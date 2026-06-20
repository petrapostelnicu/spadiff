import argparse
import os
import sys
import tarfile
import time
from glob import glob
from io import BytesIO

import h5py
import numpy as np


def convert_tar_to_h5(tar_path, h5_path):
    """Stream a (possibly gzipped) tar into a flat HDF5: one dataset per file."""
    h5_tmp = h5_path + '.tmp'
    n_entries = 0
    bytes_written = 0
    with tarfile.open(tar_path, mode='r|*') as tar:
        with h5py.File(h5_tmp, 'w') as h5:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name
                if name.startswith('./'):
                    name = name[2:]
                if not name:
                    continue
                data = tar.extractfile(member).read()
                if data is None:
                    continue
                h5.create_dataset(
                    name, data=np.frombuffer(data, dtype=np.uint8)
                )
                n_entries += 1
                bytes_written += len(data)
    os.rename(h5_tmp, h5_path)
    return n_entries, bytes_written


def verify_h5(h5_path, min_entries=100):
    """Sanity-check an .h5: openable, has entries, one random JPEG decodes.

    Returns (ok: bool, reason: str).
    """
    try:
        with h5py.File(h5_path, 'r') as h5:
            keys = list(h5.keys())
            if len(keys) < min_entries:
                return False, f"only {len(keys)} entries (<{min_entries})"
            jpg_keys = [k for k in keys if k.endswith('.jpg')]
            if not jpg_keys:
                return False, "no .jpg entries"
            # Pick the middle .jpg (deterministic, exercises tar offsets fairly).
            sample_key = jpg_keys[len(jpg_keys) // 2]
            data = bytes(h5[sample_key][:])
            try:
                from PIL import Image
                Image.open(BytesIO(data)).convert('RGB')
            except Exception as e:
                return False, f"JPEG decode failed for {sample_key}: {e}"
    except Exception as e:
        return False, f"open failed: {type(e).__name__}: {e}"
    return True, "OK"


def migrate_one(tar_path, h5_path, dry_run, min_entries):
    """Bring (tar_path, h5_path) into the desired post-state: only h5 exists.

    Returns one of: 'kept' (h5 already good, tar deleted),
                    'converted' (built h5, tar deleted),
                    'failed' (could not produce a verified h5; tar kept).
    """
    base = os.path.basename(tar_path)

    # Stage 1: do we already have a usable .h5?
    h5_exists = os.path.exists(h5_path)
    if h5_exists:
        ok, reason = verify_h5(h5_path, min_entries=min_entries)
        if ok:
            print(f"[KEEP-EXISTING] {base}: h5 OK ({reason})", flush=True)
            if os.path.exists(tar_path):
                if dry_run:
                    print(f"  [dry-run] would delete tar: {tar_path}", flush=True)
                else:
                    os.remove(tar_path)
                    print(f"  deleted tar", flush=True)
            return "kept"
        else:
            print(f"[REPAIR] {base}: h5 fails verification ({reason}); re-converting", flush=True)
            if not dry_run:
                try:
                    os.remove(h5_path)
                except OSError:
                    pass

    # Stage 2: convert tar -> h5
    if not os.path.exists(tar_path):
        print(f"[SKIP] {base}: source tar missing and no usable h5", flush=True)
        return "failed"

    if dry_run:
        print(f"[CONVERT] {base}: [dry-run] would convert tar -> h5", flush=True)
        return "converted"

    # Safety: wait until there is at least `min_free_gb` free on the h5
    # filesystem before starting.
    min_free_gb = int(os.environ.get("MIGRATE_MIN_FREE_GB", "50"))
    h5_dir = os.path.dirname(h5_path)
    import shutil
    waited = 0
    while True:
        free_gb = shutil.disk_usage(h5_dir).free / 1e9
        if free_gb >= min_free_gb:
            break
        if waited == 0:
            print(
                f"[WAIT] {base}: only {free_gb:.1f} GB free, "
                f"need {min_free_gb} GB. Sleeping...",
                flush=True,
            )
        time.sleep(60)
        waited += 60
        if waited > 3600:
            print(
                f"[GIVE UP] {base}: still no space after 1h; failing this tar",
                flush=True,
            )
            return "failed"
    if waited > 0:
        print(f"[WAIT] {base}: resumed after {waited}s ({free_gb:.1f} GB free)", flush=True)

    try:
        t0 = time.time()
        n_entries, sz = convert_tar_to_h5(tar_path, h5_path)
        dt = time.time() - t0
        print(
            f"[CONVERT] {base}: {n_entries} entries, "
            f"{sz / 1e9:.2f} GB, {dt:.1f}s",
            flush=True,
        )
    except Exception as e:
        print(f"[FAIL CONVERT] {base}: {type(e).__name__}: {e}", flush=True)
        # Best-effort cleanup of partial output.
        for p in (h5_path, h5_path + '.tmp'):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return "failed"

    # Stage 3: verify the freshly-built .h5 before deleting the source.
    ok, reason = verify_h5(h5_path, min_entries=min_entries)
    if not ok:
        print(f"[FAIL VERIFY] {base}: {reason}; keeping source tar", flush=True)
        try:
            os.remove(h5_path)
        except OSError:
            pass
        return "failed"

    # Stage 4: delete source tar.
    os.remove(tar_path)
    print(f"  verified h5; deleted tar", flush=True)
    return "converted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tar_dir', help='Source directory of .tar files')
    parser.add_argument('--h5-dir', required=True, help='Destination directory for .h5 files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Do nothing destructive; only print what would happen')
    parser.add_argument('--min-entries', type=int, default=100,
                        help='Minimum entry count for an h5 to be considered valid')
    parser.add_argument('--array-id', type=int, default=None)
    parser.add_argument('--array-count', type=int, default=None)
    args = parser.parse_args()

    if not args.dry_run:
        os.makedirs(args.h5_dir, exist_ok=True)

    tar_paths = sorted(glob(os.path.join(args.tar_dir, '*.tar')))
    if args.array_id is not None and args.array_count is not None:
        tar_paths = tar_paths[args.array_id::args.array_count]

    print(
        f"[task {args.array_id}/{args.array_count}] {len(tar_paths)} tars to migrate "
        f"(dry_run={args.dry_run})",
        flush=True,
    )

    counts = {"kept": 0, "converted": 0, "failed": 0}
    for tar_path in tar_paths:
        base = os.path.basename(tar_path)
        h5_path = os.path.join(args.h5_dir, base.replace('.tar', '.h5'))
        status = migrate_one(tar_path, h5_path, args.dry_run, args.min_entries)
        counts[status] += 1

    print(
        f"[task {args.array_id}/{args.array_count}] done. "
        f"kept={counts['kept']} converted={counts['converted']} failed={counts['failed']}",
        flush=True,
    )


if __name__ == '__main__':
    main()
