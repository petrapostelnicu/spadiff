import os
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import json
import numpy as np
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import pycocotools.mask as mask_util
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict

from torch.utils.data import Dataset
import torch

from transformers import T5Tokenizer,T5TokenizerFast

from model.src.pipelines import FluxRegionalPipeline
from model.utils.utils import mask2box, get_text_token_len
from model.utils.visualizer import Visualizer

class SACap_1M_Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        seg_caption_path,
        image_root,
        is_group_bucket = False,
        cache_root=None,
        resolution:Union[List,int]=1024,
        cond_scale_factor:int =2,
        use_h5_files: bool = False,
        h5_root: Optional[str] = None,
    ):
        super(SACap_1M_Dataset, self).__init__()
        self.image_root = image_root
        self.seg_caption_path = seg_caption_path
        self.cache_root = cache_root
        # HDF5 path: one .h5 per image_group. h5_root defaults to image_root if
        # not specified.
        self.use_h5_files = use_h5_files
        self.h5_root = h5_root if h5_root is not None else image_root

        self.resolution = [resolution,resolution] if isinstance(resolution, int) else resolution
        self.cond_resolution = [self.resolution[0]//cond_scale_factor,self.resolution[1]//cond_scale_factor]

        if is_group_bucket:
            if self.cache_root is None:
                raise ValueError("Bucket grouping is enabled. Please specify a cache_root directory to store the bucket information.")
            os.makedirs(self.cache_root,exist_ok=True)

        self.images_info = pd.read_parquet(seg_caption_path)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transforms = transforms.Compose(
            [
                transforms.Resize(self.cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.visualizer = Visualizer()
        
        if is_group_bucket:
            self.tokenizer = T5TokenizerFast.from_pretrained('google/t5-v1_1-xxl')
            self._set_group_flag()
        else:
            self.flag = np.full([len(self),], fill_value=0, dtype=np.int64)

        # If we're on the HDF5 path and only some .h5 files exist (partial
        # conversion), drop samples whose image_group has no .h5.
        if self.use_h5_files:
            self._filter_to_available_h5()
                   
    def __len__(self):
        return len(self.images_info)

    def _filter_to_available_h5(self):
        """Restrict images_info (and self.flag) to samples whose .h5 file exists.
        """
        if not os.path.isdir(self.h5_root):
            raise FileNotFoundError(f"h5_root does not exist: {self.h5_root}")

        available_groups = set()
        for fname in os.listdir(self.h5_root):
            # Accept .h5 (and reject .h5.tmp partial outputs from the converter).
            if fname.endswith('.h5') and not fname.endswith('.h5.tmp'):
                available_groups.add(fname[:-3])  # strip .h5

        if not available_groups:
            raise FileNotFoundError(
                f"No .h5 files found in {self.h5_root}; cannot run on the HDF5 path."
            )

        n_total = len(self.images_info)
        all_groups = set(self.images_info['image_group'].unique())
        mask = self.images_info['image_group'].isin(available_groups).values
        n_kept = int(mask.sum())

        print(
            f"[SACap_1M_Dataset] HDF5 subset filter: "
            f"{len(available_groups)}/{len(all_groups)} groups available, "
            f"kept {n_kept}/{n_total} samples ({100.0*n_kept/n_total:.1f}%)",
            flush=True,
        )

        if n_kept == 0:
            raise RuntimeError(
                f"None of the {len(all_groups)} image groups have a matching .h5 "
                f"in {self.h5_root}. Check h5_root and naming."
            )

        self.images_info = self.images_info[mask].reset_index(drop=True)
        # self.flag indexes the *original* images_info; filter the same mask.
        self.flag = self.flag[mask]

    def _get_h5_file(self, image_group: str):
        """Get a cached h5py file handle for an image_group's HDF5 file.
        Returns None if h5py isn't installed or the file doesn't exist.
        """
        from collections import OrderedDict
        if not hasattr(self, '_h5_files'):
            self._h5_files: OrderedDict = OrderedDict()
            self._h5_cache_max = int(os.environ.get('SACAP_H5_CACHE_SIZE', '32'))

        # Cache hit: move to end (most-recently-used) and return.
        if image_group in self._h5_files:
            self._h5_files.move_to_end(image_group)
            return self._h5_files[image_group]

        # Cache miss: validate existence + open.
        try:
            import h5py
        except ImportError:
            return None
        h5_path = os.path.join(self.h5_root, f"{image_group}.h5")
        if not os.path.exists(h5_path):
            return None

        # Evict LRU entries until there's room for the new handle.
        while len(self._h5_files) >= self._h5_cache_max:
            _, old_h5 = self._h5_files.popitem(last=False)  # pop oldest
            try:
                old_h5.close()
            except Exception:
                pass

        h5 = h5py.File(h5_path, 'r', libver='latest')
        self._h5_files[image_group] = h5
        return h5

    def _read_from_h5(self, image_group: str, image_name: str) -> tuple:
        """Read image + JSON from a per-tar HDF5 file.
        """
        h5 = self._get_h5_file(image_group)
        if h5 is None:
            raise FileNotFoundError(
                f"No HDF5 file for image_group={image_group} at {self.h5_root}"
            )
        anno_name = image_name[:image_name.rfind(".")] + ".json"

        def _read(name):
            if name in h5:
                return bytes(h5[name][:])
            alt = f"./{name}"
            if alt in h5:
                return bytes(h5[alt][:])
            raise KeyError(f"{name} not in HDF5 {h5.filename}")

        image_bytes = _read(image_name)
        anno_bytes = _read(anno_name)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        anns = json.loads(anno_bytes)
        return image, anns, f"{h5.filename}::{image_name}"

    def _set_group_flag(self):
        # Related to `GroupSampler` in `group_sampler.py`.
        # group data by cond_seq_len and txt_seq_len values.

        print("=====SACap_1M_Dataset set group flag=====")
        cache_path = os.path.join(self.cache_root, f'{self.cond_resolution[0]}H_{self.cond_resolution[1]}W-group_bucket.parquet')
        if os.path.exists(cache_path): # use cache
            result_df = pd.read_parquet(cache_path)
        else:
            # Parallel run _compute_token_num function
            num_workers = 24

            results = Parallel(n_jobs=num_workers)(
                delayed(SACap_1M_Dataset.get_token_num)(
                    id,
                    images_info=self.images_info,
                    image_root=self.image_root,
                    resolution=self.resolution,
                    tokenizer=self.tokenizer,
                    visualizer=self.visualizer,
                    cond_transforms=self.cond_transforms,
                )
                for id in tqdm(range(len(self)))
            )
            # Note: Avoid instance methods here. Parallel pickles the entire 'self', drastically slowing down execution.

            save_data = defaultdict(list)
            for res in results:
                for k in res:
                    save_data[k].append(res[k])

            # save
            result_df = pd.DataFrame(save_data)
            result_df.to_parquet(cache_path, index=False)

        assert len(result_df) == len(self)

        flags = []
        for index, row in result_df.iterrows():
            cond_seq_len = row['cond_seq_len']
            txt_seq_len = row['txt_seq_len']
            flag = cond_seq_len // 50 + txt_seq_len // 50 * 1e6
            # Bins seq_len into 50-unit buckets (//50), to create approximately equal-sized groups while allowing ±50 variation.
            # with 1e6 shift `txt_seq_len` into the higher bits, ensuring the two terms remain non-overlapping.

            flags.append(flag)

        self.flag = np.array(flags, dtype=np.int64)       

    @staticmethod
    def get_token_num(
        idx,
        images_info,
        image_root,
        resolution,
        tokenizer,
        visualizer,
        cond_transforms,
    ):
        # get valid condition token num after filtering out zero-value condition tokens, and obtain text token num via tokenizer.
        img_info = images_info.iloc[idx]
        image_name = img_info['imagename']
        image_group = img_info.get('image_group', '')

        image_path = os.path.join(image_root, image_group, image_name)
        anno_path = image_path[:image_path.rfind(".")] + ".json"
        with open(anno_path, "r", encoding="utf-8") as file:
            anns = json.load(file)
        
        img_w = anns["image"]["width"]
        img_h = anns["image"]["height"]
        
        segments_info = {seg_info['anno_id']:seg_info for seg_info in img_info['segments_info']}
        
        label = []
        
        txt_seq_len = 0
        
        global_caption = img_info['caption']
        txt_seq_len += get_text_token_len(tokenizer, global_caption)
    
        for seg in anns["annotations"]:
            if seg['id'] in segments_info:
                mask = mask_util.decode(seg["segmentation"])==1
                label.append(mask)
                
                regional_caption = segments_info[seg['id']]["caption"]
                txt_seq_len += get_text_token_len(tokenizer, regional_caption)
        
        if len(label) != 0:
            label = np.stack(label, axis=0) # n,h,w
            label = torch.from_numpy(label)
            label = label[None, ...]
            label = F.interpolate(label.float(), size=resolution, mode='nearest-exact')
            label = label[0, ...].long()  # n,h,w
            
            cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
            cond_pixel_values = visualizer.draw_contours(
                cond_pixel_values,
                label.cpu().numpy(),
                thickness=1,
                colors=[(255, 255, 255), ] * len(label)
            )
            cond_pixel_values = Image.fromarray(cond_pixel_values)
            cond_pixel_values = cond_transforms(cond_pixel_values)
            
            valid_cond_token_num =  FluxRegionalPipeline.get_valid_cond_token_num(cond_pixel_values)
        else:
            valid_cond_token_num = 0
            
        return {
            'cond_seq_len': valid_cond_token_num,
            'txt_seq_len': txt_seq_len
        }  

    def __getitem__(self, idx):
        # [DIAGNOSTIC] sample 20% of calls and time the major stages.
        import time as _time
        _diag = np.random.random() < 0.20
        _t_start = _time.time()

        img_info = self.images_info.iloc[idx]
        image_name = img_info['imagename']
        image_group = img_info.get('image_group', '')

        try:
            if self.use_h5_files:
                image, anns, image_path = self._read_from_h5(image_group, image_name)
            else:
                image_path = os.path.join(self.image_root, image_group, image_name)
                anno_path = image_path[:image_path.rfind(".")] + ".json"
                image = Image.open(image_path).convert('RGB')
                with open(anno_path, "r", encoding="utf-8") as file:
                    anns = json.load(file)
        except (EOFError, OSError, KeyError, ValueError,
                Image.UnidentifiedImageError, json.JSONDecodeError) as e:
            # Log once per bad sample, then retry with another random index.
            try:
                with open("/logs/dataset_bad_samples.log", "a") as _f:
                    _f.write(
                        f"image={image_name} group={image_group} "
                        f"err={type(e).__name__}: {str(e)[:200]}\n"
                    )
            except OSError:
                pass
            return self.__getitem__(np.random.randint(len(self)))

        _t_read = _time.time()

        img_w, img_h = image.size

        global_caption = img_info['caption']
        segments_info = {seg_info['anno_id']:seg_info for seg_info in img_info['segments_info']}

        label = []
        boxes = []
        regional_captions = []
        short_regional_captions = []
        anno_ids = []

        for seg in anns["annotations"]:
            if seg['id'] in segments_info:
                mask = mask_util.decode(seg["segmentation"])==1
                x0, y0, x1, y1 = mask2box(mask)
                box = np.array([
                    x0 / img_w,
                    y0 / img_h,
                    x1 / img_w ,
                    y1 / img_h ,
                ])
                boxes.append(box)
                anno_ids.append(seg['id'])
                label.append(mask)

                regional_captions.append(segments_info[seg['id']]["caption"])
                if "short_caption" in segments_info[seg['id']]:
                    short_regional_captions.append(segments_info[seg['id']]["short_caption"])

        if len(regional_captions)==0: # try again
            return self.__getitem__(np.random.randint(len(self)))

        _t_parse = _time.time()

        label = np.stack(label, axis=0) # n,h,w
        label = torch.from_numpy(label)
        label = label[None,...]
        label = F.interpolate(label.float(), size=self.resolution, mode='nearest-exact')
        label = label[0,...].long() # n,h,w

        pixel_values = self.image_transforms(image) # c,h,w

        cond_pixel_values = np.zeros([label.shape[-2],label.shape[-1],3],dtype=np.uint8)
        cond_pixel_values = self.visualizer.draw_contours(
            cond_pixel_values,
            label.cpu().numpy(),
            thickness=1,
            colors=[(255,255,255),]*len(regional_captions) # 黑底白边
        )
        cond_pixel_values = Image.fromarray(cond_pixel_values)
        cond_pixel_values = self.cond_transforms(cond_pixel_values)

        if _diag:
            _t_end = _time.time()
            import os as _os
            _log_path = _os.environ.get("DATASET_TIMING_LOG", "/logs/dataset_timing.log")
            try:
                with open(_log_path, "a") as _f:
                    _f.write(
                        f"pid={_os.getpid()} total={_t_end-_t_start:.2f}s "
                        f"read(img+json)={_t_read-_t_start:.2f}s "
                        f"mask_parse={_t_parse-_t_read:.2f}s "
                        f"transform={_t_end-_t_parse:.2f}s "
                        f"n_regions={len(regional_captions)} image={image_name}\n"
                    )
            except OSError:
                pass
            
        return_dict =  {
            "label":label,
            "regional_captions":regional_captions,
            "global_caption":global_caption,
            "pixel_values":pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "image_name":image_name,
            "image_path":image_path,
            "boxes":boxes,
            "anno_ids":anno_ids
        }
        
        if len(short_regional_captions)>0:
            # only used in test
            return_dict["short_regional_captions"] = short_regional_captions

        return return_dict