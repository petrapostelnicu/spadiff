from peft.tuners.tuners_utils import BaseTunerLayer
from typing import List, Any, Optional, Type

_DEBUG_LORA = False  # Set to True to debug LoRA scaling

# refer to https://github.com/Yuanshi9815/OminiControl
class select_lora:
    """Context manager to select a specific LoRA adapter and disable others."""
    def __init__(self, lora_modules: List[BaseTunerLayer], adapter_name) -> None:
        self.adapter_name = adapter_name

        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.saved_scalings = {}

    def __enter__(self) -> None:
        if _DEBUG_LORA and not self.lora_modules:
            print(f"[DEBUG select_lora] WARNING: No LoRA modules found for adapter '{self.adapter_name}'!")
        for lora_module in self.lora_modules:
            module_id = id(lora_module)
            self.saved_scalings[module_id] = {}
            for active_adapter in lora_module.active_adapters:
                # Save current scaling
                self.saved_scalings[module_id][active_adapter] = lora_module.scaling[active_adapter]
                # Set target adapter to 1, others to 0
                new_scale = 1 if active_adapter == self.adapter_name else 0
                if _DEBUG_LORA:
                    print(f"[DEBUG select_lora] {active_adapter}: {lora_module.scaling[active_adapter]} -> {new_scale}")
                lora_module.scaling[active_adapter] = new_scale

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for lora_module in self.lora_modules:
            module_id = id(lora_module)
            for active_adapter in lora_module.active_adapters:
                # Restore original scaling
                lora_module.scaling[active_adapter] = self.saved_scalings[module_id][active_adapter]


class disable_lora:
    """Context manager to disable ALL LoRA adapters (set scale=0)."""
    def __init__(self, lora_modules) -> None:
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.saved_scalings = {}
        self._logged = False

    def __enter__(self) -> None:
        if _DEBUG_LORA and not self.lora_modules and not self._logged:
            print(f"[DEBUG disable_lora] WARNING: No LoRA modules found!")
            self._logged = True
        for lora_module in self.lora_modules:
            module_id = id(lora_module)
            self.saved_scalings[module_id] = {}
            for active_adapter in lora_module.active_adapters:
                # Save current scaling and set to 0
                self.saved_scalings[module_id][active_adapter] = lora_module.scaling[active_adapter]
                lora_module.scaling[active_adapter] = 0

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for lora_module in self.lora_modules:
            module_id = id(lora_module)
            for active_adapter in lora_module.active_adapters:
                # Restore original scaling
                lora_module.scaling[active_adapter] = self.saved_scalings[module_id][active_adapter]