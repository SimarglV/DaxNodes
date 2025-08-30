import torch
import torch.nn.functional as F
import math

class DaxResolutionPickerFLF2V:
    """FLF2V resolution picker - dual image inputs with I2V scaling"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "resolution_mode": ([
                    "Native (Original Resolution)",
                    "High (1280x720 Pixel Count)", 
                    "Low (480x854 Pixel Count)"
                ],),
            },
            "optional": {
                "override_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Override width (0 = use calculated)"
                }),
                "override_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Override height (0 = use calculated)"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("width", "height", "image_1", "image_2")
    FUNCTION = "pick_resolution"
    CATEGORY = "Utilities"
    
    def pick_resolution(self, image_1, image_2, resolution_mode, override_width=0, override_height=0):
        batch_size_1, orig_height_1, orig_width_1, channels_1 = image_1.shape
        batch_size_2, orig_height_2, orig_width_2, channels_2 = image_2.shape
        
        # Use first image dimensions as reference
        orig_width = orig_width_1
        orig_height = orig_height_1
        orig_aspect = orig_width / orig_height
        
        # Use override if specified
        if override_width > 0 and override_height > 0:
            target_width, target_height = override_width, override_height
        else:
            if resolution_mode == "Native (Original Resolution)":
                target_width, target_height = orig_width, orig_height
                
            elif resolution_mode == "High (1280x720 Pixel Count)":
                target_pixels = 1280 * 720
                scale_factor = math.sqrt(target_pixels / (orig_width * orig_height))
                target_width = int(orig_width * scale_factor)
                target_height = int(orig_height * scale_factor)
                
            elif resolution_mode == "Low (480x854 Pixel Count)":
                target_pixels = 480 * 854
                scale_factor = math.sqrt(target_pixels / (orig_width * orig_height))
                target_width = int(orig_width * scale_factor)
                target_height = int(orig_height * scale_factor)
        
        # Ensure dimensions are multiples of 8 for VAE compatibility
        target_width = (target_width // 8) * 8
        target_height = (target_height // 8) * 8
        
        # Resize image_1
        if target_width != orig_width_1 or target_height != orig_height_1:
            image_1_resized = F.interpolate(
                image_1.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            image_1_resized = image_1
        
        # Resize image_2
        if target_width != orig_width_2 or target_height != orig_height_2:
            image_2_resized = F.interpolate(
                image_2.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            image_2_resized = image_2
        
        actual_pixels = target_width * target_height
        print(f"FLF2V resolution: {target_width}x{target_height} ({actual_pixels:,} pixels)")
        
        return (target_width, target_height, image_1_resized, image_2_resized)


NODE_CLASS_MAPPINGS = {
    "ResolutionPickerFLF2V": DaxResolutionPickerFLF2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionPickerFLF2V": "Resolution Picker FLF2V",
}