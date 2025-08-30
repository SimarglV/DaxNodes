import torch
import torch.nn.functional as F
import math

class DaxResolutionPickerI2V:
    """I2V resolution picker - image input with I2V scaling modes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
    
    RETURN_TYPES = ("INT", "INT", "IMAGE")
    RETURN_NAMES = ("width", "height", "image")
    FUNCTION = "pick_resolution"
    CATEGORY = "Utilities"
    
    def pick_resolution(self, image, resolution_mode, override_width=0, override_height=0):
        batch_size, orig_height, orig_width, channels = image.shape
        orig_aspect = orig_width / orig_height
        
        # Use override if specified
        if override_width > 0 and override_height > 0:
            target_width, target_height = override_width, override_height
        else:
            if resolution_mode == "Native (Original Resolution)":
                target_width, target_height = orig_width, orig_height
                
            elif resolution_mode == "High (1280x720 Pixel Count)":
                # Scale to match 1280x720 = 921,600 pixels while maintaining aspect ratio
                target_pixels = 1280 * 720
                scale_factor = math.sqrt(target_pixels / (orig_width * orig_height))
                target_width = int(orig_width * scale_factor)
                target_height = int(orig_height * scale_factor)
                
            elif resolution_mode == "Low (480x854 Pixel Count)":
                # Scale to match 480x854 = 409,920 pixels while maintaining aspect ratio  
                target_pixels = 480 * 854
                scale_factor = math.sqrt(target_pixels / (orig_width * orig_height))
                target_width = int(orig_width * scale_factor)
                target_height = int(orig_height * scale_factor)
        
        # Ensure dimensions are multiples of 8 for VAE compatibility
        target_width = (target_width // 8) * 8
        target_height = (target_height // 8) * 8
        
        # Resize image if needed
        if target_width != orig_width or target_height != orig_height:
            # Resize using torch interpolate
            image_resized = F.interpolate(
                image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            output_image = image_resized
        else:
            output_image = image
        
        actual_pixels = target_width * target_height
        print(f"I2V resolution: {target_width}x{target_height} ({actual_pixels:,} pixels)")
        
        return (target_width, target_height, output_image)


NODE_CLASS_MAPPINGS = {
    "ResolutionPickerI2V": DaxResolutionPickerI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionPickerI2V": "Resolution Picker I2V",
}