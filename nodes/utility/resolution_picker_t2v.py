import torch

class DaxResolutionPickerT2V:
    """T2V resolution picker - no inputs, presets only"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_preset": ([
                    "720x1280 (9:16 Portrait)",
                    "1280x720 (16:9 Landscape)", 
                    "1024x576 (16:9 Medium)",
                    "576x1024 (9:16 Medium)",
                    "768x768 (1:1 Square)",
                    "640x640 (1:1 Square Medium)",
                    "512x512 (1:1 Square Small)",
                    "480x854 (9:16 Low)",
                    "854x480 (16:9 Low)",
                    "Custom (use override)"
                ],),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
            },
            "optional": {
                "override_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Override width (0 = use preset)"
                }),
                "override_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "Override height (0 = use preset)"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "LATENT")
    RETURN_NAMES = ("width", "height", "latent")
    FUNCTION = "pick_resolution"
    CATEGORY = "Utilities"
    
    def pick_resolution(self, resolution_preset, batch_size, override_width=0, override_height=0):
        presets = {
            "720x1280 (9:16 Portrait)": (720, 1280),
            "1280x720 (16:9 Landscape)": (1280, 720),
            "1024x576 (16:9 Medium)": (1024, 576),
            "576x1024 (9:16 Medium)": (576, 1024),
            "768x768 (1:1 Square)": (768, 768),
            "640x640 (1:1 Square Medium)": (640, 640),
            "512x512 (1:1 Square Small)": (512, 512),
            "480x854 (9:16 Low)": (480, 854),
            "854x480 (16:9 Low)": (854, 480),
            "Custom (use override)": (512, 512)
        }
        
        if override_width > 0 and override_height > 0:
            width, height = override_width, override_height
        else:
            width, height = presets[resolution_preset]
        
        # Ensure dimensions are multiples of 8 for VAE compatibility
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        # Create empty latent tensor
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        latent_dict = {"samples": latent}
        
        print(f"T2V resolution: {width}x{height}")
        
        return (width, height, latent_dict)


NODE_CLASS_MAPPINGS = {
    "ResolutionPickerT2V": DaxResolutionPickerT2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionPickerT2V": "Resolution Picker T2V",
}