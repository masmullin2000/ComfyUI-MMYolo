from .MMYolo import MMYolo

NODE_CLASS_MAPPINGS = {
    "MMFace_Finder": MMYolo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMFace_Finder": "Face Finder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
