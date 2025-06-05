import os
import sys
from pathlib import Path


def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")


def get_path_to_assets() -> Path:
    import gdl
    pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = pythonpath.split(':')[0]
    # return Path(gdl.__file__).parents[1] / "assets"
    return Path(pythonpath) / "assets"
    # return os.path.join(pythonpath, "assets")


def get_path_to_externals() -> Path:
    # import gdl
    pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = pythonpath.split(':')[0]
    # return Path(gdl.__file__).parents[1] / "external"
    return Path(pythonpath) / "external"
    # return os.path.join(pythonpath, "external")
