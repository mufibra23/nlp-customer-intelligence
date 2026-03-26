import os, json, time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"  {func.__name__} completed in {time.time()-start:.1f}s")
        return result
    return wrapper

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_device():
    import torch
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")
