import torch


def get_cuda_autocast_kwargs(device):
    if isinstance(device, torch.device):
        device_type = device.type
        device_index = device.index
    else:
        device_str = str(device)
        device_type = "cuda" if device_str.startswith("cuda") else device_str
        if ":" in device_str:
            device_index = int(device_str.split(":", 1)[1])
        else:
            device_index = 0

    enabled = device_type == "cuda" and torch.cuda.is_available()
    if not enabled:
        return {"device_type": "cuda", "enabled": False}

    major, _minor = torch.cuda.get_device_capability(device_index)
    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    return {"device_type": "cuda", "dtype": amp_dtype, "enabled": True}
