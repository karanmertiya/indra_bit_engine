import torch

def load_teacher(model_name="cifar10_repvgg_a2", device="cuda"):
    """
    Loads high-accuracy PyTorch Hub models to act as Teachers.
    Options: 'cifar10_resnet56' (93%), 'cifar10_repvgg_a2' (95.4%)
    """
    print(f"[Indra-Bit Hub] Downloading Teacher Model: {model_name}...")
    teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
    return teacher.to(device)
