import argparse
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.architectures import IndraBitResNet
from models.teacher_hub import load_teacher
from core.turbo_optim import CudaTurboOptimizer
from core.distiller import FP16Distiller
from core.bit_packer import pack_tensor_4bit

def main():
    parser = argparse.ArgumentParser(description="Indra-Bit Training Engine")
    parser.add_argument("--width", type=int, default=40, help="Base width (40 = 1.7M params)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mode", type=str, default="futureself", choices=["teacher", "futureself"], help="Training mode: 'teacher' for standard distillation, 'futureself' for temporal self-distillation")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    tf_test  = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    
    print("[Indra-Bit] Loading CIFAR-10 Dataset...")
    train_loader = DataLoader(datasets.CIFAR10('.', True, download=True, transform=tf_train), batch_size=256, shuffle=True, num_workers=2)
    test_loader  = DataLoader(datasets.CIFAR10('.', False, transform=tf_test), batch_size=256, shuffle=False, num_workers=2)

    # Models
    student = IndraBitResNet(base_width=args.width).to(device)

    s_params = sum(p.numel() for p in student.parameters())
    print(f"\n[Indra-Bit] Student Capacity: {s_params/1000000:.2f}M Parameters")

    # Optimizers (Mixed Precision Paradigm)
    inner_p = [p for n, p in student.named_parameters() if 'prep' not in n and 'fc' not in n]
    outer_p = [p for n, p in student.named_parameters() if 'prep' in n or 'fc' in n]

    opt_bit = CudaTurboOptimizer(inner_p, lr=0.003)
    opt_fp32 = torch.optim.AdamW(outer_p, lr=0.001, weight_decay=1e-4)

    sched_fp32 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fp32, T_max=args.epochs)
    sched_bit = torch.optim.lr_scheduler.CosineAnnealingLR(opt_bit, T_max=args.epochs)

    # Dynamic Distiller selection
    if args.mode == "teacher":
        teacher = load_teacher(args.teacher, device)
        from core.distiller import FP16Distiller
        distiller = FP16Distiller(teacher, student, device)
        print(f"[Indra-Bit] Mode: Distillation from pre-trained teacher '{args.teacher}'")
    else:
        from core.distiller import FutureSelfDistiller
        distiller = FutureSelfDistiller(student, device)
        print("[Indra-Bit] Mode: Temporal Self-Distillation (Future-Self online teacher)")

    print(f"\n--- Starting FP16 Distillation ({args.epochs} Epochs) ---")
    best_acc = 0.0
    
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss, tr_acc = distiller.train_epoch(train_loader, [opt_bit, opt_fp32])
        sched_fp32.step()
        sched_bit.step()
        
        student.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                correct += (student(X).argmax(1) == y).sum().item()
        te_acc = correct / 10000
                
        if te_acc > best_acc:
            best_acc = te_acc
            
        print(f"Epoch {ep:>3} | Loss: {loss:>7.4f} | Train: {tr_acc*100:>6.2f}% | Test: {te_acc*100:>6.2f}% | Best: {best_acc*100:>6.2f}% | Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
