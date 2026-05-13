import torch
import torch.nn.functional as F

class FP16Distiller:
    def __init__(self, teacher, student, device, T=4.0, alpha=0.9):
        self.teacher = teacher.eval().to(device)
        self.student = student.to(device)
        self.device = device
        self.T = T
        self.alpha = alpha
        self.scaler = torch.amp.GradScaler('cuda')

    def loss(self, s_logits, t_logits, y):
        ce = F.cross_entropy(s_logits, y)
        kl = F.kl_div(
            F.log_softmax(s_logits / self.T, dim=1), 
            F.softmax(t_logits / self.T, dim=1), 
            reduction='batchmean'
        ) * (self.T ** 2)
        return self.alpha * kl + (1 - self.alpha) * ce

    def train_epoch(self, loader, optimizers):
        self.student.train()
        total_loss = correct = n = 0
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            for opt in optimizers: opt.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                with torch.no_grad():
                    t_out = self.teacher(X)
                s_out = self.student(X)
                L = self.loss(s_out, t_out, y)
            
            self.scaler.scale(L).backward()
            for opt in optimizers: self.scaler.step(opt)
            self.scaler.update()

            total_loss += L.item() * len(y)
            correct += (s_out.argmax(1) == y).sum().item()
            n += len(y)
            
        return total_loss / n, correct / n
