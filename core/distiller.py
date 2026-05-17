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


class FutureSelfDistiller:
    """
    Self-contained temporal self-distillation engine.
    Allows the model to learn from its own Future-Self representation
    on new data in a single pass, without requiring a pre-trained teacher.
    """
    def __init__(self, student, device, T=4.0, alpha=0.9, ema_beta=0.99):
        import copy
        self.student = student.to(device)
        self.future_self = copy.deepcopy(student).to(device)
        self.device = device
        self.T = T
        self.alpha = alpha
        self.ema_beta = ema_beta
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Freeze future self parameters (updated manually via EMA)
        for param in self.future_self.parameters():
            param.requires_grad = False

    def loss(self, s_logits, t_logits, y):
        ce = F.cross_entropy(s_logits, y)
        kl = F.kl_div(
            F.log_softmax(s_logits / self.T, dim=1), 
            F.softmax(t_logits / self.T, dim=1), 
            reduction='batchmean'
        ) * (self.T ** 2)
        return self.alpha * kl + (1 - self.alpha) * ce

    def update_future_self(self):
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.future_self.parameters()):
                teacher_param.data.mul_(self.ema_beta).add_(student_param.data, alpha=1.0 - self.ema_beta)

    def train_epoch(self, loader, optimizers):
        self.student.train()
        self.future_self.eval() # Future self is evaluative and smooth
        
        total_loss = correct = n = 0
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            for opt in optimizers: opt.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # Teacher path uses the EMA Future-Self model
                with torch.no_grad():
                    t_out = self.future_self(X)
                s_out = self.student(X)
                L = self.loss(s_out, t_out, y)
            
            self.scaler.scale(L).backward()
            for opt in optimizers: self.scaler.step(opt)
            self.scaler.update()
            
            # Smoothly transition the Future-Self weights online
            self.update_future_self()

            total_loss += L.item() * len(y)
            correct += (s_out.argmax(1) == y).sum().item()
            n += len(y)
            
        return total_loss / n, correct / n

