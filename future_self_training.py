import torch
import torch.nn as nn
import torch.optim as optim
import copy

class FutureSelfTrainer:
    """
    Online Temporal Self-Distillation Framework.
    
    This allows a model to act as its own 'Future-Self' teacher
    on new data in a SINGLE training pass (no double-training!).
    
    How it works:
      1. We keep a smoothed Exponential Moving Average (EMA) copy of the model weights.
      2. The active model (student/newborn) trains on the data.
      3. The EMA model (future self) acts as a stable soft teacher, showing the active
         model where its representations are heading.
      4. Dynamic soft losses (KL Divergence) pull the newborn model toward its own
         future-self projection.
    """
    def __init__(self, model, lr=1e-4, ema_beta=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        
        # Create the Future-Self model (deep copy of weights)
        self.future_self = copy.deepcopy(model).to(self.device)
        # Freeze future self parameters (we update it manually via EMA, no gradients)
        for param in self.future_self.parameters():
            param.requires_grad = False
            
        self.ema_beta = ema_beta
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def update_future_self(self):
        """
        Dynamically slide the teacher weights toward the active model.
        EMA = beta * EMA + (1 - beta) * Active
        """
        with torch.no_grad():
            for student_param, teacher_param in zip(self.model.parameters(), self.future_self.parameters()):
                teacher_param.data.mul_(self.ema_beta).add_(student_param.data, alpha=1.0 - self.ema_beta)

    def train_step(self, x, y, distillation_weight=0.5):
        self.model.train()
        self.future_self.eval() # Future self acts in eval mode for stable predictions
        
        self.optimizer.zero_grad()
        
        # 1. Forward pass of the Active Model (Newborn)
        student_logits = self.model(x)
        
        # 2. Forward pass of the Future Self (Teacher)
        with torch.no_grad():
            teacher_logits = self.future_self(x)
            
        # 3. Soft distillation loss (KL Divergence)
        # We want the newborn to capture the smooth probability landscape of its own future self
        p_student = torch.log_softmax(student_logits, dim=-1)
        p_teacher = torch.softmax(teacher_logits, dim=-1)
        
        soft_loss = self.kl_loss(p_student, p_teacher)
        
        # 4. Hard cross entropy loss (Standard learning from ground truth)
        hard_loss = self.ce_loss(student_logits, y)
        
        # Total single-pass loss
        loss = (1.0 - distillation_weight) * hard_loss + distillation_weight * soft_loss
        
        # 5. Optimize Newborn
        loss.backward()
        self.optimizer.step()
        
        # 6. Update Future Self weights on-the-fly
        self.update_future_self()
        
        return loss.item(), hard_loss.item(), soft_loss.item()
