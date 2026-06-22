import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(x, dim=-1):
    """Fonction d'activation non-linéaire pour les capsules."""
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, out_channels, capsule_dim, kernel_size, stride):
        super(PrimaryCaps, self).__init__()
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, out_channels * capsule_dim, kernel_size, stride)

    def forward(self, x):
        outputs = self.conv(x)
        outputs = outputs.view(x.size(0), -1, self.capsule_dim)
        return squash(outputs)

class DigitCaps(nn.Module):
    def __init__(self, num_capsules, in_capsules, in_dim, out_dim, num_routing=3):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.num_routing = num_routing
        self.in_capsules = in_capsules
        self.W = nn.Parameter(torch.randn(num_capsules, in_capsules, out_dim, in_dim) * 0.01)

    def forward(self, x):
        # x: [batch, in_capsules, in_dim]
        # W: [num_caps, in_caps, out_dim, in_dim]
        # x -> [batch, 1, in_caps, in_dim, 1]
        x = x.unsqueeze(1).unsqueeze(4)
        
        # u_hat: [batch, num_caps, in_caps, out_dim, 1]
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1) # [batch, num_caps, in_caps, out_dim]

        # Initialisation des poids de couplage b_ij
        b = torch.zeros(x.size(0), self.num_capsules, self.in_capsules, 1).to(x.device)

        for i in range(self.num_routing):
            c = F.softmax(b, dim=1) # [batch, num_caps, in_caps, 1]
            s = (c * u_hat).sum(dim=2) # [batch, num_caps, out_dim]
            v = squash(s) # [batch, num_caps, out_dim]

            if i < self.num_routing - 1:
                # v: [batch, num_caps, out_dim] -> [batch, num_caps, 1, out_dim]
                # u_hat: [batch, num_caps, in_caps, out_dim]
                agreement = torch.matmul(u_hat, v.unsqueeze(3)) # [batch, num_caps, in_caps, 1]
                b = b + agreement

        return v

class Decoder(nn.Module):
    def __init__(self, input_dim=160, output_dim=784):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1, 28, 28)

class CapsNet(nn.Module):
    def __init__(self, num_routing=3):
        super(CapsNet, self).__init__()
        # Layer 1: Conv2d
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        
        # Layer 2: Primary Capsules
        # 32 capsules of dimension 8
        self.primary_caps = PrimaryCaps(256, 32, 8, kernel_size=9, stride=2)
        
        # Layer 3: Digit Capsules
        # 10 capsules of dimension 16
        # Input: 32 * 6 * 6 = 1152 capsules
        self.digit_caps = DigitCaps(10, 1152, 8, 16, num_routing)
        
        # Layer 4: Decoder
        self.decoder = Decoder(16 * 10, 784)

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        v = self.digit_caps(x)
        
        # Norme des capsules pour la classification
        classes = torch.sqrt((v ** 2).sum(dim=-1))
        
        # Reconstruction
        if y is None:
            # En inférence, on prend la capsule avec la plus grande norme
            _, max_length_indices = classes.max(dim=1)
            y = torch.eye(10).to(x.device).index_select(dim=0, index=max_length_indices)
        
        reconstructions = self.decoder((v * y.unsqueeze(2)).view(x.size(0), -1))
        
        return v, classes, reconstructions

class CapsLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_val=0.5, reconstruction_weight=0.0005):
        super(CapsLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val
        self.reconstruction_weight = reconstruction_weight

    def forward(self, images, labels, classes, reconstructions):
        # Margin loss
        left = F.relu(self.m_plus - classes) ** 2
        right = F.relu(classes - self.m_minus) ** 2
        
        margin_loss = labels * left + self.lambda_val * (1 - labels) * right
        margin_loss = margin_loss.sum(dim=1).mean()
        
        # Reconstruction loss (MSE). If reconstructions is None (e.g., during
        # fast evaluation where reconstructions are not computed), skip the
        # reconstruction term.
        if reconstructions is None:
            reconstruction_loss = torch.tensor(0.0, device=images.device)
        else:
            reconstruction_loss = F.mse_loss(reconstructions, images)

        return margin_loss + self.reconstruction_weight * reconstruction_loss
