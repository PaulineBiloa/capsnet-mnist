import torch
import torch.optim as optim
from tqdm import tqdm
from model import CapsNet, CapsLoss
from utils import set_seed, get_dataloaders
import argparse

def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # One-hot encoding des labels pour la loss
        target_one_hot = torch.eye(10).to(device).index_select(dim=0, index=target)
        
        optimizer.zero_grad()
        _, classes, reconstructions = model(data, target_one_hot)
        
        loss = criterion(data, target_one_hot, classes, reconstructions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calcul de la précision
        _, preds = classes.max(dim=1)
        correct += (preds == target).sum().item()
        
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1), 'acc': 100. * correct / ((batch_idx + 1) * data.size(0))})

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_one_hot = torch.eye(10).to(device).index_select(dim=0, index=target)
            
            _, classes, _ = model(data)
            test_loss += criterion(data, target_one_hot, classes, None).item() # On ignore la reconstruction ici pour simplifier
            
            _, preds = classes.max(dim=1)
            correct += (preds == target).sum().item()
            
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='CapsNet sur MNIST')
    parser.add_argument('--batch-size', type=int, default=128, help='taille du batch')
    parser.add_argument('--epochs', type=int, default=10, help='nombre d\'époques')
    parser.add_argument('--lr', type=float, default=0.001, help='taux d\'apprentissage')
    parser.add_argument('--routing', type=int, default=3, help='nombre d\'itérations de routing')
    parser.add_argument('--seed', type=int, default=42, help='seed aléatoire')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")

    train_loader, test_loader = get_dataloaders(args.batch_size)
    
    model = CapsNet(num_routing=args.routing).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CapsLoss()

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, device, epoch)
        acc = test(model, test_loader, criterion, device)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'capsnet_best.pth')
            print("Modèle sauvegardé.")

if __name__ == '__main__':
    main()
