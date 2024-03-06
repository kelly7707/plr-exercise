from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.models.cnn import Net
import wandb
import optuna


wandb.login()
run = wandb.init(project="prl_exercise", dir="./results")


def get_best_params(args, model, device, train_loader, test_loader):
    """
    Performs a hyperparameter search using Optuna to find the best learning rate and number of epochs.

    Parameters:
    - args: Command-line arguments.
    - model: The model to be trained.
    - device: The computing device.
    - train_loader: DataLoader for the training data.
    - test_loader: DataLoader for the test data.
    
    Returns:
    Tuple[float, int]: The best learning rate and number of epochs found by Optuna.
    """
    def objective(trial):
        # Hyperparameters to be tuned by Optuna
        lr = trial.suggest_float("lr", 5e-4, 1, log=True)
        epochs = trial.suggest_int("epochs", 1, 20)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, print_info=False)
            test_loss = test(model, device, test_loader, epoch, print_info=False)
            scheduler.step()

            trial.report(test_loss, epoch)

        return test_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_lr = study.best_params["lr"]
    best_epochs = study.best_params["epochs"]
    return best_lr, best_epochs


def train(args, model, device, train_loader, optimizer, epoch, print_info=True):
    """
    Trains the model for one epoch.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            if print_info:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
            wandb.log({"training_loss": loss.item(), "epoch": epoch})
            if args.dry_run:
                break


def test(model, device, test_loader, epoch, print_info=True):
    """
    Tests the model on the test dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    if print_info:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
            )
        )
    wandb.log({"test_loss": test_loss, "epoch": epoch})
    return test_loss


def main():
    """
    Main function of training and testing process.
    
    Parses command-line arguments, loads the MNIST dataset, initializes the model, 
    performs hyperparameter optimization using Optuna, trains the model with the best-found parameters,
    and logs the training and testing loss with wandb.
    
    """
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    best_lr, best_epochs = get_best_params(args, model, device, train_loader, test_loader)

    best_optimizer = optim.Adam(model.parameters(), lr=best_lr)
    best_scheduler = StepLR(best_optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(best_epochs):
        train(args, model, device, train_loader, best_optimizer, epoch)
        test(model, device, test_loader, epoch)
        best_scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    artifact = wandb.Artifact("code", type="code")
    artifact.add_file("scripts/train.py")
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
