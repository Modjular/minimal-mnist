"""Trainer

Performs training on the network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms.transforms import ToTensor

from datetime import datetime
from os.path import join, exists, isdir
from os import mkdir

from model import FCNetwork


class Trainer:
    def __init__(self, root: str, results_dir: str):
        """Creates the trainer class.

        Args:
            root: path to the MNIST data root.
            results_dir: Logging path to use.
        """
        self.root = root
        self.train_data = MNIST(root, transform=ToTensor())
        self.test_data = MNIST(root, train=False, transform=ToTensor())
        self.result_dir = results_dir

        # Make sure results_dir exists
        if not exists(results_dir) or not isdir(results_dir):
            mkdir(results_dir)

    def train(self, batch_size: int, first_layer: int, second_layer: int,
              leaky: tuple, optimizer: str, optimizer_args: dict,
              epochs: int):
        """Performs training on the network.
        Args:
            batch_size: Batch size for training and validation.
            first_layer: Number of nodes in the first layer.
            second_layer: Number of nodes in the second layer.
            leaky: Configuration of leaky ReLU.
            optimizer: The optimizer to use. Either "sgd" or "adam".
            optimizer_args: Arguments for the optimizer
            epochs: Number of epochs to run for.
        """
        print("Initializing training...")
        print(f"Results saved in {self.result_dir}")
        train_loader = DataLoader(self.train_data, batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(self.test_data, batch_size, shuffle=True, pin_memory=True)
        network = FCNetwork(784, 10, first_layer, second_layer, leaky)
        if torch.cuda.is_available():
            network.cuda()
        if optimizer == 'sgd':
            optimizer = SGD(network.parameters(), **optimizer_args)

        elif optimizer == 'adam':
            optimizer = Adam(network.parameters(), **optimizer_args)

        loss_crit = CrossEntropyLoss()

        prev_epoch_val_acc = 0

        for epoch in range(epochs):
            header = "| Iteration |       Loss |        Acc |"
            underline = "|-----------|------------|------------|"
            table_format = "| {:>9} | {:1.8f} | {:1.8f} |"
            print(f'\nEpoch: {epoch + 1}')
            print(header)
            print(underline)
            for i, data in enumerate(train_loader):
                network.train()
                optimizer.zero_grad()
                img, cls = data
                if torch.cuda.is_available():
                    img = img.cuda()
                    cls = cls.cuda()
                h1, h2, out = network(img)
                loss = loss_crit(out, cls)

                # Do backprop
                if i % 100 == 0:
                    print(table_format.format(
                        i, loss.item(), self.calc_batch_accuracy(out, cls))
                    )
                loss.backward()
                optimizer.step()

            print(underline)
            # Write out the weights file
            torch.save(network.state_dict(),
                       join(self.result_dir, '{}.pth'.format(epoch + 1)))

            # Do validation
            validation_acc = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    network.eval()
                    img, cls = data
                    if torch.cuda.is_available():
                        img = img.cuda()
                        cls = cls.cuda()
                    h1, h2, out = network(img)

                    validation_acc += self.calc_batch_accuracy(out, cls)
                validation_acc /= i
                print(f"Epoch {epoch + 1} validation accuracy: "
                      f"{validation_acc}")

            if prev_epoch_val_acc - validation_acc > 0.04:
                print("Overfitted to the training set.")
                break

    @staticmethod
    def calc_batch_accuracy(output: torch.Tensor,
                            target: torch.tensor) -> float:
        """Calculates accuracy for a batch.

        Args:
            output: Output predictions of the network on one-hot encoding.
            target: Targets for the predictions.
        """
        oa = output.argmax(1)  # output argmax
        correct = (oa == target).sum()

        return float(correct) / float(target.shape[0])


if __name__ == '__main__':
    current_time_str = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    t = Trainer('/Users/tony/Documents/Github/minimal-mnist/MNIST',
                join("/Users/tony/Documents/Github/minimal-mnist/MNIST", current_time_str))
    t.train(50, 10, 15, (False, False, False), 'sgd',
            {'momentum': 0.9409782496856666,
             'lr': 0.0038795787201773}, 250)
