import poptorch
import poptorchlab
import torch
import torchvision
from tqdm import tqdm


class Block(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, pool_size):
        super(Block, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                              num_filters,
                              kernel_size=kernel_size)
        self.pool = torch.nn.MaxPool2d(kernel_size=pool_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = Block(1, 32, 3, 2)
        self.layer2 = Block(32, 64, 3, 2)
        self.layer3 = torch.nn.Linear(1600, 128)
        self.layer3_act = torch.nn.ReLU()
        self.layer3_dropout = torch.torch.nn.Dropout(0.5)
        self.layer4 = torch.nn.Linear(128, 10)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # Flatten layer
        x = x.view(-1, 1600)
        x = self.layer3_act(self.layer3(x))
        x = self.layer4(self.layer3_dropout(x))
        x = self.softmax(x)
        return x


class TrainingModelWithLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, args, labels=None):
        output = self.model(args)
        if labels is None:
            return output
        else:
            loss = self.loss(output, labels)
            return output, loss


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    labels = labels[-predictions.size()[0]:]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy


if __name__ == '__main__':
    parser = poptorchlab.ArgumentParser()
    args = parser.parse_args()

    local_dataset_path = '~/.torch/datasets'

    transform_mnist = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
    )

    training_dataset = torchvision.datasets.MNIST(
        local_dataset_path,
        train=True,
        download=True,
        transform=transform_mnist
    )

    training_opts = poptorchlab.training_settings(poptorch.Options(), args)

    training_data = poptorch.DataLoader(
        options=training_opts,
        dataset=training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    model = Network()
    model_with_loss = TrainingModelWithLoss(model)

    training_model = poptorch.trainingModel(
        model_with_loss,
        training_opts,
        optimizer=poptorch.optim.SGD(model.parameters(), lr=args.lr)
    )

    nr_steps = len(training_data)

    epochs = 10
    for epoch in tqdm(range(1, epochs + 1), leave=True, desc="Epochs", total=epochs):
        with tqdm(training_data, total=nr_steps, leave=False) as bar:
            for data, labels in bar:
                preds, losses = training_model(data, labels)

                mean_loss = torch.mean(losses).item()

                acc = accuracy(preds, labels)
                bar.set_description(
                    "Loss: {:0.4f} | Accuracy: {:05.2F}% ".format(mean_loss, acc)
                )
