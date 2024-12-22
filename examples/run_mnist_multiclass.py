from mnist import MNIST

import minitorch as torch

mndata = MNIST("examples/data/")
images, labels = mndata.load_training()

BACKEND = torch.backends.get_backend()
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


class Network(torch.nn.Module):
    """
    CNN for MNist classification based on LeNet.

    This model implements the following procedure:

    1. Apply a convolution with 4 output channels and a 3x3 kernel followed by a ReLU (save to self.mid)
    2. Apply a convolution with 8 output channels and a 3x3 kernel followed by a ReLU (save to self.out)
    3. Apply 2D average pooling with 4x4 kernel.
    4. Flatten channels, height, and width. (Should be size BATCHx392)
    5. Apply a Linear to size 64 followed by a ReLU and Dropout with rate 25%
    6. Apply a Linear to size C (number of classes).
    7. Apply a logsoftmax over the class dimension.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(1, 4, 3, 3)
        self.conv2 = torch.nn.Conv2d(4, 8, 3, 3)
        self.linear1 = torch.nn.Linear(392, 64)
        self.linear2 = torch.nn.Linear(64, C)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = torch.nn.avgpool2d(x, (4, 4))
        x = x.view(BATCH, 392)

        x = self.linear1(x).relu()
        x = torch.nn.dropout(x, 0.25, ignore=not self.training)
        x = self.linear2(x)
        return torch.nn.logsoftmax(x, 1)


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(torch.tensor([x], device=BACKEND))

    def train(
        self,
        data_train,
        data_val,
        learning_rate,
        max_epochs=500,
        log_fn=default_log_fn,
        batched: bool = False,
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = torch.optim.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(range(0, n_training_samples, BATCH)):
                if n_training_samples - example_num <= BATCH:
                    continue
                y = torch.tensor(y_train[example_num : example_num + BATCH], device=BACKEND)
                x = torch.tensor(X_train[example_num : example_num + BATCH], device=BACKEND)
                x.requires_grad_(True)
                y.requires_grad_(True)

                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.device == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = torch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            device=BACKEND,
                        )
                        x = torch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            device=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    if batched:
                        log_fn(epoch, total_loss, correct, BATCH, losses, model)
                    else:
                        log_fn(epoch, total_loss, correct, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01, batched=True)
