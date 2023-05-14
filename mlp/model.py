import torch


class mlp(torch.nn.Module):
    def __init__(self, num_in, num_hid1, num_hid2, num_out):
        super(mlp, self).__init__()

        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.classify = torch.nn.Linear(num_hid2, num_out)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.classify(x)
        x = self.sigmoid(x)
        return x
