import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torchsummary import summary


class SWING_CNN(nn.Module):
    '''
    Implementation based on the following paper:
    @article{kim2023domain,
        title={
            Domain adaptation based fault diagnosis under variable
            operating conditions of a rock drill
        },
        author={
            Kim, Yong Chae and Kim, Taehun and Ko, Jin Uk and Lee, Jinwook
            and Kim, Keon
        },
        journal={International Journal of Prognostics and Health Management},
        volume={14},
        number={2},
        year={2023}
    }
    '''
    def __init__(
            self, domain_adap=True
    ):
        super().__init__()
        self.domain_adap = domain_adap
        self.n_parts = 3
        self.n_classes = 11
        self.n_domains = 8

        self.conv = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Conv1d(
                    3, 16, kernel_size=15, stride=1,
                    padding=7,
                ),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(
                    16, 32, kernel_size=15, stride=1,
                    padding=7,
                ),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    32, 64, kernel_size=7, stride=1, padding=3
                ),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(
                    64, 128, kernel_size=7, stride=1, padding=3,
                ),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    128, 256, kernel_size=3, stride=1, padding=1,
                ),
                nn.Flatten()
            ) for k in range(self.n_parts)
        })

        flat_output_dim = 19200

        for k in range(self.n_parts):
            self.conv[str(k)].add_module(
                'fc', nn.Sequential(
                    nn.Linear(flat_output_dim, 512),
                    nn.ReLU()
                )
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.n_parts*512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, self.n_classes)
        )

        if self.domain_adap:
            self.dom_classifier = nn.Sequential(
                nn.Linear(self.n_parts*512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100, self.n_domains)
            )

    def domain_adapt_schedule(self, p, gamma=10.0):
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

    def forward(self, x, p_epoch=None):
        features = []
        crop_len = x.shape[2] // self.n_parts
        for i in range(self.n_parts):
            start = i * crop_len
            end = (i + 1) * crop_len
            features.append(
                self.conv[str(i)](x[:, :, start:end])
            )
        features = torch.cat(features, dim=1)
        class_scores = self.classifier(features)
        out = {'class': class_scores}
        if self.domain_adap:
            if p_epoch is None:
                lm = 1.0
            else:
                lm = self.domain_adapt_schedule(p_epoch)
            rev_features = GradientReversalFunction.apply(
                features, lm
            )
            domain_scores = self.dom_classifier(rev_features)
            out['domain'] = domain_scores
            out['features'] = features
        return out

    def train_mod(self, train_loader, criterion, optimizer, p_epoch=None):
        self.train()
        device = next(self.parameters()).device

        for X, y, dom, source_split in train_loader:
            X = X.to(device)
            y = y.to(device)
            dom = dom.to(device)
            res = self(X, p_epoch)
            loss = criterion[0](res['class'], y)
            if self.domain_adap:
                alpha = 1.0
                loss += alpha*criterion[1](res['domain'], dom)
                beta = 1.0
                loss += beta*criterion[2](
                    res['features'][source_split],
                    res['features'][~source_split]
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test_mod(self, loader, criterion, train=False):
        self.eval()
        device = next(self.parameters()).device
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, y, dom, source_split in loader:
                X = X.to(device)
                y = y.to(device)
                dom = dom.to(device)
                res = self(X)
                b_loss = criterion[0](res['class'], y)
                pred = res['class'].argmax(dim=1, keepdim=True)
                if train:
                    if self.domain_adap:
                        alpha = 1.0
                        b_loss += alpha*criterion[1](res['domain'], dom)
                        beta = 1.0
                        b_loss += beta*criterion[2](
                            res['features'][source_split],
                            res['features'][~source_split]
                        )
                loss += b_loss.item()
                correct += pred.eq(y.view_as(pred)).sum().item()

        loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        return loss, accuracy

    def get_summary(self, input):
        device = next(self.parameters()).device
        summary(
            self, (input[0].shape[1], input[0].shape[2]),
            device=device.type
        )


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer.
    This layer is used to reverse the gradient during backpropagation
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MMDLoss(nn.Module):

    def __init__(self, bandwith):
        super().__init__()
        self.bandwith = bandwith

    def kernel(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        nom = -L2_distances[None, ...]
        denom = (2 * self.bandwith ** 2)  # [:, None, None]
        return torch.exp(nom / denom).sum(dim=0)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
