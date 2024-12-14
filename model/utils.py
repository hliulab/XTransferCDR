import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(torch.nn.Module):
    def __init__(
        self,
        sizes,
        append_layer_width=None,
        append_layer_position=None,
        batch_norm=False,
        last_activate="relu",
        dropout=0.2,
        dtype=torch.float32,
    ):
        super().__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1], dtype=dtype),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                nn.Dropout(dropout) if s < len(sizes) - 2 else None,
                torch.nn.ReLU() if s < len(sizes) - 2 else None,
            ]

        if append_layer_width is None or append_layer_position == "first":
            layers += [self.activate(last_activate)]

        elif append_layer_width is not None and (
            append_layer_position == "last" or append_layer_position == "both"
        ):
            layers += [
                torch.nn.BatchNorm1d(sizes[-1]) if batch_norm else None,
                nn.Dropout(dropout),
                torch.nn.ReLU(),
            ]

        layers = [layer for layer in layers if layer is not None]

        if append_layer_width:
            assert append_layer_position in ("first", "last", "both")
            if append_layer_position == "first" or append_layer_position == "both":
                layers_dict = OrderedDict()
                layers_dict["first_linear"] = torch.nn.Linear(
                    append_layer_width, sizes[0], dtype=dtype
                )
                if batch_norm:
                    layers_dict["first_bn1d"] = torch.nn.BatchNorm1d(sizes[0])

                layers_dict["first_dropout"] = nn.Dropout(dropout)
                layers_dict["first_relu"] = torch.nn.ReLU()
                for i, module in enumerate(layers):
                    layers_dict[str(i)] = module

            if append_layer_position == "last" or append_layer_position == "both":
                if append_layer_position != "both":
                    layers_dict = OrderedDict(
                        {str(i): module for i, module in enumerate(layers)}
                    )

                layers_dict["last_linear"] = torch.nn.Linear(
                    sizes[-1], append_layer_width, dtype=dtype
                )

                layers_dict["last_activate"] = self.activate(last_activate)

        else:
            layers_dict = OrderedDict(
                {str(i): module for i, module in enumerate(layers)}
            )

        self.network = torch.nn.Sequential(layers_dict)

    def activate(self, activate):
        activate_dict = {
            "leakyrelu": torch.nn.LeakyReLU(),
            "relu": torch.nn.ReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "softmax": torch.nn.Softmax(dim=-1),
        }

        return activate_dict.get(activate, torch.nn.Identity())

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, dim: int, batch_norm: bool, dropout=0.2, dtype=torch.float32
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(dim) if batch_norm else nn.Identity()
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.dropout1(x1)
        x1 = self.relu1(x1)
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.dropout2(x2)
        x2 = self.relu2(x2)
        return x2 + x
