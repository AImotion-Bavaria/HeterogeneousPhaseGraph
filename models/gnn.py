import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn


class BaseGNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_classes = 11

    def forward(self):
        raise NotImplementedError

    def init_preprocessing_cnn(
            self, in_channels, n_pooling_layers=3
            ):
        # Check if n_pooling_layers is valid
        try:
            assert n_pooling_layers in [0, 1, 2, 3]
        except AssertionError:
            raise ValueError(
                "n_pooling_layers must be 0, 1, 2, or 3"
            )
        layers = [
            nn.Conv1d(in_channels, 32, 5),
            nn.ReLU()
        ]
        if n_pooling_layers == 3:
            layers.append(nn.MaxPool1d(2))
        layers.extend([
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, 5),
            nn.ReLU()
        ])
        if n_pooling_layers >= 2:
            layers.append(nn.MaxPool1d(2))
        layers.extend([
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 8, 5),
            nn.ReLU()
        ])
        if n_pooling_layers >= 1:
            layers.append(nn.MaxPool1d(2))
        layers.extend([
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 1, 5),
            nn.ReLU(),
            nn.BatchNorm1d(1)
        ])
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def init_homo_gnn_module(
            self, layer_1=512, layer_2=256
    ):
        gnn_1 = geo_nn.GATv2Conv(
            -1, layer_1, edge_dim=1,
            add_self_loops=False, share_weights=False
        )
        gnn_1_act = nn.PReLU()
        gnn_2 = geo_nn.GATv2Conv(
            layer_1, layer_2, edge_dim=1,
            add_self_loops=False, share_weights=False
        )
        gnn_2_act = nn.PReLU()
        return gnn_1, gnn_1_act, gnn_2, gnn_2_act

    def init_hetero_gnn_module(
            self, layer_1=512, layer_2=256
    ):
        gnn_1 = geo_nn.HeteroConv({
            ('p1', 'e_p1', 'p1'): geo_nn.conv.GATv2Conv(
                -1, layer_1, edge_dim=1,
                add_self_loops=False, share_weights=False
            ),
            ('p2', 'e_p2', 'p2'): geo_nn.conv.GATv2Conv(
                -1, layer_1, edge_dim=1,
                add_self_loops=False, share_weights=False
            ),
            ('p3', 'e_p3', 'p3'): geo_nn.conv.GATv2Conv(
                -1, layer_1, edge_dim=1,
                add_self_loops=False, share_weights=False
            ),
            ('p1', 'e_p1_p2', 'p2'): geo_nn.conv.SAGEConv(-1, layer_1),
            ('p2', 'e_p2_p3', 'p3'): geo_nn.conv.SAGEConv(-1, layer_1)
        }, aggr='sum')
        gnn_1_act = nn.ModuleDict({
            k: nn.PReLU() for k in ['p1', 'p2', 'p3']
        })

        gnn_2 = geo_nn.HeteroConv({
            ('p1', 'e_p1', 'p1'): geo_nn.conv.GATv2Conv(
                layer_1, layer_2, edge_dim=1,
                add_self_loops=False, share_weights=False
            ),
            ('p2', 'e_p2', 'p2'): geo_nn.conv.GATv2Conv(
                layer_1, layer_2, edge_dim=1,
                add_self_loops=False, share_weights=False
            ),
            ('p3', 'e_p3', 'p3'): geo_nn.conv.GATv2Conv(
                layer_1, layer_2, edge_dim=1,
                add_self_loops=False, share_weights=False
            ),
            ('p1', 'e_p1_p2', 'p2'): geo_nn.conv.SAGEConv(layer_1, layer_2),
            ('p2', 'e_p2_p3', 'p3'): geo_nn.conv.SAGEConv(layer_1, layer_2)
        }, aggr='sum')
        gnn_2_act = nn.ModuleDict({
            k: nn.PReLU() for k in ['p1', 'p2', 'p3']
        })
        return gnn_1, gnn_1_act, gnn_2, gnn_2_act

    def init_classifier(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, self.n_classes)
        )

    def get_summary(self, input):
        input.to(next(self.parameters()).device)
        geo_nn.summary(self, input, max_depth=5)
        # Needs to be done twice, else it doesn't print the complete summary.
        print(geo_nn.summary(self, input, max_depth=5))

    def train_mod(self, train_loader, criterion, optimizer, **kwargs):
        self.train()
        device = next(self.parameters()).device
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out = self(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    def test_mod(self, loader, criterion, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        loss = 0
        correct = 0
        with torch.no_grad():
            for data in loader:
                data.to(device)
                out = self(data)
                loss += criterion(out, data.y).item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(data.y.view_as(pred)).sum().item()

        loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        return loss, accuracy


class GNN(BaseGNN):

    def __init__(
            self, conv_pre=True,
            gnn_layer_1=512, gnn_layer_2=256,
            graph_type='hetero', phases='acceleration'
    ):
        super().__init__()

        self.conv_pre = conv_pre
        self.gnn_layer_1 = gnn_layer_1
        self.gnn_layer_2 = gnn_layer_2
        self.graph_type = graph_type
        self.phases = phases
        self.n_sensors = 3
        self.pre_pooling = [3, 1, 3]

        if conv_pre:
            if graph_type == 'hetero':
                self.conv_pre_mod = nn.ModuleDict({
                    p: nn.ModuleDict({
                        str(s): self.init_preprocessing_cnn(
                            in_channels=1,
                            n_pooling_layers=self.pre_pooling[i]
                        )
                        for s in range(self.n_sensors)
                    }) for i, p in enumerate(['p1', 'p2', 'p3'])
                })
            elif graph_type == 'homo':
                self.conv_pre_mod = nn.ModuleDict({
                    str(s): self.init_preprocessing_cnn(
                        in_channels=1,
                        # n_pooling_layers=min(pre_pooling)
                        n_pooling_layers=max(self.pre_pooling)
                    )
                    for s in range(self.n_sensors)
                })

        if graph_type == 'hetero':
            self.gnn_1, self.gnn_1_act, self.gnn_2, self.gnn_2_act = \
                self.init_hetero_gnn_module(
                    layer_1=self.gnn_layer_1, layer_2=self.gnn_layer_2
                )
        elif graph_type == 'homo':
            self.gnn_1, self.gnn_1_act, self.gnn_2, self.gnn_2_act = \
                self.init_homo_gnn_module(
                    layer_1=self.gnn_layer_1, layer_2=self.gnn_layer_2
                )

        if graph_type == 'hetero':
            in_channels = 3
        elif graph_type == 'homo':
            in_channels = 1

        in_clf = in_channels * self.gnn_layer_2

        self.clf_mod = self.init_classifier(in_clf)

    def forward(self, graph_batch):
        if self.graph_type == 'hetero':
            # Heterogenous graph input
            if self.conv_pre:
                # Cnn preprocessing module for each phase and sensor
                x_phase_sensor = {
                    k: {
                        str(s): self.conv_pre_mod[k][str(s)](
                            graph_batch.x_dict[k][
                                s::self.n_sensors, None, :
                            ]
                        )
                        for s in range(self.n_sensors)
                    } for k in ['p1', 'p2', 'p3']
                }
                # Combine separate forward passes to recreate original
                # graph structure
                x_dict_pre = {}
                for p, v_p in x_phase_sensor.items():
                    tmp = torch.empty((
                        v_p['0'].shape[0]*self.n_sensors,
                        v_p['0'].shape[1]
                    )).to((next(self.parameters()).device))
                    for s, v_s in v_p.items():
                        tmp[int(s)::self.n_sensors, :] = v_s
                    x_dict_pre[p] = tmp
            else:
                x_dict_pre = graph_batch.x_dict

            # GNN module
            graph_in = [
                graph_batch.edge_index_dict, graph_batch.edge_attr_dict
            ]
            x_dict_1 = self.gnn_1(x_dict_pre, *graph_in)
            x_dict_1 = {
                k: self.gnn_1_act[k](v)
                for k, v in x_dict_1.items()
            }
            x_dict_2 = self.gnn_2(x_dict_1, *graph_in)
            x_dict_2 = {
                k: self.gnn_2_act[k](v)
                for k, v in x_dict_2.items()
            }
            gnn_out = []
            for p in x_dict_2.keys():
                gnn_out.append(geo_nn.global_add_pool(
                    x_dict_2[p],
                    batch=graph_batch[p].batch
                ))

            # Concatenate output of GNN
            graph_emb = torch.cat(gnn_out, dim=1)

        elif self.graph_type == 'homo':
            # Homogeneous graph input
            if self.conv_pre:
                # Cnn preprocessing module for each sensor
                x_pre_sensor = {
                    str(s): self.conv_pre_mod[str(s)](
                        graph_batch.x[s::self.n_sensors, None, :]
                    )
                    for s in range(self.n_sensors)
                }
                # Combine separate forward passes to recreate original
                # graph structure
                x_pre = torch.empty((
                    x_pre_sensor['0'].shape[0]*self.n_sensors,
                    x_pre_sensor['0'].shape[1]
                )).to((next(self.parameters()).device))
                for s, v_s in x_pre_sensor.items():
                    x_pre[int(s)::self.n_sensors, :] = v_s
            else:
                x_pre = graph_batch.x

            # GNN module
            graph_in = [graph_batch.edge_index, graph_batch.edge_attr]
            x_1 = self.gnn_1(x_pre, *graph_in)
            x_1 = self.gnn_1_act(x_1)
            x_2 = self.gnn_2(x_1, *graph_in)
            x_2 = self.gnn_2_act(x_2)
            graph_emb = geo_nn.global_add_pool(x_2, batch=graph_batch.batch)

        # Classifier module
        clf_out = self.clf_mod(graph_emb)

        return clf_out
