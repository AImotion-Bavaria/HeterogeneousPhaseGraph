import numpy as np
from pathlib import Path
from itertools import product
import logging

import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import HeteroData, Data, Batch

import data.rockdrill_load as rd_load


class RockdrillDataPipeline():
    def __init__(self, force_reload):
        self.FORCE_RELOAD = force_reload

    def find_acceleration_phases(self, X_po, settings_ref, inds):
        """
        Find the index of the start and end of the retardation phase
        based on reference data.
        Rearward acceleration | Retardation | Forward acceleration

        Parameters:
        ----------
        X_po : numpy array
            Array containing the reference signal from the po sensor.
        settings_ref : numpy array
            Array containing the setting labels for the reference data.
        inds : list
            List of strings containing the indices of the reference data.

        Returns:
        ----------
        tuple
            Tuple containing two dictionaries. The first list contains the
            index of the end of the retardation phase for each setting. The
            second list contains the index of the start of the forward
            acceleration phase for each setting.
        """
        accel_rear_end = {}
        accel_for_start = {}
        for i in inds:
            accel_rear_end[i] = np.nanargmax(
                X_po[settings_ref == i, :].mean(axis=0)
            )
            accel_for_start[i] = accel_rear_end[i] + 100
        return accel_rear_end, accel_for_start

    def split_data_by_phases(
            self, X, y, settings, accel_rear_end=None, accel_for_start=None
    ):
        """
        Split data into three phases based on the indices of the start
        and end of the retardation phase.

        Parameters:
        ----------
        X: numpy array
            Array containing the signals. The first dimension is the sample
            number, the second dimension is the sensor and the third dimension
            is the sample.
        y : numpy array
            Array containing the target labels.
        settings : numpy array
            Array containing the setting labels.
        accel_rear_end : dict
            Dictionary containing the index of
            the end of the retardation phase for each setting.
        accel_for_start : dict
            Dictionary containing the index of the start of the forward
            acceleration phase for each setting.

        Returns:
        ----------
        X_phases_ref: dict
            Dictionary containing the stacked signals split into three phases.
        y_phases_ref: np.array
            Fault label to each X entry.
        settings_phases_ref: np.array
            Settings label to each X entry.
        """
        # Split data into phases according to ref data
        inds = list(np.unique(settings))
        sensors = ['pdmp', 'pin', 'po']
        phases = ['p1', 'p2', 'p3']
        X_phases_ref = {
            s: {
                p: {
                    i: [] for i in inds
                } for p in phases
            } for s in sensors
        }
        y_phases_ref = []

        for i in inds:
            phase1 = accel_rear_end[i]
            phase2 = accel_for_start[i]
            i_settings = (settings == i)
            for j, s in enumerate(sensors):
                X_phases_ref[s]['p1'][i] = X[i_settings, j, :phase1]
                X_phases_ref[s]['p2'][i] = X[i_settings, j, phase1:phase2]
                X_phases_ref[s]['p3'][i] = X[i_settings, j, phase2:]
            y_phases_ref.append(y[i_settings])
        y_phases_ref = np.concatenate(y_phases_ref)

        return X_phases_ref, y_phases_ref

    def align_arrays(
            self, arrays, stack=True, fill_value=np.nan, max_cols=None
    ):
        """
        Align numpy arrays by padding or truncating them to have the same
        number of columns according to the maximum number of columns among
        all arrays or max_cols.

        Parameters:
        ----------
        arrays : list of numpy arrays
            List containing numpy arrays to be padded.
        stack : bool, default=True
            If True, the padded arrays are stacked along the first dimension.
        fill_value : float, default=np.nan
            Value to use for padding.
        max_cols : int, default=None
            Maximum number of columns in the padded arrays. If None, the
            maximum number of columns among all arrays is used.

        Returns:
        ----------
        numpy array
            Stacked and padded arrays if stack=True, otherwise a list of padded
            arrays.
        """
        if max_cols is None:
            # Find the maximum number of columns among all arrays
            max_cols = max(array.shape[-1] for array in arrays)

        # Pad each array to have the same number of columns
        out_array = []
        for array in arrays:
            if array.shape[-1] < max_cols:
                padded_array = np.full(
                    (*array.shape[:-1], max_cols), fill_value
                )
                padded_array[..., :array.shape[-1]] = array
                out_array.append(padded_array)
            else:
                out_array.append(array[..., :max_cols])

        # Stack the padded arrays along the first dimension
        if stack:
            return np.concatenate(out_array, axis=0)
        else:
            return out_array

    def align_phase_data(
            self, X_phases_ref, max_cols=None
            ):
        """
        Align phase data by padding with zeros or truncating them to have the
        same number of columns according to the maximum number of columns among
        phases or max_cols.

        Parameters:
        ----------
        X_phases_ref : dict
            Dictionary containing the stacked signals split into three phases.
        max_cols : dict
            Dictionary containing the maximum number of columns for each phase.

        Returns:
        ----------
        X_phases_ref_padded
            Dictionary containing the stacked and padded signals split into
            three phases.
        """
        sensors = list(X_phases_ref.keys())
        phases = list(X_phases_ref[sensors[0]].keys())
        X_phases_ref_padded = {
            s: {
                p: {} for p in phases
            } for s in sensors
        }
        for phase in phases:
            for sensor in sensors:
                for ind in X_phases_ref[sensor][phase].keys():
                    X_temp = self.align_arrays(
                        [X_phases_ref[sensor][phase][ind]],
                        max_cols=max_cols[phase]
                    )
                    nan_mask = np.isnan(X_temp)
                    X_temp[nan_mask] = 0
                    X_phases_ref_padded[sensor][phase][ind] = X_temp

        return X_phases_ref_padded

    def gaussian_kernel_weight_dist(self, v1, v2, beta):
        '''
        Gaussian kernel function for distance between vectors.

        Parameters
        ----------
        v1: torch.tensor
            First vector.
        v2: torch.tensor
            Second vector.
        beta: float
            Kernel width.

        Returns
        ----------
        torch.tensor
            Gaussian kernel weight.
        '''
        return torch.exp(
            -torch.norm(v1 - v2, dim=1) ** 2 / (2 * beta ** 2)
        )

    def create_sensor_graph(self, X, y, settings, edge_attr=True):
        """
        Create a simple sensor graph without phases for each sample. Nodes
        represent the sensors and edges represent the connections between
        sensors.

        Parameters:
        ----------
        X : numpy array
            Array containing the signals. The first dimension is the sample
            number, the second dimension is the sensor and the third dimension
            is the sample.
        y : np.array
            Fault label to each X entry.
        settings : np.array
            Settings label to each X entry.
        edge_attr : bool, default=True
            If True, edge attributes are added to the graph.

        Returns:
        ----------
        graph_list: list
            List of graphs.
        """
        graph_list = []
        for i, y_i in enumerate(y):
            tmp_graph = Data()

            # Add sensor nodes
            tmp_graph.x = torch.tensor(X[i, :, :], dtype=torch.float32)

            # Add edges between sensors
            s_idx = list(range(tmp_graph.x.shape[0]))
            s_edges = list(product(s_idx, s_idx))
            s_edges = [e for e in s_edges if e[0] != e[1]]
            tmp_graph.edge_index = torch.tensor([
                s_edges
            ], dtype=torch.long).squeeze().T
            if edge_attr:
                p_node = tmp_graph.x
                p_edge_index = tmp_graph.edge_index
                tmp_graph.edge_attr = self.gaussian_kernel_weight_dist(
                    p_node[p_edge_index[0]], p_node[p_edge_index[1]], 2
                ).type(torch.float32).reshape(-1, 1)

            # Add graph label
            tmp_graph.y = torch.tensor(y_i, dtype=torch.long)

            # Add setting label
            tmp_graph.setting = torch.tensor(settings[i], dtype=torch.long)

            graph_list.append(tmp_graph)

        return graph_list

    def create_phase_graph(
            self, X, y, settings, graph_type='hetero'
            ):
        """
        Create a phase graph for each sample. Nodes represent the sensors and
        and the phases of the signals. Edges represent the connections
        between sensors in the same phase and the connections between
        the phases of the same sensor.

        Parameters:
        ----------
        X : dict
            Dictionary (X[s][p]) containing signals separated per sensors (s)
            and split into three phases (p).
        y : np.array
            Fault label to each X entry.
        settings : np.array
            Settings label to each X entry.
        graph_type : str, default='hetero'
            Type of graph. Options are
            'hetero' for heterogeneous graph and
            'homo' for homogenous graph.

        Returns:
        ----------
        graph_list: list
            List of graphs.
        """
        graph_list = []
        sensors = list(X.keys())
        phases = list(X[sensors[0]].keys())

        # Create a graph for each sample
        for i, y in enumerate(y):
            tmp_graph = HeteroData()

            # Add nodes
            for p in phases:
                tmp_graph[p].x = torch.stack([
                    torch.tensor(X[s][p][i])
                    for s in sensors
                ], axis=1).T.type(torch.float32)

            # Add edges
            # p1 <-> p1, p2 <-> p2, p3 <-> p3 ... for each sensor
            for p in phases:
                s_idx = list(range(len(sensors)))
                s_edges = list(product(s_idx, s_idx))
                s_edges = [e for e in s_edges if e[0] != e[1]]
                tmp_graph[p, f"e_{p}", p].edge_index = \
                    torch.tensor([
                        s_edges
                    ], dtype=torch.long).squeeze().T
                p_node = tmp_graph[p].x
                p_edge_index = tmp_graph[p, f"e_{p}", p].edge_index
                tmp_graph[p, f"e_{p}", p].edge_attr = \
                    self.gaussian_kernel_weight_dist(
                        p_node[p_edge_index[0]], p_node[p_edge_index[1]], 2
                    ).type(torch.float32).reshape(-1, 1)

            # p1 -> p2 -> p3 ... for each sensor
            for p_i, p in enumerate(phases[:-1]):
                p_next = phases[p_i+1]
                p_edges = list(range(len(sensors)))
                tmp_graph[p, f"e_{p}_{p_next}", p_next].edge_index = \
                    torch.tensor([
                        p_edges,
                        p_edges
                    ], dtype=torch.long)
                if graph_type == 'homo':
                    tmp_graph[p, f"e_{p}_{p_next}", p_next].edge_attr = \
                        torch.tensor(
                            [1 for _ in p_edges], dtype=torch.float32
                        ).reshape(-1, 1)

            # Add graph label
            tmp_graph['y'] = torch.tensor(y, dtype=torch.long)

            # Add setting label
            tmp_graph['setting'] = torch.tensor(
                settings[i], dtype=torch.long
            )

            if graph_type == 'homo':
                tmp_graph = tmp_graph.to_homogeneous()

            # Add to graph list
            graph_list.append(tmp_graph)

        return graph_list

    def save_graphs(self, graph_list, graph_path):
        """
        Save graph data.

        Parameters:
        ----------
        graph_list : list
            List of graphs.
        graph_path : str
            Path to save the graph data.
        """
        graph_batch = Batch.from_data_list(graph_list)
        logging.info("Saving phase graph data: %s.", graph_path)
        torch.save(graph_batch, graph_path)

    def load_graphs(self, graph_path):
        """
        Load graph data.

        Parameters:
        ----------
        graph_path : str
            Path to load the graph data.

        Returns:
        ----------
        graph_list : list
            List of graphs.
        """
        logging.info("Loading phase graph data: %s.", graph_path)
        graph_batch = torch.load(graph_path)
        graph_list = graph_batch.to_data_list()
        return graph_list

    def settings_map(self):
        return {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7
        }

    def run_pipeline(
            self, batch_size, train_s, test_s, data_type, data_path, **kwargs
    ):
        """
        Run data pipeline for Rock Drill Diagnostic Data Challenge data.

        Parameters:
        ----------
        batch_size : int
            Batch size for the dataloaders.
        train_s : list
            List of integers containing the training settings.
        test_s : list
            List of integers containing the test settings.
        data_type : str
            Type of data input. Options are 'phase_graph' or 'standard'.
        data_path : str
            Path to the directory with the raw data.
        **kwargs : dict
            Additional keyword arguments for the data pipeline.

        Returns:
        ----------
        dataloader : dict
            Dictionary containing the training and test dataloaders.
        """

        data_path = Path(data_path)
        graph_dir = Path('saved_graphs')
        max_train_length = 900

        # Load data
        X_tr, y_tr, settings_tr = rd_load.load(
            data_path, 'training',
            max_train_length, self.FORCE_RELOAD
        )
        X_v, y_v, settings_v = rd_load.load(
            data_path, 'validation',
            max_train_length, self.FORCE_RELOAD
        )
        X_te, y_te, settings_te = rd_load.load(
            data_path, 'test',
            max_train_length, self.FORCE_RELOAD
        )

        # Remap settings for domain classification
        setting_map = self.settings_map()
        settings_tr = np.array([setting_map[s] for s in settings_tr])
        settings_v = np.array([setting_map[s] for s in settings_v])
        settings_te = np.array([setting_map[s] for s in settings_te])
        train_s = [setting_map[s] for s in train_s]
        test_s = [setting_map[s] for s in test_s]

        # Combine data
        X_tr = np.stack([X_tr[s] for s in X_tr.keys()], axis=1)
        X_v = np.stack([X_v[s] for s in X_v.keys()], axis=1)
        X_te = np.stack([X_te[s] for s in X_te.keys()], axis=1)
        X_all = np.concatenate([X_tr, X_v, X_te], axis=0)
        for i in range(X_all.shape[1]):
            nan_mask = np.isnan(X_all[:, i, :])
            X_all[:, i, :][nan_mask] = 0
        y_all = np.concatenate([y_tr, y_v, y_te])
        settings_all = np.concatenate([settings_tr, settings_v, settings_te])

        # Split data into training and validation sets according to
        # settings
        X = {}
        settings = {}
        y = {}
        split_settings = {'train': train_s, 'test': test_s}
        for k in ['train', 'test']:
            rows = np.isin(settings_all, split_settings[k])
            settings[k] = settings_all[rows]
            X[k] = X_all[rows, :, :]
            y[k] = y_all[rows]

        if data_type == 'phase_graph':
            # Create graph for each sample. Nodes represent the sensors and
            # and the phases of the signals. Edges represent the connections
            # between sensors in the same phase and the connections between
            # the phases of the same sensor.

            if kwargs['phases'] == 'one':
                kwargs['graph_type'] = 'homo'

            # i+1 needed to adjust for 0 based indexing for domain
            # classification
            tr_set = ''.join([str(i+1) for i in train_s])
            te_set = ''.join([str(i+1) for i in test_s])
            g_set = (
                f"{kwargs['graph_type']}"
                f"_{kwargs['phases']}"
            )
            graph_file_tr = f"Tr{tr_set}_Te{te_set}_{g_set}.graph"
            graph_dir.mkdir(parents=True, exist_ok=True)
            graph_path = graph_dir / graph_file_tr
            graph_reload = self.FORCE_RELOAD or not graph_path.exists()

            if not graph_reload:
                graph_list_reload = self.load_graphs(graph_path)

            else:
                logging.info("Creating phase graph data.")

                # Get reference data
                X_ref, _, settings_ref = rd_load.load(
                    data_path, 'ref',
                    max_train_length, self.FORCE_RELOAD
                )
                settings_ref = np.array(
                    [setting_map[s] for s in settings_ref]
                )
                individuals = list(np.unique(settings_ref))

                # Find acceleration phases
                if kwargs['phases'] == 'acceleration':
                    accel_rear_end, accel_for_start = \
                        self.find_acceleration_phases(
                            X_ref['po'], settings_ref, individuals
                        )
                    phase_max_cols = {
                        'p1': 400,
                        'p2': 100,
                        'p3': 400
                    }
                elif kwargs['phases'] == 'equidistant':
                    phase_len = max_train_length // 3
                    accel_rear_end = {i: phase_len for i in individuals}
                    accel_for_start = {i: phase_len * 2 for i in individuals}
                    phase_max_cols = {
                        'p1': phase_len,
                        'p2': phase_len,
                        'p3': phase_len
                    }
                elif kwargs['phases'] == 'one':
                    # Simple sensor network without phases
                    phase_len = max_train_length
                    phase_max_cols = {'p1': max_train_length}
                else:
                    raise AttributeError(
                        "Invalid phase type. Choose either 'acceleration', "
                        "'equidistant' or 'one' for phases parameter."
                    )

            X_phases_ref = {}
            y_phases_ref = {}
            settings_phases_ref = {
                'train': None, 'test': None
            }
            graph_list = {}
            dataloader = {}

            for k in ['train', 'test']:

                graph_phases = kwargs['phases'] in [
                    'acceleration', 'equidistant'
                ]

                if graph_reload and graph_phases:
                    # Split data into acceleration phases based on
                    # reference signal
                    X_phases_ref[k], y_phases_ref[k] = \
                        self.split_data_by_phases(
                            X[k], y[k], settings[k],
                            accel_rear_end, accel_for_start
                        )

                    # Pad phase data
                    X_phases_ref[k] = self.align_phase_data(
                        X_phases_ref[k], phase_max_cols
                    )
                    for s in X_phases_ref[k].keys():
                        for p in X_phases_ref[k][s].keys():
                            tmp_sig = [
                                v for v in X_phases_ref[k][s][p].values()
                            ]
                            tmp_set = [
                                [k]*v.shape[0]
                                for k, v in X_phases_ref[k][s][p].items()
                            ]
                            X_phases_ref[k][s][p] = np.concatenate(tmp_sig)

                    settings_phases_ref[k] = np.concatenate(tmp_set)

                    # Create phase graph for each sample
                    graph_list[k] = self.create_phase_graph(
                        X_phases_ref[k], y_phases_ref[k],
                        settings_phases_ref[k], graph_type=kwargs['graph_type']
                    )

                    if k == 'test':
                        self.save_graphs(
                            [*graph_list['train'], *graph_list['test']],
                            graph_path
                        )

                elif graph_reload and (kwargs['phases'] == 'one'):
                    # Create simple sensor graph without phases for each sample
                    graph_list[k] = self.create_sensor_graph(
                        X[k], y[k], settings[k]
                    )

                    if k == 'test':
                        self.save_graphs(
                            [*graph_list['train'], *graph_list['test']],
                            graph_path
                        )

                else:
                    graph_list[k] = [
                        g for g in graph_list_reload
                        if g['setting'] in split_settings[k]
                    ]

                # Create dataloaders
                if k == 'test':
                    # Separate test data according to settings
                    graph_dict_s = {
                        s: [g for g in graph_list[k]
                            if g['setting'].item() == s]
                        for s in sorted(set(settings['test']))
                    }
                    dataloader[k] = {
                        s: GeoDataLoader(
                            graph_dict_s[s],
                            batch_size=batch_size, shuffle=False
                        ) for s in sorted(set(settings['test']))
                    }

                elif k == 'train':
                    dataloader[k] = GeoDataLoader(
                        graph_list[k], batch_size=batch_size, shuffle=True
                    )

            return dataloader['train'], dataloader['test']

        if data_type == 'standard':
            # Standard data input for CNN with 1D convolutions with
            # 3 channels (sensors).
            logging.info("Creating standard data input for CNN.")

            if kwargs['domain_adap']:
                # Use reference data for training
                X_ref, _, settings_ref = rd_load.load(
                    data_path, 'ref',
                    max_train_length, self.FORCE_RELOAD
                )
                settings_ref = np.array([setting_map[s] for s in settings_ref])
                y_ref = np.zeros(settings_ref.shape)
                for s in X_ref.keys():
                    nan_mask = np.isnan(X_ref[s])
                    X_ref[s][nan_mask] = 0
                X_ref = np.stack([v for v in X_ref.values()], axis=1)
                X['train'] = np.concatenate([X['train'], X_ref], axis=0)
                y['train'] = np.concatenate([y['train'], y_ref], axis=0)
                settings['train'] = np.concatenate(
                    [settings['train'], settings_ref], axis=0
                )

            # Create dataloaders
            dataloader = {}
            dataloader['train'] = DataLoader(
                dataset=TensorDataset(
                    torch.tensor(X['train'], dtype=torch.float32),
                    torch.tensor(y['train'], dtype=torch.long),
                    torch.tensor(settings['train'], dtype=torch.long),
                    torch.tensor(np.isin(settings['train'], train_s))
                ),
                batch_size=batch_size,
                shuffle=True
            )

            dataloader['test'] = {}
            for s in sorted(set(settings['test'])):
                rows = settings['test'] == s
                dataloader['test'][s] = DataLoader(
                    dataset=TensorDataset(
                        torch.tensor(X['test'][rows], dtype=torch.float32),
                        torch.tensor(y['test'][rows], dtype=torch.long),
                        torch.tensor(settings['test'][rows], dtype=torch.long),
                        torch.tensor(np.isin(settings['test'][rows], test_s))
                    ),
                    batch_size=batch_size,
                    shuffle=False
                )

            return dataloader['train'], dataloader['test']
