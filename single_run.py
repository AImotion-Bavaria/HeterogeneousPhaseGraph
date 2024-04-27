import torch
import logging
import yaml
from datetime import datetime
from pathlib import Path
import os
import random
import numpy as np

from tqdm import tqdm

from data.rockdrill_preprocessing import RockdrillDataPipeline
import models.cnn as cnn
import models.gnn as gnn


def set_seed(seed=42):
    '''Sets the seed so results are the same every
    time we run. This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(config, target_dir='single_run'):

    set_seed(config['random_state'])

    res = {}
    res['start_time'] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    if config['verbose']:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    config['model_params'] = config['model_params'][config['model_type']]

    # Set device
    device = "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info("Using device: %s", device)

    model_name = config['model_type']

    # Check if model exists and set data type
    try:
        model = getattr(gnn, model_name)
        config['data']['data_type'] = 'phase_graph'
        criterion = torch.nn.CrossEntropyLoss()
    except AttributeError:
        try:
            model = getattr(cnn, model_name)
            config['data']['data_type'] = 'standard'
            criterion = [torch.nn.CrossEntropyLoss(), None, None]
            if config['model_params']['domain_adap']:
                criterion[1] = torch.nn.CrossEntropyLoss()
                criterion[2] = cnn.MMDLoss(bandwith=1.0)
        except AttributeError:
            logging.error(
                "Invalid model type: %s. Exiting.",
                config['model_type']
            )
            return

    # Load data
    data_pipeline = RockdrillDataPipeline(config['force_reload'])
    train_dataloader, test_dataloader_dict = data_pipeline.run_pipeline(
        **config['data'],
        **config['model_params']
    )

    # Initialize model
    model = model(**config['model_params']).to(device)

    logging.info(
        "Using %s model with %s data",
        config['model_type'], config['data']['data_type']
    )

    # Set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )

    if config['verbose']:
        # Get summary of model
        d_example = next(iter(train_dataloader))
        model.get_summary(d_example)
    res['model'] = str(model)

    # Initialize metrics dictionary
    # s+1 is used to adjust for 0-based indexing for domain classification
    metrics = {
        'train_loss': [], 'train_acc': [],
        **{f'test_loss_{s+1}': [] for s in test_dataloader_dict.keys()},
        **{f'test_acc_{s+1}': [] for s in test_dataloader_dict.keys()},
        'test_loss_all': [], 'test_acc_all': []
    }

    if config['verbose']:
        epoch_progress = range(1, config['epochs']+1)
    else:
        epoch_progress = tqdm(range(1, config['epochs']+1))

    logging.info(
        "%s config: %s", model_name, config['model_params']
    )

    # Train model
    for epoch in epoch_progress:
        p_epoch = (epoch-1) / config['epochs']
        model.train_mod(
            train_dataloader, criterion, optimizer, p_epoch=p_epoch
        )
        train_loss, train_acc = model.test_mod(
            train_dataloader, criterion, train=True
        )
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)

        # Evaluate model on each test setting separately
        len_test, test_loss, test_acc = 0, 0, 0
        for s, test_dataloader_s in test_dataloader_dict.items():
            test_loss_s, test_acc_s = model.test_mod(
                test_dataloader_s, criterion
            )
            len_test += len(test_dataloader_s)
            test_loss += test_loss_s * len(test_dataloader_s)
            test_acc += test_acc_s * len(test_dataloader_s)
            metrics[f'test_loss_{s+1}'].append(test_loss_s)
            metrics[f'test_acc_{s+1}'].append(test_acc_s)
        metrics['test_loss_all'].append(test_loss / len_test)
        metrics['test_acc_all'].append(test_acc / len_test)

        # Log results
        print_res = [
            f'Epoch: {epoch:03d}, '
            f'Train Acc: {train_acc:.2f}'  # , Train Loss: {train_loss:.2f}'
        ]
        print_res.extend([
            f"| Test {' '.join(k.split('_')[-2:]).title()}: {v[-1]:.2f}"
            for k, v in metrics.items() if 'test_acc' in k
        ])

        if config['verbose']:
            logging.info(' '.join(print_res))
        else:
            epoch_progress.set_postfix_str(' '.join(print_res))

    # Save results
    res['metrics'] = metrics
    res['config'] = config
    res['config']['model_params'] = res['config']['model_params']
    res['end_time'] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    folder = Path(f"results/{target_dir}/results")
    folder.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"{config['model_type']}"
        f"_{config['data']['data_type']}"
        f"_{res['end_time']}.yaml"
    )
    with open(folder / file_name, 'w') as f:
        yaml.dump(res, f)
    logging.info("Results saved to %s", file_name)


if __name__ == "__main__":

    with open('config_files/config_single_run.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config, target_dir='single_run')
