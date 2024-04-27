import yaml
import copy
from single_run import main
from pathlib import Path
from datetime import datetime

if __name__ == "__main__":

    # Read config
    with open(
        'config_files/config_experiments.yaml', 'r'
    ) as f:
        config = yaml.safe_load(f)
    target_dir = f"{datetime.now().strftime('%Y-%m-%d_%H:%M')}_experiments"

    if config['data']['test_s'] == 'rest':
        # Fill up test settings
        all_settings = [1, 2, 3, 4, 5, 6, 7, 8]
        config['data']['test_s'] = []
        for s in config['data']['train_s']:
            config['data']['test_s'].append(
                [ts for ts in all_settings if ts not in s]
            )

    # Save experiments config
    folder = f"results/{target_dir}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(f"{folder}/00_config_experiments.yaml", 'w') as f:
        yaml.dump(config, f)

    n_runs = (
        len(config['random_state'])
        * len(config['data']['train_s'])
        * (len(config['model_params']['SWING_CNN']['domain_adap'])
           + len(config['model_params']['GNN']['conv_pre']))
    )

    def single_experiment_run(conf):
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"----- Experiment {run}/{n_runs} "
            f"----- {time} -----"
        )
        conf_in = copy.deepcopy(conf)
        mod = conf_in['model_type']
        print(
            f"{mod}: {conf_in['model_params'][mod]} "
            f"| Training settings: {conf_in['data']['train_s']} "
            f"| Random state: {conf_in['random_state']}"
        )
        main(conf_in, target_dir)

    # Run experiments for each configuration
    run = 1
    single_run_config = copy.deepcopy(config)
    for random_state in config['random_state']:
        single_run_config['random_state'] = random_state
        for t, train_s in enumerate(config['data']['train_s']):
            single_run_config['data']['train_s'] = train_s
            single_run_config['data']['test_s'] = config['data']['test_s'][t]
            for model in config['model_type']:
                params = config['model_params'][model]
                single_run_config['epochs'] = config['epochs'][model]
                single_run_config['model_type'] = model
                if model == 'SWING_CNN':
                    for domain_adap in params['domain_adap']:
                        single_run_config['model_params'][model] = {
                            'domain_adap': domain_adap
                        }
                        single_experiment_run(single_run_config)
                        run += 1
                elif model == 'GNN':
                    for i, _ in enumerate(params['conv_pre']):
                        single_run_config['model_params'][model] = {
                            'conv_pre': params['conv_pre'][i],
                            'gnn_layer_1': params['gnn_layer_1'],
                            'gnn_layer_2': params['gnn_layer_2'],
                            'graph_type': params['graph_type'][i],
                            'phases': params['phases'][i]
                        }
                        single_experiment_run(single_run_config)
                        run += 1
