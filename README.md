# Encoding Machine Phase Information into Heterogeneous Graphs for Adaptive Fault Diagnosis

This repository belongs to the paper *Encoding Machine Phase Information into Heterogeneous Graphs for Adaptive Fault Diagnosis* submitted to the 2024 *IEEE International Conference on Emerging Technologies and Factory Automation* with the following abstract:

> Machinery fault diagnosis is increasingly reliant on data-driven algorithms, yet struggles with adapting to unseen operating conditions. To address this, we propose integrating phase information into heterogeneous graphs for fault diagnostics with Graph Neural Networks. Our method involves identifying the distinct phases that a machine undergoes within a cycle and segmenting the signals accordingly. These segmented signals are then represented in a graph of multiple connected sensor networks with diverse node and edge types. Prior to graph classification with a Graph Neural Network, individual Convolutional Neural Networks preprocess the node attributes to account for their unique characteristics. The evaluation in a domain adaptation setting demonstrates the effectiveness of our approach, offering insights into improving the robustness and domain adaptability of fault diagnostic models.

## Running the code

To run the code install all required packages:

```
pip install -r requirements.txt
```

Then download  the 2022 PHM Conference Data Challenge data set on fault classification for a hydraulic rockdrill.
The best way to get the data is by downloading from Kaggle: https://www.kaggle.com/datasets/erikjakobsson/fault-classification-in-rock-drills

Adjust the value  ``data_path`` in the configuration files in ``config_files`` to the path, where you saved the data.

### Single run

Specify the parameters of the single run under ``config_files/config_single_run.yaml``. Then run:

```
python single_run.py
```

### Experiments

Rerun the experiments from the paper by executing:

```
python run_experiments.py
```

For running a subset of the experiments adjust ``config_files/config_experiments.yaml`` accordingly.

## Results

### Model configurations

| Name     | Type | Node Preprocessing | Graph         | Phase        |
|----------|------|--------------------|---------------|--------------|
| GFHeA    | GNN  | False              | Heterogeneous | Acceleration |
| GFHeU    | GNN  | False              | Heterogeneous | Uniform      |
| GFHoA    | GNN  | False              | Homogeneous   | Acceleration |
| GFHoU    | GNN  | False              | Homogeneous   | Uniform      |
| GFHoO    | GNN  | False              | Homogeneous   | One          |
| GTHeA    | GNN  | True               | Heterogeneous | Acceleration |
| GTHeU    | GNN  | True               | Heterogeneous | Uniform      |
| GTHoA    | GNN  | True               | Homogeneous   | Acceleration |
| GTHoU    | GNN  | True               | Homogeneous   | Uniform      |
| GTHoO    | GNN  | True               | Homogeneous   | One          |

| Name     | Type | Domain Adaptation  |
|----------|------|--------------------|
| SWING-DA | CNN  | True               |
| SWING    | CNN  | False              |

### Average accuracies with standard deviations per model and train domain size

| Model    |  $\|S\|$=1 |  $\|S\|$=2 |  $\|S\|$=3  |
|----------|:----------:|:----------:|:-----------:|
| GFHeA    | 41.4 (6.4) | 49.1 (3.0) | 58.9 (6.2)  |
| GFHeU    | 34.1 (4.8) | 40.3 (5.1) | 54.2 (5.6)  |
| GFHoA    | 38.3 (4.4) | 45.9 (6.7) | 58.7 (7.6)  |
| GFHoU    | 35 (3.6)   | 38.5 (5.9) | 55.1 (5.5)  |
| GFHoO    | 33.4 (4.0) | 35.5 (4.3) | 49.3 (4.6)  |
| GTHeA    | **53.3 (6.4)** | **60.3 (7.7)** | **78.4 (6.0)**  |
| GTHeU    | 39.7 (4.4) | 48.3 (6.9) | 64.6 (7.2)  |
| GTHoA    | 45.8 (6.4) | 55.4 (8.6) | 69.7 (11.2) |
| GTHoU    | 39.2 (6.4) | 46.4 (6.8) | 63.6 (7.7)  |
| GTHoO    | 38.1 (6.2) | 46.7 (6.1) | 64.6 (4.9)  |
| SWING    | 38.4 (4.3) | 45.5 (7.8) | 70.2 (6.6)  |
| SWING-DA | 37.5 (7.0) | 46.4 (7.5) | 64.3 (10.0) |

### Detailed results for train domain size $|S| = 1$

| Model    |   $S=\lbrace1\rbrace$  |   $S=\lbrace5\rbrace$   |   $S=\lbrace8\rbrace$  |
|----------|:----------:|:-----------:|:----------:|
| GFHeA    | 36.4 (3.3) | 44.9 (9.3)  | 42.8 (6.7) |
| GFHeU    | 29.6 (4.2) | 37.5 (5.7)  | 35.1 (4.5) |
| GFHoA    | 33.6 (2.7) | 43.2 (6.7)  | 38.0 (4.0) |
| GFHoU    | 30.4 (3.5) | 39.6 (2.5)  | 35.1 (4.7) |
| GFHoO    | 32.1 (2.7) | 36 (6.3)    | 32.1 (2.9) |
| GTHeA    | **46.5 (3.5)** | **57.4 (10.6)** | **55.9 (5.0)** |
| GTHeU    | 29.8 (3.4) | 49.3 (5.7)  | 39.9 (4.2) |
| GTHoA    | 39.1 (7.4) | 52.9 (7.4)  | 45.6 (4.5) |
| GTHoU    | 30.4 (3.5) | 45.5 (7.8)  | 41.7 (8.1) |
| GTHoO    | 28.5 (5.0) | 46.5 (7.0)  | 39.2 (6.6) |
| SWING    | 28.3 (3.7) | 41.1 (4.2)  | 46.0 (5.0) |
| SWING-DA | 26.5 (4.8) | 44.8 (8.3)  | 41.3 (7.8) |

### Detailed results for train domain size $|S| = 2$

| Model    | $S=\lbrace1,2\rbrace$ |  $S=\lbrace4,5\rbrace$ |  $S=\lbrace7,8\rbrace$  |
|----------|:----------:|:----------:|:-----------:|
| GFHeA    | 37.9 (2.8) | 57.4 (2.9) | 52.1 (3.2)  |
| GFHeU    | 34.9 (4.5) | 45.4 (4.7) | 40.6 (6.0)  |
| GFHoA    | 35.8 (6.6) | 55.3 (5.8) | 46.6 (7.7)  |
| GFHoU    | 31.6 (5.0) | 45.7 (5.7) | 38.2 (7.1)  |
| GFHoO    | 28.0 (6.4) | 42.5 (2.8) | 35.9 (3.7)  |
| GTHeA    | **50.9 (11)**  | **63.9 (6.1)** | **66.0 (6.1)**  |
| GTHeU    | 42.5 (5.4) | 54.9 (6.8) | 47.6 (8.5)  |
| GTHoA    | 51.0 (6.3) | 57.7 (11.0)| 57.6 (8.5)  |
| GTHoU    | 39.0 (4.8) | 54.8 (5.1) | 45.5 (10.4) |
| GTHoO    | 38.6 (7.4) | 54.8 (4.9) | 46.7 (6.0)  |
| SWING    | 25.6 (6.7) | 53.2 (9.7) | 57.7 (7.0)  |
| SWING-DA | 34.3 (5.3) | 54.8 (5.6) | 50.2 (11.5) |

### Detailed results for train domain size $|S| = 3$

| Model    | $S=\lbrace1,2,3\rbrace$ | $S=\lbrace4,5,6\rbrace$ | $S=\lbrace6,7,8\rbrace$ |
|----------|:-----------:|:-----------:|:-----------:|
| GFHeA    | 51.0 (8.7)  | 67.0 (3.7)  | 58.6 (6.1)  |
| GFHeU    | 50.1 (3.4)  | 64.9 (2.7)  | 47.8 (10.6) |
| GFHoA    | 58.5 (2.9)  | 61.3 (9.4)  | 56.2 (10.5) |
| GFHoU    | 53.0 (3.4)  | 61.4 (8.5)  | 50.9 (4.7)  |
| GFHoO    | 43.5 (4.8)  | 59.3 (5.2)  | 45.2 (3.9)  |
| GTHeA    | **74.0 (6.0)**  | **82.9 (6.1)**  | **78.3 (5.8)**  |
| GTHeU    | 65.7 (3.3)  | 70.9 (9.1)  | 57.1 (9.3)  |
| GTHoA    | 66.3 (7.8)  | 74.3 (16.2) | 68.6 (9.5)  |
| GTHoU    | 63.2 (6.0)  | 70.4 (10.3) | 57.2 (6.7)  |
| GTHoO    | 60.1 (6.0)  | 74.4 (4.8)  | 59.3 (3.9)  |
| SWING    | 64.0 (4.9)  | 75.3 (6.4)  | 71.4 (8.7)  |
| SWING-DA | 60.1 (10.9) | 71.3 (6.7)  | 61.4 (12.4) |
