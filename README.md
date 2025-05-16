# Titans-PyTorch

A PyTorch implementation of neural memory frameworks with fine-tuning capabilities for language models like Qwen3.

## Project Overview

Titans-PyTorch is a framework for building and training neural networks with memory mechanisms, specifically designed for large language model fine-tuning tasks. The project focuses on implementing memory-augmented neural networks and providing tools for training, fine-tuning, and evaluating these models.

## Features

- Neural memory mechanisms implementation
- Transformer model architectures with memory augmentation
- Fine-tuning pipeline for language models (e.g., Qwen3)
- Project Zomboid domain-specific training
- Path management utilities for consistent file handling

## Project Structure

- `titans_pytorch/`: Core module containing implementations of memory models and utilities
  - `memory_models.py`: Base memory model implementations
  - `neural_memory.py`: Neural memory implementation
  - `mac_transformer.py`: Memory-augmented contextual transformer
  - `path_manager.py`: Utilities for path management and configuration

- `train_instruction.py`: Script for fine-tuning models on instruction datasets
- `test_model_performance.py`: Script for testing trained models

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titans-pytorch.git
cd titans-pytorch
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Fine-tuning a model on an instruction dataset

```bash
python train_instruction.py
```

### Testing model performance

```bash
python test_model_performance.py
```

## Configuration

- Model parameters and training settings can be configured in the respective Python scripts.
- Data paths and directories are managed through the `path_manager.py` module.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds upon transformer architectures and neural memory mechanisms from various research papers.
- Thanks to the PyTorch and Hugging Face teams for their excellent libraries.

## Titans - Pytorch

Unofficial implementation of [Titans](https://arxiv.org/abs/2501.00663) in Pytorch. Will also contain some explorations into architectures beyond their simple 1-4 layer MLP for the neural memory module, if it works well to any degree.

## Appreciation

- [Eryk](https://github.com/sentialx) for sharing his early experimental results with me, positive for 2 layer MLP

## Install

```bash
$ pip install titans-pytorch
```

## Usage

```python
import torch
from titans_pytorch import NeuralMemory

mem = NeuralMemory(
    dim = 384,
    chunk_size = 64 # set to smaller chunk size for better perf on smaller sequence lengths (but more memory usage)
).cuda()

seq = torch.randn(2, 1024, 384).cuda()
retrieved, mem_state = mem(seq)

assert seq.shape == retrieved.shape
```

A transformer with the `MAC` configuration can be used as

```python
import torch
from titans_pytorch import MemoryAsContextTransformer

transformer = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 2,
    segment_len = 128,              # local attention window size
    num_persist_mem_tokens = 4,
    num_longterm_mem_tokens = 16,
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = transformer(token_ids, return_loss = True) # (1, 1023, 256)
loss.backward()

# after much training

sampled = transformer.sample(token_ids[:, :4], 512)
```

## Experiments

```bash
$ pip install .[examples]
```

Then modify `train_mac.py` and run it to query nature

```bash
$ python train_mac.py
```

## Citations

```bibtex
@inproceedings{Behrouz2024TitansLT,
    title   = {Titans: Learning to Memorize at Test Time},
    author  = {Ali Behrouz and Peilin Zhong and Vahab S. Mirrokni},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:275212078}
}
```

```bibtex
@article{Sun2024LearningT,
    title   = {Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
    author  = {Yu Sun and Xinhao Li and Karan Dalal and Jiarui Xu and Arjun Vikram and Genghan Zhang and Yann Dubois and Xinlei Chen and Xiaolong Wang and Oluwasanmi Koyejo and Tatsunori Hashimoto and Carlos Guestrin},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2407.04620},
    url     = {https://api.semanticscholar.org/CorpusID:271039606}
}
```

```bibtex
@inproceedings{Yang2024GatedDN,
    title   = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
    author  = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274598177}
}
```

```bibtex
@inproceedings{Nguyen2024TurningUT,
    title   = {Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs},
    author  = {Minh Nguyen and Andrew Baker and Clement Neo and Allen Roush and Andreas Kirsch and Ravid Shwartz-Ziv},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270870613}
}
```

```