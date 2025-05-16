"""
Titans PyTorch - A collection of neural network architectures and utilities
for memory-based deep learning.
"""

# Import submodules
from titans_pytorch.memory_models import (
  MemoryMLP,
  MemoryAttention,
  FactorizedMemoryMLP,
  MemorySwiGluMLP,
  GatedResidualMemoryMLP
)

from titans_pytorch.mac_transformer import MemoryAsContextTransformer
from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState, mem_state_detach
from titans_pytorch.path_manager import (
  get_project_root,
  get_data_dir,
  get_checkpoint_dir,
  get_output_dir,
  get_logs_dir,
  get_data_file_path,
  get_checkpoint_file_path,
  get_wiki_data_path,
  get_model_output_path,
  get_final_model_path,
  get_interrupted_model_path
)

__all__ = [
  # Memory models
  'MemoryMLP',
  'MemoryAttention',
  'FactorizedMemoryMLP',
  'MemorySwiGluMLP',
  'GatedResidualMemoryMLP',
  # Transformer models
  'MemoryAsContextTransformer',
  # Neural memory
  'NeuralMemory',
  'NeuralMemState',
  'mem_state_detach',
  # Path manager utilities
  'get_project_root',
  'get_data_dir',
  'get_checkpoint_dir',
  'get_output_dir',
  'get_logs_dir',
  'get_data_file_path',
  'get_checkpoint_file_path',
  'get_wiki_data_path',
  'get_model_output_path',
  'get_final_model_path',
  'get_interrupted_model_path'
]
