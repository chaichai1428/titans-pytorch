"""
Path management utilities for Titans PyTorch.
Centralizes all path-related operations to ensure consistency across the project.
"""

import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directories
def get_project_root():
  """Returns the absolute path to the project root directory"""
  # Get the directory of this file and go up one level to find project root
  return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_data_dir():
  """Returns the path to the data directory"""
  data_dir = os.path.join(get_project_root(), "data")
  os.makedirs(data_dir, exist_ok=True)
  return data_dir

def get_checkpoint_dir():
  """Returns the path to the checkpoint directory"""
  checkpoint_dir = os.path.join(get_project_root(), "checkpoint")
  os.makedirs(checkpoint_dir, exist_ok=True)
  return checkpoint_dir

def get_output_dir(model_name="model-output"):
  """Returns the path to the output directory"""
  output_dir = os.path.join(get_project_root(), model_name)
  os.makedirs(output_dir, exist_ok=True)
  return output_dir

def get_logs_dir():
  """Returns the path to the logs directory"""
  logs_dir = os.path.join(get_project_root(), "logs")
  os.makedirs(logs_dir, exist_ok=True)
  return logs_dir

# File paths
def get_data_file_path(filename):
  """Returns the path to a file in the data directory"""
  return os.path.join(get_data_dir(), filename)

def get_checkpoint_file_path(model_name):
  """Returns the path to a model checkpoint"""
  return os.path.join(get_checkpoint_dir(), model_name)

def ensure_dir_exists(dir_path):
  """Ensures a directory exists, creating it if necessary"""
  os.makedirs(dir_path, exist_ok=True)
  return dir_path

def resolve_relative_path(path):
  """
  Resolves a path relative to the project root.
  If path is absolute, it's returned unchanged.
  """
  if os.path.isabs(path):
    return path
  return os.path.join(get_project_root(), path)

# Specific paths for training scripts
def get_wiki_data_path():
  """Returns the path to the wiki data file"""
  return get_data_file_path("wiki3.jsonl")

def get_model_output_path(model_name="qwen3-pz-finetuned"):
  """Returns the path to the model output directory"""
  return get_output_dir(model_name)

def get_final_model_path(model_name="qwen3-pz-finetuned-final"):
  """Returns the path to the final model directory"""
  return get_checkpoint_file_path(model_name)

def get_interrupted_model_path(model_name="qwen3-pz-interrupted"):
  """Returns the path to the interrupted model directory"""
  return get_checkpoint_file_path(model_name) 