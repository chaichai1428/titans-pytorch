import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation_logic import ModelEvaluator # From the file we just created
from titans_pytorch.path_manager import get_final_model_path # Assuming you have this to get paths
import torch # Added for torch.float16 check

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
  parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Causal LM for Project Zomboid.")
  parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to the fine-tuned model directory. If not provided, attempts to use get_final_model_path()."
  )
  parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the evaluation dataset (JSONL format, with 'instruction'/'prompt' and 'answer'/'output' keys)."
  )
  parser.add_argument(
    "--output_dir",
    type=str,
    default="evaluation_results",
    help="Directory to save evaluation results."
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional: Limit the number of samples to evaluate from the dataset."
  )

  args = parser.parse_args()

  model_path_to_load = args.model_path
  if not model_path_to_load:
    try:
      model_path_to_load = get_final_model_path()
      logger.info(f"No model_path provided, using default final model path: {model_path_to_load}")
    except Exception as e:
      logger.error(f"Failed to get default final model path: {e}. Please specify --model_path.")
      return

  if not model_path_to_load:
      logger.error("Model path is not specified and could not be determined. Exiting.")
      return

  logger.info(f"Loading tokenizer from: {model_path_to_load}")
  try:
    tokenizer = AutoTokenizer.from_pretrained(model_path_to_load)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad_token. Setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
  except Exception as e:
    logger.error(f"Failed to load tokenizer: {e}")
    return
  
  logger.info(f"Loading model from: {model_path_to_load}")
  try:
    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16 
        # device_map="auto" can be useful but might need more VRAM or specific setup
        # model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path_to_load, **model_kwargs)
  except Exception as e:
    logger.error(f"Failed to load model: {e}")
    return

  evaluator = ModelEvaluator(model, tokenizer, output_dir=args.output_dir)
  
  logger.info(f"Starting evaluation on dataset: {args.dataset_path}")
  results = evaluator.evaluate_dataset(args.dataset_path, limit=args.limit)

  if results:
    logger.info("Evaluation finished.")
    if "error" not in results and "rouge_scores" in results:
        logger.info(f"ROUGE scores: {results.get('rouge_scores')}")
    elif "error" in results:
        logger.error(f"Evaluation error: {results.get('error')}")
    logger.info(f"Detailed results information (if generated) can be found in: {results.get('details_file')}")
  else:
    logger.error("Evaluation failed to produce results or an error occurred.")

if __name__ == "__main__":
  main() 