import torch
import json
import logging
import math
from pathlib import Path

# Try to import ROUGE, if it fails, log a warning
try:
  from rouge import Rouge
except ImportError:
  Rouge = None
  logging.warning("ROUGE library not found. ROUGE scores will not be calculated. Please install with 'pip install rouge-score py-rouge'.")


logger = logging.getLogger(__name__)

class ModelEvaluator:
  """
  Handles model evaluation tasks including text generation and metric calculation.
  """
  def __init__(self, model, tokenizer, output_dir="evaluation_results"):
    self.model = model
    self.tokenizer = tokenizer
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model.to(self.device)
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Evaluation results will be saved to: {self.output_dir.resolve()}")

  def _clean_generated_text(self, text, prompt=""):
    """
    Cleans the generated text by removing the prompt and other artifacts.
    """
    if prompt and text.startswith(prompt):
      text = text[len(prompt):]
    
    # Basic cleaning of common model output patterns
    text = text.replace(self.tokenizer.eos_token, "").strip()
    text = text.replace("<|endoftext|>", "").strip() # Common for some models

    # Further cleaning can be added here if needed
    # For example, removing specific system messages if they leak into generation
    import re
    system_patterns = [
        r"You are an assistant specialized in.*?\.",
        r"Project Zomboid is a zombie survival game.*?\.",
        r"When answering questions:.*",
        r"- Provide only factual game information.*",
        r"- Focus on specific mechanics, items, and locations.*",
        r"- Keep responses direct and to the point.*",
        r"- Never make up game features that don't exist.*",
        r"Your goal is to help players understand actual game mechanics and systems.*"
    ]
    for pattern in system_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    
    # Remove leading/trailing newlines and multiple spaces
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

  def generate_response(self, prompt_text, max_new_tokens=250, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    """
    Generates a response from the model given a prompt.
    """
    self.model.eval()
    inputs = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
    
    with torch.no_grad():
      outputs = self.model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=self.tokenizer.eos_token_id  # Ensure pad_token_id is set
      )
    
    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False) # Keep special tokens for better cleaning control
    return self._clean_generated_text(generated_text, prompt=prompt_text)

  def calculate_rouge_scores(self, hypotheses, references):
    """
    Calculates ROUGE scores.
    hypotheses: list of generated texts
    references: list of reference texts
    """
    if Rouge is None:
      logger.warning("ROUGE library not available. Skipping ROUGE calculation.")
      return None
    
    if not hypotheses or not references or len(hypotheses) != len(references):
      logger.error("Invalid input for ROUGE calculation. Hypotheses and references must be non-empty and of equal length.")
      return None

    rouge_evaluator = Rouge()
    try:
      # Filter out empty strings which can cause errors with some ROUGE implementations
      valid_hypotheses = [h if h else " " for h in hypotheses]
      valid_references = [r if r else " " for r in references]

      scores = rouge_evaluator.get_scores(valid_hypotheses, valid_references, avg=True)
      logger.info(f"ROUGE Scores (avg): {scores}")
      return scores
    except Exception as e:
      logger.error(f"Error calculating ROUGE scores: {e}")
      return None

  def evaluate_dataset(self, dataset_path, limit=None):
    """
    Evaluates the model on a dataset provided in JSONL format.
    Each line should be a JSON object with "instruction" (or "prompt") and "answer" (or "output").
    """
    prompts = []
    references = []
    try:
      with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
          if limit and i >= limit:
            break
          try:
            data_item = json.loads(line)
            prompt = data_item.get("instruction") or data_item.get("prompt")
            answer = data_item.get("answer") or data_item.get("output")
            if prompt and answer:
              prompts.append(prompt)
              references.append(answer)
            else:
              logger.warning(f"Skipping line {i+1} due to missing prompt or answer: {line.strip()}")
          except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON line {i+1}: {line.strip()}")
      logger.info(f"Loaded {len(prompts)} prompts and references from {dataset_path}")
    except FileNotFoundError:
      logger.error(f"Dataset file not found: {dataset_path}")
      return None
    except Exception as e:
      logger.error(f"Error reading dataset file {dataset_path}: {e}")
      return None

    if not prompts:
      logger.error("No valid data found in the dataset file.")
      return None

    hypotheses = []
    detailed_results = []

    for i in range(len(prompts)):
      prompt = prompts[i]
      reference = references[i]
      logger.info(f"Generating for prompt {i+1}/{len(prompts)}: {prompt[:100]}...")
      generated_text = self.generate_response(prompt)
      hypotheses.append(generated_text)
      detailed_results.append({
        "id": i,
        "prompt": prompt,
        "reference_answer": reference,
        "generated_answer": generated_text
      })

    rouge_scores = self.calculate_rouge_scores(hypotheses, references)
    
    # Save detailed generation results
    results_file = self.output_dir / f"detailed_generation_results_{Path(dataset_path).stem}.jsonl"
    with open(results_file, 'w', encoding='utf-8') as f_out:
        for res_item in detailed_results:
            f_out.write(json.dumps(res_item, ensure_ascii=False) + "\n")
    logger.info(f"Detailed generation results saved to {results_file}")

    if rouge_scores:
        metrics_summary_file = self.output_dir / f"metrics_summary_{Path(dataset_path).stem}.json"
        with open(metrics_summary_file, 'w', encoding='utf-8') as f_metrics:
            json.dump(rouge_scores, f_metrics, ensure_ascii=False, indent=2)
        logger.info(f"ROUGE metrics summary saved to {metrics_summary_file}")
        return {"rouge_scores": rouge_scores, "details_file": str(results_file)}
    else:
        return {"error": "ROUGE scores could not be calculated.", "details_file": str(results_file)} 