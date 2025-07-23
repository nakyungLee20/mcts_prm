import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project imports
from prm_trainer_mse import PRMTrainerMSE
from config import PRMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PRMInference:
    """
    Inference class for trained PRM models
    """
    def __init__(self, checkpoint_path: str, config: PRMConfig, from_scratch: bool = False):
        self.config = config
        self.from_scratch = from_scratch
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(config.model_name)
        
        # Initialize trainer (for model loading)
        self.trainer = PRMTrainerMSE(
            cfg=config,
            model=self.model,
            tokenizer=self.tokenizer,
            from_scratch=from_scratch
        )
        
        # Load checkpoint
        self.trainer.load_checkpoint(checkpoint_path)
        logger.info(f"Loaded PRM model from: {checkpoint_path}")
        
        # Set to evaluation mode
        self.trainer.prm.eval()
        if from_scratch:
            self.trainer.model.eval()
    
    def _load_model(self, model_name: str):
        """Load pretrained model and tokenizer"""
        logger.info(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    def predict_step_rewards(self, question: str, steps: List[str]) -> List[float]:
        rewards = []
        
        for i in range(len(steps)):
            # Create input text: question + steps up to current step
            input_text = f"Problem: {question}\n"
            for j in range(i + 1):
                input_text += f"{steps[j]}\n"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            )
            
            # Move to device
            input_ids = inputs["input_ids"].to(self.trainer.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.trainer.predict(input_ids)
                reward = prediction.item()
                rewards.append(reward)
        
        return rewards
    
    def evaluate_solution_quality(self, question: str, steps: List[str]) -> Dict[str, float]:
        step_rewards = self.predict_step_rewards(question, steps)
        
        # Calculate metrics
        total_reward = sum(step_rewards)
        avg_reward = np.mean(step_rewards)
        reward_progression = np.diff(step_rewards)  # How rewards change between steps
        
        return {
            "step_rewards": step_rewards,
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "reward_progression": reward_progression.tolist(),
            "final_reward": step_rewards[-1] if step_rewards else 0.0,
            "reward_consistency": np.std(step_rewards),  # Lower is more consistent
            "reward_trend": np.mean(reward_progression) if len(reward_progression) > 0 else 0.0
        }
    
    def compare_solutions(self, question: str, solutions: List[List[str]]) -> List[Dict[str, float]]:
        results = []
        
        for i, steps in enumerate(solutions):
            evaluation = self.evaluate_solution_quality(question, steps)
            evaluation["solution_id"] = i
            evaluation["num_steps"] = len(steps)
            results.append(evaluation)
        
        # Sort by total reward (descending)
        results.sort(key=lambda x: x["total_reward"], reverse=True)
        
        return results
    
    def analyze_step_contribution(self, question: str, steps: List[str]) -> List[Dict[str, float]]:
        step_rewards = self.predict_step_rewards(question, steps)
        
        analysis = []
        cumulative_reward = 0.0
        
        for i, (step, reward) in enumerate(zip(steps, step_rewards)):
            cumulative_reward += reward
            step_contribution = reward
            
            analysis.append({
                "step_id": i + 1,
                "step_text": step,
                "step_reward": reward,
                "cumulative_reward": cumulative_reward,
                "step_contribution": step_contribution,
                "contribution_ratio": step_contribution / cumulative_reward if cumulative_reward > 0 else 0.0
            })
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description="PRM Inference with MSE loss")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PRM checkpoint")
    parser.add_argument("--config", type=str, default="config.py", help="Path to config file")
    parser.add_argument("--from-scratch", action="store_true", help="Model was trained from scratch")
    parser.add_argument("--question", type=str, help="Math problem to evaluate")
    parser.add_argument("--steps", type=str, nargs="+", help="Solution steps")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = PRMConfig()
    
    # Initialize inference
    inference = PRMInference(
        checkpoint_path=args.checkpoint,
        config=config,
        from_scratch=args.from_scratch
    )
    
    if args.question and args.steps:
        # Single solution evaluation
        question = args.question
        steps = args.steps
        
        logger.info(f"Evaluating solution for: {question}")
        logger.info(f"Steps: {steps}")
        
        # Evaluate solution quality
        evaluation = inference.evaluate_solution_quality(question, steps)
        
        # Analyze step contributions
        analysis = inference.analyze_step_contribution(question, steps)
        
        # Print results
        print("\n" + "="*50)
        print("SOLUTION EVALUATION")
        print("="*50)
        print(f"Question: {question}")
        print(f"Total Reward: {evaluation['total_reward']:.4f}")
        print(f"Average Reward: {evaluation['avg_reward']:.4f}")
        print(f"Final Reward: {evaluation['final_reward']:.4f}")
        print(f"Reward Consistency: {evaluation['reward_consistency']:.4f}")
        print(f"Reward Trend: {evaluation['reward_trend']:.4f}")
        
        print("\n" + "-"*50)
        print("STEP-BY-STEP ANALYSIS")
        print("-"*50)
        for step_analysis in analysis:
            print(f"Step {step_analysis['step_id']}: {step_analysis['step_reward']:.4f}")
            print(f"  Text: {step_analysis['step_text']}")
            print(f"  Cumulative: {step_analysis['cumulative_reward']:.4f}")
            print(f"  Contribution: {step_analysis['step_contribution']:.4f}")
            print()
        
        # Save results if output file specified
        if args.output:
            results = {
                "question": question,
                "steps": steps,
                "evaluation": evaluation,
                "step_analysis": analysis
            }
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {args.output}")
    
    else:
        # Interactive mode
        print("PRM Inference Interactive Mode")
        print("Enter 'quit' to exit")
        print("-" * 30)
        
        while True:
            try:
                question = input("Enter math problem: ").strip()
                if question.lower() == 'quit':
                    break
                
                steps = []
                print("Enter solution steps (one per line, empty line to finish):")
                while True:
                    step = input("Step: ").strip()
                    if not step:
                        break
                    steps.append(step)
                
                if not steps:
                    print("No steps provided. Skipping...")
                    continue
                
                # Evaluate
                evaluation = inference.evaluate_solution_quality(question, steps)
                analysis = inference.analyze_step_contribution(question, steps)
                
                # Print results
                print(f"\nTotal Reward: {evaluation['total_reward']:.4f}")
                print(f"Average Reward: {evaluation['avg_reward']:.4f}")
                print("Step rewards:", [f"{r:.4f}" for r in evaluation['step_rewards']])
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                continue

if __name__ == "__main__":
    main() 