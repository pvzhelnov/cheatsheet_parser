"""
Evaluation system for LLM responses
"""

import os
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
from sklearn.metrics import confusion_matrix, classification_report
import hashlib
import argparse


class EvalResult:
    """Container for evaluation results"""
    def __init__(self, test_name: str, result: str, expected: Any = None, actual: Any = None, 
                 score: float = None, metadata: Dict = None):
        self.test_name = test_name
        self.result = result  # "pass", "fail", "skip"
        self.expected = expected
        self.actual = actual
        self.score = score
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class BaseEvaluator:
    """Base class for all evaluators"""
    
    def __init__(self, name: str):
        self.name = name
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        """Evaluate a response"""
        raise NotImplementedError


class ContainsEvaluator(BaseEvaluator):
    """Check if response contains specific text"""
    
    def __init__(self, text: str, case_sensitive: bool = False):
        super().__init__(f"CONTAINS_{text.upper()}")
        self.text = text
        self.case_sensitive = case_sensitive
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        content_check = response_content if self.case_sensitive else response_content.lower()
        text_check = self.text if self.case_sensitive else self.text.lower()
        
        contains = text_check in content_check
        return EvalResult(
            test_name=self.name,
            result="pass" if contains else "fail",
            expected=f"Contains '{self.text}'",
            actual=f"Contains: {contains}",
            score=1.0 if contains else 0.0
        )


class NotContainsEvaluator(BaseEvaluator):
    """Check if response does NOT contain specific text"""
    
    def __init__(self, text: str, case_sensitive: bool = False):
        super().__init__(f"NOT_CONTAINS_{text.upper()}")
        self.text = text
        self.case_sensitive = case_sensitive
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        content_check = response_content if self.case_sensitive else response_content.lower()
        text_check = self.text if self.case_sensitive else self.text.lower()
        
        not_contains = text_check not in content_check
        return EvalResult(
            test_name=self.name,
            result="pass" if not_contains else "fail",
            expected=f"Does not contain '{self.text}'",
            actual=f"Contains: {not not_contains}",
            score=1.0 if not_contains else 0.0
        )


class LengthEvaluator(BaseEvaluator):
    """Check response length constraints"""
    
    def __init__(self, min_length: int = None, max_length: int = None):
        super().__init__(f"LENGTH_{min_length or 0}_TO_{max_length or 'INF'}")
        self.min_length = min_length
        self.max_length = max_length
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        length = len(response_content)
        
        passes = True
        reasons = []
        
        if self.min_length and length < self.min_length:
            passes = False
            reasons.append(f"Too short: {length} < {self.min_length}")
            
        if self.max_length and length > self.max_length:
            passes = False
            reasons.append(f"Too long: {length} > {self.max_length}")
            
        return EvalResult(
            test_name=self.name,
            result="pass" if passes else "fail",
            expected=f"Length between {self.min_length or 0} and {self.max_length or 'inf'}",
            actual=f"Length: {length}",
            score=1.0 if passes else 0.0,
            metadata={"reasons": reasons}
        )


class GroundTruthEvaluator(BaseEvaluator):
    """Compare response against ground truth"""
    
    def __init__(self, comparison_type: str = "exact"):
        super().__init__(f"GROUND_TRUTH_{comparison_type.upper()}")
        self.comparison_type = comparison_type
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        if ground_truth is None:
            return EvalResult(
                test_name=self.name,
                result="skip",
                expected="Ground truth available",
                actual="No ground truth provided",
                score=None
            )
            
        if self.comparison_type == "exact":
            matches = str(response_content).strip() == str(ground_truth).strip()
            score = 1.0 if matches else 0.0
        elif self.comparison_type == "contains":
            matches = str(ground_truth).lower() in str(response_content).lower()
            score = 1.0 if matches else 0.0
        else:
            # Semantic similarity (placeholder - could implement with embeddings)
            matches = False
            score = 0.0
            
        return EvalResult(
            test_name=self.name,
            result="pass" if matches else "fail",
            expected=str(ground_truth),
            actual=str(response_content)[:200] + "..." if len(str(response_content)) > 200 else str(response_content),
            score=score
        )

class ManualEvaluator(BaseEvaluator):
    """Dynamic manual evaluator that accepts any name and result"""
    
    def __init__(self, name: str):
        super().__init__(name)
        
    def evaluate(self, response_content: str, ground_truth: Any = None, **kwargs) -> EvalResult:
        manual_result = kwargs.get('manual_result', 'skip')
        manual_score = kwargs.get('manual_score')
        if manual_score is None:
            manual_score = 1.0 if manual_result == "pass" else 0.0
        manual_reason = kwargs.get('manual_reason', 'Manual evaluation')
        
        return EvalResult(
            test_name=self.name,
            result=manual_result,
            expected="Manual evaluation: pass (score > 0)",
            actual=f"Manual evaluation: {manual_result} (score: {manual_score})",
            score=manual_score,
            metadata={"reason": manual_reason}
        )



class EvaluationRunner:
    """Main evaluation runner"""
    
    def __init__(self):
        self.evaluators = {}
        self.logger = logging.getLogger(__name__)
        self._register_default_evaluators()
        
    def _register_default_evaluators(self):
        """Register commonly used evaluators"""
        # Common negative checks
        self.register_evaluator("HAS_NO_SHIT_IN_RESPONSE", NotContainsEvaluator("shit"))
        self.register_evaluator("HAS_NO_FUCK_IN_RESPONSE", NotContainsEvaluator("fuck"))
        self.register_evaluator("HAS_NO_DAMN_IN_RESPONSE", NotContainsEvaluator("damn"))
        
        # Length checks
        self.register_evaluator("REASONABLE_LENGTH", LengthEvaluator(min_length=10, max_length=5000))
        self.register_evaluator("SHORT_RESPONSE", LengthEvaluator(max_length=500))
        self.register_evaluator("LONG_RESPONSE", LengthEvaluator(min_length=1000))
        
        # Ground truth
        self.register_evaluator("GROUND_TRUTH_EXACT", GroundTruthEvaluator("exact"))
        self.register_evaluator("GROUND_TRUTH_CONTAINS", GroundTruthEvaluator("contains"))
        
    def register_evaluator(self, name: str, evaluator: BaseEvaluator):
        """Register a custom evaluator"""
        self.evaluators[name] = evaluator
        
    def run_evals(self, responses_dir: str = "responses", output_dir: str = "eval_results") -> Dict[str, Any]:
        """Run evaluations on all response files"""
        responses_path = Path(responses_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not responses_path.exists():
            self.logger.warning(f"Responses directory {responses_path} does not exist")
            return {}
            
        # Load all response files
        response_files = list(responses_path.glob("*.yml"))
        if not response_files:
            self.logger.warning(f"No YAML files found in {responses_path}")
            return {}
            
        all_results = []
        
        for response_file in response_files:
            try:
                with open(response_file, 'r') as f:
                    response_data = yaml.safe_load(f)
                    
                # Run evaluations
                file_results = self._evaluate_response(response_data, response_file.name)
                all_results.extend(file_results)
                
            except Exception as e:
                self.logger.error(f"Error processing {response_file}: {e}")
                continue
                
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        # Save results
        self._save_results(aggregated, output_path)
        
        # Generate visualizations
        self._generate_visualizations(aggregated, output_path)
        
        return aggregated
        
    def _evaluate_response(self, response_data: Dict, filename: str) -> List[Dict]:
        """Evaluate a single response"""
        results = []
        
        content = response_data.get('content', '')
        evals = response_data.get('evals', {})
        ground_truth = response_data.get('ground_truth')
        name = response_data.get('name', filename)
        
        # Run specified evaluations
        for eval_name, expected_result in evals.items():
            if eval_name in self.evaluators:
                evaluator = self.evaluators[eval_name]
                result = evaluator.evaluate(content, ground_truth)
                
                results.append({
                    'response_file': filename,
                    'response_name': name,
                    'provider': response_data.get('provider'),
                    'model': response_data.get('model'),
                    'timestamp': response_data.get('timestamp'),
                    'eval_name': eval_name,
                    'expected': expected_result,
                    'actual_result': result.result,
                    'matches_expected': (result.result == expected_result),
                    'score': result.score,
                    'eval_expected': result.expected,
                    'eval_actual': result.actual,
                    'metadata': result.metadata
                })
            else:
                self.logger.warning(f"Unknown evaluator: {eval_name}")

        # Handle manual evaluations
        manual_evals = response_data.get('manual_evals', {})

        def process_nested_evals(data, parent_key=''):
            results = []
            if isinstance(data, dict):
                for key, value in data.items():
                    current_key = f"{parent_key}.{key}" if parent_key else key

                    # Placeholder dict to be filled
                    manual_dict = {
                        'manual_reason': None,
                        'manual_result': None,
                        'manual_score': None
                    }

                    # Discriminate between dicts
                    if isinstance(value, dict):
                        if any(key in value.keys() for key in manual_dict.keys()):
                            for key in manual_dict.keys():
                                if key in value:
                                    manual_dict[key] = value[key]
                        else:
                            results.extend(process_nested_evals(value, current_key))
                            continue
                    else:
                        manual_dict['manual_result'] = value

                    # Skip if already processed in regular evals
                    if current_key in evals:
                        continue
                        
                    evaluator = ManualEvaluator(current_key)
                    result = evaluator.evaluate(content, ground_truth, **manual_dict)
                    matches_expected = (
                        result.result == ('pass' if value != 'skip' else 'skip')
                    ) and (
                        (result.score > 0) if value != 'skip' else True
                    )
                    
                    results.append({
                        'response_file': filename,
                        'response_name': name,
                        'provider': response_data.get('provider'),
                        'model': response_data.get('model'),
                        'timestamp': response_data.get('timestamp'),
                        'eval_name': current_key,
                        'expected': 'pass' if value != 'skip' else 'skip',
                        'actual_result': result.result,
                        'matches_expected': matches_expected,
                        'score': result.score,
                        'eval_expected': result.expected,
                        'eval_actual': result.actual,
                        'metadata': result.metadata
                    })
            return results

        results.extend(process_nested_evals(manual_evals))
                
        return results
        
    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate evaluation results"""
        if not all_results:
            return {}
            
        df = pd.DataFrame(all_results)
        
        # Overall metrics
        total_evals = len(df)
        total_passed = len(df[df['matches_expected'] == True])
        total_failed = len(df[df['matches_expected'] == False])
        pass_rate = total_passed / total_evals if total_evals > 0 else 0
        
        # By evaluator
        by_evaluator = df.groupby('eval_name').agg({
            'matches_expected': ['count', 'sum', 'mean'],
            'score': 'mean'
        }).round(3)
        
        # By provider
        by_provider = df.groupby('provider').agg({
            'matches_expected': ['count', 'sum', 'mean'],
            'score': 'mean'
        }).round(3) if 'provider' in df.columns else pd.DataFrame()
        
        # By model
        by_model = df.groupby('model').agg({
            'matches_expected': ['count', 'sum', 'mean'],
            'score': 'mean'
        }).round(3) if 'model' in df.columns else pd.DataFrame()
        
        # Confusion matrix data
        confusion_data = []
        for eval_name in df['eval_name'].unique():
            eval_df = df[df['eval_name'] == eval_name]
            if len(eval_df) > 0:
                y_true = eval_df['expected'].tolist()
                y_pred = eval_df['actual_result'].tolist()
                confusion_data.append({
                    'eval_name': eval_name,
                    'y_true': y_true,
                    'y_pred': y_pred
                })
        
        return {
            'summary': {
                'total_evaluations': total_evals,
                'total_passed': total_passed,
                'total_failed': total_failed,
                'pass_rate': pass_rate,
                'timestamp': datetime.now().isoformat()
            },
            'by_evaluator': self._flatten_multiindex_dict(by_evaluator) if not by_evaluator.empty else {},
            'by_provider': self._flatten_multiindex_dict(by_provider) if not by_provider.empty else {},
            'by_model': self._flatten_multiindex_dict(by_model) if not by_model.empty else {},
            'raw_data': df.to_dict('records'),
            'confusion_data': confusion_data
        }

    def _flatten_multiindex_dict(self, df):
        """Flatten MultiIndex DataFrame to JSON-serializable dict"""
        result = {}
        for index, row in df.iterrows():
            result[str(index)] = {}
            for col in df.columns:
                if isinstance(col, tuple):
                    key = '_'.join(str(x) for x in col)
                else:
                    key = str(col)
                result[str(index)][key] = row[col]
        return result
        
    def _save_results(self, results: Dict, output_path: Path):
        """Save aggregated results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = output_path / f"eval_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save as YAML
        yaml_file = output_path / f"eval_results_{timestamp}.yml"
        with open(yaml_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
            
        # Save summary as CSV
        if results.get('raw_data'):
            csv_file = output_path / f"eval_results_{timestamp}.csv"
            pd.DataFrame(results['raw_data']).to_csv(csv_file, index=False)
            
        self.logger.info(f"Results saved to {output_path}")
        
    def _generate_visualizations(self, results: Dict, output_path: Path):
        """Generate visualization plots"""
        if not results.get('raw_data'):
            return
            
        df = pd.DataFrame(results['raw_data'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Pass rate by evaluator
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Pass rate heatmap by evaluator
        if 'by_evaluator' in results and results['by_evaluator']:
            eval_data = pd.DataFrame(results['by_evaluator']).T
            if 'matches_expected' in eval_data.columns:
                pass_rates = eval_data[('matches_expected', 'mean')].values.reshape(-1, 1)
                sns.heatmap(pass_rates, 
                           yticklabels=eval_data.index, 
                           xticklabels=['Pass Rate'],
                           annot=True, fmt='.2f', cmap='RdYlGn',
                           ax=axes[0, 0])
                axes[0, 0].set_title('Pass Rate by Evaluator')
        
        # Pass rate by provider
        if 'provider' in df.columns:
            provider_pass_rate = df.groupby('provider')['matches_expected'].mean()
            provider_pass_rate.plot(kind='bar', ax=axes[0, 1], color='skyblue')
            axes[0, 1].set_title('Pass Rate by Provider')
            axes[0, 1].set_ylabel('Pass Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Score distribution
        if 'score' in df.columns:
            df['score'].hist(bins=20, ax=axes[1, 0], alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Score Distribution')
            axes[1, 0].set_xlabel('Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # Confusion matrix for first evaluator (if available)
        if results.get('confusion_data'):
            confusion_info = results['confusion_data'][0]
            y_true = confusion_info['y_true']
            y_pred = confusion_info['y_pred']
            
            #labels = sorted(list(set(y_true + y_pred)))
            all_labels = ['pass', 'fail', 'skip']  # Define all possible labels
            cm = confusion_matrix(y_true, y_pred, labels=all_labels)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=all_labels, yticklabels=all_labels,
                       ax=axes[1, 1])
            axes[1, 1].set_title(f'Confusion Matrix: {confusion_info["eval_name"]}')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        viz_file = output_path / f"eval_visualization_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary to console
        self._print_summary(results)
        
    def _print_summary(self, results: Dict):
        """Print evaluation summary to console"""
        summary = results.get('summary', {})
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Evaluations: {summary.get('total_evaluations', 0)}")
        print(f"Passed: {summary.get('total_passed', 0)}")
        print(f"Failed: {summary.get('total_failed', 0)}")
        print(f"Pass Rate: {summary.get('pass_rate', 0):.2%}")
        print(f"Timestamp: {summary.get('timestamp', 'N/A')}")
        
        # By evaluator summary
        if results.get('by_evaluator'):
            print("\nBY EVALUATOR:")
            print("-" * 40)
            eval_df = pd.DataFrame(results['by_evaluator']).T
            if not eval_df.empty and 'matches_expected' in eval_df.columns:
                for idx, row in eval_df.iterrows():
                    total = row[('matches_expected', 'count')]
                    passed = row[('matches_expected', 'sum')]
                    rate = row[('matches_expected', 'mean')]
                    print(f"{idx}: {passed}/{total} ({rate:.2%})")
        
        print("="*60)

# CLI Interface
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--responses-dir', default='responses', help='Directory containing response YAML files')
    parser.add_argument('--output-dir', default='eval_results', help='Directory to save evaluation results')
    args = parser.parse_args()
    
    try:
        runner = EvaluationRunner()
        results = runner.run_evals(args.responses_dir, args.output_dir)
        if not results:
            print("No evaluations were run. Check if response files exist.")
        else:
            print(f"✅ Evaluations completed. Results saved to {args.output_dir}")
    except Exception as e:
        print(f"Error running evaluations: {e}")

if __name__ == '__main__':
    cli()