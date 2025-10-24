"""RAG Evaluation Script for Financial ESG RAG System.

Runs comprehensive evaluation of the RAG system including:
- RAGAS metrics
- Retrieval metrics
- ESG-specific metrics
- Detailed reporting with visualizations
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger
from tqdm import tqdm

from src.evaluation.metrics import RAGEvaluationMetrics
from src.evaluation.test_dataset import ESGTestDataset, ESGAdversarialDataset
from src.chatbot.esg_chatbot import ESGChatbot


class RAGEvaluator:
    """Comprehensive RAG system evaluator."""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        """Initialize evaluator."""
        self.metrics_calculator = RAGEvaluationMetrics()
        self.chatbot = ESGChatbot()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.add(
            self.output_dir / "evaluation.log",
            rotation="10 MB",
            level="INFO"
        )
        
        logger.info("RAG Evaluator initialized")
    
    def evaluate_single_query(
        self,
        test_case: Dict[str, Any],
        session_id: str = "eval_session"
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.
        
        Args:
            test_case: Test case dictionary with question and ground truth
            session_id: Session ID for chatbot
            
        Returns:
            Evaluation results dictionary
        """
        question = test_case["question"]
        ground_truth = test_case.get("ground_truth")
        expected_sources = test_case.get("expected_sources", [])
        
        logger.info(f"Evaluating: {question}")
        
        # Get RAG response
        try:
            # Retrieve contexts
            contexts_data = self.chatbot.rag_store.search(
                query=question,
                limit=5,
                score_threshold=0.15
            )
            contexts = [ctx.get("text", "") for ctx in contexts_data]
            
            # Generate answer
            answer = self.chatbot.process_message(question, session_id)
            
            # Calculate metrics
            metrics = self.metrics_calculator.evaluate_rag_response(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                expected_sources=expected_sources
            )
            
            # Add retrieval metrics if ground truth sources available
            if expected_sources:
                metrics['mrr'] = self.metrics_calculator.mean_reciprocal_rank(
                    contexts_data, expected_sources
                )
                metrics['precision@3'] = self.metrics_calculator.precision_at_k(
                    contexts_data, expected_sources, k=3
                )
                metrics['recall@5'] = self.metrics_calculator.recall_at_k(
                    contexts_data, expected_sources, k=5
                )
                metrics['ndcg@5'] = self.metrics_calculator.ndcg_at_k(
                    contexts_data, expected_sources, k=5
                )
            
            result = {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts_retrieved": len(contexts),
                "metrics": metrics,
                "expected_sources": expected_sources,
                "retrieved_sources": [
                    ctx.get("metadata", {}).get("source_file", "")
                    for ctx in contexts_data
                ],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            result = {
                "question": question,
                "answer": None,
                "ground_truth": ground_truth,
                "contexts_retrieved": 0,
                "metrics": {},
                "status": "error",
                "error": str(e)
            }
        
        return result
    
    def evaluate_dataset(
        self,
        test_cases: List[Dict[str, Any]],
        session_id: str = "eval_session"
    ) -> List[Dict[str, Any]]:
        """
        Evaluate entire dataset.
        
        Args:
            test_cases: List of test cases
            session_id: Session ID for chatbot
            
        Returns:
            List of evaluation results
        """
        results = []
        
        logger.info(f"Evaluating {len(test_cases)} test cases...")
        
        for test_case in tqdm(test_cases, desc="Evaluating"):
            result = self.evaluate_single_query(test_case, session_id)
            results.append(result)
        
        return results
    
    def generate_summary_statistics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Summary statistics dictionary
        """
        # Extract all metrics
        all_metrics = {}
        for result in results:
            if result["status"] == "success":
                metrics = result.get("metrics", {})
                for metric_name, metric_value in metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    if metric_value is not None:
                        all_metrics[metric_name].append(metric_value)
        
        # Calculate statistics
        summary = {
            "total_queries": len(results),
            "successful_queries": sum(1 for r in results if r["status"] == "success"),
            "failed_queries": sum(1 for r in results if r["status"] == "error"),
            "metrics": {}
        }
        
        for metric_name, values in all_metrics.items():
            if values:
                summary["metrics"][metric_name] = {
                    "mean": float(pd.Series(values).mean()),
                    "median": float(pd.Series(values).median()),
                    "std": float(pd.Series(values).std()),
                    "min": float(pd.Series(values).min()),
                    "max": float(pd.Series(values).max()),
                    "count": len(values)
                }
        
        return summary
    
    def generate_visualizations(
        self,
        results: List[Dict[str, Any]],
        output_path: Path
    ):
        """
        Generate evaluation visualizations.
        
        Args:
            results: List of evaluation results
            output_path: Path to save visualizations
        """
        # Extract metrics for visualization
        metrics_data = []
        for result in results:
            if result["status"] == "success":
                metrics = result.get("metrics", {})
                metrics_data.append(metrics)
        
        if not metrics_data:
            logger.warning("No successful results to visualize")
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Overall Metrics Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("RAG Evaluation Metrics Distribution", fontsize=16, fontweight='bold')
        
        key_metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'overall_score']
        
        for idx, metric in enumerate(key_metrics):
            if metric in df.columns:
                ax = axes[idx // 2, idx % 2]
                df[metric].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
                ax.axvline(df[metric].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df[metric].mean():.2f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Metrics Comparison (Box Plot)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_to_plot = [col for col in df.columns if col != 'overall_score']
        df_melted = df[metrics_to_plot].melt(var_name='Metric', value_name='Score')
        
        sns.boxplot(data=df_melted, x='Metric', y='Score', ax=ax)
        ax.set_title("RAG Metrics Comparison", fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Metrics Correlation Heatmap", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Metric Scores Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mean_scores = df.mean().sort_values(ascending=False)
        mean_scores.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title("Average Metric Scores", fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Average Score', fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(0.7, color='green', linestyle='--', label='Good Threshold (0.7)')
        ax.axhline(0.5, color='orange', linestyle='--', label='Fair Threshold (0.5)')
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "average_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def generate_detailed_report(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
        report_path: Path
    ):
        """
        Generate detailed HTML report.
        
        Args:
            results: List of evaluation results
            summary: Summary statistics
            report_path: Path to save report
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric-box {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-name {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .fair {{ color: #f39c12; font-weight: bold; }}
        .poor {{ color: #e74c3c; font-weight: bold; }}
        img {{
            max-width: 100%;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Financial ESG RAG Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric-box">
                <div class="metric-name">Total Queries</div>
                <div class="metric-value">{summary['total_queries']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-name">Successful</div>
                <div class="metric-value">{summary['successful_queries']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-name">Overall Score</div>
                <div class="metric-value">{summary['metrics'].get('overall_score', {}).get('mean', 0):.2f}</div>
            </div>
        </div>
        
        <h2>Detailed Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add metrics table
        for metric_name, stats in summary['metrics'].items():
            mean_val = stats['mean']
            status_class = 'good' if mean_val >= 0.7 else ('fair' if mean_val >= 0.5 else 'poor')
            html_content += f"""
                <tr>
                    <td><strong>{metric_name.replace('_', ' ').title()}</strong></td>
                    <td class="{status_class}">{stats['mean']:.3f}</td>
                    <td>{stats['median']:.3f}</td>
                    <td>{stats['std']:.3f}</td>
                    <td>{stats['min']:.3f}</td>
                    <td>{stats['max']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </tbody>
        </table>
        
        <h2>Visualizations</h2>
        <img src="metrics_distribution.png" alt="Metrics Distribution">
        <img src="metrics_comparison.png" alt="Metrics Comparison">
        <img src="metrics_correlation.png" alt="Metrics Correlation">
        <img src="average_scores.png" alt="Average Scores">
        
        <h2>Individual Query Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Question</th>
                    <th>Faithfulness</th>
                    <th>Relevancy</th>
                    <th>Overall</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add individual results
        for idx, result in enumerate(results, 1):
            metrics = result.get("metrics", {})
            status_class = 'good' if result['status'] == 'success' else 'poor'
            
            faithfulness = metrics.get('faithfulness', 0)
            relevancy = metrics.get('answer_relevancy', 0)
            overall = metrics.get('overall_score', 0)
            
            html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{result['question']}</td>
                    <td>{faithfulness:.2f}</td>
                    <td>{relevancy:.2f}</td>
                    <td>{overall:.2f}</td>
                    <td class="{status_class}">{result['status'].upper()}</td>
                </tr>
            """
        
        html_content += """
            </tbody>
        </table>
        
        <h2>Recommendations</h2>
        <ul>
        """
        
        # Add recommendations based on metrics
        overall_mean = summary['metrics'].get('overall_score', {}).get('mean', 0)
        faithfulness_mean = summary['metrics'].get('faithfulness', {}).get('mean', 0)
        relevancy_mean = summary['metrics'].get('answer_relevancy', {}).get('mean', 0)
        
        if overall_mean >= 0.7:
            html_content += "<li><strong>Excellent performance!</strong> The RAG system performs well across all metrics.</li>"
        elif overall_mean >= 0.5:
            html_content += "<li><strong>Good performance</strong> with room for improvement in specific areas.</li>"
        else:
            html_content += "<li><strong>Performance needs improvement.</strong> Consider system optimization.</li>"
        
        if faithfulness_mean < 0.7:
            html_content += "<li>Improve <strong>faithfulness</strong>: Ensure answers stick to retrieved context, reduce hallucinations.</li>"
        
        if relevancy_mean < 0.7:
            html_content += "<li>Improve <strong>answer relevancy</strong>: Fine-tune prompt engineering and query understanding.</li>"
        
        html_content += """
        </ul>
    </div>
</body>
</html>
        """
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report saved to {report_path}")
    
    def run_evaluation(
        self,
        include_adversarial: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Args:
            include_adversarial: Whether to include adversarial test cases
            
        Returns:
            Evaluation results and summary
        """
        logger.info("Starting RAG evaluation...")
        
        # Get test cases
        test_cases = ESGTestDataset.get_test_cases()
        logger.info(f"Loaded {len(test_cases)} standard test cases")
        
        if include_adversarial:
            adversarial_cases = ESGAdversarialDataset.get_test_cases()
            logger.info(f"Loaded {len(adversarial_cases)} adversarial test cases")
            test_cases.extend(adversarial_cases)
        
        # Run evaluation
        results = self.evaluate_dataset(test_cases)
        
        # Generate summary
        summary = self.generate_summary_statistics(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.output_dir / f"eval_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(results_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualizations
        self.generate_visualizations(results, results_dir)
        
        # Generate HTML report
        self.generate_detailed_report(results, summary, results_dir / "report.html")
        
        logger.info(f"Evaluation complete! Results saved to {results_dir}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("RAG EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Queries: {summary['total_queries']}")
        print(f"Successful: {summary['successful_queries']}")
        print(f"Failed: {summary['failed_queries']}")
        print("\nKey Metrics (Mean):")
        print("-"*60)
        
        for metric_name, stats in sorted(summary['metrics'].items()):
            print(f"{metric_name.replace('_', ' ').title():.<40} {stats['mean']:.3f}")
        
        print("="*60)
        print(f"\nFull report: {results_dir / 'report.html'}")
        print("="*60 + "\n")
        
        return {
            "results": results,
            "summary": summary,
            "output_dir": str(results_dir)
        }


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Financial ESG RAG System")
    parser.add_argument(
        "--output-dir",
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--no-adversarial",
        action="store_true",
        help="Exclude adversarial test cases"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = RAGEvaluator(output_dir=args.output_dir)
    results = evaluator.run_evaluation(include_adversarial=not args.no_adversarial)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {results['output_dir']}")
    print(f"📈 Overall Score: {results['summary']['metrics'].get('overall_score', {}).get('mean', 0):.3f}")


if __name__ == "__main__":
    main()
