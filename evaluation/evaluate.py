import os
import json
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# RAGAS
from ragas import evaluate
from ragas.llms.base import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# Plotting utils
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# =========================
# Configuration (edit here)
# =========================
EVALUATE_METHODS: List[str] = [
    "hybrid_retrieval_only",
    "hybrid_retrieval_reranked",
    # TODO: add your custom method names here and map them in TEMPLATES below
]
RUN_TAG: str = "retrieval_only_test"
RAGAS_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
]

# TODO: Map your template names to callables/classes implementing a .run(question, reference_contexts)
# Leave placeholders; user will import their implementations.
# Example:
# from src.templates.hybrid_retrieval_only import HybridRetrievalOnly
# from src.templates.hybrid_retrieval_reranked import HybridRetrievalReranked
TEMPLATES: Dict[str, object] = {
    # TODO: uncomment and ensure paths are valid for your project
    # "hybrid_retrieval_only": HybridRetrievalOnly,
    # "hybrid_retrieval_reranked": HybridRetrievalReranked,
}

# Input files (relative to repo root)
TESTSET_REL_PATH = os.path.join("data", "datasets", "testset.json")
# Optional: extra metadata/components if you need them
COMPONENTS_REL_PATH = os.path.join("data", "components.json")


# =========================
# Metric helpers
# =========================

def calculate_context_overlap(reference_contexts: List[str], retrieved_contexts: List[str]) -> float:
    if not reference_contexts or not retrieved_contexts:
        return 0.0
    ref_words = set()
    for ctx in reference_contexts:
        ref_words.update(ctx.lower().split())
    retr_words = set()
    for ctx in retrieved_contexts:
        retr_words.update(ctx.lower().split())
    if not ref_words or not retr_words:
        return 0.0
    intersection = len(ref_words.intersection(retr_words))
    union = len(ref_words.union(retr_words))
    return intersection / union if union > 0 else 0.0


def calculate_precision_at_k(reference_contexts: List[str], retrieved_contexts: List[str], k: int = 5) -> float:
    if not reference_contexts or not retrieved_contexts:
        return 0.0
    ref_words = set()
    for ctx in reference_contexts:
        ref_words.update(ctx.lower().split())
    retr_words: List[str] = []
    for ctx in retrieved_contexts:
        retr_words.extend(ctx.lower().split())
    retr_words = retr_words[:k]
    if not retr_words:
        return 0.0
    relevant_words = sum(1 for word in retr_words if word in ref_words)
    return relevant_words / len(retr_words)


def calculate_recall_at_k(reference_contexts: List[str], retrieved_contexts: List[str], k: int = 5) -> float:
    if not reference_contexts or not retrieved_contexts:
        return 0.0
    ref_words = set()
    for ctx in reference_contexts:
        ref_words.update(ctx.lower().split())
    retr_words = set()
    for ctx in retrieved_contexts:
        retr_words.update(ctx.lower().split())
    retr_words = set(list(retr_words)[:k])
    if not ref_words:
        return 0.0
    relevant_words = len(ref_words.intersection(retr_words))
    return relevant_words / len(ref_words)


# =========================
# Visualization helpers
# =========================

def create_ranking_table(template_ragas_scores: Dict[str, Dict[str, float]],
                         template_additional_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    data = []
    for template_name in template_ragas_scores.keys():
        ragas_scores = template_ragas_scores[template_name]
        additional_scores = template_additional_scores[template_name]
        data.append({
            'Template': template_name,
            'Context Recall': ragas_scores.get('context_recall', 0.0),
            'Context Precision': ragas_scores.get('context_precision', 0.0),
            'Faithfulness': ragas_scores.get('faithfulness', 0.0),
            'Answer Relevancy': ragas_scores.get('answer_relevancy', 0.0),
            'Context Overlap': additional_scores['context_overlap'],
            'Precision@K': additional_scores['precision_at_k'],
            'Recall@K': additional_scores['recall_at_k'],
            'F1 Score': additional_scores['f1_score'],
            'Response Time': additional_scores['response_time']
        })
    return pd.DataFrame(data)


def create_radar_chart(template_ragas_scores: Dict[str, Dict[str, float]],
                       output_dir: str,
                       ragas_metrics) -> None:
    categories = [metric.name for metric in ragas_metrics]
    fig = go.Figure()
    for template_name in template_ragas_scores.keys():
        scores = template_ragas_scores[template_name]
        fig.add_trace(go.Scatterpolar(
            r=[scores.get(k, 0.0) for k in categories],
            theta=categories,
            fill='toself',
            name=template_name.replace('_', ' ').title()
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="RAGAS Metrics Comparison"
    )
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "ragas_metrics_radar_chart.html"))


def create_bar_charts(template_additional_scores: Dict[str, Dict[str, float]], output_dir: str) -> None:
    metrics = list(template_additional_scores[list(template_additional_scores.keys())[0]].keys())
    template_names = list(template_additional_scores.keys())
    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics)
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=template_names,
            y=[template_additional_scores[template_name][metric] for template_name in template_names],
            name=metric,
            showlegend=False
        ), row=i+1, col=1)
    fig.update_layout(height=1200, title_text="Additional Metrics Comparison")
    os.makedirs(output_dir, exist_ok=True)
    fig.write_html(os.path.join(output_dir, "additional_metrics_bar_charts.html"))


def perform_statistical_test(template1_results: List[Dict], template2_results: List[Dict]) -> None:
    template1_response_times = [item['response_time'] for item in template1_results]
    template2_response_times = [item['response_time'] for item in template2_results]
    t_statistic, p_value = stats.ttest_ind(template1_response_times, template2_response_times)
    print(f"\nStatistical Test (Template A vs Template B):")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("The difference in response times is statistically significant.")
    else:
        print("The difference in response times is not statistically significant.")


def get_ranked_templates(template_ragas_scores: Dict[str, Dict[str, float]],
                         template_additional_scores: Dict[str, Dict[str, float]],
                         recall_weight: float = 0.3,
                         precision_weight: float = 0.3,
                         faithfulness_weight: float = 0.4) -> List[Tuple[str, float]]:
    template_scores: Dict[str, float] = {}
    for template_name in template_ragas_scores.keys():
        ragas_scores = template_ragas_scores[template_name]
        weighted_score = (
            recall_weight * ragas_scores.get('context_recall', 0.0) +
            precision_weight * ragas_scores.get('context_precision', 0.0) +
            faithfulness_weight * ragas_scores.get('faithfulness', 0.0)
        )
        template_scores[template_name] = weighted_score
    ranked_templates = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_templates


def provide_recommendations(ranked_templates: List[Tuple[str, float]],
                            template_ragas_scores: Dict[str, Dict[str, float]],
                            template_additional_scores: Dict[str, Dict[str, float]],
                            ragas_metrics) -> None:
    best_template = ranked_templates[0][0]
    print(f"\nRecommended Template: {best_template}\n")
    print("Strengths Analysis:")
    ragas_scores = template_ragas_scores[best_template]
    additional_scores = template_additional_scores[best_template]
    for metric in ragas_metrics:
        print(f"- {metric.name}: {ragas_scores.get(metric.name, 0.0):.4f}")
    for metric, score in additional_scores.items():
        print(f"- {metric}: {score:.4f}")
    print("\nProduction Deployment Recommendations:")
    print("- Consider the trade-offs between recall, precision, and response time.")
    print("- Monitor and adjust weights as you gather more data.")
    print("- Continuously evaluate with new test cases to ensure effectiveness.")


def save_results(template_results: Dict[str, List[Dict]],
                 template_ragas_scores: Dict[str, Dict[str, float]],
                 template_additional_scores: Dict[str, Dict[str, float]],
                 ranked_templates: List[Tuple[str, float]],
                 output_dir: str) -> None:
    results = {
        "template_results": {k: [{**i, 'retrieved_context': [str(c) for c in i['retrieved_context']]} for i in v] for k, v in template_results.items()},
        "template_ragas_scores": template_ragas_scores,
        "template_additional_scores": template_additional_scores,
        "ranked_templates": ranked_templates,
    }
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "rag_evaluation_results.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")


def aggregate_results(results_dir: str = "results") -> List[Dict]:
    summary_data: List[Dict] = []
    for run_folder in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_folder)
        if os.path.isdir(run_path):
            results_file = os.path.join(run_path, "rag_evaluation_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                for template_name, scores in data["template_ragas_scores"].items():
                    additional_scores = data["template_additional_scores"][template_name]
                    summary_data.append({
                        "run_folder": run_folder,
                        "template": template_name,
                        "context_recall": scores.get("context_recall"),
                        "context_precision": scores.get("context_precision"),
                        "answer_relevancy": scores.get("answer_relevancy"),
                        "response_time": additional_scores.get("response_time"),
                        "avg_score": float(np.mean([
                            scores.get("context_recall", 0.0),
                            scores.get("context_precision", 0.0),
                            scores.get("answer_relevancy", 0.0),
                        ])),
                    })
    summary_filename = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_filename, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"Aggregated summary saved to {summary_filename}")
    return summary_data


def create_comparison_charts(summary_data: List[Dict], results_dir: str = "results") -> None:
    if not summary_data:
        print("No summary data to create comparison charts.")
        return
    df = pd.DataFrame(summary_data)
    metrics_to_plot = [
        "context_recall", "context_precision", "response_time", "answer_relevancy", "avg_score"
    ]
    fig = make_subplots(rows=len(metrics_to_plot), cols=1, subplot_titles=metrics_to_plot)
    for i, metric in enumerate(metrics_to_plot):
        for template_name in df['template'].unique():
            template_df = df[df['template'] == template_name]
            fig.add_trace(go.Bar(
                x=template_df['run_folder'],
                y=template_df[metric],
                name=template_name
            ), row=i+1, col=1)
    fig.update_layout(height=1500, title_text="Cross-Evaluation Comparison")
    comparison_filename = os.path.join(results_dir, "cross_evaluation_comparison.html")
    fig.write_html(comparison_filename)
    print(f"Comparison chart saved to {comparison_filename}")


# =========================
# Core runner
# =========================

def run_template_worker(template_name: str,
                        template_cls,
                        questions_df: pd.DataFrame) -> Tuple[str, List[Dict]]:
    """Run one template over all questions inside a separate process."""
    llm = llm_factory()
    embeddings = embedding_factory()
    template = template_cls(llm=llm, embeddings=embeddings)

    results: List[Dict] = []
    for _, row in questions_df.iterrows():
        start_time = time.time()
        try:
            # Template contract: run(question, reference_contexts) -> {answer, context}
            result = template.run(row['question'], row['reference_contexts'])
            answer = result['answer']
            retrieved_context = result['context']
            if isinstance(retrieved_context, str):
                retrieved_context = [retrieved_context]
            response_time = time.time() - start_time
        except Exception as exc:
            answer = "Error occurred during processing"
            retrieved_context = []
            response_time = 0.0
            print(f"[{template_name}] Exception: {exc}")
            traceback.print_exc()

        results.append({
            "question": row["question"],
            "answer": answer,
            "retrieved_context": retrieved_context,
            "reference_contexts": row["reference_contexts"],
            "response_time": response_time,
            "ground_truths": row["ground_truths"],
        })

    return template_name, results


def run_ragas_worker(template_name: str,
                     results: List[Dict],
                     ragas_metrics) -> Tuple[str, Dict[str, float]]:
    """Compute RAGAS metrics for a single template in a separate process."""
    eval_data = {
        "question":     [r["question"] for r in results],
        "answer":       [r["answer"] for r in results],
        "contexts":     [r["retrieved_context"] for r in results],
        "ground_truth": [r["ground_truths"][0] for r in results],
    }
    ds = Dataset.from_dict(eval_data)
    try:
        ragas_result = evaluate(ds, metrics=ragas_metrics)
        df_res = ragas_result.to_pandas()
        scores = {m.name: float(df_res[m.name].mean()) for m in ragas_metrics}
    except Exception as exc:
        print(f"[RAGAS-{template_name}] Evaluation failed: {exc}")
        scores = {m.name: 0.0 for m in ragas_metrics}
    return template_name, scores


def main() -> None:
    load_dotenv()

    # Resolve project root and paths
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    testset_path = os.path.join(project_root, TESTSET_REL_PATH)
    components_path = os.path.join(project_root, COMPONENTS_REL_PATH)

    print("Loading testset...")
    with open(testset_path, 'r', encoding='utf-8') as f:
        testset_data = json.load(f)

    # Optional components file, ignore if missing
    if os.path.exists(components_path):
        with open(components_path, 'r', encoding='utf-8') as f:
            components_data = json.load(f)
        num_components = len(components_data)
    else:
        components_data = []
        num_components = 0

    num_questions = len(testset_data)

    dataset = [{
        'question': item['question'],
        'ground_truths': [item['ground_truth']],
        'reference_contexts': item['contexts'],
    } for item in testset_data]
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} test cases")

    # Select templates to run
    if not TEMPLATES:
        print("TODO: Configure TEMPLATES mapping to your implementations. No templates to run.")
        return
    templates_to_run = {name: TEMPLATES[name] for name in EVALUATE_METHODS if name in TEMPLATES}

    print("Templates requested:", EVALUATE_METHODS)
    print("Will run:", list(templates_to_run))

    # Run templates in parallel
    print("Running RAG templates in parallel...")
    template_results: Dict[str, List[Dict]] = {}
    max_workers = max(1, min(cpu_count(), len(templates_to_run)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_template_worker, name, cls, df) for name, cls in templates_to_run.items()]
        done, _ = wait(futures, return_when=ALL_COMPLETED)
        for fut in done:
            try:
                tpl_name, results = fut.result()
                template_results[tpl_name] = results
            except Exception as exc:
                print(f"[main] Worker failed: {exc}")

    if not template_results:
        print("No template results collected. Exiting.")
        return

    # Compute RAGAS in parallel
    print("Calculating RAGAS metrics in parallel...")
    template_ragas_scores: Dict[str, Dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=max(1, min(cpu_count(), len(template_results)))) as pool:
        ragas_futures = [pool.submit(run_ragas_worker, name, results, RAGAS_METRICS) for name, results in template_results.items()]
        done, _ = wait(ragas_futures, return_when=ALL_COMPLETED)
        for fut in done:
            try:
                name, scores = fut.result()
                template_ragas_scores[name] = scores
                print("  "+name+" RAGAS:", ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
            except Exception as exc:
                print(f"[main] RAGAS worker failed: {exc}")

    # Additional metrics
    print("Calculating additional metrics...")
    template_additional_scores: Dict[str, Dict[str, float]] = {}
    for template_name, results in template_results.items():
        print(f"Calculating additional metrics for {template_name}...")
        context_overlaps: List[float] = []
        precision_at_k_scores: List[float] = []
        recall_at_k_scores: List[float] = []
        f1_scores: List[float] = []
        response_times = [item['response_time'] for item in results]
        for item in results:
            overlap = calculate_context_overlap(item['reference_contexts'], item['retrieved_context'])
            context_overlaps.append(overlap)
            p_at_k = calculate_precision_at_k(item['reference_contexts'], item['retrieved_context'], 5)
            precision_at_k_scores.append(p_at_k)
            r_at_k = calculate_recall_at_k(item['reference_contexts'], item['retrieved_context'], 5)
            recall_at_k_scores.append(r_at_k)
            f1 = 2 * (p_at_k * r_at_k) / (p_at_k + r_at_k) if (p_at_k + r_at_k) > 0 else 0.0
            f1_scores.append(f1)
        template_additional_scores[template_name] = {
            'context_overlap': float(np.mean(context_overlaps)) if context_overlaps else 0.0,
            'precision_at_k': float(np.mean(precision_at_k_scores)) if precision_at_k_scores else 0.0,
            'recall_at_k': float(np.mean(recall_at_k_scores)) if recall_at_k_scores else 0.0,
            'f1_score': float(np.mean(f1_scores)) if f1_scores else 0.0,
            'response_time': float(np.mean(response_times)) if response_times else 0.0,
        }
        print(f"  {template_name} additional scores:")
        for metric, score in template_additional_scores[template_name].items():
            print(f"    {metric}: {score}")

    # Ranking / output
    print("\nCreating ranking table...")
    ranking_table = create_ranking_table(template_ragas_scores, template_additional_scores)
    print("\nRanking Table:")
    try:
        print(ranking_table.to_string(index=False))
    except Exception:
        print(ranking_table)

    ranked_templates = get_ranked_templates(template_ragas_scores, template_additional_scores)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_methods = len(EVALUATE_METHODS)
    run_folder_name = f"M{num_methods}_Q{num_questions}_C{num_components}_{RUN_TAG}_{timestamp}"
    output_dir = os.path.join("results", run_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    create_radar_chart(template_ragas_scores, output_dir, RAGAS_METRICS)
    create_bar_charts(template_additional_scores, output_dir)

    # Only test significance if two or more
    if len(template_results) >= 2:
        first_two = list(template_results.keys())[:2]
        perform_statistical_test(template_results[first_two[0]], template_results[first_two[1]])
    else:
        print("Not enough templates to run a statistical significance test — skipping.")

    provide_recommendations(ranked_templates, template_ragas_scores, template_additional_scores, RAGAS_METRICS)
    save_results(template_results, template_ragas_scores, template_additional_scores, ranked_templates, output_dir)

    summary_data = aggregate_results()
    create_comparison_charts(summary_data)
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
