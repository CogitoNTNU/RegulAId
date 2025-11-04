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

# Ensure project root (parent of this folder) is on sys.path for `src` imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.vector import VectorRetriever
from src.api.services.openai_service import OpenAIService
from src.api.config import OPENAI_MODEL, SYSTEM_PROMPT, RETRIEVER_TOP_K

# Plotting utils
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# =========================
# Configuration (edit here)
# =========================
EVALUATE_METHODS: List[str] = [
    "bm25_retrieval_only",
    "semantic_retrieval_only",
    "hybrid_retrieval",
    "bm25_rag",
    "semantic_rag",
    "hybrid_rag",
    "llm_only",
]
RUN_TAG: str = "retrieval_only_test"
RAGAS_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
]
# Leave placeholders; user will import their implementations.
# Example:
# from src.templates.hybrid_retrieval_only import HybridRetrievalOnly
# from src.templates.hybrid_retrieval_reranked import HybridRetrievalReranked
class _APIRAGBase:
    def __init__(self, retriever, k: int = None, **_):
        self.retriever = retriever
        self.k = int(k) if k is not None else int(RETRIEVER_TOP_K)
        self.llm = OpenAIService(model=OPENAI_MODEL, system_prompt=SYSTEM_PROMPT)

    def _run(self, question: str) -> Dict[str, object]:
        docs = self.retriever.search(query=question, k=self.k) or []
        # Build context string similar to API router
        context = ""
        if docs:
            context = "Context from EU AI Act documents:\n\n"
            for i, d in enumerate(docs, 1):
                content = d.get("content", "")
                context += f"[{i}] {content}\n\n"
            context += "---\n\n"
        enhanced_query = (context + question) if context else question
        llm_resp = self.llm.generate_text(prompt=enhanced_query, history=[])
        contexts = [str(d.get("content", "")) for d in docs if d.get("content")]
        return {"answer": llm_resp.content, "context": contexts}

    def run(self, question: str, reference_contexts=None):
        return self._run(question)


class BM25RAG_API(_APIRAGBase):
    def __init__(self, **kwargs):
        super().__init__(BM25Retriever(), **kwargs)


class VectorRAG_API(_APIRAGBase):
    def __init__(self, **kwargs):
        super().__init__(VectorRetriever(), **kwargs)


class HybridRAG_API(_APIRAGBase):
    def __init__(self, **kwargs):
        super().__init__(HybridRetriever(), **kwargs)


TEMPLATES: Dict[str, object] = {
    "bm25_retrieval_only": BM25Retriever,
    "semantic_retrieval_only": VectorRetriever,
    "hybrid_retrieval": HybridRetriever,
    # Use API-backed RAG pipelines (system prompt + OpenAIService)
    "bm25_rag": BM25RAG_API,
    "semantic_rag": VectorRAG_API,
    "hybrid_rag": HybridRAG_API,
    # Baseline: LLM without any retrieval
    "llm_only": None,  # placeholder, class defined below
}


class LLMOnly_API:
    """LLM baseline: answer without any retrieval contexts."""
    def __init__(self, **_):
        self.llm = OpenAIService(model=OPENAI_MODEL, system_prompt=SYSTEM_PROMPT)

    def run(self, question: str, reference_contexts=None):
        llm_resp = self.llm.generate_text(prompt=question, history=[])
        return {"answer": llm_resp.content, "context": []}


# Fill placeholder now that class exists
TEMPLATES["llm_only"] = LLMOnly_API

# Input files (relative to repo root)
TESTSET_REL_PATH = os.path.join("data", "processed", "testset.json")


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
    """Run one template over all questions inside a separate process.
    Tries to adapt to different constructor and run method signatures.
    """
    llm = llm_factory()
    embeddings = embedding_factory()

    # Best-effort construction with fallbacks
    template = None
    construct_attempts = [
        {"llm": llm, "embeddings": embeddings},
        {"embeddings": embeddings},
        {"llm": llm},
        {},
    ]
    last_exc = None
    for kwargs in construct_attempts:
        try:
            template = template_cls(**kwargs)
            break
        except TypeError as exc:
            last_exc = exc
            continue
    if template is None:
        raise last_exc if last_exc else RuntimeError(f"Failed to construct template {template_name}")

    results: List[Dict] = []
    for _, row in questions_df.iterrows():
        start_time = time.time()
        answer = ""
        retrieved_context: List[str] = []
        try:
            # Try a 'run' method first
            if hasattr(template, "run"):
                try:
                    out = template.run(row['question'], row.get('reference_contexts'))
                except TypeError:
                    out = template.run(row['question'])
                # Normalize output
                if isinstance(out, dict):
                    answer = out.get('answer', answer)
                    ctx = out.get('context') or out.get('contexts') or out.get('retrieved_context')
                    if isinstance(ctx, str):
                        retrieved_context = [ctx]
                    elif isinstance(ctx, list):
                        retrieved_context = [str(c) for c in ctx]
                elif isinstance(out, str):
                    # Treat as answer-only (for RAGAS, still requires contexts)
                    answer = out
                elif isinstance(out, (list, tuple)):
                    # Treat as contexts
                    retrieved_context = [str(c) for c in out]
            # Fallback to a 'retrieve' method if present
            elif hasattr(template, "retrieve"):
                try:
                    ctx = template.retrieve(row['question'])
                except TypeError:
                    # try without args
                    ctx = template.retrieve()
                if isinstance(ctx, str):
                    retrieved_context = [ctx]
                elif isinstance(ctx, list):
                    retrieved_context = [str(c) for c in ctx]
                # No generation => leave answer empty
            else:
                raise AttributeError(f"Template {template_name} has neither run nor retrieve")

            response_time = time.time() - start_time
        except Exception as exc:
            response_time = 0.0
            print(f"[{template_name}] Exception: {exc}")
            traceback.print_exc()
            # keep defaults (empty answer/context)

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
    # Adapt metric set based on presence of answers and contexts
    answers_present = any((a or "").strip() for a in eval_data["answer"])
    contexts_present = any(len(c or []) > 0 for c in eval_data["contexts"])
    metrics_to_use = ragas_metrics
    if answers_present and not contexts_present:
        # LLM-only baseline: only answer_relevancy is meaningful
        metrics_to_use = [m for m in ragas_metrics if m.name in ("answer_relevancy",)]
    elif not answers_present and contexts_present:
        # retrieval-only: only context metrics are meaningful
        metrics_to_use = [m for m in ragas_metrics if m.name in ("context_precision", "context_recall")]
    try:
        ragas_result = evaluate(ds, metrics=metrics_to_use)
        df_res = ragas_result.to_pandas()
        scores = {m.name: float(df_res[m.name].mean()) for m in metrics_to_use}
        # Ensure missing metrics are present with 0.0 for downstream code
        for m in ragas_metrics:
            scores.setdefault(m.name, 0.0)
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

    print("Loading testset...")
    with open(testset_path, 'r', encoding='utf-8') as f:
        testset_data = json.load(f)

    # Components are optional; set to zero if unused
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
