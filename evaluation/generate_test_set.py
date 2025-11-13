import os
import json
import getpass
import asyncio
import typing as t
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass
from openai import OpenAI

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.llms.base import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.testset.transforms import apply_transforms, HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor, OverlapScoreBuilder
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop import SingleHopQuerySynthesizer, SingleHopScenario
from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer, MultiHopScenario
from ragas.testset.synthesizers.prompts import ThemesPersonasInput, ThemesPersonasMatchingPrompt

# --- Configuration ---

# TODO: Update paths and sizes for your data/use-case
DOCS_DIR = "../data/processed/aiact-chunks.json"
OUTPUT_DIR = "../data/processed"
OUTPUT_FILE = "testset.json"
NUM_SINGLE_HOP = 25  
NUM_MULTI_HOP = 25   
LLM_MODEL = os.getenv("OPENAI_MODEL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
OPENAI_KEY = os.getenv("OPENAI_KEY")
# --- End Configuration ---

@dataclass
class MySingleHopScenario(SingleHopQuerySynthesizer):
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(self, n, knowledge_graph, persona_list, callbacks):
        property_name = "keyphrases"
        nodes = [node for node in knowledge_graph.nodes if node.type.name == "CHUNK" and node.get_property(property_name)]
        if not nodes:
            return []

        number_of_samples_per_node = max(1, n // len(nodes))
        scenarios = []
        for node in nodes:
            if len(scenarios) >= n:
                break
            themes = node.properties.get(property_name, [""])
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
            base_scenarios = self.prepare_combinations(node, themes, personas=persona_list, persona_concepts=persona_concepts.mapping)
            scenarios.extend(self.sample_combinations(base_scenarios, number_of_samples_per_node))
        return scenarios

@dataclass
class MyMultiHopQuery(MultiHopQuerySynthesizer):
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(self, n: int, knowledge_graph, persona_list, callbacks) -> t.List[MultiHopScenario]:
        results = knowledge_graph.find_two_nodes_single_rel(relationship_condition=lambda rel: (True if rel.type == "keyphrases_overlap" else False))
        if not results:
            return []
            
        num_sample_per_triplet = max(1, n // len(results))
        scenarios = []
        for triplet in results:
            if len(scenarios) >= n:
                break
            node_a, node_b = triplet[0], triplet[-1]
            overlapped_keywords = triplet[1].properties["overlapped_items"]
            if overlapped_keywords:
                themes = list(dict(overlapped_keywords).keys())
                prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
                persona_concepts = await self.theme_persona_matching_prompt.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
                overlapped_keywords = [list(item) for item in overlapped_keywords]
                base_scenarios = self.prepare_combinations([node_a, node_b], overlapped_keywords, personas=persona_list, persona_item_mapping=persona_concepts.mapping, property_name="keyphrases")
                base_scenarios = self.sample_diverse_combinations(base_scenarios, num_sample_per_triplet)
                scenarios.extend(base_scenarios)
        return scenarios

async def main():
    """
    Generates a test set with single-hop and multi-hop questions.
    """
    import json

    file = "../data/processed/aiact-chunks.json"
    #evaluation\generate_test-set.ipynb
    with open(file, 'r', encoding="utf-8") as file:
        data = json.load(file)
    # ----------------------------
    # 1. Load documents
    # ----------------------------
    # Use already-loaded chunks/data instead of markdown loader
    try:
        source_items = data
    except NameError:
        source_items = None

    if isinstance(source_items, dict) and "chunks" in source_items:
        source_items = source_items["chunks"]

    if not source_items:
        print("No preloaded chunks/data found. Provide `chunks` or `data` in the notebook environment.")
        return

    # Build Documents from structured items
    all_docs = []
    for item in source_items:
        text = item.get("text", "")
        metadata = {
            "id": item.get("id"),
            "type": item.get("type"),
            "paragraph_number": item.get("paragraph_number"),
            "page_range": item.get("page_range"),
            "chapter_number": item.get("chapter_number"),
            "chapter_name": item.get("chapter_name"),
            "section_number": item.get("section_number"),
            "section_name": item.get("section_name"),
            "article_number": item.get("article_number"),
            "article_name": item.get("article_name"),
            "annex_number": item.get("annex_number"),
            "annex_name": item.get("annex_name"),
            "source": "chunks"
        }
        all_docs.append(Document(page_content=text, metadata=metadata))

    # ----------------------------
    # 2. Create Knowledge Graph
    # ----------------------------
    kg = KnowledgeGraph()
    for doc in all_docs:
        kg.nodes.append(Node(type=NodeType.DOCUMENT, properties={"page_content": doc.page_content, "document_metadata": doc.metadata}))

    # ----------------------------
    # 3. Load environment and API keys
    # ----------------------------
    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    # ----------------------------
    # 4. Set up LLM, Embeddings and Transforms
    # ----------------------------
    client = OpenAI(api_key=os.environ["OPENAI_KEY"])

    # Provide safe defaults if env vars are missing
    llm_model = (LLM_MODEL or "gpt-4o-mini")
    embedding_model_name = (EMBEDDING_MODEL_NAME or "text-embedding-3-small")

    # Minimal OpenAI chat adapter returning an LLMResult-like object
    class _OpenAIChatLLM:
        def __init__(self, client, model):
            self._client = client
            self._model = model
        def _normalize_prompt(self, prompt):
            try:
                if hasattr(prompt, "to_string"):
                    return prompt.to_string()
                if hasattr(prompt, "to_str"):
                    return prompt.to_str()
                if hasattr(prompt, "to_messages"):
                    msgs = prompt.to_messages()
                    return "\n".join(getattr(m, "content", str(m)) for m in msgs)
            except Exception:
                pass
            return str(prompt)
        def _clean_text(self, text: str) -> str:
            try:
                import re
                # Prefer fenced JSON block if present
                m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
                if m:
                    return m.group(1).strip()
                # Otherwise, strip surrounding fences if any
                if text.strip().startswith("```") and text.strip().endswith("```"):
                    return text.strip().strip("`").strip()
            except Exception:
                pass
            return text.strip()
        def _call_sync(self, prompt, **kwargs):
            user_text = self._normalize_prompt(prompt)
            params = {}
            for k in ("stop", "temperature", "max_tokens", "top_p"):
                if k in kwargs and kwargs[k] is not None:
                    params[k] = kwargs[k]
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
                **params,
            )
            raw = resp.choices[0].message.content
            return self._clean_text(raw)
        def _as_llm_result(self, text):
            class _Gen:
                def __init__(self, text):
                    self.text = text
            class _LLMResult:
                def __init__(self, text):
                    self.generations = [[_Gen(text)]]
            return _LLMResult(text)
        def generate(self, prompt, **kwargs):
            text = self._call_sync(prompt, **kwargs)
            return self._as_llm_result(text)
        async def agenerate(self, prompt, **kwargs):
            import asyncio
            text = await asyncio.to_thread(self._call_sync, prompt, **kwargs)
            return self._as_llm_result(text)
        async def agenerate_prompt(self, prompt, **kwargs):
            return await self.agenerate(prompt, **kwargs)

    llm = _OpenAIChatLLM(client, llm_model)

    # Embeddings (keep legacy for now; consider switching to modern API when ready)
    embedding = embedding_factory(model=embedding_model_name)

    headline_extractor = HeadlinesExtractor(llm=llm)
    headline_splitter = HeadlineSplitter(min_tokens=300, max_tokens=1000)
    keyphrase_extractor = KeyphrasesExtractor(llm=llm, property_name="keyphrases", max_num=10)
    relation_builder = OverlapScoreBuilder(property_name="keyphrases", new_property_name="overlap_score", threshold=0.01, distance_threshold=0.9)
    transforms = [headline_extractor, headline_splitter, keyphrase_extractor, relation_builder]
    apply_transforms(kg, transforms=transforms)

    # ----------------------------
    # 5. Configure personas
    # ----------------------------
    # Personas relevant to your users/stakeholders
    persona_list = [
        Persona(name="Lawyer", role_description=(
            "A highly analytical and detail-oriented professional who asks precise, "
            "legally framed questions. The Lawyer focuses on clarity, accuracy, and "
            "potential implications of answers, often referencing laws, regulations, "
            "or logical consistency."
        )),

        Persona(name="Regular person", role_description=(
            "An everyday user who asks practical or curiosity-driven questions in a "
            "casual tone. The Regular person seeks understandable and useful answers "
            "without deep technical or legal jargon."
        )),

        Persona(name="Ignorant person", role_description=(
            "A person with very limited background knowledge who asks naive, vague, "
            "or sometimes confused questions. The Ignorant person seeks simple, "
            "patient explanations and may misunderstand or mix up basic concepts."
        )),
    ]

    # ---------------------------
    # 6. Generate questions
    # ----------------------------
    single_hop_query_synth = MySingleHopScenario(llm=llm)
    multi_hop_query_synth = MyMultiHopQuery(llm=llm)

    print("Generating single-hop questions...")
    single_hop_scenarios = await single_hop_query_synth.generate_scenarios(n=NUM_SINGLE_HOP, knowledge_graph=kg, persona_list=persona_list)
    single_hop_results = [await single_hop_query_synth.generate_sample(s) for s in single_hop_scenarios]

    print("Generating multi-hop questions...")
    multi_hop_scenarios = await multi_hop_query_synth.generate_scenarios(n=NUM_MULTI_HOP, knowledge_graph=kg, persona_list=persona_list)
    multi_hop_results = [await multi_hop_query_synth.generate_sample(s) for s in multi_hop_scenarios]

    # ----------------------------
    # 7. Combine and Export
    # ----------------------------
    all_scenarios = single_hop_scenarios + multi_hop_scenarios
    all_results = single_hop_results + multi_hop_results
    test_data = []
    for scenario, result in zip(all_scenarios, all_results):
        contexts = [node.properties.get("page_content", "") for node in scenario.nodes]
        node_metas = []
        answer_ids = []
        answer_numbers = []
        for node in scenario.nodes:
            meta = {}
            if "document_metadata" in node.properties:
                meta = node.properties["document_metadata"]
            # Collect id and paragraph_number if present
            if isinstance(meta, dict):
                node_metas.append(meta)
                if meta.get("id") is not None:
                    answer_ids.append(meta.get("id"))
                if meta.get("paragraph_number") is not None:
                    answer_numbers.append(meta.get("paragraph_number"))
            else:
                node_metas.append({})
        test_data.append({
            "question": result.user_input,
            # Keep model-generated reference, but also store ids/numbers as authoritative answers
            "ground_truth": result.reference,
            "answer_ids": answer_ids,
            "answer_numbers": answer_numbers,
            "contexts": contexts,
            "metadata": node_metas
        })

    dataset_df = pd.DataFrame(test_data)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    dataset_df.to_json(output_path, orient="records", indent=4)

    print(f"✅ Testset generated and saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())