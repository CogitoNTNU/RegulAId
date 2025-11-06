"""Classification Agent for EU AI Act risk assessment."""

import importlib
import json
from typing import Any

from src.agents.tools import create_retrieval_tools
from src.schemas.agent_schemas import ClassificationRequest, ClassificationResponse


CLASSIFICATION_SYSTEM_PROMPT = """You are an expert EU AI Act compliance advisor specializing in classifying AI systems according to their risk level.

Your task is to analyze the user's AI system description and classify it according to the EU AI Act risk categories:
1. **Prohibited**: AI practices that are banned (e.g., social scoring, subliminal manipulation, exploiting vulnerabilities)
2. **High-risk**: AI systems in critical areas (e.g., biometric identification, critical infrastructure, law enforcement, employment, education)
3. **Limited-risk**: AI systems with transparency obligations (e.g., chatbots, deepfakes)
4. **Minimal-risk**: AI systems with no or minimal risk (most AI applications)

You have access to tools to search the EU AI Act database for relevant information.

**Your classification process:**
1. First, analyze the AI system description to understand what it does
2. Use the retrieve_eu_ai_act tool to find relevant articles about similar systems
3. Use retrieve_system_type_info if you need information about specific system types
4. Based on the retrieved information, determine the risk level
5. If you don't have enough information, identify what clarifying questions to ask

**Important guidelines:**
- Be thorough: Use the tools to find relevant articles before making a classification
- Be cautious: If unsure, ask for more information rather than guessing
- Be specific: Identify the exact articles that support your classification
- Be clear: Explain your reasoning in simple terms

When you have enough information to classify, output your final answer in this EXACT JSON format:
{{
    "risk_level": "prohibited|high-risk|limited-risk|minimal-risk",
    "system_type": "description of system type",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "needs_more_info": false,
    "questions": [],
    "relevant_articles": ["Article X", "Article Y"]
}}

When you need more information, output:
{{
    "risk_level": null,
    "system_type": null,
    "confidence": null,
    "reasoning": "explanation of what information is missing",
    "needs_more_info": true,
    "questions": ["Question 1?", "Question 2?"],
    "relevant_articles": []
}}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the JSON response in the exact format specified above

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


class ClassificationAgent:
    """Agent for classifying AI systems according to EU AI Act risk levels."""

    def __init__(self, retriever: Any, OPENAI_KEY: str, model: str = "gpt-4o"):
        """
        Initialize the Classification Agent.

        Args:
            retriever: The retriever instance (BM25, Vector, or Hybrid)
            OPENAI_KEY: OpenAI API key
            model: OpenAI model to use (default: gpt-4o)
        """
        self.retriever = retriever
        self.model = model

        # Create LangChain tools using the retriever
        self.tools = create_retrieval_tools(retriever, top_k=5)

        # Helper: discover a ChatOpenAI-like class from several modules (prefer langchain_openai)
        def find_chat_openai():
            candidates = [
                'langchain_openai',
                'langchain.chat_models.openai',
                'langchain.chat_models',
                'langchain.llms.openai',
                'langchain.llms',
            ]
            for modname in candidates:
                try:
                    mod = importlib.import_module(modname)
                except Exception:
                    continue
                # Try common names
                for attr in ('ChatOpenAI', 'OpenAI', 'ChatModel'):
                    candidate = getattr(mod, attr, None)
                    if candidate:
                        return candidate
            return None

        def instantiate_llm(LMClass, key, model_name):
            # Try several kwarg names for API key to be compatible with different wrappers
            kwargs_variants = [
                {'model': model_name, 'api_key': key, 'temperature': 0.1},
                {'model': model_name, 'openai_api_key': key, 'temperature': 0.1},
                {'model': model_name, 'client': key, 'temperature': 0.1},
                {'model': model_name, 'temperature': 0.1},
            ]
            last_exc = None
            for kw in kwargs_variants:
                try:
                    return LMClass(**kw)
                except TypeError as e:
                    last_exc = e
                    continue
                except Exception as e:
                    last_exc = e
                    continue
            raise RuntimeError(f"Failed to instantiate LLM {LMClass}: {last_exc}")

        # Try modern LangChain agent API first
        LMClass = find_chat_openai()
        agents_mod = None
        initialize_agent = None
        AgentType = None
        if LMClass is not None:
            try:
                agents_mod = importlib.import_module('langchain.agents')
                initialize_agent = getattr(agents_mod, 'initialize_agent', None)
                AgentType = getattr(agents_mod, 'AgentType', None)
            except Exception:
                agents_mod = None

        if LMClass is not None and callable(initialize_agent) and AgentType is not None:
            # Try modern initialize_agent API but fall back silently on failure
            try:
                self.llm = instantiate_llm(LMClass, OPENAI_KEY, model)
                init_fn = initialize_agent
                if not callable(init_fn):
                    raise RuntimeError('initialize_agent is not callable')

                # `init_fn` has been checked with `callable(init_fn)` above; silence type checker
                self.agent_executor = init_fn(  # type: ignore
                    tools=self.tools,
                    llm=self.llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=10,
                    return_intermediate_steps=False,
                )
                return
            except Exception:
                # If initialize_agent fails due to API mismatch, continue to older fallback
                pass

        # Otherwise, fallback to older agent API or a simple executor
        if LMClass is None:
            raise ImportError('Could not locate a ChatOpenAI/OpenAI LLM class in langchain or langchain_openai')

        # Try to obtain older-style agent builders (create_agent / create_react_agent) and AgentExecutor
        amod = None
        try:
            amod = importlib.import_module('langchain.agents')
        except Exception:
            amod = None

        AgentExecutor = getattr(amod, 'AgentExecutor', None) if amod is not None else None
        create_agent = getattr(amod, 'create_agent', None) if amod is not None else None
        create_react_agent = getattr(amod, 'create_react_agent', None) if amod is not None else None

        # Resolve PromptTemplate from possible langchain package layouts (langchain_core or langchain)
        def _resolve_prompt_template():
            candidates = [
                'langchain_core.prompts.prompt',
                'langchain_core.prompts',
                'langchain.prompts.prompt',
                'langchain.prompts',
            ]
            for modname in candidates:
                try:
                    mod = importlib.import_module(modname)
                except Exception:
                    continue
                pt = getattr(mod, 'PromptTemplate', None)
                if pt is not None:
                    return pt
            return None

        PromptTemplate = _resolve_prompt_template()

        # Instantiate the LLM
        self.llm = instantiate_llm(LMClass, OPENAI_KEY, model)

        # 1) If a modern `create_agent` exists, prefer it. It may return an executor-like object.
        self.agent_executor = None
        if callable(create_agent):
            try:
                if PromptTemplate is not None:
                    self.prompt = PromptTemplate.from_template(CLASSIFICATION_SYSTEM_PROMPT)
                    try:
                        agent_obj = create_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
                    except TypeError:
                        # try positional fallback
                        agent_obj = create_agent(self.llm, self.tools)
                else:
                    try:
                        agent_obj = create_agent(llm=self.llm, tools=self.tools)
                    except TypeError:
                        agent_obj = create_agent(self.llm, self.tools)

                # If the returned object is already an executor (has run/invoke), use it directly
                if hasattr(agent_obj, 'run') or hasattr(agent_obj, 'invoke'):
                    self.agent_executor = agent_obj
                elif AgentExecutor is not None:
                    # Try wrapping the returned agent into an executor
                    try:
                        self.agent_executor = AgentExecutor(agent=agent_obj, tools=self.tools, verbose=True)
                    except Exception:
                        self.agent_executor = None
                else:
                    self.agent_executor = None
            except Exception:
                self.agent_executor = None

        # 2) Fallback: older create_react_agent + AgentExecutor
        if self.agent_executor is None and create_react_agent is not None and AgentExecutor is not None and PromptTemplate is not None:
            try:
                self.prompt = PromptTemplate.from_template(CLASSIFICATION_SYSTEM_PROMPT)

                self.agent = create_react_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=self.prompt,
                )

                self.agent_executor = AgentExecutor(
                    agent=self.agent,
                    tools=self.tools,
                    verbose=True,
                    max_iterations=10,
                    handle_parsing_errors=True,
                    return_intermediate_steps=False,
                )
            except Exception:
                self.agent_executor = None

        # 3) Final fallback: simple executor that calls the LLM directly
        if self.agent_executor is None:
            class SimpleLLMExecutor:
                def __init__(self, llm):
                    self.llm = llm

                def run(self, prompt: str) -> str:
                    # Try call conventions on the llm
                    try:
                        return self.llm(prompt)
                    except Exception:
                        pass
                    try:
                        if hasattr(self.llm, 'generate'):
                            gen = self.llm.generate({'input': prompt})
                            return str(gen)
                    except Exception:
                        pass
                    try:
                        return self.llm.__call__(prompt)
                    except Exception:
                        pass

                    raise RuntimeError('LLM instance is not usable with known call patterns')

                def invoke(self, kwargs: dict) -> dict:
                    inp = kwargs.get('input') or kwargs.get('prompt') or ''
                    return {'output': self.run(inp)}

                def __call__(self, prompt: str) -> str:
                    return self.run(prompt)

            self.agent_executor = SimpleLLMExecutor(self.llm)

    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """
        Classify an AI system.

        Args:
            request: Classification request with AI system description

        Returns:
            ClassificationResponse with risk level and reasoning
        """
        # Build input for the agent
        input_text = f"AI System Description: {request.ai_system_description}"

        if request.additional_info:
            input_text += f"\n\nAdditional Information: {json.dumps(request.additional_info, indent=2)}"

        # Run the agent
        try:
            # Prepend the system prompt so the agent follows the required instruction format
            full_input = f"{CLASSIFICATION_SYSTEM_PROMPT}\n\nQuestion: {input_text}"

            # AgentExecutor may expose different interfaces across versions. Try the common ones.
            output_text = None
            try:
                output_text = self.agent_executor.run(full_input)
            except Exception:
                try:
                    result = self.agent_executor.invoke({"input": input_text})
                    output_text = result.get("output", "")
                except Exception:
                    try:
                        output_text = self.agent_executor(full_input)
                    except Exception as e:
                        raise e

            if not output_text:
                raise RuntimeError("Agent returned no output")

            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                response_data = json.loads(json_str)
                return ClassificationResponse(**response_data)
            else:
                return ClassificationResponse(
                    risk_level=None,
                    reasoning="Failed to parse agent response. The agent did not return valid JSON.",
                    needs_more_info=True,
                    questions=["Could you provide more details about your AI system?"],
                    relevant_articles=[]
                )

        except Exception as e:
            return ClassificationResponse(
                risk_level=None,
                reasoning=f"Error during classification: {str(e)}",
                needs_more_info=True,
                questions=["Could you provide more details about your AI system?"],
                relevant_articles=[]
            )
