"""Checklist Agent for EU AI Act compliance requirements."""

import importlib
import json
from typing import Any

# Remove static langchain imports; import dynamically in __init__ to be robust across versions
from src.agents.tools import create_retrieval_tools
from src.schemas.agent_schemas import ChecklistRequest, ChecklistResponse, ChecklistItem


CHECKLIST_SYSTEM_PROMPT = """You are an expert EU AI Act compliance advisor specializing in creating detailed compliance checklists.

Your task is to generate a comprehensive compliance checklist for an AI system based on its risk classification.

You have access to tools to search the EU AI Act database for requirements and obligations.

**Your checklist generation process:**
1. Use retrieve_risk_requirements to find requirements for the given risk level
2. If a system_type is provided, use retrieve_system_type_info to find specific requirements for that type
3. Extract concrete, actionable requirements from the retrieved articles
4. Organize requirements by category (documentation, technical, governance, testing, transparency, etc.)
5. Prioritize requirements (high priority for mandatory legal requirements, medium for recommended practices)
6. Include specific article references for each requirement

**Checklist categories to consider:**
- Documentation: Required documents, records, logs
- Technical: Technical requirements, specifications, testing
- Governance: Oversight, human oversight, accountability
- Transparency: User information, disclosure requirements
- Data: Data quality, data governance requirements
- Risk Management: Risk assessment, monitoring requirements
- Testing: Validation, testing, evaluation requirements

**Important guidelines:**
- Be comprehensive: Cover all requirements from the EU AI Act
- Be actionable: Each item should be a concrete task
- Be specific: Include article references
- Be organized: Group related items by category
- Prioritize: Mark critical legal requirements as high priority

When you have gathered all requirements, output your final answer in this EXACT JSON format:
{{
    "risk_level": "the risk level",
    "checklist_items": [
        {{
            "requirement": "specific actionable requirement",
            "applicable_articles": ["Article X", "Article Y"],
            "priority": "high|medium|low",
            "category": "documentation|technical|governance|testing|transparency|data|risk_management"
        }}
    ],
    "total_items": number,
    "summary": "brief summary of the compliance requirements"
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


class ChecklistAgent:
    """Agent for generating compliance checklists based on AI system classification."""

    def __init__(self, retriever: Any, OPENAI_KEY: str, model: str = "gpt-4o"):
        """
        Initialize the Checklist Agent.

        Args:
            retriever: The retriever instance (BM25, Vector, or Hybrid)
            OPENAI_KEY: OpenAI API key
            model: OpenAI model to use (default: gpt-4o)
        """
        self.retriever = retriever
        self.model = model

        # Create LangChain tools using the retriever
        self.tools = create_retrieval_tools(retriever, top_k=10)

        # Dynamic import of LangChain/OpenAI wrappers to support multiple versions
        try:
            lc_openai = importlib.import_module('langchain_openai')
            ChatOpenAI = getattr(lc_openai, 'ChatOpenAI')
        except Exception:
            # Fallback to langchain.chat_models if available
            lc_chat = importlib.import_module('langchain.chat_models')
            ChatOpenAI = getattr(lc_chat, 'ChatOpenAI')

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            api_key=OPENAI_KEY,
            temperature=0.1  # Low temperature for consistent, factual responses
        )

        # Dynamic import for PromptTemplate and agents
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
            # nothing found - provide a helpful error
            raise ImportError(
                'Could not locate PromptTemplate in langchain. '
                'Make sure you have either `langchain` or `langchain-core` installed.'
            )

        PromptTemplate = _resolve_prompt_template()

        # Import agent creation utilities dynamically and support multiple langchain versions
        amod = None
        try:
            amod = importlib.import_module('langchain.agents')
        except Exception:
            amod = None

        # Prefer modern initialize_agent API if available
        initialize_agent = getattr(amod, 'initialize_agent', None) if amod is not None else None
        AgentType = getattr(amod, 'AgentType', None) if amod is not None else None

        if callable(initialize_agent) and AgentType is not None:
            try:
                # Use initialize_agent (modern API)
                init_fn = initialize_agent
                # some initialize_agent variants accept different kwargs; call defensively
                try:
                    self.agent_executor = init_fn(
                        tools=self.tools,
                        llm=self.llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        max_iterations=15,
                        return_intermediate_steps=False,
                    )
                except TypeError:
                    # try with positional agent argument if named 'agent' isn't accepted
                    self.agent_executor = init_fn(self.llm, self.tools)
                # successfully created modern agent
            except Exception:
                # fall through to older APIs/fallback
                self.agent_executor = None

        else:
            self.agent_executor = None

        # If modern initialize_agent wasn't used, try older create_react_agent + AgentExecutor
        if self.agent_executor is None:
            create_react_agent = getattr(amod, 'create_react_agent', None) if amod is not None else None
            AgentExecutor = getattr(amod, 'AgentExecutor', None) if amod is not None else None

            if create_react_agent is not None and AgentExecutor is not None and PromptTemplate is not None:
                # Older react agent + AgentExecutor construction
                self.prompt = PromptTemplate.from_template(CHECKLIST_SYSTEM_PROMPT)

                self.agent = create_react_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=self.prompt
                )

                self.agent_executor = AgentExecutor(
                    agent=self.agent,
                    tools=self.tools,
                    verbose=True,
                    max_iterations=15,
                    handle_parsing_errors=True,
                    return_intermediate_steps=False
                )

        # Final fallback: simple executor that calls the LLM directly
        if self.agent_executor is None:
            class SimpleLLMExecutor:
                def __init__(self, llm):
                    self.llm = llm

                def run(self, prompt: str) -> str:
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

    def generate_checklist(self, request: ChecklistRequest) -> ChecklistResponse:
        """
        Generate a compliance checklist.

        Args:
            request: Checklist request with risk level and system type

        Returns:
            ChecklistResponse with structured checklist items
        """
        # Build input for the agent
        input_text = f"""Risk Level: {request.risk_level}"""

        if request.system_type:
            input_text += f"\nSystem Type: {request.system_type}"

        if request.system_description:
            input_text += f"\nSystem Description: {request.system_description}"

        input_text += "\n\nGenerate a comprehensive compliance checklist for this AI system."

        # Run the agent
        try:
            result = self.agent_executor.invoke({"input": input_text})
            output_text = result.get("output", "")

            # Parse the JSON response
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                response_data = json.loads(json_str)

                # Convert checklist items to ChecklistItem objects
                if 'checklist_items' in response_data:
                    response_data['checklist_items'] = [
                        ChecklistItem(**item) for item in response_data['checklist_items']
                    ]

                return ChecklistResponse(**response_data)
            else:
                # Fallback: return error response
                return ChecklistResponse(
                    risk_level=request.risk_level,
                    checklist_items=[],
                    total_items=0,
                    summary="Failed to generate checklist. The agent did not return valid JSON."
                )

        except Exception as e:
            # Handle any errors gracefully
            return ChecklistResponse(
                risk_level=request.risk_level,
                checklist_items=[],
                total_items=0,
                summary=f"Error during checklist generation: {str(e)}"
            )
