"""Checklist Agent for EU AI Act compliance requirements."""

import json
from typing import Any, List, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from src.agents.tools import create_retrieval_tools
from src.schemas.agent_schemas import ChecklistRequest, ChecklistResponse, ChecklistItem
from src.agents.callbacks import StreamingCallbackHandler


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

    def __init__(self, retriever: Any, openai_api_key: str, model: str = "gpt-4.1"):
        """
        Initialize the Checklist Agent.

        Args:
            retriever: The retriever instance (BM25, Vector, or Hybrid)
            openai_api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4.1)
        """
        self.retriever = retriever
        self.model = model

        # Create LangChain tools using the retriever
        self.tools = create_retrieval_tools(retriever, top_k=10)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            temperature=0.1,  # Low temperature for consistent, factual responses
            streaming=True  # Enable token streaming
        )

        # Create prompt
        self.prompt = PromptTemplate.from_template(CHECKLIST_SYSTEM_PROMPT)

        # Create agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
            return_intermediate_steps=True 
        )

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

            # DEBUG: Print intermediate steps
            intermediate_steps = result.get("intermediate_steps", [])
            print(f"\n{'='*80}")
            print(f"CHECKLIST AGENT - INTERMEDIATE STEPS ({len(intermediate_steps)} steps):")
            print(f"{'='*80}")
            for i, (action, observation) in enumerate(intermediate_steps, 1):
                print(f"\nStep {i}:")
                print(f"  Tool: {action.tool}")
                print(f"  Input: {action.tool_input}")
                print(f"  Output Preview: {observation[:200]}..." if len(observation) > 200 else f"  Output: {observation}")
            print(f"{'='*80}\n")

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

    async def generate_checklist_streaming(
        self,
        request: ChecklistRequest
    ) -> AsyncGenerator[dict, None]:
        """
        Generate a compliance checklist with streaming updates.

        Args:
            request: Checklist request with risk level and system type

        Yields:
            Dict with streaming updates (tasks, progress, final result)
        """
        # Build input for the agent
        input_text = f"""Risk Level: {request.risk_level}"""

        if request.system_type:
            input_text += f"\nSystem Type: {request.system_type}"

        if request.system_description:
            input_text += f"\nSystem Description: {request.system_description}"

        input_text += "\n\nGenerate a comprehensive compliance checklist for this AI system."

        # Create streaming callback handler
        callback_handler = StreamingCallbackHandler()

        try:
            # Start agent execution in background
            import asyncio
            agent_task = asyncio.create_task(
                self.agent_executor.ainvoke(
                    {"input": input_text},
                    {"callbacks": [callback_handler]}
                )
            )

            # Stream updates as they come
            async for update in callback_handler.get_updates():
                yield update

            # Wait for agent to complete
            result = await agent_task
            output_text = result.get("output", "")

            # Parse the JSON response
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                response_data = json.loads(json_str)

                # Convert checklist items to proper format
                if 'checklist_items' in response_data:
                    response_data['checklist_items'] = [
                        item if isinstance(item, dict) else item.__dict__
                        for item in response_data['checklist_items']
                    ]

                # Yield final result
                yield {
                    "type": "final_result",
                    "data": response_data
                }
            else:
                # Yield error
                yield {
                    "type": "error",
                    "message": "Failed to parse agent response"
                }

        except Exception as e:
            # Yield error
            yield {
                "type": "error",
                "message": f"Error generating checklist: {str(e)}"
            }
