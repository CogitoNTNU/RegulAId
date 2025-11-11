"""Checklist Agent for EU AI Act compliance requirements."""

import json
from typing import Any, AsyncGenerator, cast
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
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
            api_key=lambda: openai_api_key,
            temperature=0.1,  # Low temperature for consistent, factual responses
            streaming=True  # Enable token streaming
        )

        # Create agent using LangChain v1 create_agent API
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=CHECKLIST_SYSTEM_PROMPT,
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
            result = self.agent.invoke(cast(Any, {"messages": [{"role": "user", "content": input_text}]}))
            output_text = result.get("messages", [])[-1].content if result.get("messages") else ""

            # DEBUG: Print intermediate steps
            try:
                messages = result.get("messages", [])
                tool_msgs = [m for m in messages if
                             getattr(m, "type", None) == "tool" or getattr(m, "role", None) == "tool"]
                print(f"\n{'=' * 80}")
                print(f"CHECKLIST AGENT - INTERMEDIATE TOOL MESSAGES ({len(tool_msgs)}):")
                print(f"{'=' * 80}")
                for i, m in enumerate(tool_msgs, 1):
                    print(f"\nTool Message {i}: {getattr(m, 'content', str(m))[:200]}")
                print(f"{'=' * 80}\n")
            except Exception:
                pass

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
            Dict with streaming updates (tasks, progress, sources, final result)
        """
        # Build input for the agent
        input_text = f"""Risk Level: {request.risk_level}"""

        if request.system_type:
            input_text += f"\nSystem Type: {request.system_type}"

        if request.system_description:
            input_text += f"\nSystem Description: {request.system_description}"

        input_text += "\n\nGenerate a comprehensive compliance checklist for this AI system."

        # Track sources from tool calls
        sources = []

        try:
            # Use stream_mode="values" to get the full state at each step
            # This allows us to see tool calls and their outputs
            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": input_text}]},
                stream_mode="values",
            ):
                messages = step.get("messages", [])
                if not messages:
                    continue

                latest_message = messages[-1]
                message_type = getattr(latest_message, "type", None)

                # Check for AI message with tool calls
                if message_type == "ai" or hasattr(latest_message, "tool_calls"):
                    tool_calls = getattr(latest_message, "tool_calls", [])
                    if tool_calls:
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            tool_args = tool_call.get("args", {})

                            # Map tool names to user-friendly descriptions
                            tool_descriptions = {
                                "retrieve_eu_ai_act": "Searching EU AI Act database",
                                "retrieve_risk_requirements": "Analyzing risk requirements",
                                "retrieve_system_type_info": "Searching system-specific information"
                            }

                            description = tool_descriptions.get(tool_name, f"Running {tool_name}")
                            query = tool_args.get("query", tool_args.get("risk_level", tool_args.get("system_type", "")))

                            yield {
                                "type": "task",
                                "status": "in_progress",
                                "title": description,
                                "description": f"Query: {query[:100]}" if query else "Processing...",
                                "tool": tool_name
                            }

                # Check for tool message (tool output)
                elif message_type == "tool":
                    tool_content = getattr(latest_message, "content", "")

                    # Extract sources from tool output
                    # Format: [1] Article X - Name\nContent\n\n
                    if tool_content and "[" in tool_content:
                        # Parse sources from the formatted output
                        import re
                        article_pattern = r'\[(\d+)\]\s*Article\s*([^\n-]+)(?:\s*-\s*([^\n(]+))?(?:\s*\([^)]+\))?\n([^\[]*)'
                        matches = re.findall(article_pattern, tool_content)

                        for match in matches:
                            idx, article_num, article_name, content = match
                            source = {
                                "index": len(sources) + 1,
                                "article_number": article_num.strip(),
                                "article_name": article_name.strip() if article_name else "",
                                "content": content.strip()[:500],  # Limit content length
                                "metadata": {
                                    "article_number": article_num.strip(),
                                    "article_name": article_name.strip() if article_name else ""
                                }
                            }
                            sources.append(source)

                        # Count articles retrieved
                        article_count = len(matches)
                        yield {
                            "type": "task",
                            "status": "completed",
                            "title": "Search completed",
                            "description": f"Retrieved {article_count} relevant articles"
                        }

                # Stream AI response tokens
                if message_type == "ai":
                    content = getattr(latest_message, "content", "")
                    if content:
                        # Only stream if there are no tool calls (final response)
                        tool_calls = getattr(latest_message, "tool_calls", [])
                        if not tool_calls:
                            # Stream token by token
                            for char in content:
                                yield {"type": "token", "token": char}

            # Get final result from the last message
            final_result = self.agent.invoke({"messages": [{"role": "user", "content": input_text}]})
            output_text = final_result.get("messages", [])[-1].content if final_result.get("messages") else ""

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

                # Add sources to response
                response_data["sources"] = sources

                # Yield final result
                yield {
                    "type": "final_result",
                    "data": response_data,
                    "sources": sources
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
