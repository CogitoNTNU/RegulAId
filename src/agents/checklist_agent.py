"""Checklist Agent for EU AI Act compliance requirements."""

import json
from typing import Any, List, AsyncGenerator

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

        # Import LangChain classes - using LangChain 1.0+ create_agent
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            api_key=OPENAI_KEY,
            temperature=0.1  # Low temperature for consistent, factual responses
        )

        # Create agent using LangChain 1.0+ recommended approach
        # create_agent returns a LangGraph runnable
        self.agent_executor = create_agent(
            self.llm,
            self.tools,
            system_prompt=CHECKLIST_SYSTEM_PROMPT
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

        # Run the agent using LangGraph messages format
        try:
            # LangGraph agents use messages format
            result = self.agent_executor.invoke({
                "messages": [("user", input_text)]
            })

            # Extract the output from the messages
            messages = result.get("messages", [])
            if not messages:
                raise RuntimeError("Agent returned no messages")

            # The last message should be the agent's response
            last_message = messages[-1]
            output_text = last_message.content if hasattr(last_message, 'content') else str(last_message)

            # DEBUG: Print messages
            print(f"\n{'='*80}")
            print(f"CHECKLIST AGENT MESSAGES ({len(messages)} messages):")
            print(f"{'='*80}")
            for i, msg in enumerate(messages, 1):
                msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                msg_type = msg.type if hasattr(msg, 'type') else type(msg).__name__
                print(f"\nMessage {i} ({msg_type}):")
                print(f"  Content Preview: {msg_content[:200]}..." if len(msg_content) > 200 else f"  Content: {msg_content}")
            print(f"{'='*80}\n")

            # Parse the JSON response
            # Note: The agent may return double braces {{ }} due to template escaping
            # Replace double braces with single braces
            output_text = output_text.replace('{{', '{').replace('}}', '}')

            # Strip "Final Answer:" prefix if present
            if "Final Answer:" in output_text:
                output_text = output_text.split("Final Answer:", 1)[1].strip()

            # Extract JSON from markdown code blocks if present
            if "```json" in output_text:
                json_block_start = output_text.find("```json") + 7
                json_block_end = output_text.find("```", json_block_start)
                if json_block_end != -1:
                    output_text = output_text[json_block_start:json_block_end].strip()
            elif "```" in output_text:
                # Handle plain code blocks
                json_block_start = output_text.find("```") + 3
                json_block_end = output_text.find("```", json_block_start)
                if json_block_end != -1:
                    output_text = output_text[json_block_start:json_block_end].strip()

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

        try:
            # Send initial task status
            yield {
                "type": "task",
                "status": "in_progress",
                "title": "Generating compliance checklist",
                "description": "Analyzing requirements"
            }

            # Use LangGraph's astream to get real-time updates
            final_messages = []

            async for event in self.agent_executor.astream(
                {"messages": [("user", input_text)]}
            ):
                # DEBUG: Print event structure
                print(f"\n{'='*80}")
                print(f"CHECKLIST STREAM EVENT: {list(event.keys())}")
                print(f"{'='*80}\n")

                # Check what type of event this is
                if "agent" in event:
                    # Agent is thinking or responding
                    agent_messages = event["agent"].get("messages", [])
                    if agent_messages:
                        last_message = agent_messages[-1]
                        # Check if this is a tool call
                        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                            tool_call = last_message.tool_calls[0]
                            tool_name = tool_call.get('name', 'unknown')

                            # Map tool names to user-friendly descriptions
                            tool_descriptions = {
                                "retrieve_eu_ai_act": "Searching EU AI Act database",
                                "retrieve_risk_requirements": "Analyzing risk-specific requirements",
                                "retrieve_system_type_info": "Retrieving system-specific requirements"
                            }

                            yield {
                                "type": "task",
                                "status": "in_progress",
                                "title": tool_descriptions.get(tool_name, f"Using {tool_name}"),
                                "description": "Gathering compliance requirements"
                            }

                elif "tools" in event:
                    # Tool execution completed
                    tool_messages = event["tools"].get("messages", [])
                    if tool_messages:
                        # Count articles retrieved
                        output = str(tool_messages[-1].content) if tool_messages else ""
                        article_count = output.count("Article ") if "Article " in output else 0

                        yield {
                            "type": "task",
                            "status": "completed",
                            "title": "Requirements retrieved",
                            "description": f"Found {article_count} relevant article(s)" if article_count > 0 else "Retrieved compliance information"
                        }

                # Collect all messages for final parsing
                for node_data in event.values():
                    if "messages" in node_data:
                        final_messages.extend(node_data["messages"])

            # Yield completion
            yield {
                "type": "task",
                "status": "completed",
                "title": "Checklist generation complete",
                "description": "Requirements analyzed"
            }

            # Parse the final result from the last agent message
            if final_messages:
                last_message = final_messages[-1]
                output_text = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # DEBUG: Print final output
                print(f"\n{'='*80}")
                print(f"CHECKLIST FINAL OUTPUT FROM AGENT:")
                print(f"Output preview: {output_text[:500]}")
                print(f"{'='*80}\n")

                # Parse the JSON response (same logic as generate_checklist())
                output_text = output_text.replace('{{', '{').replace('}}', '}')

                if "Final Answer:" in output_text:
                    output_text = output_text.split("Final Answer:", 1)[1].strip()

                # Extract JSON from markdown code blocks if present
                if "```json" in output_text:
                    json_block_start = output_text.find("```json") + 7
                    json_block_end = output_text.find("```", json_block_start)
                    if json_block_end != -1:
                        output_text = output_text[json_block_start:json_block_end].strip()
                elif "```" in output_text:
                    json_block_start = output_text.find("```") + 3
                    json_block_end = output_text.find("```", json_block_start)
                    if json_block_end != -1:
                        output_text = output_text[json_block_start:json_block_end].strip()

                json_start = output_text.find('{')
                json_end = output_text.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = output_text[json_start:json_end]
                    try:
                        response_data = json.loads(json_str)

                        # Convert checklist items if needed
                        if 'checklist_items' in response_data:
                            response_data['checklist_items'] = [
                                {
                                    "requirement": item.get("requirement", ""),
                                    "applicable_articles": item.get("applicable_articles", []),
                                    "priority": item.get("priority", "medium"),
                                    "category": item.get("category", "general")
                                }
                                for item in response_data['checklist_items']
                            ]

                        # Yield final result
                        yield {
                            "type": "final_result",
                            "data": response_data
                        }
                    except json.JSONDecodeError as e:
                        yield {
                            "type": "error",
                            "message": f"Failed to parse agent response: {str(e)}"
                        }
                else:
                    yield {
                        "type": "error",
                        "message": "Agent did not return valid JSON response"
                    }
            else:
                yield {
                    "type": "error",
                    "message": "No response from agent"
                }

        except Exception as e:
            # Yield error
            import traceback
            print(f"\n{'='*80}")
            print(f"ERROR in generate_checklist_streaming:")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            print(f"{'='*80}\n")

            yield {
                "type": "error",
                "message": f"Error generating checklist: {str(e)}"
            }
