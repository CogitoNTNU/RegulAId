"""Checklist Agent for EU AI Act compliance requirements."""

import json
import logging
from typing import Any, List, AsyncGenerator

from src.agents.tools import create_retrieval_tools
from src.schemas.agent_schemas import ChecklistRequest, ChecklistResponse, ChecklistItem

logger = logging.getLogger(__name__)


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

        # Initialize LLM with streaming enabled
        self.llm = ChatOpenAI(
            model=model,
            api_key=OPENAI_KEY,
            temperature=0.1,  # Low temperature for consistent, factual responses
            streaming=True  # Enable token streaming
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
            current_tool_call = None
            accumulated_content = ""
            emitted_tool_calls = set()  # Track emitted tool calls to avoid duplicates

            async for event in self.agent_executor.astream(
                {"messages": [("user", input_text)]}
            ):

                # Check what type of event this is
                if "agent" in event:
                    # Agent is thinking or responding
                    agent_messages = event["agent"].get("messages", [])
                    if agent_messages:
                        last_message = agent_messages[-1]

                        # Check if this is a tool call - try multiple ways to access tool_calls
                        tool_calls = None
                        try:
                            # Try direct attribute access
                            if hasattr(last_message, 'tool_calls'):
                                tool_calls = last_message.tool_calls
                            
                            # Try get method
                            if not tool_calls and hasattr(last_message, 'get') and callable(last_message.get):
                                tool_calls = last_message.get('tool_calls')
                            
                            # Try dict access
                            if not tool_calls and isinstance(last_message, dict):
                                tool_calls = last_message.get('tool_calls')
                            
                            # Try accessing via additional_kwargs
                            if not tool_calls and hasattr(last_message, 'additional_kwargs'):
                                additional = last_message.additional_kwargs
                                if isinstance(additional, dict) and 'tool_calls' in additional:
                                    tool_calls = additional['tool_calls']
                            
                            # Check message type
                            message_type = type(last_message).__name__
                            
                            # For AIMessage, tool_calls might be a list property
                            if message_type == 'AIMessage' and not tool_calls:
                                try:
                                    if hasattr(last_message, 'tool_calls'):
                                        val = getattr(last_message, 'tool_calls', None)
                                        if val:
                                            tool_calls = val
                                except Exception:
                                    pass
                            
                        except Exception as e:
                            logger.error(f"Error checking tool_calls: {e}", exc_info=True)

                        if tool_calls and len(tool_calls) > 0:
                            # Handle both dict and object tool calls
                            tool_call = tool_calls[0]
                            
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get('name') or tool_call.get('function', {}).get('name', 'unknown')
                                tool_input = tool_call.get('args') or tool_call.get('function', {}).get('arguments', {})
                                # If arguments is a string, try to parse it as JSON
                                if isinstance(tool_input, str):
                                    try:
                                        tool_input = json.loads(tool_input)
                                    except:
                                        tool_input = {"input": tool_input}
                            else:
                                # Object access
                                tool_name = getattr(tool_call, 'name', None) or getattr(tool_call, 'function', {}).get('name', 'unknown') if hasattr(tool_call, 'function') else 'unknown'
                                tool_input = getattr(tool_call, 'args', None) or getattr(tool_call, 'arguments', {})
                                # If it's a function object, try to get name and arguments
                                if hasattr(tool_call, 'function'):
                                    func = tool_call.function
                                    if hasattr(func, 'name'):
                                        tool_name = func.name
                                    if hasattr(func, 'arguments'):
                                        args = func.arguments
                                        if isinstance(args, str):
                                            try:
                                                tool_input = json.loads(args)
                                            except:
                                                tool_input = {"input": args}
                                        else:
                                            tool_input = args
                            
                            # Create a unique key for this tool call to avoid duplicates
                            tool_key = f"{tool_name}_{hash(str(tool_input))}"
                            
                            # Only emit if we haven't already emitted this tool call
                            if tool_key not in emitted_tool_calls:
                                emitted_tool_calls.add(tool_key)
                                current_tool_call = tool_name
                                
                                # Emit tool_start event
                                yield {
                                    "type": "tool_start",
                                    "tool_name": tool_name,
                                    "tool_input": tool_input
                                }
                                
                                # Also update task with tool call info
                                tool_query = tool_input.get('query', tool_input.get('input', str(tool_input)))
                                if isinstance(tool_query, dict):
                                    tool_query = tool_query.get('query', str(tool_query))
                                yield {
                                    "type": "task",
                                    "status": "in_progress",
                                    "title": "Generating compliance checklist",
                                    "description": f"Using {tool_name} to retrieve requirements",
                                    "items": [f"Calling {tool_name}: {str(tool_query)[:100]}"]
                                }
                        else:
                            # This might be a content chunk from the LLM
                            content = None
                            try:
                                if hasattr(last_message, 'content'):
                                    content = last_message.content
                                elif isinstance(last_message, dict):
                                    content = last_message.get('content', '')
                                
                                # Only stream if it's actual content (not empty or just whitespace)
                                if content and isinstance(content, str) and content.strip() and not content.startswith('Thought:'):
                                    # Stream content chunks
                                    accumulated_content += content
                                    yield {
                                        "type": "content_stream",
                                        "content": content
                                    }
                            except Exception as e:
                                logger.debug(f"Error processing content: {e}")

                elif "tools" in event:
                    # Tool execution completed
                    tool_messages = event["tools"].get("messages", [])
                    if tool_messages:
                        # Get the tool name from the message or use current_tool_call
                        tool_name = current_tool_call or "unknown"
                        tool_input = {}  # Default empty input
                        
                        # Try to get tool name and input from the message
                        if tool_messages:
                            last_tool_msg = tool_messages[-1]
                            if hasattr(last_tool_msg, 'name'):
                                tool_name = last_tool_msg.name
                            elif isinstance(last_tool_msg, dict) and 'name' in last_tool_msg:
                                tool_name = last_tool_msg['name']
                            
                            # Try to get input from the tool message
                            if hasattr(last_tool_msg, 'input'):
                                tool_input = last_tool_msg.input
                            elif isinstance(last_tool_msg, dict) and 'input' in last_tool_msg:
                                tool_input = last_tool_msg['input']
                            # Also check for tool_call_id which might have the original call info
                            if hasattr(last_tool_msg, 'tool_call_id'):
                                # Try to find the original tool call in agent messages
                                for node_data in event.values():
                                    if "messages" in node_data:
                                        for msg in node_data["messages"]:
                                            if hasattr(msg, 'tool_calls'):
                                                for tc in msg.tool_calls:
                                                    tc_id = getattr(tc, 'id', None) or (tc.get('id') if isinstance(tc, dict) else None)
                                                    if tc_id and hasattr(last_tool_msg, 'tool_call_id') and tc_id == last_tool_msg.tool_call_id:
                                                        # Found matching tool call, extract input
                                                        if isinstance(tc, dict):
                                                            tool_input = tc.get('args', {})
                                                        else:
                                                            tool_input = getattr(tc, 'args', {})
                                                        break
                        
                        output = str(tool_messages[-1].content) if tool_messages else ""

                        # Emit tool_complete event with input for frontend
                        yield {
                            "type": "tool_complete",
                            "tool_name": tool_name,
                            "tool_input": tool_input,  # Include input so frontend can create tool if tool_start was missed
                            "tool_output": output  # Show full output, frontend can truncate if needed
                        }
                        current_tool_call = None

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
                    logger.warning("Agent did not return valid JSON response")
                    yield {
                        "type": "error",
                        "message": "Agent did not return valid JSON response"
                    }
            else:
                logger.warning("No response from agent")
                yield {
                    "type": "error",
                    "message": "No response from agent"
                }

        except Exception as e:
            logger.error(f"Error in generate_checklist_streaming: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": f"Error generating checklist: {str(e)}"
            }
