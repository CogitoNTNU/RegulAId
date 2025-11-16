"""Classification Agent for EU AI Act risk assessment."""

import json
from typing import Any, AsyncGenerator

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

**Important guidelines:**
- Be thorough: Use the tools to find relevant articles before making a classification

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
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT
        )

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
                try:
                    response_data = json.loads(json_str)
                    return ClassificationResponse(**response_data)
                except json.JSONDecodeError as e:
                    print(f"\n{'='*80}")
                    print(f"JSON PARSING ERROR:")
                    print(f"Error: {str(e)}")
                    print(f"Attempted to parse:")
                    print(json_str[:500])
                    print(f"{'='*80}\n")
                    return ClassificationResponse(
                        risk_level=None,
                        reasoning=f"Failed to parse JSON: {str(e)}",
                        needs_more_info=True,
                        questions=["JSON parsing failed - please try again"],
                        relevant_articles=[]
                    )
            else:
                print(f"\n{'='*80}")
                print(f"NO JSON FOUND IN OUTPUT:")
                print(f"Full output text:")
                print(output_text[:500])
                print(f"{'='*80}\n")
                return ClassificationResponse(
                    risk_level=None,
                    reasoning="Failed to parse agent response. The agent did not return valid JSON.",
                    needs_more_info=True,
                    questions=["No JSON found in response"],
                    relevant_articles=[]
                )

        except Exception as e:
            # Print detailed error for debugging
            import traceback
            print(f"\n{'='*80}")
            print(f"ERROR during classification:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Full traceback:")
            traceback.print_exc()
            print(f"{'='*80}\n")

            return ClassificationResponse(
                risk_level=None,
                reasoning=f"Error during classification: {type(e).__name__}: {str(e)}",
                needs_more_info=True,
                questions=[f"Agent executor failed: {str(e)[:100]}"],
                relevant_articles=[]
            )

    async def classify_streaming(
        self,
        request: ClassificationRequest
    ) -> AsyncGenerator[dict, None]:
        """
        Classify an AI system with streaming updates.

        Args:
            request: Classification request with AI system description

        Yields:
            Dict with streaming updates (tasks, progress, final result)
        """
        # Build input for the agent
        input_text = f"""AI System Description: {request.ai_system_description}"""

        if request.additional_info:
            input_text += f"\n\nAdditional Information: {json.dumps(request.additional_info, indent=2)}"

        try:
            # Send initial task status
            yield {
                "type": "task",
                "status": "in_progress",
                "title": "Analyzing your AI system",
                "description": "Starting classification process"
            }

            # Use LangGraph's astream to get real-time updates
            final_messages = []

            async for event in self.agent_executor.astream(
                {"messages": [("user", input_text)]}
            ):

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
                                "retrieve_risk_requirements": "Analyzing risk requirements",
                                "retrieve_system_type_info": "Searching system-specific information"
                            }

                            yield {
                                "type": "task",
                                "status": "in_progress",
                                "title": tool_descriptions.get(tool_name, f"Using {tool_name}"),
                                "description": "Retrieving relevant articles and regulations"
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
                            "title": "Information retrieved",
                            "description": f"Found {article_count} relevant article(s)" if article_count > 0 else "Retrieved information"
                        }

                # Collect all messages for final parsing
                for node_data in event.values():
                    if "messages" in node_data:
                        final_messages.extend(node_data["messages"])

            # Yield completion
            yield {
                "type": "task",
                "status": "completed",
                "title": "Classification complete",
                "description": "Analysis finished"
            }

            # Parse the final result from the last agent message
            if final_messages:
                last_message = final_messages[-1]
                output_text = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # Parse the JSON response (same logic as classify())
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
            print(f"ERROR in classify_streaming:")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            print(f"{'='*80}\n")

            yield {
                "type": "error",
                "message": f"Error during classification: {str(e)}"
            }
