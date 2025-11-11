"""Classification Agent for EU AI Act risk assessment."""

import json
from typing import Any, AsyncGenerator, cast
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from src.agents.tools import create_retrieval_tools
from src.schemas.agent_schemas import ClassificationRequest, ClassificationResponse
from src.agents.callbacks import StreamingCallbackHandler


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

    def __init__(self, retriever: Any, openai_api_key: str, model: str = "gpt-4o"):
        """
        Initialize the Classification Agent.

        Args:
            retriever: The retriever instance (BM25, Vector, or Hybrid)
            openai_api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o)
        """
        self.retriever = retriever
        self.model = model

        # Create LangChain tools using the retriever
        self.tools = create_retrieval_tools(retriever, top_k=5)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            api_key=lambda: openai_api_key,
            temperature=0.1,  # Low temperature for consistent, factual responses
            streaming=True  # Enable token streaming
        )

        # Create agent with a static system prompt
        # `create_agent` expects the model (LLM object), a list of tools and a system_prompt string.
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
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
        input_text = f"""AI System Description: {request.ai_system_description}"""

        if request.additional_info:
            input_text += f"\n\nAdditional Information: {json.dumps(request.additional_info, indent=2)}"

        # Run the agent
        try:
            # The new agent API expects messages as the input shape
            result = self.agent.invoke(cast(Any, {"messages": [{"role": "user", "content": input_text}]}))
            # The agent returns a dict with messages; take the last message content
            output_text = result.get("messages", [])[-1].content if result.get("messages") else ""

            # DEBUG: Print intermediate steps
            # Try to show any tool call / tool outputs contained in the messages for debugging
            try:
                messages = result.get("messages", [])
                tool_msgs = [m for m in messages if getattr(m, "type", None) == "tool" or getattr(m, "role", None) == "tool"]
                print(f"\n{'='*80}")
                print(f"INTERMEDIATE TOOL MESSAGES ({len(tool_msgs)}):")
                print(f"{'='*80}")
                for i, m in enumerate(tool_msgs, 1):
                    print(f"\nTool Message {i}: {getattr(m, 'content', str(m))[:200]}")
                print(f"{'='*80}\n")
            except Exception:
                pass

            # Parse the JSON response
            # The agent should return JSON in the Final Answer
            # Try to extract JSON from the output
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                response_data = json.loads(json_str)
                return ClassificationResponse(**response_data)
            else:
                # Fallback: return error response
                return ClassificationResponse(
                    risk_level=None,
                    reasoning="Failed to parse agent response. The agent did not return valid JSON.",
                    needs_more_info=True,
                    questions=["Could you provide more details about your AI system?"],
                    relevant_articles=[]
                )

        except Exception as e:
            # Handle any errors gracefully
            return ClassificationResponse(
                risk_level=None,
                reasoning=f"Error during classification: {str(e)}",
                needs_more_info=True,
                questions=["Could you provide more details about your AI system?"],
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

        # Create streaming callback handler
        callback_handler = StreamingCallbackHandler()

        try:
            # Stream agent output using the new agent.stream API
            # agent.stream yields chunks; use stream_mode='messages' to get tokens with metadata
            async for token, metadata in cast(Any, self.agent.stream(
                cast(Any, {"messages": [{"role": "user", "content": input_text}]}),
                stream_mode="messages",
            )):
                # Each yielded token is a message-like object; stream token updates to the caller
                try:
                    content_blocks = getattr(token, "content_blocks", None)
                    if content_blocks:
                        # content_blocks may be a list; join or take first
                        text = "".join([b for b in content_blocks]) if isinstance(content_blocks, list) else str(content_blocks)
                    else:
                        text = getattr(token, "content", str(token))
                except Exception:
                    text = str(token)

                yield {"type": "token", "token": text}

            # After stream completes, invoke once to get final structured result (some agents include final message in stream but be safe)
            result = self.agent.invoke(cast(Any, {"messages": [{"role": "user", "content": input_text}]}))
            output_text = result.get("messages", [])[-1].content if result.get("messages") else ""

            # Parse the JSON response
            json_start = output_text.find('{')
            json_end = output_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                response_data = json.loads(json_str)

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
                "message": f"Error during classification: {str(e)}"
            }
