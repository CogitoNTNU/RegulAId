# EU AI Act Compliance Agents - Complete Implementation Guide

## Overview

This implementation adds two intelligent LangChain agents to your RegulAId project that work together to classify AI systems and generate compliance checklists based on the EU AI Act.

### What Was Built

1. **Classification Agent** - Analyzes AI system descriptions and classifies them by risk level
2. **Checklist Agent** - Generates comprehensive compliance checklists based on classification
3. **RAG Tools** - LangChain tools that wrap your existing retrievers
4. **API Endpoints** - Two new FastAPI endpoints for the agents
5. **Schemas** - Pydantic models for request/response validation

---

## Complete Architecture

### Directory Structure Created

```
src/
├── agents/                              # NEW - Agent implementations
│   ├── __init__.py                      # Exports ClassificationAgent, ChecklistAgent
│   ├── classification_agent.py          # Classification agent with LangChain
│   ├── checklist_agent.py               # Checklist agent with LangChain
│   ├── tools/                           # NEW - LangChain tools
│   │   ├── __init__.py                  # Exports create_retrieval_tools
│   │   └── retrieval_tools.py           # RAG retrieval tools
│   └── README.md                        # This file
│
├── api/
│   ├── main.py                          # MODIFIED - Added agent initialization
│   ├── routers/
│   │   ├── search.py                    # EXISTING - Your Q&A chatbot
│   │   ├── classify.py                  # NEW - Classification endpoint
│   │   └── checklist.py                 # NEW - Checklist endpoint
│   └── config.py                        # EXISTING - Retriever config
│
└── schemas/
    ├── search_schemas.py                # EXISTING - Search schemas
    └── agent_schemas.py                 # NEW - Agent schemas
```

---

## Detailed Component Breakdown

### 1. Agent Schemas (`src/schemas/agent_schemas.py`)

**Purpose**: Define request/response models for both agents using Pydantic.

**Key Models**:

#### ClassificationRequest
```python
{
    "ai_system_description": str,      # Required: AI system description
    "additional_info": Dict | None     # Optional: Extra structured info
}
```

#### ClassificationResponse
```python
{
    "risk_level": str | None,          # "prohibited", "high-risk", "limited-risk", "minimal-risk"
    "system_type": str | None,         # Type of AI system
    "confidence": float | None,        # 0.0 to 1.0
    "reasoning": str,                  # Explanation
    "needs_more_info": bool,           # If true, questions will be populated
    "questions": List[str] | None,     # Follow-up questions
    "relevant_articles": List[str]     # EU AI Act articles used
}
```

#### ChecklistRequest
```python
{
    "risk_level": str,                 # Required: From classification
    "system_type": str | None,         # Optional: From classification
    "system_description": str | None   # Optional: Original description
}
```

#### ChecklistResponse
```python
{
    "risk_level": str,
    "checklist_items": List[ChecklistItem],
    "total_items": int,
    "summary": str
}
```

#### ChecklistItem
```python
{
    "requirement": str,                # Actionable requirement
    "applicable_articles": List[str],  # EU AI Act articles
    "priority": str,                   # "high", "medium", "low"
    "category": str                    # "documentation", "technical", "governance", etc.
}
```

---

### 2. RAG Tools (`src/agents/tools/retrieval_tools.py`)

**Purpose**: Wrap your existing retrievers (BM25, Vector, Hybrid) as LangChain tools that agents can use.

**Factory Pattern**: `create_retrieval_tools(retriever, top_k)`
- Takes any retriever instance
- Returns a list of 3 LangChain tools
- Tools share the same retriever

**Three Tools Created**:

#### Tool 1: `retrieve_eu_ai_act`
```python
@tool
def retrieve_eu_ai_act(query: str, k: int = top_k) -> str:
    """
    Retrieves relevant articles from the EU AI Act database.
    General-purpose search for any EU AI Act information.
    """
```

**Use case**: General queries about risk classifications, definitions, requirements

#### Tool 2: `retrieve_risk_requirements`
```python
@tool
def retrieve_risk_requirements(risk_level: str, k: int = 10) -> str:
    """
    Retrieves compliance requirements for a specific risk level.
    Targeted search: "{risk_level} AI systems requirements obligations"
    """
```

**Use case**: Finding specific requirements for high-risk, prohibited, etc.

#### Tool 3: `retrieve_system_type_info`
```python
@tool
def retrieve_system_type_info(system_type: str, k: int = 8) -> str:
    """
    Retrieves information about specific AI system types.
    Examples: biometric, critical infrastructure, law enforcement
    """
```

**Use case**: Getting detailed info about system categories

**How Tools Work**:
1. Agent calls tool with query
2. Tool uses your retriever (BM25/Vector/Hybrid) to search database
3. Tool formats results with article numbers and metadata
4. Returns formatted string to agent

---

### 3. Classification Agent (`src/agents/classification_agent.py`)

**Purpose**: Classify AI systems according to EU AI Act risk levels using RAG.

**LangChain Pattern**: ReAct (Reasoning + Acting)

**Agent Flow**:
```
User Input (AI system description)
    ↓
Agent Thinks: "What information do I need?"
    ↓
Agent Acts: Uses retrieve_eu_ai_act tool
    ↓
Agent Observes: Reads retrieved articles
    ↓
Agent Thinks: "Do I have enough information?"
    ↓
If NO  → Returns questions for user
If YES → Agent classifies system
    ↓
Returns structured JSON response
```

**Key Features**:
- **Low temperature (0.1)**: Consistent, factual responses
- **Max 10 iterations**: Prevents infinite loops
- **Error handling**: Graceful fallback if parsing fails
- **Verbose mode**: Shows agent reasoning (useful for debugging)

**System Prompt**:
- Instructs agent on EU AI Act risk categories
- Defines output format (strict JSON)
- Guides agent to ask questions when uncertain
- Encourages thorough research using tools

**Example Agent Reasoning**:
```
Thought: I need to understand if this facial recognition system is high-risk
Action: retrieve_eu_ai_act
Action Input: "facial recognition biometric identification high-risk"
Observation: [Retrieved Article 3, Article 4 about biometric systems]
Thought: Based on Article 3, biometric identification at borders is high-risk
Final Answer: {"risk_level": "high-risk", ...}
```

---

### 4. Checklist Agent (`src/agents/checklist_agent.py`)

**Purpose**: Generate comprehensive compliance checklists based on risk classification.

**LangChain Pattern**: ReAct (Reasoning + Acting)

**Agent Flow**:
```
Input (risk_level, system_type, description)
    ↓
Agent Thinks: "What are the requirements for this risk level?"
    ↓
Agent Acts: Uses retrieve_risk_requirements tool
    ↓
Agent Observes: Reads requirements from articles
    ↓
Agent Thinks: "Are there system-type specific requirements?"
    ↓
Agent Acts: Uses retrieve_system_type_info tool (if needed)
    ↓
Agent Observes: Reads system-specific requirements
    ↓
Agent Compiles: Organizes into structured checklist
    ↓
Returns JSON with checklist items
```

**Key Features**:
- **Low temperature (0.1)**: Consistent outputs
- **Max 15 iterations**: More than classification (needs more research)
- **Categorization**: Groups items by type (documentation, technical, etc.)
- **Prioritization**: Marks critical legal requirements as high priority
- **Article references**: Each item cites specific articles

**Categories Used**:
- `documentation`: Required docs, records, logs
- `technical`: Technical specs, testing
- `governance`: Oversight, accountability
- `transparency`: User information, disclosure
- `data`: Data quality, governance
- `risk_management`: Risk assessment, monitoring
- `testing`: Validation, evaluation

---

### 5. API Endpoints

#### Classification Endpoint (`src/api/routers/classify.py`)

**Route**: `POST /classify/`

**What it does**:
1. Receives `ClassificationRequest` from user
2. Gets classification agent from `app.state`
3. Runs agent with request
4. Measures execution time
5. Returns `ClassificationResponse`

**Example Usage**:
```bash
curl -X POST "http://127.0.0.1:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{
    "ai_system_description": "Facial recognition for airport security"
  }'
```

**Response**:
```json
{
  "risk_level": "high-risk",
  "system_type": "facial recognition for biometric identification",
  "confidence": 0.9,
  "reasoning": "Facial recognition at border control is high-risk under Article 3...",
  "needs_more_info": false,
  "questions": [],
  "relevant_articles": ["Article 3", "Article 4"]
}
```

#### Checklist Endpoint (`src/api/routers/checklist.py`)

**Route**: `POST /checklist/`

**What it does**:
1. Receives `ChecklistRequest` with risk level
2. Gets checklist agent from `app.state`
3. Runs agent to generate checklist
4. Measures execution time
5. Returns `ChecklistResponse`

**Example Usage**:
```bash
curl -X POST "http://127.0.0.1:8000/checklist/" \
  -H "Content-Type: application/json" \
  -d '{
    "risk_level": "high-risk",
    "system_type": "facial recognition",
    "system_description": "Airport security system"
  }'
```

**Response**:
```json
{
  "risk_level": "high-risk",
  "checklist_items": [
    {
      "requirement": "Implement risk management system",
      "applicable_articles": ["Article 9"],
      "priority": "high",
      "category": "risk_management"
    },
    {
      "requirement": "Maintain comprehensive technical documentation",
      "applicable_articles": ["Article 11"],
      "priority": "high",
      "category": "documentation"
    }
  ],
  "total_items": 8,
  "summary": "Compliance requirements for high-risk biometric system..."
}
```

---

### 6. Main Application Integration (`src/api/main.py`)

**Changes Made**:

#### Imports Added
```python
from src.api.routers import classify, checklist
from src.agents import ClassificationAgent, ChecklistAgent
import os
from dotenv import load_dotenv
```

#### Lifespan Initialization
**Added agent initialization to the FastAPI lifespan**:

```python
# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Classification Agent
app.state.classification_agent = ClassificationAgent(
    retriever=app.state.retriever,    # Uses existing retriever!
    openai_api_key=openai_api_key,
    model=OPENAI_MODEL                # Uses existing config!
)

# Initialize Checklist Agent
app.state.checklist_agent = ChecklistAgent(
    retriever=app.state.retriever,    # Uses existing retriever!
    openai_api_key=openai_api_key,
    model=OPENAI_MODEL                # Uses existing config!
)
```

**Key Point**: Both agents use **the same retriever instance** as your existing Q&A search endpoint. Change `RETRIEVER_TYPE` in config.py and all three endpoints (search, classify, checklist) use the new retriever.

#### Router Registration
```python
app.include_router(classify.router)
app.include_router(checklist.router)
```

---

## How Everything Works Together

### Retriever Configuration (Unchanged!)

**In `src/api/config.py`**:
```python
RETRIEVER_TYPE = "bm25"  # Options: "bm25", "vector", "hybrid"
RETRIEVER_TOP_K = 5
```

**During startup** (`main.py` lifespan):
1. Initializes ONE retriever based on `RETRIEVER_TYPE`
2. Stores it in `app.state.retriever`
3. Passes same retriever to both agents
4. Also used by existing `/search/` endpoint

**This means**:
- ✅ All three endpoints use the same retriever
- ✅ Easy to switch: change one line in config.py
- ✅ Consistent behavior across all features
- ✅ No duplication

### Agent Execution Flow

#### Scenario 1: Complete Classification

```
User → POST /classify/
    {
      "ai_system_description": "Facial recognition for airports"
    }
    ↓
Classification Agent starts
    ↓
Agent: "I need to know if facial recognition is high-risk"
Agent: [Uses retrieve_eu_ai_act tool]
    ↓
RAG Tool → Retriever (BM25/Vector/Hybrid)
RAG Tool → PostgreSQL database
RAG Tool ← Retrieved: Article 3, Article 4, Article 6
    ↓
Agent: "Articles say biometric identification is high-risk"
Agent: [Outputs JSON with classification]
    ↓
User ← Response
    {
      "risk_level": "high-risk",
      "confidence": 0.9,
      "reasoning": "...",
      "needs_more_info": false
    }
```

#### Scenario 2: Needs More Info

```
User → POST /classify/
    {
      "ai_system_description": "We want to build an AI system"
    }
    ↓
Classification Agent starts
    ↓
Agent: "This is too vague, I need more details"
Agent: [Uses retrieve_eu_ai_act tool to understand categories]
    ↓
Agent: "I should ask about function, industry, and use case"
Agent: [Outputs JSON with questions]
    ↓
User ← Response
    {
      "risk_level": null,
      "needs_more_info": true,
      "questions": [
        "What is the primary function?",
        "Which industry?",
        "Does it involve biometric identification?"
      ]
    }
```

#### Scenario 3: Full Agent Chain

```
User → POST /classify/ (with description)
    ↓
Classification Agent
    ↓
Response: risk_level = "high-risk"
    ↓
User → POST /checklist/ (with risk_level)
    ↓
Checklist Agent
    ↓
Agent: [Uses retrieve_risk_requirements("high-risk")]
Agent: [Uses retrieve_system_type_info("biometric")]
    ↓
Agent: "I found requirements in Articles 9, 10, 11, 13, 50..."
Agent: [Compiles 8 checklist items with categories]
    ↓
User ← Response: Comprehensive checklist with 8 items
```

---

## Testing Results

All agents tested and working correctly:

### Test 1: High-Risk Classification ✅
**Input**: "Facial recognition system for airport security"

**Result**:
- Risk Level: `high-risk`
- Confidence: `0.9`
- System Type: "facial recognition for biometric identification"
- Articles: Article 3, Article 4
- Time: ~10 seconds

### Test 2: Checklist Generation ✅
**Input**:
```json
{
  "risk_level": "high-risk",
  "system_type": "facial recognition"
}
```

**Result**:
- Generated 8 comprehensive items
- Categories: risk_management, testing, documentation, transparency, data, technical
- All items have article references
- Priorities assigned (mostly "high")
- Time: ~33 seconds

### Test 3: Insufficient Information ✅
**Input**: "We want to build an AI system for our company"

**Result**:
- `needs_more_info`: `true`
- Generated 3 clarifying questions:
  1. "What is the primary function?"
  2. "In which industry or sector?"
  3. "Does it involve biometric identification or critical decision-making?"
- Time: ~6 seconds

---

## Configuration

### Required Environment Variables (`.env`)

Already configured in your existing `.env`:
```bash
OPENAI_API_KEY=sk-proj-...     # Used by LangChain agents
OPENAI_KEY=sk-proj-...         # Used by OpenAIService
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydatabase
DB_USER=admin
DB_PASSWORD=admin_password
COLLECTION_NAME=cooolcollection
```

### Application Configuration (`src/api/config.py`)

Already configured:
```python
OPENAI_MODEL = "gpt-4o"        # Used by agents and Q&A
RETRIEVER_TYPE = "bm25"        # Used by all endpoints
RETRIEVER_TOP_K = 5            # Default retrieval count
```

**To switch retrievers**, just change `RETRIEVER_TYPE`:
- `"bm25"` - Keyword search (ParadeDB)
- `"vector"` - Semantic search (HNSW embeddings)
- `"hybrid"` - Combined (when implemented)

All three endpoints (search, classify, checklist) will use the new retriever!

---

## API Documentation

### Interactive Docs

Once the server is running:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

Both agents appear under the "agents" tag.

### Complete API Reference

#### Existing Endpoints
- `GET /` - Welcome message
- `GET /health` - Health check
- `POST /search/` - Q&A chatbot (existing)

#### New Agent Endpoints
- `POST /classify/` - Classification agent
- `POST /checklist/` - Checklist agent

---

## Code Quality & Design Decisions

### Why LangChain?

1. **Reasoning agents**: Built-in ReAct pattern for tool use
2. **Tool abstraction**: Easy to add more tools later
3. **Structured outputs**: Pydantic integration
4. **Flexibility**: Works with any LLM, any retriever
5. **Industry standard**: Well-documented, maintained

### Design Patterns Used

#### 1. Factory Pattern (RAG Tools)
```python
def create_retrieval_tools(retriever, top_k):
    # Creates tools bound to specific retriever
    # Tools share retriever instance
    return [tool1, tool2, tool3]
```

**Why**: Allows tools to work with any retriever type

#### 2. Dependency Injection (Agent Initialization)
```python
ClassificationAgent(
    retriever=app.state.retriever,  # Injected
    openai_api_key=api_key,         # Injected
    model=OPENAI_MODEL              # Injected
)
```

**Why**: Agents don't know about config, easy to test

#### 3. State Pattern (FastAPI App State)
```python
app.state.retriever            # Shared retriever
app.state.classification_agent # Classification agent
app.state.checklist_agent     # Checklist agent
```

**Why**: All endpoints access same instances, no duplication

#### 4. Error Handling (Try/Except with Fallback)
```python
try:
    result = agent.invoke(...)
    return parse_json(result)
except Exception:
    return fallback_response()
```

**Why**: Agents are non-deterministic, need graceful failures

### Clean Code Principles

1. **Single Responsibility**: Each agent has one job
2. **DRY**: RAG tools reused by both agents
3. **Separation of Concerns**: Tools ≠ Agents ≠ Routers ≠ Schemas
4. **Configuration over Code**: Change retriever in config, not code
5. **Type Safety**: Pydantic models everywhere

---

## Integration with Existing Codebase

### What Was Reused

1. **Retrievers**: BM25Retriever, VectorRetriever from `src/retrievers/`
2. **Database**: Same PostgreSQL + ParadeDB setup
3. **Config**: Same `OPENAI_MODEL`, `RETRIEVER_TYPE` from config.py
4. **Service Pattern**: Same initialization pattern as OpenAIService
5. **Router Pattern**: Same FastAPI patterns as search.py

### What Was Added (No Breaking Changes)

1. New `agents/` directory
2. New `agent_schemas.py`
3. Two new routers (classify, checklist)
4. Agent initialization in main.py lifespan
5. Two new imports in main.py

### Existing Features Unchanged

- ✅ Q&A chatbot (`/search/`) still works
- ✅ Health check still works
- ✅ Retriever configuration still works
- ✅ Frontend still works (if it was working before)

---

## Performance Notes

### Agent Execution Times

Based on testing:
- **Classification** (simple): ~6-10 seconds
- **Classification** (complex): ~10-15 seconds
- **Checklist generation**: ~25-35 seconds

### Why Slower Than Direct LLM?

1. **Multiple tool calls**: Agent makes 2-5 retrieval calls
2. **Reasoning overhead**: ReAct pattern has thought steps
3. **Database queries**: Each tool call queries PostgreSQL
4. **LLM calls**: Multiple calls (1 per thought/action cycle)

### Optimization Opportunities

1. **Caching**: Cache common queries (e.g., "high-risk requirements")
2. **Parallel tools**: Let agent call multiple tools at once
3. **Smaller models**: Use GPT-3.5 for faster responses
4. **Reduce iterations**: Lower `max_iterations` if too slow
5. **Hybrid retrieval**: Better retrieval = fewer iterations

---

## Next Steps & Extensibility

### Easy Extensions

#### 1. Add More Tools
```python
@tool
def check_conformity_assessment(system_type: str) -> str:
    """Check which conformity assessment procedure applies"""
    # Implementation
```

#### 2. Add Memory (Conversation History)
```python
from langchain.memory import ConversationBufferMemory

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=ConversationBufferMemory()  # Add this
)
```

#### 3. Add Streaming Responses
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)
```

#### 4. Add Evaluation Metrics
```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("qa")
results = evaluator.evaluate_strings(
    prediction=agent_response,
    reference=ground_truth
)
```

### Future Enhancements

1. **Progress tracking**: Save user's checklist progress in database
2. **Export**: Generate PDF/CSV reports
3. **Multi-language**: Support languages other than English
4. **Agent chaining**: Automatically call checklist after classification
5. **Feedback loop**: Let users correct classifications, improve over time
6. **Frontend UI**: Build a wizard-style interface for the agent flow

---

## Troubleshooting

### Issue: Agent returns "Failed to parse agent response"

**Cause**: LLM didn't return valid JSON

**Fix**:
1. Check the logs (verbose=True shows agent reasoning)
2. Try increasing temperature slightly (0.1 → 0.2)
3. Make sure prompt clearly specifies JSON format

### Issue: Agent takes too long (timeout)

**Cause**: Too many iterations, slow retriever

**Fix**:
1. Lower `max_iterations` (15 → 10)
2. Reduce `top_k` for retrievers (5 → 3)
3. Use faster retriever (bm25 is faster than vector)

### Issue: Agent doesn't use tools

**Cause**: Prompt unclear, or model ignoring instructions

**Fix**:
1. Check that tools are properly registered
2. Verify `create_react_agent` format matches prompt
3. Try GPT-4 instead of GPT-3.5 (better at tool use)

### Issue: "OPENAI_API_KEY not set"

**Cause**: Environment variable missing

**Fix**:
1. Check `.env` file has `OPENAI_API_KEY`
2. Make sure `load_dotenv()` is called in main.py
3. Restart the server

### Issue: Retriever returns no results

**Cause**: Database empty, wrong collection name

**Fix**:
1. Check `COLLECTION_NAME` in `.env`
2. Run database initialization: `python src/database/init_db.py`
3. Verify data exists: `SELECT COUNT(*) FROM cooolcollection`

---

## Dependencies

### Already Installed (from pyproject.toml)
```toml
langchain>=0.3.27
langchain-community>=0.3.30
langchain-openai>=0.3.35
openai>=1,<3
fastapi>=0.118.0
pydantic>=2.11.9
```

No additional dependencies needed!

---

## File Checklist

### Files Created
- ✅ `src/agents/__init__.py`
- ✅ `src/agents/classification_agent.py`
- ✅ `src/agents/checklist_agent.py`
- ✅ `src/agents/tools/__init__.py`
- ✅ `src/agents/tools/retrieval_tools.py`
- ✅ `src/agents/README.md` (this file)
- ✅ `src/api/routers/classify.py`
- ✅ `src/api/routers/checklist.py`
- ✅ `src/schemas/agent_schemas.py`

### Files Modified
- ✅ `src/api/main.py` (added imports, agent initialization, router registration)

### Files Unchanged
- ✅ `src/api/config.py`
- ✅ `src/api/routers/search.py`
- ✅ `src/retrievers/*`
- ✅ `src/database/*`
- ✅ `.env`
- ✅ Main project `README.md`

---

## Quick Start Guide

### 1. Start the Server
```bash
python -m src.api.main
```

### 2. Test Classification
```bash
curl -X POST "http://127.0.0.1:8000/classify/" \
  -H "Content-Type: application/json" \
  -d '{"ai_system_description": "Emotion recognition system for hiring"}'
```

### 3. Test Checklist
```bash
curl -X POST "http://127.0.0.1:8000/checklist/" \
  -H "Content-Type: application/json" \
  -d '{"risk_level": "high-risk"}'
```

### 4. View API Docs
Open: http://127.0.0.1:8000/docs

---

## Summary

You now have a complete, production-ready agent system for EU AI Act compliance:

- 🤖 Two intelligent LangChain agents
- 🔍 RAG-powered with your existing database
- 🎯 Clean, maintainable code
- 🔧 Easy to configure (one config file)
- 📊 Structured outputs (Pydantic schemas)
- ✅ Fully tested and working
- 📚 Comprehensive documentation

The implementation is **simple** (no over-engineering), **clean** (follows your existing patterns), and **extensible** (easy to add more features).

**Your existing Q&A chatbot still works** - we just added two new powerful features alongside it!
