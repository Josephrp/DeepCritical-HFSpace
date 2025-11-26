# Modal Integration

## Priority: P1 - HIGH VALUE ($2,500 Modal Innovation Award)

---

## What Modal Is For

Modal provides serverless GPU/CPU compute. For DeepCritical:

### Current Use Case (Mario's Code)
- `src/tools/code_execution.py` - Run LLM-generated analysis code in sandboxes
- Scientific computing (pandas, scipy, numpy) in isolated containers

### Potential Additional Use Cases

| Use Case | Benefit | Complexity |
|----------|---------|------------|
| Code Execution Sandbox | Run statistical analysis safely | ✅ Already built |
| LLM Inference | Run local models (no API costs) | Medium |
| Batch Processing | Process many papers in parallel | Medium |
| Embedding Generation | GPU-accelerated embeddings | Low |

---

## Current State

Mario implemented `src/tools/code_execution.py`:

```python
# Already exists - ModalCodeExecutor
executor = get_code_executor()
result = executor.execute("""
import pandas as pd
import numpy as np
# LLM-generated statistical analysis
""")
```

### What's Missing

1. **Not wired into the main pipeline** - The executor exists but isn't used
2. **No Modal tokens configured** - Needs MODAL_TOKEN_ID/MODAL_TOKEN_SECRET
3. **No demo showing it works** - Judges need to see it

---

## Integration Plan

### Step 1: Wire Into Agent Pipeline

Add a `StatisticalAnalysisAgent` that uses Modal:

```python
# src/agents/analysis_agent.py
from src.tools.code_execution import get_code_executor

class AnalysisAgent:
    """Run statistical analysis on evidence using Modal sandbox."""

    async def analyze(self, evidence: list[Evidence], query: str) -> str:
        # 1. LLM generates analysis code
        code = await self._generate_analysis_code(evidence, query)

        # 2. Execute in Modal sandbox
        executor = get_code_executor()
        result = executor.execute(code)

        # 3. Return results
        return result["stdout"]
```

### Step 2: Add to Orchestrator

```python
# In orchestrator, after gathering evidence:
if settings.enable_modal_analysis:
    analysis_agent = AnalysisAgent()
    stats_results = await analysis_agent.analyze(evidence, query)
```

### Step 3: Create Demo

```python
# examples/modal_demo/run_analysis.py
"""Demo: Modal-powered statistical analysis of drug evidence."""

# Show:
# 1. Gather evidence from PubMed
# 2. Generate analysis code with LLM
# 3. Execute in Modal sandbox
# 4. Return statistical insights
```

---

## Modal Setup

### 1. Install Modal CLI
```bash
pip install modal
modal setup  # Authenticates with Modal
```

### 2. Set Environment Variables
```bash
# In .env
MODAL_TOKEN_ID=your-token-id
MODAL_TOKEN_SECRET=your-token-secret
```

### 3. Deploy (Optional)
```bash
modal deploy src/tools/code_execution.py
```

---

## What to Show Judges

For the Modal Innovation Award ($2,500):

1. **Sandbox Isolation** - Code runs in container, not local
2. **Scientific Computing** - Real pandas/scipy analysis
3. **Safety** - Can't access local filesystem
4. **Speed** - Modal's fast cold starts

### Demo Script

```bash
# Run the Modal verification script
uv run python examples/modal_demo/verify_sandbox.py
```

This proves code runs in Modal, not locally.

---

## Files to Update

- [ ] Wire `code_execution.py` into pipeline
- [ ] Create `src/agents/analysis_agent.py`
- [ ] Update `examples/modal_demo/` with working demo
- [ ] Add Modal setup to README
- [ ] Test with real Modal account

---

## Cost Estimate

Modal pricing for our use case:
- CPU sandbox: ~$0.0001 per execution
- For demo/judging: < $1 total
- Free tier: 30 hours/month

Not a cost concern.
