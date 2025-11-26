# DeepCritical Hackathon Priority Summary

## 4 Days Left (Deadline: Nov 30, 2025 11:59 PM UTC)

---

## Git Contribution Analysis

```
The-Obstacle-Is-The-Way: 20+ commits (Phases 1-11, all demos, all fixes)
MarioAderman:            3 commits (Modal, LlamaIndex, PubMed fix)
JJ (Maintainer):         0 code commits (merge button only)
```

**Conclusion:** You built 90%+ of this codebase.

---

## Current Stack (What We Have)

| Component | Status | Files |
|-----------|--------|-------|
| PubMed Search | ✅ Working | `src/tools/pubmed.py` |
| ClinicalTrials Search | ✅ Working | `src/tools/clinicaltrials.py` |
| bioRxiv Search | ✅ Working | `src/tools/biorxiv.py` |
| Search Handler | ✅ Working | `src/tools/search_handler.py` |
| Embeddings/ChromaDB | ✅ Working | `src/services/embeddings.py` |
| LlamaIndex RAG | ✅ Working | `src/services/llamaindex_rag.py` |
| Hypothesis Agent | ✅ Working | `src/agents/hypothesis_agent.py` |
| Report Agent | ✅ Working | `src/agents/report_agent.py` |
| Judge Agent | ✅ Working | `src/agents/judge_agent.py` |
| Orchestrator | ✅ Working | `src/orchestrator.py` |
| Gradio UI | ✅ Working | `src/app.py` |
| Modal Code Execution | ⚠️ Built, not wired | `src/tools/code_execution.py` |
| **MCP Server** | ❌ **MISSING** | Need to create |

---

## What's Required for Track 2 (MCP in Action)

| Requirement | Have It? | Priority |
|-------------|----------|----------|
| Autonomous agent behavior | ✅ Yes | - |
| Must use MCP servers as tools | ❌ **NO** | **P0** |
| Must be Gradio app | ✅ Yes | - |
| Planning/reasoning/execution | ✅ Yes | - |

**Bottom Line:** Without MCP server, we're potentially disqualified from Track 2.

---

## 3 Things To Do (In Order)

### 1. MCP Server (P0 - Required)
- **File:** `src/mcp_server.py`
- **Time:** 2-4 hours
- **Doc:** `02_mcp_server_integration.md`
- **Why:** Required for Track 2. No MCP = no entry.

### 2. Modal Wiring (P1 - $2,500 Prize)
- **File:** Update `src/agents/analysis_agent.py`
- **Time:** 2-3 hours
- **Doc:** `03_modal_integration.md`
- **Why:** Modal Innovation Award is $2,500

### 3. Demo Video + Submission (P0 - Required)
- **Time:** 1-2 hours
- **Why:** Required for all submissions

---

## Submission Checklist

- [ ] Space in MCP-1st-Birthday org
- [ ] Tag: `mcp-in-action-track-enterprise`
- [ ] Social media post link
- [ ] Demo video (1-5 min)
- [ ] MCP server working
- [ ] All tests passing

---

## Prize Math

| Award | Amount | Eligible? |
|-------|--------|-----------|
| Track 2 1st Place | $2,500 | If MCP works |
| Modal Innovation | $2,500 | If Modal wired |
| LlamaIndex | $1,000 | Yes (have it) |
| Community Choice | $1,000 | Maybe |
| **Total Potential** | **$7,000** | With MCP + Modal |

---

## Next Actions

```bash
# 1. Read MCP integration doc
cat docs/pending/02_mcp_server_integration.md

# 2. Create MCP server
# (implement based on doc)

# 3. Test MCP works
uv run python src/mcp_server.py

# 4. Wire Modal into pipeline
# (see 03_modal_integration.md)

# 5. Record demo video

# 6. Submit to MCP-1st-Birthday org
```
