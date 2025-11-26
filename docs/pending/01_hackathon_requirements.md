# MCP's 1st Birthday Hackathon - Requirements Analysis

## Deadline: November 30, 2025 11:59 PM UTC

---

## Track Selection: MCP in Action (Track 2)

DeepCritical fits **Track 2: MCP in Action** - AI agent applications.

### Required Tags (pick one)
```yaml
tags:
  - mcp-in-action-track-enterprise   # Drug repurposing = enterprise/healthcare
  # OR
  - mcp-in-action-track-consumer     # If targeting patients/consumers
```

### Track 2 Requirements

| Requirement | DeepCritical Status | Action Needed |
|-------------|---------------------|---------------|
| Autonomous Agent behavior | ✅ Have it | Search-Judge-Synthesize loop |
| Must use MCP servers as tools | ❌ **MISSING** | Add MCP server wrapper |
| Must be a Gradio app | ✅ Have it | `src/app.py` |
| Planning, reasoning, execution | ✅ Have it | Orchestrator + Judge |
| Context Engineering / RAG | ✅ Have it | LlamaIndex + ChromaDB |

---

## Prize Opportunities

### Current Eligibility vs With MCP Integration

| Award | Prize | Current | With MCP |
|-------|-------|---------|----------|
| MCP in Action (1st) | $2,500 | ✅ Eligible | ✅ STRONGER |
| Modal Innovation | $2,500 | ❌ Not using | ✅ ELIGIBLE (code execution) |
| Blaxel Choice | $2,500 | ❌ Not using | ⚠️ Could integrate |
| LlamaIndex | $1,000 | ✅ Using (Mario's code) | ✅ ELIGIBLE |
| Google Gemini | $10K credits | ❌ Not using | ⚠️ Could add |
| Community Choice | $1,000 | ⚠️ Possible | ✅ Better demo helps |
| **TOTAL POTENTIAL** | | ~$2,500 | **$8,500+** |

---

## Submission Checklist

- [ ] HuggingFace Space in `MCP-1st-Birthday` organization
- [ ] Track tags in Space README.md
- [ ] Social media post link (X, LinkedIn)
- [ ] Demo video (1-5 minutes)
- [ ] All team members registered
- [ ] Original work (Nov 14-30)

---

## Priority Integration Order

### P0 - MUST HAVE (Required for Track 2)
1. **MCP Server Wrapper** - Expose search tools as MCP servers
   - See: `02_mcp_server_integration.md`

### P1 - HIGH VALUE ($2,500 each)
2. **Modal Integration** - Already have code, need to wire up
   - See: `03_modal_integration.md`

### P2 - NICE TO HAVE
3. **Blaxel** - MCP hosting platform (if time permits)
4. **Gemini API** - Add as LLM option for Google prize

---

## What MCP Actually Means for Us

MCP (Model Context Protocol) is Anthropic's standard for connecting AI to tools.

**Current state:**
- We have `PubMedTool`, `ClinicalTrialsTool`, `BioRxivTool`
- They're Python classes with `search()` methods

**What we need:**
- Wrap these as MCP servers
- So Claude Desktop, Cursor, or any MCP client can use them

**Why this matters:**
- Judges will test if our tools work with Claude Desktop
- No MCP = disqualified from Track 2

---

## Reference Links

- [Hackathon Page](https://huggingface.co/MCP-1st-Birthday)
- [MCP Documentation](https://modelcontextprotocol.io/)
- [Gradio MCP Guide](https://www.gradio.app/guides/building-mcp-server-with-gradio)
- [Discord: #agents-mcp-hackathon-winter25](https://discord.gg/huggingface)
