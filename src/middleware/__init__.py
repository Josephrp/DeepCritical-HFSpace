"""Middleware for workflow state management, parallel loop coordination, and budget tracking.

This module provides:
- WorkflowState: Thread-safe state management using ContextVar
- WorkflowManager: Coordination of parallel research loops
- BudgetTracker: Token, time, and iteration budget tracking
"""

from src.middleware.budget_tracker import BudgetStatus, BudgetTracker
from src.middleware.state_machine import (
    WorkflowState,
    get_workflow_state,
    init_workflow_state,
)
from src.middleware.workflow_manager import (
    LoopStatus,
    ResearchLoop,
    WorkflowManager,
)

__all__ = [
    # State management
    "WorkflowState",
    "init_workflow_state",
    "get_workflow_state",
    # Workflow management
    "WorkflowManager",
    "ResearchLoop",
    "LoopStatus",
    # Budget tracking
    "BudgetTracker",
    "BudgetStatus",
]
