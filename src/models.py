"""I/O models for the legacy LangGraph pipeline's run services (run_local / run_hpc).

The MCP server (src/mcp/fastmcp_server.py) defines its own request/response
models internally; nothing here is used by it.
"""
from typing import Dict, Optional
from pydantic import BaseModel


class HPCScriptIn(BaseModel):
    case_id: str
    hpc_config: Dict


class HPCScriptOut(BaseModel):
    script_content: str
    script_path: str


class RunIn(BaseModel):
    case_id: str
    environment: str  # "local" | "hpc"
    extra: Optional[Dict] = None


class RunOut(BaseModel):
    job_id: Optional[str]
    status: str  # "submitted" | "completed" | "failed"


class JobStatusIn(BaseModel):
    job_id: str


class JobStatusOut(BaseModel):
    status: str
    details: Optional[Dict] = None
