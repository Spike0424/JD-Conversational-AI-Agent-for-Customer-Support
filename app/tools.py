import ast
import json
import operator
import shlex
import subprocess
from pathlib import Path
from typing import Callable

from langchain_core.tools import tool

from app.config import Settings
from app.integrations.business_service import BusinessAPIError, BusinessService
from app.rag import RAGIndex

ALLOWED_OPERATORS: dict[type, Callable[[float, float], float] | Callable[[float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

TEXT_FILE_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
IGNORED_DIR_NAMES = {".git", ".venv", "__pycache__", ".jd_browser_profile", ".cursor"}
ALLOWED_FILE_EXTENSIONS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml"}
PROTECTED_DIR_NAMES = {".git", ".venv", "__pycache__", ".jd_browser_profile", ".cursor"}
WORKSPACE_ROOT = Path(".").resolve()
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_WRITE_CHARS = 20000
BLOCKED_BASH_PATTERNS = {
    "rm -rf /",
    "shutdown",
    "reboot",
    "mkfs",
    "poweroff",
    "halt",
    "dd if=",
}
BLOCKED_BASH_TOKENS = {";", "|", "&&", "||", ">", ">>", "<", "<<", "`", "$(", "\n"}
ALLOWED_BASH_COMMANDS = {
    "ls",
    "pwd",
    "echo",
    "rg",
    "python",
    "python3",
    "uv",
    "git",
    "pytest",
}
ALLOWED_GIT_SUBCOMMANDS = {"status", "log", "diff", "show", "branch", "rev-parse"}
ALLOWED_WRITE_ROOT_DIRS = {"app"}
ALLOWED_WRITE_FILES = {"README.md"}


def _resolve_workspace_path(path: str) -> Path:
    candidate = (WORKSPACE_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not str(candidate).startswith(str(WORKSPACE_ROOT)):
        raise ValueError(f"Path '{path}' is outside workspace root.")
    return candidate


def _validate_file_path(path: Path, allow_missing: bool = False) -> None:
    if any(part in PROTECTED_DIR_NAMES for part in path.parts):
        raise ValueError("Path targets a protected directory.")
    if path.suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
        raise ValueError(f"Only file types {sorted(ALLOWED_FILE_EXTENSIONS)} are allowed.")
    if not allow_missing and (not path.exists() or not path.is_file()):
        raise ValueError("File not found.")


def _validate_write_target(path: Path) -> None:
    relative = path.relative_to(WORKSPACE_ROOT)
    if relative.name in ALLOWED_WRITE_FILES:
        return
    if not relative.parts:
        raise ValueError("Write target is invalid.")
    if relative.parts[0] not in ALLOWED_WRITE_ROOT_DIRS:
        raise ValueError(
            f"Write operations are restricted to {sorted(ALLOWED_WRITE_ROOT_DIRS)} and {sorted(ALLOWED_WRITE_FILES)}."
        )


def _safe_eval(expression: str) -> float:
    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return ALLOWED_OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
            operand = eval_node(node.operand)
            return ALLOWED_OPERATORS[type(node.op)](operand)
        raise ValueError("Unsupported expression.")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed)


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a math expression like '(2 + 3) * 4'."""
    try:
        result = _safe_eval(expression.strip())
    except Exception as exc:  # noqa: BLE001
        return f"Calculator error: {exc}"
    return str(result)


def build_search_docs_tool(docs_root: str):
    @tool
    def search_docs(query: str) -> str:
        """Search local text files and return top matching snippets."""
        query_lower = query.strip().lower()
        if not query_lower:
            return "Query is empty."

        base = Path(docs_root).resolve()
        matches = []
        for path in base.rglob("*"):
            if any(part in IGNORED_DIR_NAMES for part in path.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            if query_lower not in content.lower():
                continue

            lines = content.splitlines()
            for idx, line in enumerate(lines):
                if query_lower in line.lower():
                    start = max(0, idx - 1)
                    end = min(len(lines), idx + 2)
                    snippet = " | ".join(lines[start:end]).strip()
                    rel_path = path.relative_to(base)
                    matches.append(f"{rel_path}: {snippet}")
                    break

            if len(matches) >= 5:
                break

        if not matches:
            return f"No local docs matched '{query}'."
        return "\n".join(matches)

    return search_docs


def build_rag_search_tool(rag_index: RAGIndex):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Retrieve semantically similar context from local vector index (RAG)."""
        return rag_index.query(query.strip())

    return search_knowledge_base


def _to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def get_business_tools(settings: Settings, docs_root: str, rag_index: RAGIndex):
    service = BusinessService(settings)

    @tool
    def search_products(category: str, budget_min: int = 0, budget_max: int = 99999) -> str:
        """Search 3C products through CRM service (with mock fallback)."""
        try:
            result = service.search_products(category=category, budget_min=budget_min, budget_max=budget_max)
            return _to_json(result.to_payload())
        except BusinessAPIError as exc:
            return _to_json({"status": "error", "code": exc.code, "message": exc.message})

    @tool
    def get_order_status(order_id: str) -> str:
        """Query order and logistics status through OMS service (with mock fallback)."""
        try:
            result = service.get_order_status(order_id=order_id)
            return _to_json(result.to_payload())
        except BusinessAPIError as exc:
            return _to_json({"status": "error", "code": exc.code, "message": exc.message})

    @tool
    def check_warranty(sn_or_imei: str) -> str:
        """Query product warranty info through after-sale service (with mock fallback)."""
        try:
            result = service.check_warranty(sn_or_imei=sn_or_imei)
            return _to_json(result.to_payload())
        except BusinessAPIError as exc:
            return _to_json({"status": "error", "code": exc.code, "message": exc.message})

    @tool
    def create_after_sale_ticket(order_id: str, issue_type: str, details: str = "") -> str:
        """Create after-sale ticket through after-sale service (with mock fallback)."""
        try:
            result = service.create_after_sale_ticket(order_id=order_id, issue_type=issue_type, details=details)
            return _to_json(result.to_payload())
        except BusinessAPIError as exc:
            return _to_json({"status": "error", "code": exc.code, "message": exc.message})

    return [
        calculator,
        build_search_docs_tool(docs_root),
        build_rag_search_tool(rag_index),
        search_products,
        get_order_status,
        check_warranty,
        create_after_sale_ticket,
    ]


def get_dev_tools():
    return [read_file, write_file, edit_file, run_bash]


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from workspace."""
    try:
        target = _resolve_workspace_path(path.strip())
        _validate_file_path(target)
        content = target.read_text(encoding="utf-8", errors="ignore")
        if len(content) > MAX_TOOL_OUTPUT_CHARS:
            return content[:MAX_TOOL_OUTPUT_CHARS] + "\n... [truncated]"
        return content
    except Exception as exc:  # noqa: BLE001
        return f"Read error: {exc}"


@tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """Write UTF-8 text to a file in workspace."""
    try:
        target = _resolve_workspace_path(path.strip())
        _validate_file_path(target, allow_missing=True)
        _validate_write_target(target)
        if len(content) > MAX_WRITE_CHARS:
            return f"Write error: content exceeds {MAX_WRITE_CHARS} characters."
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as f:
            f.write(content)
        return f"Write success: {target}"
    except Exception as exc:  # noqa: BLE001
        return f"Write error: {exc}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one text occurrence in a workspace file."""
    try:
        target = _resolve_workspace_path(path.strip())
        _validate_file_path(target)
        _validate_write_target(target)
        if not old_text:
            return "Edit error: old_text cannot be empty."
        if len(new_text) > MAX_WRITE_CHARS:
            return f"Edit error: new_text exceeds {MAX_WRITE_CHARS} characters."

        original = target.read_text(encoding="utf-8", errors="ignore")
        if old_text not in original:
            return "Edit error: old_text not found."

        updated = original.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        return f"Edit success: {target}"
    except Exception as exc:  # noqa: BLE001
        return f"Edit error: {exc}"


@tool
def run_bash(command: str, timeout_seconds: int = 20) -> str:
    """Execute a single safe bash command inside workspace."""
    try:
        cmd = command.strip()
        if not cmd:
            return "Bash error: command is empty."

        lowered = cmd.lower()
        if any(pattern in lowered for pattern in BLOCKED_BASH_PATTERNS):
            return "Bash error: command blocked by safety policy."
        if any(token in cmd for token in BLOCKED_BASH_TOKENS):
            return "Bash error: shell operators are not allowed."

        parts = shlex.split(cmd)
        if not parts:
            return "Bash error: command is empty after parsing."
        if parts[0] not in ALLOWED_BASH_COMMANDS:
            return f"Bash error: command '{parts[0]}' is not in allowlist."
        if parts[0] == "git":
            if len(parts) < 2:
                return "Bash error: git subcommand is required."
            if parts[1] not in ALLOWED_GIT_SUBCOMMANDS:
                return f"Bash error: git subcommand '{parts[1]}' is not read-only allowlisted."

        completed = subprocess.run(
            parts,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=max(1, min(timeout_seconds, 30)),
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        output = output.strip() or "(no output)"
        if len(output) > MAX_TOOL_OUTPUT_CHARS:
            output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n... [truncated]"
        return f"exit_code={completed.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return "Bash error: command timed out."
    except Exception as exc:  # noqa: BLE001
        return f"Bash error: {exc}"
