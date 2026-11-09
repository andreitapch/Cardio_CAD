"""Compute node: a SymPy-backed tool the Author can call to verify a
symbolic/algebraic claim before including it in the draft.

Exposed as an in-process MCP tool, per the Claude Agent SDK custom tools
pattern (create_sdk_mcp_server + @tool).
"""

from typing import Any

import sympy
from sympy.parsing.sympy_parser import parse_expr
from claude_agent_sdk import tool, create_sdk_mcp_server, ToolAnnotations


@tool(
    "verify_algebraic_equality",
    "Check whether two math expressions are equal, using SymPy symbolic "
    "simplification. Use this before asserting any derivation step or "
    "formula in the draft. Expressions should be given in Python/SymPy "
    "syntax, e.g. 'x**2 - 1' and '(x - 1)*(x + 1)'.",
    {"lhs": str, "rhs": str},
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def verify_algebraic_equality(args: dict[str, Any]) -> dict[str, Any]:
    lhs_str, rhs_str = args["lhs"], args["rhs"]
    try:
        lhs = parse_expr(lhs_str)
        rhs = parse_expr(rhs_str)
        difference = sympy.simplify(lhs - rhs)
        is_equal = difference == 0
        text = (
            f"'{lhs_str}' == '{rhs_str}': {'VERIFIED equal' if is_equal else 'NOT equal'}"
            + ("" if is_equal else f" (simplified difference: {difference})")
        )
        return {"content": [{"type": "text", "text": text}]}
    except Exception as e:  # malformed expression, unsupported syntax, etc.
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Could not evaluate expressions: {str(e)}. "
                    "Rewrite using standard SymPy syntax and retry.",
                }
            ],
            "is_error": True,
        }


compute_server = create_sdk_mcp_server(
    name="compute_node",
    version="1.0.0",
    tools=[verify_algebraic_equality],
)
