"""数值计算工具。

安全策略：用 AST 白名单求值，而不是裸 eval，杜绝任意代码执行。
这是"让 LLM 能够算数但不给它执行任意代码的能力"的最小可用实现。
"""

from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any

from ..schemas import ToolResult

__all__ = ["SafeCalculator", "safe_eval", "calculate"]

# 允许出现的运算符与函数白名单
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "sqrt": math.sqrt,
    "log": math.log,
    "pow": math.pow,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculationError(ValueError):
    """表达式不合法或不被白名单允许。"""


def safe_eval(expression: str) -> float:
    """在白名单约束下求值算术表达式。

    支持：数字、四则运算、幂、取模、少量数学函数与常量。
    拒绝：变量赋值、属性访问、函数调用（白名单外）、下标、比较等。
    """

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise CalculationError("只允许调用白名单函数")
            name = node.func.id
            if name not in _FUNCS:
                raise CalculationError(f"不允许的函数: {name}")
            return _FUNCS[name](*(_eval(arg) for arg in node.args))
        raise CalculationError(f"表达式包含不被允许的语法: {ast.dump(node)[:80]}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"表达式语法错误: {exc}") from exc

    result = _eval(tree)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise CalculationError("结果必须是数值")
    return float(result)


class SafeCalculator:
    """以工具协议暴露的安全计算器。"""

    name = "calculator"

    def __call__(self, expression: str = "", **kwargs: Any) -> ToolResult:
        return self.run(expression or kwargs.get("query", ""))

    def run(self, expression: str) -> ToolResult:
        started = time.perf_counter()
        try:
            value = safe_eval(expression)
        except Exception as exc:
            return ToolResult(
                tool=self.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            output=f"{expression} = {value:g}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )


def calculate(expression: str) -> ToolResult:
    """函数式快捷入口。"""
    return SafeCalculator().run(expression)
