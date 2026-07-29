from __future__ import annotations
from fastmcp import FastMCP


mcp = FastMCP("arith")

def _as_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError("Expected a number (int/float) or a string representing a number.")

@mcp.tool()
async def add(a: float, b: float) -> float:
    return _as_number(a) + _as_number(b)

@mcp.tool()
async def subtract(a: float, b: float) -> float:
    return _as_number(a) - _as_number(b)
  
@mcp.tool()
async def multiply(a: float, b: float) -> float:
    return _as_number(a) * _as_number(b)
  
@mcp.tool()
async def divide(a: float, b: float) -> float:
    
    return _as_number(a) / _as_number(b)
  
if __name__ == "__main__":
    mcp.run()
    