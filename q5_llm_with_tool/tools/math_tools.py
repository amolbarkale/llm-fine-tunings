
import sys
import io

def python_exec(code: str) -> str:
    """Execute Python code and return the stdout output."""
    print("🔨 Tool Called: python_exec")
    print(f"Code: {code}")
    
    # Capture stdout
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    try:
        # Create safe environment with essential functions
        safe_globals = {
            "__builtins__": {
                "len": len,
                "print": print,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "sum": sum,
                "max": max,
                "min": min,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "abs": abs,
                "round": round,
            }
        }
        
        # Execute the code in safe environment
        exec(code, safe_globals, {})
        output = new_stdout.getvalue().strip()
        return output if output else "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Restore stdout
        sys.stdout = old_stdout