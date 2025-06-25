
import sys
import io

def string_counter(code: str) -> str:
    """Count characters, words, or analyze strings."""
    print("🔨 Tool Called: string_counter")
    print(f"Code: {code}")
    
    # Capture stdout
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    try:
        # Safe environment for string operations
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
        
        exec(code, safe_globals, {})
        output = new_stdout.getvalue().strip()
        return output if output else "Code executed successfully (no output)"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout