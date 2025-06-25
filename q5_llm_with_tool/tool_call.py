import json
import sys
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
client = genai.GenerativeModel('gemini-2.0-flash-exp')

def convert_messages_to_gemini(messages):
    """Convert OpenAI format messages to Gemini format."""
    # Combine all messages into a single prompt for Gemini
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"SYSTEM: {content}")
        elif role == "user":
            prompt_parts.append(f"USER: {content}")
        elif role == "assistant":
            prompt_parts.append(f"ASSISTANT: {content}")
    
    return "\n\n".join(prompt_parts)

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

def noop(message: str = "") -> str:
    """No operation tool - used for conversational responses."""
    print("🔨 Tool Called: noop")
    return "Ready for conversational response"

# Enhanced tool registry for counting and arithmetic
available_tools = {
    "python_exec": {
        "fn": python_exec,
        "description": "Execute Python code for counting characters, arithmetic operations, or any computation. Input should be valid Python code."
    },
    "noop": {
        "fn": noop,
        "description": "Use for regular conversational responses that don't require computation or when providing explanations."
    }
}

# Enhanced system prompt for counting/arithmetic tasks
system_prompt = f"""
You are a helpful AI Assistant specialized in solving counting and arithmetic problems accurately.
You work in start, plan, action, observe mode to overcome tokenizer limitations by using Python tools.

IMPORTANT: For any counting (characters, words, substrings) or arithmetic operations, you MUST use python_exec tool.
Never try to count or calculate manually - always use the tool for precision.

For the given user query and available tools:
1. PLAN: Analyze what needs to be done
2. ACTION: Select and call the appropriate tool
3. OBSERVE: Wait for tool output
4. OUTPUT: Provide final response based on observation

Rules:
- Follow the Output JSON Format exactly
- Always perform one step at a time and wait for next input
- For counting/arithmetic: ALWAYS use python_exec tool
- For explanations/conversations: use noop tool
- Write clean, executable Python code

Output JSON Format:
{{
    "step": "plan|action|observe|output",
    "content": "description of what you're doing",
    "function": "tool name if step is action",
    "input": "tool input parameter"
}}

Available Tools:
- python_exec: Execute Python code for counting, arithmetic, or computations
- noop: For conversational responses and explanations

Examples:

User Query: How many 'r' in 'strawberry'?
{{ "step": "plan", "content": "User wants to count occurrences of 'r' in 'strawberry'. I need to use python_exec for accurate counting." }}
{{ "step": "action", "function": "python_exec", "input": "text = 'strawberry'\\ncount = len([c for c in text if c == 'r'])\\nprint(count)" }}
[Wait for observation]
{{ "step": "output", "content": "There are 3 occurrences of 'r' in 'strawberry'." }}

User Query: What's 847 * 293 + 156?
{{ "step": "plan", "content": "User wants arithmetic calculation. I'll use python_exec for precise computation." }}
{{ "step": "action", "function": "python_exec", "input": "result = 847 * 293 + 156\\nprint(result)" }}
[Wait for observation]
{{ "step": "output", "content": "The result of 847 * 293 + 156 is [result from tool]." }}

User Query: Explain what def f(x): return x**2 does
{{ "step": "plan", "content": "User wants an explanation of a function, not execution. I'll use noop for conversational response." }}
{{ "step": "action", "function": "noop", "input": "explaining function" }}
[Wait for observation]
{{ "step": "output", "content": "This function defines f(x) that takes a parameter x and returns x squared (x to the power of 2). For example, f(3) would return 9." }}
"""

def run_agent():
    """Run the enhanced counting/arithmetic agent."""
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    print("🤖 Enhanced Tool-Calling Agent Ready!")
    print("Specialized in counting and arithmetic problems.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input('User: ')
        if user_query.lower() == 'exit':
            break
            
        messages.append({"role": "user", "content": user_query})
        
        step_count = 0
        while True:
            step_count += 1
            if step_count > 10:  # Safety limit
                print("🚨 Max steps reached, ending interaction")
                break
                
            try:
                # Convert messages to Gemini format
                gemini_messages = convert_messages_to_gemini(messages)
                
                response = client.generate_content(
                    gemini_messages,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                
                parsed_output = json.loads(response.text)
                messages.append({"role": "assistant", "content": json.dumps(parsed_output)})
                
                # Handle different steps
                if parsed_output.get("step") == "plan":
                    print(f"🧠 Planning: {parsed_output.get('content')}")
                    continue
                
                elif parsed_output.get("step") == "action":
                    tool_name = parsed_output.get("function")
                    tool_input = parsed_output.get("input", "")
                    
                    print(f"⚡ Action: Using {tool_name}")
                    
                    if tool_name in available_tools:
                        try:
                            output = available_tools[tool_name]["fn"](tool_input)
                            observe_message = {
                                "step": "observe", 
                                "output": str(output)
                            }
                            messages.append({"role": "assistant", "content": json.dumps(observe_message)})
                            print(f"👀 Observation: {output}")
                            continue
                        except Exception as e:
                            error_message = {
                                "step": "observe", 
                                "output": f"Tool error: {str(e)}"
                            }
                            messages.append({"role": "assistant", "content": json.dumps(error_message)})
                            print(f"❌ Tool Error: {str(e)}")
                            continue
                    else:
                        print(f"❌ Unknown tool: {tool_name}")
                        continue
                
                elif parsed_output.get("step") == "output":
                    print(f"🤖 Agent: {parsed_output.get('content')}")
                    break
                
                else:
                    print(f"🔄 Step: {parsed_output.get('step')} - {parsed_output.get('content', '')}")
                    continue
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break

if __name__ == "__main__":
    run_agent()