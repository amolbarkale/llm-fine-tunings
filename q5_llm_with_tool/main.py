import json
import sys
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

from tools.math_tools import python_exec
from tools.string_tools import string_counter

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


def noop(message: str = "") -> str:
    """No operation tool - used for conversational responses."""
    print("🔨 Tool Called: noop")
    return "Ready for conversational response"

available_tools = {
    "python_exec": {
        "fn": python_exec,
        "description": "Execute mathematical calculations and expressions. Input should be a valid mathematical expression."
    },
    "string_counter": {
        "fn": string_counter,
        "description": "Count characters, words, vowels, or analyze strings. Input should be valid Python code for string operations."
    },
    "noop": {
        "fn": noop,
        "description": "Use for regular conversational responses that don't require computation."
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

User Query: What's the square root of the average of 18 and 50?
{{ "step": "plan", "content": "User wants to calculate square root of average of two numbers. I need to use calculator for this mathematical operation." }}
{{ "step": "action", "function": "calculator", "input": "sqrt(avg(18, 50))" }}
[Wait for observation]
{{ "step": "output", "content": "The square root of the average of 18 and 50 is [result from tool]." }}

User Query: How many vowels are in the word 'Multimodality'?
{{ "step": "plan", "content": "User wants to count vowels in a word. I need to use string_counter for this text analysis." }}
{{ "step": "action", "function": "string_counter", "input": "word = 'Multimodality'\\nvowels = 'aeiouAEIOU'\\ncount = sum(1 for char in word if char in vowels)\\nprint(count)" }}
[Wait for observation]
{{ "step": "output", "content": "The word 'Multimodality' contains [result from tool] vowels." }}

User Query: Is the number of letters in 'machine' greater than the number of vowels in 'reasoning'?
{{ "step": "plan", "content": "User wants to compare letter count in one word with vowel count in another. I need string_counter for this comparison." }}
{{ "step": "action", "function": "string_counter", "input": "word1 = 'machine'\\nword2 = 'reasoning'\\nvowels = 'aeiouAEIOU'\\nletters_count = len(word1)\\nvowels_count = sum(1 for char in word2 if char in vowels)\\nprint(f'Letters in machine: letters_count')\\nprint(f'Vowels in reasoning: vowels_count')\\nprint(f'Is letters_count > vowels_count? letters_count > vowels_count')" }}
[Wait for observation]
{{ "step": "output", "content": "The word 'machine' has [X] letters and 'reasoning' has [Y] vowels. [Comparison result]." }}
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