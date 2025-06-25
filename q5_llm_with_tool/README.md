# Tool-Enhanced Reasoning Script

A Python script that uses Large Language Models (LLMs) to interpret natural language queries and automatically call external tools when needed. The system employs chain-of-thought reasoning to decide when mathematical calculations or string analysis operations are required.

## 🚀 Features

- **Natural Language Processing**: Interprets complex queries using Gemini AI
- **Automatic Tool Selection**: Intelligently decides when to use mathematical or string analysis tools
- **Chain-of-Thought Reasoning**: Step-by-step problem-solving approach (plan → action → observe → output)
- **Safe Code Execution**: Sandboxed environment for running calculations and string operations
- **Interactive CLI**: Real-time conversation interface with the reasoning agent

## 📁 Project Structure

```
├── main.py                 # Main script with LLM agent logic
├── tools/
│   ├── math_tools.py      # Mathematical calculation functions
│   └── string_tools.py    # String analysis and counting functions
├── outputs.md             # Example test queries and their outputs
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── .env.example          # Environment variables template
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Gemini API key from Google AI Studio

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd tool-enhanced-reasoning-script
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Get your Gemini API key**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Copy the key to your `.env` file

## 🏃‍♂️ Usage

Run the script:
```bash
python main.py
```

The interactive CLI will start. You can now ask questions that require:
- Mathematical calculations
- String analysis and counting
- Complex reasoning combining multiple operations

Type `exit` to quit the program.

## 📋 Example Queries

Here are some example queries you can try:

### Mathematical Operations
```
What's the square root of the average of 18 and 50?
Calculate 847 * 293 + 156
What's 25% of 240?
```

### String Analysis
```
How many vowels are in the word 'Multimodality'?
Count the letters in 'artificial intelligence'
How many times does 'a' appear in 'banana'?
```

### Complex Reasoning
```
Is the number of letters in 'machine' greater than the number of vowels in 'reasoning'?
Compare the word length of 'python' with the vowel count in 'programming'
What's the sum of vowels in 'hello' and 'world'?
```

### Conversational
```
Explain what machine learning is
What is the difference between AI and ML?
Tell me about Python programming
```

## 🔧 How It Works

### Tool Decision Logic

The system uses a structured JSON-based approach to decide tool usage:

1. **Plan**: Analyzes the user query and determines what type of operation is needed
2. **Action**: Selects the appropriate tool (calculator, string_counter, or noop)
3. **Observe**: Executes the tool and captures the output
4. **Output**: Provides the final answer based on the tool results

### Available Tools

- **calculator**: Handles mathematical expressions and calculations
- **string_counter**: Performs string analysis, counting, and text operations  
- **noop**: Used for conversational responses that don't require computation

### Prompt Engineering

The system uses carefully crafted prompts that help the LLM:
- Identify when calculations or string operations are needed
- Generate appropriate Python code for the tools
- Provide clear, step-by-step reasoning
- Maintain safety through restricted execution environments

## 📊 Sample Outputs

For detailed examples of how the system handles various queries, see [outputs.md](outputs.md) which contains:
- 5+ diverse test queries with complete step-by-step outputs
- Demonstrations of tool selection logic
- Examples of mathematical calculations, string analysis, and reasoning tasks

## 🤖 Technical Details

## 🚨 Limitations

- Requires internet connection for Gemini API calls
- Limited to mathematical and string operations (extensible design allows for more tools)
- API rate limits may apply based on your Gemini API plan
- Complex mathematical operations are limited to Python's built-in capabilities