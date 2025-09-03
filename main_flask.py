import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from agents import agent_workflow

# Initialize Flask and OpenAI
app = Flask(__name__)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    message = request.json.get("message", "") if request.json else ""
    
    if not message.strip():
        return jsonify({"error": "Message cannot be empty"}), 400
    
    # Check if this is a data analysis request with file upload
    use_uploaded_file = "analyze" in message.lower() and "data" in message.lower()
    result = agent_workflow(message, client, use_uploaded_file=use_uploaded_file)
    
    return jsonify(result)

def cli_mode():
    """Command line interface mode"""
    print("AI Assistant CLI Mode - Type 'quit' to exit")
    
    while True:
        try:
            message = input("\nEnter your message: ").strip()
            if message.lower() == "quit":
                break
            
            if not message:
                continue

            # For PDF and data analysis in CLI, accept file paths
            file_path = None
            if "analyze" in message.lower() and ("data" in message.lower() or "pdf" in message.lower()):
                file_input = input("Enter file path (you can drag & drop the file here): ").strip()
                if file_input:
                    file_path = file_input.strip('"').strip("'")

            result = agent_workflow(message, client, file_path=file_path)

            # Display result
            prefix = "AI (with web search)" if result.get("search_used") else "AI"
            print(f"\n{prefix}: {result['response']}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_mode()
    else:
        app.run(debug=True)