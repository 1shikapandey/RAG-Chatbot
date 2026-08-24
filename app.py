import os
import json
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from rag import RAGPipeline
from evaluator import RAGEvaluator

# Load env variables
load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")

# Global variables for pipeline and evaluator
pipeline = None
evaluator = None

def init_pipeline():
    global pipeline, evaluator
    # Reload environment to pick up any newly saved keys
    load_dotenv()
    
    # Check if AboutMLSC folder exists, if not use current dir
    docs_dir = "AboutMLSC"
    if not os.path.exists(docs_dir):
        # Fallback to current folder if needed
        docs_dir = "."
        
    pipeline = RAGPipeline(docs_dir)
    evaluator = RAGEvaluator(pipeline)
    print("RAG Pipeline and Evaluator initialized.")

# Initialize on startup
init_pipeline()

@app.route('/')
def index():
    """Serves the main application page."""
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """RAG Chat query endpoint."""
    data = request.json or {}
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is empty'}), 400
        
    try:
        response = pipeline.answer_query(query)
        # Check if running in mock/dry mode
        is_mock = not bool(os.getenv("GEMINI_API_KEY"))
        return jsonify({
            'answer': response['answer'],
            'sources': response['sources'],
            'is_mock': is_mock
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Returns raw text content of all source documents in the knowledge base."""
    docs = {}
    docs_dir = pipeline.docs_dir
    
    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            if filename.endswith(".txt") and not filename.startswith("._"):
                filepath = os.path.join(docs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        docs[filename] = f.read()
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
    return jsonify(docs)

@app.route('/api/evaluate', methods=['POST'])
def run_evaluation():
    """Runs the RAG evaluation dataset and returns metrics and results."""
    try:
        report = evaluator.evaluate_dataset("eval_set.json")
        if report:
            # Save results locally
            with open("eval_results.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return jsonify(report)
        else:
            return jsonify({'error': 'Evaluation failed to run'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns configuration status, i.e., whether the API Key is configured."""
    has_key = bool(os.getenv("GEMINI_API_KEY"))
    return jsonify({
        'has_api_key': has_key
    })

@app.route('/api/save-key', methods=['POST'])
def save_key():
    """Saves a Gemini API Key to .env and re-initializes the pipeline."""
    data = request.json or {}
    key = data.get('key', '').strip()
    
    if not key:
        return jsonify({'error': 'API Key is empty'}), 400
        
    try:
        # Write/Update the .env file in the workspace
        env_content = f"GEMINI_API_KEY={key}\n"
        
        # Read existing environment file if exists, merging values
        if os.path.exists(".env"):
            with open(".env", "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Filter out GEMINI_API_KEY if already present
            new_lines = []
            for line in lines:
                if not line.startswith("GEMINI_API_KEY="):
                    new_lines.append(line)
            new_lines.append(env_content)
            env_content = "".join(new_lines)
            
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_content)
            
        # Re-initialize the pipeline
        init_pipeline()
        return jsonify({'success': True, 'message': 'API Key saved and pipeline re-initialized.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask development server on localhost:5000
    app.run(host='127.0.0.1', port=5000, debug=True)
