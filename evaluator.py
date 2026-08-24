import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai
from rag import RAGPipeline

# Load environment
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class RAGEvaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.use_llm = bool(GEMINI_API_KEY)

    def _call_gemini_json(self, prompt):
        """Helper to call Gemini and return parsed JSON response."""
        if not self.use_llm:
            return None
        
        try:
            model = genai.GenerativeModel(
                "models/gemini-2.5-flash",
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(prompt)
            return json.loads(response.text.strip())
        except Exception as e:
            print(f"LLM-as-a-judge API error: {e}")
            return None

    def evaluate_context_precision(self, question, retrieved_contexts):
        """
        Measures Context Precision: Do the retrieved contexts contain relevant info?
        Precision = (Number of relevant retrieved chunks) / (Total retrieved chunks)
        """
        if not retrieved_contexts:
            return 0.0, "No contexts retrieved."

        if not self.use_llm:
            # Fallback mock: if sources overlap with expected, simulate high precision
            return 1.0, "Mock: Lexical match simulation."

        chunks_str = ""
        for i, ctx in enumerate(retrieved_contexts):
            chunk = ctx['chunk']
            chunks_str += f"Chunk [{i}]: (Source: {chunk['source']})\n{chunk['content']}\n\n"

        prompt = f"""Analyze the relevance of each retrieved text chunk to the user's question.
For each chunk, determine if it contains key information that directly helps in answering the question.

Question: {question}

Retrieved Chunks:
{chunks_str}

Output a JSON object with:
- "relevance": a list of objects, one per chunk, with fields:
  - "chunk_index": integer (0-indexed index of the chunk)
  - "is_relevant": boolean (true if the chunk contains information relevant to the question, false otherwise)
  - "explanation": brief reason for this decision
"""
        res = self._call_gemini_json(prompt)
        if not res or 'relevance' not in res:
            return 0.0, "Evaluation failed to return valid JSON."

        items = res['relevance']
        relevant_count = sum(1 for item in items if item.get('is_relevant', False))
        precision = relevant_count / len(items) if items else 0.0
        
        reason = "; ".join([f"Chunk [{item.get('chunk_index')}]: {'Relevant' if item.get('is_relevant') else 'Irrelevant'} ({item.get('explanation')})" for item in items])
        return precision, reason

    def evaluate_context_recall(self, question, reference, retrieved_contexts):
        """
        Measures Context Recall: Can the reference answer be derived from retrieved context?
        Recall = (Number of reference statements supported by context) / (Total reference statements)
        """
        if not retrieved_contexts:
            return 0.0, "No contexts retrieved."

        if not self.use_llm:
            # Fallback mock
            return 1.0, "Mock: Lexical match simulation."

        context_str = "\n\n".join([c['chunk']['content'] for c in retrieved_contexts])

        prompt = f"""Verify if the reference answer can be derived from the retrieved context chunks.
First, break down the reference answer into a list of distinct, atomic factual statements.
Then, for each statement, check if it can be directly inferred from the provided retrieved context.

Question: {question}
Reference Answer: {reference}

Retrieved Context:
{context_str}

Output a JSON object with:
- "statements": a list of objects, with fields:
  - "statement": string (the atomic factual statement from the reference answer)
  - "is_supported": boolean (true if it can be directly inferred from the retrieved context, false otherwise)
  - "explanation": brief reason why or why not
"""
        res = self._call_gemini_json(prompt)
        if not res or 'statements' not in res:
            return 0.0, "Evaluation failed to return valid JSON."

        items = res['statements']
        supported_count = sum(1 for item in items if item.get('is_supported', False))
        recall = supported_count / len(items) if items else 0.0
        
        reason = "; ".join([f"\"{item.get('statement')}\": {'Supported' if item.get('is_supported') else 'Unsupported'} ({item.get('explanation')})" for item in items])
        return recall, reason

    def evaluate_answer_relevancy(self, question, generated_answer):
        """
        Measures Answer Relevancy: Is the generated answer directly responsive to the question?
        Relevancy score: 0.0 to 1.0.
        """
        if not generated_answer:
            return 0.0, "Empty answer."

        # Handle standard "unanswerable" response
        if "provided knowledge base does not contain information" in generated_answer:
            return 1.0, "Correctly identified as unanswerable."

        if not self.use_llm:
            return 1.0, "Mock: Relevancy simulation."

        prompt = f"""Evaluate the relevancy of the generated answer to the user's question.
Score the answer on a scale from 0.0 (completely irrelevant or generic) to 1.0 (directly and fully answers the question with no fluff).
Consider if the answer directly addresses the question and if it contains unnecessary, redundant, or out-of-scope information.

Question: {question}
Generated Answer: {generated_answer}

Output a JSON object with:
- "score": float between 0.0 and 1.0
- "reason": brief explanation justifying the score
"""
        res = self._call_gemini_json(prompt)
        if not res or 'score' not in res:
            return 0.0, "Evaluation failed to return valid JSON."

        return float(res['score']), res.get('reason', '')

    def evaluate_faithfulness(self, generated_answer, retrieved_contexts):
        """
        Measures Faithfulness (Groundedness): Is the answer grounded *only* in retrieved context?
        Faithfulness = (Number of generated statements supported by context) / (Total generated statements)
        """
        if not generated_answer:
            return 0.0, "Empty answer."
            
        if "provided knowledge base does not contain information" in generated_answer:
            return 1.0, "Correctly refused to answer (faithful)."

        if not retrieved_contexts:
            return 0.0, "No contexts retrieved."

        if not self.use_llm:
            return 1.0, "Mock: Faithfulness simulation."

        context_str = "\n\n".join([c['chunk']['content'] for c in retrieved_contexts])

        prompt = f"""Evaluate if the generated answer is faithful to the retrieved context chunks. Do not use external knowledge.
First, break down the generated answer into distinct, atomic statements.
Second, for each statement, verify if it is directly and fully supported by the provided retrieved context.

Generated Answer: {generated_answer}

Retrieved Context:
{context_str}

Output a JSON object with:
- "statements": a list of objects, with fields:
  - "statement": string (the atomic statement from the generated answer)
  - "is_supported": boolean (true if it is fully supported by the retrieved context, false otherwise)
  - "explanation": brief reason why or why not
"""
        res = self._call_gemini_json(prompt)
        if not res or 'statements' not in res:
            return 0.0, "Evaluation failed to return valid JSON."

        items = res['statements']
        supported_count = sum(1 for item in items if item.get('is_supported', False))
        faithfulness = supported_count / len(items) if items else 0.0
        
        reason = "; ".join([f"\"{item.get('statement')}\": {'Grounded' if item.get('is_supported') else 'Hallucination'} ({item.get('explanation')})" for item in items])
        return faithfulness, reason

    def evaluate_question(self, item):
        """Evaluates a single question item from the evaluation dataset."""
        question = item['question']
        reference = item['reference_answer']
        expected_sources = item.get('expected_sources', [])
        
        # 1. Run RAG Pipeline
        retrieved = self.pipeline.retrieve(question)
        answer = self.pipeline.generate_answer(question, retrieved)
        
        # Determine actual sources retrieved
        actual_sources = [r['chunk']['source'] for r in retrieved if r['rrf_score'] > 0]
        
        # 2. Run Evaluations
        precision, prec_reason = self.evaluate_context_precision(question, retrieved)
        recall, recall_reason = self.evaluate_context_recall(question, reference, retrieved)
        relevancy, relevancy_reason = self.evaluate_answer_relevancy(question, answer)
        faithfulness, faith_reason = self.evaluate_faithfulness(answer, retrieved)
        
        # Simple Source overlap retrieval evaluation
        retrieval_success = True
        if expected_sources:
            overlap = set(expected_sources).intersection(set(actual_sources))
            retrieval_success = len(overlap) > 0

        # Wait to prevent hitting rate limits
        if self.use_llm:
            time.sleep(1.0)
            
        return {
            "id": item['id'],
            "category": item['category'],
            "question": question,
            "reference_answer": reference,
            "generated_answer": answer,
            "expected_sources": expected_sources,
            "actual_sources": list(set(actual_sources)),
            "retrieval_success": retrieval_success,
            "metrics": {
                "context_precision": precision,
                "context_recall": recall,
                "answer_relevancy": relevancy,
                "faithfulness": faithfulness
            },
            "explanations": {
                "context_precision": prec_reason,
                "context_recall": recall_reason,
                "answer_relevancy": relevancy_reason,
                "faithfulness": faith_reason
            }
        }

    def evaluate_dataset(self, dataset_path="eval_set.json"):
        """Runs the entire evaluation set and aggregates metrics."""
        if not os.path.exists(dataset_path):
            print(f"Error: Evaluation set '{dataset_path}' not found.")
            return None

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"Starting evaluation of {len(dataset)} questions...")
        results = []
        
        for item in dataset:
            print(f"Evaluating {item['id']} ({item['category']}): {item['question'][:40]}...")
            res = self.evaluate_question(item)
            results.append(res)
            
        # Aggregate metrics
        count = len(results)
        avg_precision = sum(r['metrics']['context_precision'] for r in results) / count if count else 0.0
        avg_recall = sum(r['metrics']['context_recall'] for r in results) / count if count else 0.0
        avg_relevancy = sum(r['metrics']['answer_relevancy'] for r in results) / count if count else 0.0
        avg_faithfulness = sum(r['metrics']['faithfulness'] for r in results) / count if count else 0.0
        
        category_metrics = {}
        for cat in set(r['category'] for r in results):
            cat_results = [r for r in results if r['category'] == cat]
            cat_count = len(cat_results)
            category_metrics[cat] = {
                "count": cat_count,
                "context_precision": sum(r['metrics']['context_precision'] for r in cat_results) / cat_count,
                "context_recall": sum(r['metrics']['context_recall'] for r in cat_results) / cat_count,
                "answer_relevancy": sum(r['metrics']['answer_relevancy'] for r in cat_results) / cat_count,
                "faithfulness": sum(r['metrics']['faithfulness'] for r in cat_results) / cat_count
            }

        report = {
            "summary": {
                "total_questions": count,
                "is_llm_evaluated": self.use_llm,
                "average_context_precision": avg_precision,
                "average_context_recall": avg_recall,
                "average_answer_relevancy": avg_relevancy,
                "average_faithfulness": avg_faithfulness
            },
            "category_metrics": category_metrics,
            "detailed_results": results
        }
        
        return report

if __name__ == "__main__":
    # Test runner
    pipeline = RAGPipeline("AboutMLSC")
    evaluator = RAGEvaluator(pipeline)
    
    report = evaluator.evaluate_dataset("eval_set.json")
    if report:
        print("\n=== EVALUATION REPORT SUMMARY ===")
        print(json.dumps(report['summary'], indent=2))
        
        # Save results to a file
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("\nDetailed results saved to 'eval_results.json'.")
