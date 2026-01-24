# codes/giskard.py
import os
import pandas as pd
import giskard
from giskard.rag import KnowledgeBase, generate_testset, evaluate, AgentAnswer
from rag_system import TigressTechRAG
from API.huggingface_api import huggingface_completion
from escalation_evaluator import EscalationEvaluator
from prompt_manager import PromptManager

# Paths relative to the 'codes' directory
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'prompt_config.yaml')
FILES_DIR = os.path.join(os.path.dirname(__file__), '..', 'files')

# Initialize shared components
prompt_manager = PromptManager(config_path=CONFIG_PATH)
rag = TigressTechRAG()
rag.setup_rag()

# ==================== RAGET COMPONENTS ====================

def get_model_answer(question: str, history=None) -> AgentAnswer:
    """
    Core wrapper that simulates the production pipeline.
    RAGET uses this to evaluate the Retriever vs Generator.
    """
    # 1. Retrieval Step
    context_docs = rag.get_context(question)
    context_text = "\n".join([d.page_content for d in context_docs])

    # 2. Prompt Formatting (Uses your real system prompt)
    formatted_prompt = prompt_manager.format_main_prompt(
        query_type="general",
        context=context_text,
        user_input=question
    )

    # 3. LLM Generation
    result = huggingface_completion(formatted_prompt)
    answer = result.get('response', "I am unable to answer that.")

    # 4. Return structured answer for Giskard metrics
    return AgentAnswer(
        message=answer,
        documents=[d.page_content for d in context_docs]
    )

def setup_knowledge_base():
    """Loads raw documents so Giskard can generate adversarial questions."""
    docs = []
    for filename in ["faqs.txt", "services_policies.txt"]:
        path = os.path.join(FILES_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                docs.append({"content": f.read(), "filename": filename})

    return KnowledgeBase(pd.DataFrame(docs))

# ==================== TESTING SUITES ====================

def run_raget_benchmark():
    """Runs a full RAGET assessment (Faithfulness, Answer Relevance, etc)."""
    print("🚀 Initializing Giskard RAGET...")
    kb = setup_knowledge_base()

    # Generate 20-30 questions: Simple, Complex, Distracting, and Double.
    testset = generate_testset(
        kb,
        num_questions=25,
        agent_description="Nigerian IT support bot providing technical and billing help."
    )
    testset.save("rag_testset.jsonl")

    print("📊 Evaluating RAG Components...")
    report = evaluate(get_model_answer, testset=testset, knowledge_base=kb)
    report.to_html("raget_report.html")
    print("✅ RAGET report saved to raget_report.html")

def scan_escalation_vulnerabilities():
    """
    Red-Teaming 'Attack' on the Escalation Logic.
    Tests if the bot can be manipulated into NOT escalating when it should.
    """
    evaluator = EscalationEvaluator()

    def escalation_predict(df):
        results = []
        for _, row in df.iterrows():
            should_esc, _, _ = evaluator.evaluate_escalation_need(
                current_message=row["question"],
                current_ai_response="I'm sorry, I can't help."
            )
            results.append("Escalate" if should_esc else "Normal")
        return results

    gsk_model = giskard.Model(
        model=escalation_predict,
        model_type="classification",
        classification_labels=["Escalate", "Normal"],
        name="Escalation_Logic_Red_Team",
        description="Checks for failures in handing off angry or complex users to humans."
    )

    # The 'Scan' is the actual attack
    scan_results = giskard.scan(gsk_model)
    scan_results.to_html("escalation_attack_report.html")
    print("🛡️ Escalation attack report saved to escalation_attack_report.html")

if __name__ == "__main__":
    # Execute both tests
    run_raget_benchmark()
    scan_escalation_vulnerabilities()
