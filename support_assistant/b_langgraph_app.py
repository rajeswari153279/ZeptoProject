# ============================================================
# MODULE 3 - PART B: LANGGRAPH APP (classify_intent -> retrieve_and_answer / direct_answer)
# MOCK_LLM = True means no real API key needed (required default mode)
# ============================================================
from typing import TypedDict
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import chromadb

MOCK_LLM = True  # Required: default graded version works without any API key

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="support_assistant/chroma_db")
collection = client.get_or_create_collection("policies")

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "track", "order",
    "cancel", "damaged", "missing", "gift card", "support hours",
    "shipping", "exchange"
]

class GraphState(TypedDict):
    question: str
    intent: str
    answer: str
    sources: list
    confidence: float

def classify_intent(state: GraphState) -> GraphState:
    q_lower = state["question"].lower()
    if any(keyword in q_lower for keyword in POLICY_KEYWORDS):
        state["intent"] = "policy"
    else:
        state["intent"] = "general"
    return state

def retrieve_and_answer(state: GraphState) -> GraphState:
    query_embedding = model.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    sources = [m["source"] for m in metas]
    # lower distance = more similar; convert to a rough 0-1 confidence score
    confidence = max(0.0, 1 - (sum(distances) / len(distances)))

    if MOCK_LLM:
        # Mock "generation": combine the top retrieved chunks into a grounded answer
        combined = " ".join(docs[:2])
        answer = f"Based on our policy documents: {combined[:400]}..."
    else:
        # Placeholder for a real LLM call (not required/graded by default)
        answer = "Real LLM call would go here."

    state["answer"] = answer
    state["sources"] = sources
    state["confidence"] = round(confidence, 2)
    return state

def direct_answer(state: GraphState) -> GraphState:
    state["answer"] = ("I can help with delivery, returns, membership, order tracking, "
                        "cancellations, damaged/missing items, gift cards, or support hours. "
                        "Could you rephrase your question around one of these topics?")
    state["sources"] = []
    state["confidence"] = 1.0
    return state

def route(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy" else "direct_answer"

graph = StateGraph(GraphState)
graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)

graph.set_entry_point("classify_intent")
graph.add_conditional_edges("classify_intent", route, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer",
})
graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)

app_graph = graph.compile()

if __name__ == "__main__":
    test_questions = [
        "How do I return a damaged item?",
        "What's the weather today?",
        "Can I cancel my order?",
    ]
    for q in test_questions:
        result = app_graph.invoke({"question": q, "intent": "", "answer": "", "sources": [], "confidence": 0.0})
        print(f"\nQ: {q}")
        print(f"Intent: {result['intent']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']}")
    print("\n--- PART B COMPLETE ---")