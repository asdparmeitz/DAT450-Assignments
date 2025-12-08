import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState
import math
import re
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState

# Get the directory 
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "ori_pqal.json")
tmp_data = pd.read_json(json_path).T
# some labels have been defined as "maybe", only keep the yes/no answers
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]


documents = pd.DataFrame({"abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS+[row.LONG_ANSWER]), axis=1),
             "year": tmp_data.YEAR})
questions = pd.DataFrame({"question": tmp_data.QUESTION,
             "year": tmp_data.YEAR,
             "gold_label": tmp_data.final_decision,
             "gold_context": tmp_data.LONG_ANSWER,
             "gold_document_id": documents.index})


             # Step 2: Configure LangChainLM
# Choose a model from Hugging Face, prio training speed
lm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    model_kwargs={
        "torch_dtype": "auto",
        "device_map": "auto"
        },
    pipeline_kwargs={
        "max_new_tokens": 20,
        "temperature": 0.2,
        "top_p": 0.95,
        "return_full_text": False
    }
)
lm.pipeline.tokenizer.padding_side = "left"
if lm.pipeline.tokenizer.pad_token is None:
    lm.pipeline.tokenizer.pad_token = lm.pipeline.tokenizer.eos_token

response = lm.invoke("Hello, how are you?")
print(response)


# Pre download the embedding model, LangChain download bug....
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
    )

test = "What is the capital of France?"
test_embedding = embeddings.embed_query(test)
print(test_embedding)




text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=3000,
    chunk_overlap=400,
)

metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(documents.abstract.tolist(), metadatas=metadatas)
#print(texts[0])
# print(texts[1])
# print(texts[2])
# print(texts[3])
# print(texts[4])



vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
)



# Sanity check, Chroma uses L2-score by default so scores closer to 0 means that its a good match
results = vector_store.similarity_search_with_score(
    "korvar med mos", k=3
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1] # get the user input query
        retrieved_docs = self.vector_store.similarity_search(last_message.text)  # search for documents

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)  

        augmented_message_content = (
        "<|im_start|>system\n"
        "You are a scientific QA assistant. "
        "You will be given a medical yes/no question and a context passage. "
        "CRITICAL: You MUST base your answer ONLY on the provided context. "
        "Your answer MUST only include exactly one word: 'Yes' or 'No'.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question:\n{q}\n\n"
        f"Context:\n{docs_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
         )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }


from langchain.agents import create_agent

rag_middleware = RetrieveDocumentsMiddleware(vector_store)

agent = create_agent(
    model=lm,
    tools=[],
    middleware=[rag_middleware],
)

your_query = questions["question"].iloc[2]

for step in agent.stream(
    {"messages": [{"role": "user", "content": your_query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()


def extract_label_from_answer(answer: str):
    if not isinstance(answer, str):
        return None
    first = answer.strip().split()[0].lower()
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return None


rag_preds = []
rag_golds = []
rag_valid_mask = []
doc_hit_flags = []

for i, row in questions.iterrows():
    q = row["question"]
    gold = row["gold_label"].lower()
    gold_doc_id = row["gold_document_id"]

    res = agent.invoke({"messages": [{"role": "user", "content": q}]})
    msg = res["messages"][-1]
    answer = getattr(msg, "content", getattr(msg, "text", str(msg)))

    pred = extract_label_from_answer(answer)
    if pred is not None:
        rag_preds.append(pred)
        rag_golds.append("yes" if gold == "yes" else "no")
        rag_valid_mask.append(True)
    else:
        rag_valid_mask.append(False)

    retrieved_docs = res.get("context", [])
    hit = any(doc.metadata.get("id") == gold_doc_id for doc in retrieved_docs)
    doc_hit_flags.append(hit)

n_total = len(questions)
n_valid = sum(rag_valid_mask)

tp = sum(1 for p, g in zip(rag_preds, rag_golds) if p == "yes" and g == "yes")
fp = sum(1 for p, g in zip(rag_preds, rag_golds) if p == "yes" and g == "no")
fn = sum(1 for p, g in zip(rag_preds, rag_golds) if p == "no" and g == "yes")

precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_yes = (
    2 * precision_yes * recall_yes / (precision_yes + recall_yes)
    if (precision_yes + recall_yes) > 0
    else 0.0
)

acc_rag = (
    sum(p == g for p, g in zip(rag_preds, rag_golds)) / n_valid
    if n_valid > 0
    else float("nan")
)

retrieval_acc = sum(doc_hit_flags) / n_total

print("RAG evaluation:")
print("  Total questions:", n_total)
print("  Valid answers:", n_valid)
print("  Accuracy (on valid):", acc_rag)
print("  F1 (Yes as positive):", f1_yes)
print("  Retrieval hit rate (gold doc in retrieved):", retrieval_acc)


# ===================== CONFIG: SUBSET SIZE =====================
subset_size = 100  # change this if you want fewer/more
subset_size = min(subset_size, len(questions))

questions_eval = questions.sample(n=subset_size, random_state=42).reset_index(drop=True)
print(f"Evaluating on subset of size {len(questions_eval)} out of {len(questions)}")

# ===================== RAG EVALUATION (BATCHED) =====================

rag_prompts = []
rag_golds = []
doc_hit_flags = []

for _, row in questions_eval.iterrows():
    q = row["question"]
    gold = row["gold_label"].lower()
    gold_doc_id = row["gold_document_id"]

    retrieved_docs = vector_store.similarity_search(q, k=1)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    hit = any(doc.metadata.get("id") == gold_doc_id for doc in retrieved_docs)

    # Add retrieval quality check
    print(f"\nQuestion: {q}")
    print(f"Retrieved {len(retrieved_docs)} docs")
    print(f"Gold doc ID: {gold_doc_id}")
    print(f"Retrieved doc IDs: {[doc.metadata.get('id') for doc in retrieved_docs]}")
    print(f"Gold doc in retrieved: {hit}")

    prompt = (
    "<|im_start|>system\n"
    "You are a scientific QA assistant. "
    "You will be given a medical yes/no question and a context passage. "
    "CRITICAL: You MUST base your answer ONLY on the provided context. "
    "Your answer MUST only include exactly one word: 'Yes' or 'No'.<|im_end|>\n"
    "<|im_start|>user\n"
    f"Question:\n{q}\n\n"
    f"Context:\n{docs_content}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

    # Verify context is actually used
    if not docs_content or len(docs_content.strip()) < 10:
        print(f"WARNING: Empty or very short context for question: {q}")

    rag_prompts.append(prompt)
    rag_golds.append("yes" if gold == "yes" else "no")
    hit = any(doc.metadata.get("id") == gold_doc_id for doc in retrieved_docs)
    doc_hit_flags.append(hit)

rag_answers = lm.batch(rag_prompts)
print(f"***THIS IS THE RAG ANSWERS: {rag_answers}***")

# ===================== INSPECTION OF RETRIEVED DOC + ANSWERS =====================

print("\n================ SAMPLE INSPECTION OF RETRIEVAL & ANSWERS ================\n")

for i, row in questions_eval.iterrows():
    q = row["question"]
    gold_doc_id = row["gold_document_id"]

    # Retrieve again so we can inspect the chunks
    retrieved_docs = vector_store.similarity_search(q, k=1)

    print(f"\n----- QUESTION {i+1} / {len(questions_eval)} -----")
    print("Question:", q)
    print("Gold document ID:", gold_doc_id)

    # Retrieved document info
    if retrieved_docs:
        rid = retrieved_docs[0].metadata.get("id")
        print("Retrieved doc ID:", rid)
        print("Retrieval hit? ", "YES" if rid == gold_doc_id else "NO")

        print("\n--- Retrieved Context Chunk ---")
        print(retrieved_docs[0].page_content[:800], "...\n")  # print at most 800 chars
    else:
        print("!! No documents retrieved !!")

    # Model answer
    print("--- Model Answer ---")
    print(f"***THIS IS THE ANSWER: {rag_answers[i]}***")
    print("----------------------------------------------")
rag_preds = [extract_label_from_answer(ans) for ans in rag_answers]

valid_idx = [i for i, p in enumerate(rag_preds) if p is not None]
n_total = len(rag_preds)
n_valid = len(valid_idx)

rag_preds_valid = [rag_preds[i] for i in valid_idx]
rag_golds_valid = [rag_golds[i] for i in valid_idx]

tp = sum(1 for p, g in zip(rag_preds_valid, rag_golds_valid) if p == "yes" and g == "yes")
fp = sum(1 for p, g in zip(rag_preds_valid, rag_golds_valid) if p == "yes" and g == "no")
fn = sum(1 for p, g in zip(rag_preds_valid, rag_golds_valid) if p == "no" and g == "yes")

precision_yes = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_yes = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_yes = (
    2 * precision_yes * recall_yes / (precision_yes + recall_yes)
    if (precision_yes + recall_yes) > 0
    else 0.0
)

acc_rag = (
    sum(p == g for p, g in zip(rag_preds_valid, rag_golds_valid)) / n_valid
    if n_valid > 0
    else float("nan")
)

retrieval_acc = sum(doc_hit_flags) / n_total

print("\nRAG evaluation (batched, subset):")
print("  Total questions (subset):", n_total)
print("  Valid answers:", n_valid)
print("  Accuracy (on valid):", acc_rag)
print("  F1 (Yes as positive):", f1_yes)
print("  Retrieval hit rate (gold doc in retrieved):", retrieval_acc)

# ===================== BASELINE EVALUATION (NO CONTEXT, BATCHED) =====================

baseline_prompts = []
baseline_golds = []

baseline_template = (
    "<|im_start|>system\n"
    "You are a medical QA classifier.\n"
    "You will be given a medical yes/no question.\n"
    "Answer with 'Yes' or 'No' as the first word, then a short explanation.\n\n"
    "<|im_start|>user\n"
    "Question:\n{question}\n"
    "<|im_start|>assistant\n"
)


for _, row in questions_eval.iterrows():
    q = row["question"]
    gold = row["gold_label"].lower()
    prompt = baseline_template.format(question=q)
    baseline_prompts.append(prompt)
    baseline_golds.append("yes" if gold == "yes" else "no")

baseline_answers = lm.batch(baseline_prompts)

baseline_preds = [extract_label_from_answer(a) for a in baseline_answers]
valid_idx_b = [i for i, p in enumerate(baseline_preds) if p is not None]

n_total_b = len(baseline_preds)
n_valid_b = len(valid_idx_b)

baseline_preds_valid = [baseline_preds[i] for i in valid_idx_b]
baseline_golds_valid = [baseline_golds[i] for i in valid_idx_b]

tp_b = sum(1 for p, g in zip(baseline_preds_valid, baseline_golds_valid) if p == "yes" and g == "yes")
fp_b = sum(1 for p, g in zip(baseline_preds_valid, baseline_golds_valid) if p == "yes" and g == "no")
fn_b = sum(1 for p, g in zip(baseline_preds_valid, baseline_golds_valid) if p == "no" and g == "yes")

precision_yes_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
recall_yes_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
f1_yes_b = (
    2 * precision_yes_b * recall_yes_b / (precision_yes_b + recall_yes_b)
    if (precision_yes_b + recall_yes_b) > 0
    else 0.0
)

acc_baseline = (
    sum(p == g for p, g in zip(baseline_preds_valid, baseline_golds_valid)) / n_valid_b
    if n_valid_b > 0
    else float("nan")
)

print("\nBaseline (no context, batched, subset):")
print("  Total questions (subset):", n_total_b)
print("  Valid answers:", n_valid_b)
print("  Accuracy (on valid):", acc_baseline)
print("  F1 (Yes as positive):", f1_yes_b)


