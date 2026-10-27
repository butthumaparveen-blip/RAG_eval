from deepeval import evaluate
import litellm
from deepeval.metrics import ContextualRecallMetric,ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import LiteLLMModel
import json
from src.retriever_py import retriever_system
from langchain_groq import ChatGroq
JUDGE_MODEL=LiteLLMModel(model="groq/llama3-70b-8192")
dataset_path=r"C:\Users\butth\Documents\RAG_practice\RAG\goldenDataset\goldendata.json"
def golden_dataset(path):
    with open(path,"r") as f:
        return json.load(f)
golden=golden_dataset(dataset_path)

test_case=[]
retriever=retriever_system()
for g in golden:
    context=retriever.invoke(g["query"])
    retrieved_context=[doc.page_content for doc in context]
    test_case.append(
        LLMTestCase(
            input=g["query"],
            expected_output=g["ideal_answer"],
            retrieval_context=retrieved_context,
            actual_output="(generator not evalated it is freeze)"
        )
    )

metrics=[
    ContextualPrecisionMetric(
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True
    ),
     ContextualRecallMetric(
            threshold=0.7,
            model=JUDGE_MODEL,
            include_reason=True
        )
    
]
evaluate(
    test_cases=test_case,
    metrics=metrics,
    max_concurrent_tasks=2
)

