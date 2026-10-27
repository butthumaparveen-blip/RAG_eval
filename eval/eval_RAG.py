from src.RAG import RAG_building
from deepeval import evaluate
import json
from langchain_ollama import ChatOllama
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric,ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
JUDGE_MODEL=OllamaModel(
    model="llama3.1"
)
dataset_path=r"C:\Users\butth\Documents\RAG_practice\RAG\goldenDataset\generatorData.json"
with open(dataset_path) as f:
    goldens=json.load(f)
RAG=RAG_building()
test_cases=[]
for g in goldens:
    input=g["input"]
    data=RAG.invoke(input)
    test_cases.append(
        LLMTestCase(
            input=input,
            retrieval_context=data["context"],
            actual_output=data["generated_answer"]
        )
    )

metrics=[
    FaithfulnessMetric(
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True
    ),
    AnswerRelevancyMetric(
       threshold=0.7,
       model=JUDGE_MODEL,
       include_reason=True 
    ),
    ContextualRelevancyMetric(
        threshold=0.7,
        model=JUDGE_MODEL,
        include_reason=True
    )
]

evaluate(
    test_cases=test_cases,
    metrics=metrics
)


