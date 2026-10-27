from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import litellm
import json
from src.generator_code import generate_answer
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from deepeval.models import OllamaModel
eval_model = OllamaModel(
    model="llama3.1")
golden_path=r"C:\Users\butth\Documents\RAG_practice\RAG\goldenDataset\generatorData.json"
with open(golden_path) as f:
    golden=json.load(f)
test_case=[]
for g in golden:
    input=g["input"]
    context=g["context"]
    generated_answer=generate_answer(g["input"],context)
    test_case.append(
        LLMTestCase(
            input=g["input"],
            actual_output=generated_answer,
            retrieval_context=g["context"]
        )
    )
metrics=[
    FaithfulnessMetric(
        threshold=0.7,
        model=eval_model,
        include_reason=True,
        async_mode=False
    ),
    AnswerRelevancyMetric(
        threshold=0.7,
        model=eval_model,
        include_reason=True,
        async_mode=False
    )
]
evaluate(
    test_cases=test_case,
    metrics=metrics
)
