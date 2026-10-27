from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
template=ChatPromptTemplate.from_template(
    """
you are expert AI assistant
1- provide the answer of the question from given context
2- if the context is not enough to answer the question just say answer is not in context
3- and cite the sources from which the answer is taken
question {question}
context {context}
"""
) 
llm=ChatGroq(model="openai/gpt-oss-120b")
parser=StrOutputParser()
chain=template | llm | parser
def generate_answer(query:str,context:str):
    output=chain.invoke({
        "context":context,
        "question":query
    })
    return output
if __name__=="__main__":
   context="A neural network is a machine learning model inspired by the structure and function of the human brain. It consists of interconnected artificial neurons arranged in layers: an input layer that receives raw data, one or more hidden layers that process information through weighted connections and activation functions, and an output layer that delivers the final result. During training, the network adjusts its internal weights and biases via a process called backpropagation to minimize errors, allowing it to recognize complex patterns, classify data, and make predictions. These models power modern artificial intelligence applications such as image recognition, natural language processing, and speech-to-text transcription"
   question="explain neural network in simple word that 6 year old child can understand it"
   print(generate_answer(question,context))