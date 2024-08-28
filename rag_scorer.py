# rag_scorer.py
import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from examples import POLICY_EXAMPLES

def create_rag():
  # Initialize embeddings (still using OpenAI)
  embeddings = OpenAIEmbeddings()

  # Create a Langchain wrapper for Pinecone
  vectorstore = LangchainPinecone.from_existing_index(
      index_name="civis",
      embedding=embeddings,
      text_key="text",
      namespace=""
  )

  # Initialize the Anthropic language model
  llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

  # Create the example prompt
  example_prompt = PromptTemplate(
      input_variables=["context", "scores"],
      template="Context: {context}\n\nScores:\n{scores}"
  )

  # Create the few-shot prompt template
  few_shot_prompt = FewShotPromptTemplate(
      examples=POLICY_EXAMPLES,
      example_prompt=example_prompt,
      prefix="You are an AI assistant tasked with evaluating policy drafts. Based on the given context and criteria, provide scores and reasons for Justification, Essential Elements, Impact Assessment, and Comprehension. Format your response exactly as shown in the examples, with each score on its own line and each reason on a new line immediately following its corresponding score.\n\nHere are some examples:\n\n",
      suffix="\nNow, evaluate the following policy draft:\n\nContext: {context}\n\nProvide scores and reasons in the exact same format as the examples, without any introductory text.",
      input_variables=["context"]
  )
  # Create the RetrievalQA chain
  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectorstore.as_retriever(),
      chain_type_kwargs={"prompt": few_shot_prompt}
  )

  return qa_chain

def get_scores_and_recommendations(qa_chain):
  # This query is designed to retrieve relevant information from the document
  query = """Provide a summary of the policy change, its justification, 
  the outlined changes or new provisions, its potential impacts,
  and an assessment of its clarity and comprehensibility."""
  
  response = qa_chain.invoke(query)

  # Extract the result from the response
  result = response['result']

  return result

def run_rag_scorer():
  rag_system = create_rag()
  result = get_scores_and_recommendations(rag_system)
  return result