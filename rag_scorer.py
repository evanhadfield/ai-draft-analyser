# rag_scorer.py
import os
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Pinecone as LangchainPinecone

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

  # Create the prompt template
  template = """
  Based on the following context and criteria, provide scores for Justification and Essential Elements, along with reasons and recommendations:

  Context: {context}

  Justification Criteria:
  5 – Outstanding: Provides a thorough and compelling explanation of the need for policy change, with detailed context and strong evidence supporting the proposal.
  4 – Exceeds Expectations: Clearly explains the need for the policy change with good contextual information and adequate supporting evidence.
  3 – Meets Expectations: Satisfactorily explains the need for the policy change with basic contextual information, but some details or evidence may be lacking.
  2 – Needs Improvement: Provides a vague explanation of the need for policy change with minimal context and little supporting evidence.
  1 – Unacceptable: Fails to explain the need for policy change or provide any contextual information or evidence.

  Essential Elements Criteria:
  5 – Outstanding: All changes or new provisions are clearly outlined, logically structured, and thoroughly detailed.
  4 – Exceeds Expectations: Changes or new provisions are well-outlined and detailed, with minor areas for improvement in clarity or structure.
  3 – Meets Expectations: Changes or new provisions are satisfactorily outlined but lack some detail or clarity.
  2 – Needs Improvement: Changes or new provisions are vaguely outlined and lack significant detail or clarity.
  1 – Unacceptable: Changes or new provisions are not outlined or are extremely unclear.

  Please provide:
  1. Justification Score (1-5)
  2. Reason for Justification Score
  3. Recommendations for improvement of Justification Score
  4. Essential Elements Score (1-5)
  5. Reason for Essential Elements Score
  6. Recommendations for improvement of Essential Elements Score

  Format your response as follows:
  Justification Score: [score]
  Justification Reason: [reason]
  Justification Improvement Recommendations: [recommendations]
  Essential Elements Score: [score]
  Essential Elements Reason: [reason]
  Essential Elements Improvement Recommendations: [recommendations]
  """

  prompt = PromptTemplate(
      template=template,
      input_variables=["context"]
  )

  # Create the RetrievalQA chain
  qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=vectorstore.as_retriever(),
      chain_type_kwargs={"prompt": prompt}
  )

  return qa_chain

def get_scores_and_recommendations(qa_chain):
  # This query is designed to retrieve relevant information from the document
  query = "Provide a summary of the policy change, its justification, and the outlined changes or new provisions."
  
  # # Count tokens in the query
  # query_tokens = count_tokens(query)
  # print(f"Input query tokens: {query_tokens}")

  response = qa_chain.invoke(query)

  # Extract the result from the response
  result = response['result']

  # Count tokens in the result
  # result_tokens = count_tokens(result)
  # print(f"Output response tokens: {result_tokens}")

  return result
def run_rag_scorer():
    rag_system = create_rag()
    result = get_scores_and_recommendations(rag_system)
    return result