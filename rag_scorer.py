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

  # Create the few-shot prompt template with added criteria definitions
  few_shot_prompt = FewShotPromptTemplate(
      examples=POLICY_EXAMPLES,
      example_prompt=example_prompt,
      prefix="""You are an AI assistant tasked with evaluating policy drafts. Based on the given context and criteria, provide scores and reasons for Justification, Essential Elements, Impact Assessment, and Comprehension. Format your response exactly as shown in the examples, with each score on its own line and each reason on a new line immediately following its corresponding score.

Justification Criteria:
5 - Outstanding: Provides a thorough and compelling explanation of the need for policy change, with detailed context and strong evidence supporting the proposal.
4 - Exceeds Expectations: Clearly explains the need for the policy change with good contextual information and adequate supporting evidence.
3 - Meets Expectations: Satisfactorily explains the need for the policy change with basic contextual information, but some details or evidence may be lacking.
2 - Needs Improvement: Provides a vague explanation of the need for policy change with minimal context and little supporting evidence.
1 - Unacceptable: Fails to explain the need for policy change or provide any contextual information or evidence.

Essential Elements Criteria:
5 - Outstanding: All changes or new provisions are clearly outlined, logically structured, and thoroughly detailed.
4 - Exceeds Expectations: Changes or new provisions are well-outlined and detailed, with minor areas for improvement in clarity or structure.
3 - Meets Expectations: Changes or new provisions are satisfactorily outlined but lack some detail or clarity.
2 - Needs Improvement: Changes or new provisions are vaguely outlined and lack significant detail or clarity.
1 - Unacceptable: Changes or new provisions are not outlined or are extremely unclear.

Impact Assessment Criteria:
5 - Outstanding: Provides comprehensive and detailed analysis of financial, social, and/or environmental impacts, supported by robust data.
4 - Exceeds Expectations: Provides good analysis of potential impacts with adequate data, though some areas could be more detailed.
3 - Meets Expectations: Provides satisfactory analysis of impacts but lacks depth or detail.
2 - Needs Improvement: Provides minimal analysis of impacts with insufficient data or detail.
1 - Unacceptable: Fails to provide any meaningful analysis of impacts.

Comprehension Criteria:
5 - Outstanding: Draft is exceptionally clear, easy to understand, and accessible to readers with or without domain expertise.
4 - Exceeds Expectations: Draft is clear and understandable, with minor areas for improvement in accessibility for the average reader.
3 - Meets Expectations: Draft is generally understandable but may have some areas that are unclear or complex for the average citizen to understand.
2 - Needs Improvement: Draft is difficult to understand and lacks clarity, requiring significant effort to comprehend.
1 - Unacceptable: Draft is highly confusing and incomprehensible to most readers.

Here are some examples:

""",
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