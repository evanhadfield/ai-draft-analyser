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
      prefix="""You are an AI assistant tasked with evaluating policy drafts. Based on the given context and criteria, provide scores and reasons for Justification, Essential Elements, Impact Assessment, Comprehension, Feedback Collection, and Translations. Format your response exactly as shown in the examples, with each score on its own line and each reason on a new line immediately following its corresponding score.

Justification Criteria:
(Whether the need for the policy change and the context in which it is being proposed has been adequately explained) 
5 - Outstanding: Provides a thorough and compelling explanation of the need for policy change, with detailed context and strong evidence supporting the proposal.
4 - Exceeds Expectations: Clearly explains the need for the policy change with good contextual information and adequate supporting evidence.
3 - Meets Expectations: Satisfactorily explains the need for the policy change with basic contextual information, but some details or evidence may be lacking.
2 - Needs Improvement: Provides a vague explanation of the need for policy change with minimal context and little supporting evidence.
1 - Unacceptable: Fails to explain the need for policy change or provide any contextual information or evidence.

Essential Elements Criteria:
(Whether the changes being introduced or new provisions being introduced are clearly outlined in the draft) 
5 - Outstanding: All changes or new provisions are clearly outlined, logically structured, and thoroughly detailed.
4 - Exceeds Expectations: Changes or new provisions are well-outlined and detailed, with minor areas for improvement in clarity or structure.
3 - Meets Expectations: Changes or new provisions are satisfactorily outlined but lack some detail or clarity.
2 - Needs Improvement: Changes or new provisions are vaguely outlined and lack significant detail or clarity.
1 - Unacceptable: Changes or new provisions are not outlined or are extremely unclear.

Impact Assessment Criteria:
(Whether the Financial, Social, or Environmental impact that the proposed draft may have has been adequately analysed and stated) 
5 - Outstanding: Provides comprehensive and detailed analysis of financial, social, and/or environmental impacts, supported by robust data.
4 - Exceeds Expectations: Provides good analysis of potential impacts with adequate data, though some areas could be more detailed.
3 - Meets Expectations: Provides satisfactory analysis of impacts but lacks depth or detail.
2 - Needs Improvement: Provides minimal analysis of impacts with insufficient data or detail.
1 - Unacceptable: Fails to provide any meaningful analysis of impacts.

Comprehension Criteria:
(Whether efforts have been made to ensure that the draft can be easily understood by the average reader who may or may not have domain expertise in the subject of the draft) 
5 - Outstanding: Draft is exceptionally clear, easy to understand, and accessible to readers with or without domain expertise.
4 - Exceeds Expectations: Draft is clear and understandable, with minor areas for improvement in accessibility for the average reader.
3 - Meets Expectations: Draft is generally understandable but may have some areas that are unclear or complex for the average citizen to understand.
2 - Needs Improvement: Draft is difficult to understand and lacks clarity, requiring significant effort to comprehend.
1 - Unacceptable: Draft is highly confusing and incomprehensible to most readers.

Feedback Collection Criteria:
(Whether multiple avenues to respond to the consultation were provided to the stakeholders to enable responses from a variety of citizens, who may or may not be digitally enabled)
5 - Outstanding: Multiple and varied avenues for response, easily accessible to all citizens, including those not digitally enabled (4 or more methods)
4 - Exceeds Expectations: Several avenues for response, accessible to most citizens (3 methods)
3 - Meets Expectations: Basic avenues for response, accessible to most citizens but with some limitations (2 methods)
2 - Needs Improvement: Few avenues for response, not easily accessible to all citizens (single method)
1 - Unacceptable: Minimal or no avenues for response, not accessible to most citizens (no clear method of response outlined)

Translations Criteria:
(Whether the consultation was able to reach larger segments of the population by providing translations in multiple languages)
5 - Outstanding: Provides translations in multiple languages, reaching a wide and diverse population (4 or more languages including sign language, braille etc)
4 - Exceeds Expectations: Provides translations in several languages, reaching a diverse population (provides translations in English, Hindi and the state language or any 3 languages)
3 - Meets Expectations: Provides translations in a few languages, reaching an expected population (provides translations in 2 languages)
2 - Needs Improvement: Provides minimal translations, reaching a limited population (Provides no translations but issues instructions to sub departments/states to issue translations)
1 - Unacceptable: Provides no translations, failing to reach a diverse population, single language
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
an assessment of its clarity and comprehensibility,
the feedback collection methods, and the availability of translations."""
  
  response = qa_chain.invoke(query)

  # Extract the result from the response
  result = response['result']

  return result

def run_rag_scorer():
  rag_system = create_rag()
  result = get_scores_and_recommendations(rag_system)
  return result