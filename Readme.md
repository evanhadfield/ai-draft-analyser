# Civis AI Draft Analyzer

This repository contains the Civis AI-powered Draft Analyzer, designed to provide various scores after analyzing PDF documents. The objective is to deliver accurate scores, along with explanations for the scores and recommendations for improvement.

## Platform Architecture

1. **Upload or Input PDF**:
   - Upload the PDF file or enter the URL of the PDF.
2. **Document Processing**:
   - Split the document into chunks using a text splitter.
   - Upload the chunks into the Pinecone Vector Database.
3. **Embedding and Scoring**:
   - Use OpenAI embeddings for storing data in the vector database.
   - Run the RAG (Retrieval-Augmented Generation) scorer to generate scores.
   - Optionally, enhance the score further using prompt engineering and few-shot examples from Langchain.

## Tech Stack

- **Languages/Frameworks**: Python, Streamlit
- **Database**: Pinecone Vector Database

## System Requirements

- **Python**: Version 3.0 and above

## Deployment

- **Platform**: Streamlit

## Credentials

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `PINECONE_API_KEY`

## Running Locally

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>

   ```

2. Set Up Environment:

Create a `.env` file and provide the necessary credentials as mentioned above.

Create a virtual environment and install all dependencies from `requirements.txt`

Run the Application:
`streamlit run app.py`

## Prompt used to Check Score

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
Assess whether the consultation was made accessible to non-English-speaking audiences by providing translations. If the consultation was only available in English, the translation score should be 1. If translations were provided in other languages in addition to English, the translation score should be based on the total number of languages available (e.g., if 3 languages are present, including English, the score should be 3)
5 - Outstanding: Provides translations in multiple languages, reaching a wide and diverse population (4 or more languages including sign language, braille etc)
4 - Exceeds Expectations: Provides translations in several languages, reaching a diverse population (provides translations in English, Hindi and the state language or any 3 languages)
3 - Meets Expectations: Provides translations in a few languages, reaching an expected population (provides translations in 2 languages)
2 - Needs Improvement: Provides minimal translations, reaching a limited population (Provides no translations but issues instructions to sub departments/states to issue translations)
1 - Unacceptable: Provides no translations, failing to reach a diverse population, single language

## Policy Examples

As mentioned in exapmple.py
