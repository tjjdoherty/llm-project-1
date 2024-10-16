# llm-project
LLM Project for Document Summarization

## LLM Project Journal

## Goals

- ### Summarizing lengthy legal documents / decisions into key point summaries
- ### Summarizing lengthy business RFPs and requirements into key points

### General Structure:

- Provide a word / PDF document or text file
- Removes stop words
- Create topics 
- Return key topics
- Challenge - could it have some text generation? e.g. "The main topic is about [Requirements 1] and 2 but [Requirement 3] has [context]


### Sources for discussion:

Article 1: [A very Long Discussion of Legal Document Summarization using LLMs](https://www.linkedin.com/pulse/very-long-discussion-legal-document-summarization-using-leonard-park/)
Key Takeaways
- **Large Documents are hard to summarize due to Context Length Limitation**. There are **limited token inputs** for most models, and these are how the model operate, most are around 4.1k - 16.4k (depending on model, as of August 2023 writing of this article) which is about 6 - 26 pages of text. If the input document creates more tokens that the model limit allows, then sections need to be removed before the model can process it.


ChatGPT "What are some of the limitations of large document summarizer tools built by LLMs?"
Bullet point summaries:
- **Token Limitations:** Referred to in Article 1: what is the token limit of the model you're using? It is likely going to lead to a loss of context or information in lengthy documents e.g. business RFPs/Tenders or legal case outcomes.

- **Context Truncation:** LLM's typically use a sliding window of tokens to maintain context. The problem is that the context is lost from early in the document when it gets to working on later documents. The way that an RFP or legal document is laid out typically is demarcated with sections that deal in detail with some issue before moving on to the next. (Human readers know and expect this structure and can refer back and forth to sections but the model may 'forget' the earlier context when working on later sections.)

- **Abstraction vs. Extractiveness:** LLMs tend to be abstractive summarizers. This means generating new sentences to summarize content rather than extract the important sentences (how would it know which sentences are the most important?). This means summaries can be vague and miss specific important details. (Summarization may prioritize fluency over factuality which distorts messages - like sophistry in humans!).

- **Inconsistencies in Summary:** Ultimately LLMs are probabilistic text generators so summaries will be inconsistent even across the same document depending on the starting point, token context or even a slight rewording.

- **Token Misinterpretation:** Misinterpretation of specialized vocabulary and technical terms is a real issue in domain-specific applications. Legal may be a huge issue - a lawyer reading that **consideration** occurred has a very specific meaning that the layperson gave consideration to an issue does not capture, and the former may have major consequences on the lawyer's decision-making.

- **Bias and Hallucination:** This is a symptom of the above issues. The model simply generates factually incorrect content that wasn't in the original text.