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
- Challenge - could it have some generative properties? e.g. "This RFP is about [Requirements 1 and 2] but [Requirement 3] has [context] which is important, or may be ignored depending on [Challenge 1]"


### Sources for discussion:

Article 1: [A very Long Discussion of Legal Document Summarization using LLMs (Aug 2023)](https://www.linkedin.com/pulse/very-long-discussion-legal-document-summarization-using-leonard-park/)
Key Takeaways
- **Large Documents are hard to summarize due to Context Length Limitation**. There are **limited token inputs** for most models, and these are how the model operate, most are around 4.1k - 16.4k (depending on model, as of August 2023 writing of this article) which is about 6 - 26 pages of text. If the input document creates more tokens that the model limit allows, then sections need to be removed before the model can process it. 
    - This is referred to at the **context** by some, e.g. **8k context** or 4k context.
    - Some models have much larger limits e.g. 32k, Clause v2 has a 100k token limit but it doesn't actually change much for this author.

- **Currently, all model performance degrades as the amount of text you provide to them increases**.
    - This is **Particularly true in the middle of the text** [Lost in the Middle, Nov 2023](https://arxiv.org/abs/2307.03172) "We observe that performance is often highest when relevant information occurs at the beginning or end of the input context, and signiciatly degrades when models must access relevant information in the middle of long contexts, even for explicitly long-context models". So optimal performance in current models meant moderating the input to the model with each API call.
    - See image below, where GPT 3.5 and Claude 1.3 9k and 100k both degrade nearly identically. See images below.
    - Retrieval of information is better in the beginning and end of the context window.
    - A really noticeable trend here is that **even when these models have substantially larger token limits (GPT 4k or 16k, Claude 8k or 100k) their performances are nearly identical.** All the models have better performance at the beginning and end of the context window, and a degradation in the middle. Larger context models are more convenient but optimal performance involves smaller chunks of information.

    ![Accuracy vs Number of Documents in Input Context, by Model](image.png)
    ![Document position vs Accuracy of answer retrieval](image-1.png)

- **LLMs struggle with domain specificity and are "reality agnostic"**.
    - Great for general knowledge and some reasoning.



Article 2: [Legal Summarization through LLMs: The PRODIGIT Project (Aug 2023)](https://arxiv.org/pdf/2308.04416)



Article 3: [Unlocking Legal Insights (OpenAI's LLM and LangChain)](https://www.velotio.com/engineering-blog/unlocking-legal-insights-effortless-document-summarization-with-openais-llm-and-langchain#:~:text=For%20each%20document%2C%20we%20employ,translating%20legalese%20into%20understandable%20insights.)



ChatGPT "What are some of the limitations of large document summarizer tools built by LLMs?"
Bullet point summaries:
- **Token Limitations:** Referred to in Article 1: what is the token limit of the model you're using? It is likely going to lead to a loss of context or information in lengthy documents e.g. business RFPs/Tenders or legal case outcomes. 
    - **Tokenization rules might have hyperparameters but the rules are often predefined by the model architecture and the training process, so it is not easily tunable post-training**. For example, you couldn't tune the token limit without retraining the model with a larger or different architecture. 
    - **For most pre-built LLMs, tokenization is set in stone unless you build your own tokenizer.**

- **Context Truncation:** LLM's typically use a sliding or **rolling window of tokens** to maintain context. **When the token limit is reached, most models simply discard the earliest tokens rather than removing the least frequently found tokens, retaining the tokens closer to where the model is currently generating predictions.** Context is lost from early in the document when it gets to working on later documents. 
    - In some contexts, splitting could be better, but usually the recency of tokens leads to more coherent outputs. The way that an RFP or legal decision document is laid out is often demarcated with sections that specifically deal in detail and preserve information here with one issue before moving on to the next. Human readers know and expect this structure but the model may 'forget' the earlier context when working on later sections. 
    - For this, you could use 'splitting' or generate section-specific summaries or answers before combining them into a broader document summary. **longformer or BigBird are better suited for handling very long documents and incorporate strategies for retaining more distant context** which is critical for such structured documents

- **Abstraction vs. Extractiveness:** LLMs tend to be abstractive summarizers. This means generating new sentences to summarize content rather than extract the important sentences (how would it know which sentences are the most important?). This means summaries can be vague and miss specific important details. 
    - Summarization may unintentionally prioritize fluency over factuality which distorts messages. See Token Misinterpretation below.

- **Token Misinterpretation:** Misinterpretation of specialized vocabulary and technical terms is a real issue in domain-specific applications. 
    - Legal may be a huge issue - a lawyer reading that **consideration** occurred has a very specific meaning that the layperson gave consideration to an issue does not capture, and the former may have major consequences on the lawyer's decision-making. 
    - Same with the physical sciences - **organic** is a vague term that has been co-opted by health food marketers but has a specific definition in chemistry and a chemist will act on that word differently than a layperson. 
    - This happens because the model has more general training on common language patterns than on rare or technical terms. There are just more instances of the general-purpose and layperson context of the word; **LLMs have been trained on massive amounts of general-purpose text, where common English words and connective phrases appear frequently.**

- **Inconsistencies in Summary:** Ultimately LLMs are probabilistic text generators so summaries will be inconsistent even across the same document depending on the starting point, token context or even a slight rewording.

- **Bias and Hallucination:** This is a symptom of the above issues. The model simply generates factually incorrect content that wasn't in the original text.