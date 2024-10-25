# llm-project: Document Summarisation

## Project Journal

## Goals

- ### Summarising News Articles: 
    - ### Given a training set of longer form news reports, use an LLM to produce a summary of the long form
    - ### Compare the LLM-generated summary to a human expert-provided summary also supplied in the training data
    - ### Inference with spot-checking specific summaries from the dataset, and new unseen texts

## General Structure:

- Provide a text input e.g. News Article or transcript (from a known dataset)
    - Previous: Remove stop words, lower casing, preprocessing including lemmatization. This is done automatically with LLM's own tokeniser but I also worked this manually in llm_project_1_preprocess
- Use the dataset as training data on a pretrained model and produce a summary of the news reports
- Compare the LLM-generated summary of the news report to the human summary provided using ROUGE Test evaluation

## Strategy with the Dataset: [Alex Fabbri Multi-News, from HuggingFace](https://huggingface.co/datasets/alexfabbri/multi_news)

- 45,000 records for training, 5.6k for validation (hyperparam tuning) and 5.6k for testing. Fields are the document itself and a summary that was professionaly written by a human editor.
    - There is a large linguistic variety in the training documents. Some are pure opinion/gossip news releases, some matter-of-fact official reporting, some travel journalism type casual fare.

- I uncovered the lost-in-the-middle issue with doc summarisation and that ideal "chunks" of documents using models like GPT4 is about 4 - 6k tokens (around 1,000 - 1,500 words) as input to summarise an input text. Our News articles are likely not going to be that long and we could preprocess out the extremely long outliers, or truncate the input texts.

- Once we have the model's generated summary, I will use ROUGE evaluation to compare it to the expert 'summary' column. That would make the goal of having model-generated summaries as close to the human experts as possible.
    - See Article below - Extract-then-Evaluate Method For Future phase potentially testing an improvement in the performance.

- This is going to be extremely memory/RAM intensive with the full dataset so I did my **first full run with only 25% of the dataset** from Hugging Face.

## Results:

- I trained the T5 model on 25% of the original data from the source linked above. Dynamic padding of the tokenized sentences was done, other than this most settings/parameters were default other than smaller batch size to manage compute / time (also the motive for the reduced dataset)

- Over each consecutive Epoch (4 total):
    - Training Loss and Validation Loss gradually declined, with Validation Loss slightly lower than Training
    - ROUGE scores gently increased 

- ROUGE1 Score around 0.15 in first full training, ROUGE2 around 0.05. These show that the model's summarisations have a 3x better overlap with single words than consecutive word pairs with the reference summary. Model is probably capturing the key information with not much fluency.

- GEN LEN was 19 tokens in the first default training. This is **highly likely** to be impacting the poor ROUGE scores when typical reference summaries are 150-300 tokens. **For improved ROUGE scores, adding a minimum_length token output is quite likely to help.**

- Inference: 
    - Human inspection / Spot checking the model output against reference showed the lost-in-the-middle problem anecdotally - ignoring of entire sentences in middle of an article, focusing on introduction and small context near the end of the text input.


## Learnings from initial research: background-research.md for more.

- I initially had some interest on looking at long legal document summarisation and the initial online research I did focused on that, before I narrowed the initial scope down to news article summarisation on a publicly available dataset on Hugging face. This still helped because the current state of LLM based text summarisation has the most pronounced issues with long, legal documents.

- One of the biggest challenges in document summarisation is the **Lost-in-the-Middle** problem: Summarisation from middle parts of input documents tends to be worse than from the beginning and end.
    - It's unlikely news articles will get to those very long lengths, but Chain Summarisation could help even if they did. 
        - Chain summarisation is when a large text is split into chunks, a summary is generated for each chunk and these intermediate summaries are then combined and further summarized, iteratively producing increasingly concise versions of the text. 
        - Each step feeds the next, giving a collective summary by the end.
    - Segments of 6 to 9 pages in length (4 to 6k token input) is a good starting point. This aligns with lots of models limited token inputs which is about 4 - 16k when using GPT4, which provides good results in a number of similar tasks without taking the time to fine tune the pretrained model.

- You can specify a **token output limit** in your training setup - a 20:1 ratio of input to output tokens is a good place to start.
    - Don't chase a big token limit LLM - the research shows the larger token limit 'upgrade' model degrades nearly identically in performance of its 'junior' sibling and longer documents consistently give poorer performance. You can also end up with models repeating sentences in summarisation just to make up the larger token output allowance.

- Look at the different types of summarisations you could do:
    - Extractive vs Abstractive
    - In Abstractive: free form summary or concept/principle based? For summarising news, abstractive is perfectly fine as there are many ways to summarise news articles, and quotes could still be taken extractively in between.

- **I will start with a T5-small model**. Once I have familiarised with managing GPU resources and want to dedicate more compute to a more powerful model for better results, I could use ChatGPT 4 which has generally very good performance with the right prompt sequence and explicit structures. This is true without fine-tuning or custom GPT training - the added benefit there appears marginal at this time when you trade off further work and compute needed to train the GPT in a specific domain.

- Extract-then-Evaluate has gotten much better alignment between LLM and human expert evaluation of model-derived summaries. Instead of comparing the original document (x) to the model-generated summary of it, you extract key semantically important sentences (x') and compare it to that, which costs less compute too.