# llm-project

## Jan 2025 Update
This project repo will stay active and once some work on pipelines and deployment is done, I will return to use multiple models in an ensemble method - one LLM model, staying with T5 for summary generation and another LLM model to compare the summary results. The Ensemble could be:

- One model (T5) generates the summary
- Another model (e.g., BLEURT, BERTScore, or a custom LLM) evaluates the quality of the generated summary against the human expert summary
- By combining the strengths of these two models, you effectively form an ensemble

## Project Task: Summarising News Articles: 
- ### Given a training set of news reports that vary in style and length, use an LLM (T5-small) to produce a summary of the long form text
    - I manually preprocessed and tokenized the dataset initially, before learning the document summarisation can be done end to end through an LLM with its own tokenizer needing to be used
    - I set on the [T5-small (Text-to-Text Transfer Transformer)](https://databasecamp.de/en/ml-blog/t5-model)
- ### Compare the LLM-generated summary to a human expert-provided summary (the reference summary) by ROUGE Test evaluation
    - Rouge 1, Rouge2 and RougeL compare individual words, consecutive word pairs and sentence-level similarities.
- ### Inference with spot-checking specific summaries from the dataset, and new unseen texts from news of the day
- ### Identify the hyperparameters and parameters we would want to adjust for a second round of fine-tuning, given the results of the first fine tuning with my dataset

## General Week 22-24 Structure:
- ### Week 22-23
    - Decided on text summarisation (possibly legal document, then confirmed to news summarisation for phase 1), online literature review of some papers (see background-research.md).
    - Found the dataset (link below), practiced tokenization and preprocessing manually using NLP tools (NLTK, remove stopwords etc). Saved preprocessed data
- ### Week 23-24
    - Selected T5-small model for GPU/RAM considerations, first run with just 25% of the data, preprocessing from the raw text (I understand that you need to use the LLM's own tokenizer in almost all cases, so my manual preprocessing wouldn't be appropriate here).
    - Ran the preprocessed, tokenized data in T5-small, interpret ROUGE score results, manually inspect some outputs and inference with new unseen inputs.
    - Considered hyperparameter changes to improve performance (output min length, prompt engineering the input for the T5, learning rate
    - Deployment and ethical considerations discussed in the notebook. The main issue IMO is around training on "news articles" which could be anything from matter-of-fact Reuters/BBC "just the facts" or internet forum gossiping about celebrities in controversial news which was found in some of the training data. There is risk of factually incorrect claims about individuals in certain events which could amount to libel.

## The Dataset: [Alex Fabbri Multi-News, from HuggingFace](https://huggingface.co/datasets/alexfabbri/multi_news)

- 45,000 records for training, 5.6k for validation (hyperparam tuning) and 5.6k for testing. Fields are the document itself and a summary that was professionaly written by a human editor.
    - There is a large linguistic variety in the training documents. Some are pure opinion/gossip news releases, some matter-of-fact official reporting, some travel journalism type casual fare.

- I uncovered the lost-in-the-middle issue with doc summarisation and that ideal "chunks" of documents using models like GPT4 is about 4 - 6k tokens (around 1,000 - 1,500 words) as input to summarise an input text. Our News articles are likely not going to be that long and we could preprocess out the extremely long outliers, or truncate the input texts.

- Once we have the model's generated summary, I will use ROUGE evaluation to compare it to the expert 'summary' column. That would make the goal of having model-generated summaries as close to the human experts as possible.
    - See Article below - Extract-then-Evaluate Method For Future phase potentially testing an improvement in the performance.

- This is going to be very memory/RAM intensive with the full dataset so I did my **first full run with only 25% of the dataset** from Hugging Face.

## The pre-trained Model:

- I trained the **T5-small model** on 25% of the original data from the source linked above. I chose T5 as it is built for text-to-text / sequence-to-sequence NLP tasks like summarisation and question answering. The prompt to the LLM can also be easily modified e.g. "Summarise this text:" "Summarise this passage in at least 6 sentences:" or something similar.
- T5 is said to need sizeable amounts of data for fine-tuning so I will be mindful of this and try 50% of the data for better results next time.
- Almost all settings/parameters were default other than smaller batch size to manage compute / time (also the motive for the reduced dataset) to complete a successful run and gather baseline metrics, before we inspected summarisation results and where we could improve.

## Results / Performance Metrics:

- Over each consecutive Epoch (4 total):
    - Training Loss (around 2.9) and Validation Loss (around 2.6) gradually declined, with Validation Loss slightly lower than Training
    - ROUGE scores gently increasing

- ROUGE1 Score around 0.15 in first full training, ROUGE2 around 0.05. These show that the model's summarisations have a 3x better overlap with single words than consecutive word pairs with the reference summary. Model is probably capturing the key information with not much fluency.

- GEN LEN was 19 tokens in the first default training. This is **highly likely** to be impacting the poor ROUGE scores when typical reference summaries are 150-300 tokens. **For improved ROUGE scores, adding a minimum_length token output is quite likely to help.**

- Inference: 
    - Human inspection / Spot checking the model output against reference showed the lost-in-the-middle problem anecdotally - ignoring of entire sentences in middle of an article, focusing on introduction and small context near the end of the text input.

## Hyperparameters:

At this time I have not re-run the fine tuned model with different hyperparameters as I would like time to decide which parameters to change for the next training, but it is very likely I will focus on:

- **Learning rate:** generally is something we should experiment with, perhaps a lower learning rate would help summarise given the nature of news being highly varied in writing and journalistic style, subject matter and vocabulary the model will come across.
- Max_length and Min_length. I strongly suspect that larger min_length would force longer than 19 token GEN LEN outputs. This would noticeably improve the ROUGE Score, initially by virtue of the generated summary having more n-grams that could match the reference.
- Related to length, length_penalty, encouraging longer or shorter outputs.
- **Change the T5 prompt:** At this time I have the prompt simply as "Summarize:" but this prompt could be used to ask for longer summaries which also solves the issue of small GEN LEN which contributes to underwhelming ROUGE scores.

## Relevant Links:

- My [small-multi-news-model](https://huggingface.co/tjjdoherty/small-multi-news-model)
- ![model-card](images/image-3.png)

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
