---
title: ResearchFlow AI
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Cited, fact-checked answers from your own documents
---

# ResearchFlow AI

Ask questions about PDFs and text files and get answers where every claim links to the exact
passage it came from. A LangGraph agent retrieves passages (FAISS + sentence-transformers),
drafts an answer with Gemini, fact-checks it against the evidence, retries with a new search
when evidence is thin, and says so honestly when the documents don't contain the answer.

The demo starts with two collections: three short fictional reports (**Default**) and ten
foundational AI papers from arXiv (**AI research papers**).

Source code and documentation: see the project repository.
