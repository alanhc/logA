---
title: inferencing-llm
date: 2024-07-04
tags:
  - llm
updated: 2024-07-04
up:
  - "[[llm]]"
---
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ## mac
## 安裝
- `pip install mlx-lm`
### 使用 mac cli
- `python -m mlx_lm.generate --model mlx-community/Phi-3-mini-4k-instruct-4bit  --prompt "how are you"`
- 

```
import bs4

from langchain import hub

from langchain_community.document_loaders import WebBaseLoader

from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
```

```
import getpass

import os

  

os.environ["OPENAI_API_KEY"] = "sk-proj-"#getpass.getpass()
from langchain_community.document_loaders import PyPDFLoader

  

loader = PyPDFLoader("files/替代役管理作業規定.pdf")

pages = loader.load_and_split()

```

```
from langchain_community.llms.mlx_pipeline import MLXPipeline

  

llm = MLXPipeline.from_model_id(

"mlx-community/quantized-gemma-2b-it",

# pipeline_kwargs={"max_tokens": 512, "temp": 0.1},

pipeline_kwargs={"temp": 0.1},

)
```

```
from huggingface_hub import login

login(token="")
```
```
import os

  
  

docs = loader.load()

  

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

  

# Retrieve and generate using the relevant snippets of the blog.

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

  

def format_docs(docs):

return "\n\n".join(doc.page_content for doc in docs)

  
  

rag_chain = (

{"context": retriever | format_docs, "question": RunnablePassthrough()}

| prompt

| llm

| StrOutputParser()

)
```

```
question = "替代役管理作業規定是依據甚麼法律?"

rag_chain.invoke(f"{question} ")
# ' 根據上述法律，替代役管理作業規定主要依據兵役法第 26條。'
```

## Ollama 
-  [使用 Ollama 執行 TAIDE 的 TAIDE-LX-7B-Chat-4bit 大語言模型](https://blog.miniasp.com/post/2024/04/17/Run-TAIDE-LX-7B-Chat-4bit-model-in-Ollama)
	- `ollama create taide-lx-7b-chat-4bit:latest -f Modelfile`
-  [介紹好用工具：Ollama 快速在本地啟動並執行大型語言模型](https://blog.miniasp.com/post/2024/03/04/Useful-tool-Ollama)
[透過Ollama在本機電腦執行大型語言模型（LLM）：Windows與VS Code篇](https://swf.com.tw/?p=1952)
## Ref
- 
