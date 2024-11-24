## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
The growing volume of research articles across various domains makes it increasingly difficult for users to retrieve and synthesize information efficiently. Traditional search systems often retrieve results that lack relevance or context. A multidocument retrieval agent powered by LlamaIndex addresses this problem by enabling a modular, efficient, and context-aware query-response system, providing users with accurate and concise information from multiple sources.

### DESIGN STEPS:

#### Step 1: Prepare the Documents**
Collect research articles (PDFs or text files) and preprocess them to extract clean text.
Split the text into manageable chunks for efficient indexing.

#### Step 2: Build the Retrieval System**
Create indexes for the document chunks using **LlamaIndex** and vector embeddings.
Set up tools for semantic search (vector tool) and summarization (summary tool).

#### Step 3: Implement and Query the Agent**
Combine the tools into a retrieval agent using **LlamaIndex's AgentRunner**.
 Accept user queries, retrieve relevant chunks, and provide synthesized responses.



### PROGRAM:
```python
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

# List of papers to load
papers = [
    "knowledge_card.pdf",
    "swebench.pdf",
    "longlora.pdf"
]

# Loading vector tools and summary tools for each paper
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print("Loading,wait")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]


from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=False
)
agent = AgentRunner(agent_worker)


print("Available documents are:")
for p in papers:
    print(p)


question = input("Ask your query related to these documents: \n")

response = agent.query(question) 

print("Response: \n")
print(response)
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/bd532f2a-7020-4bae-a65f-8bf440e87186)


### RESULT:
The multidocument retrieval agent was successfully implemented using LlamaIndex. It demonstrated the ability to process and retrieve relevant information from multiple research articles, synthesizing concise and accurate responses to diverse queries. The system is efficient and modular, making it adaptable to various domains and datasets.
