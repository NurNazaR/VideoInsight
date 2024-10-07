import os
from dotenv import load_dotenv
import asyncio

load_dotenv() 
TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain_together import ChatTogether
from map_reduce_summarize import Map_Summary

class Summary():
    def __init__(self,llm:ChatTogether):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([("system", "Write a concise summary of the following:\\n\\n{context}")])
    def summarize(self, text:str):
        if self.llm.get_num_tokens(text) < 17500:
            print('Context size is {} tokens'.format(self.llm.get_num_tokens(text)))
            # Instantiate chain
            chain = self.prompt | self.llm | StrOutputParser()
            # Invoke chain
            result = chain.invoke({"context": text})
            return result
        else:
            print('Context is too big for LLM model. We try to split it into smaller parts')
            return 'Context is too big for LLM model'
            documents = [Document(page_content=text)]
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=40000,
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            docs = text_splitter.split_documents(documents)
            
            map_summary_graph = Map_Summary(llm=self.llm).construct_graph()
            
            async def process_documents():
                async for step in map_summary_graph.astream(
                    {"contents": [doc.page_content for doc in docs]},
                    {"recursion_limit": 50},
                ):
                    print(list(step.keys()))
                return step['generate_final_summary']["final_summary"]
            
            # Call the asynchronous function
            return asyncio.run(process_documents())

