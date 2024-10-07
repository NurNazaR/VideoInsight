import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages

from typing import List

import tiktoken
from langchain_core.messages import (
    BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
)


load_dotenv()

class Rag():
    def __init__(self, llm, session_id, transcript_text):
        self.llm = llm
        self.session_id = session_id
        self.transcript_text = transcript_text
        
        chunks = self.chunks(self.transcript_text)
        self.db = self.vector_store(chunks)


    def str_token_counter(self, text: str) -> int:
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))


    def tiktoken_counter(self, messages: List[BaseMessage]) -> int:
        """Approximately reproduce https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        For simplicity only supports str Message.contents.
        """
        num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
        tokens_per_message = 3
        tokens_per_name = 1
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                raise ValueError(f"Unsupported messages type {msg.__class__}")
            num_tokens += (
                tokens_per_message
                + self.str_token_counter(role)
                + self.str_token_counter(msg.content)
            )
            if msg.name:
                num_tokens += tokens_per_name + self.str_token_counter(msg.name)
        return num_tokens

    def chunks(self, transcript_text:str):
        documents = [Document(page_content=transcript_text)]
        # text_splitter = CharacterTextSplitter(
        #     separator="\n",
        #     chunk_size=1000,
        #     chunk_overlap=100,
        #     length_function=len,
        #     is_separator_regex=False,
        # )
        
        text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=2000, 
                                    chunk_overlap=200,
                                    length_function=len,
                                )
        docs = text_splitter.split_documents(documents)

        return docs

    def vector_store(self, documents):
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        db = Chroma.from_documents(documents, embeddings)
        return db
    
    def retrieve(self, query:str, top_k:int):
        return self.db.search(query, search_type='mmr', search_kwargs={'k': top_k, 'lambda_mult': 0.25})
        
    def generate(self, query:str):
        trimmer = trim_messages(
            max_tokens=4000,
            strategy="last",
            token_counter=self.tiktoken_counter,
            include_system=True,
        )
        system_prompt = """You are an assistant that answers questions over a knowledge base.
                    - On each message you will receive context from the knowledge base and an user message.
                    - Be brief and polite.
                    - Be conversational and friendly."""
        user_prompt = """"Use the following pieces of retrieved context to answer the question. Context: {context}
        
        question: {question}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", user_prompt),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        context = format_docs(self.retrieve(query, 4))

        rag_chain = (prompt
            | trimmer
            | self.llm
            | StrOutputParser()
        )
        
        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: MongoDBChatMessageHistory(
                session_id=session_id,
                connection_string=os.environ['MONGODB_URL'],
                database_name="chat_history_db",
                collection_name="chat_messages",
            ),
            input_messages_key="question",
            history_messages_key="history", 
        )
        
        results = chain_with_history.invoke(input = {'context':context, 'question':query}, config={"configurable": {"session_id":self.session_id}})

        return results 


        



