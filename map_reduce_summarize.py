from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_together import ChatTogether

# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str


class Map_Summary:
    def __init__(self, llm: ChatTogether, token_max: int = 17000):
        self.llm = llm
        self.token_max = token_max

        self.map_prompt = ChatPromptTemplate.from_messages(
            [("system", "Extract the important information or ideas from the following podcast text, maintaining the order as in the podcast text:\\n\\n{context}")]
        )
        self.map_chain = self.map_prompt | self.llm | StrOutputParser()

        reduce_template = """
        The following is a list of important information or ideas extracted from the podcast text:
        {docs}
        Take these and distill it into a final, consolidated summary.
        """
        self.reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
        self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()

    def length_function(self, documents: List[Document]) -> int:
        """Get number of tokens for input contents."""
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)

    async def generate_summary(self, state: SummaryState):
        response = await self.map_chain.ainvoke(state["content"])
        return {"summaries": [response]}

    def map_summaries(self, state: OverallState):
        return [
            Send("generate_summary", {"content": content}) for content in state["contents"]
        ]

    def collect_summaries(self, state: OverallState):
        return {
            "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
        }

    async def collapse_summaries(self, state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], self.length_function, self.token_max
        )
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, self.reduce_chain.ainvoke))

        return {"collapsed_summaries": results}

    def should_collapse(self, state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = self.length_function(state["collapsed_summaries"])
        if num_tokens > self.token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    async def generate_final_summary(self, state: OverallState):
        response = await self.reduce_chain.ainvoke(state["collapsed_summaries"])
        return {"final_summary": response}

    def construct_graph(self):
        graph = StateGraph(OverallState)
        graph.add_node("generate_summary", self.generate_summary)
        graph.add_node("collect_summaries", self.collect_summaries)
        graph.add_node("collapse_summaries", self.collapse_summaries)
        graph.add_node("generate_final_summary", self.generate_final_summary)

        graph.add_conditional_edges(START, self.map_summaries, ["generate_summary"])
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self.should_collapse)
        graph.add_conditional_edges("collapse_summaries", self.should_collapse)
        graph.add_edge("generate_final_summary", END)

        return graph.compile()
