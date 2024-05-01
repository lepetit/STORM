from langchain_core.pydantic_v1 import validator
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.runnables import RunnableLambda, chain as as_runnable

from common_classes import WorkState, Perspectives


class PerspectiveGenerator():
    def __init__(self, state : WorkState):
        self.state = state

        self.gen_perspectives_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You need to select a diverse (and distinct) group of {perspective_roles} who will work together to complete a comprehensive analysis on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.\
                    You can search on the Internet for related topics for inspiration. For each expert, add a description of what they will focus on.

                    These are sample outlines of related topics for inspiration:
                    {examples}""",
                ),
                ("user", "Topic of interest: {topic}"),
            ]
        )

        '''self.gen_perspectives_chain = self.gen_perspectives_prompt | ChatOpenAI(
            model= state.fast_llm
        ).with_structured_output(Perspectives)'''
        self.gen_perspectives_chain = self.gen_perspectives_prompt | state.fast_llm.with_structured_output(Perspectives)


    def format_doc(self, doc, max_length=1000):
        related = "- ".join(doc.metadata["categories"])
        return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[
            :max_length
        ]

    def format_docs(self, docs):
        return "\n\n".join(self.format_doc(doc) for doc in docs)

    def survey_subjects(self):
        wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)

        retrieved_docs = wikipedia_retriever.batch(
            self.state.related_subjects.topics, return_exceptions=True
        )
        all_docs = []
        for docs in retrieved_docs:
            if isinstance(docs, BaseException):
                continue
            all_docs.extend(docs)
        formatted = self.format_docs(all_docs)
        return self.gen_perspectives_chain.invoke({"examples": formatted, "topic": self.state.topic, "perspective_roles" : self.state.perspective_roles})