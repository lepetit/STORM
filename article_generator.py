from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from common_classes import Subsection

class SubSection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    content: str = Field(
        ...,
        title="Full content of the subsection. Include [#] citations to the cited sources where relevant.",
    )

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.content}".strip()


class ArticleSection(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    content: str = Field(..., title="Full content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the article.",
    )
    citations: List[str] = Field(default_factory=list)

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            subsection.as_str for subsection in self.subsections or []
        )
        citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
        return (
            f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
            + f"\n\n{citations}".strip()
        )


class ArticleGenerator:
    def  __init__(self, long_context_llm, role):
        self.role = role
        
        self.section_writer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{role}. Complete your assigned section from the following outline:\n\n"
                    "{outline}\n\nCite your sources, using the following references:\n\n<Documents>\n{docs}\n<Documents>",
                ),
                ("user", "Write the full content for the '{section}' section."),
            ]
        )
        
        self.section_writer = (
            self.retrieve
            | self.section_writer_prompt
            | long_context_llm.with_structured_output(ArticleSection)
        )


        self.writer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{role}. Write a long, complete and exhaustive composition on {topic}. Use the following drafts, which was prepared by a group of industry experts:\n\n"
                    "{draft}\n\nUse a style appropriate to the topic, be accurate, original, engaging and cite as many details as possible. ",
                ),
                (
                    "user",
                    'Write the complete composition using markdown format. Organize citations using footnotes like "[1]",'
                    ' avoiding duplicates in the footer. Include URLs in the footer.',
                ),
            ]
        )

        self.writer = self.writer_prompt | long_context_llm | StrOutputParser()        

    def refences_lister(self, interview):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        ref = {k: v for iv in interview for k, v in iv["references"].items()}

        reference_docs = [
            Document(page_content=v, metadata={"source": k})
            for k, v in ref.items()
        ]

        # This really doesn't need to be a vectorstore for this size of data.
        # It could just be a numpy matrix. Or you could store documents
        # across requests if you want.
        vectorstore = SKLearnVectorStore.from_documents(
            reference_docs,
            embedding=embeddings,
        )
        self.refences = vectorstore.as_retriever(k=10)


    def retrieve(self, inputs: dict):        
        docs = self.refences.invoke(inputs["topic"] + ": " + inputs["section"])
        formatted = "\n".join(
            [
                f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )
        return {"docs": formatted, **inputs}    


    def article_section_writer(self, topic, refined_outline):
        sections = []

        for section in refined_outline.sections:
            print("Drafting ", section.section_title)
            write_section = self.section_writer.invoke(
                {
                    "outline": refined_outline.as_str,
                    "section": section.section_title,
                    "role": self.role,
                    "topic": topic,
                }
            )
            sections.append(write_section)

        return sections
    
    def write_article(self, topic, refined_outline):
        article = ""

        sections = self.article_section_writer(topic, refined_outline)
        draft = "\n\n".join([section.as_str for section in sections])
        article = self.writer.invoke({"topic": topic, "draft": draft, "role" : self.role})

        return article, draft
    