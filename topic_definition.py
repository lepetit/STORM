from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate

from common_classes import Outline, RelatedSubjects
from common_classes import WorkState


class InitialTopicGenerator():
    def __init__(self, state : WorkState):
        self.llm = state.fast_llm
        self.role = state.role
        self.topic = state.topic

        self.direct_gen_outline_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "{role}. Write an outline for the user-provided topic. Be comprehensive and specific.",
            ),
            ("user", "{topic}"),
        ])

        self.generate_outline_direct = self.direct_gen_outline_prompt | self.llm.with_structured_output(Outline)

        self.gen_related_topics_prompt = ChatPromptTemplate.from_template(
            """I'm writing about the topic mentioned below. Please identify and recommend some web content on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in articles for similar topics.

            Please list the as many subjects and urls as you can.

            Topic of interest: {topic}
            """
        )

        self.expand_chain = self.gen_related_topics_prompt | self.llm.with_structured_output(
            RelatedSubjects
        )

    def generate_outline(self):
        return self.generate_outline_direct.invoke({"topic": self.topic, "role": self.role})

    def related_subjects(self):
        return self.expand_chain.invoke({"topic": self.topic})

