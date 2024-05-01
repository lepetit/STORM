from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from common_classes import WorkState, Outline


class OutlineRefiner:
    def __init__(self, state : WorkState):
        self.state = state

        self.refine_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """{role}. You have gathered information from experts and search engines. Now, you are refining the outline of the content. \
        You need to make sure that the outline is comprehensive and specific. \
        Topic you are writing about: {topic}

        Old outline:

        {old_outline}""",
                ),
                (
                    "user",
                    "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined outline:",
                ),
            ]
        )

        # Using turbo preview since the context can get quite long
        self.refine_outline_chain = self.refine_outline_prompt | state.long_context_llm.with_structured_output(
            Outline
        )

    def refine_outline(self):
        all_messages = []
        for interview in self.state.interviews:
            all_messages.extend(interview['messages'])

        refined_outline = self.refine_outline_chain.invoke(
            {
                "topic": self.state.topic,
                "old_outline": self.state.initial_outline.as_str,
                "role": self.state.role,
                "conversations": "\n\n".join(
                    f"### {m.name}\n\n{m.content}" for m in all_messages
                ),
            }
        )

        return refined_outline