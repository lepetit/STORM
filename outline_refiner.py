from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from common_classes import Outline, RelatedSubjects


class OutlineRefiner:
    def __init__(self, long_context_llm):

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
        self.refine_outline_chain = self.refine_outline_prompt | long_context_llm.with_structured_output(
            Outline
        )

    def refine_outline(self, topic, initial_outline, interviews, role):
        all_messages = []
        for interview in interviews:
            all_messages.extend(interview['messages'])

        refined_outline = self.refine_outline_chain.invoke(
            {
                "topic": topic,
                "old_outline": initial_outline.as_str,
                "role": role,
                "conversations": "\n\n".join(
                    f"### {m.name}\n\n{m.content}" for m in all_messages
                ),
            }
        )

        return refined_outline