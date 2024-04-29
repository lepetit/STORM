import json

from typing import List, Optional

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, chain as as_runnable
from langchain_core.runnables import RunnableConfig

from common_classes import InterviewState
from tools import search_engine



class Queries(BaseModel):
    queries: List[str] = Field(
        description="Comprehensive list of search engine queries to answer the user's questions.",
    )

class AnswerWithCitations(BaseModel):
    answer: str = Field(
        description="Comprehensive answer to the user's question with citations.",
    )
    cited_urls: List[str] = Field(
        description="List of urls cited in the answer.",
    )

    @property
    def as_str(self) -> str:
        return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
            f"[{i+1}]: {url}" for i, url in enumerate(self.cited_urls)
        )



class EditorManager:
    def __init__(self, llm, editor_role):
        self.llm = llm
        self.editor_role = editor_role
        self.gen_qn_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """{editor_role}. \
                    Besides your expertise, you have a specific focus when researching the topic. \
                    Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

                    When you have no more questions to ask, say 'WORK_DONE' to end the conversation.\
                    Please only ask one question at a time and don't ask what you have asked before.\
                    Your questions should be related to the topic you want to write.
                    Be comprehensive and curious, gaining as much unique insight from the expert as possible.\

                    Stay true to your specific perspective:

                    {persona}""",
                ),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )

        self.gen_queries_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful research assistant. Query the search engine to answer the user's questions.",
                ),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )

        self.gen_queries_chain = self.gen_queries_prompt | ChatOpenAI(
            model=llm.model_name
        ).with_structured_output(Queries, include_raw=True)

        self.gen_answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert who can use information effectively. You are chatting with a review writer who wants\
                to write an article on the topic you know. You have gathered the related information and will now use the information to form a response.

                Make your response as informative as possible and make sure every sentence is supported by the gathered information.
                Each response must be backed up by a citation from a reliable source, formatted as a footnote, reproducing the URLS after your response.""",
                ),
                MessagesPlaceholder(variable_name="messages", optional=True),
            ]
        )

        self.gen_answer_chain = self.gen_answer_prompt | llm.with_structured_output(
            AnswerWithCitations, include_raw=True
        ).with_config(run_name="GenerateAnswer")


    def tag_with_name(self, ai_message: AIMessage, name: str):
        ai_message.name = name
        return ai_message


    def swap_roles(self, state: InterviewState, name: str):
        converted = []
        for message in state["messages"]:
            if isinstance(message, AIMessage) and message.name != name:
                message = HumanMessage(**message.dict(exclude={"type"}))
            converted.append(message)
        return {"messages": converted}

    def generate_question(self, state: InterviewState):
        editor = state["editor"]
        gn_chain = (
            RunnableLambda(self.swap_roles).bind(name=editor.name)
            | self.gen_qn_prompt.partial(persona=editor.persona, editor_role=self.editor_role)
            | self.llm
            | RunnableLambda(self.tag_with_name).bind(name=editor.name)
        )
        result = gn_chain.invoke(state)
        return {"messages": [result]}
    
    #def generate_answer(self, messages):
    #    queries = self.gen_queries_chain.invoke(
    #        {"messages": messages}
    #    )        
    #    return queries

    def gen_answer(self,
        state: InterviewState,
        config: Optional[RunnableConfig] = None,
        name: str = "SubjectMatterExpert",
        max_str_len: int = 15000,
    ):
        swapped_state = self.swap_roles(state, name)  # Convert all other AI messages
        queries = self.gen_queries_chain.invoke(swapped_state)
        query_results = search_engine.batch(
            queries["parsed"].queries, config, return_exceptions=True
        )
        successful_results = [
            res for res in query_results if not isinstance(res, Exception)
        ]
        all_query_results = {
            res["url"]: res["content"] for results in successful_results for res in results
        }
        # We could be more precise about handling max token length if we wanted to here
        dumped = json.dumps(all_query_results)[:max_str_len]
        ai_message: AIMessage = queries["raw"]
        tool_call = queries["raw"].additional_kwargs["tool_calls"][0]
        tool_id = tool_call["id"]
        tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
        swapped_state["messages"].extend([ai_message, tool_message])
        # Only update the shared state with the final answer to avoid
        # polluting the dialogue history with intermediate messages
        generated = self.gen_answer_chain.invoke(swapped_state)
        cited_urls = set(generated["parsed"].cited_urls)
        # Save the retrieved information to a the shared state for future reference
        cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
        formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
        return {"messages": [formatted_message], "references": cited_references}    