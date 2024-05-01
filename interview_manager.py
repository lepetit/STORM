
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

from editor_manager import EditorManager
from common_classes import WorkState, InterviewState, Perspectives
from tools import search_engine



class InterviewManager:
    max_num_turns = 5

    def __init__(self, state : WorkState):
        self.perspectives : Perspectives = state.perspectives
        self.topic : str = state.topic

    def initial_state(self, editor):
        return {
            #"editor": perspectives.editors[0],
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {self.topic}?",
                    name="SubjectMatterExpert",
                )
            ]
        }

    
    def route_messages(self, state: InterviewState, name: str = "SubjectMatterExpert"):
        messages = state["messages"]
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= self.max_num_turns:
            return END
        last_question = messages[-2]
        if last_question.content.endswith("WORK_DONE"):
            return END
        return "ask_question"
    
    
    def build_interview_graph(self, editorManager : EditorManager):
        builder = StateGraph(InterviewState)

        builder.add_node("ask_question", editorManager.generate_question)
        builder.add_node("answer_question", editorManager.gen_answer)
        builder.add_conditional_edges("answer_question", self.route_messages)
        builder.add_edge("ask_question", "answer_question")

        builder.set_entry_point("ask_question")
        interview_graph = builder.compile().with_config(run_name="Conduct Interviews")

        return interview_graph


    def conduct_interview(self, editorManager : EditorManager):
        
        outcomes = []
        interview_graph = self.build_interview_graph(editorManager)

        for editor in self.perspectives.editors:
            print(f"\nInterview with {editor.card}")
            initial_state = self.initial_state(editor)
            outcome = interview_graph.invoke(initial_state)
            #print(outcome)
            outcomes.append(outcome)

        return outcomes
