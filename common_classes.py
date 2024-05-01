
import pickle

from typing_extensions import TypedDict
from typing import List, Optional

#langchain
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import validator
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AnyMessage
from typing import Annotated, Sequence


FAST_LLM = "gpt-3.5-turbo"
LONG_CONTEXT_LLM= "gpt-3.5-turbo"  #SHOULD BE "gpt-4-turbo-preview"

class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(
        default=None,
        title="Titles and descriptions for each subsection of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(
            f"### {subsection.subsection_title}\n\n{subsection.description}"
            for subsection in self.subsections or []
        )
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[Section] = Field(
        default_factory=list,
        title="Titles and descriptions for each section of the Wikipedia page.",
    )

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


class RelatedSubjects(BaseModel):
    topics: List[str] = Field(
        description="Comprehensive list of related subjects as background research.",
    )



class Editor(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the editor.",
    )
    name: str = Field(
        description="Name of the editor.",
    )
    role: str = Field(
        description="Role of the editor in the context of the topic.",
    )
    description: str = Field(
        description="Description of the editor's focus, concerns, and motives.",
    )

    @validator("name", pre=True)
    def strip_whitespaces(cls, v: str) -> str:
        return v.replace(' ', '').replace('.','')

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

    @property
    def card(self) -> str:
        return f"* {self.name} - {self.role}, {self.affiliation}\n    {self.description}"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )



class InterviewState(TypedDict):

    def add_messages(left, right):
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]
        return left + right


    def update_references(references, new_references):
        if not references:
            references = {}
        references.update(new_references)
        return references


    def update_editor(editor, new_editor):
        # Can only set at the outset
        if not editor:
            return new_editor
        return editor
        
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], update_references]
    editor: Annotated[Optional[Editor], update_editor]


class WorkState():
    pickle_file : str
    topic : str
    role : str
    team_roles : str
    initial_outline : Outline = None
    related_subjects : RelatedSubjects = None
    perspectives : Perspectives = None
    interviews : List[InterviewState] = None
    refined_outline : Outline = None
    article_draft : str = None
    article_final : str = None

    fast_llm = ChatOpenAI(model=FAST_LLM)
    long_context_llm = ChatOpenAI(model=LONG_CONTEXT_LLM)

    def __init__(self, pickle_file : str, topic : str, role : str, team_roles : str):
        self.pickle_file = pickle_file
        self.topic = topic
        self.role = role
        self.team_roles = team_roles

    def store_pickle(self):
        with open(self.pickle_file, 'wb') as outp:
            pickle.dump(self.initial_outline, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.related_subjects, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.perspectives, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.interviews, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.refined_outline, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.article_draft, outp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.article_final, outp, pickle.HIGHEST_PROTOCOL)

    def load_pickle(self):
        with open(self.pickle_file, 'rb') as inp:
            self.initial_outline = pickle.load(inp)
            self.related_subjects = pickle.load(inp)
            self.perspectives = pickle.load(inp)
            self.interviews = pickle.load(inp)
            self.refined_outline = pickle.load(inp)
            self.article_draft = pickle.load(inp)
            self.article_final = pickle.load(inp)
