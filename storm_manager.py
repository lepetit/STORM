import os 
import pickle
from typing import List, Optional

#langchain
from langchain_openai import ChatOpenAI

#App specific
from topic_definition import InitialTopicGenerator
from perspective_generator import PerspectiveGenerator
from editor_manager import EditorManager
from interview_manager import InterviewManager
from outline_refiner import OutlineRefiner
from article_generator import ArticleGenerator
from common_classes import Outline, RelatedSubjects, Perspectives, InterviewState

FAST_LLM = "gpt-3.5-turbo"
LONG_CONTEXT_LLM= "gpt-3.5-turbo"  #SHOULD BE "gpt-4-turbo-preview"

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


class StormManager():
    def  __init__(self, work_state : WorkState):
        self.state = work_state

    def execute(self):
        load_pick = True
        prepare_work = False
        do_interview = False
        create_refined_outline = False
        create_article = True
        save_pick = True
        write_report = True

        if load_pick:
            self.state.load_pickle()

        if prepare_work:
            self.prepare_article()
            self.generate_perspective()

        if do_interview:
            self.interview()

        if create_refined_outline:
            self.refine_outline()

        if create_article:
            self.write_article()
            
        if save_pick:
            self.state.store_pickle()

        if write_report:
            self.write_report()

    def prepare_article(self):
        initial_topic_generator = InitialTopicGenerator(self.state.fast_llm, self.state.role)
        self.state.initial_outline = initial_topic_generator.generate_outline(self.state.topic)
        print("Initial_topic") #, self.state.initial_outline.as_str)

        self.state.related_subjects = initial_topic_generator.related_subjects(self.state.topic)
        print("Related_subjects") #, self.state.related_subjects.topics)

    def generate_perspective(self):
        perspective_generator = PerspectiveGenerator(FAST_LLM)
        self.state.perspectives = perspective_generator.survey_subjects(self.state.related_subjects, self.state.topic, self.state.team_roles)
        print("Perspectives")

    def interview(self):
        editorManager = EditorManager(self.state.fast_llm, self.state.role)
        im = InterviewManager(self.state.perspectives, self.state.topic)
        self.state.interviews = im.conduct_interview(editorManager)
        print("Interviews")

    def refine_outline(self):
        refiner = OutlineRefiner(self.state.long_context_llm)
        self.state.refined_outline = refiner.refine_outline(self.state.topic, self.state.initial_outline, self.state.interviews, self.state.role)
        print("Refined outline")

    def write_article(self):
        article_generator = ArticleGenerator(self.state.long_context_llm, self.state.role)
        article_generator.refences_lister(self.state.interviews)
        article, draft = article_generator.write_article(self.state.topic, self.state.refined_outline)

        self.state.article_draft = draft
        self.state.article_final = article
        print("Article written")

    def write_report(self):
        # Convert file name to .md extension
        output_file_path = os.path.splitext(self.state.pickle_file)[0] + ".md"

        with open(output_file_path, 'w') as f:
            #Quit, if there is no data
            if self.state.initial_outline == None:
                return
            
            f.write(self.state.initial_outline.as_str)            
            f.write("\n## Related_subjects:\n" + ",".join(str(x) for x in self.state.related_subjects.topics))
            f.write("\n## Team:\n")
            for editor in self.state.perspectives.editors:
                f.write(f"{editor.card}\n")

            if self.state.interviews is not None:
                f.write("\n## Interviews:\n")
                for iv in self.state.interviews:
                    editor = iv["editor"]
                    messages = iv["messages"]
                    references = iv["references"]

                    f.write(f"#### Interview with {editor.name}, {editor.role}:\n")
                    for message in messages:
                        f.write(f"\n**{message.name}**: {message.content}\n")

            if self.state.refined_outline != None:
                f.write("\n## Refined outline:\n")
                f.write(self.state.refined_outline.as_str)

            if self.state.article_draft != None:
                f.write("\n\n## Article DRAFT:\n")
                f.write(self.state.article_draft)

            if self.state.article_final != None:
                f.write("\n\n## Article final version:\n")
                f.write(self.state.article_final)

        print("Report written")
