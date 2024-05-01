import os 
from typing import List, Optional


#App specific
from topic_definition import InitialTopicGenerator
from perspective_generator import PerspectiveGenerator
from editor_manager import EditorManager
from interview_manager import InterviewManager
from outline_refiner import OutlineRefiner
from article_generator import ArticleGenerator
from common_classes import WorkState


class StormManager():
    def  __init__(self, work_state : WorkState):
        self.state = work_state

    def execute(self):
        load_pick = True
        prepare_work = True
        do_interview = False
        create_refined_outline = False
        create_article = False
        save_pick = False
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
        initial_topic_generator = InitialTopicGenerator(self.state)
        self.state.initial_outline = initial_topic_generator.generate_outline()
        print("Initial_topic") #, self.state.initial_outline.as_str)

        self.state.related_subjects = initial_topic_generator.related_subjects()
        print("Related_subjects") #, self.state.related_subjects.topics)

    def generate_perspective(self):
        perspective_generator = PerspectiveGenerator(self.state)
        self.state.perspectives = perspective_generator.survey_subjects()
        print("Perspectives")

    def interview(self):
        editorManager = EditorManager(self.state)
        im = InterviewManager(self.state)
        self.state.interviews = im.conduct_interview(editorManager)
        print("Interviews")

    def refine_outline(self):
        refiner = OutlineRefiner(self.state.long_context_llm)
        self.state.refined_outline = refiner.refine_outline()
        print("Refined outline")

    def write_article(self):
        article_generator = ArticleGenerator(self.state)
        article_generator.refences_lister()
        article, draft = article_generator.write_article()

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
