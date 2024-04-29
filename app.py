from dotenv import load_dotenv
load_dotenv()

#generic
import os
import pickle

#langchain
from langchain_openai import ChatOpenAI

#App specific
from topic_definition import InitialTopicGenerator
from perspective_generator import PerspectiveGenerator
from editor_manager import EditorManager
from interview_manager import InterviewManager
from outline_refiner import OutlineRefiner
from article_generator import ArticleGenerator

FAST_LLM = "gpt-3.5-turbo"
LONG_CONTEXT_LLM= "gpt-3.5-turbo"  #SHOULD BE "gpt-4-turbo-preview"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "STORM"

fast_llm = ChatOpenAI(model=FAST_LLM)
long_context_llm = ChatOpenAI(model=FAST_LLM)


pickle_file = "boardgame_data.pkl"
topic = "The wargame 'Struggle of Empires'"
role = "You are a board game enthusiast expert in reviewing them for specialized magazines"
team_roles = "boardgame magazine editors"

def store_pickle():
    with open(pickle_file, 'wb') as outp:
        pickle.dump(initial_outline, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(related_subjects, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(perspectives_result, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(interview_result, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(refined_outline, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(article_draft, outp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(article_final, outp, pickle.HIGHEST_PROTOCOL)

def load_pickle():
    global initial_outline, related_subjects, perspectives_result, interview_result, refined_outline, article_draft, article_final
    with open(pickle_file, 'rb') as inp:
        initial_outline = pickle.load(inp)
        related_subjects = pickle.load(inp)
        perspectives_result = pickle.load(inp)
        interview_result = pickle.load(inp)
        refined_outline = pickle.load(inp)
        article_draft = pickle.load(inp)
        article_final = pickle.load(inp)

def prepare_article():
    initial_topic_generator = InitialTopicGenerator(fast_llm, role)
    initial_outline = initial_topic_generator.generate_outline(topic)
    print("initial_topic:", initial_outline.as_str)

    related_subjects = initial_topic_generator.related_subjects(topic)
    print("related_subjects:", related_subjects.topics)

    return initial_outline, related_subjects

def generate_perspective(related_subjects):
    perspective_generator = PerspectiveGenerator(FAST_LLM)
    perspectives = perspective_generator.survey_subjects(related_subjects, topic, team_roles)
    #for editor in perspectives.editors:
    #    print(f"{editor.card}")
    return perspectives

def interview(perspectives):
    editorManager = EditorManager(fast_llm, role)
    im = InterviewManager(perspectives, topic)
    interview = im.conduct_interview(editorManager)
    return interview

def refine_outline(interview, outline):
    refiner = OutlineRefiner(long_context_llm)
    refined_outline = refiner.refine_outline(topic, outline, interview, role)
    return refined_outline

def write_article(interview, refined_outline):

    print("GENERATING ARTICLE")

    article_generator = ArticleGenerator(long_context_llm, role)
    #sections = article_generator.article_section_writer(topic, refined_outline)
    #print("sections:", sections)
    article_generator.refences_lister(interview)
    article, draft = article_generator.write_article(topic, refined_outline)

    return article, draft

def report_activity(output_file_path):
    # Convert file name to .md extension
    output_file_path = os.path.splitext(output_file_path)[0] + ".md"

    with open(output_file_path, 'w') as f:
        f.write(initial_outline.as_str)
        f.write("\n## Related_subjects:\n" + related_subjects.topics)
        f.write("\n## Team:")
        for editor in perspectives_result.editors:
            f.write(f"{editor.card}\n")

        if interview_result is not None:
            f.write("\n## Interview:")
            for iv in interview_result:
                editor = iv["editor"]
                messages = iv["messages"]
                references = iv["references"]

                f.write(f"#### Interview with {editor.name}, {editor.role}:\n")
                for message in messages:
                    f.write(f"\n**{message.name}**: {message.content}\n")

        f.write("\n## Refined outline:\n")
        f.write(refined_outline.as_str)

        f.write("\n\n## Article DRAFT:\n")
        f.write(article_draft)

        f.write("\n\n## Article final version:\n")
        f.write(article_final)

initial_outline = None
related_subjects = None
perspectives_result = None
interview_result = None
refined_outline = None
article_draft = None
article_final = None

# Very basic debug code to see if the functions are working as expected
load_pick = False
prepare_work = False
do_interview = False
create_refined_outline = False
create_article = False
save_pick = False
write_report = True


if load_pick:
    load_pickle()

if prepare_work:
    initial_outline, related_subjects = prepare_article()
    perspectives_result = generate_perspective(related_subjects)

if do_interview:
    interview_result= interview(perspectives_result)

if create_refined_outline:
    refined_outline = refine_outline(interview_result, initial_outline)
    print("refined_outline:", refined_outline.as_str)

if create_article:
    article_final, article_draft = write_article(interview_result, refined_outline)
    print(article_draft, "\n\n------------ FINAL ARTICLE ---------------\n\n")
    print(article_final)
    
if save_pick:
    store_pickle()

if write_report:
    report_activity(pickle_file)