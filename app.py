from dotenv import load_dotenv
load_dotenv()

#App specific
from storm_manager import  StormManager, WorkState

pickle_file = "boardgame_data.pkl"
topic = "The wargame 'Struggle of Empires'"
role = "You are a board game enthusiast expert in reviewing them for specialized magazines"
team_roles = "boardgame magazine editors"

state : WorkState = WorkState(pickle_file= pickle_file, topic= topic, role= role, team_roles= team_roles)
storm : StormManager = StormManager(work_state=state)
storm.execute()

print("Done!")
exit()
