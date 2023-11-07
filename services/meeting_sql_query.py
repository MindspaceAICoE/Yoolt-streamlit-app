from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os

def add_reminder(message):

    load_dotenv()
    host=os.getenv('DB_HOST'),  # this is the name you find in the .env file
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
    print(host, user, password, database)
    db = SQLDatabase.from_uri(f"mysql+pymysql://admin:Yoolt12345@yoolt-db.csmeqs709mpv.us-east-2.rds.amazonaws.com/yoolt_db")
    llm = OpenAI(temperature=0, verbose=True)
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    db_chain.run(message)

# message = "set a reminder to meet with rebin at 2pm on 12th june"
# add_reminder(message)