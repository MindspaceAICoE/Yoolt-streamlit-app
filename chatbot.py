
# Run this command from the command line, not within the Python script:
# pip install openai

import os
import re
import openai
from PIL import Image
# import spacy
# from spacy import displacy
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from frontend.html_template import css, bot_template, user_template
# from dotenv import load_dotenv
import pickle
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from streamlit_extras import verticla
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from services.email_service import send_email_with_sendgrid
from services.pdf_reader_service import get_text_from_pdf
from services.sql_service import get_data_from_database
from services.meeting_sql_query import add_reminder
from services.pattern_recognition_service import meeting_minutes
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from services.video_rendering_service import transcribeVideoOrchestrator
from services.ocr_service import load_model
import numpy as np  
load_dotenv()
openai.api_key  = st.secrets["OPENAI_API_KEY"]

def detect_chat_mode(query):
    # Email detection: For simplicity, we'll check if it starts with "email". Adjust this as needed.
    if is_email_request(query):
        return "Email"
    
    # Chat about document detection: Check for document-specific keywords or contents.
    # This is a basic example. Modify with actual keywords or better logic.
    elif is_doc_query(query):
        return "Document"
    elif is_reminder_request(query):
        return "Reminder"
    
    # Default to general chat
    return "General Chat" 
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def is_doc_query(query):
    # Implement logic to determine the type of query
    # For example, check if query contains certain keywords
    document_keywords = ['pdf', 'document', 'file']
    is_document_query = any(keyword in query.lower() for keyword in document_keywords)
    return is_document_query

def is_email_request(query):
    email_keywords = ['send', 'email', 'mail', 'forward']
    return any(keyword in query.lower() for keyword in email_keywords)

def is_reminder_request(query):
    # Expanded keywords related to reminders and scheduling
    reminder_keywords = ['reminder', 'remember', 'remind me', 'appointment', 'schedule', 'meeting']
    return any(keyword in query.lower() for keyword in reminder_keywords)

def extract_email_info(query):
    recipient = None
    email_type = None
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", query)
    if email_match:
        recipient = email_match.group(0)

   

    email_keywords = ['resignation', 'invitation', 'reminder', 'thank you', 'follow-up']
    for keyword in email_keywords:
        if keyword in query.lower():
            email_type = keyword
            break

    print(f"Recipient: {recipient} \n Email Type: {email_type}")

    return recipient, email_type

def generate_email_content(email_type, recipient_name):
    prompt = f"Compose a {email_type} email addressed to {recipient_name}."
    response = get_completion(prompt)
    return response

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# def add_reminder(message):
#     has_run = False
#     load_dotenv()
#     host=os.getenv('DB_HOST'),  # this is the name you find in the .env file
#     user=os.getenv("DB_USER"),
#     password=os.getenv("DB_PASSWORD"),
#     database=os.getenv("DB_NAME")
#     print(host, user, password, database)
#     db = SQLDatabase.from_uri(f"mysql+pymysql://admin:Yoolt12345@yoolt-db.csmeqs709mpv.us-east-2.rds.amazonaws.com/yoolt_db")
#     llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'),model_name='gpt-3.5-turbo')

#     QUERY = """
# Translate the following user request into a SQL query to insert a reminder into the database:
# Question: "{question}"
# """
#     db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
#     formatted_message = QUERY.format(question=message)

#     try:
#         if not has_run:
#             print(db_chain.run(formatted_message))
#             has_run = True
#     except Exception as e:
#         print(e)
#     # try:
#     #     question = QUERY.format(question=message)
#     #     print(db_chain.run(question))
#     # except Exception as e:
#     #     print(e)


   
#     # db_chain.run(message)


def get_conversation_chain(vectorstore):
    llm = OpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    chat_mode = detect_chat_mode(user_question)

    if chat_mode == "Email":
        recipient, email_type = extract_email_info(user_question)

        if recipient and email_type:  # Email-related query
            if 'generated_email_content' not in st.session_state:
                st.session_state.generated_email_content = generate_email_content(email_type, recipient)
            
            st.write(f"Generated Email Content: {st.session_state.generated_email_content}")

            # Check if the user wants to edit the email
            if 'edit_email' in st.session_state and st.session_state.edit_email:
                edited_content = st.text_area("Edit Email Content:", st.session_state.generated_email_content,height=400)
                
                if st.button("Confirm Changes"):
                    st.session_state.generated_email_content = edited_content
                    st.session_state.edit_email = False

                # Button to continue editing after changes
                if st.button("Edit Again"):
                    st.session_state.edit_email = True
            else:
                if st.button("Edit Email"):
                    st.session_state.edit_email = True

            # Button to send the email
            if st.button("Send Email"):
                send_email_with_sendgrid(recipient, st.session_state.generated_email_content)
                response_message = f"Email sent to {recipient}!"
                st.write(response_message)
                
                # Clear the generated email content from session state
                del st.session_state.generated_email_content

                # Optional: Append email-related interactions to chat history
                # if "chat_history" not in st.session_state:
                #     st.session_state.chat_history = []
                # st.session_state.chat_history.append(response_message)
        else:
            st.write("Invalid email request. Please format it correctly.")


    elif chat_mode == "Document":

        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    
    elif chat_mode == "Reminder":
        if st.button("Set Reminder"):
            add_reminder(user_question,db_url=st.secrets["COCKROACH_DB_URL"])
        # try:
        #     # Call your function to handle the reminder
        #     add_reminder(user_question)
            
        #     # Provide a success message to the user
        #     st.write("Meeting set successfully!")
        # except Exception as e:
        #     # Handle exceptions and inform the user
        #     st.write(f"Error setting the reminder: {e}")
    else:
        response_content = get_completion(user_question, model="gpt-3.5-turbo")
        st.write(bot_template.replace("{{MSG}}", response_content), unsafe_allow_html=True)
        

def main():
    st.set_page_config(page_title="Chatbot", page_icon=":robot_face:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with Yoolt 💬")
   # st.text_input("Ask a question about your uploaded documents or enter any other query:")
   
    # user_question = st.text_input("Ask a question about your documents:")
    # if user_question:
    #     handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Navigation")
        selected_tab = st.radio("Choose a Tab:", ["Yoolt", "My Reminders","Video Rendering","OCR"])
        if selected_tab == "Yoolt":
            st.subheader("Your Documents")
            pdf_files=st.file_uploader("Upload your PDFs and click on 'Process'",accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    if pdf_files is None:
                        st.error("Please upload a PDF file")
                    else:
                        raw_text=get_text_from_pdf(pdf_files)
                        text_chunks=get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        
    if selected_tab == "Yoolt":
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

    elif selected_tab == "My Reminders":
        st.title("My Reminders")
        data = get_data_from_database()
        st.table(data)  

    elif selected_tab == "Video Rendering":
        st.title("Video Rendering")
        url = st.text_input("Enter YouTube URL:")
        models = ["tiny", "base", "small", "medium", "large"]
        model = st.selectbox("Select Model:", models)
        st.write(
            "If you take a smaller model it is faster but not as accurate, whereas a larger model is slower but more accurate.")
        if st.button("Transcribe"):
            if url:
                transcript = transcribeVideoOrchestrator(url, model)

            if transcript:
                st.subheader("Transcription:")
                st.write(transcript)
                meeting_insights = meeting_minutes(transcript)
                st.subheader("Meeting Summary:")
                st.write(meeting_insights['abstract_summary'])
                st.subheader("Meeting Key-Points:")
                st.write(meeting_insights['key_points'])
                st.subheader("Action Items:")
                st.write(meeting_insights['action_items'])
                st.subheader("Sentiment Analysis:")
                st.write(meeting_insights['sentiment'])
                


            else:
                st.error("Error occurred while transcribing.")
                st.write("Please try again.")

    elif selected_tab=="OCR":
        #title
        st.title("Easy OCR - Extract Text from Images")

        #subtitle
        st.markdown("## Optical Character Recognition - Using `easyocr`, `streamlit`")

        st.markdown("")

        #image uploader
        image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

        reader = load_model() #load model

        if image is not None:

            input_image = Image.open(image) #read image
            st.image(input_image) #display image

            with st.spinner("🤖 AI is at Work! "):
                

                result = reader.readtext(np.array(input_image))

                result_text = [] #empty list for results


                for text in result:
                    result_text.append(text[1])

                st.write(result_text)
            st.balloons()
        else:
            st.write("Upload an Image")    






    




if __name__ == '__main__':
    main()
