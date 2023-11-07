import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import streamlit as st
from dotenv import load_dotenv, find_dotenv
load_dotenv()
_ = load_dotenv(find_dotenv())

def extract_subject_and_body(email_content):
    lines = email_content.strip().split("\n")
    subject = lines[0].replace("Subject: ", "").strip()
    body = "\n".join(lines[1:]).strip()
    return subject, body

def prepare_body_for_html(body):
    return body.replace('\n', '<br>')


def send_email_with_sendgrid(to_email, content):
    subject, body = extract_subject_and_body(content)
    body = prepare_body_for_html(body)
    message = Mail(
        from_email='Soumyaprakash@mindspace.llc',
        to_emails=to_email,
        subject=subject,
        html_content=body)
    try:
        sg = SendGridAPIClient(st.secrets["SENDGRID_API_KEY"])
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e)