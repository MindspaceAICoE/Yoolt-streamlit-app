import mysql.connector
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv()
# Set up the database connection
def get_database_connection():
    connection = mysql.connector.connect(
        host=st.secrets["DB_HOST"],  # this is the name you find in the .env file
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_NAME"]
    )
    return connection

# Fetch data from your table
def get_data_from_database():
    connection = get_database_connection()
    cursor = connection.cursor(dictionary=True)  # Use dictionary=True to get column names
    cursor.execute("SELECT * FROM reminders")
    data = cursor.fetchall()
    connection.close()
    return data

# # Display data on Streamlit
# def main():
#     # st.title('My Database Data')
#     data = get_data_from_database()
#     print(data)
#     # st.write(data)  # You can use st.table(data) for a more table-like format

# if __name__ == "__main__":
#     main()
