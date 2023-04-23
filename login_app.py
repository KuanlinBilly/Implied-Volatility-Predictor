import streamlit as st
import hashlib
from predictor_app import * 
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# MongoDB connection details
mongo_id = 'mongodb+srv://ndhu:ndhu@cluster0.vnqxzcd.mongodb.net/?retryWrites=true&w=majority'
mongo_client = MongoClient(mongo_id)
mongo_db = mongo_client["ndhu"]
mongo_collection = mongo_db['user_login_info']  # mini database

class User:
    def __init__(self, id, username=None, password=None, mongo_collection=None):
        self.id = id
        self.username = username
        self.password = password
        self.mongo_collection = mongo_collection

    def save_to_mongo(self):
        user_doc = {
            "_id": self.id,
            "username": self.username,
            "password": generate_password_hash(self.password),
        }
        self.mongo_collection.insert_one(user_doc)

    @staticmethod
    def get_user(username, mongo_collection):
        user_doc = mongo_collection.find_one({"username": username})
        if user_doc:
            return User(user_doc["_id"], user_doc["username"], user_doc["password"])
        return None

    def verify_password(self, password):
        return check_password_hash(self.password, password)


# Function to set the background image
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            -webkit-background-size: cover;
            -moz-background-size: cover;
            -o-background-size: cover;
            background-size: cover;
        }}
        .stApp .stMarkdown h1, .stApp .stMarkdown h2, .stApp .stMarkdown p {{
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
def reset_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: none;
        }}
        .stApp .stMarkdown h1, .stApp .stMarkdown h2, .stApp .stMarkdown p {{
            color: inherit;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Function to verify the user's password (you should replace this with your actual authentication method)
def verify_password(stored_password, provided_password):
    return stored_password == hashlib.sha256(provided_password.encode()).hexdigest()

# Define a function to display the login page
def show_login_page(placeholders, mongo_collection):
    # Set the background image here
    picture_url = "https://i.postimg.cc/t45ryBwW/Adobe-Stock-328664888.jpg"
    set_background_image(picture_url)  # Change the URL to your custom image

    placeholders[0].title("Welcome!")
    placeholders[1].markdown("**This is an Implied Volatility Predictor**")
    placeholders[2].write("Please enter your username and password:")

    # Set default values for username and password
    default_username = "ndhu"
    default_password = "1234"

    username = placeholders[3].text_input("Username", value=default_username)
    password = placeholders[4].text_input("Password", type="password", value=default_password)

    # User Login
    if placeholders[5].button("Log In"):
        user = User.get_user(username, mongo_collection)
        if user and user.verify_password(password):
            placeholders[9].success("Logged in successfully!")
            time.sleep(1)  # 等待1秒钟
 
            for placeholder in placeholders:
                placeholder.empty()
            return True
        else:
            placeholders[6].error("Invalid username or password.")
            

    # User Registration
    if placeholders[7].button("Sign Up a New Account"):
        if username and password:
            existing_user = User.get_user(username, mongo_collection)
            if existing_user is None:
                user_id = mongo_collection.count_documents({})
                user = User(user_id, username, password, mongo_collection)
                user.save_to_mongo()
                st.success("Account created successfully!")
                st.session_state.login_status = True
            else:
                placeholders[6].error("Username already exists. Please choose a different username.")
        else:
            placeholders[6].error("Please enter a username and password.")

    # Add a new placeholder for the GitHub hyperlink
    placeholders[8].markdown(
        f'<p style="text-align: center; color: black; margin-top: 15px;"><a href="https://github.com/KuanlinBilly/Implied-Volatility-Predictor" target="_blank">Click here</a> to see the source code on GitHub</p>',
        unsafe_allow_html=True,
    )

     

# Define a function to display the main content of the web app
def show_main_content():
 
    reset_background()  # Add this line to reset the background
    implied_volatility_predictor()

# Main function to display the login page or the main content depending on the login status
def main(mongo_collection):
    if "login_status" not in st.session_state:
        st.session_state.login_status = False

    if not st.session_state.login_status:
        placeholders = [st.empty() for _ in range(10)]
        st.session_state.login_status = show_login_page(placeholders, mongo_collection) 
        
    if st.session_state.login_status: #當用戶成功登錄時，應用將顯示主內容
        show_main_content()

if __name__ == "__main__":
    main(mongo_collection)
