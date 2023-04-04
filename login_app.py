# add login page for app3
import streamlit as st
import hashlib
from predictor_app import * 

# Function to verify the user's password (you should replace this with your actual authentication method)
def verify_password(stored_password, provided_password):
    return stored_password == hashlib.sha256(provided_password.encode()).hexdigest()

# Define a function to display the login page
def show_login_page(placeholders):
    my_username = "ndhu"
    my_password = "1234"
    placeholders[0].title("Welcome!")
    placeholders[1].write("Please enter your username and password:")

    username = placeholders[2].text_input("Username", value = my_username ) #為了方便，先預設填入帳密
    password = placeholders[3].text_input("Password", type="password", value= my_password)

    if placeholders[4].button("Login"):
        # You should replace these credentials with your actual username and password
        stored_username = my_username
        stored_password = hashlib.sha256(my_password.encode()).hexdigest()

        if username == stored_username and verify_password(stored_password, password):
            for placeholder in placeholders:
                placeholder.empty()
            return True
        else:
            placeholders[5].error("Incorrect username or password")
            return False

# Define a function to display the main content of the web app
def show_main_content():
    implied_volatility_predictor()

# Main function to display the login page or the main content depending on the login status
def main():
    if "login_status" not in st.session_state:
        st.session_state.login_status = False

    if not st.session_state.login_status:
        placeholders = [st.empty() for _ in range(6)]
        st.session_state.login_status = show_login_page(placeholders)

    if st.session_state.login_status:
        show_main_content()

if __name__ == "__main__":
    main()
