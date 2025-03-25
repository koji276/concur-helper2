import streamlit as st
import subprocess

def main():
    st.title("Debug: Show installed packages")

    # Show weaviate-client info
    st.write("## pip show weaviate-client")
    cmd = subprocess.run(["pip", "show", "weaviate-client"], capture_output=True, text=True)
    st.text(cmd.stdout)

    # Show all installed packages
    st.write("## pip freeze")
    cmd2 = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    st.text(cmd2.stdout)

if __name__ == "__main__":
    main()
