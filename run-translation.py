import streamlit as st
import translation

#page = st.sidebar.selectbox("选择模式", ["智能翻译 📖"])

#if page == "国际工程管理智能专用翻译":
st.title("国际工程管理智能专用翻译 📖")

if "messages_human" not in st.session_state:
    st.session_state["messages_human"] = []
if "messages_assistant" not in st.session_state:
    st.session_state["messages_assistant"] = []

messages_human = st.session_state["messages_human"]
messages_assistant = st.session_state["messages_assistant"]
max_length = max(len(messages_human), len(messages_assistant))

for i in range(max_length):
    if i < len(messages_human) and messages_human[i] is not None:
        with st.chat_message("user", avatar='🧐'):
            st.markdown(messages_human[i])
    if i < len(messages_assistant) and messages_assistant[i] is not None:
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown(messages_assistant[i])

question = st.chat_input("Shift+Enter换行，Enter发送")
history_data = []
if question:
    st.session_state["messages_human"].append(question)
    with st.chat_message("user", avatar='🧐'):
        st.markdown(question)

    answer, history_data = translation.run_model(question, history_data)

    st.session_state["messages_assistant"].append(answer)
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown(answer)

container = st.container()
container.write("")

if container.button("清空对话"):
    st.session_state["messages_human"] = []
    st.session_state["messages_assistant"] = []
