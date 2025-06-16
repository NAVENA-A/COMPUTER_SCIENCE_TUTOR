

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from operator import itemgetter
import streamlit as st

import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyDFRWR-lHyHseM878Z7qNeW6lLkG-TdXbc"

#gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)
st.set_page_config(page_title="AI Assistant")
st.title("Welcome I am AI tutor")

class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)
#chatgpt  =ChatOpenAI(model_name = gemini_model, temperature=0.1,streaming=True)

SYS_PROMPT = """
You are a helpful tutor for Computer Science students and you must answer only the comupter science related questions. 
When answering:
- Use simple language and examples.
- For theory questions: explain using analogies if possible.
- For code: use triple backticks and comment each line.
- If unsure, say: 'Please consult your textbook or instructor.
- Don't answer any questions that are asked from different domain even if it is a generic one.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        MessagesPlaceholder(variable_name="history"),
       ("human", "{input}"),
    ]
)

llm_chain = ( 
  prompt
  | 
  gemini_model
) 

streamlit_msg_history = StreamlitChatMessageHistory()

conversation_chain = RunnableWithMessageHistory(
    llm_chain,
    lambda session_id: streamlit_msg_history,
    input_messages_key="input",
    history_messages_key="history",
)

if len(streamlit_msg_history.messages) == 0:
  #streamlit_msg_history.add_ai_message("How can I help you?")
  pass

for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)
  

if user_prompt := st.chat_input():
  st.chat_message("human").write(user_prompt)
  with st.chat_message("ai"):
    stream_handler = StreamHandler(st.empty())
    config = {"configurable":{"session_id":"any"},
              "callbacks":[stream_handler]}
    response = conversation_chain.invoke({"input": user_prompt}, config)  
    st.write(str(response.content))
