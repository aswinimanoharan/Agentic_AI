import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import random

# 🔐 Hardcoded Gemini API Key (Not recommended for production)
GEMINI_API_KEY = "AIzaSyC4KWzVoB8uGmiOQBqldPC-PoChlIN_8KY"

# 🔍 Search Tool
search = DuckDuckGoSearchRun()

# 💬 Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# 🤖 LangChain Agent Setup
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for answering questions about current events or factual queries"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# 🎨 Streamlit UI
st.set_page_config(page_title="🧠 Real-Time Q&A with Gemini", layout="wide")
st.title("🧠 Ask Me Anything - Powered by Gemini & DuckDuckGo")
st.markdown("Ask questions about the world 🌍, tech 💻, news 🗞️, or anything else!")

# 🧠 Input Area
query = st.text_input("Type your question here 👇")

# 📚 Feature Toggles
col1, col2, col3 = st.columns(3)
with col1:
    summarize = st.toggle("🔍 Smart Answer Summarization", value=False)
with col2:
    show_image = False  # 🚫 Disabled image toggle
with col3:
    show_follow_ups = st.checkbox("💡 Suggest Follow-Up Questions")

# 📊 Trending Questions Sidebar
st.sidebar.title("📊 Trending Questions")
trending_questions = [
    "What's happening in the world today?",
    "Who is the president of the USA?",
    "Latest tech news",
    "What is the stock market doing today?",
    "Weather forecast in Paris"
]
for q in trending_questions:
    st.sidebar.markdown(f"- {q}")

# 🔍 Handle Query
if st.button("Get Answer") and query:
    try:
        with st.spinner("Thinking..."):
            response = agent.run(query)

            # Smart Summarization (basic prompt tuning)
            if summarize:
                summarizer_template = PromptTemplate(
                    template="Summarize the following answer in 2-3 sentences: {answer}",
                    input_variables=["answer"]
                )
                summary_prompt = summarizer_template.format(answer=response)
                response = llm(summary_prompt)

            st.markdown("### ✅ Answer")
            st.success(response)

            # 💡 Follow-Up Suggestions
            if show_follow_ups:
                follow_up_template = PromptTemplate(
                    template="Suggest 3 related follow-up questions for: {question}",
                    input_variables=["question"]
                )
                follow_prompt = follow_up_template.format(question=query)
                suggestions = llm(follow_prompt)
                st.markdown("### 💡 You might also ask:")
                st.info(suggestions)

    except Exception as e:
        st.error("Oops! Something went wrong. Please try again.")
        st.stop()

elif not query:
    st.info("Please enter a question to get started.")
