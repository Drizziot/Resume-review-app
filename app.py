import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS


# Load environment variables from .env file
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Resume Review Assistant",
    page_icon="📄",
    layout="wide"
)


# Check for API key
def check_api_key():
    """Check if Groq API key is available"""
    # First check environment variable
    api_key = os.getenv("GROQ_API_KEY")

    # If not in environment, check Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            api_key = None

    # If still no key, ask user to input it
    if not api_key:
        st.warning("🔑 Groq API key required!")
        st.markdown("""
        **To get started:**
        1. Get your free API key from [Groq Console](https://console.groq.com/)
        2. Enter it below or set it as an environment variable `GROQ_API_KEY`
        """)

        api_key = st.text_input(
            "Enter your Groq API Key:",
            type="password",
            help="Your API key will not be stored and is only used for this session"
        )

        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("✅ API key set successfully!")
            st.rerun()
        else:
            st.stop()

    return api_key


# Initialize session state variables
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'resume_uploaded' not in st.session_state:
    st.session_state.resume_uploaded = False


# Initialize the language model and embeddings
@st.cache_resource
def initialize_llm(_api_key):
    return ChatGroq(
        api_key=_api_key,  # Explicitly pass the API key
        # You can also use "mixtral-8x7b-32768" or "llama-3.1-8b-instant"
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )


@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def create_qa_chain(vectordb, llm, memory):
    """Create the QA chain with custom prompt template"""
    template = """
    You are a Career Guide. Provide concise and actionable feedback on resumes and specific questions. When asked to rewrite or improve content, offer direct examples and suggestions. Avoid repeating greetings if the conversation is ongoing. Focus on transforming given content into more impactful statements using action verbs and quantifiable results. Do not provide unsolicited career path suggestions unless explicitly requested.


    Conversation History:
    {chat_history}


    Resume Context:
    {context}


    Student Question: {question}


    Response:
    """

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 1}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )


def process_uploaded_file(uploaded_file, api_key):
    """Process the uploaded PDF file and create vector database"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load the PDF
        loader = PyMuPDFLoader(temp_file_path)
        data = loader.load()

        # Create vector database
        embedding = initialize_embeddings()
        vectordb = FAISS.from_documents(
            documents=data,
            embedding=embedding,
            persist_directory=None
        )

        # Create QA chain
        llm = initialize_llm(api_key)
        qa_chain = create_qa_chain(vectordb, llm, st.session_state.memory)

        # Clean up temporary file
        os.remove(temp_file_path)

        return qa_chain, True

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, False


def main():
    # Check API key first
    api_key = check_api_key()

    st.title("📄 Resume Review Assistant")
    st.markdown("Upload your resume and get AI-powered feedback to improve it!")

    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload your resume in PDF format"
        )

        if uploaded_file is not None:
            if st.button("Process Resume", type="primary"):
                with st.spinner("Processing your resume..."):
                    qa_chain, success = process_uploaded_file(
                        uploaded_file, api_key)
                    if success:
                        st.session_state.qa_chain = qa_chain
                        st.session_state.resume_uploaded = True
                        st.success("✅ Resume processed successfully!")
                        st.rerun()

        # Clear conversation button
        if st.session_state.resume_uploaded:
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                st.session_state.memory.clear()
                st.success("Conversation cleared!")
                st.rerun()

        # Reset application button
        if st.button("Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application reset!")
            st.rerun()

    # Main chat interface
    if not st.session_state.resume_uploaded:
        st.info("👆 Please upload your resume using the sidebar to get started!")

        # Show sample questions
        st.subheader("What you can ask:")
        sample_questions = [
            "What are the main strengths of my resume?",
            "What areas of my resume need improvement?",
            "How can I make my work experience more impactful?",
            "Are there any formatting issues with my resume?",
            "How can I better highlight my skills?",
            "What action verbs should I use to improve my bullet points?"
        ]

        for question in sample_questions:
            st.markdown(f"• {question}")

    else:
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {question}")
                st.markdown(f"**AI Career Guide:** {answer}")
                st.divider()

        # Chat input
        user_question = st.chat_input("Ask a question about your resume...")

        if user_question and st.session_state.qa_chain:
            # Add user message to chat
            with st.container():
                st.markdown(f"**You:** {user_question}")

                # Get AI response
                with st.spinner("Analyzing your resume..."):
                    try:
                        result = st.session_state.qa_chain.invoke(
                            {"question": user_question})
                        answer = result['answer']

                        # Display AI response
                        st.markdown(f"**AI Career Guide:** {answer}")

                        # Add to chat history
                        st.session_state.chat_history.append(
                            (user_question, answer))

                        # Update memory
                        st.session_state.memory.chat_memory.add_user_message(
                            user_question)
                        st.session_state.memory.chat_memory.add_ai_message(
                            answer)

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** Be specific with your questions for better feedback. "
        "Ask about particular sections, formatting, or request specific improvements."
    )
    st.markdown(
        "🤖 **Powered by Groq's free API** - Fast and efficient AI responses!")


if __name__ == "__main__":
    main()

