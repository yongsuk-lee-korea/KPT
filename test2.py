import streamlit as st
from langchain_core.messages.chat import ChatMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


# 임베딩 모델 정의
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 저장된 데이터를 로드
vectorstore = FAISS.load_local(
    folder_path="faiss_db",
    index_name="k-ics",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever()


# API KEY 정보로드
load_dotenv()

st.title("K-ICS Helper📝")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 대화초기화 버튼
    clear_btn = st.button("대화 초기화")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 매세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인을 생성
# 체인을 생성
def create_chain():
    # prompt | llm | output_parser
    prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    you must include 'page' number and document name.
    Answer in Korean.

    #Context: 
    {context}

    #Question:
    {question}

    #Answer:"""
    )
 
    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    # 출력파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = (
        {"context": lambda question: retriever.get_relevant_documents(str(question)),  # 질문을 문자열로 변환
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    return chain

# 초기화 버튼이 눌리면?
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 생성
print_messages()

# 사용자의 입력
user_input = st.chat_input("K-ICS 에 대해 물어보세요")

# 만약에 사용자의 입력이 들어오면
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)

    # chain 을 생성
    chain = create_chain()
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간을 만들어서 여기에 토큰을 스트리밍 출력)
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다
    add_message("user", user_input)
    add_message("assistant", ai_answer)



# retriever 테스트
test_query = "K-ICS 문서의 일부 내용 예시"  # 테스트할 질문
docs = retriever.get_relevant_documents(test_query)

# 반환된 문서 출력
if docs:
    for doc in docs:
        print(doc.page_content)  # 각 문서의 내용 출력
else:
    print("관련된 문서를 찾을 수 없습니다.")
