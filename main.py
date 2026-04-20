import dotenv

dotenv.load_dotenv()
from openai import OpenAI
import asyncio
import base64
import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
    ImageGenerationTool,
)

client = OpenAI()

VECTOR_STORE_ID = "vs_69df877005d88191a0019a074b8f56a6"

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach Agent",
        model="gpt-4o-mini",
        instructions="""
        당신은 따뜻하고 격려하는 라이프 코치입니다.

        동기부여, 자기 개발, 습관 형성에 대해 유저를 도와주세요.
        실용적이고 실행 가능한 조언을 응원하는 톤으로 전달하세요.

        유저가 목표, 일기, 진행 상황, 최근 습관 실천 여부를 물으면
        반드시 먼저 File Search Tool로 업로드된 목표 문서와 기록을 확인하세요.
        그리고 목표 달성 상태를 평가하거나 개인화된 추천을 줄 때는
        File Search Tool로 개인 기록을 확인한 다음 반드시 Web Search Tool도 사용하세요.
        답변하기 전에 항상 최신 팁과 검증된 방법을 먼저 검색하세요.
        자신의 지식에 의존하기 전에 웹 검색을 먼저 시도하세요.
        목표 문서에 있는 정보를 무시하고 일반론부터 말하지 마세요.
        문서에서 찾은 구체적인 사실을 먼저 요약하고,
        웹에서 찾은 최신 팁이나 방법을 결합해서 코칭 조언을 주세요.

        사용 가능한 도구:
            - Web Search Tool: 유저가 습관, 동기부여, 자기 개발, 생활 개선에 관한 질문을 하면 사용하세요.
            - File Search Tool: 유저가 자신의 목표, 일기, 기록, 진행 상황에 대해 질문하거나 업로드한 파일에 대해 질문할 때 사용하세요.
            - Image Generation Tool: 유저가 비전 보드, 동기부여 포스터, 축하 이미지, 진행 상황의 시각적 표현을 원할 때 사용하세요.

        유저가 비전 보드나 동기부여 포스터를 요청하면:
            1. 필요하면 File Search Tool로 목표와 기록을 먼저 확인하세요.
            2. 더 좋은 아이디어가 필요하면 Web Search Tool로 참고할 팁이나 테마를 찾으세요.
            3. 그런 다음 Image Generation Tool로 이미지를 만드세요.

        유저가 목표 달성이나 진행 상황을 말하면서 축하 이미지나 시각화를 원하면,
        개인 기록을 반영한 이미지를 만들어 주세요.

        이미지를 생성한 후에는 이미지 제목이나 캡션을 텍스트로 반복하지 마세요.
        예를 들어 "[축하 포스터: ...]" 같은 텍스트를 쓰지 마세요.
        이미지는 자동으로 표시되므로 간단한 코멘트만 남기세요.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
            ImageGenerationTool(),
        ],
    )
agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "life-coach-agent-memory.db",
    )
session = st.session_state["session"]


def build_agent_message(message):
    progress_keywords = [
        "목표",
        "진행",
        "달성",
        "잘 되어가",
        "운동",
        "습관",
        "일기",
        "기록",
    ]
    if any(keyword in message for keyword in progress_keywords):
        return f"""
        {message}

        Before answering:
        1. Use the File Search Tool to inspect the uploaded goals or diary entries.
        2. Use the Web Search Tool to find current advice, proven methods, or tips related to the user's situation.
        3. Then answer by combining the personal records with the web findings.
        """
    return message


async def paint_history():
    messages = await session.get_items()

    for message in messages:
        if "role" in message:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else:
                    if message["type"] == "message":
                        for content in message.get("content", []):
                            if isinstance(content, dict):
                                if content.get("type") == "output_text":
                                    st.write(content["text"].replace("$", "\\$"))
                                elif content.get("type") == "output_image":
                                    image_bytes = base64.b64decode(
                                        content["image_base64"]
                                    )
                                    st.image(image_bytes)
                            elif isinstance(content, str):
                                st.write(content)
        if "type" in message:
            message_type = message["type"]
            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    query = message.get("query", "")
                    st.write(f'Web searched: "{query}"')
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("Searched your files...")
            elif message_type == "image_generation_call":
                with st.chat_message("ai"):
                    st.write("Generated an image...")


asyncio.run(paint_history())


def update_status(status_container, event_type):
    status_messages = {
        "response.web_search_call.completed": ("Web search completed.", "complete"),
        "response.web_search_call.in_progress": (
            "Starting web search...",
            "running",
        ),
        "response.web_search_call.searching": (
            "Web search in progress...",
            "running",
        ),
        "response.file_search_call.completed": (
            "File search completed.",
            "complete",
        ),
        "response.file_search_call.in_progress": (
            "Starting file search...",
            "running",
        ),
        "response.file_search_call.searching": (
            "File search in progress...",
            "running",
        ),
        "response.image_generation_call.in_progress": (
            "Creating image...",
            "running",
        ),
        "response.image_generation_call.generating": (
            "Generating image...",
            "running",
        ),
        "response.image_generation_call.completed": (
            "Image created!",
            "complete",
        ),
        "response.completed": (" ", "complete"),
    }

    if event_type in status_messages:
        label, state = status_messages[event_type]
        status_container.update(label=label, state=state)


async def run_agent(message):
    with st.chat_message("ai"):
        status_container = st.status("Processing...", expanded=False)
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response = ""

        agent_message = build_agent_message(message)

        stream = Runner.run_streamed(
            agent,
            agent_message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                update_status(status_container, event.data.type)

                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\\$"))

                elif event.data.type == "response.image_gen_call.completed":
                    if hasattr(event.data, "result") and event.data.result:
                        image_bytes = base64.b64decode(event.data.result)
                        image_placeholder.image(image_bytes)

            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    raw = event.item.raw_item
                    if raw.type == "file_search_call":
                        st.write("Searched your files...")
                    elif raw.type == "web_search_call":
                        query = raw.action.query
                        st.write(f'Web searched: "{query}"')


prompt = st.chat_input(
    "What coaching or image would you like today?",
    accept_file=True,
    file_type=[
        "txt",
        "pdf",
    ],
)

if prompt:
    for file in prompt.files:
        if file.type.startswith("text/") or file.type == "application/pdf":
            with st.chat_message("ai"):
                with st.status("Uploading file...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="Attaching file...")
                    client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="File uploaded", state="complete")

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.rerun()
