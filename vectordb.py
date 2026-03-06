from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

# ── 1. 모델 설정 ──────────────────────────────────────
# TODO Gemini, Claude 등 크래딧을 충전해야 응답 품질을 높일 수 있습니다.
Settings.llm = Ollama(
    model="llama3.2",
    request_timeout=120.0,
    system_prompt=(
        "You are a technical document assistant. "
        "ALWAYS respond in Korean only. "
        "NEVER mix other languages. "
        "Only use information from the provided documents. "
        "If not found, say '문서에서 찾을 수 없습니다'."
    )
)

# bge-m3 → 한국어 질문 + 영어 문서 동시 지원
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)

Settings.chunk_size = 512
Settings.chunk_overlap = 50

# ── 2. Vector DB 연결 ─────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("research_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ── 3. 인덱싱 (기존 DB 있으면 재사용) ────────────────
if chroma_collection.count() > 0:
    print(f"✅ 기존 인덱스 재사용 ({chroma_collection.count()}개 청크)")
    index = VectorStoreIndex.from_vector_store(vector_store)
else:
    print("📄 PDF 로딩 중...")
    documents = SimpleDirectoryReader("./docs").load_data()
    print(f"✅ {len(documents)}페이지 로드 완료")
    print("🔍 인덱싱 중...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("✅ 인덱싱 완료")

print(f"📦 저장된 청크 수: {chroma_collection.count()}")

# ── 4. 질문/답변 ──────────────────────────────────────
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"
)

print("\n💬 질문을 입력하세요 (종료: q)\n")
while True:
    question = input("질문: ").strip()
    if question.lower() == "q":
        break

    response = query_engine.query(question)
    print(f"\n📝 답변:\n{response}\n")

    print("📚 참조 출처:")
    for node in response.source_nodes:
        fname = node.metadata.get("file_name", "알 수 없음")
        score = round(node.score, 3) if node.score else "-"
        print(f"  - {fname} (유사도: {score})")
    print("-" * 50)