import os, time, json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# .env を読む
load_dotenv()

# フォルダの場所を決める（このファイルの2つ上）
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "storage"
LOG_DIR = ROOT / "logs"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

def load_docs():
    """data_raw/ 以下の pdf/docx/txt/md を全部読み込む。"""
    docs = []
    for p in DATA_DIR.glob("**/*"):
        if p.is_dir():
            continue
        suf = p.suffix.lower()
        if suf == ".pdf":
            docs += PyPDFLoader(str(p)).load()
        elif suf == ".docx":
            docs += Docx2txtLoader(str(p)).load()
        elif suf in [".txt", ".md"]:
            docs += TextLoader(str(p), encoding="utf-8").load()
    return docs

def main():
    t0 = time.time()
    raw_docs = load_docs()
    if not raw_docs:
        print("⚠️ data_raw/ にファイルがありません。pdf/docx/txt/mdを置いてください。")
        return

    # 文字数ベースで素直に分割（日本語もOK）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    docs = splitter.split_documents(raw_docs)

    # OpenAI の埋め込みを使う（.env の API キー必要）
    embeddings = OpenAIEmbeddings()

    # ベクトル化→FAISSへ保存
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(OUT_DIR / "faiss_index"))

    # シンプルなログ
    log = {
        "event": "ingest",
        "num_files": len([p for p in DATA_DIR.glob('**/*') if p.is_file()]),
        "num_chunks": len(docs),
        "latency_ms": int((time.time()-t0)*1000)
    }
    with open(LOG_DIR / "run.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"✅ Ingest done: {log}")

if __name__ == "__main__":
    main()