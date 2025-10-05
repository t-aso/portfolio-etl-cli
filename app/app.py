import os, time, json, shutil
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain_community.vectorstores import FAISS

# ====== 初期設定 ======
load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "storage"
LOG_DIR = ROOT / "logs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="RAG Indexer & Search", page_icon="📚", layout="wide")

# ====== ユーティリティ ======
def write_log(event: str, payload: dict):
    payload = {"event": event, **payload, "ts": int(time.time())}
    with open(LOG_DIR / "run.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def load_docs_from_dir(dirpath: Path):
    docs = []
    for p in dirpath.glob("**/*"):
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

# ====== スプリット（賢く切る） ======
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document

def split_documents_smart(raw_docs, chunk_size: int, overlap: int):
    """DOC_IDやMarkdown見出しを境界にしてから、最後に再帰スプリッタで整える"""
    chunks = []
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]
    )
    for d in raw_docs:
        text = d.page_content
        parts = re.split(r"\n(?=DOC_ID:\s*)", text) if "DOC_ID:" in text else [text]
        for part in parts:
            # Markdown見出しで切れるなら優先
            try:
                md_docs = md_splitter.split_text(part)
                for m in md_docs:
                    m.metadata = {**d.metadata, **m.metadata}  # 元のmetadata（source, page 等）を上書きマージ
                docs_lvl = md_docs if md_docs else [Document(page_content=part, metadata=d.metadata)]
            except Exception:
                docs_lvl = [Document(page_content=part, metadata=d.metadata)]
            # 最後に長さ調整
            rc = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=overlap,
                separators=["\n\n","\n","。","、"," ",""]
            )
            chunks.extend(rc.split_documents(docs_lvl))
    return chunks

def build_index(chunk_size: int, overlap: int) -> Tuple[int, int, int]:
    t0 = time.time()
    raw_docs = load_docs_from_dir(DATA_DIR)
    if not raw_docs:
        return 0, 0, 0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    docs = splitter.split_documents(raw_docs)
    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(str(OUT_DIR / "faiss_index"))
    dt_ms = int((time.time() - t0) * 1000)
    write_log("ui_ingest", {
        "num_files": len([p for p in DATA_DIR.glob("**/*") if p.is_file()]),
        "num_chunks": len(docs),
        "latency_ms": dt_ms,
        "chunk_size": chunk_size,
        "overlap": overlap
    })
    return len(raw_docs), len(docs), dt_ms

def load_index() -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        str(OUT_DIR / "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True
    )

def do_search(vs: FAISS, query: str, k: int, use_mmr: bool):
    if use_mmr:
        docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=max(10, k*3))
        scores = [None] * len(docs)
        return list(zip(docs, scores))
    else:
        return vs.similarity_search_with_score(query, k=k)

def move_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile], dest: Path):
    dest.mkdir(exist_ok=True, parents=True)
    saved = []
    for uf in uploaded_files:
        # 拡張子チェック
        if not any(uf.name.lower().endswith(ext) for ext in (".pdf", ".docx", ".txt", ".md")):
            continue
        outpath = dest / uf.name
        with open(outpath, "wb") as f:
            shutil.copyfileobj(uf, f)
        saved.append(outpath)
    return saved

# ====== サイドバー ======
with st.sidebar:
    st.header("⚙️ 設定")
    # ★追加：検索/インデックスの対象スコープ
    scope = st.selectbox("検索スコープ", ["全部", "req", "manuals", "misc"], index=0)
    chunk_size = st.slider("Chunk size", 300, 2000, 800, 50)
    overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
    k = st.slider("Top-K", 1, 20, 5, 1)
    use_mmr = st.toggle("MMR（重複抑制）", value=False)
    gen_answer = st.toggle("回答文を生成（要API）", value=False,
        help="上位チャンクから根拠付きの短い回答文を生成します。")
    st.caption("※ まずはアップロード→インデックス再構築→検索。")

# ====== インデックス構築（scope対応 & スマート分割） ======
def build_index(chunk_size: int, overlap: int, scope: str) -> tuple[int,int,int,str]:
    t0 = time.time()
    target_dir = DATA_DIR if scope == "全部" else DATA_DIR / scope
    raw_docs = load_docs_from_dir(target_dir)
    if not raw_docs:
        return 0, 0, 0, target_dir.as_posix()

    docs = split_documents_smart(raw_docs, chunk_size, overlap)

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(docs, embeddings)

    index_name = "faiss_all" if scope == "全部" else f"faiss_{scope}"
    vs.save_local(str(OUT_DIR / index_name))

    dt_ms = int((time.time() - t0) * 1000)
    write_log("ui_ingest", {
        "scope": scope,
        "dir": target_dir.as_posix(),
        "num_files": len([p for p in target_dir.glob("**/*") if p.is_file()]),
        "num_chunks": len(docs),
        "latency_ms": dt_ms,
        "chunk_size": chunk_size,
        "overlap": overlap
    })
    return len(raw_docs), len(docs), dt_ms, target_dir.as_posix()

def load_index(scope: str) -> FAISS:
    embeddings = OpenAIEmbeddings()
    index_name = "faiss_all" if scope == "全部" else f"faiss_{scope}"
    return FAISS.load_local(
        str(OUT_DIR / index_name),
        embeddings,
        allow_dangerous_deserialization=True
    )

# ====== メイン：アップロード ======
st.subheader("1) ドキュメントをアップロード")
uploaded = st.file_uploader(
    f"{' → data_raw/' + scope if scope != '全部' else '（※アップロード時はスコープを選択してください）'}",
    type=["pdf","docx","txt","md"],
    accept_multiple_files=True
)

cols = st.columns(2)
if uploaded:
    upload_dir = DATA_DIR if scope == "全部" else DATA_DIR / scope
    saved_paths = move_uploaded_files(uploaded, upload_dir)
    with cols[0]:
        st.success(f"保存: {len(saved_paths)} 件 → {upload_dir.as_posix()}")
    with cols[1]:
        st.write("\n".join([p.name for p in saved_paths]) or "-")

reindex = st.button("📦 インデックスを再構築する（ingest）")
if reindex:
    with st.status(f"インデックス構築中…（scope={scope}）", expanded=True) as status:
        n_files, n_chunks, dt_ms, where = build_index(chunk_size, overlap, scope)
        if n_files == 0:
            st.warning("対象フォルダにファイルがありません。先にアップロードしてください。")
            status.update(label="構築失敗", state="error")
        else:
            st.write(f"Scope: {scope}  /  Path: {where}")
            st.write(f"ファイル数: {n_files} / チャンク数: {n_chunks} / 時間: {dt_ms} ms")
            status.update(label="構築完了", state="complete")

st.divider()

# ====== メイン：検索 ======
st.subheader("2) 検索（ベクトル近傍）")
query = st.text_input("質問・キーワードを入力", value="社内検索システムについて教えて")
col1, col2 = st.columns([1,1])
search_clicked = col1.button("🔎 検索する")
hint = col2.caption("※ 例：『納期は？』『データ更新フロー』など資料に含まれる語で。")

if search_clicked and query.strip():
    try:
        vs = load_index(scope)
    except Exception as e:
        st.error("インデックスがありません。先に『インデックスを再構築』を実行してください。")
        st.exception(e)
    else:
        t0 = time.time()
        results = do_search(vs, query.strip(), k=k, use_mmr=use_mmr)
        dt_ms = int((time.time() - t0) * 1000)
        write_log("ui_query", {"scope": scope, "query": query, "k": k, "mmr": use_mmr, "latency_ms": dt_ms})

        st.caption(f"検索時間: {dt_ms} ms / Top-K: {k} / {'MMR: ON' if use_mmr else 'MMR: OFF'}")

        if not results:
            st.warning("該当が見つかりませんでした。キーワードを変えるか、資料を追加してください。")
        else:
            for idx, item in enumerate(results, start=1):
                if use_mmr:
                    doc, score = item[0], None
                else:
                    doc, score = item

                src = Path(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page")
                loc = f"{src.name}" + (f"  p.{page+1}" if isinstance(page, int) else "")
                snippet = doc.page_content[:400].replace("\n", " ")

                with st.container(border=True):
                    st.markdown(f"**[{idx}] {loc}**" + (f"  _(distance: {score:.4f})_" if score is not None else ""))
                    st.write(snippet + " …")
                    with st.expander("メタデータ"):
                        st.json(doc.metadata)

            # ====== 回答生成（オプション） ======
            if gen_answer:
                st.divider()
                st.subheader("📝 回答（上位文書に基づく短い要約）")
                try:
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                    # 短い根拠付きの要約（幻覚抑制のため、参照テキストのみを材料に）
                    context = "\n\n".join([d[0].page_content if not use_mmr else d.page_content for d in results[:k]])
                    prompt = (
                        "あなたはドキュメントQAアシスタントです。以下のコンテキストの範囲内で、"
                        "ユーザーの質問に日本語で簡潔に答えてください。出典となるキーワードや章タイトルがあれば文末に括弧で付けてください。"
                        "コンテキスト外の推測は避け、『不明』と述べてください。\n\n"
                        f"# 質問:\n{query}\n\n# コンテキスト:\n{context}"
                    )
                    ans = llm.invoke(prompt).content
                    st.success(ans)
                except Exception as e:
                    st.warning("回答生成に失敗しました（APIキーや権限をご確認ください）。")
                    st.exception(e)

# ====== フッター ======
st.divider()
with st.expander("🧪 使い方ヒント", expanded=False):
    st.markdown("""
- **手順**：①ドキュメントをアップロード → ②「インデックスを再構築」 → ③検索  
- **MMR**：似た内容の重複を減らし、多様性を上げます（距離スコアは表示されません）。  
- **Chunk設計**：要約中心は短め（500/100）、仕様書中心は長め（1000/200）を試すのがおすすめ。  
- **PDFの注意**：画像だけのPDFはテキストが取れません（OCRは別途）。  
- **保存場所**：`data_raw/`（原文書）、`storage/faiss_index/`（ベクトル）、`logs/run.jsonl`（ログ）
""")