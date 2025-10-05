import sys, time, json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "storage"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def main():
    # 質問文を引数 or 入力から取得
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("質問> ").strip()
    if not q:
        print("⚠️ 質問文を入力してください。例：python scripts/query.py \"納期は？\"")
        return

    t0 = time.time()

    # 保存済みのインデックスを読み込む
    vs = FAISS.load_local(
        str(OUT_DIR / "faiss_index"),
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )

    # 上位5件を検索
    docs = vs.similarity_search(q, k=5)
    dt = int((time.time()-t0)*1000)

    # 結果表示
    print("\n--- 検索結果（上位5件） ---")
    if not docs:
        print("該当なしでした。クエリを変えるか、data_raw/ の資料を増やしてください。")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        snippet = d.page_content[:160].replace("\n"," ")
        print(f"[{i}] {src}\n{snippet}…\n")

    # ログ保存
    with open(LOG_DIR / "run.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"event":"query","latency_ms":dt,"k":5,"query":q}, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
