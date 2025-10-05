[▶ Live Demo](https://portfolio-etl-cli.streamlit.app/)


# RAG Indexer & Search（Streamlit + CLI）

アップロードした文書を **ETL → ベクトル化 → 近傍検索（RAGの下回り）** までノーコードで実行。
スコープ（req / manuals / misc / 全部）ごとにインデックスを分離し、結果に出典（ファイル名＋ページ）と距離を表示します。

## 使い方（3行）
1) `pip install -r requirements.txt`  
2) プロジェクト直下に `.env` を作成し `OPENAI_API_KEY` を設定（埋め込み/回答生成で使用）  
3) `python -m streamlit run app/app.py`

> CLI派は下の「CLIモード」を参照。

---

## できること
- pdf / docx / txt / md をアップロード → **ワンクリックでベクトル化**（FAISS）
- **検索スコープ**：`req` / `manuals` / `misc` / `全部` を選んでインデックス作成・検索
- 検索結果：**ファイル名 + ページ + 距離(distance)** を表示、本文の**簡易ハイライト**
- ログ：`logs/run.jsonl` に `ui_ingest / ui_query`（scope, latency など）を記録
- （任意）上位チャンクから**短い回答生成**（.env の API キーが必要）

---

## ディレクトリ
```
portfolio-etl-cli/
├─ app/ # Streamlit UI
│ └─ app.py
├─ data_raw/ # アップロード先（中に req/manuals/misc を作成）
│ ├─ req/
│ ├─ manuals/
│ └─ misc/
├─ storage/ # スコープ別FAISS（faiss_req, faiss_manuals, faiss_misc, faiss_all）
├─ logs/ # JSONLログ（ui_ingest / ui_query）
├─ scripts/ # CLI（ingest.py / query.py）
├─ .env # OPENAI_API_KEY など
└─ requirements.txt
```

---

## 使い方（UI）
1. 左サイドバーの **「検索スコープ」** を選択（例：`req`）  
2. 画面で **pdf / docx / txt / md** をアップロード（選んだスコープのフォルダに保存）  
3. **「📦 インデックスを再構築」** をクリック（ファイル数・チャンク数・処理時間が表示）  
4. キーワードを入力して **🔎 検索**  
   - **MMR** をONにすると重複が減って多様性UP  
   - 見出しに **`<ファイル名> p.<ページ> (distance: 0.xxxx)`** が表示されます

> 横断検索したい場合はスコープを **「全部」** にして一度「📦 再構築」してください。

---

## 動作確認用クエリ（例）
- **req**：`納期` / `データ更新フロー` / `社内検索`  
- **manuals**：`ペアリング` / `エラー` / `再起動`  
- **misc**：自由（FAQ, レシピなど雑多を隔離）

---

## CLIモード（任意）
> `data_raw/` 以下（サブフォルダ含む）を一括インデックス化します。

```bash
# インデックス作成（FAISSを storage/faiss_index/ に保存）
python scripts/ingest.py

# 検索（上位5件を表示）
python scripts/query.py "テスト質問"
```

## 注意事項

- **機微情報はアップロードしない**でください（ローカル検証用の想定）。
- 画像ベースPDFは現状 **OCR未対応**（`pypdf` のテキスト抽出のみ）。
- Windows で FAISS を使う場合、**プロジェクトパスに日本語など非ASCIIを含めない**のが安全です（例: `C:\dev\...`）。

## よくある詰まり
- 「インデックスがありません」 → 先に対象スコープで **「📦 インデックスを再構築」** を実行。
- 出典が `unknown` と出る → 分割時の **metadata 引き継ぎ**が必要（`split_documents_smart` を参照）。
- Git Bash で `streamlit` が見つからない → `python -m streamlit run app/app.py` で起動。

## ログ例
`logs/run.jsonl` に追記されます：
```text
{"event":"ui_ingest","scope":"req","num_files":1,"num_chunks":2,"latency_ms":1493,...}
{"event":"ui_query","scope":"req","query":"納期","k":5,"mmr":false,"latency_ms":483,...}
```

## （任意）デプロイ：Streamlit Community Cloud

1. GitHubへpush → Streamlit Cloudでリポを選択
2. Secrets に以下を追加
    ```
    OPENAI_API_KEY=xxxxx
    ```
3. Entry point を app/app.py に設定
4. Deploy をクリック

## ライセンス
MIT