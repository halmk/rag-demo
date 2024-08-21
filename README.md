## QdrantとOpenAIのAPIを用いたRAGの構築

### はじめに
こちらの記事「[ゼロからRAGを作るならこんなふうに](https://zenn.dev/minedia/articles/8f4ef7f2daed11)」に触発されたので実際にRAGを構築してみました。

感想としては、ユーザの質問に対してできるだけ関連度の高いドキュメントをできるだけ多く生成AIに渡すことで、AIによる回答精度を上げるというものだと感じました。
コンテキストの質は精度の高い埋め込みモデルを用いることである程度実現できます。また、コンテキストの量はベクトルデータベースから取り出すドキュメントの量を増やし、コンテキストウィンドウの多いLLMを用いることで実現できます。コンテキストウィンドウはLLMで様々で、GPT-4oでは現在128,000トークンまで可能です。Gemini Proでは1,000,000トークンまで可能なので用途に応じて選択する場合もあるかと思います。

今回はベクトル検索のみ使用していますがコンテキストの質を上げられるなら全文検索でも可能です。より質を上げるためには両方を組み合わせたり、検索結果のリランキングを行うなどの工夫が必要になってくるようです。

このブログでは、以下の手順でQdrantとOpenAIのAPIを用いてRAGを構築する方法について説明します。

- データのダウンロードと前処理
- エンベディングの生成
- Qdrantへのドキュメントのアップロード
- 検索と回答生成のプロセス

### 0. 環境構築
Dockerコンテナを用いて、Python/Gradioの実行環境とQdrantを構築します。

```bash
git clone https://github.com/halmk/qdrant-rag-demo.git
cd qdrant-rag-demo
docker compose up
```

### 1. データのダウンロードと前処理
まず、Hugging Faceからデータセットをダウンロードし、Polarsを用いてデータを読み込みます。このデータセットは、日本政府のFAQデータを含んでおり、22,000件の質問と回答が含まれています。[https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k](https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k)

埋め込みの対象テキストとして、`question : 質問 | answer : 回答`という形式になるように、`combined`カラムを追加します。

```python
import polars as pl
df = pl.read_ndjson('hf://datasets/matsuxr/JaGovFaqs-22k/data.jsonl')
df = df.with_columns(
    ("question : " + pl.col("Question") + " | " + "answer : " + pl.col("Answer")).alias("combined")
)
```

### 2. エンベディングの生成
OpenAIのAPIを用いて、質問と回答の結合テキストからエンベディングを生成します。

OpenAIのAPIを使用するには、事前にAPIキーを発行し、環境変数`OPENAI_API_KEY`に設定する必要があります。

ここでは、埋め込みモデルとしてOpenAIが提供している`text-embedding-3-small`を使用していますが、この他にも、OpenAIが提供している`text-embedding-3-large`や、Hugging Faceの`pkshatech/GLuCoSE-base-ja`などを使用することも可能です。

ここではOpenAIのEmbedding APIに対してレコード毎にエンべディングを取得していますが結構時間がかかります（48分かかりました）。`pkshatech/GLuCoSE-base-ja`はローカルで実行できるのでスペックに応じて選択してください（私の環境では2時間ほどかかりました ア）。SentenceTransformersではdeviceを指定することでGPUを使用できます。

```python
    from openai import OpenAI
    client = OpenAI()

    def get_embedding(text, model="text-embedding-3-small"):
        return client.embeddings.create(input = [text], model=model).data[0].embedding
    df = df.with_columns(
        pl.col("combined").map_elements(lambda x: get_embedding(x, model='text-embedding-3-small')).alias("ada_embedding")
    )
    df.write_csv('output/embedded_faq.csv')
```


### 3. Qdrantへのデータのアップロード
生成したエンベディングをQdrantにアップロードします。
Qdrantのクライアントを初期化し、コレクションを作成します。コレクションのベクトルサイズは、OpenAIのエンベディングAPIの出力サイズに合わせておきます。
その後、作成したコレクションにエンべディングとペイロードを挿入していきます。

```python
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(url="http://qdrant:6333")

    client.create_collection(
        collection_name="kankocho_faq",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    points = embeded_df.iter_rows(named=True)
    point_structs = [
        PointStruct(id=i, vector=row["ada_embedding"], payload={
            "question": row["Question"],
            "answer": row["Answer"],
            "copyright": row["copyright"],
            "url": row["url"],
            "combined": row["combined"],
        })
        for i, row in tqdm(enumerate(points))
    ]
    chunk_size = 100
    for chunk in tqdm(point_structs[i:i + chunk_size] for i in range(0, len(point_structs), chunk_size)):
        operation_info = client.upsert(
            collection_name="kankocho_faq",
        wait=True,
        points=chunk,
    )
```


### 4. 検索と回答生成
Gradioを用いて、ユーザーインターフェースを構築します。ユーザーが質問を入力すると、Qdrantから関連するドキュメントを検索し、GPT-4oを用いて回答を生成します。
処理の流れは以下のようになります。
1. ユーザーの質問を受け取る
2. 質問を埋め込みモデル(`text-embedding-3-small`)でエンベディングする
3. Qdrantから関連するドキュメントを検索する
4. 検索結果をGPT-4oに渡し、回答を生成する

Gradioは適当に入出力のUIを指定するだけでPythonのみでリッチなUIを簡単に構築できるのでデモには最適です。

検索結果を渡して回答を生成するときのプロンプトはより工夫したほうが精度が上がると思います。
LLMをよりシステム内で活用するためには、Function CallingやStructured Outputなどを活用するケースもあります。
Structured Outputは`gpt-4o-2024-08-06`から与えられたJSONスキーマに従うように回答を生成することができます。

```python
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI
import polars as pl

qdrant_client = QdrantClient(url="http://qdrant:6333")
openai_client = OpenAI()

def to_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return openai_client.embeddings.create(input = [text], model=model).data[0].embedding

def search_results_to_dataframe(results: list[PointStruct]) -> pl.DataFrame:
    data = []
    for result in results:
        payload = result.payload
        data.append({
            "id": result.id,
            "score": result.score,
            "question": payload.get("question", ""),
            "answer": payload.get("answer", ""),
            "copyright": payload.get("copyright", ""),
            "url": payload.get("url", ""),
            "combined": payload.get("combined", "")
        })
    return pl.DataFrame(data)

def get_response(input: str, context: list[str], use_context: bool = True, model: str ="gpt-4o"):
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"以下の検索結果を引用して、'{input}'に対する回答を生成してください。\n\n検索結果:\n{context}"}
        ] if use_context else [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"'{input}'に対する回答を生成してください。"}
        ],
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip(), response.usage.total_tokens


def query_qdrant_and_gpt(collection_name, question):
    input_embedding = to_embedding(question)
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=input_embedding,
        limit=10
    )
    df_results = search_results_to_dataframe(search_results)
    context = df_results["combined"].to_list()
    response_with_context, response_with_context_tokens = get_response(question, context)
    response_without_context, response_without_context_tokens = get_response(question, [], use_context=False)
    return response_with_context, response_with_context_tokens, response_without_context, response_without_context_tokens, df_results


# Gradioインターフェースの設定
demo = gr.Interface(
    fn=query_qdrant_and_gpt,
    inputs=[
        gr.Dropdown(choices=["kankocho_faq"], label="Collection Name"),
        gr.Textbox(lines=2, placeholder="質問を入力してください", label="Question")
    ],
    outputs=[
        gr.Textbox(lines=2, label="回答 (検索結果を使用)"),
        gr.Number(label="使用トークン数 (検索結果を使用)", precision=0),
        gr.Textbox(lines=2, label="回答 (検索結果を使用しない)"),
        gr.Number(label="使用トークン数 (検索結果を使用しない)", precision=0),
        gr.Dataframe(label="検索結果 (関連ドキュメント)"),
    ],
    title="RAG with Qdrant (embedding: text-embedding-3-small) & GPT-4o",
    description="Qdrantのコレクションと質問を入力すると、GPT-4oが関連するドキュメントの情報を用いて回答を生成します。 質問やドキュメントはOpenAIのtext-embedding-3-smallでエンベディングされています。"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
```

### まとめ
本記事では、QdrantとOpenAIの埋め込みモデルとLLMを組み合わせたRAG（Retrieval-Augmented Generation）を紹介しました。
QdrantとAPIを用いることでローカルでシンプルにRAGを構築することができました。また、Gradioを用いることでデモアプリも簡単に構築しました。


### 参考
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [Gradio Documentation](https://gradio.app/docs/)
