import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from openai import OpenAI
import polars as pl
import os

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
        model="gpt-4o-2024-08-06",
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

USER_NAME = os.environ.get("USER_NAME")
PASSWORD = os.environ.get("PASSWORD")
demo.launch(server_name="0.0.0.0", server_port=7860, auth=(USER_NAME, PASSWORD))
