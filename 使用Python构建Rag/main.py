from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from sentence_transformers import CrossEncoder
from openai import OpenAI
import os

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

def generate_answer(query: str, retrieved_chunks: List[str]):
    prompt = ""
    prompt += "# [信息开始]\n"
    for i, chunk in enumerate(retrieved_chunks):
        prompt += f"信息{i+1}: {chunk}\n"
    prompt += "# [信息结束]\n"
    prompt += f"请根据上述信息回答用户的问题：{query}"
    print(prompt)
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen3.5-flash",
        messages=[
            {"role": "system", "content": "你将根据文本信息回答用户的问题。请仔细阅读信息并给出准确的回答。不要编造，如果信息不足以回答问题，请说明信息不足。最后的回答格式为：\n\n回答：你的回答内容\n\n,准确干练，不要多余的内容。"},
            {"role": "user", "content": prompt},
        ]
    )
    # print(completion.model_dump_json())
    return completion.choices[0].message.content

def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]

def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]  # type: ignore

def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )

def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()

def split_text(text_path: str) -> List[str]:
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [chunk.strip() for chunk in text.split("---") if chunk.strip()]

def main(query: str):
    text_chunks = split_text("doc.md")
    # for chunk in text_chunks:
        # print(chunk)
    # print(f"Total chunks: {len(text_chunks)}")
    embeddings = [embed_chunk(chunk) for chunk in text_chunks]
    # print(text_chunks[0])
    # print(embeddings[0])
    save_embeddings(text_chunks, embeddings)
    results = retrieve(query, top_k=8)
    # print("Embeddings saved successfully.")
    # print("Retrieved results:")
    # for doc in results:
    #     print("-" * 40)
    #     print(doc)
    results = rerank(query, results, top_k=5)
    # print("\nReranked results:")
    # for doc in results:
    #     print("-" * 40)
    #     print(doc)
    print(generate_answer(query, results))

if __name__ == "__main__":
    main("我最终将书怎么处理掉了？")
