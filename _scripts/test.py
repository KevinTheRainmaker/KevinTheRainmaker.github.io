import os
import openai
import numpy as np
import faiss
import json
from update_knowledge_base import get_embedding

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        print(f"[!] Embedding 오류: {e}")
        return None

def load_index_and_metadata(index_path="vector.index", metadata_path="metadata.json"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def search_similar_docs(query, index, metadata, top_k=3):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []

    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, top_k)
    return [metadata[i]["text"] for i in indices[0] if i < len(metadata)]

def generate_answer(query, context_texts):
    prompt = (
        "다음 문서를 참고하여 질문에 답해줘.\n\n"
        "문서:\n" + "\n---\n".join(context_texts) +
        f"\n\n질문: {query}\n답변:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 친절한 블로그 도우미야. 질문에 대해 정확하게 응답해야해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[!] Chat 생성 오류: {e}")
        return "답변을 생성하는 데 실패했습니다."

def ask(query):
    index, metadata = load_index_and_metadata()
    context_texts = search_similar_docs(query, index, metadata)
    if not context_texts:
        return "관련 문서를 찾을 수 없습니다."
    answer = generate_answer(query, context_texts)
    return answer

if __name__ == "__main__":
    print('\n\n')
    query = "이 사람의 현재 어떤 일을 하고 있나요? 주요 관심사는 무엇인가요?"
    print(f'질문: {query}')
    print(f'답변: {ask(query)}')
