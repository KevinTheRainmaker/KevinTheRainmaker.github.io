import frontmatter
import glob
import os
import openai
import numpy as np
import faiss
import json
from tqdm import tqdm

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_markdown_texts(path):
    md_files = glob.glob(f"{path}/**/*.md", recursive=True)
    texts = []

    for file in md_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                post = frontmatter.load(f)
                content = post.content.strip()
                if content and len(content) > 50:  # 너무 짧은 문서 제외
                    texts.append({"text": content, "source": file})
        except Exception as e:
            print(f"[!] Failed to load {file}: {e}")

    return texts

def chunk_text(text, max_tokens=500):
    # 개선 가능 지점: token 수 기반 분할 (tiktoken 등)
    return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]

def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype="float32")
    except Exception as e:
        print(f"[!] Embedding failed: {e}")
        return None

def build_index(embeddings, metadata_list):
    if not embeddings:
        print("[!] No embeddings to index.")
        return

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # 인덱스 저장
    faiss.write_index(index, "vector.index")

    # 메타데이터 저장
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

def main():
    path = "_pages"
    raw_docs = load_markdown_texts(path)

    embeddings = []
    metadata_list = []

    for doc in tqdm(raw_docs, desc="Embedding documents"):
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            embedding = get_embedding(chunk)
            if embedding is not None:
                embeddings.append(embedding)
                metadata_list.append({
                    "source": doc["source"],
                    "text": chunk
                })

    build_index(embeddings, metadata_list)

if __name__ == "__main__":
    main()
