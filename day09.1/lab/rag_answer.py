"""
rag_answer.py — Sprint 2 + Sprint 3: Retrieval & Grounded Answer
================================================================
Sprint 2 (60 phút): Baseline RAG
  - Dense retrieval từ ChromaDB
  - Grounded answer function với prompt ép citation
  - Trả lời được ít nhất 3 câu hỏi mẫu, output có source

Sprint 3 (60 phút): Tuning tối thiểu
  - Thêm hybrid retrieval (dense + sparse/BM25)
  - Hoặc thêm rerank (cross-encoder)
  - Hoặc thử query transformation (expansion, decomposition, HyDE)
  - Tạo bảng so sánh baseline vs variant

Definition of Done Sprint 2:
  ✓ rag_answer("SLA ticket P1?") trả về câu trả lời có citation
  ✓ rag_answer("Câu hỏi không có trong docs") trả về "Không đủ dữ liệu"

Definition of Done Sprint 3:
  ✓ Có ít nhất 1 variant (hybrid / rerank / query transform) chạy được
  ✓ Giải thích được tại sao chọn biến đó để tune
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

TOP_K_SEARCH = 10    # Số chunk lấy từ vector store trước rerank (search rộng)
TOP_K_SELECT = 3     # Số chunk gửi vào prompt sau rerank/select (top-3 sweet spot)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Singleton clients/models — khởi tạo một lần, dùng lại
_openai_client = None
_cross_encoder_model = None
_bm25_index = None          # (bm25_object, all_chunks) tuple
_st_embedding_model = None


# =============================================================================
# RETRIEVAL — DENSE (Vector Search)
# =============================================================================

def retrieve_dense(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Dense retrieval: tìm kiếm theo embedding similarity trong ChromaDB.

    Returns:
        List các dict với "text", "metadata", "score"
    """
    import chromadb
    from index import get_embedding, CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB cosine distance = 1 - similarity → score = 1 - distance
    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": float(1.0 - dist),
        })

    return chunks


# =============================================================================
# RETRIEVAL — SPARSE / BM25 (Keyword Search)
# =============================================================================

def _load_bm25_index() -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Lazy-load BM25 index. Build từ ChromaDB nếu chưa có.
    Trả về (bm25_object, all_chunks).
    """
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank-bm25 chưa được cài.\n"
            "Chạy: pip install rank-bm25"
        )

    import chromadb
    from index import CHROMA_DB_DIR

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection("rag_lab")

    # Lấy tất cả chunks
    results = collection.get(include=["documents", "metadatas"])
    all_chunks = [
        {"text": doc, "metadata": meta, "score": 0.0}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]

    # Tokenize đơn giản — split theo khoảng trắng, lowercase
    # Với tiếng Việt có thể dùng underthesea.word_tokenize để tốt hơn
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    _bm25_index = (bm25, all_chunks)
    return _bm25_index


def retrieve_sparse(query: str, top_k: int = TOP_K_SEARCH) -> List[Dict[str, Any]]:
    """
    Sparse retrieval: tìm kiếm theo keyword (BM25).

    Mạnh ở: exact term, mã lỗi, tên riêng (ví dụ: "ERR-403", "P1", "refund")
    Hay hụt: câu hỏi paraphrase, đồng nghĩa
    """
    bm25, all_chunks = _load_bm25_index()

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k indices theo score giảm dần
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []
    max_score = max(scores[top_indices[0]], 1e-9)  # tránh chia cho 0
    for idx in top_indices:
        chunk = dict(all_chunks[idx])
        chunk["score"] = float(scores[idx] / max_score)  # normalize về [0, 1]
        results.append(chunk)

    return results


# =============================================================================
# RETRIEVAL — HYBRID (Dense + Sparse với Reciprocal Rank Fusion)
# =============================================================================

def retrieve_hybrid(
    query: str,
    top_k: int = TOP_K_SEARCH,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval: kết hợp dense và sparse bằng Reciprocal Rank Fusion (RRF).

    RRF_score(doc) = dense_weight * 1/(60 + dense_rank)
                   + sparse_weight * 1/(60 + sparse_rank)

    60 là hằng số RRF tiêu chuẩn — giảm ảnh hưởng của rank đầu tiên.
    """
    dense_results = retrieve_dense(query, top_k=top_k)
    sparse_results = retrieve_sparse(query, top_k=top_k)

    # Dùng text làm key để merge (cùng chunk có thể xuất hiện ở cả hai)
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # Dense ranks
    for rank, chunk in enumerate(dense_results):
        key = chunk["text"][:120]  # dùng 120 ký tự đầu làm key đủ unique
        chunk_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight * (1.0 / (60 + rank))

    # Sparse ranks
    for rank, chunk in enumerate(sparse_results):
        key = chunk["text"][:120]
        chunk_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight * (1.0 / (60 + rank))

    # Sort theo RRF score giảm dần
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

    results = []
    for key in sorted_keys:
        chunk = dict(chunk_map[key])
        chunk["score"] = rrf_scores[key]
        results.append(chunk)

    return results


# =============================================================================
# RERANK — Cross-Encoder
# =============================================================================

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = TOP_K_SELECT,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates bằng cross-encoder ms-marco-MiniLM-L-6-v2.

    Funnel: Search rộng (top-10) → Rerank (cross-encoder) → Select (top-3)

    Cross-encoder chấm từng cặp (query, chunk) → score chính xác hơn bi-encoder
    nhưng chậm hơn → chỉ dùng sau search rộng.
    """
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers chưa được cài.\n"
                "Chạy: pip install sentence-transformers"
            )
        print("  [Rerank] Đang load CrossEncoder model (lần đầu có thể chậm)...")
        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("  [Rerank] Model đã sẵn sàng.")

    pairs = [[query, chunk["text"]] for chunk in candidates]
    scores = _cross_encoder_model.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []
    for chunk, score in ranked[:top_k]:
        chunk = dict(chunk)
        chunk["rerank_score"] = float(score)
        results.append(chunk)

    return results


# =============================================================================
# QUERY TRANSFORMATION
# =============================================================================

def transform_query(query: str, strategy: str = "expansion") -> List[str]:
    """
    Biến đổi query để tăng recall.

    Strategies:
      - "expansion"     : Sinh 2-3 cách diễn đạt khác (alias, đồng nghĩa)
      - "decomposition" : Tách query phức tạp thành sub-queries
      - "hyde"          : Sinh câu trả lời giả (Hypothetical Document Embedding)

    Returns:
        List queries (gồm cả query gốc ở đầu)
    """
    if strategy == "expansion":
        prompt = (
            f"Given the query: '{query}'\n"
            "Generate 2-3 alternative phrasings or related terms in the same language "
            "(Vietnamese if query is Vietnamese, English otherwise).\n"
            "Include synonyms, abbreviations, or related policy terms.\n"
            'Output ONLY a JSON array of strings, no explanation. Example: ["...", "..."]'
        )
    elif strategy == "decomposition":
        prompt = (
            f"Break down this query into 2-3 simpler sub-queries: '{query}'\n"
            "Each sub-query should be answerable independently.\n"
            'Output ONLY a JSON array of strings, no explanation. Example: ["...", "..."]'
        )
    elif strategy == "hyde":
        prompt = (
            f"Write a short, factual answer (2-3 sentences) to this question "
            f"as if it appeared in an internal company policy document: '{query}'\n"
            "Output the hypothetical passage only, no explanation."
        )
    else:
        raise ValueError(f"strategy không hợp lệ: {strategy}. Chọn expansion | decomposition | hyde")

    raw = call_llm(prompt)

    if strategy == "hyde":
        # HyDE: dùng đoạn văn giả làm query thay thế
        return [query, raw.strip()]

    # expansion / decomposition: parse JSON array
    import json, re
    try:
        # Tìm array JSON trong output
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            alternatives = json.loads(match.group())
            return [query] + [q for q in alternatives if isinstance(q, str)]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: split theo dòng
    lines = [line.strip().strip('"').strip("'").strip("-").strip()
             for line in raw.splitlines() if line.strip()]
    return [query] + [l for l in lines if l]


# =============================================================================
# GENERATION
# =============================================================================

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Đóng gói chunks thành context block có số thứ tự [1], [2], ...
    để model dễ trích dẫn trong câu trả lời.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        eff_date = meta.get("effective_date", "")
        score = chunk.get("rerank_score", chunk.get("score", 0))
        text = chunk.get("text", "")

        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        if eff_date and eff_date != "unknown":
            header += f" | effective={eff_date}"
        if score > 0:
            header += f" | score={score:.3f}"

        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def build_grounded_prompt(query: str, context_block: str) -> str:
    """
    Grounded prompt theo 4 quy tắc:
    1. Evidence-only  — chỉ trả lời từ context
    2. Abstain        — thiếu context thì nói không đủ dữ liệu
    3. Citation       — gắn [số] khi có thể
    4. Short & clear  — ngắn, rõ, nhất quán
    """
    prompt = f"""You are a helpful internal assistant. Answer ONLY from the retrieved context below.

Rules:
1. Use ONLY the provided context. Do not use prior knowledge.
2. If the context does not contain enough information, respond with:
   "Không đủ dữ liệu trong tài liệu để trả lời câu hỏi này." (or the English equivalent if query is in English)
3. Cite sources using their bracket number like [1] or [2] when referencing specific information.
4. Keep your answer concise and factual (2-5 sentences max).
5. Respond in the same language as the question.

Question: {query}

Context:
{context_block}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Gọi LLM để sinh câu trả lời.

    Chọn backend qua biến môi trường:
      - LLM_BACKEND=openai  (mặc định) → cần OPENAI_API_KEY
      - LLM_BACKEND=gemini              → cần GOOGLE_API_KEY
    """
    backend = os.getenv("LLM_BACKEND", "openai").lower()

    if backend == "openai":
        return _call_openai(prompt)
    elif backend == "gemini":
        return _call_gemini(prompt)
    else:
        raise ValueError(
            f"LLM_BACKEND không hợp lệ: '{backend}'. Chọn 'openai' hoặc 'gemini'."
        )


def _call_openai(prompt: str) -> str:
    """Option A: OpenAI Chat Completions"""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY không tìm thấy trong .env.\n"
                "Thêm OPENAI_API_KEY=sk-... vào file .env\n"
                "Hoặc đặt LLM_BACKEND=gemini và cung cấp GOOGLE_API_KEY."
            )
        _openai_client = OpenAI (base_url="https://models.inference.ai.azure.com/",api_key=api_key)

    response = _openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,      # output ổn định, dễ đánh giá
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str) -> str:
    """Option B: Google Gemini"""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai chưa được cài.\n"
            "Chạy: pip install google-generativeai"
        )
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY không tìm thấy trong .env.\n"
            "Thêm GOOGLE_API_KEY=... vào file .env"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def rag_answer(
    query: str,
    retrieval_mode: str = "dense",
    top_k_search: int = TOP_K_SEARCH,
    top_k_select: int = TOP_K_SELECT,
    use_rerank: bool = False,
    query_transform: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline RAG hoàn chỉnh: query → (transform) → retrieve → (rerank) → generate.

    Args:
        query           : Câu hỏi
        retrieval_mode  : "dense" | "sparse" | "hybrid"
        top_k_search    : Số chunk lấy từ vector store (search rộng)
        top_k_select    : Số chunk đưa vào prompt (sau rerank/select)
        use_rerank      : Bật cross-encoder rerank
        query_transform : None | "expansion" | "decomposition" | "hyde"
        verbose         : In debug info

    Returns:
        Dict với "answer", "sources", "chunks_used", "query", "config"
    """
    config = {
        "retrieval_mode": retrieval_mode,
        "top_k_search": top_k_search,
        "top_k_select": top_k_select,
        "use_rerank": use_rerank,
        "query_transform": query_transform,
        "llm_model": LLM_MODEL,
    }

    # --- Bước 1: Query Transformation (optional) ---
    queries = [query]
    if query_transform:
        queries = transform_query(query, strategy=query_transform)
        if verbose:
            print(f"[RAG] Transformed queries: {queries}")

    # --- Bước 2: Retrieve (merge nếu có nhiều queries) ---
    if retrieval_mode == "dense":
        retrieve_fn = retrieve_dense
    elif retrieval_mode == "sparse":
        retrieve_fn = retrieve_sparse
    elif retrieval_mode == "hybrid":
        retrieve_fn = retrieve_hybrid
    else:
        raise ValueError(f"retrieval_mode không hợp lệ: '{retrieval_mode}'")

    # Nếu có nhiều queries (từ transform), merge kết quả và dedup
    all_candidates: Dict[str, Dict[str, Any]] = {}
    for q in queries:
        for chunk in retrieve_fn(q, top_k=top_k_search):
            key = chunk["text"][:120]
            if key not in all_candidates or chunk["score"] > all_candidates[key]["score"]:
                all_candidates[key] = chunk

    # Sort theo score, lấy top_k_search tốt nhất
    candidates = sorted(all_candidates.values(), key=lambda c: c["score"], reverse=True)
    candidates = candidates[:top_k_search]

    if verbose:
        print(f"\n[RAG] Query: {query}")
        print(f"[RAG] Retrieved {len(candidates)} candidates (mode={retrieval_mode})")
        for i, c in enumerate(candidates[:5]):
            print(f"  [{i+1}] score={c.get('score', 0):.3f} | "
                  f"{c['metadata'].get('source', '?')} | "
                  f"{c['metadata'].get('section', '')[:40]}")

    # --- Bước 3: Rerank (optional) ---
    if use_rerank:
        candidates = rerank(query, candidates, top_k=top_k_select)
        if verbose:
            print(f"[RAG] After rerank: {len(candidates)} chunks")
    else:
        candidates = candidates[:top_k_select]

    # --- Bước 4: Build context và prompt ---
    context_block = build_context_block(candidates)
    prompt = build_grounded_prompt(query, context_block)

    if verbose:
        print(f"\n[RAG] Context block preview:\n{context_block[:400]}...\n")

    # --- Bước 5: Generate ---
    answer = call_llm(prompt)

    # --- Bước 6: Extract sources ---
    sources = list({
        c["metadata"].get("source", "unknown")
        for c in candidates
    })

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "chunks_used": candidates,
        "config": config,
    }


# =============================================================================
# SPRINT 3: SO SÁNH BASELINE VS VARIANT
# =============================================================================

def compare_retrieval_strategies(
    query: str,
    strategies: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    So sánh các retrieval strategies với cùng một query.

    Mỗi strategy là một dict kwargs truyền vào rag_answer().
    A/B Rule: chỉ đổi MỘT biến mỗi lần để kết quả có thể so sánh được.

    Ví dụ:
        compare_retrieval_strategies(
            "Approval Matrix để cấp quyền là tài liệu nào?",
            strategies=[
                {"retrieval_mode": "dense",  "label": "Baseline (Dense)"},
                {"retrieval_mode": "hybrid", "label": "Variant A (Hybrid)"},
                {"retrieval_mode": "dense",  "use_rerank": True, "label": "Variant B (Dense+Rerank)"},
            ]
        )
    """
    if strategies is None:
        strategies = [
            {"retrieval_mode": "dense",  "label": "Baseline (Dense)"},
            {"retrieval_mode": "hybrid", "label": "Variant A (Hybrid RRF)"},
            {"retrieval_mode": "dense",  "use_rerank": True, "label": "Variant B (Dense+Rerank)"},
        ]

    print(f"\n{'='*65}")
    print(f"Query: {query}")
    print("="*65)

    for strategy in strategies:
        label = strategy.pop("label", str(strategy))
        print(f"\n--- {label} ---")
        try:
            result = rag_answer(query, **strategy, verbose=False)
            print(f"Answer : {result['answer']}")
            print(f"Sources: {result['sources']}")
        except NotImplementedError as e:
            print(f"Chưa implement: {e}")
        except Exception as e:
            print(f"Lỗi: {type(e).__name__}: {e}")
        # restore label cho lần chạy sau
        strategy["label"] = label


# =============================================================================
# MAIN — Demo và Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 2 + 3: RAG Answer Pipeline")
    print("=" * 60)

    test_queries = [
        "SLA xử lý ticket P1 là bao lâu?",
        "Khách hàng có thể yêu cầu hoàn tiền trong bao nhiêu ngày?",
        "Ai phải phê duyệt để cấp quyền Level 3?",
        "ERR-403-AUTH là lỗi gì và cách xử lý?",    # query exact-keyword → BM25 mạnh
        "Chính sách nghỉ phép năm của nhân viên mới là gì?",  # có thể không có trong docs → test abstain
    ]

    # --- Sprint 2: Baseline Dense ---
    print("\n--- Sprint 2: Baseline (Dense Retrieval) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_answer(query, retrieval_mode="dense", verbose=True)
            print(f"Answer : {result['answer']}")
            print(f"Sources: {result['sources']}")
        except Exception as e:
            print(f"Lỗi: {type(e).__name__}: {e}")

    # --- Sprint 3: So sánh strategies ---
    print("\n\n--- Sprint 3: So sánh strategies ---")
    compare_retrieval_strategies(
        "Approval Matrix để cấp quyền là tài liệu nào?",
    )
    compare_retrieval_strategies(
        "ERR-403-AUTH",
    )

    print("\n\nChecklist Sprint 2:")
    print("  ✓ retrieve_dense()     — ChromaDB cosine search")
    print("  ✓ call_llm()           — OpenAI (gpt-4o-mini) hoặc Gemini flash")
    print("  ✓ build_grounded_prompt — evidence-only, abstain, citation, short")
    print("  ✓ rag_answer()         — full pipeline")

    print("\nChecklist Sprint 3:")
    print("  ✓ retrieve_sparse()    — BM25 (rank-bm25) với lazy index build")
    print("  ✓ retrieve_hybrid()    — RRF (dense 0.6 + sparse 0.4)")
    print("  ✓ rerank()             — CrossEncoder ms-marco-MiniLM-L-6-v2")
    print("  ✓ transform_query()    — expansion / decomposition / HyDE")
    print("  ✓ compare_retrieval_strategies() — A/B test framework")