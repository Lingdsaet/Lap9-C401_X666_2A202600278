"""
index.py — Sprint 1: Build RAG Index
====================================
Mục tiêu Sprint 1 (60 phút):
  - Đọc và preprocess tài liệu từ data/docs/
  - Chunk tài liệu theo cấu trúc tự nhiên (heading/section)
  - Gắn metadata: source, section, department, effective_date, access
  - Embed và lưu vào vector store (ChromaDB)

Definition of Done Sprint 1:
  ✓ Script chạy được và index đủ docs
  ✓ Có ít nhất 3 metadata fields hữu ích cho retrieval
  ✓ Có thể kiểm tra chunk bằng list_chunks()
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CẤU HÌNH
# =============================================================================

DOCS_DIR = Path(__file__).parent / "data" / "docs"
CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"

# Chunk size và overlap theo gợi ý từ slide: chunk 300-500 tokens, overlap 50-80 tokens
CHUNK_SIZE = 400       # tokens (ước lượng bằng số ký tự / 4)
CHUNK_OVERLAP = 80     # tokens overlap giữa các chunk


# =============================================================================
# STEP 1: PREPROCESS
# Làm sạch text trước khi chunk và embed
# =============================================================================

def preprocess_document(raw_text: str, filepath: str) -> Dict[str, Any]:
    """
    Preprocess một tài liệu: extract metadata từ header và làm sạch nội dung.

    Args:
        raw_text: Toàn bộ nội dung file text
        filepath: Đường dẫn file để làm source mặc định

    Returns:
        Dict chứa:
          - "text": nội dung đã clean
          - "metadata": dict với source, department, effective_date, access
    """
    lines = raw_text.strip().split("\n")
    metadata = {
        "source": filepath,
        "section": "",
        "department": "unknown",
        "effective_date": "unknown",
        "access": "internal",
    }
    content_lines = []
    header_done = False

    for line in lines:
        if not header_done:
            # Parse metadata từ các dòng "Key: Value"
            if line.startswith("Source:"):
                metadata["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Department:"):
                metadata["department"] = line.replace("Department:", "").strip()
            elif line.startswith("Effective Date:"):
                metadata["effective_date"] = line.replace("Effective Date:", "").strip()
            elif line.startswith("Access:"):
                metadata["access"] = line.replace("Access:", "").strip()
            elif line.startswith("==="):
                # Gặp section heading đầu tiên → kết thúc header
                header_done = True
                content_lines.append(line)
            elif line.strip() == "" or line.isupper():
                # Dòng tên tài liệu (toàn chữ hoa) hoặc dòng trống
                continue
        else:
            content_lines.append(line)

    cleaned_text = "\n".join(content_lines)

    # Normalize text:
    # 1. Rút gọn nhiều dòng trắng liên tiếp xuống còn tối đa 2
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    # 2. Xóa khoảng trắng thừa ở đầu/cuối mỗi dòng
    cleaned_text = "\n".join(line.rstrip() for line in cleaned_text.split("\n"))
    # 3. Chuẩn hóa dấu ngoặc kép kiểu curly → thẳng
    cleaned_text = cleaned_text.replace("\u201c", '"').replace("\u201d", '"')
    cleaned_text = cleaned_text.replace("\u2018", "'").replace("\u2019", "'")
    # 4. Xóa ký tự null và các ký tự điều khiển không cần thiết
    cleaned_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned_text)

    return {
        "text": cleaned_text,
        "metadata": metadata,
    }


# =============================================================================
# STEP 2: CHUNK
# Chia tài liệu thành các đoạn nhỏ theo cấu trúc tự nhiên
# =============================================================================

def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk một tài liệu đã preprocess thành danh sách các chunk nhỏ.

    Args:
        doc: Dict với "text" và "metadata" (output của preprocess_document)

    Returns:
        List các Dict, mỗi dict là một chunk với:
          - "text": nội dung chunk
          - "metadata": metadata gốc + "section" của chunk đó

    Chiến lược:
    1. Split theo heading "=== Section ... ===" trước
    2. Nếu section quá dài (> CHUNK_SIZE * 4 ký tự), split tiếp theo paragraph
    3. Thêm overlap: lấy đoạn cuối của chunk trước vào đầu chunk tiếp theo
    4. Mỗi chunk giữ metadata đầy đủ từ tài liệu gốc
    """
    text = doc["text"]
    base_metadata = doc["metadata"].copy()
    chunks = []

    # Bước 1: Split theo heading pattern "=== ... ==="
    # re.split với capturing group giữ lại delimiter trong kết quả
    sections = re.split(r"(===.*?===)", text)

    current_section = "General"
    current_section_text = ""

    for part in sections:
        if re.match(r"===.*?===", part):
            # Lưu section trước (nếu có nội dung)
            if current_section_text.strip():
                section_chunks = _split_by_size(
                    current_section_text.strip(),
                    base_metadata=base_metadata,
                    section=current_section,
                )
                chunks.extend(section_chunks)
            # Bắt đầu section mới — làm sạch tên section
            current_section = part.strip("= ").strip()
            current_section_text = ""
        else:
            current_section_text += part

    # Lưu section cuối cùng
    if current_section_text.strip():
        section_chunks = _split_by_size(
            current_section_text.strip(),
            base_metadata=base_metadata,
            section=current_section,
        )
        chunks.extend(section_chunks)

    return chunks


def _split_by_size(
    text: str,
    base_metadata: Dict,
    section: str,
    chunk_chars: int = CHUNK_SIZE * 4,
    overlap_chars: int = CHUNK_OVERLAP * 4,
) -> List[Dict[str, Any]]:
    """
    Helper: Split text dài thành chunks với overlap.

    Thuật toán:
    1. Split text theo paragraph (\\n\\n)
    2. Ghép các paragraph liên tiếp cho đến khi tổng độ dài vượt chunk_chars
    3. Khi vượt ngưỡng → lưu chunk hiện tại, lấy overlap từ đoạn cuối
       rồi bắt đầu chunk mới
    4. Đảm bảo không bao giờ cắt giữa paragraph
    """
    if len(text) <= chunk_chars:
        return [{
            "text": text,
            "metadata": {**base_metadata, "section": section},
        }]

    # Split theo paragraph
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current_paragraphs: List[str] = []
    current_length = 0
    overlap_buffer = ""  # Đoạn overlap từ chunk trước

    for para in paragraphs:
        para_len = len(para)

        # Nếu thêm paragraph này sẽ vượt quá chunk_chars → flush chunk hiện tại
        if current_length + para_len > chunk_chars and current_paragraphs:
            chunk_text = overlap_buffer + "\n\n".join(current_paragraphs)
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {**base_metadata, "section": section},
            })

            # Tính overlap: lấy các paragraph cuối sao cho tổng <= overlap_chars
            overlap_paragraphs = []
            overlap_len = 0
            for p in reversed(current_paragraphs):
                if overlap_len + len(p) <= overlap_chars:
                    overlap_paragraphs.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            overlap_buffer = "\n\n".join(overlap_paragraphs) + "\n\n" if overlap_paragraphs else ""

            # Bắt đầu chunk mới với paragraph hiện tại
            current_paragraphs = [para]
            current_length = para_len
        else:
            # Nếu một paragraph đơn lẻ đã dài hơn chunk_chars → cắt cứng theo câu
            if para_len > chunk_chars and not current_paragraphs:
                sub_chunks = _split_long_paragraph(para, chunk_chars, overlap_chars)
                for sc in sub_chunks:
                    chunks.append({
                        "text": sc.strip(),
                        "metadata": {**base_metadata, "section": section},
                    })
                overlap_buffer = ""
            else:
                current_paragraphs.append(para)
                current_length += para_len

    # Flush chunk cuối cùng
    if current_paragraphs:
        chunk_text = overlap_buffer + "\n\n".join(current_paragraphs)
        chunks.append({
            "text": chunk_text.strip(),
            "metadata": {**base_metadata, "section": section},
        })

    return chunks


def _split_long_paragraph(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    """
    Fallback: cắt một paragraph rất dài theo ranh giới câu (dấu chấm, chấm than, chấm hỏi).
    """
    # Tách theo dấu kết thúc câu
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    overlap = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_chars and current:
            chunks.append(overlap + current)
            # Lấy overlap từ cuối chunk hiện tại
            words = current.split()
            overlap_words = []
            overlap_len = 0
            for w in reversed(words):
                if overlap_len + len(w) + 1 <= overlap_chars:
                    overlap_words.insert(0, w)
                    overlap_len += len(w) + 1
                else:
                    break
            overlap = " ".join(overlap_words) + " " if overlap_words else ""
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(overlap + current)

    return chunks


# =============================================================================
# STEP 3: EMBED + STORE
# Embed các chunk và lưu vào ChromaDB
# =============================================================================

# Cache model để không load lại mỗi lần gọi (dùng cho Option B)
_sentence_transformer_model = None

def get_embedding(text: str) -> List[float]:
    """
    Tạo embedding vector cho một đoạn text.

    Ưu tiên Option A (OpenAI) nếu có OPENAI_API_KEY trong môi trường.
    Fallback về Option B (Sentence Transformers) nếu không có API key.

    Option A — OpenAI Embeddings:
        Yêu cầu: OPENAI_API_KEY trong file .env
        Model: text-embedding-3-small (1536 chiều, nhanh và rẻ)

    Option B — Sentence Transformers (local, không cần API key):
        Model: paraphrase-multilingual-MiniLM-L12-v2
        Hỗ trợ tiếng Việt tốt, chạy offline hoàn toàn.
        Cài đặt: pip install sentence-transformers
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        # Option A: OpenAI Embeddings
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://models.inference.ai.azure.com/", api_key=api_key)
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small",
            )
            return response.data[0].embedding
        except ImportError:
            print("Cảnh báo: openai chưa được cài. Chạy: pip install openai")
            print("Đang fallback sang Sentence Transformers...")

    # Option B: Sentence Transformers (local)
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Đang tải Sentence Transformer model lần đầu (có thể mất vài phút)...")
            _sentence_transformer_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            print("Model đã sẵn sàng.")
        except ImportError:
            raise ImportError(
                "Thiếu thư viện embedding. Hãy cài một trong hai:\n"
                "  pip install openai          # Và đặt OPENAI_API_KEY trong .env\n"
                "  pip install sentence-transformers  # Chạy local, không cần API key"
            )

    return _sentence_transformer_model.encode(text, normalize_embeddings=True).tolist()


def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Pipeline hoàn chỉnh: đọc docs → preprocess → chunk → embed → store.

    Chạy lần đầu: pip install chromadb
    """
    import chromadb

    print(f"Đang build index từ: {docs_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo ChromaDB persistent client
    client = chromadb.PersistentClient(path=str(db_dir))

    # Dùng cosine similarity — phù hợp với normalized embeddings
    collection = client.get_or_create_collection(
        name="rag_lab",
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    doc_files = list(docs_dir.glob("*.txt"))

    if not doc_files:
        print(f"Không tìm thấy file .txt trong {docs_dir}")
        return

    for filepath in doc_files:
        print(f"  Processing: {filepath.name}")
        raw_text = filepath.read_text(encoding="utf-8")

        # Preprocess
        doc = preprocess_document(raw_text, str(filepath))

        # Chunk
        chunks = chunk_document(doc)
        print(f"    → {len(chunks)} chunks")

        # Embed và lưu từng chunk vào ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_{i}"
            embedding = get_embedding(chunk["text"])

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])

        # Upsert theo batch để tránh gọi từng cái một
        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

        total_chunks += len(chunks)

    print(f"\n✓ Hoàn thành! Tổng số chunks đã index: {total_chunks}")
    print(f"  Database lưu tại: {db_dir}")


# =============================================================================
# STEP 4: INSPECT / KIỂM TRA
# Dùng để debug và kiểm tra chất lượng index
# =============================================================================

def list_chunks(db_dir: Path = CHROMA_DB_DIR, n: int = 5) -> None:
    """
    In ra n chunk đầu tiên trong ChromaDB để kiểm tra chất lượng index.

    Kiểm tra:
    - Chunk có giữ đủ metadata không? (source, section, effective_date)
    - Chunk có bị cắt giữa điều khoản không?
    - Metadata effective_date có đúng không?
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(limit=n, include=["documents", "metadatas"])

        print(f"\n=== Top {n} chunks trong index ===\n")
        for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"])):
            print(f"[Chunk {i+1}]")
            print(f"  Source    : {meta.get('source', 'N/A')}")
            print(f"  Section   : {meta.get('section', 'N/A')}")
            print(f"  Department: {meta.get('department', 'N/A')}")
            print(f"  Eff. Date : {meta.get('effective_date', 'N/A')}")
            print(f"  Access    : {meta.get('access', 'N/A')}")
            print(f"  Length    : {len(doc)} ký tự")
            print(f"  Preview   : {doc[:150].strip()}...")
            print()
    except Exception as e:
        print(f"Lỗi khi đọc index: {e}")
        print("Hãy chạy build_index() trước.")


def inspect_metadata_coverage(db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Kiểm tra phân phối metadata trong toàn bộ index.

    Checklist Sprint 1:
    - Mọi chunk đều có source?
    - Có bao nhiêu chunk từ mỗi department?
    - Chunk nào thiếu effective_date?
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")
        results = collection.get(include=["metadatas"])

        total = len(results["metadatas"])
        print(f"\n=== Metadata Coverage Report ===")
        print(f"Tổng chunks: {total}\n")

        # Phân tích theo department
        departments: Dict[str, int] = {}
        missing_date = 0
        missing_source = 0
        access_levels: Dict[str, int] = {}

        for meta in results["metadatas"]:
            # Department
            dept = meta.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1

            # Effective date
            if meta.get("effective_date") in ("unknown", "", None):
                missing_date += 1

            # Source
            if not meta.get("source") or meta.get("source") in ("", None):
                missing_source += 1

            # Access level
            access = meta.get("access", "unknown")
            access_levels[access] = access_levels.get(access, 0) + 1

        print("Phân bố theo department:")
        for dept, count in sorted(departments.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {dept:30s}: {count:4d} chunks ({pct:.1f}%)")

        print("\nPhân bố theo access level:")
        for access, count in sorted(access_levels.items(), key=lambda x: -x[1]):
            print(f"  {access:20s}: {count:4d} chunks")

        print(f"\nChunks thiếu effective_date : {missing_date} / {total}")
        print(f"Chunks thiếu source         : {missing_source} / {total}")

        # Cảnh báo nếu dữ liệu có vấn đề
        if missing_date > total * 0.2:
            print("\n⚠️  Cảnh báo: Hơn 20% chunk thiếu effective_date.")
            print("   Hãy kiểm tra lại định dạng header của file .txt.")
        if missing_source > 0:
            print(f"\n⚠️  Có {missing_source} chunk thiếu source metadata!")

    except Exception as e:
        print(f"Lỗi: {e}. Hãy chạy build_index() trước.")


def search(query: str, n_results: int = 5, db_dir: Path = CHROMA_DB_DIR) -> None:
    """
    Tìm kiếm semantic trong index với một câu truy vấn.
    Hữu ích để kiểm tra chất lượng embedding và retrieval.

    Args:
        query: Câu truy vấn tự nhiên
        n_results: Số kết quả trả về
        db_dir: Thư mục ChromaDB
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db_dir))
        collection = client.get_collection("rag_lab")

        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        print(f"\n=== Kết quả tìm kiếm cho: '{query}' ===\n")
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            similarity = 1 - dist  # cosine: distance → similarity
            print(f"[Kết quả {i+1}] Similarity: {similarity:.3f}")
            print(f"  Source  : {meta.get('source', 'N/A')}")
            print(f"  Section : {meta.get('section', 'N/A')}")
            print(f"  Preview : {doc[:200].strip()}...")
            print()
    except Exception as e:
        print(f"Lỗi khi search: {e}")
        print("Hãy chạy build_index() trước.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 1: Build RAG Index")
    print("=" * 60)

    # Bước 1: Kiểm tra docs
    doc_files = list(DOCS_DIR.glob("*.txt"))
    print(f"\nTìm thấy {len(doc_files)} tài liệu:")
    for f in doc_files:
        print(f"  - {f.name}")

    if not doc_files:
        print(f"\nKhông tìm thấy file nào trong {DOCS_DIR}")
        print("Hãy thêm file .txt vào thư mục data/docs/")
        exit(1)

    # Bước 2: Test preprocess và chunking (không cần API key)
    print("\n--- Test preprocess + chunking ---")
    for filepath in doc_files[:1]:  # Test với 1 file đầu
        raw = filepath.read_text(encoding="utf-8")
        doc = preprocess_document(raw, str(filepath))
        chunks = chunk_document(doc)
        print(f"\nFile: {filepath.name}")
        print(f"  Metadata  : {doc['metadata']}")
        print(f"  Số chunks : {len(chunks)}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  [Chunk {i+1}] Section: {chunk['metadata']['section']}")
            print(f"  Độ dài   : {len(chunk['text'])} ký tự")
            print(f"  Text     : {chunk['text'][:150].strip()}...")

    # Bước 3: Build full index
    print("\n--- Build Full Index ---")
    print("Đang build index (cần embedding model — OpenAI hoặc Sentence Transformers)...")
    try:
        build_index()

        # Bước 4: Kiểm tra index
        print("\n--- Kiểm tra Index ---")
        list_chunks(n=3)
        inspect_metadata_coverage()

        # Bước 5: Test search
        print("\n--- Test Semantic Search ---")
        search("chính sách hoàn tiền", n_results=3)

    except ImportError as e:
        print(f"\n⚠️  Lỗi import: {e}")
    except Exception as e:
        print(f"\n⚠️  Lỗi: {e}")

    print("\n✓ Sprint 1 hoàn thành!")