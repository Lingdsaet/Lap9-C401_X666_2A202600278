# Báo Cáo Cá Nhân — Lab Day 09: Multi-Agent Orchestration

**Họ và tên:** Phan Hoài Linh
**Vai trò trong nhóm:** Supervisor Owner / Worker Owner / MCP Owner / Trace & Docs Owner  
**Ngày nộp:** 15/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

> **Lưu ý quan trọng:**
> - Viết ở ngôi **"tôi"**, gắn với chi tiết thật của phần bạn làm
> - Phải có **bằng chứng cụ thể**: tên file, đoạn code, kết quả trace, hoặc commit
> - Nội dung phân tích phải khác hoàn toàn với các thành viên trong nhóm
> - Deadline: Được commit **sau 18:00** (xem SCORING.md)
> - Lưu file với tên: `reports/individual/[ten_ban].md` (VD: `nguyen_van_a.md`)

---

## 1. Tôi phụ trách phần nào? (100–150 từ)

> Mô tả cụ thể module, worker, contract, hoặc phần trace bạn trực tiếp làm.
> Không chỉ nói "tôi làm Sprint X" — nói rõ file nào, function nào, quyết định nào.

**Module/file tôi chịu trách nhiệm:**
- File chính: Lập trình toàn bộ source code (`graph.py`, `mcp_server.py`, `eval_trace.py`) và toàn bộ thư mục `workers/*.py`.
- Functions tôi implement: 
  - `supervisor_node` và router builder trong `graph.py` (điều tiết State qua các layer).
  - Logic phân tích ngoại lệ (Flash sale, digital) từ `analyze_policy` của `policy_tool.py`.
  - Semantic Retrieval và module MCP Discovery Server.

**Cách công việc của tôi kết nối với phần của thành viên khác:**

Vì tôi triển khai độc lập hệ thống nên tôi chịu trách nhiệm chắp vá (integrate) toàn bộ contract chung: Input Task đi qua `Supervisor` tạo `AgentState`, đẩy vào `Worker` trích xuất Context và kết thúc ở `Synthesis Worker`.

**Bằng chứng (commit hash, file có comment tên bạn, v.v.):**

Sự hiện diện của việc tôi đảm nhiệm toàn bộ kiến trúc phản ánh qua State Dict (`make_initial_state`) trong `graph.py` với cấu trúc JSON hoàn chỉnh cho Trace logs. Bạn có thể mở trực tiếp folder dự án để thấy dấu ấn code liên hoàn trên `history` append.

---

## 2. Tôi đã ra một quyết định kỹ thuật gì? (150–200 từ)

> Chọn **1 quyết định** bạn trực tiếp đề xuất hoặc implement trong phần mình phụ trách.
> Giải thích:
> - Quyết định là gì?
> - Các lựa chọn thay thế là gì?
> - Tại sao bạn chọn cách này?
> - Bằng chứng từ code/trace cho thấy quyết định này có effect gì?

**Quyết định:** Caching Embedding Model Model ở cấp Runtime Module trong `retrieval.py` (`_embed_fn` global) thay vì gọi Load model cho mỗi lần Node Retrieval được Invoke.

**Lý do:**
Worker `retrieval` có thể được trigger nhiều lần trong một luồng: Lần 1 khi Supervisor gọi trực tiếp, Lần 2 khi Policy Checker thấy thiếu dữ liệu và dùng Tool-call MCP `search_kb` (bản chất gọi lồng function Retrieval). Việc khởi tạo lại Object model `SentenceTransformers("all-MiniLM-L6-v2")` mỗi lần gọi pipeline sẽ ngốn 2 đến 3 giây tải weights từ Disk. Tôi chọn Global cache để triệt tiêu overhead này.

**Trade-off đã chấp nhận:**
Găm object Model trong RAM vĩnh viễn lúc chạy Terminal, tốn Mem/VRAM dư thừa nhưng đổi lấy Latency của multi-hop queries tiệm cận 50ms - 200ms cho Vector Search sau lần fetch đầu tiên.

**Bằng chứng từ trace/code:**

```python
# Trong workers/retrieval.py
_embed_fn = None  # module-level cache — tránh load lại model mỗi lần gọi

def _get_embedding_fn():
    global _embed_fn
    if _embed_fn is not None:  # Bỏ qua Load object nếu đã có trong Memory
        return _embed_fn

    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed(text: str) -> list:
            return _st_model.encode([text])[0].tolist()
        _embed_fn = embed
        return _embed_fn
```

---

## 3. Tôi đã sửa một lỗi gì? (150–200 từ)

> Mô tả 1 bug thực tế bạn gặp và sửa được trong lab hôm nay.
> Phải có: mô tả lỗi, symptom, root cause, cách sửa, và bằng chứng trước/sau.

**Lỗi:** Graph State Key bị "đè" (overwrite) thay vì tiếp nối (append), khiến Trace Debug mất logs.

**Symptom (pipeline làm gì sai?):**
Biến State `history` hoặc `worker_io_logs` trong `AgentState` được xuất ra ở CLI sau khi Pipeline chạy xong chỉ hiển thị log duy nhất của Worker cuối cùng vòng lặp (Synthesis Worker). Quá trình Router đi thế nào hay rút Policy ra sao hoàn toàn "tàng hình".

**Root cause (lỗi nằm ở đâu — indexing, routing, contract, worker logic?):**
Khi gán state trong các file Workers (như `synthesis.py` hay `policy_tool.py`), do sơ suất, ban đầu worker định nghĩa lại List rỗng nếu State truyền vào không rõ ràng: `state["history"] = ["..."]` chứ không dùng lệnh nối append. Các Workers khác nhau pass con trỏ Dict State nhưng bị replace mất giá trị Array cũ của node trước.

**Cách sửa:**
Sử dụng hàm `.setdefault` của Python Dict để đảm bảo khởi tạo an toàn, theo sau là `.append()`.

**Bằng chứng trước/sau:**

```python
# TRƯỚC KHI SỬA
history = state.get("history", [])
history.clear() # <- Bug xóa dấu vết
history.append(f"[{WORKER_NAME}] error...")
state["history"] = history

# TRÚNG KẾT QUẢ ĐÃ SỬA: (Trong tất cả workers)
state.setdefault("workers_called", [])
state.setdefault("history", [])
state["workers_called"].append(WORKER_NAME)
state["history"].append(f"[{WORKER_NAME}] policy_applies=True")
```

---

## 4. Tôi tự đánh giá đóng góp của mình (100–150 từ)

> Trả lời trung thực — không phải để khen ngợi bản thân.

**Tôi làm tốt nhất ở điểm nào?**
Tôi thiết lập **Contract Interface** (Schema Discovery in MCP và State Node in Graph) vô cùng chặt chẽ. Việc quản lý IO Contract giúp các sub-workers dễ dàng phát triển và debug unit_test rời mà không dội lỗi.

**Tôi làm chưa tốt hoặc còn yếu ở điểm nào?**
Tuning Prompts! System prompt ở `Synthesis worker` tôi viết bị rập khuôn và Overfit. Đôi lúc mô hình Generate phản hồi khá cứng nhắc "Không có đủ thông tin..." kể cả khi có context liên quan một phần.

**Nhóm phụ thuộc vào tôi ở đâu?** _(Phần nào của hệ thống bị block nếu tôi chưa xong?)_
Tất cả. Là "Sole Developer" (Solo) cho dự án Lab lần này, nếu tôi kẹt ở khâu thiết kế `Router`, việc kết nối ChromaDB hay build Synthesis sẽ vô nghĩa.

**Phần tôi phụ thuộc vào thành viên khác:** _(Tôi cần gì từ ai để tiếp tục được?)_
Không có. Nhưng nếu được, tôi hy vọng có ai đó test các câu hỏi Grading (Eval) song song trong quá trình tôi code để phản hồi lỗi Routing sớm.

---

## 5. Nếu có thêm 2 giờ, tôi sẽ làm gì? (50–100 từ)

> Nêu **đúng 1 cải tiến** với lý do có bằng chứng từ trace hoặc scorecard.
> Không phải "làm tốt hơn chung chung" — phải là:
> *"Tôi sẽ thử X vì trace của câu gq___ cho thấy Y."*

Tôi sẽ tích hợp **LangGraph Interrupt API** (`interrupt_before=["human_review_node"]`) cho node Human Review.
Bằng chứng: Trace hiện tại cho thấy tuy trigger cờ rủi ro (`"hitl_triggered": True`), code lại tự động *Auto-approving in lab mode* rồi tuột thẳng xuống Retrieval, không hề có Webhook pause hay human thực sự can thiệp. Việc thêm API này sẽ khiến đồ án tiệm cận chuẩn mức độ Production.

---

*Lưu file này với tên: `reports/individual/phan_hoai_linh.md`*
