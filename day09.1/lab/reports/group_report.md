# Báo Cáo Nhóm — Lab Day 09: Multi-Agent Orchestration

**Tên nhóm:** X666
**Thành viên:**
Phan Hoài Linh| Supervisor Owner, Worker Owner, MCP Owner, Trace & Docs Owner | 

**Ngày nộp:** 15/04/2026  
**Repo:** https://github.com/phanlinhfbi/day09.1  
**Độ dài khuyến nghị:** 600–1000 từ

---

> **Hướng dẫn nộp group report:**
> 
> - File này nộp tại: `reports/group_report.md`
> - Deadline: Được phép commit **sau 18:00** (xem SCORING.md)
> - Tập trung vào **quyết định kỹ thuật cấp nhóm** — không trùng lặp với individual reports
> - Phải có **bằng chứng từ code/trace** — không mô tả chung chung
> - Mỗi mục phải có ít nhất 1 ví dụ cụ thể từ code hoặc trace thực tế của nhóm

---

## 1. Kiến trúc nhóm đã xây dựng (150–200 từ)

> Mô tả ngắn gọn hệ thống nhóm: bao nhiêu workers, routing logic hoạt động thế nào,
> MCP tools nào được tích hợp. Dùng kết quả từ `docs/system_architecture.md`.

**Hệ thống tổng quan:**

Hệ thống được thiết kế theo pattern **Supervisor-Worker**, gồm 1 node điều phối trung tâm (Supervisor) và 3 workers xử lý theo nhiệm vụ cụ thể: Retrieval Worker (lo trích xuất ChromaDB), Policy Tool Worker (lo validate các quy tắc, permission qua MCP tools), và Synthesis Worker (Tổng hợp câu trả lời và tự chấm điểm độ tin cậy).

**Routing logic cốt lõi:**
> Mô tả logic supervisor dùng để quyết định route (keyword matching, LLM classifier, rule-based, v.v.)

Supervisor định tuyến dựa trên **Rule-based & Regex Keyword Matching**. Việc này giúp định tuyến cực nhanh và có tính dự đoán cao (predictable routing).
- Các từ khóa như `"hoàn tiền", "kỹ thuật số", "cấp quyền"` sẽ điều hướng tự động sang `policy_tool_worker`.
- Các từ khóa thuộc về thủ tục như `"sla", "ticket", "tài liệu"` điều hướng sang `retrieval_worker`.
- Nhận dạng mã lỗi Regex `err[-_]\d+` hoặc báo động khẩn cấp `"khẩn cấp", "emergency"` sẽ văng cờ rủi ro và điều hướng qua `human_review` node.

**MCP tools đã tích hợp:**
> Liệt kê tools đã implement và 1 ví dụ trace có gọi MCP tool.

- `search_kb`: Kết nối thẳng vào function Retrieval để semantic search các đoạn tài liệu tương đồng.
- `get_ticket_info`: Mock database cho hệ thống Jira lấy status của ticket.
- `check_access_permission`: Validation điều kiện cho các Level Permissions, cho phép emergency overwrite.
- `create_ticket`: Tạo ra mock payload cho service request. (VD: `[policy_tool_worker] called MCP get_ticket_info`)

---

## 2. Quyết định kỹ thuật quan trọng nhất (200–250 từ)

> Chọn **1 quyết định thiết kế** mà nhóm thảo luận và đánh đổi nhiều nhất.
> Phải có: (a) vấn đề gặp phải, (b) các phương án cân nhắc, (c) lý do chọn phương án đã chọn.

**Quyết định:** Tách biệt tác vụ Policy Checker thành một Worker Độc lập thay vì gộp chung Policy Constraint Validation vào Prompt của Synthesis Writer hoặc Supervisor.

**Bối cảnh vấn đề:**

Cấu trúc nghiệp vụ (Policy) liên quan rất nhiều đến conflict (các quy định cắn đuôi nhau như "hoàn tiền trong 30 ngày cho mọi đơn" nhưng ngoại lệ "đơn thiết bị số không hoàn"). Nếu nhồi nhét quy định vào Prompt cho Synthesis Node tự xử lý, LLM thường trực tiếp ảo giác (hallucinate) làm ngơ các trường hợp biên hoặc sinh ra lý do không có căn cứ. 

**Các phương án đã cân nhắc:**

| Phương án | Ưu điểm | Nhược điểm |
|-----------|---------|-----------|
| Gộp vào 1 Single Agent xử lý (Synthesis) | Triển khai nhanh, State chỉ cần vài parameter text nhỏ. | Token dài, gây Overload prompt và bỏ sót Rule nếu Rule đó chiếm trọng số quá nhỏ lúc Retrieval. |
| Xây dựng Node Policy_Tool riêng | Ép chuẩn (contract) và logic rule rõ ràng. Dễ dàng dùng Tool Call ra MCP Server. | Mất công define State phúc tạp, tốn thời gian design và Overhead latency. |

**Phương án đã chọn và lý do:**

Nhóm chọn xây dựng `policy_tool_worker` riêng biệt kết hợp với Workflow Fallback. Lý do lớn nhất là đảm bảo **Tính nhất quán, có thể giải thích được** (Visibility). Logic check policy phải được ưu tiên, nếu chưa có Data ở Policy Check, Worker này vẫn có thể Invoke gián tiếp công cụ `search_kb` lấy về Context.

**Bằng chứng từ trace/code:**
> Dẫn chứng cụ thể (VD: route_reason trong trace, đoạn code, v.v.)

```python
# Trace output (graph.py execution logic) lý do redirect qua Policy
"supervisor_route": "policy_tool_worker",
"route_reason": "Phát hiện yêu cầu về policy/cấp quyền/tác vụ đặc biệt → policy_tool_worker"

# AgentState Schema trace history:
"history": [
  "[supervisor] received task: Cần cấp quyền Level 3 để khắc phục P1 khẩn cấp. Quy trình là gì?",
  "[supervisor] route=policy_tool_worker reason=Phát hiện yêu cầu về policy/cấp quyền...",
  "[policy_tool_worker] called MCP search_kb",
  "[policy_tool_worker] policy_applies=True, exceptions=0"
]
```

---

## 3. Kết quả grading questions (150–200 từ)

> Sau khi chạy pipeline với grading_questions.json (public lúc 17:00):
> - Nhóm đạt bao nhiêu điểm raw?
> - Câu nào pipeline xử lý tốt nhất?
> - Câu nào pipeline fail hoặc gặp khó khăn?

**Tổng điểm raw ước tính:** 92 / 96

**Câu pipeline xử lý tốt nhất:**
- ID: `gq04` (Nhiều rules xung đột) — Nhóm giải quyết rất tốt vì worker `policy_tool` tự list ra được chính xác Exceptions mà Retrieval Node thông thường hay bị lạc mất context. Điểm tự tin do Synthesis chấm luôn chạm ngưỡng > 0.90.

**Câu pipeline fail hoặc partial:**
- ID: `gq15` — Fail ở khâu Routing: Câu hỏi trộn lẫn ngôn ngữ mô tả khẩn cấp và quy định chính sách.
  Root cause: Router của nhóm hiện dùng string parsing (keywords tracking), khi keywords của hai bên quá chéo nhau, Rule matching nhảy qua default branch không phù hợp, làm rẽ nhầm sang `retrieval` thuần túy thay vì `HR`.

**Câu gq07 (abstain):** Nhóm xử lý thế nào?

Nếu Context lấy về bị Drop (empty chunks) hoặc các Policy trả về không hợp lệ, worker Synthesis được ghim system prompt cực đoan: *"Nếu context không đủ để trả lời → nói rõ Không đủ thông tin trong tài liệu nội bộ"*. Hệ quả là Output sẽ luôn an toàn, và Confidence Point sẽ bị kéo thẳng xuống `0.3` (heuristic score).

**Câu gq09 (multi-hop khó nhất):** Trace ghi được 2 workers không? Kết quả thế nào?

Trace đã capture thành công. State chuyển qua 2 vòng:
Supervisor -> Policy Worker (thấy thiếu Context/Tool) -> Tự trigger `get_ticket_info` -> Đọc kết quả -> Synthesis lưu context và Generate trả lời gốc.

---

## 4. So sánh Day 08 vs Day 09 — Điều nhóm quan sát được (150–200 từ)

> Dựa vào `docs/single_vs_multi_comparison.md` — trích kết quả thực tế.

**Metric thay đổi rõ nhất (có số liệu):**

- **Latency tăng**: Single-agent mất TB 800ms để response do gọi thẳng Open AI/Gemini, nhưng Supervisor tốn 1.8s - 2.5s vì có nhiều overhead gọi qua lại và check MCP. 
- **Độ tin cậy tăng (Hallucination giảm)**: Số lần sinh ảo giác giảm hẳn (Từ khoảng 12% theo Single-agent xuống <3% cho Multi-agent) khi ép chạy bằng Worker Synthesis có Temperature 0.1 và System prompt kiểm tra.

**Điều nhóm bất ngờ nhất khi chuyển từ single sang multi-agent:**

Việc viết prompt cho LLM không còn là bottleneck lớn nhất. Khó khăn lớn nhất chuyển đổi sang việc thiết kế cấu trúc `AgentState` và làm Schema Discovery cho MCP Protocol. Hệ thống bị gián đoạn rất nhiều lần lúc code chỉ vì key `retrieved_chunks` bị mismatch hoặc mất biến giữa các Graph Nodes.

**Trường hợp multi-agent KHÔNG giúp ích hoặc làm chậm hệ thống:**

Khi User chỉ hỏi Greeting/Common Sense ("Hello", "Bên mình IT có quy trình không?"). Lúc đó Pipeline bị overload logic, mất thì giờ chạy xuống Supervisor xong đi vòng lại mà đáng lẽ có thể Reject ngay từ Gateway.

---

## 5. Phân công và đánh giá nhóm (100–150 từ)

> Đánh giá trung thực về quá trình làm việc nhóm.

**Phân công thực tế:**

| Thành viên | Phần đã làm | Sprint |
|------------|-------------|--------|
| Phan Hoài Linh | Supervisor Owner & HR Node | Sprint 1 |
| Phan Hoài Linh | Retrieval Worker, Policy Tool Worker | Sprint 2 |
| Phan Hoài Linh | MCP Server Tooling & API Schema | Sprint 3 |
| Phan Hoài Linh | Writing System Architecture, Bug Trace Fix | Sprint 3 |

**Điều nhóm làm tốt:**

- Thiết kế rất vững chắc và quy củ ở các Node. File codebase hoàn thiện `MCP_Server` và có fallback cực kì xuất sắc với Local model mock/Offline (Sentence Transformer).

**Điều nhóm làm chưa tốt hoặc gặp vấn đề về phối hợp:**

- Quá trình quản lý và pass state dict đôi khi còn bị overwrite giữa các worker (Ví dụ worker B bị override `history` của Worker A nếu không dùng list append cẩn thận).

**Nếu làm lại, nhóm sẽ thay đổi gì trong cách tổ chức?**

- Implement LangGraph hoặc StateGraph Library từ đầu thay vì code thuần Python Function passing (Option A) để được thụ hưởng các checkpoint tự động của thư viện.

---

## 6. Nếu có thêm 1 ngày, nhóm sẽ làm gì? (50–100 từ)

> 1–2 cải tiến cụ thể với lý do có bằng chứng từ trace/scorecard.

1. **Thay Router Rule-base bằng LLM Classifier:** Nhìn vào `gq15` pipeline fail vì string parsing if/else đơn giản. Có thêm thời gian, nhóm sẽ build một con LLM nhỏ (Claude Haiku/GPT-4o Mini) chỉ có nhiệm vụ Output Enum định tuyến chuyên biệt.
2. **Implement Checkpoint/Interrupt:** Viết chức năng Breakpoint cho "human_review" node để chặn Graph chờ phê duyệt thực sự thay vì Auto-Approve qua CLI, nhằm giúp HR flow được bảo vệ an toàn.

---

*File này lưu tại: `reports/group_report.md`*  
*Commit sau 18:00 được phép theo SCORING.md*
