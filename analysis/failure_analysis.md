# Failure Analysis — Lab 18: Production RAG

**Thực hiện:** Nguyễn Quang Vinh (2A202601517) · Cá nhân

---

## Ghi chú quan trọng về phạm vi dữ liệu

Toàn bộ 5 module (`src/m1_chunking.py` → `src/m5_enrichment.py`) đã implement đầy đủ, không còn TODO, và
`pytest tests/ -v` pass 37/37 (bao gồm M3 khi chạy cô lập từng test — xem lý do ở `reflection`). Pipeline
`python src/pipeline.py` / `python main.py` chạy được end-to-end về mặt logic.

Tuy nhiên, **RAGAS evaluation trong lần chạy cuối chỉ hoàn tất được 2/5 câu của Naive Baseline** (không kịp
chạy tới Production pipeline) trước khi phải dừng lại, vì lý do kỹ thuật thực tế:

- Ban đầu dùng `OPENAI_API_KEY` nhưng tài khoản hết credit (`insufficient_quota`) → chuyển sang Google
  Gemini API (free tier).
- `ragas.evaluate()` (thư viện ragas 0.1.22) bị deadlock vĩnh viễn trên máy Windows này (native
  asyncio/gRPC AsyncClient conflict) → phải viết lại `evaluate_ragas()` trong `src/m4_eval.py` để gọi
  LLM đồng bộ trực tiếp (`llm.invoke()`), giữ đúng phương pháp luận RAGAS (LLM-as-judge trên statements)
  nhưng thực thi tuần tự.
- Gemini free tier giới hạn **15 requests/phút**. Mỗi câu hỏi cần ~7 lời gọi LLM (statements + verdicts ×
  4 metrics) → dù đã thêm pacing (5s/call) và giảm max statements xuống 3, tốc độ thực tế chỉ đạt
  ~1 câu / 30–60 phút do bị throttle liên tục (429 `ResourceExhausted`) từ các lần test trước đó trong
  cùng session đã tiêu tốn phần lớn quota.

Do giới hạn thời gian, phần dưới đây dùng **2 câu Naive Baseline có scores RAGAS thật** (Gemini
`gemini-3.5-flash-lite`, không phải dữ liệu giả lập) làm minh hoạ phương pháp phân tích, thay vì bộ đầy đủ
20 câu × Naive + Production như đề bài kỳ vọng.

---

## RAGAS Scores (dữ liệu thật, mẫu 2/20 câu — chỉ Naive Baseline)

| Metric | Câu 1 | Câu 2 | Trung bình |
|--------|------:|------:|-----------:|
| Faithfulness | 0.50 | 0.67 | 0.585 |
| Answer Relevancy | 0.59 | 0.99 | 0.79 |
| Context Precision | 0.33 | 0.33 | 0.33 |
| Context Recall | 1.00 | *(chưa chạy xong)* | — |

Cột **Production** không có vì pipeline production (M5 enrichment → M2 hybrid → M3 rerank → RAGAS) chưa
kịp chạy tới bước eval trong thời gian cho phép — xem ghi chú ở trên.

## Chi tiết 2 câu đã đánh giá

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected (ground truth):** Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm.
- **Retrieved context (basic/dense-only, top-3):** 1 đoạn liên quan tìm được (context_precision=0.33 → 2/3 context không liên quan).
- **Scores:** faithfulness=0.50, answer_relevancy=0.59, context_precision=0.33, context_recall=1.00
- **Worst metric:** Context Precision (0.33)
- **Error Tree:** Output sai một phần → Context đúng nhưng lẫn nhiều đoạn không liên quan (basic chunking cắt theo
  paragraph, không phân biệt chủ đề) → Query OK (đúng ý định) → **Root cause: retrieval trả về top-3 nhưng basic
  dense-only không rerank nên 2/3 context nhiễu.**
- **Suggested fix:** Thêm reranking (M3) để lọc context nhiễu trước khi đưa vào LLM — đây chính xác là lý do
  Production pipeline có M3 CrossEncoder rerank mà Naive Baseline không có.

### #2
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected (ground truth):** Hạn mức bảo hiểm sức khỏe PVI cho nhân viên là 200.000.000 VNĐ/năm, bao gồm nội trú, ngoại trú và nha khoa.
- **Scores:** faithfulness=0.67, answer_relevancy=0.99, context_precision=0.33
- **Worst metric:** Context Precision (0.33) — cùng pattern với câu #1.
- **Error Tree:** Answer relevancy rất cao (0.99, LLM hiểu đúng câu hỏi) nhưng context_precision thấp →
  **Root cause: cùng nguyên nhân — basic chunking + dense-only search không đủ để loại nhiễu context.**
- **Suggested fix:** Giống câu #1 — reranking + hybrid search (BM25 bắt số liệu chính xác "200.000.000" tốt
  hơn dense-only vì đây là exact-match numeric, đúng use case BM25 được thiết kế để giải quyết).

## Pattern quan sát được (dù mẫu nhỏ)

Cả 2/2 câu đều có **Context Precision thấp nhất (0.33)** trong 4 metrics — đây là tín hiệu nhất quán cho
thấy Naive Baseline (paragraph chunking + dense-only, không rerank) đưa quá nhiều context không liên quan
vào prompt. Đây đúng là vấn đề mà Production pipeline (M2 Hybrid Search + M3 Reranking) được thiết kế để
giải quyết — dù không có số liệu Production thật để so sánh trực tiếp trong lần chạy này, pattern retrieval
noise này khớp với lý do lý thuyết đã học (Diagnostic Tree: context_precision thấp → "Too many irrelevant
chunks" → suggested fix "Add reranking or metadata filter", xem `src/m4_eval.py::failure_analysis()`).

## Case Study

**Question chọn phân tích:** "Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?"

**Error Tree walkthrough:**
1. Output đúng? → Answer relevancy 0.99, câu trả lời đúng ý câu hỏi.
2. Context đúng? → Chỉ 1/3 context liên quan (context_precision=0.33) — có context nhiễu từ tài liệu khác.
3. Query rewrite OK? → Không cần rewrite, câu hỏi đã rõ ràng, đây không phải vấn đề query.
4. Fix ở bước: **Retrieval/Reranking** — cần M3 CrossEncoder rerank để đẩy context liên quan lên top vị trí,
   hoặc M2 Hybrid (BM25) để bắt chính xác số liệu "200.000.000 VNĐ" thay vì chỉ dựa vào dense embedding.

**Nếu có thêm thời gian (không bị giới hạn API):**
- Chạy lại `main.py` với `OPENAI_API_KEY` có credit hoặc Gemini paid tier để có đủ 20 câu × Naive + Production,
  điền bảng so sánh đầy đủ theo đúng format đề bài.
- Với dữ liệu đầy đủ, kỳ vọng Production pipeline cải thiện context_precision đáng kể nhờ M3 reranking —
  đây là giả thuyết dựa trên lecture (cross-encoder rerank cải thiện precision@k) nhưng cần số liệu thật để xác nhận.
