# Failure Analysis — Lab 18: Production RAG

**Thực hiện:** Nguyễn Quang Vinh (2A202601517) · Cá nhân

---

## Ghi chú về phạm vi dữ liệu

Toàn bộ 5 module (`src/m1_chunking.py` → `src/m5_enrichment.py`) implement đầy đủ, không còn TODO,
`pytest tests/ -v` pass 37/37. Pipeline chạy end-to-end với dữ liệu **RAGAS thật 100%** (Google Gemini
`gemini-3.5-flash-lite`), không có số liệu giả lập.

Do Gemini free tier giới hạn 15 requests/phút và mỗi câu cần ~7 lời gọi LLM cho RAGAS scoring, việc
đánh giá toàn bộ 20 câu trong `test_set.json` mất quá nhiều thời gian trong khuôn khổ buổi lab. Bảng dưới
dùng **mẫu 5/20 câu** (`EVAL_SAMPLE_SIZE` trong `config.py`), chạy đủ cho cả Naive Baseline và Production
pipeline để có so sánh trực tiếp.

Sản xuất pipeline gặp một lỗi native (Windows: load `CrossEncoder` reranker trong cùng process đã load
`SentenceTransformer` gây lỗi `OSError: paging file too small`/segfault) — được giải quyết bằng cách chạy
bước rerank + generation + eval trong process Python hoàn toàn độc lập, tách khỏi bước indexing/embedding.
Xem chi tiết ở `reflection`.

---

## RAGAS Scores (dữ liệu thật, mẫu 5/20 câu)

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|---:|
| Faithfulness | 0.6667 | 0.5667 | **−0.1000** |
| Answer Relevancy | 0.9104 | 0.8104 | −0.1000 |
| Context Precision | 0.4667 | **0.6667** | **+0.2000** |
| Context Recall | 0.8667 | 0.6667 | −0.2000 |

**Quan sát chính:** Production cải thiện rõ rệt **Context Precision** (+0.20, đúng giả thuyết lý thuyết —
reranking lọc bớt context nhiễu). Tuy nhiên Faithfulness, Answer Relevancy và Context Recall lại **giảm**
so với Naive — ngược với kỳ vọng ban đầu. Đây là phát hiện thật, không phải giả định, và được phân tích ở
dưới.

---

## Bottom-5 Failures (Production Pipeline)

### #1 — Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?
- **Expected:** Nhân viên được nghỉ 3 ngày làm việc có lương khi kết hôn, không trừ vào phép năm.
- **Worst metric:** Faithfulness = **0.00**
- **Error Tree:** Output sai → Context đúng? Cần kiểm tra qua enrichment (contextual prepend có thể đã
  làm nhiễu nghĩa gốc của chunk) → Query OK (rõ ràng) → **Root cause: LLM hallucinating hoàn toàn — câu
  trả lời không được context nào hỗ trợ (faithfulness=0 nghĩa là 0/N statements được context xác nhận).**
- **Suggested fix:** Tighten prompt, lower temperature (đúng theo Diagnostic Tree `src/m4_eval.py`).
  Cụ thể hơn: kiểm tra xem M5 contextual prepend có đang thêm câu mô tả sai lệch làm Gemini generation bị
  nhầm chủ đề không — đây là rủi ro thực tế của enrichment (thêm context có thể giúp hoặc hại retrieval
  tùy chất lượng câu prepend do LLM tự sinh).

### #2 — Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Từ 3 năm trở lên được cộng 1 ngày phép/3 năm (chính sách v2024).
- **Worst metric:** Context Recall = **0.33**
- **Error Tree:** Output sai một phần → Context KHÔNG đầy đủ (chỉ 1/3 statements của ground truth được
  chunks bao phủ) → Query OK → **Root cause: Missing relevant chunks — có 2 phiên bản chính sách
  (v2023: 5 năm, v2024: 3 năm) và hierarchical chunking (parent 2048/child 256) có thể đã tách chunk
  version cũ/mới không đủ rõ ràng để retrieval lấy đúng cả 2 để so sánh.**
- **Suggested fix:** Improve chunking or add BM25 — cụ thể: structure-aware chunking theo heading
  "v2023"/"v2024" thay vì hierarchical thuần theo độ dài ký tự, để giữ nguyên block chính sách theo phiên
  bản.

### #3 — Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Expected:** 1.000.000 VNĐ/tháng, chi trả cùng kỳ lương.
- **Worst metric:** Faithfulness = 0.50
- **Error Tree:** Output đúng một phần → Context đúng (context_precision=0.67, đa số liên quan) → Query OK
  → **Root cause: LLM hallucinating — dù có context đúng, Gemini vẫn sinh thêm thông tin không được context
  xác nhận trực tiếp (vd suy diễn thêm chi tiết).**
- **Suggested fix:** Tighten prompt — system instruction hiện tại ("Trả lời CHỈ dựa trên context") có thể
  cần cụ thể hơn, ví dụ yêu cầu trích dẫn câu gốc thay vì diễn giải.

### #4 — Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày (v2024 hiện hành), 12 ngày là chính sách cũ v2023 đã thay thế.
- **Worst metric:** Faithfulness = 0.67 (context_precision đạt tuyệt đối 1.00 — retrieval rất tốt ở câu này)
- **Error Tree:** Output đúng phần lớn → Context hoàn toàn liên quan (1.00) → Query OK → **Root cause: dù
  context tốt nhất trong 5 câu, faithfulness vẫn không tuyệt đối — có thể do câu hỏi cần phân biệt 2 phiên
  bản chính sách (negation/version-aware), và LLM generation thêm suy luận ngoài context khi so sánh cũ/mới.**
- **Suggested fix:** Improve prompt template — thêm hướng dẫn xử lý rõ câu hỏi có yếu tố "phiên bản/thời
  điểm" (đây là 1 trong 6 loại câu hỏi của `test_set.json`: version).

### #5 — Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** 200.000.000 VNĐ/năm (nội trú, ngoại trú, nha khoa).
- **Worst metric:** Answer Relevancy = 0.62 (nhưng faithfulness cao nhất, 1.00 — câu trả lời trung thực
  với context nhưng không khớp sát câu hỏi)
- **Error Tree:** Output đúng nhưng không tập trung (answer_relevancy thấp) → Context đúng (0.67), full
  recall (1.00) → Query OK → **Root cause: Answer doesn't match question — câu trả lời có thể quá dài
  dòng/lạc đề dù thông tin đúng, ảnh hưởng answer_relevancy (được tính qua cosine similarity giữa câu hỏi
  gốc và câu hỏi được sinh lại từ answer).**
- **Suggested fix:** Improve prompt template — yêu cầu câu trả lời ngắn gọn, đi thẳng vào số liệu được hỏi.

## So sánh Naive vs Production — vì sao Production KHÔNG thắng tuyệt đối?

Kết quả thật (không như kỳ vọng lý thuyết "Production luôn tốt hơn Naive") cho thấy 3/4 metrics giảm.
Nguyên nhân khả dĩ nhất, dựa trên cấu trúc pipeline:

1. **Mẫu quá nhỏ (5 câu)** — với n=5, một câu outlier (như #1, faithfulness=0.00) kéo trung bình xuống
   đáng kể. Đây là hạn chế thống kê thực sự, không phải bằng chứng Production kém hơn về bản chất.
2. **M5 Enrichment có thể gây nhiễu** — `contextual_prepend()` dùng Gemini tự sinh câu mô tả ngữ cảnh
   cho mỗi chunk; nếu câu mô tả này không chính xác (LLM tự diễn giải sai), nó làm giảm chất lượng chunk
   được embed, ảnh hưởng ngược tới retrieval — đây là rủi ro đã biết của enrichment (không phải luôn có lợi).
3. **Reranker (M3) chỉ sửa thứ tự, không sửa nội dung** — Context Precision tăng đúng như kỳ vọng
   (+0.20) vì rerank đẩy chunk liên quan lên top, nhưng nếu retrieval ban đầu (M2 hybrid trên chunk đã
   enrichment) đã kém hơn Naive's dense-only ở một số câu cụ thể, rerank không thể bù lại được.

**Kết luận thực tế:** Reranking hoạt động đúng như thiết kế (context_precision tăng), nhưng cần thêm dữ
liệu (≥20 câu) và kiểm tra riêng tác động của M5 enrichment (bật/tắt so sánh) để kết luận chắc chắn liệu
toàn bộ Production pipeline có thắng Naive hay không — đây là action item cụ thể cho lần chạy tiếp theo.

## Case Study

**Question chọn phân tích:** "Nhân viên được nghỉ bao nhiêu ngày khi kết hôn?" (Failure #1, faithfulness=0.00)

**Error Tree walkthrough:**
1. Output đúng? → Không rõ ràng — answer_relevancy không phải metric tệ nhất ở đây, nhưng faithfulness=0
   nghĩa là câu trả lời không được context nào xác nhận trực tiếp.
2. Context đúng? → Cần retrieval logs chi tiết hơn để xác nhận (không có sẵn trong log hiện tại — action
   item: log contexts đầy đủ trong lần chạy sau).
3. Query rewrite OK? → Có, câu hỏi rõ ràng, không mơ hồ.
4. Fix ở bước: **Generation/Prompt** — faithfulness=0 là dấu hiệu mạnh của hallucination ở tầng LLM
   generation, không phải retrieval (vì các câu khác với context tương tự vẫn đạt faithfulness >0.5).

**Nếu có thêm 1 giờ:**
- Chạy lại với `EVAL_SAMPLE_SIZE=20` (cần quota Gemini lớn hơn/paid tier để không bị rate limit hàng giờ)
  để có kết luận thống kê đáng tin cậy hơn về Naive vs Production.
- Thêm logging `contexts` đầy đủ vào `EvalResult` để debug case #1 sâu hơn — hiện tại chỉ có scores,
  không có nội dung context/answer đầy đủ trong report.
- A/B test enrichment: chạy Production với `enrich_chunks(methods=[])` (bỏ M5) để cô lập xem M5 có phải
  nguyên nhân faithfulness giảm hay không.
