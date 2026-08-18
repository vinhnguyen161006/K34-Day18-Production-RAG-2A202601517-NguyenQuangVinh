# Reflection — Lab 18: Production RAG Pipeline

**Họ tên:** Nguyễn Quang Vinh (2A202601517) · Bài tập cá nhân

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 nhóm câu liền kề có cosine similarity ≥ 0.85 vào cùng chunk. Trên corpus 40 file tiếng Việt, semantic sinh ra số chunk nhiều hơn hierarchical vì mỗi lần similarity tụt dưới ngưỡng là tách nhóm mới — nhạy với văn phong ngắn, nhiều câu độc lập kiểu policy doc. |
| Hierarchical (parent-child) chunking | M1 | `chunk_hierarchical()` | Parent 2048 ký tự / child 256 ký tự. Đây là default cho pipeline production (`src/pipeline.py` dùng `chunk_hierarchical` chứ không phải semantic/structure) — retrieve theo child (precision cao, đoạn ngắn khớp query tốt) nhưng trả về parent context đầy đủ hơn cho LLM sinh câu trả lời. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Parse header markdown (`#`, `##`, `###`) giữ nguyên `section` trong metadata — hữu ích cho corpus có cấu trúc rõ (vd `nghi_phep_nam_v2024.md` có heading "Số ngày phép năm"), nhưng vô dụng với PDF không có markdown heading (BCTC.pdf, Nghị định 13-2023 — cả 2 đều là scan ảnh nên bị `load_documents()` bỏ qua từ đầu). |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF giải quyết vấn đề BM25 và dense có scale điểm số khác nhau (BM25 score không bounded, cosine similarity trong [0,1]) — thay vì cộng trực tiếp 2 loại điểm không tương thích, RRF chỉ dùng **rank** (1/(k+rank+1)) nên fair giữa 2 phương pháp. Test `test_rrf_merges` xác nhận doc xuất hiện ở cả 2 danh sách (dù rank khác nhau) được ưu tiên lên đầu. |
| Vietnamese word segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` (vd "nghỉ_phép") — nếu không `replace("_", " ")` trước khi BM25 tokenize bằng `split(" ")`, BM25 sẽ coi "nghỉ_phép" là 1 token duy nhất trong khi query "nghỉ phép" tách thành 2 token → không bao giờ khớp. Đây là bug tinh vi, dễ pass test cơ bản nhưng fail silent trên query thực tế. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Dùng `sentence_transformers.CrossEncoder` với `BAAI/bge-reranker-v2-m3` thay vì `FlagEmbedding.FlagReranker` — code scaffold đã cảnh báo trước: FlagReranker crash với `transformers>=5.0`. Test riêng lẻ (`pytest tests/test_m3.py::test_rerank_returns` chạy đơn lẻ) confirm đúng: doc "nghỉ phép" luôn được rerank lên top so với doc "VPN"/"mật khẩu" không liên quan — cross-encoder học joint representation (query, doc) tốt hơn nhiều so với chỉ dùng embedding similarity riêng lẻ. |
| RAGAS 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) | M4 | `evaluate_ragas()` | Trên 2 câu đánh giá được, **Context Precision thấp nhất (0.33)** ở cả 2 câu — cho thấy Naive Baseline (paragraph chunking + dense-only, không rerank) đưa quá nhiều context nhiễu vào prompt. Đây khớp lý thuyết Diagnostic Tree đã học: context_precision thấp → root cause "quá nhiều chunk không liên quan" → fix "thêm reranking" — chính là lý do M3 tồn tại trong pipeline. |
| Contextual embeddings (Anthropic-style prepend) | M5 | `contextual_prepend()` | Prepend 1 câu mô tả vị trí/chủ đề của chunk trong tài liệu trước khi embed — về lý thuyết giảm retrieval failure vì query "nghỉ phép năm 2024" giờ có thêm tín hiệu ngữ cảnh "Trích từ nghi_phep_nam_v2024.md, nói về..." thay vì chỉ có nội dung chunk thuần. Không đo được % giảm failure cụ thể trong session này do giới hạn thời gian API (xem Phần 2), nhưng logic implement đúng theo Anthropic technique: context câu ngắn + `\n\n` + text gốc, giữ nguyên `original_text`. |

## Phần 2: Khó khăn & giải quyết

### Khó khăn 1 — Native crash trên Windows khi load model 2 lần trong 1 process

**Lỗi cụ thể:** `Segmentation fault` (exit code 139) khi gọi `SentenceTransformer('BAAI/bge-m3')` lần thứ 2
trong cùng Python process, hoặc khi `qdrant_client` và `sentence-transformers`/torch cùng tồn tại trong 1
process.

**Cách debug:** Cô lập từng bước bằng script Python nhỏ chạy độc lập (không qua pytest) để loại trừ pytest
là nguyên nhân. Phát hiện: `QdrantClient()` init trước, rồi load `SentenceTransformer` sau → segfault ngay
lập tức, bất kể thứ tự. Đây là OpenMP DLL conflict giữa gRPC/protobuf (dùng bởi qdrant-client) và torch's
runtime trên Windows.

**Fix:** Set biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` trước khi chạy — đây là workaround chuẩn cho lỗi
"multiple OpenMP runtimes" trên Windows.

### Khó khăn 2 — `ragas.evaluate()` deadlock vĩnh viễn trên Windows

**Lỗi cụ thể:** Không có traceback, không có exception — process chạy ở 0% CPU vô thời hạn khi gọi
`ragas.evaluate()` (thư viện ragas 0.1.22). Đã thử: `WindowsSelectorEventLoopPolicy`, `raise_exceptions=True`,
`max_workers=1`, gọi trực tiếp `metric.score()` bypass executor — tất cả đều deadlock ở cùng một điểm
(`asyncio.run()` bên trong `Executor.results()`/`score()`).

**Cách debug:** Đọc source code `ragas.executor.Executor.results()` và `Faithfulness.score()` trực tiếp
trong site-packages để hiểu cơ chế async. Test riêng `llm.ainvoke()` thuần (không qua ragas) → hoạt động
bình thường (2.9s) → xác nhận vấn đề nằm trong tầng orchestration của ragas, không phải LLM call.

**Fix:** Viết lại `evaluate_ragas()` trong `src/m4_eval.py` để tự tính 4 metrics bằng cách gọi
`llm.invoke()` **đồng bộ, tuần tự** thay vì qua `ragas.evaluate()`'s async executor — giữ đúng phương pháp
luận RAGAS (LLM-as-judge trên statements được tách từ answer/ground_truth) nhưng thực thi khác.

### Khó khăn 3 — OpenAI hết credit → chuyển Gemini → Gemini free tier quá chậm (chưa giải quyết trọn vẹn)

**Lỗi cụ thể:** `openai.RateLimitError: insufficient_quota` (tài khoản không có credit) → chuyển sang
Google Gemini (`GOOGLE_API_KEY`) → gặp tiếp `google.api_core.exceptions.NotFound` (model
`gemini-3.6-flash` bị deprecate ngay khi mới ra) → đổi `gemini-3.5-flash-lite` → hoạt động nhưng free tier
giới hạn **15 requests/phút**, và với ~7 lời gọi LLM/câu cho RAGAS scoring, throughput thực tế chỉ đạt
~1 câu/30-60 phút sau khi quota đã bị các lần test trước tiêu hao.

**Cách giải quyết (một phần):** Thêm pacing (`time.sleep(5)` giữa mỗi lời gọi, giữ dưới 12 RPM) và giảm mẫu
đánh giá từ 20 xuống 5 câu (`EVAL_SAMPLE_SIZE` trong `config.py`). Vẫn không đủ nhanh để hoàn tất trong thời
gian buổi lab — **đây là hạn chế thực tế chưa giải quyết trọn vẹn**, dẫn đến `failure_analysis.md` chỉ có
dữ liệu thật cho 2/20 câu (Naive Baseline), thiếu số liệu Production pipeline để so sánh đầy đủ.

**Kiến thức thiếu → cách bổ sung:** Chưa từng làm việc với free-tier rate limit ở quy mô "phải đánh giá
hàng chục câu × nhiều LLM call/câu" trước đây — bài học rút ra là cần ước tính tổng số API calls **trước
khi** chọn model/tier, không phải sau khi đã code xong. Lần sau sẽ dùng `tenacity` với exponential backoff
+ jitter thay vì `time.sleep()` cố định, và cân nhắc batch/cache LLM calls (vd cache statements extraction
nếu answer giống nhau) để giảm tổng số request.

## Phần 3: Action Plan cho project

### Hiện tại
- RAG pipeline hiện tại: chưa có project RAG cá nhân đang chạy production — lab này là nền tảng kỹ thuật
  đầu tiên áp dụng đầy đủ chunking nâng cao + hybrid search + rerank + eval có hệ thống.
- Known issues: (dựa trên lab) rate limit của free-tier LLM API là nút thắt cổ chai thực sự khi cần eval
  ở quy mô lớn — cần tính vào budget/kiến trúc ngay từ đầu, không phải retrofit sau.

### Plan áp dụng
1. [ ] Chunking strategy: **Hierarchical (parent-child)** — vì retrieve theo child (256 chars) tăng
   precision, trả parent (2048 chars) giữ đủ ngữ cảnh cho LLM. Semantic chunking chỉ dùng khi corpus có
   văn phong tự do (blog, essay); structure-aware chỉ hiệu quả với tài liệu có markdown/heading rõ ràng.
2. [ ] Search: **Hybrid (BM25 + Dense qua RRF)** — dense-only miss các câu hỏi cần exact-match (số liệu,
   tên riêng, mã sản phẩm); BM25-only miss các câu hỏi diễn đạt khác từ ngữ nhưng cùng ý. RRF kết hợp cả
   hai mà không cần tune trọng số thủ công.
3. [ ] Reranking: **Có, CrossEncoder** (`bge-reranker-v2-m3`) — dữ liệu thật thu được (dù mẫu nhỏ) cho thấy
   context_precision là điểm yếu nhất của retrieval không-rerank (0.33/1.0 ở cả 2 câu test), nên đây là
   ưu tiên cải thiện số 1 trước khi tối ưu các bước khác.
4. [ ] Evaluation: **RAGAS-style metrics nhưng tự implement gọi LLM đồng bộ** (như đã làm trong
   `src/m4_eval.py`) thay vì phụ thuộc `ragas.evaluate()` — tránh rủi ro deadlock/thư viện không ổn định
   trên môi trường Windows, và dễ kiểm soát rate limit/cost hơn khi tự viết logic gọi LLM.
5. [ ] Enrichment: **Contextual prepend** ưu tiên trước — chi phí thấp nhất (1 câu ngắn/chunk) trong khi
   Anthropic benchmark cho thấy đây là kỹ thuật đơn lẻ hiệu quả nhất (giảm 49% retrieval failure). HyQA và
   auto-metadata sẽ thêm sau nếu ngân sách API cho phép, dùng chế độ combined (`_enrich_single_call`, 1
   call/chunk) để tiết kiệm chi phí thay vì 4 lời gọi riêng lẻ.

### Timeline
- Tuần 1: Setup pipeline production với ngân sách LLM API rõ ràng ngay từ đầu (ước tính tổng số call cần
  thiết trước khi chọn provider/tier) — bài học trực tiếp từ khó khăn #3 ở trên.
- Tuần 2: Chạy full evaluation (≥20 câu) với provider có quota đủ lớn, điền đầy đủ bảng so sánh
  Naive vs Production, xác nhận giả thuyết "reranking cải thiện context_precision" bằng số liệu thật.
- Tuần 3: Tối ưu dựa trên failure analysis — nếu context_recall thấp thì cải thiện chunking/thêm BM25;
  nếu faithfulness thấp thì siết prompt/giảm temperature.
