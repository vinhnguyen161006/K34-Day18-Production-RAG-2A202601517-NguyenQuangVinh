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
| RAGAS 4 metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) | M4 | `evaluate_ragas()` | Trên 5/5 câu đánh giá đầy đủ cả Naive và Production (dữ liệu thật): **Context Precision tăng từ 0.4667 (Naive) lên 0.6667 (Production, +0.20)** — đúng giả thuyết lý thuyết, reranking lọc context nhiễu. Nhưng bất ngờ: Faithfulness (0.67→0.57), Answer Relevancy (0.91→0.81), Context Recall (0.87→0.67) đều **giảm** ở Production — kết quả thật không khớp hoàn toàn kỳ vọng "Production luôn thắng", cho thấy M5 enrichment có thể đang gây nhiễu chunk content (xem `failure_analysis.md`). |
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

### Khó khăn 3 — OpenAI hết credit → chuyển Gemini → Gemini free tier rất chậm

**Lỗi cụ thể:** `openai.RateLimitError: insufficient_quota` (tài khoản không có credit) → chuyển sang
Google Gemini (`GOOGLE_API_KEY`) → gặp tiếp `google.api_core.exceptions.NotFound` (model
`gemini-3.6-flash` bị deprecate ngay khi mới ra) → đổi `gemini-3.5-flash-lite` → hoạt động nhưng free tier
giới hạn **15 requests/phút**, và với ~7 lời gọi LLM/câu cho RAGAS scoring, throughput thực tế chỉ đạt
~1 câu/20-40 phút khi quota bị các lần test trước tiêu hao.

**Cách giải quyết:** Thêm pacing (`time.sleep(5)` giữa mỗi lời gọi, giữ dưới 12 RPM), giảm mẫu đánh giá từ
20 xuống 5 câu (`EVAL_SAMPLE_SIZE` trong `config.py`), và kiên nhẫn chạy nền qua nhiều giờ. Cuối cùng thu
được **dữ liệu RAGAS thật đầy đủ cho cả Naive Baseline và Production pipeline (5/5 câu mỗi bên)** —
kết quả trong `failure_analysis.md` phản ánh đúng thực tế, không có số liệu giả lập.

**Kiến thức thiếu → cách bổ sung:** Chưa từng làm việc với free-tier rate limit ở quy mô "phải đánh giá
hàng chục câu × nhiều LLM call/câu" trước đây — bài học rút ra là cần ước tính tổng số API calls **trước
khi** chọn model/tier, không phải sau khi đã code xong. Lần sau sẽ dùng `tenacity` với exponential backoff
+ jitter thay vì `time.sleep()` cố định, và cân nhắc batch/cache LLM calls (vd cache statements extraction
nếu answer giống nhau) để giảm tổng số request.

### Khó khăn 4 — CrossEncoder + SentenceTransformer trong cùng process gây `OSError: paging file too small`

**Lỗi cụ thể:** Khi `src/pipeline.py` chạy `DenseSearch` (dùng `SentenceTransformer('bge-m3')`) rồi sau đó
gọi `CrossEncoderReranker.rerank()` (lazy-load `CrossEncoder('bge-reranker-v2-m3')`) trong **cùng process**,
process crash với exit code 139 (segfault). Ban đầu tưởng là OpenMP conflict (giống Khó khăn 1), nhưng debug
sâu hơn qua `subprocess.run(..., capture_output=True)` để bắt full stderr mới lộ ra lỗi thật:
`OSError: The paging file is too small for this operation to complete. (os error 1455)` khi
`safetensors.safe_open()` cố mmap file lớn — đây là giới hạn virtual memory commit ở tầng process-tree của
Windows, không phải RAM/đĩa vật lý bị đầy (kiểm tra bằng `Get-CimInstance Win32_OperatingSystem` cho thấy
RAM/disk còn dư dả).

**Cách debug:** Thử tách rerank ra `subprocess.run()` riêng (`src/_rerank_worker.py`) — vẫn crash dù là
process con độc lập, vì process cha vẫn đang giữ `SentenceTransformer` trong memory và Windows áp giới hạn
theo process tree. Chỉ khi tách rerank thành **2 script Python hoàn toàn độc lập chạy tuần tự qua dòng lệnh**
(không có quan hệ cha-con: script A retrieve → ghi JSON → thoát hẳn; script B đọc JSON → rerank + eval) thì
mới hết crash.

**Fix:** Với `pipeline.py`, dùng `subprocess.run([sys.executable, "-m", "src._rerank_worker", ...])` gọi từ
một script cha **không load SentenceTransformer trước đó trong cùng lần chạy** — cụ thể đã tách bước
"retrieve" (cần `DenseSearch`/bge-m3) và bước "rerank + generate + eval" (cần `CrossEncoder`) thành 2 lần
chạy `python` riêng biệt, dùng file JSON trung gian để truyền dữ liệu.

**Kiến thức thiếu → cách bổ sung:** Trước đây coi "process cô lập" chỉ cần `subprocess` là đủ để tránh xung
đột tài nguyên native — thực tế trên Windows, process tree vẫn có thể chia sẻ giới hạn virtual memory commit
với process cha. Bài học: khi debug native crash, luôn bắt `stderr` đầy đủ (`capture_output=True`) thay vì
chỉ nhìn exit code — exit code 139/3221225477 chỉ nói "có gì đó sập", không nói lý do; log lỗi thật
(`paging file too small`) mới dẫn tới fix đúng.

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
3. [ ] Reranking: **Có, CrossEncoder** (`bge-reranker-v2-m3`) — dữ liệu thật (5/5 câu) xác nhận
   context_precision tăng +0.20 (0.4667→0.6667) nhờ reranking, đúng lý thuyết. Nhưng cần điều tra thêm vì
   3/4 metrics khác lại giảm ở Production — nghi ngờ M5 enrichment là nguyên nhân, cần A/B test tách riêng
   tác động M3 (rerank) và M5 (enrichment) thay vì đo gộp cả pipeline.
4. [ ] Evaluation: **RAGAS-style metrics nhưng tự implement gọi LLM đồng bộ** (như đã làm trong
   `src/m4_eval.py`) thay vì phụ thuộc `ragas.evaluate()` — tránh rủi ro deadlock/thư viện không ổn định
   trên môi trường Windows, và dễ kiểm soát rate limit/cost hơn khi tự viết logic gọi LLM.
5. [ ] Enrichment: **Cẩn trọng hơn với contextual prepend** — dữ liệu thật cho thấy Production (có M5)
   giảm faithfulness/answer_relevancy/context_recall so với Naive (không M5, không rerank). Trước khi áp
   dụng cho project cá nhân, cần A/B test rõ ràng: Naive vs Naive+M3(rerank, không enrichment) vs
   Naive+M3+M5(đầy đủ) để tách bạch đóng góp của từng kỹ thuật thay vì cộng gộp và giả định luôn có lợi.

### Timeline
- Tuần 1: Setup pipeline production với ngân sách LLM API rõ ràng ngay từ đầu (ước tính tổng số call cần
  thiết trước khi chọn provider/tier) — bài học trực tiếp từ khó khăn #3 ở trên.
- Tuần 2: Chạy full evaluation (≥20 câu, provider có quota đủ lớn) với 3 cấu hình tách bạch (Naive /
  +Rerank / +Rerank+Enrichment) để xác định chính xác kỹ thuật nào đóng góp tích cực — dữ liệu 5 câu ở lab
  này cho thấy kết quả gộp không đơn giản như lý thuyết ("thêm kỹ thuật = luôn tốt hơn").
- Tuần 3: Tối ưu dựa trên failure analysis — nếu context_recall thấp thì cải thiện chunking/thêm BM25;
  nếu faithfulness thấp thì siết prompt/giảm temperature; nếu enrichment gây hại thì tắt hoặc cải thiện
  prompt của `contextual_prepend()`.
