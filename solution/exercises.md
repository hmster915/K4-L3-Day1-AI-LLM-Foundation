# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Qua 4 phản hồi, tôi nhận thấy khi temperature ở mức thấp (0.0), câu trả lời mang tính chuẩn mực, an toàn và giống văn bản bách khoa toàn thư. Khi tăng dần temperature lên 0.5 và 1.0, model bắt đầu thêm thắt các chi tiết phong phú hơn (như năm phát hiện) và cách diễn đạt sinh động, cảm xúc hơn (so sánh với tòa nhà chọc trời, dùng dấu chấm than). Ở mức cao (1.5), văn phong trở nên phá cách và mang tính hình tượng rất cao. Nhìn chung, temperature càng cao, câu trả lời càng sáng tạo và đa dạng về từ vựng.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Trong môi trường thực tế (production) cho chatbot hỗ trợ khách hàng, tôi sẽ đặt temperature ở mức rất thấp, khoảng 0.0 đến 0.2. Lý do là vì trong dịch vụ khách hàng, tính chính xác, nhất quán và an toàn là quan trọng nhất. Việc đặt temperature thấp giúp hạn chế tối đa tình trạng bot "ảo giác" (hallucinate) như tự bịa ra chính sách bảo hành sai lệch hoặc hứa hẹn những điều công ty không thể đáp ứng. Điều này đảm bảo bot luôn cung cấp thông tin chuẩn xác và chuyên nghiệp.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Dựa vào bảng giá, giá token đầu ra (output) của GPT-4o là 15$/1M token, trong khi GPT-4o-mini chỉ là 0.6$/1M token. Vì vậy, đối với cùng một workload, GPT-4o đắt hơn chính xác 25 lần so với model mini.*
-Trường hợp xứng đáng dùng GPT-4o: Giải quyết các bài toán lập trình phức tạp, phân tích dữ liệu chuyên sâu, hoặc các tác vụ đòi hỏi khả năng suy luận logic cao.
-Trường hợp nên dùng GPT-4o-mini: Chatbot giao tiếp thông thường, tóm tắt văn bản dài, trích xuất dữ liệu, hoặc các hệ thống có lượng người dùng khổng lồ cần tối ưu chi phí.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Hai phản hồi khác biệt hoàn toàn về từ vựng, độ dài và cách dùng ví dụ. Vai giáo viên trả lời rất ngắn gọn, dùng từ ngữ đơn giản và lấy ví dụ gần gũi như "một cuốn sổ lớn" để trẻ em 8 tuổi dễ hình dung. Ngược lại, vai chuyên gia đưa ra câu trả lời dài hơn, sử dụng dày đặc các thuật ngữ kỹ thuật (sổ cái phân tán, mã băm, Proof of Work, hợp đồng thông minh). Điều này cho thấy system prompt đóng vai trò cốt lõi trong việc định hình hành vi, mức độ chuyên sâu và văn phong của model để phù hợp với từng đối tượng cụ thể.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
>*Đoạn hội thoại: Trí tuệ nhân tạo đang thay đổi cách chúng ta sống và làm việc mỗi ngày. Từ những trợ lý ảo trên điện thoại thông minh cho đến các hệ thống tự động hóatrong các nhà máy lớn, công nghệ này mang lại vô số lợi ích về hiệu suất và sự tiện lợi. Tuy nhiên, sự phát triển nhanh chóng của nó cũng đặt ra nhiềuthách thức mới. Các vấn đề về bảo mật dữ liệu, quyền riêng tư và nguy cơ mất việc làm đang trở thành những chủ đề được tranh luận gay gắt trên toàn cầu hiện nay.* 
*Khi thử nghiệm một đoạn hội thoại tiếng Việt (bao gồm khoảng 100 từ đầu vào và câu trả lời của bot), tổng số từ rơi vào khoảng hơn 160 từ. Theo công thức ước lượng (số từ / 0.75), đáng lẽ lượng token tiêu thụ chỉ khoảng 215 token. Tuy nhiên, thống kê thực tế bằng `tiktoken` cho ra kết quả lên tới 231 token.*
*Sự chênh lệch này xảy ra vì tiếng Việt tốn nhiều token hơn tiếng Anh. Các bộ Tokenizer của AI được huấn luyện chủ yếu bằng tiếng Anh, nên một từ tiếng Anh thường chỉ tính là 1 token. Ngược lại, các từ tiếng Việt (đặc biệt là các chữ cái có dấu) thường không có sẵn trong từ điển gốc của AI. Do đó, Tokenizer buộc phải cắt vụn từ tiếng Việt ra thành nhiều sub-word (âm tiết nhỏ) hoặc chia thành các byte lẻ, làm cho tổng số token tăng vọt so với số từ thực tế.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming đặc biệt quan trọng trong các ứng dụng tương tác trực tiếp (như chatbot), vì nó giúp người dùng đọc được ngay từng chữ AI đang viết, xóa bỏ cảm giác chờ đợi và tránh việc giao diện bị "đơ" khi xử lý câu trả lời dài. Ngược lại, non-streaming lại phù hợp hơn cho các tác vụ chạy ngầm (background jobs) như trích xuất dữ liệu, kiểm duyệt nội dung, hoặc khi hệ thống cần AI trả về một file JSON hoàn chỉnh, vì máy tính chỉ cần kết quả cuối cùng để xử lý chứ không cần xem hiệu ứng gõ từng chữ.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp giảm áp lực lên server bằng cách kéo giãn thời gian chờ sau mỗi lần thử thất bại (ví dụ: chờ 1s, rồi 2s, rồi 4s), tạo "khoảng nghỉ" để hệ thống API kịp phục hồi. Nếu hàng nghìn client cùng dùng một delay cố định (ví dụ 1 giây), khi server báo lỗi, tất cả hàng nghìn máy đó sẽ đồng loạt gửi lại request vào đúng 1 giây sau. Hiện tượng này gọi là "thác lũ" (thundering herd), nó tạo ra một lượng truy cập khổng lồ cùng lúc và sẽ ngay lập tức đánh sập server vừa mới phục hồi.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Tôi chọn persona là một "Trợ lý kinh doanh nội bộ". System prompt: "Bạn là trợ lý phân tích dữ liệu và kinh doanh. Hãy trả lời cực kỳ ngắn gọn, trực diện và giữ thái độ chuyên nghiệp. Tuyệt đối không tự bịa đặt số liệu, nếu không biết hãy nói không biết."*
Giải thích từ ngữ: 
(1) "Cực kỳ ngắn gọn": Giúp tiết kiệm thời gian đọc cho nhân viên và tối ưu hóa chi phí token API cho công ty. 
(2) "Tuyệt đối không tự bịa đặt số liệu": Là ranh giới bảo mật tối quan trọng trong môi trường doanh nghiệp để ngăn chặn rủi ro bot bị "ảo giác" (hallucinate), dẫn đến các quyết định kinh doanh sai lầm.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý hiện tại là bộ nhớ ngắn hạn bị cắt cứng (chỉ giữ 3 lượt hội thoại). Trong môi trường doanh nghiệp, khi nhân viên đang trao đổi để giải quyết một chuỗi vấn đề hoặc phân tích dữ liệu phức tạp, việc bot đột ngột "quên" các thông số đã cung cấp ở đầu cuộc hội thoại sẽ làm đứt gãy luồng công việc.*
Đề xuất cải thiện: Triển khai cơ chế "Tóm tắt ngữ cảnh" (Memory Summarization). Cách triển khai: Thay vì xóa hoàn toàn các tin nhắn cũ khi vượt quá 3 lượt, hệ thống sẽ chạy ngầm một lời gọi API model nhỏ (như GPT-4o-mini) để tóm tắt các nội dung cũ thành một đoạn văn bản ngắn. Đoạn tóm tắt này sau đó được tiêm (inject) ngược lại vào system prompt. Bằng cách này, bot có bộ nhớ dài hạn (long-term memory) về toàn bộ bối cảnh làm việc mà công ty vẫn kiểm soát được chi phí token.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
