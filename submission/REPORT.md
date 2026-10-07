# Lab 21 — Evaluation Report

**Họ tên**: Trần Quốc Bảo Long  **MSSV**: 2A202602696  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4, CUDA, sm_75, 14.6 GB

> **Trạng thái kết quả:** NB2 và NB5 đã chạy đầy đủ (`n=50` target, `n=15` regression). NB3/NB4 đã chạy đủ cấu hình, 2 epochs / 30 bước. Cổng NB5 báo `FAILED` vì regression giảm 0.180; fine-tune tăng target nhưng chưa giữ được năng lực tổng quát.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (dataset mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 theo tier T4 — p95 đo được là 98 token, gợi ý làm tròn là 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 bước optimizer |

**Template có giữ khối `<think>` không?** Có — kiểm tra NB1 cho thấy reasoning được giữ trong render. Dataset mặc định có câu trả lời JSON, không có trace suy luận thực tế.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (trích từ log):

```text
</think>
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

NB1 đo p95 là 98 token; cấu hình tier vẫn dùng `max_length=1024`. Tôi giữ mặc định của tier để các run tuân theo cấu hình lab, dù dữ liệu hiện tại có thể dùng độ dài ngắn hơn.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

NB2 đã đo baseline trước khi train với toàn bộ tập: 50 target và 15 regression (`EVAL_LIMIT` bỏ trống). NB5 sau đó đánh giá fine-tune trên cùng các tập đầy đủ.

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3153.5 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1025.2 |
| (c) LoRA fine-tune | 0.970 | 0.611 | 1.000 | 1384.5 |

**(b) có thật sự mạnh hơn (a) không?** Có. Trên 50 mẫu target, prompt tối ưu tăng target từ 0 lên 0.765, format từ 0 lên 1.0, giữ regression ở 0.791 (15 mẫu) và giảm latency từ khoảng 3154 xuống 1025 ms/mẫu. Log không cho biết tôi có sửa `OPTIMIZED_PROMPT` hay không; đây là prompt tối ưu có sẵn trong pipeline.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target (NB5 §4)** | thời gian train (s) | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6262 | 0.970 (n=50) | 404.9 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5372 | 0.970 (n=50) | 275.4 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 (n=50) | 409.9 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 (n=50) | 484.3 | 3.86 |

Các target ở bảng là kết quả NB5 đầy đủ trên 50 mẫu.

**4.1 — `attn_only` so với `correct`.** Hai cấu hình gần như khớp số tham số huấn luyện. Trên 50 mẫu target, `attn_only` hoà với `correct` ở target (0.970) và format (1.0). Train loss của `attn_only` thấp hơn (0.5372 so với 0.6262), nhưng target không cao hơn; vì vậy loss thấp hơn không chứng minh vị trí adapter tốt hơn. Trong phép đo này, tăng rank cho q,v không vượt all-linear dù điểm target ngang nhau.

**4.2 — `wrong_lr`.** Run dùng learning rate 0.00001, thấp hơn 10 lần so với cấu hình chính. Train loss cuối là 1.5702, so với 0.6262 của `correct`, và target/format đều bằng 0 trên 50 mẫu. Nếu chỉ nhìn loss mà không biết learning rate, có thể quy kết sai rằng kiến trúc hay dữ liệu là nguyên nhân; ở đây cấu hình chỉ đổi LR nhưng chất lượng đầu ra khác hẳn. Log có một số `grad_norm: nan`, nhưng trainer vẫn hoàn tất và lưu adapter.

**4.3 — `qlora`.** QLoRA đạt target 0.940, thấp hơn `correct` 0.970 trên 50 mẫu, trong khi format vẫn là 1.0. VRAM đỉnh giảm từ 8.78 xuống 3.86 GB, tiết kiệm khoảng 4.92 GB (56%). Ở phép đo này, QLoRA đánh đổi 0.03 điểm target để giảm bộ nhớ đáng kể; tuy nhiên nó cũng chưa giải quyết được regression của cấu hình chính.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.180` · `valid_trace_rate = 0.00` (n=50 target, n=15 regression)

Fine-tune đạt target 0.970, cao hơn baseline prompt tối ưu 0.765 đúng 0.205 điểm trên 50 mẫu, đồng thời giữ format ở 1.0. Tuy nhiên regression giảm từ 0.791 xuống 0.611 trên 15 mẫu, tức giảm 0.180, vượt xa ngưỡng cho phép 0.020; cổng vì vậy báo `FAILED`. Model học tốt hơn cho ticket CSKH nhưng suy giảm trên nhóm câu hỏi tổng quát được kiểm tra. `valid_trace_rate=0.0` không ảnh hưởng bài toán JSON hiện tại vì dữ liệu huấn luyện không có reasoning trace thực tế. Tôi chưa nên deploy adapter này như một thay thế trực tiếp cho base model. Hướng xử lý theo log là thêm 1–5% replay data để giảm regression, sau đó huấn luyện và đánh giá lại cả target lẫn regression. Kết quả thất bại vẫn có giá trị: chỉ số target cao không đủ để chấp nhận model nếu regression giảm mạnh.

---

## 6. Định tính — bắt buộc có cả ca THUA

Log in ba ca điểm thấp nhất và cao nhất của fine-tune, nhưng không in nhãn vàng hoặc dự đoán baseline prompt theo từng ticket. Vì vậy, bảng dưới ghi ticket và điểm fine-tune; các output bị cắt trong log.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Hỏi tiền hoàn cho bình giữ nhiệt chưa về | Chưa có trong log | Chưa có trong log | score 0.75 | 3/4 trường đúng |
| 2 | Nồi chiên không dầu thiếu phụ kiện | Chưa có trong log | Chưa có trong log | score 0.75 | 3/4 trường đúng |
| 3 | Áo khoác gió bị lỗi, “khi nào tiện” | Chưa có trong log | Chưa có trong log | score 0.75 | 3/4 trường đúng |
| 4 | Ốp lưng, hỏi shipper | Chưa có trong log | Chưa có trong log | score 1.00 | 4/4 trường đúng |
| 5 | Ốp lưng, hỏi giá | Chưa có trong log | Chưa có trong log | score 1.00 | 4/4 trường đúng |
| 6 | Ốp lưng sai màu, cần sớm | Chưa có trong log | Chưa có trong log | score 1.00 | 4/4 trường đúng |

Ba ca điểm thấp nhất đều đạt 0.75, tức mỗi ca sai một trong bốn trường. Chúng liên quan đến hoàn tiền, thiếu phụ kiện và lỗi sản phẩm; ba ca điểm cao nhất đều là ticket về ốp lưng. Đây chỉ là sáu ví dụ được log in ra, chưa đủ để quy kết nguyên nhân lỗi; log cũng không kèm nhãn vàng.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi chưa nên deploy bản fine-tune này làm model thay thế trực tiếp. Trên 50 mẫu target, nó đạt 0.970, vượt baseline prompt tối ưu 0.765; format giữ ở 1.0. Nhưng regression giảm từ 0.791 xuống 0.611 trên 15 mẫu, nên cổng báo `FAILED`. Model chuyên biệt hóa tốt hơn cho ticket nhưng làm suy giảm năng lực tổng quát, và mức tăng target không bù được regression theo tiêu chí lab. NB4 cho thấy learning rate thấp hơn 10 lần đi kèm train loss cao hơn và target bằng 0; QLoRA giảm VRAM khoảng 56% nhưng target thấp hơn cấu hình chính 0.03 điểm; `attn_only` hoà target với `correct` dù train loss thấp hơn. Vì vậy, loss, VRAM và target riêng lẻ đều không đủ để chọn model. Tôi sẽ thử thêm 1–5% replay data như NB5 gợi ý, rồi chạy lại đánh giá đầy đủ. Chỉ cân nhắc triển khai thử nghiệm nếu regression nằm trong ngưỡng và lợi thế target vẫn còn; nếu không, baseline prompt tối ưu hiện là lựa chọn tốt hơn.

**Ba điều tôi học được:**
1. Mask `assistant-only` đã xác nhận câu hỏi không bị tính loss và câu trả lời có nằm trong loss; đây là kiểm tra cần làm trước khi huấn luyện.
2. Prompt baseline là một đối thủ thực sự: trên 50 mẫu, chỉ tối ưu prompt đã đưa target từ 0 lên 0.765.
3. Train loss thấp hơn không đảm bảo target cao hơn; `attn_only` có loss thấp hơn `correct` nhưng target chỉ hoà, còn `wrong_lr` cho thấy learning rate tác động mạnh.

**Tiếp theo tôi sẽ:** thêm 1–5% replay data, huấn luyện lại rồi đo xem có giữ target cao mà phục hồi regression không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
