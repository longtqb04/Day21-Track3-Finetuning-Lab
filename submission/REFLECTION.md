# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

> Những prompt tốt đã đưa target từ 0 lên 0.765 trước cả khi fine-tune. Fine-tune tăng tiếp lên 0.970 trên 50 mẫu, nhưng regression lại giảm từ 0.791 xuống 0.611. Tôi đã nghĩ model chỉ cần học đúng định dạng và nhãn ticket là sẽ tốt hơn; kết quả cho thấy cải thiện trên tác vụ chính có thể đi kèm với việc làm hỏng năng lực khác.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

> Phần tốn thời gian nhất là NB4, khoảng 22 phút cho ba cấu hình đối chứng; lần tải model đầu tiên cũng mất thêm khoảng 7 phút. Tôi đoán huấn luyện sẽ là phần lâu nhất, nên NB4 đúng là đáng kể, nhưng không ngờ riêng việc tải checkpoint 9.32 GB lại chiếm nhiều thời gian như vậy. Các lượt đánh giá đầy đủ NB2 và NB5 sau đó nhanh hơn dự kiến: lần lượt khoảng 5 và 11 phút trên T4.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

> Trước lab, tôi nghĩ train loss giảm mạnh là dấu hiệu khá chắc rằng model đã học tốt, và fine-tune thường sẽ làm model tốt hơn cho tác vụ mình quan tâm mà ít ảnh hưởng phần khác. Giờ tôi không còn tin hai điều đó nếu chưa nhìn eval, `attn_only` có train loss thấp hơn `correct` nhưng target chỉ hòa; còn model chính tăng target 0.205 điểm mà mất 0.180 điểm regression. Cần chốt baseline và tập đánh giá trước khi train, rồi chấp nhận cả kết quả `FAILED` nếu số đo nói vậy.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

> Tôi dùng AI assistant để đọc log từng notebook, giải thích các chỉ số và điền report từ kết quả đã chạy. Sai sót đáng kể là assistant ban đầu diễn giải NB5 smoke run (8 mẫu) như một kết quả khả quan, trước khi đối chiếu với NB2 chạy đầy đủ trên 50 mẫu. Kết quả đó chưa thể so trực tiếp với baseline đầy đủ. Sau khi chạy lại NB5 không giới hạn, phép so sánh đúng cho thấy regression giảm mạnh và gate `FAILED`. Bài học của tôi là luôn kiểm tra số mẫu, chế độ smoke/full và artefact gốc trước khi tin vào kết luận do AI tóm tắt.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

> Đầu tiên, tôi sẽ cùng khách hàng xác định điều gì được xem là thành công và điều gì tuyệt đối không được suy giảm; sau đó tạo một tập eval đại diện, tách riêng và đóng băng trước khi huấn luyện. Tôi sẽ đo base model với prompt tốt trên tập đó để biết fine-tuning cần vượt qua mốc nào. Với dữ liệu khách hàng, tôi cũng sẽ xác nhận quyền sử dụng và loại bỏ thông tin nhạy cảm trước khi đưa vào bất kỳ pipeline nào. Chỉ khi baseline và cách đo đã rõ, tôi mới chọn model và bắt đầu train.
