# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh**. Trong JSON chỉ có `rank`, `class_id`, `class_name`, `score` cho toàn ảnh, hoàn toàn không có toạ độ box hay polygon, nên không chỉ vào vật thể cụ thể nào. Model chỉ trả lời “cả ảnh này giống lớp nào nhất” (top-5: cab, minibus, police_van, recreational_vehicle, streetcar).

3. Danh sách tên lớp **đi kèm taxonomy/checkpoint**, không phải model tự nghĩ ra. Checkpoint `yolo11n-cls.pt` được huấn luyện sẵn với bộ lớp ImageNet-1K (1000 lớp); model chỉ chọn trong danh sách cố định đó. Vì vậy nó không thể trả về lớp nằm ngoài danh sách.

4. Vì cùng một tên có thể khác nghĩa hoặc khác số thứ tự giữa các taxonomy/phiên bản. `class_id` (468) chỉ có nghĩa khi biết đang dùng bảng nào; `taxonomy_name` (ImageNet-1K) cho biết bảng nào; `class_name` (cab) giúp người đọc hiểu nhanh. Thiếu `taxonomy_name` thì id 468 có thể bị map sai sang lớp khác (ví dụ COCO-80 có id khác hẳn), làm dữ liệu không tái lập và không so sánh được.

5. Guideline phải nói rõ **quy tắc chọn một lớp khi ảnh có nhiều vật/chủ đề**: ví dụ chọn theo chủ thể chính (vật to nhất / ở trung tâm / được chụp để làm nổi bật), cách xử lý khi hai chủ thể ngang nhau (ưu tiên theo thứ tự nào), và khi nào dùng lớp “khác/không rõ”. Ảnh `traffic` có bus, car và người cùng lúc, nên nếu không có quy tắc này thì mỗi annotator sẽ chọn một kiểu (model chọn “cab” dù ảnh chứa bus).

6. Vì `score` chỉ là độ “tin” của model về dự đoán của chính nó, không do người xác nhận. Model có thể tự tin nhưng sai: ở đây điểm cao nhất chỉ 0.51 và lớp “cab” khác với những gì thực sự thấy trong ảnh (xe buýt, ô tô). Ground truth phải do người gán theo guideline rồi được reviewer kiểm tra; dùng score làm nhãn đúng sẽ lặp lại lỗi của model.


## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

. Record (ảnh `traffic` trong file JSON của tôi): `class_name`=bus ; `score`=0.912558 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96

2. Định dạng `xyxy` là [x_min, y_min, x_max, y_max] tính bằng pixel, gốc toạ độ ở **góc trên-trái ảnh**, x tăng sang phải, y tăng xuống dưới. Góc trên-trái của box ở (93.17; 187.95), góc dưới-phải ở (223.01; 320.91). Rộng = 223.01 − 93.17 = 129.84 px, cao = 320.91 − 187.95 = 132.96 px (khớp với `bbox_width`, `bbox_height`; ảnh rộng 640 × cao 428 px).

3. Hai ngưỡng (ảnh `kitchen`, theo output notebook):
   - Ngưỡng **0.20**: **17** prediction (có cả spoon, potted plant, dining table, bottle).
   - Ngưỡng **0.60**: **6** prediction (person, bowl, bowl, oven, oven, person).
   - Tham khảo ngưỡng 0.35 (mặc định của bài): 11 prediction.

4. Khi hạ ngưỡng thì giữ nhiều hộp hơn, nên reviewer phải **kiểm tra nhiều hộp hơn** (từ 6 lên 17, gấp gần 3 lần): xoá hộp sai/không có vật thật, sửa lại lớp hoặc box lệch, và để ý các vật nhỏ, điểm thấp (spoon, bottle, potted plant) dễ là nhận diện nhầm. Đồng thời phải kiểm tra thêm các vật bị bỏ sót. Khối lượng QC và thời gian tăng lên.

5. **Quy tắc box:** vẽ box chữ nhật chặt sát phần nhìn thấy của vật (không chừa khoảng trống thừa, không cắt vào vật), mỗi vật thể một box riêng, và không vẽ vật thể nhỏ hơn kích thước tối thiểu đã quy định trong guideline (ví dụ < 10 px thì bỏ qua).

6. Ví dụ từ dữ liệu detection có sẵn (ảnh `traffic`): record `person` có `bbox_xyxy` bắt đầu từ x = 0.08, tức người **bị cắt mép trái ảnh** (rộng chỉ 18.32 px, score 0.77); hai box `car` ([268.46, 217.4, 326.52, 268.12] và [300.78, 252.08, 370.32, 320.41]) **chồng lên nhau** vì một xe che xe kia. Với trường hợp **cắt mép** (vật chỉ lộ một phần ở rìa ảnh) hay **che khuất** phổ biến, tôi sẽ viết thành **quy tắc trong guideline** (vẽ box quanh phần nhìn thấy, vẫn gán nhãn nếu thấy đủ để nhận ra) vì nó lặp lại nhiều lần và cần cả team làm giống nhau. Chỉ khi gặp ca không rõ (vật che gần hết, không chắc là vật gì) mới **escalation lên Lab Coach** thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Record (ảnh `traffic` trong file JSON của tôi): `instance_id`=traffic-001 ; `class_name`=bus ; `score`=0.925745 ; `polygon_point_count`=120 ; 5 điểm đầu của `polygon_xy`=[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], [142.0, 192.0]

2. Box chỉ cho biết vị trí và kích thước một hình chữ nhật bao quanh. Polygon (120 điểm) cho biết **đường biên thật** của xe buýt: các đoạn xiên ở góc trên-trái (từ (148, 189) xuống (96, 222)), cạnh trái thẳng đứng, đáy lượn quanh bánh xe. Nhờ đó biết được phần nào trong box là vật, phần nào là nền, tính được diện tích/hình dạng chính xác, và các góc trống của box không bị tính nhầm là xe.

3. `instance_id` (ví dụ `traffic-001`) là **mã định danh của một đối tượng cụ thể trong một ảnh**, để phân biệt từng mask/polygon (bus này khác bus kia, car này khác car kia). Nó **không phải class ID**: class ID (bus = 5) là mã loại, nhiều vật cùng loại cùng class ID nhưng mỗi vật có instance_id riêng. Nó **không phải tracking ID**: tracking ID dùng để nối cùng một vật qua nhiều khung hình video; ở đây chỉ có một ảnh tĩnh, `traffic-001` không cho biết gì về vật đó ở khung hình khác.

4. **Quy tắc vẽ biên mask:** đặt các điểm polygon bám sát đường biên thật của vật, đủ dày ở chỗ cong để mask không bị góc cạnh, không lẫn nền vào mask, không bỏ sót phần vật thật; không để điểm trùng nhau, và mỗi vật một polygon (instance) riêng.

5. Ví dụ từ dữ liệu segmentation có sẵn (ảnh `traffic`): record `traffic-002` (car) có điểm đầu [497, 287] **lặp lại hai lần liên tiếp**, và biên xe là các bậc thang nhỏ (toạ độ số nguyên), chứng tỏ biên không mượt ở chỗ cong. Ở bus (`traffic-001`), cạnh trái chạy thẳng đứng từ (96, 222) tới (96, 312) có thể là biên bị cắt hoặc bị che. Lỗi nhỏ kiểu “điểm trùng, biên bậc thang” thì dùng **guideline** (tạo quy tắc: bỏ điểm trùng, độ chính xác biên bao nhiêu pixel). Còn chỗ **biên mờ, hai vật chạm/che nhau mà không rõ ranh giới** (như hai xe chồng nhau) thì **escalation** hỏi Lab Coach, vì nếu tự quyết thì mỗi annotator sẽ vẽ một kiểu và guideline chưa có câu trả lời.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp cho **cả ảnh** (`class_id`, `class_name`, `taxonomy_name`), không có toạ độ | Ảnh `traffic` có bus, car, người nhưng model xếp hạng 1 là `cab` (score 0.51), top-5 toàn xe khác nhau → khó chọn một lớp khi ảnh nhiều chủ thể | Chọn đúng **một** lớp theo quy tắc chủ thể chính của guideline, trong taxonomy cho sẵn | Nhãn có đúng lớp và đúng taxonomy không; có tuân thủ quy tắc “chủ thể chính” không; không dùng `score` làm đúng/sai |
| Phát hiện vật thể | Một **box** `xyxy` (pixel) + một lớp cho **mỗi vật** | Person bị cắt mép trái ảnh (x = 0.08, rộng 18.32 px); hai car chồng lên nhau; hạ ngưỡng 0.35 → 0.20 ở ảnh kitchen tăng từ 11 lên 17 hộp, thêm nhiều vật nhỏ điểm thấp | Vẽ box chặt quanh phần nhìn thấy, mỗi vật một box, gán đúng lớp; hỏi Lab Coach ca che khuất nặng | Box có sát vật không, có sót/thừa vật không, lớp có đúng không, box của vật bị che/cắt mép có theo quy tắc không |
| Instance segmentation | Một **polygon/mask** + lớp + `instance_id` riêng cho **mỗi đối tượng** | Điểm polygon trùng nhau ở `traffic-002`; biên bậc thang; cạnh trái bus thẳng đứng (biên cắt/che); hai xe chạm nhau → khó chia ranh giới | Vẽ biên bám sát vật, tách từng instance, không để điểm trùng, escalation khi biên mờ/chạm nhau | Mask có bám biên không, có lẫn nền hoặc thiếu phần vật không, các instance có bị gộp/tách sai không, `instance_id` có duy nhất không |

## 5. An toàn dữ liệu

1. Chỉ dùng đúng ảnh COCO được giao cho bài lab, không đưa dữ liệu cá nhân (họ tên, MSSV, email, số điện thoại…) vào báo cáo hay output; họ tên/MSSV chỉ nằm trong tên repository.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (ví dụ ảnh có khuôn mặt/thông tin cá nhân thật, ảnh không thuộc bộ được giao), tôi sẽ **dừng ngay và báo cho Lab Coach (giảng viên/GV hướng dẫn)**, không tự xử lý hay chia sẻ tiếp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
