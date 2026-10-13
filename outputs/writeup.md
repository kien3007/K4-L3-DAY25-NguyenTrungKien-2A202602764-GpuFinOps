# Báo cáo Phân tích FinOps & Đề xuất Tối ưu Hạ tầng GPU — NimbusAI

**Tác giả:** Kỹ sư FinOps Hạ tầng AI  
**Dự án:** NimbusAI GPU Optimization (Lab 25)  
**Thời gian đánh giá:** Tháng 6/2026  
**Trạng thái hệ thống:** Đã kiểm định tự động (verify.py: 11/11 PASSED, pytest: 19/19 PASSED)

---

## 1. Tổng quan Chi phí: Baseline vs. Optimized

Khi tiếp nhận dữ liệu hạ tầng GPU tại NimbusAI, hóa đơn hàng tháng đang ở mức **$27,133/tháng** do thói quen cấu hình on-demand toàn bộ, chọn GPU kích cỡ lớn nhất theo cảm tính và không có cơ chế thu hồi tài nguyên nhàn rỗi. 

Sau khi áp dụng đồng bộ 4 đòn bẩy FinOps (Inference Levers, Purchasing Strategy, Right-sizing và Kill Idle):
- **Chi phí vận hành hàng tháng:** Giảm từ **$27,133/tháng** xuống còn **$14,626/tháng**.
- **Số tiền tiết kiệm thực tế:** **$12,507/tháng** (tương đương cắt giảm **46.1%** tổng hóa đơn).
- **Đơn giá kinh tế trên sản lượng token ($/1M-token):** Giảm từ **$6.488 / 1M-token** (baseline không cache, không batch, dồn hết vào model lớn) xuống còn **$1.126 / 1M-token** ở trạng thái tối ưu (giảm **82.6%**).

Con số này chứng minh sự thay đổi cốt lõi trong tư duy FinOps: thay vì chỉ nhìn vào đơn giá thuê máy tính theo giờ (`$/GPU-hr`), chúng ta đo lường chính xác chi phí trên một đơn vị giá trị tạo ra (`$/1M-token`).

```
+-------------------------------------------------------------------------+
| NimbusAI GPU Spend: $27,133/mo  --->  $14,626/mo  (Tiết kiệm 46.1%)     |
| Token Unit Economics: $6.488/1M --->  $1.126/1M   (Cắt giảm 82.6%)      |
+-------------------------------------------------------------------------+
```

---

## 2. Phân tích Đóng góp của Từng Đòn bẩy (Levers Breakdown)

| Đòn bẩy tối ưu | Tiết kiệm / tháng | Tỷ trọng tiết kiệm | Bản chất kỹ thuật & Nguyên nhân đóng góp |
|---|:---:|:---:|---|
| **Purchasing (Spot + Reserved)** | **$10,040** | **80.3%** | **Đóng góp số tiền lớn nhất:** Hơn 90% ngân sách hạ tầng nằm ở các cụm training và job chạy nền tảng. Việc chuyển 5 job training/batch sang Spot kết hợp checkpoint định kỳ và cam kết 3-year Reserved cho các cụm inference chạy 24/7 mang lại hiệu quả tức thì mà không cần can thiệp logic mã nguồn mô hình. |
| **Inference (Cascade + Cache + Batch)** | **$1,212** | **9.7%** | **Đòn bẩy có tỷ lệ giảm sâu nhất (82.6%):** Hiện tại lưu lượng token chỉ ở mức ~7.5 triệu token/ngày. Nhưng khi lượng người dùng tăng trưởng 10x đến 100x, đây sẽ là đòn bẩy cứu cánh giúp biên lợi nhuận công ty không bị bào mòn. Cascade định tuyến 70% request dễ sang model nhỏ rẻ hơn 15x, prompt cache giảm 90% chi phí input lặp lại, và batch API giảm 50% chi phí các tác vụ chạy đêm. |
| **Right-size "Util-Lie" Nodes** | **$655** | **5.2%** | Hạ cấp node `gpu-h100-4` từ H100 ($2.50/h) xuống A100 ($1.79/h). Workload trên node này thực chất chỉ cần băng thông HBM, hoàn toàn lãng phí năng lực tính toán FLOPs của H100. |
| **Kill Idle GPUs** | **$600** | **4.8%** | Loại bỏ lãng phí từ node `gpu-h100-5` (bỏ không 8 giờ mỗi ngày sau khi train xong nhưng dev quên tắt instance). Thu hồi ngay $20/ngày ($600/tháng) với rủi ro vận hành bằng 0. |

---

## 3. Bản chất của "GPU-Util Lie" và Tác động Tài chính

Trong báo cáo giám sát telemetry, hai GPU bị gắn cờ "nói dối" (GPU-Util Lie) gồm:
1. **`gpu-h100-4` (H100):** `GPU-Util = 98.2%`, nhưng `MFU = 19.4%` và `MBU = 20.7%`.
2. **`gpu-a10g-1` (A10G):** `GPU-Util = 96.9%`, nhưng `MFU = 26.8%` và `MBU = 30.2%`.

### Cơ chế kỹ thuật gốc rễ:
- **Hạn chế cố hữu của nvidia-smi:** Chỉ số `gpu_util_pct` chỉ là một chiếc đồng hồ bấm giờ (time-active clock). Nó ghi nhận tỷ lệ thời gian có *bất kỳ kernel nào* đang chiếm dụng Streaming Multiprocessor (SM). Nó **không đo lường** số lượng nhân Tensor Core đang thực sự tính toán hay đang đứng chờ.
- **Hiện tượng Memory Stall & Roofline Bottleneck:** Theo mô hình Roofline, pha sinh từ (decode) của LLM có cường độ tính toán cực kỳ thấp (Arithmetic Intensity chỉ đạt ~1–2 FLOP/byte), nằm sâu phía dưới điểm gãy (ridge point ~295 FLOP/byte của H100). Trong mỗi bước sinh token, GPU phải nạp toàn bộ trọng số mô hình và KV-cache từ bộ nhớ HBM ngoài vào SRAM trên chip. Hơn 80% thời gian xung nhịp, các nhân Tensor Cores bị bỏ đói (starvation) và phải dừng chờ dữ liệu di chuyển qua bus bộ nhớ (memory stall).
- **Overhead khởi chạy Kernel và Batch size nhỏ:** Việc phục vụ các request lẻ tẻ với batch size nhỏ khiến thời gian điều phối kernel và đồng bộ luồng CUDA chiếm ưu thế, giữ cho clock GPU luôn ở mức 98% nhưng sản lượng tính toán thực sự rất nghèo nàn.

### Tác động tài chính:
NimbusAI đang trả đủ $2.50/giờ cho H100 để mua 990 TFLOPS năng lực tính toán đỉnh, nhưng thực tế chỉ thu về ~192 TFLOPS. Khi thấy GPU-Util 98%, đội ngũ kỹ thuật trước đây đã kết luận sai lầm rằng hệ thống "quá tải" và tiếp tục xin cấp thêm hạn mức thuê H100 mới, khiến ngân sách bị rò rỉ nghiêm trọng.

---

## 4. Báo cáo Chi tiết 5 Phần Mở Rộng "Your Turn"

Chúng tôi đã triển khai, đo lường thực nghiệm và viết bộ kiểm thử tự động cho toàn bộ 5 phần mở rộng:

### Extension 1 — Tối ưu hóa `recommend_tier()` theo Ma trận Rủi ro & Kỳ hạn
- **Vị trí triển khai:** `finops/pricing.py` + `missions/m3_purchasing.py` + test `test_recommend_tier_advanced`.
- **Cải tiến:** 
  - Đưa tỷ lệ gián đoạn thực tế (interruption rate) theo từng dòng chip vào mô hình (H100: 4%, A100: 6%, A10G: 9%, L4: 8%).
  - So sánh chi tiết giữa cam kết 1 năm (chiết khấu ~20%, điểm hòa vốn duty cycle ~80%) và cam kết 3 năm (chiết khấu ~45%, điểm hòa vốn ~55%).
  - Đối với các job chạy không liên tục 24/7 (như `job-infer-search` chạy 18h/ngày, duty cycle 75%), chính sách nâng cao thận trọng đề xuất cam kết 1 năm thay vì khóa hợp đồng 3 năm, tránh rủi ro "idle lock-in" nếu lưu lượng tìm kiếm thay đổi trong tương lai.
- **Kết quả đo lường:** 
  - Chính sách cơ bản: Tiết kiệm 39.1% ($10,040/tháng).
  - Chính sách nâng cao (thận trọng): Tiết kiệm 38.0% ($9,745/tháng).
- **Insight:** Việc giảm nhẹ 1.1% mức tiết kiệm trên giấy tờ đổi lại sự an toàn thanh khoản và linh hoạt kiến trúc cho startup, tránh bị ràng buộc công nợ 3 năm trên những dòng chip tầm trung như L4.

### Extension 2 — Right-sizing theo Băng thông Bộ nhớ (MBU)
- **Vị trí triển khai:** `missions/m1_efficiency_audit.py` + `finops/metrics.py` + test `test_mbu_rightsizing_logic`.
- **Cải tiến:** Tính toán đơn giá VRAM (`$/GB-VRAM/h`) và băng thông (`$/(TB/s)/h`) cho toàn bộ danh mục GPU. Với các GPU memory-bound (MBU < 0.35), tự động tìm GPU thay thế có đủ băng thông chịu tải nhưng chi phí rẻ hơn.
- **Kết quả đo lường:**
  - `gpu-h100-4` (H100, MBU 0.207, đạt 0.69 TB/s) -> Đổi sang A100 (băng thông đỉnh 2.0 TB/s, VRAM 80GB): Tiết kiệm $0.71/giờ = **$511/tháng** (giảm 28.4%).
  - `gpu-a10g-0` (A10G, MBU 0.235, đạt 0.14 TB/s) -> Đổi sang L4 (băng thông đỉnh 0.30 TB/s, VRAM 24GB): Tiết kiệm $0.20/giờ = **$144/tháng** (giảm 20.0%).
  - Tổng tiềm năng tiết kiệm từ MBU right-sizing: **$1,166/tháng**.
- **Insight trả lời câu hỏi cốt lõi:** *"Tại sao không chỉ chọn GPU rẻ nhất theo $/GPU-hr (như L4 ở mức $0.80/h)?"*
  Bởi vì workload LLM decode bị nghẽn trực tiếp bởi dung lượng VRAM (chứa KV-cache) và băng thông HBM. Nếu ép một tác vụ đang cần 0.69 TB/s chạy trên L4 (chỉ có 0.3 TB/s), tốc độ sinh token sẽ tụt dốc thảm hại, vi phạm SLA độ trễ nghiêm trọng; chưa kể nếu kích thước KV-cache vượt quá 24GB của L4 sẽ dẫn tới lỗi sập bộ nhớ (OOM) tức thì!

### Extension 3 — Cơ chế Gating `cache_is_worth_it()`
- **Vị trí triển khai:** `finops/pricing.py` + `missions/m2_inference_levers.py` + test `test_cache_is_worth_it_logic`.
- **Cải tiến:** Thiết lập ngưỡng hòa vốn toán học cho việc lưu cache:  
  `N_breakeven = WriteCost / (BaseReadCost * (1 - ReadDiscount))`
- **Kết quả đo lường:**
  - Với chiết khấu đọc 90% (read discount 0.10), ngưỡng hòa vốn là **1.11 lần đọc lại**. Tức chỉ cần prefix được tái sử dụng từ lần thứ 2 trở lên là hệ thống đã sinh lời tài chính ròng.
  - Phân tích nhật ký `token_usage.csv` cho thấy tần suất tái sử dụng trung bình của các prompt hệ thống và context RAG đạt **3.8 lần/prefix**.
  - Hàm `cache_is_worth_it()` trả về `True`, cho phép hệ thống áp dụng đầy đủ đòn bẩy tiết kiệm từ prompt cache một cách an toàn.

### Extension 4 — Quản trị Ngân sách Reasoning (Cost & Energy)
- **Vị trí triển khai:** `missions/m2_inference_levers.py` + test suite.
- **Số liệu thực nghiệm gây sốc:**
  - Lưu lượng Reasoning (`is_reasoning = 1`): Chỉ chiếm **8.4% tổng số request** (201 / 2,400) và 16.5% tổng số token.
  - Tuy nhiên, nhóm này tiêu thụ tới **94.0% tổng điện năng inference** (29.79 kWh trên 31.68 kWh tổng cộng).
- **Lý giải nguyên nhân:** Mô hình suy luận sâu sử dụng cơ chế test-time compute (sinh chuỗi Chain-of-Thought dài và lặp lại nhiều nhánh tìm kiếm), tiêu tốn năng lượng gấp ~80 lần một câu hỏi thông thường.
- **Quy tắc điều phối (Routing Rule) đề xuất:** Tích hợp bộ phân loại độ phức tạp tác vụ ở tầng gateway. Chỉ cho phép kích hoạt reasoning model khi câu hỏi thuộc nhóm coding chuyên sâu hoặc toán học phức tạp (complexity score > 0.75). Nếu cắt giảm 50% các cuộc gọi reasoning không cần thiết, NimbusAI sẽ tiết kiệm ngay **$20.9/ngày ($627/tháng)** tiền token và **446.8 kWh điện/tháng**.

### Extension 5 — Điều phối Thông minh Hướng Carbon (Carbon-aware Scheduling)
- **Vị trí triển khai:** `missions/m3_purchasing.py` + `finops/sustainability.py` + test `test_carbon_aware_savings`.
- **Phạm vi phân tích:** 5 job training/batch có tính chất `interruptible = 1`, tiêu thụ **4,227.0 kWh điện mỗi tháng**.
- **Bảng so sánh 5 vùng đám mây:**

| Vùng triển khai (Region) | Cường độ carbon (gCO2/kWh) | Giá điện ($/kWh) | Tiền điện / tháng | Phát thải carbon / tháng | Đánh giá & Khuyến nghị |
|---|:---:|:---:|:---:|:---:|---|
| **europe-north1** (Na Uy) | **30** | $0.090 | $380.43 | **126.8 kgCO2e** | **Sạch nhất:** Giảm **92.1% lượng phát thải carbon** (tiết kiệm 1.48 tấn CO2e/tháng). Phù hợp nhất cho offline batch training. |
| **us-east-wa** (Washington) | 90 | **$0.055** | **$232.49** | 380.4 kgCO2e | **Rẻ nhất:** Tiết kiệm **54.2% tiền điện** ($274.75/tháng), dùng thủy điện sạch, độ trễ tốt cho người dùng Mỹ. |
| **us-west-2** (Oregon) | 120 | $0.070 | $295.89 | 507.2 kgCO2e | Cân bằng tốt, năng lượng tái tạo ổn định. |
| **us-east-1** (Virginia) | 380 | $0.120 | $507.24 | 1,606.3 kgCO2e | Vùng mặc định hiện tại, lưới điện dùng nhiều nhiệt điện than. |
| **europe-central2** (Ba Lan) | 660 | $0.180 | $760.86 | 2,789.8 kgCO2e | Tệ nhất cả về tiền điện (+50%) lẫn mức độ ô nhiễm (+73.7%). Tránh xa tuyệt đối. |

- **Đánh giá Trade-off về Độ trễ (Latency):**
  Vùng sạch nhất (`europe-north1`) có khoảng cách địa lý xa, độ trễ mạng (RTT) sang Mỹ lên tới 110–140ms. Do đó:
  - **Không bao giờ đặt User-facing Inference tại europe-north1** vì sẽ làm chậm trải nghiệm người dùng.
  - **Đưa 100% batch training sang europe-north1**, bởi các job huấn luyện chạy nền qua đêm hoàn toàn không bị ảnh hưởng bởi độ trễ mạng RTT, giúp công ty đạt chứng chỉ xanh ESG với chi phí điện hợp lý.

---

## 5. Khuyến nghị Chiến lược cho FinOps Lead NimbusAI

Nếu được trao quyền FinOps Lead tại NimbusAI, đây là 3 hành động ưu tiên cao nhất tôi sẽ thực thi ngay trong tuần đầu tiên:

1. **Khóa ngay cam kết Reserved và cấu hình Spot Orchestration (Tác động tài chính: +$10,040/tháng):**
   - Làm việc với nhà cung cấp đám mây để chuyển cụm inference 24/7 (`job-infer-chat` và `job-infer-rag`) sang hợp đồng 3-year Reserved (chiết khấu ~44%).
   - Cấu hình Ray Train / Slurm cho 5 job training offline chạy trên Spot instance với chu kỳ checkpoint 30 phút một lần lên cloud storage.
2. **Triển khai Auto-termination và Right-sizing hạ tầng (Tác động tài chính: +$1,111/tháng):**
   - Thiết lập Kubernetes Cluster Autoscaler và cron script quét telemetry: tự động dừng bất kỳ node GPU nào có utilization < 10% kéo dài quá 15 phút (triệt tiêu ngay $600/tháng lãng phí idle).
   - Hạ cấp node `gpu-h100-4` từ H100 sang A100 để tiết kiệm thêm $511/tháng mà không làm suy giảm hiệu năng.
3. **Nâng cấp Gateway Inference và Thiết lập Kiểm soát Phân bổ Chi phí (Tác động tài chính: Giảm 82.6% đơn giá token, mở cổng Chargeback):**
   - Kích hoạt tính năng Prompt Caching trên toàn bộ endpoint phục vụ Chat và RAG để tận dụng mức giảm giá 90% cho system prompts.
   - Bật cascade router đẩy các tác vụ dịch thuật/tóm tắt đơn giản sang model nhỏ ($0.20/1M-token thay vì $3.00/1M-token).
   - Đặt hạn mức kiểm soát (budget alert) cho Reasoning queries qua semantic intent classifier.
   - Với Tag Coverage hiện đã đạt **92%** (vượt ngưỡng 80%), chính thức kích hoạt cổng **Chargeback** theo chuẩn FOCUS để phân bổ chi phí minh bạch về từng phòng ban (Assistant, Search, Eval, RAG), gắn trách nhiệm tài chính trực tiếp với các kỹ sư AI.
