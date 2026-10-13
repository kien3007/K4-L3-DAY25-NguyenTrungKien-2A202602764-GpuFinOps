# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) | Share of Savings | Impact Description |
|---|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 9.7% | Định tuyến model nhỏ cascade, tận dụng 90% discount prompt cache và 50% discount batch API. |
| Purchasing (spot/reserved) | $10,040 | 80.3% | Chuyển đổi workload sang Spot instance kèm checkpointing và cam kết 3yr Reserved cho dịch vụ 24/7. |
| Right-size util-lies | $655 | 5.2% | Hạ cấp GPU từ H100 sang A100 cho các tác vụ bị nghẽn bộ nhớ thay vì tính toán. |
| Kill idle GPUs | $600 | 4.8% | Thu hồi tự động các GPU nhàn rỗi (utilization < 10% kéo dài) qua scheduler. |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

---

## Phân tích Chuyên sâu: GPU-Util Lie & Cơ chế Thất thoát Chi phí

### 1. Bản chất của "GPU-Util Lie"
Trong kiểm toán Mission 1, telemetry ghi nhận `gpu-h100-4` có chỉ số `GPU-Util = 98.2%` nhưng hiệu năng thực tế `MFU = 19.4%` và `MBU = 20.7%`. Tương tự, `gpu-a10g-1` đạt `GPU-Util = 96.9%` nhưng `MFU = 26.8%`.

**Nguyên nhân kỹ thuật gốc rễ:**
- **Cơ chế đo của nvidia-smi:** Chỉ số `gpu_util_pct` thực chất là *time-active clock* — đo tỷ lệ thời gian có ít nhất một kernel đang chạy trên GPU trong chu kỳ lấy mẫu, hoàn toàn **không phản ánh mức độ bão hòa (occupancy)** của các nhân tính toán Tensor Cores.
- **Memory Stall & Roofline Bottleneck:** Khác với pha prefill (compute-bound, arithmetic intensity ~455 FLOP/byte), pha decode của LLM có arithmetic intensity rất thấp (~1–2 FLOP/byte), nằm sâu dưới ridge point của H100 (~295 FLOP/byte). Quá trình sinh token phải liên tục nạp ma trận trọng số và KV-cache từ bộ nhớ HBM vào SRAM cho từng token đơn lẻ. Khi đó, GPU tiêu tốn 80% thời gian chu kỳ xung nhịp để *chờ dữ liệu từ bộ nhớ* (memory stall) thay vì thực hiện phép nhân ma trận.
- **Kernel Launch Overhead & Tensor Core Starvation:** Khi phục vụ các batch nhỏ với độ trễ thấp, chi phí điều phối kernel và đồng bộ CUDA stream khiến SM luôn ở trạng thái 'bận' trên danh nghĩa nhưng Tensor Cores thực tế không đạt công suất.

**Hệ quả tài chính:**
NimbusAI đang chi trả trọn vẹn đơn giá on-demand $2.50/giờ cho H100 (tương đương công suất đỉnh 990 TFLOPS FP16), nhưng chỉ khai thác được ~192 TFLOPS. Việc nhìn vào GPU-Util 98% khiến đội ngũ kỹ thuật ngộ nhận hệ thống đã hết tải và tiếp tục yêu cầu cấp thêm GPU đắt tiền, gây lãng phí hàng nghìn USD mỗi tháng.

---

## Kế hoạch Hành động Cụ thể theo Thứ tự ROI

Để đạt hiệu quả tài chính tối đa trong thời gian ngắn nhất, các đòn bẩy FinOps được sắp xếp theo thứ tự ROI và độ phức tạp triển khai:

| Thứ tự | Đòn bẩy | Tiết kiệm dự kiến | Độ phức tạp | Lý do ưu tiên & Hành động cụ thể |
|:---:|---|:---:|:---:|---|
| **1** | **Purchasing (Spot + Reserved)** | **$10,040 / tháng** (80.3%) | Thấp | **Quick Win lớn nhất:** Không cần sửa code mô hình. Áp dụng ngay 3-year Reserved cho cụm inference production chạy 24/7 (tiết kiệm ~44%). Chuyển các job training offline sang Spot instance kết hợp lưu checkpoint định kỳ 30 phút. |
| **2** | **Inference Levers (Cascade + Cache + Batch)** | **$1,212 / tháng** (82.6% $/1M-tok) | Trung bình | **Đòn bẩy chiến lược cho quy mô:** Giảm đơn giá phục vụ từ $6.488 xuống $1.126 / 1M-token. Khi lượng request tăng trưởng 10x-50x, đây sẽ là đòn bẩy tiết kiệm hàng chục nghìn USD mỗi tháng. Bật cascade router để đẩy request đơn giản sang model nhỏ, kích hoạt prompt caching cho system prompt dài và gom batch các tác vụ eval/offline. |
| **3** | **Kill Idle GPUs** | **$600 / tháng** (4.8%) | Rất thấp | **Hành động ngay trong ngày:** Thu hồi ngay GPU bỏ trống (như `gpu-h100-5` chạy không 8 tiếng/ngày). Cấu hình Kubernetes cluster autoscaler hoặc cronjob tự động shutdown instance sau 15 phút idle. |
| **4** | **Right-sizing util-lies** | **$655 / tháng** (5.2%) | Trung bình | **Tối ưu hóa kiến trúc hạ tầng:** Hạ cấp `gpu-h100-4` từ H100 ($2.50/h) xuống A100 ($1.79/h) vì workload chỉ cần băng thông bộ nhớ và không tận dụng được FLOPs của H100. Tiết kiệm thêm $511/tháng cho riêng node này mà không ảnh hưởng latency. |

---

## Tính Bền vững & Kinh tế Carbon (Sustainability Economics)

### 1. Phân tích Chi phí Điện và Phát thải theo 5 Vùng
Chuyển đổi 5 job training/batch có tính chất `interruptible` (tiêu thụ 4,227 kWh/tháng) sang các trung tâm dữ liệu xanh mang lại tác động kép về cả ngân sách và trách nhiệm môi trường:

| Region | Lưới điện & Năng lượng | Cường độ carbon (gCO2/kWh) | Giá điện ($/kWh) | Chi phí điện/tháng | Phát thải carbon/tháng | So với us-east-1 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **europe-north1** (Norway) | Thủy điện Bắc Âu | **30** | $0.090 | $380.43 | **126.8 kgCO2e** | **-92.1% CO2** (Sạch nhất) |
| **us-east-wa** (Washington) | Thủy điện sông Columbia | 90 | **$0.055** | **$232.49** | 380.4 kgCO2e | **-54.2% tiền điện** (Rẻ nhất) |
| **us-west-2** (Oregon) | Năng lượng hỗn hợp | 120 | $0.070 | $295.89 | 507.2 kgCO2e | -68.4% CO2, -41.7% $ |
| **us-east-1** (Virginia - Baseline) | Lưới điện hỗn hợp than/khí | 380 | $0.120 | $507.24 | 1,606.3 kgCO2e | Điểm mốc cơ sở |
| **europe-central2** (Poland) | Nhiệt điện than | 660 | $0.180 | $760.86 | 2,789.8 kgCO2e | +73.7% CO2, +50% $ (Ô nhiễm nhất) |

**Khuyến nghị triển khai:**
- **Offline Batch & Training:** Đưa toàn bộ sang `europe-north1`. Tiết kiệm **1,479.5 kg CO2e/tháng** (tương đương giảm phát thải khí nhà kính 1.48 tấn CO2) và giảm $126.81/tháng tiền điện. Độ trễ mạng ~120ms hoàn toàn không ảnh hưởng đến batch training.
- **Online User-facing Serving:** Triển khai tại `us-east-wa` hoặc `us-west-2` để đảm bảo độ trễ phản hồi < 50ms cho người dùng Bắc Mỹ trong khi vẫn hưởng mức giá điện siêu rẻ ($0.055/kWh) và năng lượng thủy điện sạch.

### 2. "Bom Năng lượng" từ Reasoning Traffic (Test-Time Compute)
- Telemetry cho thấy các truy vấn suy luận sâu (`is_reasoning = 1`) chỉ chiếm **8.4% tổng request** (201 / 2,400) nhưng tiêu thụ tới **94.0% tổng năng lượng inference** (29.79 kWh trên tổng số 31.68 kWh).
- Nguyên nhân: Kỹ thuật reasoning dựa trên CoT (Chain of Thought) và search tree bắt GPU sinh hàng nghìn token nội tâm ẩn, tiêu tốn năng lượng gấp ~80 lần một câu truy vấn thông thường.
- Quy tắc quản trị: Cần triển khai semantic router phân loại intent; chỉ kích hoạt reasoning model khi câu hỏi thuộc nhóm lập trình phức tạp hoặc bài toán logic khó (confidence < 0.75), giúp tiết kiệm ngay ~447 kWh điện mỗi tháng.

_Figures are June-2026 as-of snapshots; re-baseline before acting._