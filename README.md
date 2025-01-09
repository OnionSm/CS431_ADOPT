<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>CÁC KĨ THUẬT HỌC SÂU VÀ ỨNG DỤNG</b></h1>


# Giới thiệu
* **Tên môn học:** Các kĩ thuât học sâu và ứng dụng - CS431.P11
* **Năm học:** HK1 (2024 - 2025)
* **Giảng viên**: Nguyễn Vinh Tiệp
* **Sinh viên thực hiện:**
  
  | STT | MSSV     | Họ và Tên        | Email                   |
  |-----|----------|------------------|-------------------------|
  |1    | 22520019 | Nguyễn Ấn | 22520019@gm.uit.edu.vn |
  |2    | 22520083 | Trịnh Thị Lan Anh  | 22520083@gm.uit.edu.vn |
  |3    | 22520363 | Lê Văn Giáp    | 22520363@gm.uit.edu.vn |
  |4    | 22520375 | Vương Dương Thái Hà | 22520375@gm.uit.edu.vn |

# Thông tin đồ án
* **Đề tài:** Nghiên cứu phương pháp tối ưu mới `ADOPT`
* **Giới thiệu chung:**  Trong lĩnh vực học sâu, việc lựa chọn thuật toán tối ưu hóa đóng vai trò quan trọng trong quá trình huấn luyện mô hình. Một trong những thuật toán phổ biến nhất là `Adam` [^1], được biết đến với khả năng điều chỉnh tốc độ học theo từng tham số dựa trên các moment bậc nhất và bậc hai. Tuy nhiên, Adam cũng tồn tại những vấn đề khi không thể đảm bảo hội tụ về mặc lý thuyết chung và phụ thuộc vào việc chọn siêu tham số β2[^2]. `ADOPT` được ra đời dựa trên việc khắc phục các điểm yếu mà vẫn giữ được những ưu điểm của `Adam`[^3].
* **Thông tin chi tiết:** [Report](Report.pdf)
* **Thực nghiệm:** Để chứng minh khả năng của `ADOPT`, chúng tôi tiến hành các thực nghiệm sau:
    * [Toy problem](Experiment/toy-problem.ipynb)
    * [MLP with MNIST](Experiment/mnist_classification.ipynb)
    * [Resnet18 with Cifa-10](Experiment/adopt-adam-cifa10.ipynb)
    * [Resnet50 with IMAGENET](Experiment/adopt-adam-cifa10.ipynb)

## References

[^1]: Diederik P. Kingma and Jimmy Ba, *Adam: A method for stochastic optimization*. arXiv preprint [arXiv:1412.6980](https://arxiv.org/abs/1412.6980), 2014.
[^2]: Dongruo Zhou, Jinghui Chen, Yuan Cao, Yiqi Tang, Ziyan Yang, and Quanquan Gu, *On the convergence of adaptive gradient methods for nonconvex optimization*. arXiv preprint [arXiv:1808.05671](https://arxiv.org/abs/1808.05671), 2018.
[^3]: Zhen Zhang, Min Li, Wei Xu, and Yu Wang, *Adopt: Modified adam can converge with any β2 with the optimal rate*. arXiv preprint [arXiv:2411.02853v3](https://arxiv.org/abs/2411.02853), 2024.
