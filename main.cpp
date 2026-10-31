#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

struct Sample {
    std::vector<double> x; // các đặc trưng đầu vào của một mẫu
    double y; // giá trị thực tế cần dự đoán

    Sample(int nFeatures) : x(nFeatures), y(0.0) {}
};

double predict(const std::vector<double>& x, const std::vector<double>& weights, double bias) {
    double prediction = bias;
    for (int col = 0; col < static_cast<int>(weights.size()); col++) {
        prediction += weights[col] * x[col];
    }
    return prediction;
}

void standardize(std::vector<Sample>& samples, std::vector<double>& mean, std::vector<double>& stddev) {
    int nSamples = static_cast<int>(samples.size());
    if (nSamples == 0) return;

    int nFeatures = static_cast<int>(samples[0].x.size());

    for (int col = 0; col < nFeatures; col++) { // tính trung bình từng đặc trưng
        for (int row = 0; row < nSamples; row++) {
            mean[col] += samples[row].x[col];
        }
        mean[col] /= nSamples;
    }

    for (int col = 0; col < nFeatures; col++) { // tính độ lệch chuẩn từng đặc trưng
        for (int row = 0; row < nSamples; row++) {
            stddev[col] += (samples[row].x[col] - mean[col]) * (samples[row].x[col] - mean[col]);
        }
        stddev[col] /= nSamples;

        if (stddev[col] != 0.0) {
            stddev[col] = std::sqrt(stddev[col]);
        }
        else {
            stddev[col] = 1.0; // tránh chia cho 0 nếu đặc trưng không đổi
        }
    }

    for (int row = 0; row < nSamples; row++) { // chuẩn hóa các đặc trưng, giữ nguyên y
        for (int col = 0; col < nFeatures; col++) {
            samples[row].x[col] = (samples[row].x[col] - mean[col]) / stddev[col];
        }
    }
}

bool train(const std::vector<Sample>& samples, std::vector<double>& weights, double& bias) {
    const double learningRate = 0.01;
    const double tolerance = 1e-6;
    const int maxIterations = 100000;

    const int nSamples = static_cast<int>(samples.size());
    const int nFeatures = static_cast<int>(weights.size());
    std::vector<double> gradW(nFeatures); // gradient của MSE theo từng trọng số

    for (int iteration = 0; iteration < maxIterations; iteration++) {
        for (int col = 0; col < nFeatures; col++) { // đặt lại gradient trước mỗi vòng lặp
            gradW[col] = 0.0;
        }
        double gradB = 0.0; // gradient của MSE theo bias

        for (int row = 0; row < nSamples; row++) { // cộng dồn gradient từ tất cả các mẫu
            double error = predict(samples[row].x, weights, bias) - samples[row].y;

            for (int col = 0; col < nFeatures; col++) {
                gradW[col] += error * samples[row].x[col];
            }
            gradB += error;
        }

        for (int col = 0; col < nFeatures; col++) {
            gradW[col] *= 2.0 / nSamples; // hệ số đạo hàm của MSE
        }
        gradB *= 2.0 / nSamples;

        double gradNormSq = gradB * gradB; // bình phương độ lớn vector gradient
        for (int col = 0; col < nFeatures; col++) {
            gradNormSq += gradW[col] * gradW[col];
        }

        if (std::sqrt(gradNormSq) <= tolerance) { // dừng khi gradient đủ nhỏ
            return true;
        }

        for (int col = 0; col < nFeatures; col++) { // cập nhật trọng số ngược hướng gradient
            weights[col] -= learningRate * gradW[col];
        }
        bias -= learningRate * gradB;
    }

    return false;
}

int main() {
    std::ifstream input("input.txt"); // đọc dữ liệu huấn luyện và đầu vào cần dự đoán
    std::ofstream output("output.txt"); // tự tạo file nếu chưa có, ghi đè nếu đã có

    int nSamples, nFeatures;
    input >> nSamples >> nFeatures;

    std::vector<Sample> samples(nSamples, Sample(nFeatures));
    for (int row = 0; row < nSamples; row++) { // đọc dữ liệu huấn luyện
        for (int col = 0; col < nFeatures; col++) {
            input >> samples[row].x[col];
        }
        input >> samples[row].y;
    }

    std::vector<double> mean(nFeatures, 0.0);
    std::vector<double> stddev(nFeatures, 0.0);
    standardize(samples, mean, stddev); // chuẩn hóa dữ liệu trước khi huấn luyện

    std::vector<double> weights(nFeatures, 0.0);
    double bias = 0.0;

    if (!train(samples, weights, bias)) {
        output << "training did not converge within the iteration limit\n";
        return 1;
    }

    double originalBias = bias; // chuyển bias về thang đo gốc
    for (int col = 0; col < nFeatures; col++) {
        originalBias -= weights[col] * mean[col] / stddev[col];
    }

    output << std::setprecision(10);
    output << "y = " << originalBias; // in phương trình theo thang đo gốc

    for (int col = 0; col < nFeatures; col++) {
        double originalWeight = weights[col] / stddev[col];
        output << (originalWeight < 0 ? " - " : " + ") << std::abs(originalWeight) << " * x" << col + 1;
    }
    output << '\n';

    std::vector<double> x(nFeatures);
    for (int col = 0; col < nFeatures; col++) { // đọc đầu vào cần dự đoán
        input >> x[col];
    }

    for (int col = 0; col < nFeatures; col++) { // chuẩn hóa bằng mean và stddev đã tính
        x[col] = (x[col] - mean[col]) / stddev[col];
    }

    output << "prediction = " << predict(x, weights, bias) << '\n';
    return 0;
}