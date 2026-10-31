#include <cmath>
#include <iomanip>
#include <fstream>
#include <vector>

struct Sample {
    std::vector<double> x; // các đặc trưng đầu vào của một mẫu
    double y;             // giá trị thực tế cần dự đoán

    Sample(int nFeatures) : x(nFeatures), y(0.0) {} // khởi tạo y trước khi Sample được sao chép vào vector
};

double predict(const std::vector<double>& x, const std::vector<double>& weights, double bias) {
    double prediction = bias;
    for (int j = 0; j < static_cast<int>(weights.size()); j++) {
        prediction += weights[j] * x[j];
    }
    return prediction;
}

bool train(const std::vector<Sample>& samples, std::vector<double>& weights, double& bias) { // true khi độ lớn gradient đạt ngưỡng dừng
    const double learningRate = 0.01;
    const double tolerance = 1e-6;
    const int maxIterations = 100000;

    const int nSamples = static_cast<int>(samples.size());
    const int nFeatures = static_cast<int>(weights.size());
    std::vector<double> gradW(nFeatures); // gradient của MSE theo từng trọng số

    for (int iteration = 0; iteration < maxIterations; iteration++) {
        for (int j = 0; j < nFeatures; j++) {
            gradW[j] = 0.0;
        }
        double gradB = 0.0; // gradient của MSE theo bias

        for (int i = 0; i < nSamples; i++) {
            double error = predict(samples[i].x, weights, bias) - samples[i].y;

            for (int j = 0; j < nFeatures; j++) {
                gradW[j] += error * samples[i].x[j];
            }
            gradB += error;
        }

        for (int j = 0; j < nFeatures; j++) {
            gradW[j] *= 2.0 / nSamples; // MSE lấy trung bình bình phương sai số, nên đạo hàm có hệ số 2/nSamples
        }
        gradB *= 2.0 / nSamples;

        double gradNormSq = gradB * gradB;
        for (int j = 0; j < nFeatures; j++) {
            gradNormSq += gradW[j] * gradW[j];
        }
        if (std::sqrt(gradNormSq) <= tolerance) { // kiểm tra bộ tham số hiện tại trước khi cập nhật
            return true;
        }

        for (int j = 0; j < nFeatures; j++) {
            weights[j] -= learningRate * gradW[j];
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
    for (int i = 0; i < nSamples; i++) {
        for (int j = 0; j < nFeatures; j++) {
            input >> samples[i].x[j];
        }
        input >> samples[i].y;
    }

    std::vector<double> weights(nFeatures, 0.0);
    double bias = 0.0;
    if (!train(samples, weights, bias)) {
        output << "training did not converge within the iteration limit\n";
        return 1;
    }

    output << std::setprecision(10);
    output << "y = " << bias;
    for (int j = 0; j < nFeatures; j++) {
        output << (weights[j] < 0 ? " - " : " + ") << std::abs(weights[j]) << " * x" << j + 1;
    }
    output << '\n';

    std::vector<double> x(nFeatures);
    for (int j = 0; j < nFeatures; j++) {
        input >> x[j];
    }

    output << "prediction = " << predict(x, weights, bias) << '\n';
    return 0;
}