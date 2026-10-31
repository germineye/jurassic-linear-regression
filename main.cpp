#include <cmath>
#include <iomanip>
#include <fstream>
#include <vector>

struct Sample {
    std::vector<double> x; // các đặc trưng đầu vào của một mẫu
    double y;             // giá trị thực tế cần dự đoán

    Sample(int nFeatures) : x(nFeatures), y(0.0) {} // khởrow tạo y trước khi Sample được sao chép vào vector
};

double predict(const std::vector<double>& x, const std::vector<double>& weights, double bias) {
    double prediction = bias;
    for (int col = 0; col < static_cast<int>(weights.size()); col++) {
        prediction += weights[col] * x[col];
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
        for (int col = 0; col < nFeatures; col++) {
            gradW[col] = 0.0;
        }
        double gradB = 0.0; // gradient của MSE theo bias

        for (int row = 0; row < nSamples; row++) {
            double error = predict(samples[row].x, weights, bias) - samples[row].y;

            for (int col = 0; col < nFeatures; col++) {
                gradW[col] += error * samples[row].x[col];
            }
            gradB += error;
        }

        for (int col = 0; col < nFeatures; col++) {
            gradW[col] *= 2.0 / nSamples; // MSE lấy trung bình bình phương sai số, nên đạo hàm có hệ số 2/nSamples
        }
        gradB *= 2.0 / nSamples;

        double gradNormSq = gradB * gradB;
        for (int col = 0; col < nFeatures; col++) {
            gradNormSq += gradW[col] * gradW[col];
        }
        if (std::sqrt(gradNormSq) <= tolerance) { // kiểm tra bộ tham số hiện tạrow trước khi cập nhật
            return true;
        }

        for (int col = 0; col < nFeatures; col++) {
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
    for (int row = 0; row < nSamples; row++) {
        for (int col = 0; col < nFeatures; col++) {
            input >> samples[row].x[col];
        }
        input >> samples[row].y;
    }

    std::vector<double> weights(nFeatures, 0.0);
    double bias = 0.0;
    if (!train(samples, weights, bias)) {
        output << "training did not converge within the iteration limit\n";
        return 1;
    }

    output << std::setprecision(10);
    output << "y = " << bias;
    for (int col = 0; col < nFeatures; col++) {
        output << (weights[col] < 0 ? " - " : " + ") << std::abs(weights[col]) << " * x" << col + 1;
    }
    output << '\n';

    std::vector<double> x(nFeatures);
    for (int col = 0; col < nFeatures; col++) {
        input >> x[col];
    }

    output << "prediction = " << predict(x, weights, bias) << '\n';
    return 0;
}