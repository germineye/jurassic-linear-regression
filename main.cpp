#include <iostream>
#include <vector>
#include <cmath>

class Point {
public:
	double x, y;
};

std::istream& operator>>(std::istream& in, Point& a) {
	in >> a.x >> a.y;
	return in;
}

int main() {
	int n;
	std::cin >> n;
	std::vector<Point> point(n);
	for (int i = 0; i < n; i++) {
		std::cin >> point[i];
	}

	double weight = 0, bias = 0;
	while (36) {
		double dWeight = 0, dBias = 0;
		for (int i = 0; i < n; i++) {
			double error = weight * point[i].x + bias - point[i].y;

			dWeight += error * point[i].x;
			dBias += error;
		}
		dWeight *= (2.0 / n);
		dBias *= (2.0 / n);

		weight -= 0.01 * dWeight;
		bias -= 0.01 * dBias;

		if (std::hypot(dWeight, dBias) < 1e-6)	break;
	}

	double x;
	std::cin >> x;

	std::cout << "\n\ny = " << weight << "x + " << bias << "\n"
		<< "prediction = " << weight * x + bias << std::endl;
}