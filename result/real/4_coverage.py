import os
import csv

def read_interval_data(file_path):
    dates = []
    expected_returns = []
    actual_returns = []
    posterior_intervals = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            date = parts[0]
            expected = float(parts[1])
            actual = float(parts[2])
            lower = float(parts[3])
            upper = float(parts[4])

            dates.append(date)
            expected_returns.append(expected)
            actual_returns.append(actual)
            posterior_intervals.append((lower, upper))

    return {
        "dates": dates,
        "expected_returns": expected_returns,
        "actual_returns": actual_returns,
        "posterior_intervals": posterior_intervals
    }

def calculate_coverage(result):
    actual_returns = result["actual_returns"]
    posterior_intervals = result["posterior_intervals"]
    dates = result["dates"]

    total = 0
    inside = 0
    outliers = []

    for date, actual, (lower, upper) in zip(dates, actual_returns, posterior_intervals):
        if lower <= actual <= upper:
            inside += 1
        else:
            outliers.append(date)
        total += 1

    coverage_probability = inside / total if total > 0 else 0
    return coverage_probability, outliers

def evaluate_all_methods(data_folder, save_path="coverage_results.csv"):
    """
    Evaluate all .dat files in folder and save results to CSV.
    """
    results = []

    print("Coverage Probability Summary:\n")
    for file in os.listdir(data_folder):
        if file.endswith(".dat"):
            method_name = os.path.splitext(file)[0]
            file_path = os.path.join(data_folder, file)

            result = read_interval_data(file_path)
            coverage, outliers = calculate_coverage(result)

            print(f"Method: {method_name}")
            print(f"  Coverage Probability: {coverage:.3f}")
            print(f"  Out-of-interval dates: {outliers}\n")

            results.append({
                "method": method_name,
                "coverage_probability": coverage,
                "outlier_count": len(outliers),
                "outlier_dates": ";".join(outliers)
            })

    # Save to CSV
    with open(save_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["method", "coverage_probability", "outlier_count", "outlier_dates"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Coverage results saved to: {save_path}")

if __name__ == "__main__":
    evaluate_all_methods("4_predict_years") 