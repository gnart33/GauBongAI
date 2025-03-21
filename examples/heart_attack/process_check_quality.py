from gaubongai.analysis.quality import QualityManager
from gaubongai.data_management.processing import DataProcessingManager
from pathlib import Path


def main():
    """Main function."""
    data_processor = DataProcessingManager()
    quality_manager = QualityManager()

    csv_file = Path("examples/heart_attack/heart_attack_dataset.csv")

    # Process with different plugins
    pandas_data = data_processor.process_file(csv_file, plugin_name="pandas_csv")

    report = quality_manager.run_all_checks(pandas_data)
    print(
        f"\nQuality Analysis Report for {csv_file.name}, datatype: {type(pandas_data.data)}"
    )
    print(f"Timestamp: {report.timestamp}")
    print(f"Total checks: {len(report.checks_performed)}\n")
    print("Check Results:")
    for result in report.checks_performed:
        print(f"- {result.check_name}: {'Passed' if result.status else 'Failed'}")
        print(f"  Summary: {result.summary}")
        print()


if __name__ == "__main__":
    main()
