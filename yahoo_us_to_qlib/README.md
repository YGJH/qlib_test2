# Initialize Qlib
        init(provider_uri=str(self.qlib_data_dir))

    def download_data(self):
        collector = YahooCollectorUS1d(
            save_dir=self.source_dir,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            max_workers=4,
            delay=0.5
        )
        collector.collector_data()
        print(f"Data downloaded to {self.source_dir}")

    def normalize_data(self):
        # Assuming you have a normalization class defined in your collector
        normalizer = YahooNormalize1d()  # Replace with the actual normalizer class
        for csv_file in self.source_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            normalized_df = normalizer.normalize(df)
            normalized_df.to_csv(self.normalize_dir / csv_file.name, index=False)
            print(f"Normalized data saved to {self.normalize_dir / csv_file.name}")

    def run(self):
        self.download_data()
        self.normalize_data()

if __name__ == "__main__":
    fire.Fire(FetchAndConvertUSStockData)
```

### Instructions to Use the Script

1. **Save the Script**: Save the above script as `fetch_us_stock_data.py`.

2. **Set Up Directories**: Make sure to create the directories for `source_dir`, `normalize_dir`, and `qlib_data_dir` where you want to store the downloaded and normalized data.

3. **Run the Script**: You can run the script from the command line as follows:
   ```bash
   python fetch_us_stock_data.py --source_dir ~/path/to/source --normalize_dir ~/path/to/normalize --qlib_data_dir ~/path/to/qlib_data --start_date "2023-01-01" --end_date "2023-12-31"
   ```

### Notes
- Ensure that the `YahooNormalize1d` class is correctly defined and imported in your script. You may need to adjust the normalization logic based on your specific requirements.
- The script uses the `fire` library to handle command-line arguments easily. Make sure you have it installed.
- Adjust the parameters like `max_workers` and `delay` as needed based on your network conditions and the volume of data you are fetching.