"""
Main data processing script
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import with absolute imports
from src.data_processing.data_processor import YelpDataProcessor
from src.utils.config import config


def main():
    """Main function to run data processing"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Yelp data processing...")
    
    # Initialize processor
    processor = YelpDataProcessor()
    
    # Process data (use sample for testing)
    sample_size = 10000  # Set to None for full dataset
    processed_data = processor.process_all_data(sample_size=sample_size)
    
    # Create merged dataset
    merged_df = processor.create_merged_dataset()
    
    logger.info("Data processing completed successfully!")
    print(f"Processed {len(processed_data)} datasets")
    print(f"Merged dataset shape: {merged_df.shape}")


if __name__ == "__main__":
    main()
