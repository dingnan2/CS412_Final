"""
Yelp Dataset Download Script using Kagglehub
"""

import kagglehub
import os
import shutil
from pathlib import Path
import logging

def setup_logging():
    """Setup logging"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def download_yelp_dataset():
    """Download Yelp dataset using Kagglehub"""
    logger = setup_logging()
    
    logger.info("Starting Yelp dataset download...")
    
    try:
        # Download the dataset
        logger.info("Downloading Yelp dataset from Kaggle...")
        path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
        
        logger.info(f"Dataset downloaded to: {path}")
        
        # Check what files were downloaded
        dataset_files = list(Path(path).glob("*"))
        logger.info(f"Downloaded files: {[f.name for f in dataset_files]}")
        
        return path
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise

def copy_files_to_project(dataset_path, target_dir="data/raw"):
    """Copy dataset files to project data directory"""
    logger = setup_logging()
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Copying files to {target_path}...")
    
    # Find JSON files in the dataset
    dataset_path = Path(dataset_path)
    json_files = list(dataset_path.glob("*.json"))
    
    if not json_files:
        # Sometimes files are in subdirectories
        json_files = list(dataset_path.rglob("*.json"))
    
    logger.info(f"Found JSON files: {[f.name for f in json_files]}")
    
    # Copy each JSON file
    copied_files = []
    for json_file in json_files:
        target_file = target_path / json_file.name
        shutil.copy2(json_file, target_file)
        copied_files.append(target_file)
        logger.info(f"Copied: {json_file.name}")
    
    return copied_files

def verify_dataset():
    """Verify that the dataset files are properly downloaded"""
    logger = setup_logging()
    
    data_dir = Path("data/raw")
    expected_files = [
        "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_review.json", 
        "yelp_academic_dataset_user.json",
        "yelp_academic_dataset_checkin.json",
        "yelp_academic_dataset_tip.json"
    ]
    
    logger.info("Verifying dataset files...")
    
    missing_files = []
    existing_files = []
    
    for file_name in expected_files:
        file_path = data_dir / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            logger.info(f"✅ {file_name} ({file_size:.1f} MB)")
            existing_files.append(file_name)
        else:
            logger.warning(f"❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning(f"Missing files: {missing_files}")
        return False
    else:
        logger.info("✅ All dataset files are present!")
        return True

def main():
    """Main function to download and setup Yelp dataset"""
    logger = setup_logging()
    
    print("🎯 CS 412 Research Project - Yelp Dataset Download")
    print("=" * 50)
    
    try:
        # Download dataset
        dataset_path = download_yelp_dataset()
        
        # Copy files to project
        copied_files = copy_files_to_project(dataset_path)
        
        # Verify dataset
        if verify_dataset():
            print("\n🎉 SUCCESS! Yelp dataset is ready!")
            print("\nNext steps:")
            print("1. Run: python src/data_processing/preprocess.py")
            print("2. Run: python run_baseline.py")
            print("3. Start your experiments!")
        else:
            print("\n⚠️  Some files may be missing. Check the logs above.")
            
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Kagglehub is installed: pip install kagglehub")
        print("3. Try running the script again")

if __name__ == "__main__":
    main()
