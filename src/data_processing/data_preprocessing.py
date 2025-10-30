"""
Data Preprocessing: Clean individual Yelp datasets and write processed CSVs.
"""

import pandas as pd
import numpy as np
import json
import gc
import logging
from pathlib import Path
from typing import Optional, Dict
import os
import warnings

warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean individual Yelp datasets
    Goal: Remove noise, handle missing values, remove duplicates
    """

    def __init__(self, raw_path: str = "data/raw", output_path: str = "data/processed"):
        self.raw_path = Path(raw_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cleaning_summary: Dict[str, Dict[str, int]] = {}

    def load_json_data(self, filename: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load JSON file line by line (memory efficient)"""
        filepath = self.raw_path / filename
        logger.info(f"Loading: {filename}")

        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logger.warning(f"Skipped malformed JSON at line {i+1}")

        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df):,} records from {filename}")
        return df

    def clean_business_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Clean business data:
        Requirements:
        1. Remove: postal_code, longitude, latitude, hours
        2. Handle missing values
        3. Remove duplicates
        4. Keep: business_id, name, address, city, state, stars,
                 review_count, is_open, attributes, categories
        """
        logger.info("="*70)
        logger.info("CLEANING BUSINESS DATA")
        logger.info("="*70)

        # Load
        df = self.load_json_data("yelp_academic_dataset_business.json", sample_size)
        logger.info(f"Original shape: {df.shape}")
        original_cols = list(df.columns)
        logger.info(f"Original columns: {original_cols}")
        summary = {"input_rows": len(df)}

        # Step 1: Remove specified columns
        cols_to_remove = ['postal_code', 'longitude', 'latitude', 'hours']
        existing_to_remove = [col for col in cols_to_remove if col in df.columns]
        df = df.drop(columns=existing_to_remove)
        logger.info(f"Removed columns: {existing_to_remove}")

        # Step 2: Data type conversions
        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
        df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
        df['is_open'] = df['is_open'].astype(int)

        # Step 3: Handle missing values
        logger.info(f"\nMissing values BEFORE cleaning:")
        missing_before = df.isnull().sum()
        logger.info(f"\n{missing_before[missing_before > 0]}")

        # Numeric columns - fill with median
        stars_missing = int(df['stars'].isna().sum())
        review_count_missing = int(df['review_count'].isna().sum())
        df['stars'].fillna(df['stars'].median(), inplace=True)
        df['review_count'].fillna(df['review_count'].median(), inplace=True)

        # Categorical columns - fill appropriately
        df['name'].fillna('Unknown', inplace=True)
        df['address'].fillna('', inplace=True)
        df['city'].fillna('Unknown', inplace=True)
        df['state'].fillna('Unknown', inplace=True)
        df['categories'].fillna('', inplace=True)
        df['attributes'].fillna('{}', inplace=True)

        logger.info(f"\nMissing values AFTER cleaning:")
        missing_after = df.isnull().sum()
        logger.info(f"\n{missing_after[missing_after > 0]}")

        # Step 4: Remove duplicates
        before_dup = len(df)
        df = df.drop_duplicates(subset=['business_id'], keep='first')
        after_dup = len(df)
        removed_dups = before_dup - after_dup
        logger.info(f"\nRemoved {removed_dups:,} duplicate businesses")

        # Step 4.5: Handle outliers using IQR method
        logger.info(f"\nHandling outliers using IQR method:")
        outlier_columns = ['stars', 'review_count']
        total_outliers_handled = 0

        for col in outlier_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"  {col}: Clipped {outliers_count:,} outliers (range: [{lower_bound:.2f}, {upper_bound:.2f}])")
                    total_outliers_handled += outliers_count

        # Step 5: Data quality check
        logger.info(f"\nData Quality Summary:")
        logger.info(f"  Final shape: {df.shape}")
        logger.info(f"  Target distribution (is_open):")
        target_counts = df['is_open'].value_counts()
        logger.info(f"    Open (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.2f}%)")
        logger.info(f"    Closed (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.2f}%)")

        # Save
        output_file = self.output_path / "business_clean.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nSaved: {output_file}")

        # Update and write summary
        summary.update({
            "final_rows": int(len(df)),
            "duplicates_removed": int(removed_dups),
            "stars_filled": int(stars_missing),
            "review_count_filled": int(review_count_missing),
            "columns_removed": int(len(existing_to_remove)),
            "outliers_handled": int(total_outliers_handled),
        })
        self.cleaning_summary["business"] = summary
        try:
            with open(self.output_path / "cleaning_summary.json", 'w', encoding='utf-8') as f:
                json.dump(self.cleaning_summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write cleaning summary: {e}")

        return df

    def clean_review_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Clean review data:
        Requirements:
        1. Combine funny + cool → total_count (similar metrics)
        2. Handle missing values
        3. Remove duplicates
        4. Keep: review_id, user_id, business_id, stars, text, date, useful, total_count
        """
        logger.info("="*70)
        logger.info("CLEANING REVIEW DATA")
        logger.info("="*70)

        filepath = self.raw_path / "yelp_academic_dataset_review.json"
        output_file = self.output_path / "review_clean.csv"

        if sample_size is not None:
            # Small sample path (in-memory)
            df = self.load_json_data("yelp_academic_dataset_review.json", sample_size)
            logger.info(f"Original shape: {df.shape}")
            logger.info(f"Original columns: {list(df.columns)}")

            # Combine funny + cool → funny_cool
            df['funny_cool'] = 0
            if 'funny' in df.columns:
                df['funny_cool'] = df['funny_cool'] + df['funny'].fillna(0)
            if 'cool' in df.columns:
                df['funny_cool'] = df['funny_cool'] + df['cool'].fillna(0)
            logger.info(f"\nCreated 'funny_cool' = funny + cool")

            cols_to_drop = [col for col in ['funny', 'cool'] if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # Type conversions
            df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
            df['useful'] = pd.to_numeric(df['useful'], errors='coerce')
            df['funny_cool'] = pd.to_numeric(df['funny_cool'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Missing values
            logger.info(f"\nMissing values BEFORE cleaning:")
            missing_before = df.isnull().sum()
            logger.info(f"\n{missing_before[missing_before > 0]}")
            df['stars'].fillna(df['stars'].median(), inplace=True)
            df['useful'].fillna(0, inplace=True)
            df['funny_cool'].fillna(0, inplace=True)
            df['text'].fillna('', inplace=True)

            # Calculate text length (important feature)
            if 'text' in df.columns:
                df['text_length'] = df['text'].str.len()
                logger.info(f"\nCalculated text_length (mean: {df['text_length'].mean():.0f} chars)")

            dropped_na_date = int(df['date'].isna().sum())
            df = df.dropna(subset=['date'])
            logger.info(f"\nMissing values AFTER cleaning:")
            missing_after = df.isnull().sum()
            logger.info(f"\n{missing_after[missing_after > 0]}")

            # Duplicates
            before_dup = len(df)
            df = df.drop_duplicates(subset=['review_id'], keep='first')
            after_dup = len(df)
            removed_dups = before_dup - after_dup
            logger.info(f"\nRemoved {removed_dups:,} duplicate reviews")

            # Handle outliers using IQR method
            logger.info(f"\nHandling outliers using IQR method:")
            outlier_columns = ['useful', 'funny_cool']
            total_outliers = 0

            for col in outlier_columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers_count > 0:
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        logger.info(f"  {col}: Clipped {outliers_count:,} outliers")
                        total_outliers += outliers_count

            # Save
            df.to_csv(output_file, index=False)
            logger.info(f"\nSaved: {output_file}")
            # Summary
            self.cleaning_summary["review"] = {
                "input_rows": int(len(df) + removed_dups + dropped_na_date),
                "final_rows": int(len(df)),
                "duplicates_removed": int(removed_dups),
                "rows_dropped_invalid_date": int(dropped_na_date),
            }
            try:
                with open(self.output_path / "cleaning_summary.json", 'w', encoding='utf-8') as f:
                    json.dump(self.cleaning_summary, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not write cleaning summary: {e}")
            return df

        # Full dataset path (chunked, low-memory)
        logger.info(f"Streaming and cleaning in chunks: {filepath}")
        if output_file.exists():
            output_file.unlink()

        # Use smaller chunk size and manual line reading to avoid memory issues
        chunksize = 100_000  # Reduced from 200k
        total_rows = 0
        total_written = 0
        total_na_date = 0
        total_dups_removed = 0
        header_written = False

        # Manual chunked reading to avoid pandas memory issues
        chunk_data = []
        line_count = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk_data.append(json.loads(line.strip()))
                    line_count += 1

                    if line_count >= chunksize:
                        # Process chunk
                        df = pd.DataFrame(chunk_data)

                        # funny_cool
                        df['funny_cool'] = 0
                        if 'funny' in df.columns:
                            df['funny_cool'] = df['funny_cool'] + df['funny'].fillna(0)
                        if 'cool' in df.columns:
                            df['funny_cool'] = df['funny_cool'] + df['cool'].fillna(0)
                        drop_cols = [c for c in ['funny', 'cool'] if c in df.columns]
                        if drop_cols:
                            df.drop(columns=drop_cols, inplace=True)
                        # types
                        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
                        df['useful'] = pd.to_numeric(df['useful'], errors='coerce')
                        df['funny_cool'] = pd.to_numeric(df['funny_cool'], errors='coerce')
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        # fills
                        df['stars'].fillna(df['stars'].median(), inplace=True)
                        df['useful'].fillna(0, inplace=True)
                        df['funny_cool'].fillna(0, inplace=True)
                        df['text'] = df.get('text', '').fillna('') if 'text' in df.columns else ''
                        # Calculate text length
                        if 'text' in df.columns:
                            df['text_length'] = df['text'].str.len()
                        na_date = int(df['date'].isna().sum())
                        df = df.dropna(subset=['date'])
                        # dup within chunk
                        before = len(df)
                        df = df.drop_duplicates(subset=['review_id'], keep='first')
                        dups_removed = before - len(df)
                        # Handle outliers (IQR method)
                        for col in ['useful', 'funny_cool']:
                            if col in df.columns and len(df) > 0:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                if IQR > 0:
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR
                                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        # Save append
                        df.to_csv(output_file, index=False, mode='a', header=not header_written)
                        header_written = True
                        total_na_date += na_date
                        total_dups_removed += dups_removed
                        total_rows += line_count
                        total_written += len(df)
                        logger.info(f"  Wrote chunk: {len(df):,} rows (written total: {total_written:,})")

                        # Clear memory
                        chunk_data = []
                        line_count = 0
                        del df
                        gc.collect()

                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

        # Process final chunk if any remaining data
        if chunk_data:
            df = pd.DataFrame(chunk_data)
            # funny_cool
            df['funny_cool'] = 0
            if 'funny' in df.columns:
                df['funny_cool'] = df['funny_cool'] + df['funny'].fillna(0)
            if 'cool' in df.columns:
                df['funny_cool'] = df['funny_cool'] + df['cool'].fillna(0)
            drop_cols = [c for c in ['funny', 'cool'] if c in df.columns]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
            # types
            df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
            df['useful'] = pd.to_numeric(df['useful'], errors='coerce')
            df['funny_cool'] = pd.to_numeric(df['funny_cool'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # fills
            df['stars'].fillna(df['stars'].median(), inplace=True)
            df['useful'].fillna(0, inplace=True)
            df['funny_cool'].fillna(0, inplace=True)
            df['text'] = df.get('text', '').fillna('') if 'text' in df.columns else ''
            # Calculate text length
            if 'text' in df.columns:
                df['text_length'] = df['text'].str.len()
            na_date = int(df['date'].isna().sum())
            df = df.dropna(subset=['date'])
            # dup within chunk
            before = len(df)
            df = df.drop_duplicates(subset=['review_id'], keep='first')
            dups_removed = before - len(df)
            # Handle outliers (IQR method)
            for col in ['useful', 'funny_cool']:
                if col in df.columns and len(df) > 0:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            # Save append
            df.to_csv(output_file, index=False, mode='a', header=not header_written)
            header_written = True
            total_na_date += na_date
            total_dups_removed += dups_removed
            total_rows += len(chunk_data)  # Use chunk_data length
            total_written += len(df)
            logger.info(f"  Wrote final chunk: {len(df):,} rows (written total: {total_written:,})")
            del df
            gc.collect()
        logger.info(f"\nSaved: {output_file} (total rows: {total_rows:,})")
        # Save summary
        self.cleaning_summary["review"] = {
            "input_rows": int(total_rows),
            "final_rows": int(total_written),
            "duplicates_removed": int(total_dups_removed),
            "rows_dropped_invalid_date": int(total_na_date),
        }
        try:
            with open(self.output_path / "cleaning_summary.json", 'w', encoding='utf-8') as f:
                json.dump(self.cleaning_summary, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write cleaning summary: {e}")
        # Return only schema (empty DF) to keep type contract
        return pd.read_csv(output_file, nrows=0)

    def clean_user_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Clean user data:
        Requirements:
        1. Handle missing values
        2. Remove duplicates
        3. Calculate user tenure for weighting
        4. Keep all fields
        """
        logger.info("="*70)
        logger.info("CLEANING USER DATA")
        logger.info("="*70)

        filepath = self.raw_path / "yelp_academic_dataset_user.json"
        if sample_size is not None:
            df = self.load_json_data("yelp_academic_dataset_user.json", sample_size)
            logger.info(f"Original shape: {df.shape}")
            logger.info(f"Original columns: {list(df.columns)}")
        else:
            logger.info(f"Streaming and cleaning in chunks: {filepath}")
            output_file = self.output_path / "user_clean.csv"
            if output_file.exists():
                output_file.unlink()

            # Use manual line-by-line reading to avoid memory issues
            chunksize = 100_000  # Reduced chunk size
            total_rows = 0
            total_written = 0
            total_dups_removed = 0
            header_written = False

            chunk_data = []
            line_count = 0

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk_data.append(json.loads(line.strip()))
                        line_count += 1

                        if line_count >= chunksize:
                            # Process chunk
                            df_chunk = pd.DataFrame(chunk_data)
                            # numeric conversions
                            numeric_cols = ['review_count', 'useful', 'funny', 'cool', 'fans',
                                           'average_stars', 'compliment_hot', 'compliment_more',
                                           'compliment_profile', 'compliment_cute', 'compliment_list',
                                           'compliment_note', 'compliment_plain', 'compliment_cool',
                                           'compliment_funny', 'compliment_writer', 'compliment_photos']
                            for col in numeric_cols:
                                if col in df_chunk.columns:
                                    df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
                            df_chunk['yelping_since'] = pd.to_datetime(df_chunk['yelping_since'], errors='coerce')
                            # funny_cool
                            has_funny = 'funny' in df_chunk.columns
                            has_cool = 'cool' in df_chunk.columns
                            if has_funny or has_cool:
                                df_chunk['funny_cool'] = (df_chunk['funny'].fillna(0) if has_funny else 0) + (df_chunk['cool'].fillna(0) if has_cool else 0)
                            # tenure
                            reference_date = pd.Timestamp('2025-01-31')
                            df_chunk['user_tenure_days'] = (reference_date - df_chunk['yelping_since']).dt.days
                            df_chunk['user_tenure_years'] = df_chunk['user_tenure_days'] / 365.25
                            # fills
                            for col in numeric_cols:
                                if col in df_chunk.columns:
                                    df_chunk[col].fillna(0, inplace=True)
                            if 'name' in df_chunk.columns:
                                df_chunk['name'].fillna('Anonymous', inplace=True)
                            if 'elite' in df_chunk.columns:
                                df_chunk['elite'].fillna('', inplace=True)
                            if 'friends' in df_chunk.columns:
                                df_chunk['friends'].fillna('', inplace=True)
                            # drop originals funny/cool
                            drop_cols = [c for c in ['funny', 'cool'] if c in df_chunk.columns]
                            if drop_cols:
                                df_chunk.drop(columns=drop_cols, inplace=True)
                            # dedup within chunk
                            before = len(df_chunk)
                            df_chunk = df_chunk.drop_duplicates(subset=['user_id'], keep='first')
                            dups_removed = before - len(df_chunk)
                            # append
                            df_chunk.to_csv(output_file, index=False, mode='a', header=not header_written)
                            header_written = True
                            total_rows += line_count
                            total_written += len(df_chunk)
                            total_dups_removed += dups_removed
                            logger.info(f"  Wrote user chunk: {len(df_chunk):,} rows (written total: {total_written:,})")

                            # Clear memory
                            chunk_data = []
                            line_count = 0
                            del df_chunk
                            gc.collect()

                    except json.JSONDecodeError:
                        continue

            # Process final chunk if any remaining
            if chunk_data:
                df_chunk = pd.DataFrame(chunk_data)
                # numeric conversions
                numeric_cols = ['review_count', 'useful', 'funny', 'cool', 'fans',
                               'average_stars', 'compliment_hot', 'compliment_more',
                               'compliment_profile', 'compliment_cute', 'compliment_list',
                               'compliment_note', 'compliment_plain', 'compliment_cool',
                               'compliment_funny', 'compliment_writer', 'compliment_photos']
                for col in numeric_cols:
                    if col in df_chunk.columns:
                        df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
                df_chunk['yelping_since'] = pd.to_datetime(df_chunk['yelping_since'], errors='coerce')
                # funny_cool
                has_funny = 'funny' in df_chunk.columns
                has_cool = 'cool' in df_chunk.columns
                if has_funny or has_cool:
                    df_chunk['funny_cool'] = (df_chunk['funny'].fillna(0) if has_funny else 0) + (df_chunk['cool'].fillna(0) if has_cool else 0)
                # tenure
                reference_date = pd.Timestamp('2025-01-31')
                df_chunk['user_tenure_days'] = (reference_date - df_chunk['yelping_since']).dt.days
                df_chunk['user_tenure_years'] = df_chunk['user_tenure_days'] / 365.25
                # fills
                for col in numeric_cols:
                    if col in df_chunk.columns:
                        df_chunk[col].fillna(0, inplace=True)
                if 'name' in df_chunk.columns:
                    df_chunk['name'].fillna('Anonymous', inplace=True)
                if 'elite' in df_chunk.columns:
                    df_chunk['elite'].fillna('', inplace=True)
                if 'friends' in df_chunk.columns:
                    df_chunk['friends'].fillna('', inplace=True)
                # drop originals funny/cool
                drop_cols = [c for c in ['funny', 'cool'] if c in df_chunk.columns]
                if drop_cols:
                    df_chunk.drop(columns=drop_cols, inplace=True)
                # dedup within chunk
                before = len(df_chunk)
                df_chunk = df_chunk.drop_duplicates(subset=['user_id'], keep='first')
                dups_removed = before - len(df_chunk)
                # append
                df_chunk.to_csv(output_file, index=False, mode='a', header=not header_written)
                total_rows += len(chunk_data)
                total_written += len(df_chunk)
                total_dups_removed += dups_removed
                logger.info(f"  Wrote final user chunk: {len(df_chunk):,} rows (written total: {total_written:,})")
                del df_chunk
                gc.collect()

            logger.info(f"\nSaved: {output_file} (input rows: {total_rows:,}, written rows: {total_written:,})")
            # Save summary
            self.cleaning_summary["user"] = {
                "input_rows": int(total_rows),
                "final_rows": int(total_written),
                "duplicates_removed": int(total_dups_removed),
                "rows_dropped_invalid_date": 0,
            }
            try:
                with open(self.output_path / "cleaning_summary.json", 'w', encoding='utf-8') as f:
                    json.dump(self.cleaning_summary, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not write cleaning summary: {e}")
            return pd.read_csv(output_file, nrows=0)

        # If a small in-memory sample was loaded above
        numeric_cols = ['review_count', 'useful', 'funny', 'cool', 'fans',
                       'average_stars', 'compliment_hot', 'compliment_more',
                       'compliment_profile', 'compliment_cute', 'compliment_list',
                       'compliment_note', 'compliment_plain', 'compliment_cool',
                       'compliment_funny', 'compliment_writer', 'compliment_photos']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['yelping_since'] = pd.to_datetime(df['yelping_since'], errors='coerce')

        reference_date = pd.Timestamp('2025-01-31')
        df['user_tenure_days'] = (reference_date - df['yelping_since']).dt.days
        df['user_tenure_years'] = df['user_tenure_days'] / 365.25

        logger.info(f"\nCalculated user tenure (days since joining Yelp)")
        logger.info(f"  Mean tenure: {df['user_tenure_years'].mean():.2f} years")
        logger.info(f"  Median tenure: {df['user_tenure_years'].median():.2f} years")

        has_funny = 'funny' in df.columns
        has_cool = 'cool' in df.columns
        if has_funny or has_cool:
            df['funny_cool'] = (df['funny'].fillna(0) if has_funny else 0) + (df['cool'].fillna(0) if has_cool else 0)

        logger.info(f"\nMissing values BEFORE cleaning:")
        missing_before = df.isnull().sum()
        logger.info(f"\n{missing_before[missing_before > 0]}")

        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

        if 'name' in df.columns:
            df['name'].fillna('Anonymous', inplace=True)
        if 'elite' in df.columns:
            df['elite'].fillna('', inplace=True)
        if 'friends' in df.columns:
            df['friends'].fillna('', inplace=True)

        logger.info(f"\nMissing values AFTER cleaning:")
        missing_after = df.isnull().sum()
        logger.info(f"\n{missing_after[missing_after > 0]}")

        drop_cols = [c for c in ['funny', 'cool'] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        before_dup = len(df)
        df = df.drop_duplicates(subset=['user_id'], keep='first')
        after_dup = len(df)
        removed_dups = before_dup - after_dup
        logger.info(f"\nRemoved {removed_dups:,} duplicate users")

        logger.info(f"\nHandling outliers using IQR method:")
        outlier_columns = ['review_count', 'useful', 'fans', 'average_stars']
        if 'funny_cool' in df.columns:
            outlier_columns.append('funny_cool')

        total_outliers = 0
        for col in outlier_columns:
            if col in df.columns and len(df) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers_count > 0:
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        logger.info(f"  {col}: Clipped {outliers_count:,} outliers")
                        total_outliers += outliers_count

        logger.info(f"\nData Quality Summary:")
        logger.info(f"  Final shape: {df.shape}")
        logger.info(f"  Average reviews per user: {df['review_count'].mean():.2f}")
        logger.info(f"  Average useful votes: {df['useful'].mean():.2f}")

        output_file = self.output_path / "user_clean.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"\nSaved: {output_file}")

        return df


def main():
    """Run Phase 1: Data Cleaning only."""
    print("="*70)
    print("CS 412 RESEARCH PROJECT - DATA PREPROCESSING")
    print("Business Success Prediction using Yelp Dataset")
    print("="*70)
    print("\nPipeline: Data Cleaning")
    print("")

    # Non-interactive: default to FULL dataset; override via SAMPLE_SIZE env
    env_sample = os.environ.get('SAMPLE_SIZE', '').strip()
    if env_sample:
        try:
            sample_size = int(env_sample)
            print(f"Using SAMPLE_SIZE from environment: {sample_size}")
        except ValueError:
            sample_size = None
            print("Invalid SAMPLE_SIZE; defaulting to full dataset")
    else:
        sample_size = None

    print("\n" + "="*70)
    print("PHASE 1: DATA CLEANING")
    print("="*70)

    cleaner = DataCleaner()

    print("\n1/3 Cleaning business data...")
    business_df = cleaner.clean_business_data(sample_size)

    print("\n2/3 Cleaning review data...")
    review_df = cleaner.clean_review_data(sample_size)

    print("\n3/3 Cleaning user data...")
    user_df = cleaner.clean_user_data(sample_size)

    print("\n[OK] Data cleaning complete!")
    # Report on-disk counts for streamed datasets
    def _count_rows_fast(path: str) -> int:
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                return max(sum(1 for _ in f) - 1, 0)
        except Exception:
            return -1

    review_cols = len(pd.read_csv('data/processed/review_clean.csv', nrows=0).columns) if (Path('data/processed/review_clean.csv')).exists() else 0
    user_cols = len(pd.read_csv('data/processed/user_clean.csv', nrows=0).columns) if (Path('data/processed/user_clean.csv')).exists() else 0
    print(f"  - business_clean.csv: {business_df.shape}")
    print(f"  - review_clean.csv: ({_count_rows_fast('data/processed/review_clean.csv')}, {review_cols})")
    print(f"  - user_clean.csv: ({_count_rows_fast('data/processed/user_clean.csv')}, {user_cols})")


if __name__ == "__main__":
    main()


