"""
Data Preprocessing: Clean individual Yelp datasets and write processed CSVs.
"""

import pandas as pd
import numpy as np
import json
import gc
from pathlib import Path
from typing import Optional, Dict
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


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

        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                
                data.append(json.loads(line.strip()))

        df = pd.DataFrame(data)

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

        # Load
        df = self.load_json_data("yelp_academic_dataset_business.json", sample_size)
        original_cols = list(df.columns)
        summary = {"input_rows": len(df)}

        # Step 1: Remove specified columns
        cols_to_remove = ['postal_code', 'longitude', 'latitude', 'hours']
        existing_to_remove = [col for col in cols_to_remove if col in df.columns]
        df = df.drop(columns=existing_to_remove)

        # Step 2: Data type conversions
        df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
        df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
        df['is_open'] = df['is_open'].astype(int)

        # Step 3: Handle missing values
        missing_before = df.isnull().sum()

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

        missing_after = df.isnull().sum()

        # Step 4: Remove duplicates
        before_dup = len(df)
        df = df.drop_duplicates(subset=['business_id'], keep='first')
        after_dup = len(df)
        removed_dups = before_dup - after_dup

        # Step 4.5: Handle outliers using IQR method
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
                    total_outliers_handled += outliers_count

        # Step 5: Data quality check
        target_counts = df['is_open'].value_counts()

        # Save
        output_file = self.output_path / "business_clean.csv"
        df.to_csv(output_file, index=False)

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
        # Main summary location under src/data_processing for reporting/final aggregation
        summary_path = Path("src/data_processing/cleaning_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_summary, f, indent=2)

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

        filepath = self.raw_path / "yelp_academic_dataset_review.json"
        output_file = self.output_path / "review_clean.csv"

        if sample_size is not None:
            # Small sample path (in-memory)
            df = self.load_json_data("yelp_academic_dataset_review.json", sample_size)

            # Combine funny + cool → funny_cool
            df['funny_cool'] = 0
            if 'funny' in df.columns:
                df['funny_cool'] = df['funny_cool'] + df['funny'].fillna(0)
            if 'cool' in df.columns:
                df['funny_cool'] = df['funny_cool'] + df['cool'].fillna(0)

            cols_to_drop = [col for col in ['funny', 'cool'] if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # Type conversions
            df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
            df['useful'] = pd.to_numeric(df['useful'], errors='coerce')
            df['funny_cool'] = pd.to_numeric(df['funny_cool'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Missing values
            missing_before = df.isnull().sum()
            df['stars'].fillna(df['stars'].median(), inplace=True)
            df['useful'].fillna(0, inplace=True)
            df['funny_cool'].fillna(0, inplace=True)
            df['text'].fillna('', inplace=True)

            # Calculate text length (important feature)
            if 'text' in df.columns:
                df['text_length'] = df['text'].str.len()

            dropped_na_date = int(df['date'].isna().sum())
            df = df.dropna(subset=['date'])
            missing_after = df.isnull().sum()

            # Duplicates
            before_dup = len(df)
            df = df.drop_duplicates(subset=['review_id'], keep='first')
            after_dup = len(df)
            removed_dups = before_dup - after_dup

            # Handle outliers using IQR method
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
                        total_outliers += outliers_count

            # Save
            df.to_csv(output_file, index=False)
            # Summary
            self.cleaning_summary["review"] = {
                "input_rows": int(len(df) + removed_dups + dropped_na_date),
                "final_rows": int(len(df)),
                "duplicates_removed": int(removed_dups),
                "rows_dropped_invalid_date": int(dropped_na_date),
            }
            
            with open(self.output_path / "cleaning_summary.json", 'w', encoding='utf-8') as f:
                json.dump(self.cleaning_summary, f, indent=2)
            
            return df

        # Full dataset path (chunked, low-memory)
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
            del df
            gc.collect()
        # Save summary
        self.cleaning_summary["review"] = {
            "input_rows": int(total_rows),
            "final_rows": int(total_written),
            "duplicates_removed": int(total_dups_removed),
            "rows_dropped_invalid_date": int(total_na_date),
        }
        
        summary_path = Path("src/data_processing/cleaning_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_summary, f, indent=2)
        
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

        filepath = self.raw_path / "yelp_academic_dataset_user.json"
        if sample_size is not None:
            df = self.load_json_data("yelp_academic_dataset_user.json", sample_size)
        else:
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
                del df_chunk
                gc.collect()

            # Save summary
            self.cleaning_summary["user"] = {
                "input_rows": int(total_rows),
                "final_rows": int(total_written),
                "duplicates_removed": int(total_dups_removed),
                "rows_dropped_invalid_date": 0,
            }
            summary_path = Path("src/data_processing/cleaning_summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(self.cleaning_summary, f, indent=2)
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


        has_funny = 'funny' in df.columns
        has_cool = 'cool' in df.columns
        if has_funny or has_cool:
            df['funny_cool'] = (df['funny'].fillna(0) if has_funny else 0) + (df['cool'].fillna(0) if has_cool else 0)

        missing_before = df.isnull().sum()

        for col in numeric_cols:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

        if 'name' in df.columns:
            df['name'].fillna('Anonymous', inplace=True)
        if 'elite' in df.columns:
            df['elite'].fillna('', inplace=True)
        if 'friends' in df.columns:
            df['friends'].fillna('', inplace=True)

        missing_after = df.isnull().sum()

        drop_cols = [c for c in ['funny', 'cool'] if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        before_dup = len(df)
        df = df.drop_duplicates(subset=['user_id'], keep='first')
        after_dup = len(df)
        removed_dups = before_dup - after_dup

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
                        total_outliers += outliers_count


        output_file = self.output_path / "user_clean.csv"
        df.to_csv(output_file, index=False)

        return df
    

    def filter_low_quality_businesses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out low-quality businesses that have insufficient data for prediction.
        
        Criteria:
        - Minimum 5 reviews (changed from 3 to ensure quality)
        - Remove suspicious patterns (e.g., very few reviews but perfect rating)
        
        Args:
            df: Business dataframe
            
        Returns:
            Filtered dataframe
        """
        
        initial_count = len(df)
        
        # Filter 1: Minimum review count
        min_reviews = 5
        df = df[df['review_count'] >= min_reviews]
        after_min_reviews = len(df)
        removed_min_reviews = initial_count - after_min_reviews
        
        # Filter 2: Suspicious patterns (few reviews + perfect rating)
        # These might be fake or biased entries
        suspicious = (df['review_count'] < 10) & (df['stars'] == 5.0)
        df = df[~suspicious]
        after_suspicious = len(df)
        removed_suspicious = after_min_reviews - after_suspicious
        
        # Filter 3: Businesses with invalid categories
        df = df[df['categories'].str.len() > 0]
        after_categories = len(df)
        removed_categories = after_suspicious - after_categories
        
        total_removed = initial_count - len(df)
        
        # Persist quality filter statistics into cleaning_summary for reporting
        business_summary = self.cleaning_summary.get("business", {})
        business_summary["quality_filter"] = {
            "min_reviews_removed": int(removed_min_reviews),
            "suspicious_removed": int(removed_suspicious),
            "no_category_removed": int(removed_categories),
            "final_after_quality_filter": int(len(df)),
        }
        self.cleaning_summary["business"] = business_summary
        summary_path = Path("src/data_processing/cleaning_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleaning_summary, f, indent=2)
        

        return df

    def generate_preprocessing_report(self) -> None:
        """
        Generate a comprehensive markdown report summarizing preprocessing results.
        
        Outputs:
            - src/data_processing/preprocessing_report.md
        """
        
        report_path = Path("src/data_processing/preprocessing_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load summary data (main copy under src/data_processing)
        summary_file = Path("src/data_processing/cleaning_summary.json")
        if not summary_file.exists():
            return
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Generate report content
        report_lines = []
        report_lines.append("# Data Preprocessing Report")

        report_lines.append("---")
        report_lines.append("")
        
        
        # Business Data
        if 'business' in summary:
            bus = summary['business']
            qf = bus.get('quality_filter', {})
            final_after_qf = qf.get('final_after_quality_filter', bus.get('final_rows', bus.get('input_rows', 0)))
            report_lines.append("## 1. Business Data")
            report_lines.append("")
            report_lines.append("### Overview")
            report_lines.append("")
            report_lines.append(f"- **Input rows**: {bus.get('input_rows', 'N/A'):,}")
            report_lines.append(f"- **Final rows**: {final_after_qf:,}")
            report_lines.append(f"- **Data retention**: {final_after_qf/max(bus.get('input_rows', 1), 1)*100:.2f}%")
            report_lines.append("")
            
            report_lines.append("### Cleaning Operations")
            report_lines.append("")
            report_lines.append(f"1. **Duplicates removed**: {bus.get('duplicates_removed', 0):,}")
            report_lines.append(f"2. **Missing values filled**:")
            report_lines.append(f"   - Stars: {bus.get('stars_filled', 0):,}")
            report_lines.append(f"   - Review count: {bus.get('review_count_filled', 0):,}")
            report_lines.append(f"3. **Columns removed**: {bus.get('columns_removed', 0)}")
            report_lines.append(f"4. **Outliers handled**: {bus.get('outliers_handled', 0):,}")
            report_lines.append("")
            
            # Quality filtering
            if 'quality_filter' in bus:
                qf = bus['quality_filter']
                report_lines.append("### Quality Filtering")
                report_lines.append("")
                report_lines.append(f"- **Minimum reviews filter**: {qf.get('min_reviews_removed', 0):,} businesses")
                report_lines.append(f"- **Suspicious patterns**: {qf.get('suspicious_removed', 0):,} businesses")
                report_lines.append(f"- **Invalid categories**: {qf.get('no_category_removed', 0):,} businesses")
                report_lines.append("")
        
        # Review Data
        if 'review' in summary:
            rev = summary['review']
            report_lines.append("## 2. Review Data")
            report_lines.append("")
            report_lines.append("### Overview")
            report_lines.append("")
            report_lines.append(f"- **Input rows**: {rev.get('input_rows', 'N/A'):,}")
            report_lines.append(f"- **Final rows**: {rev.get('final_rows', 'N/A'):,}")
            report_lines.append(f"- **Data retention**: {rev.get('final_rows', 0)/max(rev.get('input_rows', 1), 1)*100:.2f}%")
            report_lines.append("")
            
            report_lines.append("### Cleaning Operations")
            report_lines.append("")
            report_lines.append(f"1. **Duplicates removed**: {rev.get('duplicates_removed', 0):,}")
            report_lines.append(f"2. **Invalid dates dropped**: {rev.get('rows_dropped_invalid_date', 0):,}")
            report_lines.append(f"3. **Feature engineering**: Created `funny_cool` and `text_length`")
            report_lines.append("")
        
        # User Data
        if 'user' in summary:
            usr = summary['user']
            report_lines.append("## 3. User Data")
            report_lines.append("")
            report_lines.append("### Overview")
            report_lines.append("")
            report_lines.append(f"- **Input rows**: {usr.get('input_rows', 'N/A'):,}")
            report_lines.append(f"- **Final rows**: {usr.get('final_rows', 'N/A'):,}")
            report_lines.append(f"- **Data retention**: {usr.get('final_rows', 0)/max(usr.get('input_rows', 1), 1)*100:.2f}%")
            report_lines.append("")
            
            report_lines.append("### Cleaning Operations")
            report_lines.append("")
            report_lines.append(f"1. **Duplicates removed**: {usr.get('duplicates_removed', 0):,}")
            report_lines.append(f"2. **Feature engineering**: Created `funny_cool` and `user_tenure_years`")
            report_lines.append("")
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        

    
def main():
    """Run Phase 1: Data Cleaning with quality filtering and report generation."""
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

    print("\n1/4 Cleaning business data...")
    business_df = cleaner.clean_business_data(sample_size)

    print("\n2/4 Filtering low-quality businesses...")
    business_df = cleaner.filter_low_quality_businesses(business_df)
    
    # Save filtered business data
    filtered_output = Path("data/processed/business_clean.csv")
    business_df.to_csv(filtered_output, index=False)
    print(f"Saved filtered business data: {filtered_output}")
    
    # Update summary with quality filter stats
    # cleaning_summary['business']['quality_filter'] is already populated inside
    # filter_low_quality_businesses; no need to overwrite here.

    print("\n3/4 Cleaning review data...")
    review_df = cleaner.clean_review_data(sample_size)

    print("\n4/4 Cleaning user data...")
    user_df = cleaner.clean_user_data(sample_size)

    # Generate preprocessing report
    print("\n" + "="*70)
    print("GENERATING PREPROCESSING REPORT")
    print("="*70)
    cleaner.generate_preprocessing_report()

    print("\n[OK] Data preprocessing complete!")
    
    # Report final counts
    def _count_rows_fast(path: str) -> int:
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                return max(sum(1 for _ in f) - 1, 0)
        except Exception:
            return -1

    review_cols = len(pd.read_csv('data/processed/review_clean.csv', nrows=0).columns) if (Path('data/processed/review_clean.csv')).exists() else 0
    user_cols = len(pd.read_csv('data/processed/user_clean.csv', nrows=0).columns) if (Path('data/processed/user_clean.csv')).exists() else 0
    
    print("\nFinal dataset sizes:")
    print(f"  - business_clean.csv: {business_df.shape}")
    print(f"  - review_clean.csv: ({_count_rows_fast('data/processed/review_clean.csv')}, {review_cols})")
    print(f"  - user_clean.csv: ({_count_rows_fast('data/processed/user_clean.csv')}, {user_cols})")
    print("")
    print("Output files:")
    print("  - data/processed/*.csv (cleaned datasets)")
    print("  - data/processed/cleaning_summary.json")
    print("  - src/data_processing/preprocessing_report.md")
    print("")


if __name__ == "__main__":
    main()