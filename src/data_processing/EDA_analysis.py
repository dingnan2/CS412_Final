"""
EDA Analysis: Load cleaned datasets and produce visualizations and a markdown report.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
import re

warnings.filterwarnings('ignore')

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class EDAAnalyzer:
    """
    Comprehensive EDA with visualizations
    Focus: Understanding patterns for business success prediction
    """

    def __init__(self, processed_path: str = "data/processed", plot_path: Optional[str] = None):
        # Resolve directories relative to this script to avoid CWD issues
        self.base_dir = Path(__file__).parent
        self.processed_path = Path(processed_path)
        # Default plots to src/data_processing/plots
        resolved_plot_path = Path(plot_path) if plot_path else (self.base_dir / "plots")
        self.plot_path = resolved_plot_path
        self.plot_path.mkdir(parents=True, exist_ok=True)

        # Load cleaned data
        self.business_df = None
        self.review_df = None
        self.user_df = None

    def load_cleaned_data(self):
        """Load all cleaned datasets"""

        self.business_df = pd.read_csv(self.processed_path / "business_clean.csv")

        # Default to FULL dataset unless EDA_SAMPLE_N is specified
        eda_sample_env = os.environ.get('EDA_SAMPLE_N', '').strip()
        review_path = self.processed_path / "review_clean.csv"

        if eda_sample_env:
            try:
                nrows = int(eda_sample_env)
                self.review_df = pd.read_csv(review_path, nrows=nrows)
            except Exception as e:
                self.review_df = pd.read_csv(review_path)
        else:
            chunk_size = 50000
            chunks = []
            row_count = 0
            for i, chunk in enumerate(pd.read_csv(review_path, chunksize=chunk_size)):
                chunks.append(chunk)
                row_count += len(chunk)
                
            self.review_df = pd.concat(chunks, ignore_index=True)
        self.review_df['date'] = pd.to_datetime(self.review_df['date'])

        user_path = self.processed_path / "user_clean.csv"
        if eda_sample_env:
            try:
                nrows = int(eda_sample_env)
                self.user_df = pd.read_csv(user_path, nrows=nrows)
            except Exception as e:
                chunk_size = 50000
                chunks = []
                for chunk in pd.read_csv(user_path, chunksize=chunk_size):
                    chunks.append(chunk)
                    if len(chunks) * chunk_size >= nrows:
                        break
                self.user_df = pd.concat(chunks, ignore_index=True).iloc[:nrows]
        else:
            # Use chunked reading for full file to avoid memory issues
            chunk_size = 50000
            chunks = []
            row_count = 0
            for i, chunk in enumerate(pd.read_csv(user_path, chunksize=chunk_size)):
                chunks.append(chunk)
                row_count += len(chunk)
                
            self.user_df = pd.concat(chunks, ignore_index=True)

        self.user_df['yelping_since'] = pd.to_datetime(self.user_df['yelping_since'])

    def analyze_business_data(self):

        df = self.business_df

        # Basic statistics

        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Business Data Analysis', fontsize=16, fontweight='bold')

        # 1. Business Status Distribution (Bar Chart)
        ax = axes[0, 0]
        status_counts = df['is_open'].value_counts()
        labels = ['Open', 'Closed']
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(labels, [status_counts.get(1, 0), status_counts.get(0, 0)], color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Business Status Distribution')
        ax.text(0, status_counts.get(1, 0), f"{status_counts.get(1, 0):,}\n({status_counts.get(1, 0)/len(df)*100:.1f}%)",
                ha='center', va='bottom')
        ax.text(1, status_counts.get(0, 0), f"{status_counts.get(0, 0):,}\n({status_counts.get(0, 0)/len(df)*100:.1f}%)",
                ha='center', va='bottom')

        # 2. Rating Distribution (Histogram)
        ax = axes[0, 1]
        ax.hist(df['stars'], bins=20, edgecolor='black', alpha=0.7, color='#3498db')
        ax.set_xlabel('Stars')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Rating Distribution (Mean: {df["stars"].mean():.2f})')
        ax.axvline(df['stars'].mean(), color='red', linestyle='--', label='Mean')
        ax.legend()

        # 3. Review Count Distribution (Histogram with log scale)
        ax = axes[0, 2]
        ax.hist(df['review_count'], bins=50, edgecolor='black', alpha=0.7, color='#9b59b6')
        ax.set_xlabel('Review Count')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Review Count Distribution (Median: {df["review_count"].median():.0f})')
        ax.set_yscale('log')

        # 4. Stars vs Business Status (Box Plot)
        ax = axes[1, 0]
        open_stars = df[df['is_open'] == 1]['stars']
        closed_stars = df[df['is_open'] == 0]['stars']
        ax.boxplot([open_stars, closed_stars], labels=['Open', 'Closed'])
        ax.set_ylabel('Stars')
        ax.set_title('Rating Distribution by Business Status')
        ax.grid(True, alpha=0.3)

        # 5. Top 10 States by Business Count (Bar Chart)
        ax = axes[1, 1]
        top_states = df['state'].value_counts().head(10)
        ax.barh(range(len(top_states)), top_states.values, color='#1abc9c')
        ax.set_yticks(range(len(top_states)))
        ax.set_yticklabels(top_states.index)
        ax.set_xlabel('Number of Businesses')
        ax.set_title('Top 10 States by Business Count')
        ax.invert_yaxis()

        # 6. Success Rate by State (Top 10 states)
        ax = axes[1, 2]
        top_10_states = df['state'].value_counts().head(10).index
        success_rates = []
        for state in top_10_states:
            state_df = df[df['state'] == state]
            success_rate = state_df['is_open'].mean() * 100
            success_rates.append(success_rate)

        ax.barh(range(len(top_10_states)), success_rates, color='#e67e22')
        ax.set_yticks(range(len(top_10_states)))
        ax.set_yticklabels(top_10_states)
        ax.set_xlabel('Success Rate (%)')
        ax.set_title('Business Success Rate by State (Top 10)')
        ax.axvline(df['is_open'].mean()*100, color='red', linestyle='--', label='Overall Avg')
        ax.legend()
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.plot_path / 'business_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Additional analysis: Top categories (Top 20, counts and success rate)
        all_categories = []
        for cats in df['categories'].fillna(''):
            if cats:
                all_categories.extend([c.strip() for c in str(cats).split(',') if c.strip()])
        category_series = pd.Series(all_categories)
        top20 = category_series.value_counts().head(20)
        # Plot Top-20 category counts
        plt.figure(figsize=(12, 8))
        top20.sort_values().plot(kind='barh', color='#34495e', edgecolor='black')
        plt.title('Top 20 Business Categories (Count)')
        plt.xlabel('Number of Businesses')
        plt.tight_layout()
        plt.savefig(self.plot_path / 'business_top_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Success rate by category (Top 20 only)
        cat_success = []
        for cat in top20.index:
            mask = df['categories'].fillna('').str.contains(fr'\b{re.escape(cat)}\b', case=False, na=False)
            if mask.any():
                cat_success.append((cat, df.loc[mask, 'is_open'].mean() * 100))
        if cat_success:
            cats, rates = zip(*cat_success)
            plt.figure(figsize=(12, 8))
            order = np.argsort(rates)
            plt.barh(np.array(cats)[order], np.array(rates)[order], color='#27ae60', edgecolor='black')
            plt.axvline(df['is_open'].mean()*100, color='red', linestyle='--', label='Overall Avg')
            plt.xlabel('Success Rate (%)')
            plt.title('Success Rate by Top 20 Categories')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_path / 'business_top_categories_success.png', dpi=300, bbox_inches='tight')
            plt.close()

    def analyze_review_data(self):

        df = self.review_df

        # Basic statistics

        # Text length analysis (memory-safe batching)
        try:
            batch_size = 500_000
            n = len(df)
            lengths = np.empty(n, dtype=np.int32)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                lengths[start:end] = df['text'].iloc[start:end].astype(str).str.len().to_numpy(dtype=np.int32, copy=False)
            df['text_length'] = lengths
        except MemoryError:
            sample_n = min(1_000_000, len(df))
            sample_idx = np.random.RandomState(42).choice(len(df), size=sample_n, replace=False)
            df_sample = df.iloc[sample_idx].copy()
            df_sample['text_length'] = df_sample['text'].astype(str).str.len()
            df_for_text_plots = df_sample
        else:
            df_for_text_plots = df

        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Review Data Analysis', fontsize=16, fontweight='bold')

        # 1. Review Rating Distribution (Bar Chart)
        ax = axes[0, 0]
        rating_counts = df['stars'].value_counts().sort_index()
        ax.bar(rating_counts.index, rating_counts.values, color='#3498db', edgecolor='black')
        ax.set_xlabel('Stars')
        ax.set_ylabel('Count')
        ax.set_title('Review Rating Distribution')
        ax.set_xticks([1, 2, 3, 4, 5])

        # 2. Reviews Over Time (Line Chart - Yearly)
        ax = axes[0, 1]
        reviews_by_year = df.groupby(df['date'].dt.year).size()
        ax.plot(reviews_by_year.index, reviews_by_year.values, marker='o', linewidth=2, color='#e74c3c')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Reviews')
        ax.set_title('Review Volume Over Time')
        ax.grid(True, alpha=0.3)

        # 3. Useful Votes Distribution (Histogram)
        ax = axes[0, 2]
        useful_data = df['useful'][df['useful'] <= df['useful'].quantile(0.95)]
        ax.hist(useful_data, bins=50, edgecolor='black', alpha=0.7, color='#2ecc71')
        ax.set_xlabel('Useful Votes')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Useful Votes Distribution (Median: {df["useful"].median():.0f})')
        ax.set_yscale('log')

        # 4. Text Length Distribution (Histogram)
        ax = axes[1, 0]
        text_length_data = df_for_text_plots['text_length'][df_for_text_plots['text_length'] <= df_for_text_plots['text_length'].quantile(0.95)]
        ax.hist(text_length_data, bins=50, edgecolor='black', alpha=0.7, color='#9b59b6')
        ax.set_xlabel('Text Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Review Text Length (Median: {df_for_text_plots["text_length"].median():.0f})')

        # 5. Reviews per Business (Histogram)
        ax = axes[1, 1]
        reviews_per_business = df.groupby('business_id').size()
        ax.hist(reviews_per_business, bins=50, edgecolor='black', alpha=0.7, color='#f39c12')
        ax.set_xlabel('Reviews per Business')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Reviews per Business (Median: {reviews_per_business.median():.0f})')
        ax.set_yscale('log')

        # 6. Average Rating Over Time (Line Chart)
        ax = axes[1, 2]
        avg_rating_by_year = df.groupby(df['date'].dt.year)['stars'].mean()
        ax.plot(avg_rating_by_year.index, avg_rating_by_year.values, marker='o', linewidth=2, color='#1abc9c')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Stars')
        ax.set_title('Average Rating Trend Over Time')
        ax.set_ylim([3.5, 4.5])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_path / 'review_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Monthly trends (reviews, average rating, average text length)
        monthly = df.assign(year_month=df['date'].dt.to_period('M')).groupby('year_month').agg(
            reviews=('review_id', 'count'),
            avg_rating=('stars', 'mean'),
            avg_text_length=('text_length', 'mean'),
        )
        monthly.index = monthly.index.astype(str)
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        axes[0].plot(monthly.index, monthly['reviews'], color='#2980b9')
        axes[0].set_title('Monthly Review Volume')
        axes[0].set_ylabel('Reviews')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(monthly.index, monthly['avg_rating'], color='#27ae60')
        axes[1].set_title('Monthly Average Rating')
        axes[1].set_ylabel('Avg Stars')
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(monthly.index, monthly['avg_text_length'], color='#8e44ad')
        axes[2].set_title('Monthly Average Review Length')
        axes[2].set_ylabel('Avg Characters')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_path / 'review_monthly_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Rating velocity (last 6 months vs prior period)
        max_date = df['date'].max()
        cutoff_recent = max_date - pd.Timedelta(days=180)
        recent = df[df['date'] > cutoff_recent]
        prior = df[df['date'] <= cutoff_recent]
        bus_recent = recent.groupby('business_id')['stars'].mean()
        bus_prior = prior.groupby('business_id')['stars'].mean()
        velocity = (bus_recent - bus_prior).dropna()
        plt.figure(figsize=(10, 6))
        clipped = velocity[(velocity >= velocity.quantile(0.01)) & (velocity <= velocity.quantile(0.99))]
        plt.hist(clipped, bins=50, color='#16a085', edgecolor='black', alpha=0.8)
        plt.title('Business Rating Velocity (Recent 6M - Prior)')
        plt.xlabel('Delta Stars')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(self.plot_path / 'review_rating_velocity.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Sentiment analysis (if available)
        # OPTIMIZED: Save sentiment scores to CSV for reuse in Phase 3 (feature engineering)
        if VADER_AVAILABLE:
            
            analyzer = SentimentIntensityAnalyzer()
                
            # Compute sentiment in batches to show progress
            n_reviews = len(df)
            batch_size = 100000
            sentiments = []
                
            for start in range(0, n_reviews, batch_size):
                end = min(start + batch_size, n_reviews)
                batch_texts = df['text'].iloc[start:end].fillna('')
                batch_sentiments = batch_texts.apply(lambda t: analyzer.polarity_scores(str(t))['compound'])
                sentiments.extend(batch_sentiments.tolist())
                
            # Create sentiment DataFrame with review_id for joining
            df_sent = df[['review_id', 'business_id', 'date']].copy()
            df_sent['sentiment'] = sentiments
                
            # [SAVE] SENTIMENT TO CSV FOR PHASE 3 REUSE
            sentiment_output_path = self.processed_path / 'review_sentiment.csv'
            df_sent.to_csv(sentiment_output_path, index=False)
                
            # Plot monthly sentiment trend
            monthly_sent = df_sent.assign(year_month=df_sent['date'].dt.to_period('M')).groupby('year_month')['sentiment'].mean()
            monthly_sent.index = monthly_sent.index.astype(str)
            plt.figure(figsize=(14, 5))
            plt.plot(monthly_sent.index, monthly_sent.values, color='#c0392b')
            plt.title('Monthly Average Review Sentiment (VADER)')
            plt.ylabel('Compound Sentiment')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.plot_path / 'review_monthly_sentiment.png', dpi=300, bbox_inches='tight')
            plt.close()


    def analyze_user_data(self):

        df = self.user_df

        # Basic statistics

        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('User Data Analysis', fontsize=16, fontweight='bold')

        # 1. Review Count Distribution (Histogram, log scale)
        ax = axes[0, 0]
        review_count_data = df['review_count'][df['review_count'] > 0]
        ax.hist(review_count_data, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax.set_xlabel('Reviews per User')
        ax.set_ylabel('Frequency')
        ax.set_title(f'User Review Count (Median: {df["review_count"].median():.0f})')
        ax.set_yscale('log')

        # 2. User Tenure Distribution (Histogram)
        ax = axes[0, 1]
        ax.hist(df['user_tenure_years'], bins=30, edgecolor='black', alpha=0.7, color='#e74c3c')
        ax.set_xlabel('Years on Yelp')
        ax.set_ylabel('Frequency')
        ax.set_title(f'User Tenure (Mean: {df["user_tenure_years"].mean():.1f} years)')

        # 3. Useful Votes Distribution (Histogram, log scale)
        ax = axes[0, 2]
        useful_data = df['useful'][df['useful'] > 0]
        ax.hist(useful_data, bins=50, edgecolor='black', alpha=0.7, color='#2ecc71')
        ax.set_xlabel('Useful Votes Received')
        ax.set_ylabel('Frequency')
        ax.set_title(f'User Useful Votes (Median: {df["useful"].median():.0f})')
        ax.set_yscale('log')

        # 4. Fans Distribution (Histogram, log scale)
        ax = axes[1, 0]
        fans_data = df['fans'][df['fans'] > 0]
        if len(fans_data) > 0:
            ax.hist(fans_data, bins=50, edgecolor='black', alpha=0.7, color='#9b59b6')
            ax.set_xlabel('Number of Fans')
            ax.set_ylabel('Frequency')
            ax.set_title(f'User Fans (Median: {df["fans"].median():.0f})')
            ax.set_yscale('log')

        # 5. User Tenure vs Review Count (Scatter Plot)
        ax = axes[1, 1]
        sample_df = df.sample(min(10000, len(df)))
        ax.scatter(sample_df['user_tenure_years'], sample_df['review_count'],
                   alpha=0.3, s=10, color='#f39c12')
        ax.set_xlabel('Years on Yelp')
        ax.set_ylabel('Review Count')
        ax.set_title('User Tenure vs Activity Level')
        ax.set_yscale('log')

        # 6. User Credibility Score Distribution (Composite Score A)
        ax = axes[1, 2]
        useful_rate = df['useful'] / (df['review_count'] + 1)
        tenure_weight = np.log1p(df['user_tenure_days']).fillna(0) / 10.0
        experience_weight = np.log1p(df['review_count']).fillna(0) / 10.0
        df['credibility_score'] = (useful_rate * 0.5) + (tenure_weight * 0.3) + (experience_weight * 0.2)
        credibility_data = df['credibility_score'][df['credibility_score'] <= df['credibility_score'].quantile(0.95)]
        ax.hist(credibility_data, bins=50, edgecolor='black', alpha=0.7, color='#1abc9c')
        ax.set_xlabel('Credibility Score')
        ax.set_ylabel('Frequency')
        ax.set_title('User Credibility Score Distribution')
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.plot_path / 'user_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # User segments analysis

    def analyze_correlations(self):

        # Aggregate review data by business
        if 'text_length' not in self.review_df.columns:
            self.review_df['text_length'] = self.review_df['text'].astype(str).str.len()
        review_agg = self.review_df.groupby('business_id').agg({
            'stars': ['mean', 'std', 'count'],
            'useful': 'sum',
            'funny_cool': 'sum',
            'text_length': 'mean'
        }).reset_index()

        # Avoid name collision with business_df['review_count'] by renaming aggregated count
        review_agg.columns = [
            'business_id',
            'avg_review_stars',
            'std_review_stars',
            'num_reviews',
            'total_useful',
            'total_funny_cool',
            'avg_text_length',
        ]

        # Merge with business data
        merged_df = self.business_df.merge(review_agg, on='business_id', how='left')

        # Create correlation visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Business Success Correlation Analysis', fontsize=16, fontweight='bold')

        # 1. Average Review Stars vs Business Success (Box Plot)
        ax = axes[0, 0]
        open_reviews = merged_df[merged_df['is_open'] == 1]['avg_review_stars'].dropna()
        closed_reviews = merged_df[merged_df['is_open'] == 0]['avg_review_stars'].dropna()
        ax.boxplot([open_reviews, closed_reviews], labels=['Open', 'Closed'])
        ax.set_ylabel('Average Review Stars')
        ax.set_title('Review Rating vs Business Status')
        ax.grid(True, alpha=0.3)

        # 2. Review Count vs Business Success (Box Plot, log scale)
        ax = axes[0, 1]
        open_count = merged_df[merged_df['is_open'] == 1]['num_reviews'].dropna()
        closed_count = merged_df[merged_df['is_open'] == 0]['num_reviews'].dropna()
        ax.boxplot([open_count, closed_count], labels=['Open', 'Closed'])
        ax.set_ylabel('Review Count')
        ax.set_title('Review Volume vs Business Status')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # 3. Correlation heatmap
        ax = axes[1, 0]
        corr_cols = [
            'is_open',
            'stars',
            'avg_review_stars',
            'num_reviews',
            'total_useful',
            'total_funny_cool',
        ]
        corr_data = merged_df[corr_cols].corr()

        sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix')

        # 4. Success rate by review count quartiles (Bar Chart)
        ax = axes[1, 1]
        try:
            merged_df['review_quartile'] = pd.qcut(
                merged_df['num_reviews'].fillna(0),
                q=4,
                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
                duplicates='drop',
            )
        except ValueError:
            temp_bins = pd.qcut(
                merged_df['num_reviews'].fillna(0), q=4, labels=False, duplicates='drop'
            )
            num_bins = int(pd.Series(temp_bins).nunique())
            fallback_labels = [f"Q{i+1}" for i in range(num_bins)]
            merged_df['review_quartile'] = pd.Categorical(
                [fallback_labels[int(x)] if pd.notna(x) else np.nan for x in temp_bins],
                categories=fallback_labels,
                ordered=True,
            )
        success_by_quartile = merged_df.groupby('review_quartile')['is_open'].mean() * 100

        ax.bar(range(len(success_by_quartile)), success_by_quartile.values, color='#16a085', edgecolor='black')
        ax.set_xticks(range(len(success_by_quartile)))
        ax.set_xticklabels(success_by_quartile.index, rotation=45)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Business Success Rate by Review Volume Quartile')
        ax.axhline(merged_df['is_open'].mean()*100, color='red', linestyle='--', label='Overall Avg')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.plot_path / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Print key correlations
        correlations = corr_data['is_open'].sort_values(ascending=False)
        
    def generate_eda_report(self):
        """Generate comprehensive EDA report (Markdown) with analysis and plot references"""

        report_lines = []
        report_lines.append(f"# CS 412 Research Project - EDA Report")
        report_lines.append("")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("## Overview")
        report_lines.append("This report summarizes data cleaning results and exploratory analysis for the Yelp-based business success prediction project. It includes dataset summaries and commentary for each visualization.")
        report_lines.append("")
        eda_sample_env = os.environ.get('EDA_SAMPLE_N', '').strip()
        if eda_sample_env:
            report_lines.append(f"Note: EDA visualizations use a sample of {eda_sample_env} rows for reviews/users.")
        else:
            report_lines.append("Note: EDA visualizations use the full reviews/users datasets.")
        report_lines.append("")

        # Business summary
        report_lines.append("## 1. Business Data Summary")
        report_lines.append(f"Total businesses: {len(self.business_df):,}")
        report_lines.append(f"Open businesses: {self.business_df['is_open'].sum():,} ({self.business_df['is_open'].mean()*100:.2f}%)")
        report_lines.append(f"Closed businesses: {(1-self.business_df['is_open']).sum():,} ({(1-self.business_df['is_open'].mean())*100:.2f}%)")
        report_lines.append(f"Average rating: {self.business_df['stars'].mean():.2f}")
        report_lines.append(f"Unique states: {self.business_df['state'].nunique()}")
        report_lines.append(f"Unique cities: {self.business_df['city'].nunique()}")
        report_lines.append("")
        report_lines.append("### Figure: Business Analysis")
        report_lines.append("![Business Analysis](./plots/business_analysis.png)")
        report_lines.append("")
        report_lines.append("### Figures: Top Categories")
        report_lines.append("![Top Categories](./plots/business_top_categories.png)")
        report_lines.append("![Category Success Rates](./plots/business_top_categories_success.png)")
        report_lines.append("")
        report_lines.append("- Ratings skew modestly positive; median review counts are low with a long tail.")
        report_lines.append("- Success rate (open) aligns with expectations (~80%), confirming class imbalance.")
        report_lines.append("- Success varies across states; location features may add signal.")
        report_lines.append("")

        # Review summary
        report_lines.append("## 2. Review Data Summary")
        report_lines.append(f"Total reviews: {len(self.review_df):,}")
        report_lines.append(f"Unique businesses reviewed: {self.review_df['business_id'].nunique():,}")
        report_lines.append(f"Average review rating: {self.review_df['stars'].mean():.2f}")
        report_lines.append(f"Date range: {self.review_df['date'].min()} to {self.review_df['date'].max()}")
        report_lines.append("")
        report_lines.append("### Figure: Review Analysis")
        report_lines.append("![Review Analysis](./plots/review_analysis.png)")
        report_lines.append("")
        report_lines.append("### Figures: Temporal & Text Trends")
        report_lines.append("![Monthly Trends](./plots/review_monthly_trends.png)")
        report_lines.append("![Rating Velocity](./plots/review_rating_velocity.png)")
        if VADER_AVAILABLE:
            report_lines.append("![Monthly Sentiment](./plots/review_monthly_sentiment.png)")
        report_lines.append("")
        report_lines.append("- Reviews trend upward over years; ratings average ~3.8–4.0.")
        report_lines.append("- Useful votes and text lengths are heavy-tailed; consider robust aggregations.")
        report_lines.append("- Temporal features like rating velocity and review frequency are promising.")
        report_lines.append("")

        # User summary
        report_lines.append("## 3. User Data Summary")
        report_lines.append(f"Total users: {len(self.user_df):,}")
        report_lines.append(f"Average reviews per user: {self.user_df['review_count'].mean():.2f}")
        report_lines.append(f"Average user tenure: {self.user_df['user_tenure_years'].mean():.2f} years")
        report_lines.append(f"Average useful votes per user: {self.user_df['useful'].mean():.2f}")
        report_lines.append("")
        report_lines.append("### Figure: User Analysis")
        report_lines.append("![User Analysis](./plots/user_analysis.png)")
        report_lines.append("")
        report_lines.append("- User activity and engagement are highly skewed; small power-user group dominates.")
        report_lines.append("- Tenure distribution supports credibility weighting; combine with useful votes.")
        report_lines.append("")

        # Key insights
        report_lines.append("## 4. Correlation & Key Insights for Modeling")
        report_lines.append("### Figure: Correlation Analysis")
        report_lines.append("![Correlation Analysis](./plots/correlation_analysis.png)")
        report_lines.append("")
        report_lines.append("[OK] Class Imbalance: ~80% open, ~20% closed - Need stratified sampling/SMOTE")
        report_lines.append("[OK] Text Data: Reviews contain rich text - Sentiment analysis needed")
        report_lines.append("[OK] Temporal Patterns: Reviews span multiple years - Temporal features important")
        report_lines.append("[OK] User Weighting: High variance in user credibility - User weighting critical")
        report_lines.append("[OK] Geographic Variation: Success rates vary by state/city - Location features needed")
        report_lines.append("")

        report_text = "\n".join(report_lines)
        report_file = self.base_dir / "EDA_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)



def main():
    """Run Phase 2: EDA only (assumes cleaned CSVs exist)."""
    print("="*70)
    print("CS 412 RESEARCH PROJECT - EDA ANALYSIS")
    print("Business Success Prediction using Yelp Dataset")
    print("="*70)
    print("\nPipeline: EDA -> Visualizations -> Report")
    print("")

    print("\n" + "="*70)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("="*70)

    eda = EDAAnalyzer()
    eda.load_cleaned_data()

    print("\n1/4 Analyzing business data...")
    eda.analyze_business_data()

    print("\n2/4 Analyzing review data...")
    eda.analyze_review_data()

    print("\n3/4 Analyzing user data...")
    eda.analyze_user_data()

    print("\n4/4 Analyzing correlations...")
    eda.analyze_correlations()

    print("\n5/5 Generating EDA report...")
    eda.generate_eda_report()

    print("\n" + "="*70)
    print("EDA COMPLETE!")
    print("="*70)
    print("\nVisualizations: src/data_processing/plots/")
    print("  - business_analysis.png")
    print("  - review_analysis.png")
    print("  - user_analysis.png")
    print("  - correlation_analysis.png")
    print("\nReport:")
    print("  - src/data_processing/EDA_report.md")


if __name__ == "__main__":
    main()


