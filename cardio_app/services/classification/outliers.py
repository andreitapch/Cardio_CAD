import numpy as np
import pandas as pd



def remove_outliers_by_class_global(combined_df, feature_col_name, y_col_name, threshold=5.0, aggregation='mean'):
    df = combined_df.copy()
    filtered_list = []

    # Process each class separately.
    for class_label in df[y_col_name].unique():
        subset = df[df[y_col_name] == class_label].copy()

        # Ensure that each feature is a numeric array.
        subset[feature_col_name] = subset[feature_col_name].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

        # Stack all time series into a 2D array.
        time_series_list = subset[feature_col_name].tolist()
        X = np.vstack([np.asarray(ts, dtype=np.float64) for ts in time_series_list])

        # Compute point-wise median and MAD.
        median_series = np.median(X, axis=0)
        mad_series = np.median(np.abs(X - median_series), axis=0)

        # Compute the outlier score for each time series.
        scores = []
        eps = 1e-6  # Small constant to avoid division by zero.
        for row in X:
            norm_diff = np.abs(row - median_series) / (mad_series + eps)
            if aggregation == 'max':
                score = np.max(norm_diff)
            elif aggregation == 'mean':
                score = np.mean(norm_diff)
            else:
                raise ValueError("Aggregation must be 'max' or 'mean'")
            scores.append(score)

        subset['outlier_score'] = scores
        # Flag rows where the score exceeds the threshold.
        subset['is_outlier'] = subset['outlier_score'] > threshold

        print(f"Class '{class_label}': outlier scores (first 5 rows): {subset['outlier_score'].head().tolist()}")
        removed_count = subset['is_outlier'].sum()
        total_count = len(subset)
        print(f"Class '{class_label}': {removed_count} rows flagged as outliers out of {total_count} total rows")

        # Only keep rows that are NOT flagged as outliers.
        filtered_subset = subset[~subset['is_outlier']]
        filtered_list.append(filtered_subset)

    result_df = pd.concat(filtered_list, ignore_index=True)
    print(f"Total rows retained after removal: {len(result_df)} out of {len(df)} total rows")
    return result_df


def remove_outliers_from_df(features, combined_df, lower_percentile, upper_percentile):
    combined_df = combined_df.copy()

    if isinstance(features, str):
        features = [features]

    for feature in features:
        if feature in combined_df.columns:
            combined_df[feature] = pd.to_numeric(combined_df[feature], errors='coerce')

            if pd.api.types.is_numeric_dtype(combined_df[feature]):
                lower_bound = np.percentile(combined_df[feature], lower_percentile)
                upper_bound = np.percentile(combined_df[feature], upper_percentile)

                print(f"\n🔍 Debugging '{feature}':")
                print(f"   - Lower Bound ({lower_percentile}%): {lower_bound}")
                print(f"   - Upper Bound ({upper_percentile}%): {upper_bound}")

                before_count = combined_df.shape[0]
                combined_df = combined_df[(combined_df[feature] >= lower_bound) & (combined_df[feature] <= upper_bound)]
                after_count = combined_df.shape[0]

                print(
                    f"   - Rows Before: {before_count}, Rows After: {after_count}, Removed: {before_count - after_count}")

    return combined_df
