"""
This script evaluates how manual judgments (judgmental_base, judgmental_exchange)
and the political shift of word_word pairs relate to truthfulness assessments
predicted by language models. It runs a logistic regression per model/prompt/axis
and saves parameter estimates and p-values for the variables of interest
(judgmental_base, judgmental_exchange, shift) to data/results.csv.

Note: Code logic remains unchanged; this file adds documentation and comments only.
"""

import json
import pandas as pd
import numpy as np

import statsmodels.formula.api as sfa

INPUT_PATH = 'data/sample_output.json'
MODEL_NAMES = ['mixtral', 'llama']

def split_data_frame_list(df, target_column, output_type=int):
    """
    Expand rows where the column `target_column` contains a list into multiple rows,
    one per element. Elements equal to None or the string "refusal" are skipped.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    target_column : str
        Column name to expand. Values may be lists or scalars.
    output_type : callable, default=int
        Casting function applied to each element.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per list element in `target_column`.
    """
    row_accumulator = []

    def split_list_to_rows(row):
        split_row = row[target_column]
        if isinstance(split_row, list):
            for s in split_row:
                if s is not None and s != "refusal":
                    new_row = row.to_dict()
                    new_row[target_column] = output_type(s)
                    row_accumulator.append(new_row)
        else:
            new_row = row.to_dict()
            new_row[target_column] = output_type(split_row)
            row_accumulator.append(new_row)

    df.apply(split_list_to_rows, axis=1)
    new_df = pd.DataFrame(row_accumulator)

    return new_df

def create_empty_results(relevant_columns):
    """Create an empty result Series with params_/pvalues_ keys for columns.

    Used when there are too few observations or on model errors to preserve the
    expected result schema.
    """
    empty_params = pd.Series({col: None for col in relevant_columns})
    empty_pvalues = pd.Series({col: None for col in relevant_columns})
    return pd.concat([
        empty_params.rename(lambda x: f"params_{x}"),
        empty_pvalues.rename(lambda x: f"pvalues_{x}")
    ])


def calculate_results(dataframe, model_name, prompt, relevant_columns):
    """
    Fit the logistic regression for a given `model_name` and `prompt` subset.

    Dependent variable: exchange truthfulness list (expanded to per-element rows).
    Covariates: base_truthfulness (mean), C(golden_truthfulness_base_claim),
    judgmental_base, judgmental_exchange, shift.

    Returns a Series with params_*/pvalues_* for `relevant_columns`.
    """
    exchange_model = f'truthfulness_probability_exchange_claim_{prompt}_prompt_{model_name}'
    base_model = f'truthfulness_probability_base_claim_{prompt}_prompt_{model_name}'

    try:
        # First, expand the DataFrame (expand list-valued dependent variable)
        split_df = split_data_frame_list(dataframe, exchange_model)
        
        # Compute base_truthfulness and handle problematic values (take mean)
        split_df["base_truthfulness"] = split_df[base_model].apply(
            lambda x: np.mean([i for i in x if pd.notna(i) and isinstance(i, (int, float))]) 
            if isinstance(x, list) else (
                x if pd.notna(x) and isinstance(x, (int, float)) else np.nan
            )
        )

        # Remove rows with NaN values in essential columns (model inputs)
        columns_to_check = [
            exchange_model, 
            "base_truthfulness",
            "golden_truthfulness_base_claim",
            "judgmental_base",
            "judgmental_exchange",
            "shift",
            "axis"
        ]
        
        clean_df = split_df.dropna(subset=columns_to_check)

        # Check if there are enough observations left (minimal sample size for stability)
        if len(clean_df) < 10:  # minimal number of observations
            print(f"Too few rows for {model_name}, {prompt}: {len(clean_df)} rows")
            return create_empty_results(relevant_columns)

        model = sfa.logit(
            f"{exchange_model} ~ base_truthfulness + C(golden_truthfulness_base_claim) + judgmental_base + judgmental_exchange  + shift",
            missing='drop', 
            data=clean_df
        )
        return results_to_series(model.fit(), relevant_columns)
    
    except Exception as e:
        print(f"Error for {model_name}, {prompt}: {str(e)}")
        return create_empty_results(relevant_columns)


def results_to_series(results, relevant_columns):
    """Collect params and p-values from a fitted model for selected columns."""
    params = results.params
    pvalues = results.pvalues
    params = params[relevant_columns]
    pvalues = pvalues[relevant_columns]
    return pd.concat([params.rename(lambda x: f"params_{x}"), pvalues.rename(lambda x: f"pvalues_{x}")])

def has_valid_values(x):
    """Return True if `x` is a non-empty list; otherwise False."""
    return isinstance(x, list) and len(x) > 0


if __name__ == '__main__':
    # Load nested JSON and flatten to underscore-separated columns

    with open(INPUT_PATH) as file:
        data_dict = json.load(file)

    df = pd.json_normalize(data_dict, sep='_')
    df = df.replace('none', pd.NA).replace('refusal', pd.NA)  # unify missing markers

    # Normalize types/values used in the model
    df["golden_truthfulness_base_claim"] = df["golden_truthfulness_base_claim"].apply(lambda x: str(x).lower())
    df["exchange_len"] = df["exchange_word"].apply(lambda x: len(x))

    # Map textual shift to numeric (-1, 0, 1); keep NA for unknowns
    mapping = {
        "left": -1,
        "libertarian": -1,
        "none": 0,
        "right": 1,
        "authoritarian": 1,
    }
    df["shift"] = (
        df["shift"]
        .astype("string")               # optional, normalized
        .str.lower()
        .map(mapping)                   # unknown/NA -> NaN
        .astype("Int64")                # allows NA
    )

    results_list = []
    for model_name in MODEL_NAMES:
        
        for prompt in ['simple', 'advanced']:

            # Columns containing non-empty lists are required for modeling
            base_col = f"truthfulness_probability_base_claim_{prompt}_prompt_{model_name}"
            exchange_col = f"truthfulness_probability_exchange_claim_{prompt}_prompt_{model_name}"

            filtered_base_df = df[
                (df[base_col].apply(has_valid_values)) &
                (df[exchange_col].apply(has_valid_values))
            ]

            for axis in ["social", "economic", "both"]:
                # Variables of interest for coefficient and p-value export
                relevant_columns = ["judgmental_base",
                                "judgmental_exchange",
                                "shift"]

                if axis == "both":
                    axis_filtered_df = filtered_base_df
                    # relevant_columns.append("axis[T.social]")
                else:
                    axis_filtered_df = filtered_base_df[filtered_base_df['axis'] == axis]


                result_series = calculate_results(axis_filtered_df, model_name, prompt, relevant_columns)
                result_series["model_name"] = model_name
                result_series["prompt"] = prompt
                result_series["axis"] = axis
                results_list.append(result_series)


    results_df = pd.DataFrame(results_list)


    # Save parameter estimates and p-values to CSV for publication
    results_df.to_csv('data/results.csv', index=False)