import datetime
import numpy as np
import pandas as pd
from src.config import CONFIG


def load_population_data(population_path):
    # Read data
    population_df = pd.read_csv(population_path)

    # Convert total to float
    population_df["TOTAL"] = population_df["TOTAL"].str.replace(",", "").astype(float)

    # Create date column
    population_df["Date"] = population_df.apply(
        lambda row: datetime.datetime(row["YEAR"], 1, 1)
        + datetime.timedelta(days=row["DOY"] - 1),
        axis=1,
    )

    # Drop unnecessary columns
    population_df = population_df[["Date", "TOTAL"]]

    # Rename columns
    population_df.columns = ["date", "population"]

    # Handle duplicate dates by aggregating
    population_df = population_df.groupby("date").sum().reset_index()

    return population_df


def interpolate_population(population_df):
    # Sort by date
    population_df = population_df.sort_values("date")

    # Set the date column as index
    population_df.set_index("date", inplace=True)

    # Create a date range from the first day until the last day
    date_range = pd.date_range(
        start=population_df.index.min(), end=population_df.index.max(), freq="D"
    )

    # Use the date range as new index
    population_df = population_df.reindex(date_range)

    # Interpolate to fill population values
    population_df = population_df.interpolate(method="linear")

    # Reset the index and turn the dates back into a column
    population_df.reset_index(inplace=True)

    # Rename the columns appropriately
    population_df.rename(columns={"index": "date"}, inplace=True)

    return population_df


def create_future_population(population_df):
    # Calculate the average population of the current day and the next two days
    population_df["population_2d"] = (
        population_df["population"].rolling(window=2, min_periods=1).mean().shift(-1)
    )

    # Calculate the average population of the current day and the next three days
    population_df["population_3d"] = (
        population_df["population"].rolling(window=3, min_periods=1).mean().shift(-2)
    )

    # Forward fill the NaN values
    population_df["population_2d"].fillna(method="ffill", inplace=True)
    population_df["population_3d"].fillna(method="ffill", inplace=True)

    return population_df


def generate_weather_data(weather_path):
    # Retrieve weather data
    weather_df = pd.read_csv(weather_path, parse_dates=[1])

    # Select necessary column
    weather_df = weather_df[CONFIG["data"]["selected_weather_columns"]]

    # Average temperature in the last week
    weather_df["lw_avg_temp"] = weather_df["temp"].rolling(window=7).mean()

    # Average humidity in the last week
    weather_df["lw_avg_humidity"] = weather_df["humidity"].rolling(window=7).mean()

    # Precipitation in the last week
    weather_df["lw_precip"] = weather_df["precip"].rolling(window=7).sum()

    # Rename the columns appropriately
    weather_df.rename(columns={"datetime": "date"}, inplace=True)

    return weather_df


def merge_population_and_weather(population_df, weather_df, shift):
    # Shift weather data by 1 day
    if shift:
        weather_df["date"] = weather_df["date"] + pd.DateOffset(days=1)

    # Merge population and weather data
    data = pd.merge(population_df, weather_df, on="date", how="left")

    # Create a new column indicating if the date is 2004 or before
    data['is_2004_or_before'] = (data['date'].dt.year <= 2004).astype(int)

    return data


def prepare_data():
    population_df = interpolate_population(
        load_population_data(CONFIG["data"]["population_path"])
    )
    population_df = create_future_population(population_df)
    weather_df = generate_weather_data(CONFIG["data"]["weather_path"])
    return merge_population_and_weather(population_df, weather_df, shift=True)
