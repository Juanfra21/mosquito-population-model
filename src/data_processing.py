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
    # Create a yearweek column
    population_df["date"] = population_df['date'].dt.isocalendar().year.astype(str) + "_" + population_df['date'].dt.isocalendar().week.astype(str).str.zfill(2)

    # Group by date column
    population_df = population_df.groupby("date").sum().reset_index()

    # Sort by date
    population_df.sort_values("date")

    return population_df


def create_future_population(population_df):
    # Calculate the average population of the current day and the next two weeks
    population_df["population_2d"] = (
        population_df["population"].rolling(window=2, min_periods=1).mean().shift(-1)
    )

    # Calculate the average population of the current day and the next three weeks
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

    # Create yearweek column
    weather_df["datetime"] = weather_df['datetime'].dt.isocalendar().year.astype(str) + "_" + weather_df['datetime'].dt.isocalendar().week.astype(str).str.zfill(2)

    # Group by datetime column
    weather_df = weather_df.groupby("datetime").agg({'precip': 'sum','humidity': 'mean','temp': 'mean'}).reset_index()

    # Average temperature in the last 3 weeks
    weather_df["lw_avg_temp"] = weather_df["temp"].rolling(window=3).mean()

    # Average humidity in the last 3 weeks
    weather_df["lw_avg_humidity"] = weather_df["humidity"].rolling(window=3).mean()

    # Precipitation in the last 3 weeks
    weather_df["lw_precip"] = weather_df["precip"].rolling(window=3).sum()

    # Rename the columns appropriately
    weather_df.rename(columns={"datetime": "date"}, inplace=True)
    return weather_df


def merge_population_and_weather(population_df, weather_df, population_rolling_window, shift=True):
    # Shift weather data by 1 day
    if shift:
        # Shift all columns except the first one down by 1 row
        weather_df.iloc[1:] = weather_df.iloc[:-1].values

    # Merge population and weather data
    data = pd.merge(population_df, weather_df, on="date", how="left")

    # Smooth data using a 3 week rolling window
    data["population"] = data["population"].rolling(window = population_rolling_window).mean().fillna(method='ffill')

    return data


def prepare_data(population_path, weather_path, population_rolling_window):
    population_df = interpolate_population(load_population_data(population_path))
    population_df = create_future_population(population_df)
    weather_df = generate_weather_data(weather_path)

    return merge_population_and_weather(population_df, weather_df, population_rolling_window)