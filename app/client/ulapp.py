from groq import Groq
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class Ulapp:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    WEATHER_API = os.getenv("WEATHER_API")

    def __init__(self):
        self.client = Groq(
            api_key=os.getenv(Ulapp.GROQ_API_KEY),
        )
    

    def _call_weather_api(api_key: str, location: str = "Tuguegarao"):
        """
            Calls WeatherAPI forecast endpoint and extracts laundry-relevant parameters.

            Parameters:
                api_key (str): Your WeatherAPI key.
                location (str): Location query (default: Tuguegarao).

            Returns:
                dict: Extracted weather parameters for laundry decision.
        """
        url = f"https://api.weatherapi.com/v1/forecast.json?q={location}&days=1&key={Ulapp.WEATHER_API}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Weather API error: {response.status_code}, {response.text}")

        data = response.json()
        forecast = data["forecast"]["forecastday"][0]["day"]

        return {
            "avg_temp_c": forecast["avgtemp_c"],
            "max_temp_c": forecast["maxtemp_c"],
            "min_temp_c": forecast["mintemp_c"],
            "avg_humidity": forecast["avghumidity"],
            "daily_chance_of_rain": forecast.get("daily_chance_of_rain", "N/A"),
            "max_wind_kph": forecast["maxwind_kph"],
            "total_precip_mm": forecast["totalprecip_mm"],
            "condition": forecast["condition"]["text"],
        }

