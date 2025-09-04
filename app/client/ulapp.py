from groq import Groq
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class Ulapp:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    WEATHER_API = os.getenv("WEATHER_API")
    DEFAULT_DAY_FORECAST = 3

    SYSTEM_PROMPT = (
        "You are a helpful assistant that identifies the best laundry day for the user given the weather forecast."
    )

    def __init__(self, location: str):
        self.client = Groq(
            api_key=Ulapp.GROQ_API_KEY,
        )
        self.location = location    

    def _call_weather_api(self) -> list[dict[str: str | float | int]]:
        """
        Calls WeatherAPI forecast endpoint and extracts laundry-relevant parameters.

        Parameters:
            location (str): Location query (default: Tuguegarao).

        Returns:
            list[dict]: Extracted weather parameters for each forecast day.
        """
        url = f"https://api.weatherapi.com/v1/forecast.json?q={self.location}&days={Ulapp.DEFAULT_DAY_FORECAST}&key={Ulapp.WEATHER_API}"
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Weather API error: {response.status_code}, {response.text}")

        data = response.json()
        forecast_days = data["forecast"]["forecastday"]

        results = []
        for day in forecast_days:
            summary = {
                "date": day["date"],
                "avg_temp_c": day["day"]["avgtemp_c"],
                "max_temp_c": day["day"]["maxtemp_c"],
                "min_temp_c": day["day"]["mintemp_c"],
                "avg_humidity": day["day"]["avghumidity"],
                "daily_chance_of_rain": day["day"].get("daily_chance_of_rain", "N/A"),
                "max_wind_kph": day["day"]["maxwind_kph"],
                "total_precip_mm": day["day"]["totalprecip_mm"],
                "condition": day["day"]["condition"]["text"],
            }
            results.append(summary)

        return results

    

    def get_response(self) -> str:
        prompt = f"Identify the best laundry day based on the weather forecast for {self.location}. Call the weather API to get current forecast data."

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": Ulapp.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "call_weather_api",
                        "description": "Calls WeatherAPI forecast endpoint and extracts laundry-relevant parameters for the specified location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "Location query for weather forecast"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ],
            tool_choice="auto"
        )

        # Check if the model wants to call a tool
        if response.choices[0].message.tool_calls:
            # Get weather data (now returns a list of dictionaries)
            weather_data = self._call_weather_api()
            
            # Format the weather data as JSON for better AI comprehension
            weather_data_json = json.dumps(weather_data, indent=2)
            
            # Create a follow-up message with the weather data
            follow_up_response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": Ulapp.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "tool_calls": response.choices[0].message.tool_calls},
                    {"role": "tool", "content": weather_data_json, "tool_call_id": response.choices[0].message.tool_calls[0].id}
                ]
            )
            
            return follow_up_response.choices[0].message.content
        
        return response.choices[0].message.content


if __name__ == "__main__":
    # Test the Ulapp class
    print("Testing Ulapp Weather-Based Laundry Assistant...")
    print("-" * 50)
    
    # You can change this location to test different cities
    test_location = "Tuguegarao"
    
    try:
        # Create an instance of Ulapp
        ulapp = Ulapp(location=test_location)
        print(f"Location: {test_location}")
        print()
        
        # Test the weather API call directly
        print("Testing weather API call...")
        weather_data = ulapp._call_weather_api()
        print("Weather data retrieved:")
        
        # Handle the list of dictionaries format
        if isinstance(weather_data, list):
            for i, day_data in enumerate(weather_data):
                print(f"  Day {i + 1}:")
                for key, value in day_data.items():
                    print(f"    {key}: {value}")
                print()
        else:
            # Fallback for single dictionary (backward compatibility)
            for key, value in weather_data.items():
                print(f"  {key}: {value}")
        print()
        
        # Test the full response with AI recommendation
        print("Getting AI recommendation...")
        recommendation = ulapp.get_response()
        print("AI Recommendation:")
        print(recommendation)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set up your environment variables:")
        print("- GROQ_API_KEY: Your Groq API key")
        print("- WEATHER_API: Your WeatherAPI key")
        print("- Create a .env file with these variables")