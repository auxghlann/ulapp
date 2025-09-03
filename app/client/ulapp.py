from groq import Groq
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class Ulapp:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    WEATHER_API = os.getenv("WEATHER_API")

    SYSTEM_PROMPT = (
        "You are a helpful assistant that identifies the best laundry day for the user given the weather forecast."
    )

    def __init__(self, location: str):
        self.client = Groq(
            api_key=Ulapp.GROQ_API_KEY,
        )
        self.location = location    

    def _call_weather_api(self):
        """
            Calls WeatherAPI forecast endpoint and extracts laundry-relevant parameters.

            Parameters:
                location (str): Location query (default: Tuguegarao).

            Returns:
                dict: Extracted weather parameters for laundry decision.
        """
        url = f"https://api.weatherapi.com/v1/forecast.json?q={self.location}&days=1&key={Ulapp.WEATHER_API}"
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
    

    def get_response(self) -> str:
        prompt = f"Identify the best laundry day based on the weather forecast for {self.location}. Please call the weather API to get current forecast data."

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
            # Get weather data
            weather_data = self._call_weather_api()
            
            # Create a follow-up message with the weather data
            follow_up_response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": Ulapp.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "tool_calls": response.choices[0].message.tool_calls},
                    {"role": "tool", "content": str(weather_data), "tool_call_id": response.choices[0].message.tool_calls[0].id}
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