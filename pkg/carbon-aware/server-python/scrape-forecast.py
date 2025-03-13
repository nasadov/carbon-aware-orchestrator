import json
import requests

API_KEY = "REbFju22LwHPVT3t1Y0IKh1I"
ZONE_CODES = ["DE", "FR", "ES", "IT-NO"]
OUTPUT_FILE = "all_forecasts.json"

def main():
    # Dictionary to hold forecast data for each zone
    all_forecasts = {}
    
    for zone in ZONE_CODES:
        url = "https://api.electricitymap.org/v3/carbon-intensity/forecast"
        params = {"zone": zone}
        headers = {"auth-token": API_KEY}

        # Send the request
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            # Store the JSON data for this zone in the dictionary
            all_forecasts[zone] = response.json()
            print(f"Retrieved forecast data for {zone}.")
        else:
            # Store an error message rather than skipping
            all_forecasts[zone] = {"error": response.text}
            print(
                f"Failed to retrieve forecast for {zone} "
                f"(HTTP {response.status_code}): {response.text}"
            )
    
    # Write the entire dictionary to one JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_forecasts, f, indent=2)

    print(f"\nAll forecasts saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()