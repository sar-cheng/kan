import requests

# URL to the stores GraphQL API
url = 'https://www.platypusshoes.com.au/graphql'

# Query parameters
query = """
query stores($search: String!, $pageSize: Int!, $filters: StoreFilterInput, $currentPage: Int!) {
  stores(search: $search, pageSize: $pageSize, filters: $filters, currentPage: $currentPage) {
    items {
      additional_information
      city
      distance
      country_id
      detailed_page_url
      flag_image
      holiday_trading_hours
      latitude
      longitude
      name
      phone
      postcode
      region
      region_id
      source_code
      street
      working_hours
      url_key
      monday_closed
      monday_opening
      monday_closing
      tuesday_closed
      tuesday_opening
      tuesday_closing
      wednesday_closed
      wednesday_opening
      wednesday_closing
      thursday_closed
      thursday_opening
      thursday_closing
      friday_closed
      friday_opening
      friday_closing
      saturday_closed
      saturday_opening
      saturday_closing
      sunday_closed
      sunday_opening
      sunday_closing
      __typename
    }
    page_info {
      current_page
      page_size
      total_pages
      __typename
    }
    total_count
    __typename
  }
}
"""

variables = {
  "search": "",
  "pageSize": 10000,
  "filters": {
    "latitude": -33.7693709,
    "longitude": 151.0696512
  },
  "currentPage": 1
}

# Headers (include any necessary headers like User-Agent if needed)
headers = {
    'Content-Type': 'application/json'
}

# Making the request
response = requests.post(url, json={'query': query, 'variables': variables}, headers=headers)

# Parsing the response
data = response.json()

# Processing the data
for store in data['data']['stores']['items']:
    print(f"Name: {store['name']}, Address: {store['street']}, {store['city']}, Coordinates: ({store['latitude']}, {store['longitude']})")

# Save the data to a CSV if needed
import csv

with open('stores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Street', 'City', 'Latitude', 'Longitude'])

    for store in data['data']['stores']['items']:
        writer.writerow([store['name'], store['street'], store['city'], store['latitude'], store['longitude']])
