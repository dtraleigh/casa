"""Open-Meteo weather lookup for Mycroft's `get_current_weather` tool (Phase 4).

Open-Meteo is free and needs no API key. A lookup is at most two small HTTPS
requests: a geocode, which is skipped for the default location, and then a
forecast. They only fire when the model asks for weather, which is the "as
offline as possible" principle. This module only fetches and shapes the data.
Registering it as a tool and dispatching calls to it happens in `tools.py`.
"""
from dataclasses import dataclass
from datetime import date, datetime

import requests
from django.conf import settings

GEOCODE_URL = 'https://geocoding-api.open-meteo.com/v1/search'
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'


class WeatherError(Exception):
    """A lookup failure whose message is safe to hand back to the model, which
    relays it to the user (e.g. "Couldn't find a place called 'Atlantis'.")."""


@dataclass(frozen=True)
class Place:
    name: str
    latitude: float
    longitude: float

    @classmethod
    def from_geocode(cls, result):
        parts = [result.get('name'), result.get('admin1'), result.get('country')]
        # De-dup while keeping order (city-states repeat name/admin1).
        label = ', '.join(dict.fromkeys(p for p in parts if p))
        return cls(label, result['latitude'], result['longitude'])


# Where the household is. Used when the user doesn't say where, so the most
# common question ("what's the weather?") costs no geocode call.
DEFAULT_PLACE = Place('Raleigh, North Carolina, United States', 35.7796, -78.6382)

# WMO weather interpretation codes, as used by Open-Meteo.
WEATHER_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Freezing fog',
    51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle',
    56: 'Light freezing drizzle', 57: 'Freezing drizzle',
    61: 'Light rain', 63: 'Rain', 65: 'Heavy rain',
    66: 'Light freezing rain', 67: 'Freezing rain',
    71: 'Light snow', 73: 'Snow', 75: 'Heavy snow', 77: 'Snow grains',
    80: 'Light rain showers', 81: 'Rain showers', 82: 'Violent rain showers',
    85: 'Light snow showers', 86: 'Snow showers',
    95: 'Thunderstorm', 96: 'Thunderstorm with hail',
    99: 'Thunderstorm with heavy hail',
}

# "Cary, NC" style qualifiers: Open-Meteo's geocoder matches on the place name
# alone, so the part after the comma is matched against the results' state or
# country to pick the right Cary.
US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut',
    'DE': 'Delaware', 'DC': 'District of Columbia', 'FL': 'Florida',
    'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky',
    'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
    'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
    'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
    'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming',
}

# Hourly outlook: the next N hours, sampled every STEP hours, which is enough
# to reason about "tonight" without flooding the context window.
_OUTLOOK_HOURS = 12
_OUTLOOK_STEP = 2
_FORECAST_DAYS = 7


def _get(url, params):
    """GET JSON from Open-Meteo, mapping every transport/API failure to a
    WeatherError the model can relay."""
    try:
        resp = requests.get(
            url, params=params, timeout=settings.MYCROFT_WEATHER_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise WeatherError(
            "The weather service couldn't be reached right now.") from exc


def _qualifier_matches(result, qualifier):
    q = qualifier.lower()
    if q in ('us', 'usa', 'united states'):
        return result.get('country_code') == 'US'
    wanted = {q, US_STATES.get(qualifier.upper(), '').lower()} - {''}
    have = {
        (result.get('admin1') or '').lower(),
        (result.get('country') or '').lower(),
        (result.get('country_code') or '').lower(),
    }
    return bool(wanted & have)


def geocode(query):
    """Resolve a free-text location ("Asheville", "Cary, NC", "Paris, France")
    to a Place. Raises WeatherError if nothing matches."""
    name, _, qualifier = query.partition(',')
    name, qualifier = name.strip(), qualifier.strip()
    if not name:
        raise WeatherError("No location was given.")
    data = _get(GEOCODE_URL, {
        'name': name, 'count': 10, 'language': 'en', 'format': 'json',
    })
    results = [r for r in (data.get('results') or [])
               if 'latitude' in r and 'longitude' in r]
    if not results:
        raise WeatherError(f"Couldn't find a place called {query!r}.")
    if qualifier:
        # Results come back most-populous first, so the first match wins. With
        # no match at all, fall back to the top result. The reply names the
        # place it used, so a wrong guess is visible to the user.
        results = ([r for r in results if _qualifier_matches(r, qualifier)]
                   or results)
    return Place.from_geocode(results[0])


def _round(value):
    return None if value is None else round(value)


def _conditions(code):
    return WEATHER_CODES.get(code, 'Unknown')


def _clock(dt, minutes=False):
    """'3 PM' / '7:42 PM', without the zero-padding strftime adds (the
    no-pad flag isn't portable across platforms)."""
    text = dt.strftime('%I:%M %p' if minutes else '%I %p')
    return text.lstrip('0')


def _day_label(day, today):
    delta = (day - today).days
    return {0: 'today', 1: 'tomorrow'}.get(delta, day.strftime('%A'))


def _shape(place, data):
    """Trim Open-Meteo's response to what the model needs, with plain labelled
    fields (US units) so a small model can't misread arrays or units."""
    current = data.get('current') or {}
    observed = datetime.fromisoformat(current['time'])
    tz = data.get('timezone_abbreviation') or data.get('timezone') or ''

    hourly = data.get('hourly') or {}
    outlook = []
    for i in range(0, len(hourly.get('time') or []), _OUTLOOK_STEP):
        at = datetime.fromisoformat(hourly['time'][i])
        outlook.append({
            'time': f"{at.strftime('%a')} {_clock(at)}",
            'temperature_f': _round(hourly['temperature_2m'][i]),
            'precipitation_chance_pct': hourly['precipitation_probability'][i],
            'conditions': _conditions(hourly['weather_code'][i]),
        })

    daily = data.get('daily') or {}
    today = observed.date()
    days = []
    for i, day in enumerate(daily.get('time') or []):
        d = date.fromisoformat(day)
        days.append({
            'day': _day_label(d, today),
            'date': day,
            'conditions': _conditions(daily['weather_code'][i]),
            'high_f': _round(daily['temperature_2m_max'][i]),
            'low_f': _round(daily['temperature_2m_min'][i]),
            'precipitation_chance_pct': daily['precipitation_probability_max'][i],
            'sunset': (_clock(datetime.fromisoformat(daily['sunset'][i]), True)
                       if daily.get('sunset') else None),
        })

    stamp = f"{observed.strftime('%A %Y-%m-%d')} {_clock(observed, True)} {tz}"
    return {
        'location': place.name,
        'observed_at': stamp.strip(),
        'current': {
            'conditions': _conditions(current.get('weather_code')),
            'temperature_f': _round(current.get('temperature_2m')),
            'feels_like_f': _round(current.get('apparent_temperature')),
            'humidity_pct': current.get('relative_humidity_2m'),
            'wind_mph': _round(current.get('wind_speed_10m')),
            'wind_gusts_mph': _round(current.get('wind_gusts_10m')),
            'precipitation_in': current.get('precipitation'),
            'daytime': bool(current.get('is_day')),
        },
        'next_hours': outlook,
        'forecast': days,
    }


def get_current_weather(location=None):
    """Current conditions, a short hourly outlook and a 7-day forecast for
    `location`. A blank location means the household's (Raleigh). Returns a
    plain dict; raises WeatherError on a lookup failure."""
    location = (location or '').strip()
    place = geocode(location) if location else DEFAULT_PLACE
    data = _get(FORECAST_URL, {
        'latitude': place.latitude,
        'longitude': place.longitude,
        'current': ','.join([
            'temperature_2m', 'apparent_temperature', 'relative_humidity_2m',
            'precipitation', 'weather_code', 'wind_speed_10m',
            'wind_gusts_10m', 'is_day',
        ]),
        'hourly': 'temperature_2m,precipitation_probability,weather_code',
        'daily': ','.join([
            'weather_code', 'temperature_2m_max', 'temperature_2m_min',
            'precipitation_probability_max', 'sunset',
        ]),
        'forecast_hours': _OUTLOOK_HOURS,
        'forecast_days': _FORECAST_DAYS,
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'precipitation_unit': 'inch',
        'timezone': 'auto',
    })
    try:
        return _shape(place, data)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise WeatherError(
            "The weather service sent back something unexpected.") from exc
