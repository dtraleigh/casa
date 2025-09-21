from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
import subprocess, logging

logger = logging.getLogger(__name__)

def casa_login(request):
    message = "Log in to Emo Casa"

    if request.POST:
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_active:
                login(request, user)

                return HttpResponseRedirect("/")
            else:
                message = "Account is disabled."
        else:
            message = "Invalid login."

    return render(request, "pages/login.html", {"message": message})


def casa_logout(request):
    logout(request)

    return HttpResponseRedirect("/")

@login_required
def dashboard_view(request):
    ups_data = {}
    ups_error = None
    runtime_minutes = None
    current_watts = 'NA'

    try:
        result = subprocess.run(
            ['upsc', 'cyberpower@localhost'],
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug("UPSC Output:\n%s", result.stdout)

        for line in result.stdout.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace('.', '_')
                ups_data[key] = value.strip()

        logger.debug("Parsed UPS data: %s", ups_data)

        runtime_minutes = int(ups_data.get("battery_runtime", 0)) // 60

        STATUS_MAP = {
            "OL": "Online (using utility power)",
            "OB": "On Battery (power outage)",
            "LB": "Low Battery",
            "HB": "High Battery",
            "CHRG": "Charging",
            "DISCHRG": "Discharging",
            "RB": "Replace Battery",
            "BYPASS": "Bypass Active",
            "CAL": "Calibrating",
            "OFF": "Offline",
            "OVER": "Overload",
            "TRIM": "Voltage Too High (Trimming)",
            "BOOST": "Voltage Too Low (Boosting)",
            "FSD": "Forced Shutdown",
        }

        raw_status = ups_data.get("ups_status", "")
        status_descriptions = [STATUS_MAP.get(code, code) for code in raw_status.split()]
        ups_data["ups_status_readable"] = ", ".join(status_descriptions)

        load_percent = int(ups_data["ups_load"])
        nominal_power = int(ups_data["ups_realpower_nominal"])
        current_watts = (load_percent / 100) * nominal_power
        logger.info(f"Calculated current watts: {current_watts}")

    except subprocess.CalledProcessError as e:
        logger.error("Error running upsc: %s", e)
        ups_error = "Unable to connect to UPS. Check power or USB connection."
    except Exception as e:
        logger.exception("Unexpected error in dashboard_view")
        ups_error = "Unexpected error occurred while fetching UPS data."

    return render(request, 'core/dashboard.html', {
        'ups_data': ups_data,
        'ups_error': ups_error,
        'runtime_minutes': runtime_minutes,
        'current_watts': current_watts
    })