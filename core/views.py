from django.http import FileResponse, Http404, HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login, logout
import os, subprocess, logging

from . import logs

logger = logging.getLogger(__name__)

LOG_LINE_CHOICES = [200, 500, 1000, 5000, 20000]
LOG_LEVEL_CHOICES = ['ALL', 'INFO', 'WARNING', 'ERROR']

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


def _staff(user):
    return user.is_staff


def _selected_log(request, available):
    """(log dict, backup index, path) from the query string, 404 if not whitelisted."""
    key = request.GET.get('log') or (available[0]['key'] if available else '')
    try:
        backup = int(request.GET.get('backup', 0))
    except ValueError:
        raise Http404("Unknown log")
    path = logs.resolve_log(key, backup)
    if path is None:
        raise Http404("Unknown log")
    log = next(l for l in available if l['key'] == key)
    return log, backup, path


@login_required
@user_passes_test(_staff)
def logs_view(request):
    available = logs.available_logs()
    if not available:
        return render(request, 'core/logs.html', {'available': available})

    log, backup, path = _selected_log(request, available)

    try:
        lines = int(request.GET.get('lines', 500))
    except ValueError:
        lines = 500
    if lines not in LOG_LINE_CHOICES:
        lines = 500
    level = request.GET.get('level', 'ALL')
    if level not in LOG_LEVEL_CHOICES:
        level = 'ALL'
    query = request.GET.get('q', '').strip()

    entries = []
    error = None
    try:
        entries = logs.group_entries(logs.tail_lines(path, lines))
        entries = logs.filter_entries(entries, min_level=None if level == 'ALL' else level, query=query)
    except FileNotFoundError:
        error = f"{os.path.basename(path)} doesn't exist yet."
    except PermissionError:
        error = f"The web server doesn't have permission to read {path}."

    return render(request, 'core/logs.html', {
        'available': available,
        'log': log,
        'backup': backup,
        'file_name': os.path.basename(path),
        'file_info': logs.file_info(path),
        'entries': entries,
        'error': error,
        'lines': lines,
        'level': level,
        'query': query,
        'line_choices': LOG_LINE_CHOICES,
        'level_choices': LOG_LEVEL_CHOICES,
    })


@login_required
@user_passes_test(_staff)
def log_download_view(request):
    _, _, path = _selected_log(request, logs.available_logs())
    try:
        return FileResponse(open(path, 'rb'), as_attachment=True, filename=os.path.basename(path))
    except OSError:
        raise Http404("Log not readable")
