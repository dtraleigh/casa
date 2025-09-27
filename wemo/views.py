# views.py
import json
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from .models import WemoSwitch
import requests
import logging

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["POST"])
def wemo_discover(request):
    """AJAX endpoint to discover/update Wemo devices."""
    try:
        # Import here to avoid startup issues if pywemo isn't installed
        import pywemo
        import socket

        def safe_gethost(ip):
            try:
                return socket.gethostbyaddr(ip)[0]
            except Exception:
                return None

        def get_attr_any(obj, *names, default=None):
            for name in names:
                if hasattr(obj, name):
                    return getattr(obj, name)
            return default

        def device_exists_and_update(device):
            """Check if device exists and update if needed."""
            udn = getattr(device, 'udn', None)
            serial = get_attr_any(device, 'serial_number', 'serial')
            mac = getattr(device, 'mac', None)
            existing_switch = None

            # Match by UDN, Serial, or MAC
            if udn:
                existing_switch = WemoSwitch.objects.filter(udn=udn).first()
                match_type = 'UDN'
            elif serial:
                existing_switch = WemoSwitch.objects.filter(serial_number=serial).first()
                match_type = 'Serial'
            elif mac:
                existing_switch = WemoSwitch.objects.filter(mac_address=mac).first()
                match_type = 'MAC'
            else:
                # Fallback to IP + Name
                host = getattr(device, "host", None)
                name = getattr(device, 'name', None)
                if host and name:
                    existing_switch = WemoSwitch.objects.filter(ip_address=host, name=name).first()
                    match_type = 'IP+Name'

            if existing_switch:
                # Update existing device
                host = getattr(device, "host", None)
                port = getattr(device, "port", None)
                hostname = safe_gethost(host) if host else None
                mac = getattr(device, 'mac', None)

                changes = []
                if existing_switch.ip_address != host:
                    changes.append(f"IP: {existing_switch.ip_address} → {host}")
                    existing_switch.ip_address = host

                if existing_switch.port != port:
                    changes.append(f"Port: {existing_switch.port} → {port}")
                    existing_switch.port = port

                if existing_switch.hostname != hostname:
                    changes.append(f"Hostname: '{existing_switch.hostname}' → '{hostname}'")
                    existing_switch.hostname = hostname

                if mac and existing_switch.mac_address != mac:
                    changes.append(f"MAC: {existing_switch.mac_address} → {mac}")
                    existing_switch.mac_address = mac

                if changes:
                    existing_switch.save()
                    return 'updated', existing_switch.name, changes
                else:
                    return 'unchanged', existing_switch.name, []

            return None, None, []

        # Discover devices
        devices = pywemo.discover_devices()

        if not devices:
            return JsonResponse({
                'success': True,
                'message': 'No Wemo devices found on network',
                'discovered': 0,
                'new': 0,
                'updated': 0,
                'details': []
            })

        new_count = 0
        updated_count = 0
        unchanged_count = 0
        details = []

        for device in devices:
            name = getattr(device, 'name', 'Unknown')
            host = getattr(device, 'host', 'Unknown')

            result, device_name, changes = device_exists_and_update(device)

            if result == 'updated':
                updated_count += 1
                details.append({
                    'action': 'updated',
                    'name': device_name,
                    'changes': changes
                })
            elif result == 'unchanged':
                unchanged_count += 1
                details.append({
                    'action': 'unchanged',
                    'name': device_name,
                    'ip': host
                })
            else:
                # New device - create it
                try:
                    switch = WemoSwitch(
                        name=name,
                        hostname=safe_gethost(host),
                        ip_address=host,
                        port=getattr(device, "port", None),
                        model=get_attr_any(device, 'model', 'model_name'),
                        model_name=getattr(device, 'model_name', None),
                        serial_number=get_attr_any(device, 'serial_number', 'serial'),
                        udn=getattr(device, 'udn', None),
                        mac_address=getattr(device, 'mac', None),
                        manufacturer=getattr(device, 'manufacturer', None),
                        firmware_version=getattr(device, 'firmware_version', None),
                    )
                    switch.save()
                    new_count += 1
                    details.append({
                        'action': 'added',
                        'name': name,
                        'ip': host
                    })
                except Exception as e:
                    details.append({
                        'action': 'error',
                        'name': name,
                        'error': str(e)
                    })

        return JsonResponse({
            'success': True,
            'message': f'Discovery complete: {new_count} added, {updated_count} updated, {unchanged_count} unchanged',
            'discovered': len(devices),
            'new': new_count,
            'updated': updated_count,
            'unchanged': unchanged_count,
            'details': details
        })

    except ImportError:
        return JsonResponse({
            'success': False,
            'error': 'pywemo library not available'
        })
    except Exception as e:
        logger.error(f"Error in wemo_discover: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Discovery failed: {str(e)}'
        })


@login_required
def wemo_main(request):
    """Main Wemo control page with device status."""
    switches = WemoSwitch.objects.filter(disabled=False).order_by('name')

    # Get current state for each switch
    switch_data = []
    for switch in switches:
        try:
            # Attempt to get current state
            state = switch.get_state()
            online = True
            current_state = state if state is not None else 0
        except Exception as e:
            logger.warning(f"Failed to get state for {switch.name}: {e}")
            online = False
            current_state = 0

        switch_data.append({
            'switch': switch,
            'online': online,
            'current_state': current_state,
            'is_on': current_state == 1
        })

    context = {
        'switch_data': switch_data,
        'total_switches': len(switches),
        'online_count': sum(1 for data in switch_data if data['online'])
    }

    return render(request, 'wemo/wemo_main.html', context)


@login_required
@require_http_methods(["POST"])
@csrf_exempt  # You might want to handle CSRF properly in production
def wemo_toggle(request, switch_id):
    """AJAX endpoint to toggle a Wemo switch."""
    try:
        switch = get_object_or_404(WemoSwitch, id=switch_id, disabled=False)

        # Get current state first
        try:
            current_state = switch.get_state()
            if current_state is None:
                return JsonResponse({
                    'success': False,
                    'error': 'Could not determine current switch state',
                    'online': False
                })
        except Exception as e:
            logger.error(f"Failed to get state for switch {switch.name}: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Device appears to be offline: {str(e)}',
                'online': False
            })

        # Toggle the switch
        try:
            if current_state == 1:
                switch.turn_off()
                new_state = 0
                action = 'turned off'
            else:
                switch.turn_on()
                new_state = 1
                action = 'turned on'

            # Update last_seen timestamp
            switch.last_seen = timezone.now()
            switch.save(update_fields=['last_seen'])

            return JsonResponse({
                'success': True,
                'new_state': new_state,
                'is_on': new_state == 1,
                'message': f'{switch.name} {action} successfully',
                'online': True,
                'last_seen': switch.last_seen.isoformat()
            })

        except Exception as e:
            logger.error(f"Failed to toggle switch {switch.name}: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Failed to control device: {str(e)}',
                'online': False
            })

    except Exception as e:
        logger.error(f"Error in wemo_toggle: {e}")
        return JsonResponse({
            'success': False,
            'error': 'An unexpected error occurred'
        })


@login_required
@require_http_methods(["GET"])
def wemo_refresh_status(request, switch_id):
    """AJAX endpoint to refresh the status of a specific switch."""
    try:
        switch = get_object_or_404(WemoSwitch, id=switch_id, disabled=False)

        try:
            state = switch.get_state()
            switch.last_seen = timezone.now()
            switch.save(update_fields=['last_seen'])

            return JsonResponse({
                'success': True,
                'current_state': state,
                'is_on': state == 1,
                'online': True,
                'last_seen': switch.last_seen.isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to refresh status for {switch.name}: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e),
                'online': False
            })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': 'Switch not found'
        })