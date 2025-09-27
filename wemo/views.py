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
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


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