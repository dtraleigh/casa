#!/usr/bin/env python3
"""
Print-friendly Wemo device info dump using pywemo.
"""

import pywemo
import socket
import pprint

def safe_gethost(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return None

def get_attr_any(obj, *names, default=None):
    """Return first attribute that exists on obj from names (works with camelCase/snake_case)."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default

def print_service_info(name, svc):
    print(f"  Service name: {name}")
    print(f"    repr: {repr(svc)}")

    svc_type = get_attr_any(svc, "service_type", "serviceType", "service")
    if svc_type:
        print(f"    service_type: {svc_type}")

    svc_id = get_attr_any(svc, "service_id", "serviceId")
    if svc_id:
        print(f"    service_id: {svc_id}")

    desc = get_attr_any(svc, "description_url", "descriptionURL", "description")
    if desc:
        print(f"    description_url: {desc}")

    ctrl = get_attr_any(svc, "control_url", "controlURL", "control_url")
    if ctrl:
        print(f"    control_url: {ctrl}")

    event = get_attr_any(svc, "event_subscription_url", "eventSubscriptionURL", "event_subscription_url")
    if event:
        print(f"    event_subscription_url: {event}")

    # Try to show declared actions (if available)
    actions = get_attr_any(svc, "actions", "Actions")
    if actions:
        try:
            # actions might be dict-like or list-like
            if isinstance(actions, dict):
                action_list = list(actions.keys())
            else:
                action_list = list(actions)
            print(f"    actions: {', '.join(map(str, action_list))}")
        except Exception:
            print(f"    actions: {repr(actions)}")
    else:
        # Fallback: show callable public methods on the service object
        callables = [a for a in dir(svc) if not a.startswith("_") and callable(getattr(svc, a))]
        if callables:
            print(f"    methods: {', '.join(callables)}")

def main():
    devices = pywemo.discover_devices()
    if not devices:
        print("No Wemo devices discovered.")
        return

    for d in devices:
        print("=" * 60)
        print(f"Name:           {getattr(d, 'name', None)}")
        host = getattr(d, "host", None)
        port = getattr(d, "port", None)
        print(f"IP:             {host}:{port}")
        print(f"Hostname:       {safe_gethost(host)}")
        print(f"Model:          {get_attr_any(d, 'model', 'model_name')}")
        print(f"Model Name:     {getattr(d, 'model_name', None)}")
        print(f"Serial Number:  {get_attr_any(d, 'serial_number', 'serial')}")
        print(f"UDN:            {getattr(d, 'udn', None)}")
        print(f"MAC:            {getattr(d, 'mac', None)}")
        print(f"Manufacturer:   {getattr(d, 'manufacturer', None)}")
        print(f"Firmware:       {getattr(d, 'firmware_version', None)}")

        # _config_any has useful raw config values
        config = getattr(d, "_config_any", None)
        if config:
            print("\nExtra Config:")
            for k, v in config.items():
                print(f"  {k}: {v}")

        # Services: print per-service info safely
        if hasattr(d, "services") and d.services:
            print("\nServices:")
            try:
                for svc_name, svc in d.services.items():
                    print_service_info(svc_name, svc)
            except Exception as e:
                # Some implementations use a different services structure (ServiceProperties objects)
                print(f"  Error enumerating services: {e!r}")
                # Try fallback: iterate through d._services if present
                _services = getattr(d, "_services", None)
                if _services:
                    for s in _services:
                        print(f"  Service (fallback repr): {repr(s)}")

        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

