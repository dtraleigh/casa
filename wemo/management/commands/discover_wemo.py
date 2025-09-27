import socket

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from wemo.models import WemoSwitch

try:
    import pywemo
except ImportError:
    raise CommandError("pywemo is required. Install it with: pip install pywemo")


class Command(BaseCommand):
    help = 'Discover Wemo switches on the network and add new ones to the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be added without actually adding devices',
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed information about discovered devices',
        )

    def safe_gethost(self, ip):
        """Safely get hostname from IP address."""
        try:
            return socket.gethostbyaddr(ip)[0]
        except Exception:
            return None

    def get_attr_any(self, obj, *names, default=None):
        """Return first attribute that exists on obj from names."""
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def device_exists(self, device):
        """Check if device already exists in database using multiple identifiers."""
        udn = getattr(device, 'udn', None)
        serial = self.get_attr_any(device, 'serial_number', 'serial')
        mac = getattr(device, 'mac', None)
        existing_switch = None

        # Priority 1: Match by UDN (most reliable)
        if udn:
            existing_switch = WemoSwitch.objects.filter(udn=udn).first()
            if existing_switch:
                return True, 'UDN', existing_switch

        # Priority 2: Match by Serial Number
        if serial:
            existing_switch = WemoSwitch.objects.filter(serial_number=serial).first()
            if existing_switch:
                return True, 'Serial Number', existing_switch

        # Priority 3: Match by MAC Address (handles IP/port changes)
        if mac:
            existing_switch = WemoSwitch.objects.filter(mac_address=mac).first()
            if existing_switch:
                return True, 'MAC Address', existing_switch

        # Fallback: check by IP and name combination
        host = getattr(device, "host", None)
        name = getattr(device, 'name', None)
        if host and name:
            existing_switch = WemoSwitch.objects.filter(ip_address=host, name=name).first()
            if existing_switch:
                return True, 'IP + Name', existing_switch

        return False, None, None

    def update_existing_device(self, existing_switch, device):
        """Update existing device with new network information."""
        host = getattr(device, "host", None)
        port = getattr(device, "port", None)
        hostname = self.safe_gethost(host) if host else None
        mac = getattr(device, 'mac', None)

        # Track what changed
        changes = []

        if existing_switch.ip_address != host:
            changes.append(f"IP: {existing_switch.ip_address} -> {host}")
            existing_switch.ip_address = host

        if existing_switch.port != port:
            changes.append(f"Port: {existing_switch.port} -> {port}")
            existing_switch.port = port

        if existing_switch.hostname != hostname:
            changes.append(f"Hostname: '{existing_switch.hostname}' -> '{hostname}'")
            existing_switch.hostname = hostname

        if mac and existing_switch.mac_address != mac:
            changes.append(f"MAC: {existing_switch.mac_address} -> {mac}")
            existing_switch.mac_address = mac

        # Update other fields that might have changed
        firmware = getattr(device, 'firmware_version', None)
        if firmware and existing_switch.firmware_version != firmware:
            changes.append(f"Firmware: {existing_switch.firmware_version} -> {firmware}")
            existing_switch.firmware_version = firmware

        # Update name if it changed
        name = getattr(device, 'name', None)
        if name and existing_switch.name != name:
            changes.append(f"Name: '{existing_switch.name}' -> '{name}'")
            existing_switch.name = name

        if changes:
            existing_switch.save()
            return changes
        return None

    def create_wemo_switch(self, device):
        """Create a WemoSwitch instance from discovered device."""
        host = getattr(device, "host", None)
        port = getattr(device, "port", None)
        name = getattr(device, 'name', None)

        if not host or not name:
            self.stdout.write(
                self.style.WARNING(f'Skipping device - missing required host or name')
            )
            return None

        # Get hostname
        hostname = self.safe_gethost(host)

        # Create the switch object
        switch = WemoSwitch(
            name=name,
            hostname=hostname,
            ip_address=host,
            port=port,
            model=self.get_attr_any(device, 'model', 'model_name'),
            model_name=getattr(device, 'model_name', None),
            serial_number=self.get_attr_any(device, 'serial_number', 'serial'),
            udn=getattr(device, 'udn', None),
            mac_address=getattr(device, 'mac', None),
            manufacturer=getattr(device, 'manufacturer', None),
            firmware_version=getattr(device, 'firmware_version', None),
        )

        return switch

    def handle(self, *args, **options):
        self.stdout.write("Discovering Wemo devices...")

        try:
            devices = pywemo.discover_devices()
        except Exception as e:
            raise CommandError(f"Failed to discover devices: {e}")

        if not devices:
            self.stdout.write(self.style.WARNING("No Wemo devices discovered on the network."))
            return

        self.stdout.write(f"Found {len(devices)} device(s)")

        new_devices = []
        existing_count = 0
        updated_count = 0

        for device in devices:
            if options['verbose']:
                self.stdout.write(f"\nProcessing: {getattr(device, 'name', 'Unknown')}")
                self.stdout.write(f"  IP: {getattr(device, 'host', 'Unknown')}")
                self.stdout.write(f"  UDN: {getattr(device, 'udn', 'Unknown')}")
                self.stdout.write(
                    f"  Serial: {self.get_attr_any(device, 'serial_number', 'serial', default='Unknown')}")

            # Check if device already exists
            exists, match_type, existing_switch = self.device_exists(device)
            if exists:
                # Check if we need to update network information
                if existing_switch:
                    changes = self.update_existing_device(existing_switch, device)
                    if changes:
                        updated_count += 1
                        if options['verbose']:
                            self.stdout.write(f"  Status: Updated ({', '.join(changes)})")
                    else:
                        existing_count += 1
                        if options['verbose']:
                            self.stdout.write(f"  Status: No changes needed (matched by {match_type})")
                else:
                    existing_count += 1
                    if options['verbose']:
                        self.stdout.write(f"  Status: Already exists (matched by {match_type})")
                continue

            # Create new device
            switch = self.create_wemo_switch(device)
            if switch:
                new_devices.append(switch)
                if options['verbose']:
                    self.stdout.write(f"  Status: Will be added")

        # Summary
        self.stdout.write(f"\nSummary:")
        self.stdout.write(f"  Existing devices: {existing_count}")
        self.stdout.write(f"  Updated devices: {updated_count}")
        self.stdout.write(f"  New devices to add: {len(new_devices)}")

        if options['dry_run']:
            self.stdout.write(self.style.WARNING("\nDRY RUN - No changes were actually made"))
            if new_devices:
                self.stdout.write("Would add these devices:")
                for switch in new_devices:
                    self.stdout.write(f"  - {switch.name} ({switch.ip_address})")
            return

        # Save new devices
        if new_devices:
            try:
                with transaction.atomic():
                    for switch in new_devices:
                        switch.save()
                        self.stdout.write(
                            self.style.SUCCESS(f"Added: {switch.name} ({switch.ip_address})")
                        )

                self.stdout.write(
                    self.style.SUCCESS(
                        f"\nSuccessfully added {len(new_devices)} new device(s) and updated {updated_count} existing device(s)")
                )
            except Exception as e:
                raise CommandError(f"Failed to save devices: {e}")
        else:
            self.stdout.write("No new devices to add.")

        # Update last_seen for existing devices (optional)
        if existing_count > 0:
            self.stdout.write(f"Updated last_seen timestamp for {existing_count} existing device(s)")

        self.stdout.write(self.style.SUCCESS("Discovery complete!"))