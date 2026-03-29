import requests
import xml.etree.ElementTree as ET

from django.db import models
from django.utils import timezone


class WemoSwitch(models.Model):
    """Model to represent Wemo smart switches discovered on the network."""

    name = models.CharField(max_length=255, help_text="Device name")
    hostname = models.CharField(max_length=255, null=True, blank=True, help_text="Device hostname")
    ip_address = models.GenericIPAddressField(help_text="Device IP address")
    port = models.IntegerField(null=True, blank=True, help_text="Device port")
    model = models.CharField(max_length=100, null=True, blank=True, help_text="Device model")
    model_name = models.CharField(max_length=100, null=True, blank=True, help_text="Device model name")
    serial_number = models.CharField(max_length=100, null=True, blank=True, help_text="Device serial number",
                                     unique=True)
    udn = models.CharField(max_length=255, null=True, blank=True, help_text="Unique Device Name", unique=True)
    mac_address = models.CharField(max_length=17, null=True, blank=True, help_text="MAC address")
    manufacturer = models.CharField(max_length=100, null=True, blank=True, help_text="Device manufacturer")
    firmware_version = models.CharField(max_length=50, null=True, blank=True, help_text="Firmware version")
    date_added = models.DateTimeField(default=timezone.now, help_text="When this device was added to the database")
    disabled = models.BooleanField(default=False, help_text="Whether this device is disabled")
    last_seen = models.DateTimeField(auto_now=True, help_text="Last time this device was discovered")

    class Meta:
        verbose_name = "Wemo Switch"
        verbose_name_plural = "Wemo Switches"
        ordering = ['-date_added']

    def __str__(self):
        return f"{self.name} ({self.ip_address})"

    def save(self, *args, **kwargs):
        if not self.udn and not self.serial_number:
            raise ValueError("Either UDN or Serial Number must be provided")
        super().save(*args, **kwargs)

    def ping(self):
        try:
            state = self.get_state()
            self.last_seen = timezone.now()
            self.save(update_fields=["last_seen"])
            return state
        except Exception:
            # Device is unreachable, return None to indicate offline status
            return None

    # --------------------
    # Internal SOAP helper
    # --------------------
    def _soap_request(self, service_type, control_url, action, body=""):
        headers = {
            "SOAPACTION": f'"{service_type}#{action}"',
            "Content-Type": 'text/xml; charset="utf-8"',
        }

        envelope = f"""
        <?xml version="1.0" encoding="utf-8"?>
        <s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"
                    s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
            <s:Body>
                <u:{action} xmlns:u="{service_type}">
                    {body}
                </u:{action}>
            </s:Body>
        </s:Envelope>
        """

        url = f"http://{self.ip_address}:{self.port}{control_url}"
        resp = requests.post(url, data=envelope.strip(), headers=headers, timeout=5)
        resp.raise_for_status()
        return resp.text

    # --------------------
    # Switch controls
    # --------------------
    def turn_on(self, notes=""):
        result = self._soap_request(
            "urn:Belkin:service:basicevent:1",
            "/upnp/control/basicevent1",
            "SetBinaryState",
            "<BinaryState>1</BinaryState>")
        SwitchEvent.objects.create(
            event_type='switch_on',
            switch=self,
            notes=notes)
        return result

    def turn_off(self, notes=""):
        result = self._soap_request(
            "urn:Belkin:service:basicevent:1",
            "/upnp/control/basicevent1",
            "SetBinaryState",
            "<BinaryState>0</BinaryState>")
        SwitchEvent.objects.create(
            event_type='switch_off',
            switch=self,
            notes=notes)
        return result

    def get_state(self):
        """
        Fetch the current power state from the Wemo device.

        Returns:
            int: Device state as reported by the Wemo.
                 - 0: Switch is OFF (not providing power).
                 - 1: Switch is ON (providing power).
                 - Other values (e.g., 8): Transitional or model-specific states.

        Raises:
            requests.exceptions.RequestException: If device is unreachable
        """
        try:
            xml = self._soap_request(
                "urn:Belkin:service:basicevent:1",
                "/upnp/control/basicevent1",
                "GetBinaryState"
            )
            root = ET.fromstring(xml)
            state = root.find(".//BinaryState")
            return int(state.text) if state is not None else None
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, ConnectionRefusedError) as e:
            # Re-raise connection-related errors so calling code can handle them
            raise e
        except Exception as e:
            # For other errors (XML parsing, etc.), also raise
            raise e

    # --------------------
    # Device info
    # --------------------
    def get_device_info(self):
        return self._soap_request(
            "urn:Belkin:service:deviceinfo:1",
            "/upnp/control/deviceinfo1",
            "GetDeviceInformation"
        )

    def get_firmware_version(self):
        xml = self._soap_request(
            "urn:Belkin:service:firmwareupdate:1",
            "/upnp/control/firmwareupdate1",
            "GetFirmwareVersion"
        )
        root = ET.fromstring(xml)
        version = root.find(".//FirmwareVersion")
        return version.text if version is not None else None

    # --------------------
    # Rules (timers/schedules)
    # --------------------
    def fetch_rules(self):
        return self._soap_request(
            "urn:Belkin:service:rules:1",
            "/upnp/control/rules1",
            "FetchRules"
        )

    def store_rules(self, rules_xml):
        return self._soap_request(
            "urn:Belkin:service:rules:1",
            "/upnp/control/rules1",
            "StoreRules",
            f"<rules>{rules_xml}</rules>"
        )


class AwayModeSettings(models.Model):
    """Settings for Away Mode - simulates presence when away from home."""

    enabled = models.BooleanField(default=False, help_text="Whether Away Mode is currently active")
    sunset_window_minutes = models.IntegerField(
        default=15,
        help_text="Minutes before/after sunset to randomly turn on lights (total window = 2x this value)"
    )
    off_time_hour = models.IntegerField(default=22, help_text="Hour (24h format) to turn off lights")
    off_time_minute = models.IntegerField(default=30, help_text="Minute to turn off lights")
    off_window_minutes = models.IntegerField(
        default=15,
        help_text="Minutes before/after off time to randomly turn off lights"
    )
    last_sunset_on = models.DateField(null=True, blank=True, help_text="Last date lights were turned on at sunset")
    last_night_off = models.DateField(null=True, blank=True, help_text="Last date lights were turned off at night")

    class Meta:
        verbose_name = "Away Mode Settings"
        verbose_name_plural = "Away Mode Settings"

    def __str__(self):
        return f"Away Mode ({'Enabled' if self.enabled else 'Disabled'})"

    @classmethod
    def get_settings(cls):
        """Get or create the singleton settings instance."""
        settings, created = cls.objects.get_or_create(pk=1)
        return settings


class SwitchAwaySchedule(models.Model):
    """Per-switch scheduled on/off times for away mode each day."""

    switch = models.ForeignKey(
        'WemoSwitch',
        on_delete=models.CASCADE,
        related_name='away_schedules',
        help_text="The switch this schedule applies to"
    )
    date = models.DateField(help_text="The date this schedule is for")
    planned_on_time = models.DateTimeField(
        null=True, blank=True,
        help_text="Planned time to turn this switch on (within sunset window)"
    )
    planned_off_time = models.DateTimeField(
        null=True, blank=True,
        help_text="Planned time to turn this switch off (within off-time window)"
    )
    on_executed = models.BooleanField(default=False, help_text="Whether the on action has been executed")
    off_executed = models.BooleanField(default=False, help_text="Whether the off action has been executed")

    class Meta:
        verbose_name = "Switch Away Schedule"
        verbose_name_plural = "Switch Away Schedules"
        unique_together = [('switch', 'date')]
        ordering = ['date', 'planned_on_time']

    def __str__(self):
        return f"{self.switch.name} - {self.date}"


class SwitchEvent(models.Model):
    """Historical log of switch state changes and away mode events."""

    EVENT_TYPES = [
        ('switch_on', 'Switch Turned On'),
        ('switch_off', 'Switch Turned Off'),
        ('away_mode_on', 'Away Mode Enabled'),
        ('away_mode_off', 'Away Mode Disabled'),
    ]

    event_type = models.CharField(max_length=20, choices=EVENT_TYPES, help_text="Type of event")
    timestamp = models.DateTimeField(default=timezone.now, help_text="When the event occurred", db_index=True)
    switch = models.ForeignKey(
        WemoSwitch,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='events',
        help_text="Related switch (null for away mode events)"
    )
    notes = models.TextField(blank=True, help_text="Additional context (e.g., 'Manual', 'Away Mode', 'Scheduled')")

    class Meta:
        verbose_name = "Switch Event"
        verbose_name_plural = "Switch Events"
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['-timestamp', 'event_type']),
        ]

    def __str__(self):
        if self.switch:
            return f"{self.switch.name}: {self.get_event_type_display()} at {self.timestamp}"
        return f"{self.get_event_type_display()} at {self.timestamp}"