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
    def turn_on(self):
        return self._soap_request(
            "urn:Belkin:service:basicevent:1",
            "/upnp/control/basicevent1",
            "SetBinaryState",
            "<BinaryState>1</BinaryState>"
        )

    def turn_off(self):
        return self._soap_request(
            "urn:Belkin:service:basicevent:1",
            "/upnp/control/basicevent1",
            "SetBinaryState",
            "<BinaryState>0</BinaryState>"
        )

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