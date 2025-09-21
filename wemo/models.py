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