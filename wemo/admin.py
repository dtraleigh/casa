from django.contrib import admin
from django.utils.html import format_html
from .models import WemoSwitch, AwayModeSettings


@admin.register(AwayModeSettings)
class AwayModeSettingsAdmin(admin.ModelAdmin):
    list_display = ['__str__', 'enabled', 'sunset_window_minutes', 'off_time_display', 'last_activity']

    fieldsets = (
        ('Status', {
            'fields': ('enabled',)
        }),
        ('Sunset Settings', {
            'fields': ('sunset_window_minutes',),
            'description': 'Lights will turn on randomly within this window around sunset'
        }),
        ('Night Off Settings', {
            'fields': ('off_time_hour', 'off_time_minute', 'off_window_minutes'),
            'description': 'Lights will turn off randomly within this window around the specified time'
        }),
        ('Activity Log', {
            'fields': ('last_sunset_on', 'last_night_off'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ['last_sunset_on', 'last_night_off']

    def off_time_display(self, obj):
        """Display the off time in readable format."""
        return f"{obj.off_time_hour:02d}:{obj.off_time_minute:02d}"

    off_time_display.short_description = 'Off Time'

    def last_activity(self, obj):
        """Display last activity."""
        activities = []
        if obj.last_sunset_on:
            activities.append(f"Sunset: {obj.last_sunset_on}")
        if obj.last_night_off:
            activities.append(f"Night: {obj.last_night_off}")
        return " | ".join(activities) if activities else "No activity yet"

    last_activity.short_description = 'Last Activity'

    def has_add_permission(self, request):
        # Only allow one instance
        return not AwayModeSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        # Don't allow deletion
        return False


@admin.register(WemoSwitch)
class WemoSwitchAdmin(admin.ModelAdmin):
    list_display = [
        'name',
        'ip_address',
        'hostname',
        'model_name',
        'status_badge',
        'date_added',
        'last_seen'
    ]

    list_filter = [
        'disabled',
        'manufacturer',
        'model_name',
        'date_added',
        'last_seen'
    ]

    search_fields = [
        'name',
        'ip_address',
        'hostname',
        'serial_number',
        'udn',
        'mac_address'
    ]

    readonly_fields = [
        'date_added',
        'last_seen',
        'udn',
        'serial_number',
        'device_info_display'
    ]

    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'disabled', 'date_added', 'last_seen')
        }),
        ('Network Information', {
            'fields': ('ip_address', 'port', 'hostname', 'mac_address')
        }),
        ('Device Information', {
            'fields': ('manufacturer', 'model', 'model_name', 'firmware_version', 'device_info_display'),
            'classes': ('collapse',)
        }),
        ('Identifiers', {
            'fields': ('serial_number', 'udn'),
            'classes': ('collapse',)
        }),
    )

    list_per_page = 25
    ordering = ['-last_seen']

    def status_badge(self, obj):
        """Display a colored status badge."""
        if obj.disabled:
            return format_html(
                '<span style="color: white; background-color: #dc3545; padding: 2px 8px; border-radius: 3px;">Disabled</span>'
            )
        else:
            # Check if device is reachable
            try:
                state = obj.get_state()
                if state == 1:
                    return format_html(
                        '<span style="color: white; background-color: #28a745; padding: 2px 8px; border-radius: 3px;">Active (ON)</span>'
                    )
                else:
                    return format_html(
                        '<span style="color: white; background-color: #28a745; padding: 2px 8px; border-radius: 3px;">Active (OFF)</span>'
                    )
            except Exception:
                return format_html(
                    '<span style="color: white; background-color: #ffc107; padding: 2px 8px; border-radius: 3px; color: black;">Offline</span>'
                )

    status_badge.short_description = 'Status'

    def device_info_display(self, obj):
        """Display formatted device information."""
        info_parts = []

        if obj.manufacturer:
            info_parts.append(f"<strong>Manufacturer:</strong> {obj.manufacturer}")
        if obj.model:
            info_parts.append(f"<strong>Model:</strong> {obj.model}")
        if obj.firmware_version:
            info_parts.append(f"<strong>Firmware:</strong> {obj.firmware_version}")
        if obj.serial_number:
            info_parts.append(f"<strong>Serial:</strong> {obj.serial_number}")
        if obj.udn:
            info_parts.append(f"<strong>UDN:</strong> <code>{obj.udn}</code>")

        if info_parts:
            return format_html("<br>".join(info_parts))
        return "No additional device information available"

    device_info_display.short_description = 'Device Details'

    actions = ['enable_devices', 'disable_devices']

    def enable_devices(self, request, queryset):
        """Enable selected devices."""
        updated = queryset.update(disabled=False)
        self.message_user(
            request,
            f'{updated} device(s) were successfully enabled.'
        )

    enable_devices.short_description = "Enable selected devices"

    def disable_devices(self, request, queryset):
        """Disable selected devices."""
        updated = queryset.update(disabled=True)
        self.message_user(
            request,
            f'{updated} device(s) were successfully disabled.'
        )

    disable_devices.short_description = "Disable selected devices"

    def get_queryset(self, request):
        """Optimize queryset for admin display."""
        return super().get_queryset(request).select_related()