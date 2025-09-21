from django.contrib import admin
from django.utils.html import format_html
from .models import WemoSwitch


@admin.register(WemoSwitch)
class WemoSwitchAdmin(admin.ModelAdmin):
    list_display = [
        'name',
        'ip_address',
        'hostname',
        'model_name',
        'status_badge',
        'live_status',
        'power_state',
        'live_status',
        'power_state',
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
        """Display a colored status badge based on the disabled flag."""
        if obj.disabled:
            return format_html(
                '<span style="color: white; background-color: #dc3545; padding: 2px 8px; border-radius: 3px;">Disabled</span>'
            )
        else:
            return format_html(
                '<span style="color: white; background-color: #28a745; padding: 2px 8px; border-radius: 3px;">Enabled</span>'
            )

    status_badge.short_description = 'Status'

    def live_status(self, obj):
        """Show if device is reachable."""
        if obj.disabled:
            return format_html(
                '<span style="color: white; background-color: #6c757d; padding: 2px 8px; border-radius: 3px;">Disabled</span>'
            )

        state = obj.ping()  # <-- updates last_seen automatically
        if state is not None:
            return format_html(
                '<span style="color: white; background-color: #28a745; padding: 2px 8px; border-radius: 3px;">Online</span>'
            )
        else:
            return format_html(
                '<span style="color: white; background-color: #dc3545; padding: 2px 8px; border-radius: 3px;">Offline</span>'
            )

    live_status.short_description = 'Live Status'

    def power_state(self, obj):
        """Show ON/OFF state if reachable."""
        if obj.disabled:
            return "-"
        state = obj.ping()  # <-- updates last_seen automatically
        if state is None:
            return "Unknown"
        return "On" if state == 1 else "Off"

    power_state.short_description = 'Power'

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
