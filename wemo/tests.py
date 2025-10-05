# tests.py
import time
import os
from django.test import TestCase
from django.utils import timezone
from django.core.exceptions import ValidationError
from .models import WemoSwitch
import requests


class WemoSwitchModelTests(TestCase):
    """Tests for WemoSwitch model with real devices and dummy devices."""

    def setUp(self):
        """Set up test fixtures."""
        # Load real device data from environment variables
        self.real_device_data = {
            'name': os.getenv('TEST_WEMO_NAME', 'Test Wemo'),
            'ip_address': os.getenv('TEST_WEMO_IP', '192.168.1.58'),
            'port': int(os.getenv('TEST_WEMO_PORT', '49153')),
            'hostname': os.getenv('TEST_WEMO_HOSTNAME', 'wemo.lan'),
            'serial_number': os.getenv('TEST_WEMO_SERIAL', 'TEST123'),
            'udn': os.getenv('TEST_WEMO_UDN', 'uuid:Test-Device-1'),
            'mac_address': os.getenv('TEST_WEMO_MAC', '58:EF:68:FB:C2:37'),
            'manufacturer': os.getenv('TEST_WEMO_MANUFACTURER', 'Belkin'),
            'model_name': os.getenv('TEST_WEMO_MODEL', 'Socket')
        }

        # Create a real device for testing
        self.real_switch = WemoSwitch.objects.create(**self.real_device_data)

        # Create a dummy (offline) device for testing error handling
        self.dummy_switch = WemoSwitch.objects.create(
            name='Dummy Offline Switch',
            ip_address='192.168.1.254',  # Non-existent IP
            port=49999,
            serial_number='DUMMY123',
            udn='uuid:Dummy-Device-1',
            mac_address='00:00:00:00:00:00'
        )

    def test_model_creation(self):
        """Test basic model creation and string representation."""
        self.assertEqual(str(self.real_switch), f"{self.real_switch.name} ({self.real_switch.ip_address})")
        self.assertFalse(self.real_switch.disabled)
        self.assertIsNotNone(self.real_switch.date_added)

    def test_model_validation(self):
        """Test that model requires either UDN or serial number."""
        with self.assertRaises(ValueError):
            switch = WemoSwitch(
                name='Invalid Switch',
                ip_address='192.168.1.100',
                port=49153
            )
            switch.save()

    def test_unique_constraints(self):
        """Test that UDN and serial_number must be unique."""
        from django.db import IntegrityError, transaction

        # Test 1: Try to create duplicate serial number (should fail)
        with transaction.atomic():
            try:
                duplicate_serial = WemoSwitch(
                    name='Duplicate Serial',
                    ip_address='192.168.1.101',
                    port=49153,
                    serial_number=self.real_switch.serial_number,  # Use actual serial from existing switch
                    udn='uuid:Different-UDN-123'
                )
                duplicate_serial.save()
                self.fail("Expected IntegrityError for duplicate serial_number but none was raised")
            except IntegrityError:
                pass  # Expected behavior

        # Test 2: Try to create duplicate UDN (should fail)
        with transaction.atomic():
            try:
                duplicate_udn = WemoSwitch(
                    name='Duplicate UDN',
                    ip_address='192.168.1.102',
                    port=49153,
                    serial_number='UNIQUE-DIFFERENT-SERIAL-456',
                    udn=self.real_switch.udn  # Use actual UDN from existing switch
                )
                duplicate_udn.save()
                self.fail("Expected IntegrityError for duplicate UDN but none was raised")
            except IntegrityError:
                pass  # Expected behavior

    def test_last_seen_auto_update(self):
        """Test that last_seen updates automatically."""
        original_time = self.real_switch.last_seen
        time.sleep(1)

        self.real_switch.name = "Updated Name"
        self.real_switch.save()

        self.assertGreater(self.real_switch.last_seen, original_time)

    def test_disabled_field_default(self):
        """Test that disabled field defaults to False."""
        self.assertFalse(self.real_switch.disabled)
        self.assertFalse(self.dummy_switch.disabled)

    # --------------------
    # Real Device Tests (require human verification)
    # --------------------

    def test_get_state_real_device(self):
        """Test getting state from a real device."""
        print("\n" + "=" * 60)
        print("TEST: Getting state from real device")
        print("=" * 60)

        try:
            state = self.real_switch.get_state()
            self.assertIsNotNone(state)
            self.assertIn(state, [0, 1, 8])  # Valid states

            print(f"✓ Successfully got state: {state}")
            print(f"  0 = OFF, 1 = ON, 8 = Standby/Other")

            # Ask user to verify
            user_confirm = input(f"\nIs the switch currently {'ON' if state == 1 else 'OFF'}? (y/n): ")
            self.assertEqual(user_confirm.lower(), 'y', "User reported state doesn't match")
            print("✓ User confirmed state is correct")

        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to communicate with real device: {e}")

    def test_turn_on_real_device(self):
        """Test turning on a real device (requires human verification)."""
        print("\n" + "=" * 60)
        print("TEST: Turning ON real device")
        print("=" * 60)

        try:
            # Get initial state
            initial_state = self.real_switch.get_state()
            print(f"Initial state: {initial_state} ({'ON' if initial_state == 1 else 'OFF'})")

            # If already on, turn it off first
            if initial_state == 1:
                print("Device is already ON, turning OFF first...")
                self.real_switch.turn_off()
                time.sleep(2)
                initial_state = self.real_switch.get_state()
                print(f"State after turning off: {initial_state} ({'ON' if initial_state == 1 else 'OFF'})")

            # Now turn on
            print("Turning ON...")
            self.real_switch.turn_on()
            time.sleep(2)  # Give device time to respond

            # Verify new state
            new_state = self.real_switch.get_state()
            print(f"New state: {new_state} ({'ON' if new_state == 1 else 'OFF'})")

            # Verify state changed from initial
            self.assertNotEqual(initial_state, new_state, "State did not change")

            # Ask user to verify
            print("\nPlease check the physical device:")
            user_confirm = input("Is the switch NOW ON (light/device powered)? (y/n): ")
            self.assertEqual(user_confirm.lower(), 'y', "Switch did not turn on")

            # Verify state is 1
            self.assertEqual(new_state, 1, "State should be 1 (ON)")
            print("✓ Switch successfully turned ON")

        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to communicate with real device: {e}")

    def test_turn_off_real_device(self):
        """Test turning off a real device (requires human verification)."""
        print("\n" + "=" * 60)
        print("TEST: Turning OFF real device")
        print("=" * 60)

        try:
            # Get initial state
            initial_state = self.real_switch.get_state()
            print(f"Initial state: {initial_state} ({'ON' if initial_state == 1 else 'OFF'})")

            # If already off, turn it on first
            if initial_state == 0:
                print("Device is already OFF, turning ON first...")
                self.real_switch.turn_on()
                time.sleep(2)
                initial_state = self.real_switch.get_state()
                print(f"State after turning on: {initial_state} ({'ON' if initial_state == 1 else 'OFF'})")

            # Now turn off
            print("Turning OFF...")
            self.real_switch.turn_off()
            time.sleep(2)  # Give device time to respond

            # Verify new state
            new_state = self.real_switch.get_state()
            print(f"New state: {new_state} ({'ON' if new_state == 1 else 'OFF'})")

            # Verify state changed from initial
            self.assertNotEqual(initial_state, new_state, "State did not change")

            # Ask user to verify
            print("\nPlease check the physical device:")
            user_confirm = input("Is the switch NOW OFF (no power to device)? (y/n): ")
            self.assertEqual(user_confirm.lower(), 'y', "Switch did not turn off")

            # Verify state is 0
            self.assertEqual(new_state, 0, "State should be 0 (OFF)")
            print("✓ Switch successfully turned OFF")

        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to communicate with real device: {e}")

    def test_toggle_sequence_real_device(self):
        """Test a complete toggle sequence (requires human verification)."""
        print("\n" + "=" * 60)
        print("TEST: Toggle sequence (ON -> OFF -> ON)")
        print("=" * 60)

        try:
            # Turn on
            print("\n1. Turning ON...")
            self.real_switch.turn_on()
            time.sleep(2)
            state1 = self.real_switch.get_state()
            user_confirm1 = input("Is the switch ON? (y/n): ")
            self.assertEqual(user_confirm1.lower(), 'y')
            self.assertEqual(state1, 1)
            print("✓ Step 1 complete")

            # Turn off
            print("\n2. Turning OFF...")
            self.real_switch.turn_off()
            time.sleep(2)
            state2 = self.real_switch.get_state()
            user_confirm2 = input("Is the switch OFF? (y/n): ")
            self.assertEqual(user_confirm2.lower(), 'y')
            self.assertEqual(state2, 0)
            print("✓ Step 2 complete")

            # Turn on again
            print("\n3. Turning ON again...")
            self.real_switch.turn_on()
            time.sleep(2)
            state3 = self.real_switch.get_state()
            user_confirm3 = input("Is the switch ON? (y/n): ")
            self.assertEqual(user_confirm3.lower(), 'y')
            self.assertEqual(state3, 1)
            print("✓ Step 3 complete")

            print("\n✓ Complete toggle sequence successful!")

        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to communicate with real device: {e}")

    def test_ping_real_device(self):
        """Test ping method updates last_seen."""
        print("\n" + "=" * 60)
        print("TEST: Ping method")
        print("=" * 60)

        try:
            original_time = self.real_switch.last_seen
            time.sleep(1)

            state = self.real_switch.ping()

            self.assertIsNotNone(state)
            self.assertGreater(self.real_switch.last_seen, original_time)
            print(f"✓ Ping successful, state: {state}")
            print(f"✓ last_seen updated from {original_time} to {self.real_switch.last_seen}")

        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to communicate with real device: {e}")

    # --------------------
    # Offline/Dummy Device Tests
    # --------------------

    def test_get_state_offline_device(self):
        """Test that offline device raises appropriate exception."""
        print("\n" + "=" * 60)
        print("TEST: Offline device - get_state")
        print("=" * 60)

        with self.assertRaises(requests.exceptions.RequestException):
            self.dummy_switch.get_state()

        print("✓ Correctly raised exception for offline device")

    def test_turn_on_offline_device(self):
        """Test that turning on offline device raises exception."""
        print("\n" + "=" * 60)
        print("TEST: Offline device - turn_on")
        print("=" * 60)

        with self.assertRaises(requests.exceptions.RequestException):
            self.dummy_switch.turn_on()

        print("✓ Correctly raised exception for offline device")

    def test_turn_off_offline_device(self):
        """Test that turning off offline device raises exception."""
        print("\n" + "=" * 60)
        print("TEST: Offline device - turn_off")
        print("=" * 60)

        with self.assertRaises(requests.exceptions.RequestException):
            self.dummy_switch.turn_off()

        print("✓ Correctly raised exception for offline device")

    def test_ping_offline_device(self):
        """Test that ping returns None for offline device."""
        print("\n" + "=" * 60)
        print("TEST: Offline device - ping")
        print("=" * 60)

        result = self.dummy_switch.ping()

        self.assertIsNone(result)
        print("✓ Ping correctly returned None for offline device")

    def test_multiple_offline_attempts(self):
        """Test that multiple attempts to reach offline device fail gracefully."""
        print("\n" + "=" * 60)
        print("TEST: Multiple attempts on offline device")
        print("=" * 60)

        for i in range(3):
            with self.assertRaises(requests.exceptions.RequestException):
                self.dummy_switch.get_state()
            print(f"  Attempt {i + 1}: ✓ Correctly raised exception")

        print("✓ All attempts handled gracefully")

    # --------------------
    # Additional Info Methods Tests
    # --------------------

    def test_get_device_info_real_device(self):
        """Test getting device info from real device."""
        print("\n" + "=" * 60)
        print("TEST: Get device info")
        print("=" * 60)

        try:
            info = self.real_switch.get_device_info()
            self.assertIsNotNone(info)
            print("✓ Successfully retrieved device info")
            print(f"  Info length: {len(info)} characters")

        except requests.exceptions.RequestException as e:
            print(f"⚠ Warning: Could not get device info: {e}")
            # Don't fail the test as not all devices support this

    def test_get_firmware_version_real_device(self):
        """Test getting firmware version from real device."""
        print("\n" + "=" * 60)
        print("TEST: Get firmware version")
        print("=" * 60)

        try:
            version = self.real_switch.get_firmware_version()
            if version:
                print(f"✓ Firmware version: {version}")
            else:
                print("⚠ Warning: No firmware version returned")

        except requests.exceptions.RequestException as e:
            print(f"⚠ Warning: Could not get firmware version: {e}")
            # Don't fail the test as not all devices support this


class WemoSwitchQueryTests(TestCase):
    """Tests for querying WemoSwitch models."""

    def setUp(self):
        """Create test switches."""
        self.enabled_switch = WemoSwitch.objects.create(
            name='Enabled Switch',
            ip_address='192.168.1.100',
            port=49153,
            serial_number='ENABLED123',
            udn='uuid:Enabled-Device',
            disabled=False
        )

        self.disabled_switch = WemoSwitch.objects.create(
            name='Disabled Switch',
            ip_address='192.168.1.101',
            port=49153,
            serial_number='DISABLED123',
            udn='uuid:Disabled-Device',
            disabled=True
        )

    def test_filter_enabled_switches(self):
        """Test filtering for enabled switches only."""
        enabled = WemoSwitch.objects.filter(disabled=False)
        self.assertEqual(enabled.count(), 1)
        self.assertEqual(enabled.first(), self.enabled_switch)

    def test_filter_disabled_switches(self):
        """Test filtering for disabled switches."""
        disabled = WemoSwitch.objects.filter(disabled=True)
        self.assertEqual(disabled.count(), 1)
        self.assertEqual(disabled.first(), self.disabled_switch)

    def test_ordering(self):
        """Test default ordering by date_added."""
        all_switches = WemoSwitch.objects.all()
        self.assertEqual(all_switches[0], self.disabled_switch)  # Created second, appears first