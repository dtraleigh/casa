# Wemo UPnP Services Reference

This document summarizes available services/actions for Wemo devices.
You can extend the `WemoSwitch` class using these.

---

## basicevent (urn:Belkin:service:basicevent:1)
- **SetBinaryState**: Turn plug on/off (`BinaryState=1` / `0`)
- **GetBinaryState**: Query current state
- **GetFriendlyName**: Return user-friendly device name
- **GetMacAddr**: Return MAC address
- **GetSerialNo**: Return device serial
- **GetSignalStrength**: WiFi signal info

---

## deviceinfo (urn:Belkin:service:deviceinfo:1)
- **GetDeviceInformation**: Returns XML with hardware model, manufacturer, UUID, etc.
- **GetInformation**: Alternative device details
- **GetRouterInformation**: Router connection details

---

## firmwareupdate (urn:Belkin:service:firmwareupdate:1)
- **GetFirmwareVersion**: Query firmware version
- **UpdateFirmware**: Trigger firmware update

---

## rules (urn:Belkin:service:rules:1)
- **FetchRules**: Retrieve schedule database (timers)
- **StoreRules**: Save/update schedule database
- **UpdateWeeklyCalendar**: Update weekly schedule
- **EditWeeklyCalendar**: Edit schedule entries

---

## timesync (urn:Belkin:service:timesync:1)
- **GetTime**: Query current device time
- **TimeSync**: Sync device time with host

---

## wifiSetup (urn:Belkin:service:WiFiSetup:1)
- **GetNetworkList**: Scan WiFi networks
- **ConnectHomeNetwork**: Connect to new SSID

---

## smartsetup / manufacture / metainfo
Additional low-level provisioning or diagnostic services.
Rarely needed for normal use.
