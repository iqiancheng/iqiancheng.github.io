---
layout: post
title: How to Check Charging Power (Watts) on macOS
date: 2023-07-06 00:33:00 +0800
categories: [tooling, hardware]
tags: [tooling, hardware, cli]
---

Want to know exactly how much power your MacBook is drawing from its charger? Here are several ways—ranging from command-line tools to a GUI app—that help you find out the charging wattage on macOS.

---

## 🔧 Method 1: ioreg (CLI)

`ioreg` digs into the system's battery info and pulls out wattage data.

```bash
ioreg -r -n AppleSmartBattery | grep -i "watts"
````

If that doesn’t show anything useful, try this extended version:

```bash
ioreg -w0 -l | grep -i "AppleSmartBattery" -A20 | grep -i "watts"
```

You may see output like:

```
"AppleRawAdapterDetails" = ({
  ...
  "Watts" = 60,
  ...
})
```

⚠️ **Note:** This is **not** the actual power being consumed in real time, but the maximum your charger offers during the USB PD negotiation. **60w** is the maximum power your charger can offer.

---

## 🧠 Method 2: system_profiler

`system_profiler` is macOS’s native way to show detailed hardware reports.

```bash
system_profiler SPPowerDataType
```

Look for:

```
AC Charger Information:
    Connected: Yes
    Wattage (W): 60
```

⚠️ **Note:** Again, this value reflects the charger's *rated output*, not live power draw.

---

## 🖥️ Method 3: GUI App — Powerflow


**Powerflow** shows the **real-time charging wattage** in a visual interface:

1. Install Powerflow:

   ```bash
   brew tap lzt1008/powerflow
   brew install --cask powerflow
   # Remove quarantine attribute to allow execution
   xattr -c '/Applications/Powerflow.app'
   # Open the app
   open /Applications/Powerflow.app'
   ```

2. The app displays **live power draw**, helping you understand charging behavior dynamically.

![screenshot_powerflow](/assets/static/2025/macos_battery_watts_check/screenshot_powerflow_2025.png)

---

## ✅ Summary

| Method            | Type    | Shows Wattage | Notes                          |
| ----------------- | ------- | ------------- | ------------------------------ |
| `ioreg`           | CLI     | ✅             | Advanced battery info          |
| `system_profiler` | CLI     | ✅             | Built-in, readable             |
| Powerflow         | GUI App | ✅             | Simple and visual, easy to use |

---

## ⚙️ What Is USB-PD Handshake?

When you plug in a USB-C charger, your Mac negotiates with it using the **USB Power Delivery (USB-PD)** protocol. This "handshake" determines:

* The **maximum voltage and current** the charger can supply
* The **maximum power** (watts) your Mac is allowed to draw

For example, if a charger advertises:

* Max Voltage = 20V
* Max Current = 3A
  Then negotiated power is 20V × 3A = **60W**

But this **DOES NOT** mean your Mac is **actually drawing** 60W. Real-time power depends on:

* Battery level
* System load
* Thermal management

So, the `ioreg` and `system_profiler` commands show **negotiated** (theoretical) power, while apps like Powerflow display **real-time consumption**.