---
layout: post
title: Practical SSH Port Forwarding Scenarios
date: 2014-05-27 00:33:00 +0800
categories: [tooling, networking]
tags: [networking, cli, security]
---

SSH port forwarding is a powerful tool for securely tunneling network traffic. Here are three common types—local (`-L`), remote (`-R`), and dynamic (`-D`)—explained with practical use cases.

## 1. Local Forwarding (`-L`)

**Usage:**

```bash
ssh user@remote_host -L 8080:target_host:80
```

**Scenario:**

You’re at home and need to access an internal web service (`target_host:80`) at your workplace. The company server (`remote_host`) can access it, but you can’t connect directly from home.

**How to:**
- Run the above command on your home computer.
- Visit `http://localhost:8080` in your browser. This forwards your request through the SSH tunnel to `target_host:80` at your workplace.

**How it works:**  
The SSH client listens on your local port 8080, forwards traffic through the SSH tunnel to `remote_host`, which then connects to `target_host:80`.

---

## 2. Remote Forwarding (`-R`)

**Usage:**

```bash
ssh user@remote_host -R 9090:localhost:3000
```

**Scenario:**

You’re developing a web service on your home computer (port `3000`) and want colleagues at work to access it via the company server (`remote_host`).

**How to:**
- Run the above command on your home computer.
- Your colleagues can access `http://localhost:9090` on `remote_host` to reach your local service.

**How it works:**  
The SSH server (`remote_host`) listens on port `9090` and forwards incoming traffic through the SSH tunnel to your home computer’s port `3000`.

---

## 3. Dynamic Forwarding (`-D`)

**Usage:**

```bash
ssh user@remote_host -D 1080
```

**Scenario:**

You want to route all your local traffic through `remote_host`—for example, to browse the internet securely or bypass restrictions.

**How to:**
- Run the above command locally.
- Set your browser’s SOCKS5 proxy to `127.0.0.1:1080`.
- All browser traffic is tunneled through the SSH connection to `remote_host`, which then accesses the internet.

**How it works:**  
The SSH client listens on local port `1080`, acting as a SOCKS proxy and dynamically forwarding all traffic through the SSH tunnel.

---

**Summary:**
- `-L`: Local forwarding—access remote internal services from your local machine.
- `-R`: Remote forwarding—let remote hosts access services running on your local machine.
- `-D`: Dynamic forwarding—route all traffic through a remote host as a proxy.
