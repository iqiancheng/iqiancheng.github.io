#!/usr/bin/env ruby
# Convert Clash Verge config -> mihomo NAS config.
# Node details stay in-memory and are written only to the output file — never printed.
require 'yaml'

SRC = File.expand_path('~/.config/clash-verge/clash-verge.yaml')
DST = '/tmp/mihomo_config.yaml'

config = YAML.safe_load(File.read(SRC))

# --- Adjustments for NAS mihomo ---
config['allow-lan']             = true
config['external-controller']   = '0.0.0.0:9090'
config['bind-address']          = '*'
config['ipv6']                  = false

# mihomo-specific enhancements (compatible superset of clash)
config['tcp-concurrent']        = true
config['unified-delay']         = true
config['geodata-mode']          = true
config['sniffer']               = { 'enable' => true, 'sniffing' => %w[tls http] }
config['profile']               = { 'store-selected' => true }

# Quieter log level on NAS
config['log-level']             = 'warning'

# Remove secret if present (dashboard auth — we rely on iptables instead)
config.delete('secret')

# --- Counts (safe to print) ---
proxy_count       = config['proxies']&.length       || 0
group_count       = config['proxy-groups']&.length  || 0
rule_count        = config['rules']&.length         || 0
dns_keys          = config['dns']&.keys&.join(', ') || 'none'
tun_enabled       = config.dig('tun', 'enable')     || false

# Write config (node details go to file only, never to stdout)
File.write(DST, YAML.dump(config))
puts "Config written to #{DST}"
puts "  proxies:      #{proxy_count} nodes"
puts "  proxy-groups: #{group_count} groups"
puts "  rules:        #{rule_count} rules"
puts "  dns keys:     #{dns_keys}"
puts "  tun:          #{tun_enabled ? 'enabled' : 'disabled'}"
puts "  allow-lan:    true (for LAN device access)"
puts "  controller:   0.0.0.0:9090"
