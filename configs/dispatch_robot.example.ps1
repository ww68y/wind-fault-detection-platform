# Robot dispatch configuration template.
# Copy this file to a local-only script and fill in your real webhook.

# Supported values: "wecom" or "dingtalk".
$env:WIND_DISPATCH_PROVIDER = "wecom"

# WeCom group robot webhook.
$env:WIND_WECOM_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY"

# DingTalk group robot webhook. Use this instead when WIND_DISPATCH_PROVIDER="dingtalk".
# $env:WIND_DINGTALK_WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN"

# Optional DingTalk signing secret.
# $env:WIND_DINGTALK_SECRET = "SEC..."

# Network timeout in seconds.
$env:WIND_DISPATCH_TIMEOUT = "15"

# Public URL used in robot message links. For a LAN demo, replace it with
# this machine's reachable IP, for example http://192.168.1.20:8000.
$env:WIND_PUBLIC_BASE_URL = "http://127.0.0.1:8000"
