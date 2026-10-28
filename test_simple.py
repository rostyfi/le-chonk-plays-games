#!/usr/bin/env python3
"""
Minimal test: screenshot → mistral-large-4 → print response.
No parsing, no actions, no complexity. Just see what the AI says.
"""

import base64
import json
import os
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screenshot import take_screenshot

API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-4"

api_key = os.environ.get('MISTRAL_API_KEY', '')
if not api_key:
    print("Error: Set MISTRAL_API_KEY first.")
    sys.exit(1)

# 1. Screenshot
print("[1/3] Taking screenshot...")
shot_path = take_screenshot('/tmp/minimal_test.png')
print(f"      Saved: {shot_path} ({os.path.getsize(shot_path)} bytes)")

# 2. Encode
with open(shot_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')

# 3. Call API — dead simple prompt
print(f"[2/3] Calling {MODEL} with simple prompt...")
payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What would be the best move for this game?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
    "max_tokens": 1024,
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(
    API_URL,
    data=data,
    headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    },
    method='POST',
)

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    body = e.read().decode('utf-8')
    print(f"\nHTTP {e.code} error:")
    print(body[:2000])
    sys.exit(1)
except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)

# 4. Print response
print("[3/3] Response:\n")
print("=" * 60)

choices = result.get('choices', [])
if not choices:
    print("No choices in response:")
    print(json.dumps(result, indent=2)[:2000])
    sys.exit(1)

msg = choices[0]['message']
content = msg.get('content', '')

if isinstance(content, str):
    print(content)
elif isinstance(content, list):
    for part in content:
        if isinstance(part, dict) and part.get('type') == 'text':
            print(part.get('text', ''))
        elif isinstance(part, str):
            print(part)
else:
    print(str(content))

print("=" * 60)

# Also print usage info if available
usage = result.get('usage', {})
if usage:
    print(f"\nTokens: prompt={usage.get('prompt_tokens', '?')}, "
          f"completion={usage.get('completion_tokens', '?')}, "
          f"total={usage.get('total_tokens', '?')}")
