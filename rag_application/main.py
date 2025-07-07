import requests

token = "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" # your real token here
res = requests.get("https://huggingface.co/api/whoami", headers={
    "Authorization": f"Bearer {token}"
})

print("Status code:", res.status_code)
print("Response:", res.text)
