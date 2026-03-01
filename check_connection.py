"""Quick connectivity & API key check for GetStream."""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    key = os.getenv("STREAM_API_KEY", "")
    secret = os.getenv("STREAM_API_SECRET", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")

    print(f"STREAM_API_KEY    = {key[:6]}...{key[-4:]}" if len(key) > 10 else f"STREAM_API_KEY    = {key!r} (TOO SHORT!)")
    print(f"STREAM_API_SECRET = {secret[:6]}...{secret[-4:]}" if len(secret) > 10 else f"STREAM_API_SECRET = {secret!r} (TOO SHORT!)")
    print(f"GOOGLE_API_KEY    = {google_key[:6]}...{google_key[-4:]}" if len(google_key) > 10 else f"GOOGLE_API_KEY    = {google_key!r} (TOO SHORT!)")
    print()

    # Test 1: Raw HTTP to GetStream
    import httpx
    print("--- Test 1: Raw HTTPS to GetStream ---")
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://chat.stream-io-api.com/api/v2/app",
                            headers={"Authorization": key, "stream-auth-type": "jwt"})
            print(f"  Status: {r.status_code} (401=auth issue but server reachable, 200=OK)")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Test 2: GetStream SDK create client
    print("\n--- Test 2: GetStream SDK client ---")
    try:
        from getstream import Stream
        client = Stream(api_key=key, api_secret=secret)
        print(f"  Client created OK (base_url={getattr(client, 'base_url', 'unknown')})")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Test 3: create_user via SDK
    print("\n--- Test 3: SDK create_user (the step that fails) ---")
    try:
        from getstream import Stream
        client = Stream(api_key=key, api_secret=secret)
        resp = client.update_users({"agent": {"id": "agent", "name": "test"}})
        print(f"  SUCCESS: user created/updated")
    except httpx.ConnectTimeout as e:
        print(f"  CONNECT TIMEOUT: {e}")
        print("  → Your network may be blocking outbound HTTPS to GetStream.")
        print("  → Try: check firewall, VPN, or proxy settings.")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # Test 4: Google Gemini
    print("\n--- Test 4: Google API key ---")
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={google_key}")
            print(f"  Status: {r.status_code} (200=OK, 400/403=bad key)")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

asyncio.run(main())
