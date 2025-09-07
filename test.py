import os
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv
load_dotenv()  # add this at the top



async def main():
    GATEWAY_URL = os.getenv("GATEWAY_URL")
    GATEWAY_ACCESS_TOKEN = os.getenv("GATEWAY_ACCESS_TOKEN")

    if not GATEWAY_URL or not GATEWAY_ACCESS_TOKEN:
        raise RuntimeError("❌ Missing GATEWAY_URL or GATEWAY_ACCESS_TOKEN in env")

    print("🔌 Connecting to Gateway:", GATEWAY_URL)

    async with streamablehttp_client(
        GATEWAY_URL, headers={"Authorization": f"Bearer {GATEWAY_ACCESS_TOKEN}"}
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            print("✅ Tools registered with Gateway:")
            for tool in tools_result.tools:
                print(f" - {tool.name}: {tool.description}")

if __name__ == "__main__":
    asyncio.run(main())
