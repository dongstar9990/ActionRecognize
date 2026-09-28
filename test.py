import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/recognize/ws") as ws:
        await ws.send(json.dumps({
            "mode": "fuzzy",
            "threshold": 80,
            "stop_on_match": True,
            "chunk_suffix": ".wav",
            "language": "vi"
        }))

        for chunk in ["chunks/chunk_000.wav", "chunks/chunk_001.wav", "chunks/chunk_002.wav"]:
            with open(chunk, "rb") as f:
                await ws.send(f.read())

            result = json.loads(await ws.recv())
            print(result)

            if result.get("matched"):
                break

asyncio.run(main())