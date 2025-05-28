import asqlite
from PIL import Image
import io
import requests
import traceback
import builtins


f = open("stdout.log", "w+")

_print = print

def print(*args, **kwargs):
    _print(*args, **kwargs)
    f.write(" ".join(str(arg) for arg in args))
    f.write("\n")
    f.flush()

class Database:
    """Everything relating to persistent readable & writable data."""
    conn: asqlite.Connection
    filter_cache: dict[str, (Image.Image, bool)]

    async def connect(self, db: str) -> None:
        """Startup."""
        # not checking for same thread probably is a terrible idea but
        # whateverrr
        self.conn = await asqlite.connect(db, check_same_thread=False)

        def regexp(x, y):
            return bool(re.search(x, y))

        self.conn.get_connection().create_function('regexp', 2, regexp)
        print("Initialized database connection.")

    async def close(self) -> None:
        """Teardown."""
        if hasattr(self, "conn"):
            await self.conn.close()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

async def main():
    print("Connecting...")
    db = Database()
    await db.connect("../robot.db")
    print("Connected!")

    try:
        filters = {}
        with open("sql_output.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                name, absolute, url, author = (*(val.strip() for val in line[1:-1].split("|")), )
                if url.endswith("gif"): continue
                print(f"[.] Porting {name} ({url})...")
                buf = io.BytesIO()
                try:
                    res = requests.get(url, headers = headers, stream = True)
                    res.raise_for_status()
                    with Image.open(res.raw) as im:
                        width, height = im.size
                        assert width <= 120 and height <= 120, "Image is too large!"
                        im.convert("RGBA").save(buf, format = "PNG")
                    buf.seek(0)
                    async with db.conn.cursor() as cursor:
                        await cursor.execute(
                            "INSERT INTO filters VALUES (?, ?, ?, ?, ?);",
                            name, int(absolute) > 0, int(author), None, buf.getvalue()
                        )
                except Exception as err:
                    print(f"[!] {err.__class__.__name__}: {err}")
                    continue


    finally:
        await db.close()

import asyncio

if __name__ == "__main__":
    asyncio.run(main())
