#!.venv/bin/python

import macrosia_glue
import asyncio
import time

def main():
    macrosia_glue.connect_to_db("robot.db")
    macrosia_glue.update_macros()
    ok, res, tb = macrosia_glue.evaluate_sync("[mazegen.display/5/5]", ord("T"), None)
    print(ok)
    print(res)
    print(tb)

if __name__ == "__main__":
    main()
