# ROBOT IS CHILL
---

This is a self-sustained fork of the "Robot Is You" project by RocketRace.

If you want to see the original project, go here:
https://github.com/RocketRace/robot-is-you#readme

[Support Server](https://discord.gg/ktk8XkAfGD)

---

### Setup
If on Windows, set up WSL for this, or it may get a bit messy.

Windows _is_ kind of supported, but here be dragons if you do that!

Step by step:
1. Clone the repository
2. `pip install -r requirements.txt`
3. Set up auth.py: 
   ```py
   token: str = "<TOKEN>"
   ```
4. Set up webhooks.py:
   ```py
   logging_id: int = <command logging id>
   error_id: int = <error logging id>
   ```
5. Make directory `target/renders/`
6. Configure `config.py`
7. Run the bot
8. Run setup commands (in order)

   | Command | What it does |
   | :------ | :----------- |
   | `loadbaba <path>`| Loads required assets from the game from the path. Must have a copy of the game to do this. |
   | `loaddata`| Loads tile metadata from the files. |
   | `loadworld *`| Loads all maps. |
   | `loadletters`| Slices letters from text objects for custom text. |
   | `loadpalettes`| Loads all palettes1. |

9. Restart the bot

Everything should be working fine!

---

## Licensing

<img width="1144" height="46" alt="image" src="https://github.com/user-attachments/assets/1266af14-4a34-412c-9f0a-8c7cd61e8ac0" />

Everything except for the `data/` folder is licensed under the BSD 3-Clause License. Everything in the `data/custom` folder is licensed under the BSD 3-Clause License. Everything else has all rights reserved to their respective authors. Ask the respective authors about usage rights.
