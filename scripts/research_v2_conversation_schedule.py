def prefill(route, prompt, down, up, mode, progress=lambda *args: None):
    if not 0 < down < up < len(prompt):
        raise ValueError("Invalid conversational boundaries")
    if mode not in ("q4", "q3", "retained", "rebuild"):
        raise ValueError("Invalid treatment")
    route.switch(int(mode == "q3"), "initial")
    for pos, token in enumerate(prompt):
        if mode in ("retained", "rebuild") and pos in (down, up):
            route.switch(int(pos == down), "down" if pos == down else "up")
            if mode == "rebuild":
                route.event("rebuild_start", position=pos)
                for previous, t in enumerate(prompt[:pos]):
                    route.token(previous, t)
                    if previous % 64 == 0:
                        progress("rebuild", previous, pos)
                route.event("rebuild_end", position=pos)
        value = route.token(pos, token)
        if pos % 64 == 0:
            progress("prefill", pos, len(prompt))
    return value
