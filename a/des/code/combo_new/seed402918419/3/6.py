from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
FRAME = 5
ARROW = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = ("N", "E", "S", "W")

GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
PAIR_COUNT_RANGE = (1, 3)
RELAY_COUNT_RANGE = (0, 3)
TRAVEL_MIN = 2
OUTER_ATTEMPTS = 400
PAIR_ATTEMPTS = 300
PADDING = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each 2x2 source block along its arrow-directed path into the matching target frame.

    Input structure:
    - Background is black.
    - Several colored 2x2 solid squares are sources, each adjacent to one maroon L-shaped arrow.
    - Several grey 4x4 hollow squares with black 2x2 interiors are target frames.
    - Along the straight lane indicated by an arrow, zero or more single-pixel relay stations may appear.

    Rule:
    - For each source, read the adjacent arrow to determine the travel direction.
    - Follow that direction to the aligned target frame, collecting relay stations in encounter order.
    - The source color changes to each relay's color in turn; if there are no relays, the original source color is kept.
    - Remove the source, its arrow, and all relay stations, then fill the target frame's 2x2 interior with the final color.

    Key insight:
    - The arrow identifies which straight lane to trace, while the ordered relay colors independently determine the final fill color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    arrows = _find_arrows(g)
    sources = _find_sources(g)

    for source in sources:
        touching = [arrow for arrow in arrows if _touching(source["cells"], arrow["cells"])]
        assert len(touching) == 1, "each source must touch exactly one arrow"
        arrow = touching[0]

        sr, sc = source["top"], source["left"]
        frame = _find_target_frame(frames, sr, sc, arrow["direction"])
        relays = _relay_sequence(g, sr, sc, frame["top"], frame["left"], arrow["direction"])
        final_color = relays[-1][1] if relays else source["color"]

        for r, c in source["cells"]:
            out[r, c] = BG
        for r, c in arrow["cells"]:
            out[r, c] = BG
        for (r, c), _color in relays:
            out[r, c] = BG

        fr, fc = frame["top"], frame["left"]
        out[fr + 1 : fr + 3, fc + 1 : fc + 3] = final_color

    return out


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for comp in _color_components(grid, FRAME):
        if len(comp) != 12:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1][[0, 0, 0, 1, 2, 3, 3, 3], [0, 1, 2, 3, 3, 3, 2, 1]] == FRAME):
            continue
        if not np.all(grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] == BG):
            continue
        frames.append({"top": r0, "left": c0})
    return frames


def _find_sources(grid: np.ndarray) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, FRAME, ARROW):
            continue
        for comp in _color_components(grid, color):
            if len(comp) != 4:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            cells = {(r, c) for r, c in comp}
            sources.append({"color": color, "top": r0, "left": c0, "cells": cells})
    return sources


def _arrow_direction(comp: list[tuple[int, int]]) -> str:
    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for r, c in comp:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)

    for r, values in rows.items():
        if len(values) == 3:
            rod_cols = sorted(values)
            wing = next((rr, cc) for rr, cc in comp if rr != r)
            if wing[1] == rod_cols[0]:
                return "E"
            if wing[1] == rod_cols[-1]:
                return "W"
    for c, values in cols.items():
        if len(values) == 3:
            rod_rows = sorted(values)
            wing = next((rr, cc) for rr, cc in comp if cc != c)
            if wing[0] == rod_rows[0]:
                return "S"
            if wing[0] == rod_rows[-1]:
                return "N"
    raise AssertionError("invalid arrow shape")


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _color_components(grid, ARROW):
        if len(comp) != 4:
            continue
        direction = _arrow_direction(comp)
        arrows.append({"direction": direction, "cells": {(r, c) for r, c in comp}})
    return arrows


def _touching(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _find_target_frame(frames: list[dict[str, int]], sr: int, sc: int, direction: str) -> dict[str, int]:
    if direction == "E":
        candidates = [f for f in frames if f["top"] == sr - 1 and f["left"] > sc]
        assert candidates, "missing target frame to the east"
        return min(candidates, key=lambda f: f["left"])
    if direction == "W":
        candidates = [f for f in frames if f["top"] == sr - 1 and f["left"] < sc]
        assert candidates, "missing target frame to the west"
        return max(candidates, key=lambda f: f["left"])
    if direction == "S":
        candidates = [f for f in frames if f["left"] == sc - 1 and f["top"] > sr]
        assert candidates, "missing target frame to the south"
        return min(candidates, key=lambda f: f["top"])
    candidates = [f for f in frames if f["left"] == sc - 1 and f["top"] < sr]
    assert candidates, "missing target frame to the north"
    return max(candidates, key=lambda f: f["top"])


def _relay_sequence(
    grid: np.ndarray,
    sr: int,
    sc: int,
    fr: int,
    fc: int,
    direction: str,
) -> list[tuple[tuple[int, int], int]]:
    relays: list[tuple[tuple[int, int], int]] = []

    if direction == "E":
        rows = (sr, sr + 1)
        for c in range(sc + 2, fc):
            hits = [(r, c) for r in rows if int(grid[r, c]) not in (BG, FRAME, ARROW)]
            assert len(hits) <= 1, "relay order must be unambiguous"
            if hits:
                relays.append((hits[0], int(grid[hits[0]])))
        return relays

    if direction == "W":
        rows = (sr, sr + 1)
        for c in range(sc - 1, fc + 3, -1):
            hits = [(r, c) for r in rows if int(grid[r, c]) not in (BG, FRAME, ARROW)]
            assert len(hits) <= 1, "relay order must be unambiguous"
            if hits:
                relays.append((hits[0], int(grid[hits[0]])))
        return relays

    if direction == "S":
        cols = (sc, sc + 1)
        for r in range(sr + 2, fr):
            hits = [(r, c) for c in cols if int(grid[r, c]) not in (BG, FRAME, ARROW)]
            assert len(hits) <= 1, "relay order must be unambiguous"
            if hits:
                relays.append((hits[0], int(grid[hits[0]])))
        return relays

    cols = (sc, sc + 1)
    for r in range(sr - 1, fr + 3, -1):
        hits = [(r, c) for c in cols if int(grid[r, c]) not in (BG, FRAME, ARROW)]
        assert len(hits) <= 1, "relay order must be unambiguous"
        if hits:
            relays.append((hits[0], int(grid[hits[0]])))
    return relays


def _source_cells(sr: int, sc: int) -> list[tuple[int, int]]:
    return [(sr, sc), (sr, sc + 1), (sr + 1, sc), (sr + 1, sc + 1)]


def _frame_cells(fr: int, fc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(fr, fr + 4):
        for c in range(fc, fc + 4):
            if r in (fr, fr + 3) or c in (fc, fc + 3):
                cells.append((r, c))
    return cells


def _arrow_cells(sr: int, sc: int, direction: str, variant: int) -> list[tuple[int, int]]:
    if direction == "E":
        patterns = [
            [(sr, sc - 3), (sr, sc - 2), (sr, sc - 1), (sr + 1, sc - 3)],
            [(sr + 1, sc - 3), (sr + 1, sc - 2), (sr + 1, sc - 1), (sr, sc - 3)],
        ]
        return patterns[variant]
    if direction == "W":
        patterns = [
            [(sr, sc + 2), (sr, sc + 3), (sr, sc + 4), (sr + 1, sc + 4)],
            [(sr + 1, sc + 2), (sr + 1, sc + 3), (sr + 1, sc + 4), (sr, sc + 4)],
        ]
        return patterns[variant]
    if direction == "S":
        patterns = [
            [(sr - 3, sc), (sr - 2, sc), (sr - 1, sc), (sr - 3, sc + 1)],
            [(sr - 3, sc + 1), (sr - 2, sc + 1), (sr - 1, sc + 1), (sr - 3, sc)],
        ]
        return patterns[variant]
    patterns = [
        [(sr + 2, sc), (sr + 3, sc), (sr + 4, sc), (sr + 4, sc + 1)],
        [(sr + 2, sc + 1), (sr + 3, sc + 1), (sr + 4, sc + 1), (sr + 4, sc)],
    ]
    return patterns[variant]


def _reserve_ok(occupied: np.ndarray, cells: list[tuple[int, int]], h: int, w: int) -> bool:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = max(0, min(rows) - PADDING)
    c0 = max(0, min(cols) - PADDING)
    r1 = min(h - 1, max(rows) + PADDING)
    c1 = min(w - 1, max(cols) + PADDING)
    return not occupied[r0 : r1 + 1, c0 : c1 + 1].any()


def _reserve_box(occupied: np.ndarray, cells: list[tuple[int, int]], h: int, w: int) -> None:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = max(0, min(rows) - PADDING)
    c0 = max(0, min(cols) - PADDING)
    r1 = min(h - 1, max(rows) + PADDING)
    c1 = min(w - 1, max(cols) + PADDING)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _choose_relays(
    rng: np.random.Generator,
    travel_cells: list[int],
    relay_count: int,
    source_color: int,
) -> list[tuple[int, int]]:
    if relay_count == 0:
        return []
    choices = rng.choice(travel_cells, size=relay_count, replace=False)
    colors = rng.choice([c for c in OBJECT_COLORS if c != source_color], size=relay_count, replace=False)
    return list(zip(map(int, choices), map(int, colors)))


def _place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: np.random.Generator,
    source_color: int,
) -> dict[str, object] | None:
    h, w = grid.shape

    for _ in range(PAIR_ATTEMPTS):
        direction = str(rng.choice(DIRS))
        relay_count = int(rng.integers(RELAY_COUNT_RANGE[0], RELAY_COUNT_RANGE[1] + 1))
        travel_min = max(TRAVEL_MIN, relay_count)
        variant = int(rng.integers(0, 2))

        if direction == "E":
            sr = int(rng.integers(1, h - 2))
            sc_high = w - 6 - travel_min
            if sc_high < 3:
                continue
            sc = int(rng.integers(3, sc_high + 1))
            max_travel = w - sc - 6
            travel = int(rng.integers(travel_min, max_travel + 1))
            fr, fc = sr - 1, sc + 2 + travel
            relay_data = _choose_relays(rng, list(range(sc + 2, fc)), relay_count, source_color)
            relay_cells = [(sr + int(rng.integers(0, 2)), axis) for axis, _color in relay_data]
        elif direction == "W":
            sr = int(rng.integers(1, h - 2))
            sc_low = 4 + travel_min
            if sc_low > w - 5:
                continue
            sc = int(rng.integers(sc_low, w - 4))
            max_travel = sc - 4
            if max_travel < travel_min:
                continue
            travel = int(rng.integers(travel_min, max_travel + 1))
            fr, fc = sr - 1, sc - 4 - travel
            relay_data = _choose_relays(rng, list(range(fc + 4, sc)), relay_count, source_color)
            relay_cells = [(sr + int(rng.integers(0, 2)), axis) for axis, _color in relay_data]
        elif direction == "S":
            sc = int(rng.integers(1, w - 2))
            sr_low = 3
            sr_high = h - 6 - travel_min
            if sr_high < sr_low:
                continue
            sr = int(rng.integers(sr_low, sr_high + 1))
            max_travel = h - sr - 6
            if max_travel < travel_min:
                continue
            travel = int(rng.integers(travel_min, max_travel + 1))
            fr, fc = sr + 2 + travel, sc - 1
            relay_data = _choose_relays(rng, list(range(sr + 2, fr)), relay_count, source_color)
            relay_cells = [(axis, sc + int(rng.integers(0, 2))) for axis, _color in relay_data]
        else:
            sc = int(rng.integers(1, w - 2))
            sr_low = 4 + travel_min
            sr_high = h - 5
            if sr_low > sr_high:
                continue
            sr = int(rng.integers(sr_low, sr_high + 1))
            max_travel = sr - 4
            if max_travel < travel_min:
                continue
            travel = int(rng.integers(travel_min, max_travel + 1))
            fr, fc = sr - 4 - travel, sc - 1
            relay_data = _choose_relays(rng, list(range(fr + 4, sr)), relay_count, source_color)
            relay_cells = [(axis, sc + int(rng.integers(0, 2))) for axis, _color in relay_data]

        source_cells = _source_cells(sr, sc)
        frame_cells = _frame_cells(fr, fc)
        arrow_cells = _arrow_cells(sr, sc, direction, variant)
        all_cells = source_cells + frame_cells + arrow_cells + relay_cells

        if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
            continue
        if any(grid[r, c] != BG for r, c in all_cells):
            continue
        if not _reserve_ok(occupied, all_cells, h, w):
            continue

        for r, c in frame_cells:
            grid[r, c] = FRAME
        for r, c in source_cells:
            grid[r, c] = source_color
        for r, c in arrow_cells:
            grid[r, c] = ARROW
        for (r, c), (_axis, relay_color) in zip(relay_cells, relay_data):
            grid[r, c] = relay_color

        _reserve_box(occupied, all_cells, h, w)

        final_color = relay_data[-1][1] if relay_data else source_color
        return {
            "source_cells": source_cells,
            "arrow_cells": arrow_cells,
            "relay_cells": relay_cells,
            "frame_top": fr,
            "frame_left": fc,
            "final_color": final_color,
        }

    return None


def _build_expected(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        for r, c in item["source_cells"]:
            out[r, c] = BG
        for r, c in item["arrow_cells"]:
            out[r, c] = BG
        for r, c in item["relay_cells"]:
            out[r, c] = BG
        fr = int(item["frame_top"])
        fc = int(item["frame_left"])
        out[fr + 1 : fr + 3, fc + 1 : fc + 3] = int(item["final_color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    pair_count = int(rng.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    source_colors = rng.choice(OBJECT_COLORS, size=pair_count, replace=False)

    placements: list[dict[str, object]] = []
    for source_color in source_colors:
        placed = _place_pair(grid, occupied, rng, int(source_color))
        if placed is None:
            return None
        placements.append(placed)

    expected = _build_expected(grid, placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(OUTER_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-with-relays puzzle")
