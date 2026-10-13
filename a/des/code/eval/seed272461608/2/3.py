import random
from collections import deque

import numpy as np


BG = 0
SOURCE = 6
LAMP = 9
LIT = 3
WIRE_PALETTE = (1, 2, 4, 5, 7, 8)


def _cell(g: np.ndarray, orient: str, fixed: int, pos: int) -> int:
    return int(g[fixed, pos]) if orient == "h" else int(g[pos, fixed])


def _coord(orient: str, fixed: int, pos: int) -> tuple[int, int]:
    return (fixed, pos) if orient == "h" else (pos, fixed)


def _marker_matches_orientation(g: np.ndarray, r: int, c: int, orient: str) -> bool:
    h, w = g.shape
    if orient == "h":
        for rr in (r - 1, r + 1):
            if 0 <= rr < h and int(g[rr, c]) != BG:
                return False
    else:
        for cc in (c - 1, c + 1):
            if 0 <= cc < w and int(g[r, cc]) != BG:
                return False
    return True


def _recover_wire(g: np.ndarray, color: int, orient: str, fixed: int) -> dict | None:
    limit = g.shape[1] if orient == "h" else g.shape[0]
    support = [pos for pos in range(limit) if _cell(g, orient, fixed, pos) == color]
    if not support:
        return None

    seen = set(support)
    q = deque(support)
    while q:
        pos = q.popleft()
        r, c = _coord(orient, fixed, pos)
        for step in (-1, 1):
            nxt = pos + step
            if not (0 <= nxt < limit):
                continue
            val = _cell(g, orient, fixed, nxt)
            if val == color:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
                continue
            if val in (SOURCE, LAMP):
                nr, nc = _coord(orient, fixed, nxt)
                if _marker_matches_orientation(g, nr, nc, orient) and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
                continue
            if val == BG:
                continue
            jump = pos + 2 * step
            if 0 <= jump < limit and _cell(g, orient, fixed, jump) == color:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
                if jump not in seen:
                    seen.add(jump)
                    q.append(jump)

    lo, hi = min(seen), max(seen)
    if hi - lo < 1:
        return None

    for pos in range(lo, hi + 1):
        val = _cell(g, orient, fixed, pos)
        if val == BG:
            return None
        if val in (SOURCE, LAMP) and pos not in (lo, hi):
            return None

    if _cell(g, orient, fixed, lo) not in (color, SOURCE, LAMP):
        return None
    if _cell(g, orient, fixed, hi) not in (color, SOURCE, LAMP):
        return None

    return {
        "color": int(color),
        "orient": orient,
        "fixed": int(fixed),
        "lo": int(lo),
        "hi": int(hi),
    }


def _infer_wires(g: np.ndarray) -> dict[int, dict]:
    colors = sorted(set(map(int, np.unique(g))) - {BG, SOURCE, LAMP})
    wires: dict[int, dict] = {}
    for color in colors:
        pts = np.argwhere(g == int(color))
        assert len(pts) > 0, "every wire color must remain visible somewhere"
        rows = sorted(set(map(int, pts[:, 0])))
        cols = sorted(set(map(int, pts[:, 1])))
        candidates = []
        if len(rows) == 1:
            wire = _recover_wire(g, int(color), "h", rows[0])
            if wire is not None:
                candidates.append(wire)
        if len(cols) == 1:
            wire = _recover_wire(g, int(color), "v", cols[0])
            if wire is not None:
                candidates.append(wire)
        assert len(candidates) == 1, f"ambiguous reconstruction for wire color {color}"
        wires[int(color)] = candidates[0]
    assert wires, "expected at least one wire"
    return wires


def _wire_cells(wire: dict) -> list[tuple[int, int]]:
    return [_coord(wire["orient"], wire["fixed"], pos) for pos in range(wire["lo"], wire["hi"] + 1)]


def _advance(wire: dict, r: int, c: int, sign: int) -> tuple[int, int]:
    if wire["orient"] == "h":
        return (r, c + sign)
    return (r + sign, c)


def _in_wire(wire: dict, r: int, c: int) -> bool:
    if wire["orient"] == "h":
        return r == wire["fixed"] and wire["lo"] <= c <= wire["hi"]
    return c == wire["fixed"] and wire["lo"] <= r <= wire["hi"]


def _endpoint_sign(wire: dict, r: int, c: int) -> int:
    if wire["orient"] == "h":
        assert r == wire["fixed"]
        if c == wire["lo"]:
            return 1
        assert c == wire["hi"]
        return -1
    assert c == wire["fixed"]
    if r == wire["lo"]:
        return 1
    assert r == wire["hi"]
    return -1


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wires = _infer_wires(g)
    cell_wires: dict[tuple[int, int], list[int]] = {}
    for color, wire in wires.items():
        for cell in _wire_cells(wire):
            cell_wires.setdefault(cell, []).append(int(color))

    for cell, colors in cell_wires.items():
        assert len(colors) <= 2, "only pairwise crossings are allowed"
        r, c = cell
        val = int(g[r, c])
        if len(colors) == 2:
            assert val not in (BG, SOURCE, LAMP), "crossings must show a wire color"
            assert wires[colors[0]]["orient"] != wires[colors[1]]["orient"], "crossings must be orthogonal"
        else:
            assert val != BG, "wire cells cannot disappear"

    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == SOURCE)]
    lamps = [tuple(map(int, rc)) for rc in np.argwhere(g == LAMP)]
    assert sources, "expected at least one source"
    assert lamps, "expected at least one lamp"
    for cell in sources + lamps:
        assert cell in cell_wires and len(cell_wires[cell]) == 1, "markers must sit on unique wire endpoints"

    seen_states = set()
    q = deque()
    for r, c in sources:
        color = cell_wires[(r, c)][0]
        q.append((int(color), int(r), int(c), _endpoint_sign(wires[color], r, c)))

    lit_lamps: set[tuple[int, int]] = set()
    straight_count = 0
    turn_count = 0
    dead_end_count = 0

    while q:
        color, r, c, sign = q.popleft()
        state = (int(color), int(r), int(c), int(sign))
        if state in seen_states:
            continue
        seen_states.add(state)

        wire = wires[color]
        nr, nc = _advance(wire, r, c, sign)
        if not _in_wire(wire, nr, nc):
            if int(g[r, c]) != LAMP:
                dead_end_count += 1
            continue

        val = int(g[nr, nc])
        if val == LAMP:
            lit_lamps.add((nr, nc))
            continue

        passing = cell_wires[(nr, nc)]
        if len(passing) == 2 and val != color:
            turn_count += 1
            other = passing[0] if passing[1] == color else passing[1]
            # The prompt does not distinguish the two outgoing directions on the new wire,
            # so signal propagation explores both visible continuations.
            q.append((other, nr, nc, -1))
            q.append((other, nr, nc, 1))
        else:
            if len(passing) == 2:
                straight_count += 1
            q.append((color, nr, nc, sign))

    return {
        "wires": wires,
        "cell_wires": cell_wires,
        "sources": sources,
        "lamps": lamps,
        "lit_lamps": lit_lamps,
        "straight_count": straight_count,
        "turn_count": turn_count,
        "dead_end_count": dead_end_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden straight wires from the visible colored overlay and light every
    lamp that is reachable from a magenta source under the crossing-priority rule.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    out = g.copy()
    for r, c in info["lit_lamps"]:
        out[r, c] = LIT
    return out


GRID_SIDE_RANGE = (14, 19)
ROW_COL_CHOICES = ((2, 2), (2, 3), (3, 2))
MIN_AXIS_MARGIN = 2
MIN_GAP = 4
GEN_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 200
MAX_SOURCES = 3
MAX_LAMPS = 4


def _sample_spaced_positions(limit: int, count: int) -> list[int] | None:
    candidates = list(range(MIN_AXIS_MARGIN, limit - MIN_AXIS_MARGIN))
    if count == 0:
        return []
    if len(candidates) < count:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= MIN_GAP for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _wire_endpoint(wire: dict, side: int) -> tuple[int, int]:
    pos = wire["lo"] if side == 0 else wire["hi"]
    return _coord(wire["orient"], wire["fixed"], pos)


def _draw_scene(shape: tuple[int, int], wires: list[dict], order: list[int], sources: set[tuple[int, int]], lamps: set[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for idx in order:
        wire = wires[idx]
        if wire["orient"] == "h":
            grid[wire["fixed"], wire["lo"] : wire["hi"] + 1] = wire["color"]
        else:
            grid[wire["lo"] : wire["hi"] + 1, wire["fixed"]] = wire["color"]
    for r, c in sources:
        grid[r, c] = SOURCE
    for r, c in lamps:
        grid[r, c] = LAMP
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_h, n_v = random.choice(ROW_COL_CHOICES)

    rows = _sample_spaced_positions(h, n_h)
    cols = _sample_spaced_positions(w, n_v)
    if rows is None or cols is None:
        return None

    colors = random.sample(list(WIRE_PALETTE), n_h + n_v)
    wires: list[dict] = []
    for row in rows:
        wires.append({"color": colors[len(wires)], "orient": "h", "fixed": row, "lo": 0, "hi": w - 1})
    for col in cols:
        wires.append({"color": colors[len(wires)], "orient": "v", "fixed": col, "lo": 0, "hi": h - 1})

    order = list(range(len(wires)))
    random.shuffle(order)

    n_sources = random.randint(1, min(MAX_SOURCES, len(wires)))
    source_wire_ids = set(random.sample(range(len(wires)), n_sources))
    sources: set[tuple[int, int]] = set()
    used_endpoints: set[tuple[int, int]] = set()
    for wire_id in source_wire_ids:
        cell = _wire_endpoint(wires[wire_id], random.randint(0, 1))
        sources.add(cell)
        used_endpoints.add(cell)

    all_endpoints = [_wire_endpoint(wire, side) for wire in wires for side in (0, 1)]
    remaining = [cell for cell in all_endpoints if cell not in used_endpoints]
    if len(remaining) < 2:
        return None

    max_lamps = min(MAX_LAMPS, len(remaining) - 1)
    if max_lamps < 1:
        return None
    n_lamps = random.randint(1, max_lamps)
    lamps = set(random.sample(remaining, n_lamps))

    grid = _draw_scene((h, w), wires, order, sources, lamps)

    for wire in wires:
        if int(np.count_nonzero(grid == wire["color"])) < 2:
            return None

    try:
        info = _analyze_grid(grid)
        out = main(grid)
    except AssertionError:
        return None

    if not info["lit_lamps"]:
        return None
    if info["turn_count"] == 0 or info["straight_count"] == 0:
        return None
    if info["dead_end_count"] == 0:
        return None
    if len(info["lamps"]) > 1 and len(info["lit_lamps"]) == len(info["lamps"]):
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-priority circuit puzzle")
