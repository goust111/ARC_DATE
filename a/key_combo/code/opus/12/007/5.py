from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
MARKER_COLORS = (1, 3, 4, 5, 9)
STAMP_RULES = (
    ("up", -1, 0, 2),
    ("down", 1, 0, 8),
    ("left", 0, -1, 7),
    ("right", 0, 1, 6),
)

GRID_HEIGHT_RANGE = (11, 17)
GRID_WIDTH_RANGE = (8, 12)
MARKER_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 600


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != BG)]


def _sorted_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    return sorted(_extract_markers(grid), key=lambda item: (-item[0], item[1]))


def _simulate(grid: np.ndarray, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    markers = _sorted_markers(g)

    out = g.copy()
    for r, c, _color in markers:
        out[r, c] = BG

    trace = []
    for start_r, start_c, color in markers:
        end_r = int(start_r)
        while end_r + 1 < H and int(out[end_r + 1, start_c]) == BG:
            end_r += 1

        stop = "bottom" if end_r == H - 1 else "obstacle"
        out[end_r, start_c] = color

        blocked: list[str] = []
        clipped: list[str] = []
        for name, dr, dc, stamp_color in STAMP_RULES:
            rr = end_r + dr
            cc = start_c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                clipped.append(name)
                continue
            if int(out[rr, cc]) != BG:
                blocked.append(name)
                continue
            out[rr, cc] = stamp_color

        if with_trace:
            trace.append(
                {
                    "color": int(color),
                    "start": (int(start_r), int(start_c)),
                    "end": (int(end_r), int(start_c)),
                    "stop": stop,
                    "blocked": tuple(blocked),
                    "clipped": tuple(clipped),
                }
            )

    if with_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored marker downward, then let its direction-colored cross become a new obstacle for later markers.

    Input structure:
    - The grid is mostly black background with 3-5 isolated single-pixel colored markers.
    - Marker colors are distinct and come from a small palette of non-construction colors.
    - Markers occupy different rows and columns, leaving empty space below for falling.

    Rule:
    - Process markers from lower initial row to higher initial row; if rows tie, process the leftmost one first.
    - Remove all markers from their starts, then for each marker drop it straight down in its own column until it reaches the bottom row or the cell below is already non-background.
    - Place the marker at its resting cell and immediately stamp a clipped four-neighbor cross: above red(2), below teal(8), left orange(7), right pink(6), without overwriting any existing non-background cell.

    Key insight:
    - The cross is not just decoration: each landed marker creates fresh obstacles, so earlier landings can suspend later markers above the floor and can also block parts of later stamps.
    """
    return _simulate(np.asarray(grid, dtype=int), with_trace=False)


def _pick_chain_columns(width: int, chain_len: int, direction: int, want_edge: bool) -> list[int]:
    if direction not in (-1, 1):
        raise ValueError("direction must be +/-1")

    if want_edge:
        start = 0 if direction == 1 else width - 1
        return [start + i * direction for i in range(chain_len)]

    if direction == 1:
        starts = list(range(1, width - chain_len))
        if not starts:
            starts = list(range(0, width - chain_len + 1))
    else:
        starts = list(range(chain_len - 1, width - 1))
        if not starts:
            starts = list(range(chain_len - 1, width))
    start = random.choice(starts)
    return [start + i * direction for i in range(chain_len)]


def _pick_extra_columns(width: int, used: list[int], extra_count: int) -> list[int] | None:
    if extra_count == 0:
        return []

    candidates = [c for c in range(width) if all(abs(c - u) >= 2 for u in used)]
    random.shuffle(candidates)
    picked: list[int] = []
    for c in candidates:
        if all(abs(c - p) >= 2 for p in picked):
            picked.append(c)
            if len(picked) == extra_count:
                return picked
    return None


def _sample_rows(height: int, cols: list[int], chain_len: int) -> list[int] | None:
    top_band = list(range(0, height - 2))
    if len(top_band) < len(cols):
        return None

    chain_cols = cols[:chain_len]
    extra_cols = cols[chain_len:]
    for _ in range(250):
        chain_rows = sorted(random.sample(top_band, chain_len), reverse=True)
        if any(chain_rows[i] - chain_rows[i + 1] < 2 for i in range(chain_len - 1)):
            continue

        remaining_rows = [r for r in top_band if r not in chain_rows]
        extra_rows = random.sample(remaining_rows, len(extra_cols))
        rows = chain_rows + extra_rows
        positions = list(zip(rows, cols))
        if all(abs(r1 - r2) + abs(c1 - c2) >= 3 for (r1, c1), (r2, c2) in combinations(positions, 2)):
            return rows
    return None


def _build_candidate() -> tuple[np.ndarray, dict] | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    marker_count = random.randint(*MARKER_COUNT_RANGE)

    want_edge = random.random() < 0.65
    chain_len = random.randint(2, min(4, marker_count))
    direction = random.choice((-1, 1))

    chain_cols = _pick_chain_columns(W, chain_len, direction, want_edge)
    extra_cols = _pick_extra_columns(W, chain_cols, marker_count - chain_len)
    if extra_cols is None:
        return None

    cols = chain_cols + extra_cols
    rows = _sample_rows(H, cols, chain_len)
    if rows is None:
        return None

    colors = random.sample(list(MARKER_COLORS), marker_count)
    grid = np.zeros((H, W), dtype=int)
    for r, c, color in zip(rows, cols, colors):
        grid[r, c] = color

    metadata = {
        "chain_colors": tuple(colors[:chain_len]),
        "extra_colors": tuple(colors[chain_len:]),
        "chain_cols": tuple(chain_cols),
        "chain_len": int(chain_len),
        "want_edge": bool(want_edge),
        "direction": int(direction),
    }
    return grid, metadata


def _validate_candidate(grid: np.ndarray, meta: dict) -> bool:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    markers = _sorted_markers(g)

    if not (3 <= len(markers) <= 5):
        return False
    rows = [r for r, _c, _color in markers]
    cols = [c for _r, c, _color in markers]
    colors = [color for _r, _c, color in markers]

    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols) or len(set(colors)) != len(colors):
        return False
    if any(color not in MARKER_COLORS for color in colors):
        return False
    if any(r >= H - 1 for r in rows):
        return False
    if any(abs(r1 - r2) + abs(c1 - c2) < 3 for (r1, c1), (r2, c2) in combinations(zip(rows, cols), 2)):
        return False

    out, trace = _simulate(g, with_trace=True)
    trace_by_color = {int(item["color"]): item for item in trace}
    if len(trace_by_color) != len(markers):
        return False

    chain_colors = meta["chain_colors"]
    chain_cols = meta["chain_cols"]
    for index, color in enumerate(chain_colors):
        info = trace_by_color[int(color)]
        expected_row = H - 1 - index
        expected_col = int(chain_cols[index])
        if info["end"] != (expected_row, expected_col):
            return False
        if index == 0:
            if info["stop"] != "bottom":
                return False
        else:
            if info["stop"] != "obstacle":
                return False
            if "down" not in info["blocked"]:
                return False

    for color in meta["extra_colors"]:
        if trace_by_color[int(color)]["stop"] != "bottom":
            return False

    if meta["want_edge"]:
        first_info = trace_by_color[int(chain_colors[0])]
        if chain_cols[0] == 0 and "left" not in first_info["clipped"]:
            return False
        if chain_cols[0] == W - 1 and "right" not in first_info["clipped"]:
            return False

    obstacle_stops = sum(1 for item in trace if item["stop"] == "obstacle")
    blocked_stamps = sum(len(item["blocked"]) for item in trace)
    if obstacle_stops < meta["chain_len"] - 1:
        return False
    if blocked_stamps < meta["chain_len"] - 1:
        return False

    stamp_colors = {int(v) for v in np.unique(out)} - {BG} - set(colors)
    if len(stamp_colors) < 3:
        return False
    if np.array_equal(g, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        grid, meta = built
        if _validate_candidate(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid cascading gravity-stamp input")
