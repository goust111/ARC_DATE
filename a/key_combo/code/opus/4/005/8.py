from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, get_bg
except Exception:  # pragma: no cover - fallback for direct SourceFileLoader use
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREY = 5
        GRAY = 5
        ORANGE = 7

    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREY = int(Color.GREY)
ORANGE = int(Color.ORANGE)

GRID_H_RANGE = (13, 18)
GRID_W_CAP = 22
TOTAL_SOURCE_RANGE = (5, 8)
MAX_ATTEMPTS = 400


def _summary_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    return [
        (h - 2, w - 3),
        (h - 2, w - 2),
        (h - 2, w - 1),
        (h - 1, w - 3),
        (h - 1, w - 2),
        (h - 1, w - 1),
    ]


def _find_divider_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert len(rows) == 1, "expected exactly one full grey divider row"
    return int(rows[0])


def _source_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid == RED) | (grid == BLUE))
    return sorted((int(r), int(c), int(grid[r, c])) for r, c in coords)


def _ray_success_and_path(
    grid: np.ndarray, r: int, c: int, color: int, divider: int
) -> tuple[bool, list[tuple[int, int]]]:
    h, _w = map(int, grid.shape)
    path: list[tuple[int, int]] = []

    if color == RED:
        step = 1 if r < divider else -1
        cur = int(r)
        while True:
            nxt = cur + step
            if nxt == divider:
                return True, path
            assert 0 <= nxt < h, "red ray must encounter divider before leaving grid"
            cell = int(grid[nxt, c])
            if cell == ORANGE:
                return False, []
            assert cell == BG, "only black cells may lie on an open red-ray path"
            path.append((nxt, c))
            cur = nxt

    assert color == BLUE, "expected only red or blue source pixels"
    step = -1 if r < divider else 1
    cur = int(r)
    while True:
        nxt = cur + step
        assert 0 <= nxt < h, "blue ray must stay in-bounds until the boundary cell"
        cell = int(grid[nxt, c])
        if cell == ORANGE:
            return False, []
        if nxt == 0 or nxt == h - 1:
            return True, path
        assert cell == BG, "only black cells may lie on an open blue-ray path"
        path.append((nxt, c))
        cur = nxt


def _render_output_from_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 6
    assert int(get_bg(g)) == BG, "expected black background"

    divider = _find_divider_row(g)
    slot = _summary_cells(g.shape)
    for r, c in slot:
        assert int(g[r, c]) == BG, "summary slot must be empty in the input"

    out = g.copy()
    red_success = 0
    blue_success = 0

    for r, c, color in _source_cells(g):
        assert r != divider, "sources cannot lie on the divider"
        success, path = _ray_success_and_path(g, r, c, color, divider)
        if not success:
            continue
        for rr, cc in path:
            out[rr, cc] = color
        if color == RED:
            red_success += 1
        else:
            blue_success += 1

    fill = [RED] * red_success + [BLUE] * blue_success
    for idx, (r, c) in enumerate(slot):
        out[r, c] = int(fill[idx]) if idx < len(fill) else BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast red and blue vertical rays around a grey divider, keep only unobstructed rays, then summarize the success counts in a corner bar.

    Input structure:
    - Black background with exactly one full horizontal grey divider row.
    - Above and below the divider sit isolated red and blue source pixels, plus isolated orange obstacle pixels.
    - The bottom-right 2x3 area is reserved as an empty black summary slot.

    Rule:
    - Each red source casts vertically toward the divider; each blue source casts vertically away from the divider toward the nearest top or bottom boundary.
    - Rays advance one cell at a time through black cells; hitting an orange obstacle makes that ray fail immediately, while reaching the divider (red) or the boundary (blue) makes it succeed.
    - Draw only the successful ray paths, excluding the source cell and the terminal divider/boundary cell, then count successful red rays R and successful blue rays B.
    - Fill the 2x3 summary slot in row-major order with R red cells followed by B blue cells; leave any remaining slot cells black.

    Key insight:
    - The visible output combines two coupled effects: orange blockers filter which rays actually appear, and those same success/failure decisions are re-encoded as a compact counting bar in the corner.
    """
    return _render_output_from_input(grid)


def _sample_category_counts(total_sources: int) -> dict[str, int]:
    counts = {"top_red": 1, "top_blue": 1, "bot_red": 1, "bot_blue": 1}
    for _ in range(total_sources - 4):
        choices = [
            key
            for key in counts
            if counts["top_red" if key.startswith("top") else "bot_red"]
            + counts["top_blue" if key.startswith("top") else "bot_blue"]
            < 4
        ]
        counts[random.choice(choices)] += 1
    return counts


def _sample_specs() -> list[dict[str, object]]:
    total_sources = random.randint(*TOTAL_SOURCE_RANGE)
    counts = _sample_category_counts(total_sources)

    specs: list[dict[str, object]] = []
    for key, color in (("top_red", RED), ("top_blue", BLUE), ("bot_red", RED), ("bot_blue", BLUE)):
        region = "top" if key.startswith("top") else "bottom"
        for _ in range(counts[key]):
            specs.append({"region": region, "color": color})

    for color in (RED, BLUE):
        subset = [i for i, spec in enumerate(specs) if int(spec["color"]) == color]
        fail_idx, succ_idx = random.sample(subset, 2)
        for i in subset:
            specs[i]["success"] = random.choice((True, False))
        specs[fail_idx]["success"] = False
        specs[succ_idx]["success"] = True

    return specs


def _sample_dims(specs: list[dict[str, object]]) -> tuple[int, int, int]:
    top_count = sum(1 for spec in specs if spec["region"] == "top")
    bottom_count = len(specs) - top_count
    min_h = max(GRID_H_RANGE[0], top_count + bottom_count + 7)
    h = random.randint(min_h, GRID_H_RANGE[1])
    divider_lo = top_count + 3
    divider_hi = h - bottom_count - 4
    divider = random.randint(divider_lo, divider_hi)

    total_sources = len(specs)
    min_w = max(14, 2 * total_sources + 3)
    w = random.randint(min_w, GRID_W_CAP)
    return h, w, divider


def _allowed_rows(spec: dict[str, object], h: int, divider: int) -> list[int]:
    color = int(spec["color"])
    success = bool(spec["success"])
    if spec["region"] == "top":
        lo, hi = 2, divider - 2
        if not success and color == BLUE:
            lo = 3
        if not success and color == RED:
            hi = divider - 3
    else:
        lo, hi = divider + 2, h - 3
        if not success and color == RED:
            lo = divider + 3
        if not success and color == BLUE:
            hi = h - 4
    return list(range(lo, hi + 1))


def _assign_rows(specs: list[dict[str, object]], h: int, divider: int) -> bool:
    top_ids = [i for i, spec in enumerate(specs) if spec["region"] == "top"]
    bot_ids = [i for i, spec in enumerate(specs) if spec["region"] == "bottom"]

    def assign(indices: list[int]) -> bool:
        options = {i: _allowed_rows(specs[i], h, divider) for i in indices}
        if any(len(rows) == 0 for rows in options.values()):
            return False
        order = sorted(indices, key=lambda i: (len(options[i]), int(specs[i]["color"])))
        used: set[int] = set()

        def rec(pos: int) -> bool:
            if pos == len(order):
                return True
            idx = order[pos]
            rows = [r for r in options[idx] if r not in used]
            random.shuffle(rows)
            for r in rows:
                used.add(r)
                specs[idx]["row"] = int(r)
                if rec(pos + 1):
                    return True
                used.remove(r)
                specs[idx].pop("row", None)
            return False

        return rec(0)

    return assign(top_ids) and assign(bot_ids)


def _sample_spaced_columns(count: int, w: int) -> list[int]:
    max_col = w - 4
    base = sorted(random.sample(range(1, max_col - count + 2), count))
    return [int(base[i] + i) for i in range(count)]


def _blocker_rows(spec: dict[str, object], h: int, divider: int) -> list[int]:
    r = int(spec["row"])
    color = int(spec["color"])
    if spec["region"] == "top":
        if color == RED:
            return list(range(r + 2, divider))
        return list(range(1, r - 1))
    if color == RED:
        return list(range(divider + 1, r - 1))
    return list(range(r + 2, h - 1))


def _can_place_orange(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = map(int, grid.shape)
    if not (0 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) != BG:
        return False
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
            return False
    return True


def _place_sources_and_obstacles(specs: list[dict[str, object]], h: int, w: int, divider: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    grid[divider, :] = GREY

    cols = _sample_spaced_columns(len(specs), w)
    random.shuffle(cols)
    for spec, c in zip(specs, cols):
        spec["col"] = int(c)
        grid[int(spec["row"]), int(c)] = int(spec["color"])

    for spec in specs:
        if bool(spec["success"]):
            continue
        blockers = [(r, int(spec["col"])) for r in _blocker_rows(spec, h, divider)]
        blockers = [(r, c) for r, c in blockers if _can_place_orange(grid, r, c)]
        if not blockers:
            return None
        r, c = random.choice(blockers)
        spec["blocker"] = (int(r), int(c))
        grid[r, c] = ORANGE

    if random.random() < 0.6:
        slot = set(_summary_cells(grid.shape))
        source_cols = {int(spec["col"]) for spec in specs}
        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if c not in source_cols and (r, c) not in slot and r != divider and _can_place_orange(grid, r, c)
        ]
        random.shuffle(candidates)
        for r, c in candidates[: random.randint(0, 2)]:
            grid[r, c] = ORANGE

    return grid


def _semantic_expected(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    divider = _find_divider_row(g)
    out = g.copy()
    red_success = 0
    blue_success = 0
    for r, c, color in _source_cells(g):
        success, path = _ray_success_and_path(g, r, c, color, divider)
        if not success:
            continue
        for rr, cc in path:
            out[rr, cc] = color
        if color == RED:
            red_success += 1
        else:
            blue_success += 1
    fill = [RED] * red_success + [BLUE] * blue_success
    for idx, (r, c) in enumerate(_summary_cells(g.shape)):
        out[r, c] = int(fill[idx]) if idx < len(fill) else BG
    return out, {"red_success": red_success, "blue_success": blue_success}


def _validate_generated_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    if g.ndim != 2 or max(h, w) > 30:
        return False
    if int(get_bg(g)) != BG:
        return False

    divider = _find_divider_row(g)
    slot = _summary_cells(g.shape)
    if any(int(g[r, c]) != BG for r, c in slot):
        return False

    sources = _source_cells(g)
    if not (5 <= len(sources) <= 8):
        return False

    rows = [r for r, _c, _color in sources]
    cols = [c for _r, c, _color in sources]
    if len(rows) != len(set(rows)) or len(cols) != len(set(cols)):
        return False

    top_sources = [src for src in sources if src[0] < divider]
    bot_sources = [src for src in sources if src[0] > divider]
    if not (2 <= len(top_sources) <= 4 and 2 <= len(bot_sources) <= 4):
        return False

    for color in (RED, BLUE):
        if not any(src[2] == color and src[0] < divider for src in sources):
            return False
        if not any(src[2] == color and src[0] > divider for src in sources):
            return False

    orange = np.argwhere(g == ORANGE)
    for r, c in orange:
        r = int(r)
        c = int(c)
        if r == divider or (r, c) in slot:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == ORANGE:
                return False

    expected, counts = _semantic_expected(g)
    successes = {RED: 0, BLUE: 0}
    failures = {RED: 0, BLUE: 0}
    for r, c, color in sources:
        success, _path = _ray_success_and_path(g, r, c, color, divider)
        if success:
            successes[color] += 1
        else:
            failures[color] += 1
    if not all(successes[color] >= 1 and failures[color] >= 1 for color in (RED, BLUE)):
        return False
    if successes[RED] != counts["red_success"] or successes[BLUE] != counts["blue_success"]:
        return False

    actual = main(g)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, g):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    h, w, divider = _sample_dims(specs)
    if not _assign_rows(specs, h, divider):
        return None
    grid = _place_sources_and_obstacles(specs, h, w, divider)
    if grid is None:
        return None
    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-filter summary puzzle")
