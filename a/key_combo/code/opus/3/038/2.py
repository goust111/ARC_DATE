from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
CUP = 5
BAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components_mask(grid == int(color))


def _detect_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for coords in _color_components(grid, CUP):
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 4:
            continue

        top = grid[r0, c0 : c1 + 1]
        opening_rel = np.flatnonzero(top != CUP)
        if opening_rel.size == 0:
            continue
        if opening_rel[0] == 0 or opening_rel[-1] == c1 - c0:
            continue
        if not np.array_equal(opening_rel, np.arange(opening_rel[0], opening_rel[-1] + 1)):
            continue

        if not np.all(grid[r0 : r1 + 1, c0] == CUP):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == CUP):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == CUP):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        interior_patch = inner == BG
        if not interior_patch.any():
            continue

        interior_mask = np.zeros_like(grid, dtype=bool)
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = interior_patch
        cups.append(
            {
                "bbox": (r0, c0, r1, c1),
                "open_c0": c0 + int(opening_rel[0]),
                "open_c1": c0 + int(opening_rel[-1]),
                "mouth_w": int(opening_rel.size),
                "interior_mask": interior_mask,
            }
        )

    cups.sort(key=lambda cup: (cup["bbox"][0], cup["bbox"][1]))
    return cups


def _detect_bars(grid: np.ndarray) -> list[dict]:
    bars: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG, CUP}):
        for coords in _color_components(grid, color):
            rows = np.unique(coords[:, 0])
            if rows.size != 1:
                continue
            cols = np.sort(coords[:, 1])
            if not np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1)):
                continue
            bars.append(
                {
                    "color": int(color),
                    "row": int(rows[0]),
                    "c0": int(cols[0]),
                    "c1": int(cols[-1]),
                    "width": int(cols.size),
                    "coords": coords,
                }
            )

    bars.sort(key=lambda bar: (bar["row"], bar["c0"]))
    return bars


def _assign_bars_to_cups(bars: list[dict], cups: list[dict]) -> dict[int, dict]:
    assigned: dict[int, dict] = {}
    for bar in bars:
        candidates = []
        for idx, cup in enumerate(cups):
            cup_r0 = cup["bbox"][0]
            if bar["row"] >= cup_r0:
                continue
            if cup["open_c0"] <= bar["c0"] and bar["c1"] <= cup["open_c1"]:
                candidates.append((cup_r0, idx))
        if candidates:
            _r0, best_idx = min(candidates)
            assigned[best_idx] = bar
    return assigned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop horizontal seed bars into the first cup whose opening fully admits them, then fill that cup with the seed color.

    Input structure:
    - The background is black.
    - Gray pixels form 2-3 upward-opening cup shapes near the bottom; each cup is a U-shaped connected component with a black mouth on its top row.
    - Separate non-gray, non-black objects near the top are monochromatic horizontal bars, each occupying one row of consecutive cells.

    Rule:
    - Detect every gray cup, its top-row mouth interval, and the black interior cells enclosed below that mouth.
    - Detect every colored horizontal bar and erase all bars from the output.
    - For each bar, look straight downward and choose the first cup below whose mouth fully contains the bar's column span; equivalently the bar width must fit within that opening and stay column-aligned with it.
    - Fill the chosen cup's enclosed interior black cells with the bar color, leaving uncaptured cups black and keeping the gray cup walls unchanged.

    Key insight:
    - Color does not choose the destination cup; the deciding signal is geometric compatibility between a falling bar's horizontal span and the cup mouth directly below it.
    """
    g = np.asarray(grid, dtype=int)
    cups = _detect_cups(g)
    bars = _detect_bars(g)

    out = g.copy()
    for bar in bars:
        rr = bar["coords"][:, 0]
        cc = bar["coords"][:, 1]
        out[rr, cc] = BG

    assigned = _assign_bars_to_cups(bars, cups)
    for idx, cup in enumerate(cups):
        bar = assigned.get(idx)
        if bar is None:
            continue
        out[cup["interior_mask"]] = bar["color"]
    return out


def _draw_cup(grid: np.ndarray, top: int, left: int, height: int, mouth_w: int) -> dict:
    outer_w = mouth_w + 2
    bottom = top + height - 1
    right = left + outer_w - 1
    grid[top : bottom + 1, left] = CUP
    grid[top : bottom + 1, right] = CUP
    grid[bottom, left : right + 1] = CUP
    return {
        "bbox": (top, left, bottom, right),
        "open_c0": left + 1,
        "open_c1": right - 1,
        "mouth_w": mouth_w,
        "interior": (slice(top + 1, bottom), slice(left + 1, right)),
    }


def _draw_bar(grid: np.ndarray, row: int, c0: int, width: int, color: int) -> dict:
    grid[row, c0 : c0 + width] = color
    return {"row": row, "c0": c0, "c1": c0 + width - 1, "width": width, "color": color}


def _make_expected(grid: np.ndarray, cups: list[dict], captures: list[tuple[int, dict]]) -> np.ndarray:
    out = grid.copy()
    out[(out != BG) & (out != CUP)] = BG
    for cup_idx, bar in captures:
        rs, cs = cups[cup_idx]["interior"]
        out[rs, cs] = bar["color"]
    return out


def _gap_starts(total: int, slots: int, minimum: int) -> list[int]:
    base = [minimum] * slots
    extra = total - minimum * slots
    for _ in range(extra):
        base[random.randrange(slots)] += 1
    return base


def _candidate_starts(width: int, w: int, cups: list[dict]) -> list[int]:
    starts = []
    for c0 in range(0, w - width + 1):
        c1 = c0 + width - 1
        if all(not (cup["open_c0"] <= c0 and c1 <= cup["open_c1"]) for cup in cups):
            starts.append(c0)
    return starts


def _choose_row_pool(limit: int, needed: int) -> list[int] | None:
    if limit < needed:
        return None
    rows = list(range(limit))
    chosen = random.sample(rows, needed)
    chosen.sort()
    return chosen


def _sample_mouths(n_cups: int) -> list[int]:
    if n_cups == 2:
        mouths = [2, 3]
        random.shuffle(mouths)
        return mouths
    mouths = [2, 3, random.choice([2, 3])]
    random.shuffle(mouths)
    return mouths


def _generate_one() -> np.ndarray | None:
    n_cups = random.choices([2, 3], weights=[3, 1], k=1)[0]
    mouths = _sample_mouths(n_cups)
    heights = [random.randint(4, 6) for _ in range(n_cups)]
    outer_ws = [mouth + 2 for mouth in mouths]

    min_w = sum(outer_ws) + (n_cups + 1)
    max_w = min(24, min_w + 6)
    if min_w > max_w:
        return None
    w = random.randint(max(14, min_w), max_w)

    min_h = max(13, max(heights) + 8)
    max_h = min(22, min_h + 5)
    if min_h > max_h:
        return None
    h = random.randint(min_h, max_h)

    grid = np.zeros((h, w), dtype=int)

    gap_total = w - sum(outer_ws)
    gaps = _gap_starts(gap_total, n_cups + 1, 1)
    lefts = []
    cur = gaps[0]
    for idx, outer_w in enumerate(outer_ws):
        lefts.append(cur)
        cur += outer_w + gaps[idx + 1]

    cup_specs: list[dict] = []
    for left, height, mouth in zip(lefts, heights, mouths):
        bottom = h - random.choice([1, 2])
        top = bottom - height + 1
        cup_specs.append(_draw_cup(grid, top, left, height, mouth))

    open_top = min(spec["bbox"][0] for spec in cup_specs)
    extra_unmatched = 1 if random.random() < 0.6 else 0
    capture_count = random.randint(1, n_cups)
    total_bars = capture_count + 1 + extra_unmatched
    rows = _choose_row_pool(open_top - 1, total_bars)
    if rows is None:
        return None
    row_iter = iter(rows)

    colors = random.sample(BAR_COLORS, total_bars)
    color_iter = iter(colors)
    captures: list[tuple[int, dict]] = []

    capture_indices = sorted(random.sample(range(n_cups), capture_count))
    for cup_idx in capture_indices:
        cup = cup_specs[cup_idx]
        if random.random() < 0.4:
            width = cup["mouth_w"]
        else:
            width = random.randint(1, cup["mouth_w"])
        c0 = random.randint(cup["open_c0"], cup["open_c1"] - width + 1)
        bar = _draw_bar(grid, next(row_iter), c0, width, next(color_iter))
        captures.append((cup_idx, bar))

    oversize_cup = random.randrange(n_cups)
    oversize_spec = cup_specs[oversize_cup]
    oversize_left_min = oversize_spec["bbox"][1]
    oversize_left_max = oversize_spec["bbox"][3] - 3
    oversize_c0 = random.randint(oversize_left_min, oversize_left_max)
    _draw_bar(grid, next(row_iter), oversize_c0, 4, next(color_iter))

    if extra_unmatched:
        width = random.randint(1, 3)
        starts = _candidate_starts(width, w, cup_specs)
        if not starts:
            return None
        _draw_bar(grid, next(row_iter), random.choice(starts), width, next(color_iter))

    expected = _make_expected(grid, cup_specs, captures)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid width-gated cup input")
