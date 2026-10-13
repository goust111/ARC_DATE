from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_7_056_7",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

bbox_h = _ARC_UTILS.bbox_h
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
RECT_COUNT = 4
WIDTH_RANGE = (2, 4)
HEIGHT_CHOICES = (2, 3, 4, 5)
LANE_GAP = 2
MAX_ATTEMPTS = 400


def _dot_clearance(k: int) -> int:
    if k <= 0:
        return 0
    return 2 + 2 * ((k - 1) // 2)


def _dot_positions(r0: int, c0: int, h: int, w: int, side: str, k: int) -> list[tuple[int, int]]:
    if k == 0:
        return []
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    out: list[tuple[int, int]] = []
    for idx in range(k):
        level = idx // 2
        col = c0 - 1 if idx % 2 == 0 else c1 + 1
        if side == "up":
            row = r0 - 2 - 2 * level
        else:
            row = r1 + 2 + 2 * level
        out.append((int(row), int(col)))
    return out


def _parse_by_color(grid: np.ndarray) -> dict[int, dict]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one foreground component"

    grouped: dict[int, list[dict]] = {}
    for comp in comps:
        grouped.setdefault(int(comp["color"]), []).append(comp)

    parsed: dict[int, dict] = {}
    for color, parts in grouped.items():
        rectangles = [comp for comp in parts if int(comp["area"]) > 1]
        dots = [comp for comp in parts if int(comp["area"]) == 1]
        assert len(rectangles) == 1, "each foreground color must have one solid rectangle"
        rect = rectangles[0]

        r0, c0, r1, c1 = map(int, rect["bbox"])
        h = int(bbox_h(rect["bbox"]))
        w = int(c1 - c0 + 1)
        assert int(rect["area"]) == h * w, "rectangle must fill its bounding box"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle must be monochrome"

        side = "none"
        if dots:
            rows = [int(dot["coords"][0, 0]) for dot in dots]
            cols = [int(dot["coords"][0, 1]) for dot in dots]
            assert all(col < c0 or col > c1 for col in cols), "dots must avoid rectangle columns"
            has_above = any(row < r0 for row in rows)
            has_below = any(row > r1 for row in rows)
            assert has_above ^ has_below, "same-color dots must be all above or all below"
            side = "up" if has_above else "down"
            assert all((row < r0) if side == "up" else (row > r1) for row in rows)

        parsed[color] = {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "h": h,
            "w": w,
            "k": len(dots),
            "side": side,
            "dot_coords": [tuple(map(int, dot["coords"][0])) for dot in dots],
        }

    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each solid rectangle by its own height times the number of matching guide dots, then erase the dots.

    Input structure:
    - The background is black and the foreground contains several differently colored solid rectangles.
    - Some colors also have isolated single-pixel dots of the same color; each such color has dots only above its rectangle or only below it.
    - Matching dots never lie in the rectangle's occupied columns, and some rectangles may have no matching dots.

    Rule:
    - For each rectangle, count same-color dots k and determine whether they are above or below that rectangle.
    - Erase the original rectangle, then redraw it shifted toward that side by exactly k times its own height.
    - Remove every isolated dot from the output; rectangles with k = 0 stay in place.

    Key insight:
    - Same-color dots are not targets to land on: they jointly encode a direction (above or below) and a jump length multiplier (their count).
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    parsed = _parse_by_color(g)

    out = np.full_like(g, bg)
    occupied = np.zeros(g.shape, dtype=bool)

    for info in parsed.values():
        r0, c0, r1, c1 = info["bbox"]
        h = int(info["h"])
        k = int(info["k"])
        side = info["side"]
        dr = 0 if side == "none" else (-h * k if side == "up" else h * k)
        dst_r0 = int(r0 + dr)
        dst_r1 = int(r1 + dr)
        assert 0 <= dst_r0 <= dst_r1 < g.shape[0], "translated rectangle must stay in bounds"
        target = occupied[dst_r0 : dst_r1 + 1, c0 : c1 + 1]
        assert not target.any(), "translated rectangles must not overlap"
        target[:, :] = True
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        paste(out, patch, dst_r0, c0)

    return out


def _choose_specs() -> list[dict]:
    common_h = random.choice(HEIGHT_CHOICES[:-1])
    alt_h = random.choice([h for h in HEIGHT_CHOICES if h != common_h])
    idle_h = random.choice(HEIGHT_CHOICES)

    templates = [
        [("up", 1, common_h), ("down", 2, common_h), ("rand", 1, alt_h), ("none", 0, idle_h)],
        [("down", 1, common_h), ("up", 3, common_h), ("rand", 1, alt_h), ("none", 0, idle_h)],
        [("up", 2, common_h), ("down", 1, common_h), ("rand", 2, alt_h), ("none", 0, idle_h)],
        [("down", 2, common_h), ("up", 1, common_h), ("rand", 2, alt_h), ("none", 0, idle_h)],
    ]
    chosen = random.choice(templates)

    specs: list[dict] = []
    for side, k, h in chosen:
        real_side = random.choice(["up", "down"]) if side == "rand" else side
        specs.append({"side": real_side, "k": int(k), "h": int(h), "w": random.randint(*WIDTH_RANGE)})
    random.shuffle(specs)
    return specs


def _lane_columns(specs: list[dict]) -> list[int]:
    lane_widths = [int(spec["w"]) + 2 for spec in specs]
    base_width = sum(lane_widths) + LANE_GAP * (len(specs) - 1)
    assert base_width <= 30
    extra = random.randint(0, 30 - base_width)
    left_pad = random.randint(0, extra)

    c0s: list[int] = []
    cursor = left_pad
    for lane_w, spec in zip(lane_widths, specs):
        c0s.append(int(cursor + 1))
        cursor += int(lane_w) + LANE_GAP
    return c0s


def _sample_input_row(grid_h: int, side: str, h: int, k: int) -> int | None:
    clearance = _dot_clearance(k)
    if side == "up":
        lo = max(h * k, clearance)
        hi = grid_h - h
    elif side == "down":
        lo = 0
        hi = min(grid_h - h * (k + 1), grid_h - h - clearance)
    else:
        lo = 0
        hi = grid_h - h
    if hi < lo:
        return None
    return random.randint(int(lo), int(hi))


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> None:
    parsed = _parse_by_color(inp)
    assert len(parsed) == RECT_COUNT
    sides = [info["side"] for info in parsed.values()]
    ks = [int(info["k"]) for info in parsed.values()]
    hs = [int(info["h"]) for info in parsed.values()]

    assert "up" in sides and "down" in sides and "none" in sides
    assert 0 in ks and any(k >= 2 for k in ks)
    assert any(info["k"] == 0 for info in parsed.values())

    moving = [info for info in parsed.values() if int(info["k"]) > 0]
    assert any(a["h"] == b["h"] and a["k"] != b["k"] for a in moving for b in moving if a is not b)
    assert any(a["h"] != b["h"] and a["k"] == b["k"] for a in moving for b in moving if a is not b)

    for info in parsed.values():
        for r, c in info["dot_coords"]:
            r0, c0, r1, c1 = info["bbox"]
            assert c < c0 or c > c1
            assert (r < r0) if info["side"] == "up" else (info["side"] == "none" or r > r1)

    dot_mask = np.zeros(inp.shape, dtype=bool)
    for info in parsed.values():
        for r, c in info["dot_coords"]:
            assert not dot_mask[r, c]
            dot_mask[r, c] = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
                        assert inp[rr, cc] == BG

    actual = main(inp)
    assert np.array_equal(actual, expected)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    specs = _choose_specs()
    c0s = _lane_columns(specs)
    grid_h = random.randint(22, 30)
    grid_w = max(c0 + spec["w"] + 1 for c0, spec in zip(c0s, specs))
    assert grid_w <= 30

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full_like(grid, BG)
    colors = random.sample(PALETTE, RECT_COUNT)

    rows: list[int] = []
    for spec in specs:
        row = _sample_input_row(grid_h, spec["side"], int(spec["h"]), int(spec["k"]))
        if row is None:
            raise RuntimeError("failed to sample input row")
        rows.append(int(row))

    for color, spec, r0, c0 in zip(colors, specs, rows, c0s):
        h = int(spec["h"])
        w = int(spec["w"])
        k = int(spec["k"])
        side = str(spec["side"])

        rect = np.full((h, w), int(color), dtype=int)
        paste(grid, rect, r0, c0)

        dr = 0 if side == "none" else (-h * k if side == "up" else h * k)
        paste(expected, rect, r0 + dr, c0)

        for rr, cc in _dot_positions(r0, c0, h, w, side, k):
            assert 0 <= rr < grid_h and 0 <= cc < grid_w
            assert grid[rr, cc] == BG
            grid[rr, cc] = int(color)

    _semantic_checks(grid, expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, _expected = _generate_case()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

