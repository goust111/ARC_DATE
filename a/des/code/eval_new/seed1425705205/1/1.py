from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np


_BG = 0
_GREY = 5
_MAROON = 9

_LEGEND_H = 3
_LEGEND_ENTRY_W = 6
_LEGEND_GAP = 1

_SEED_COLORS = [1, 2, 3, 4, 6, 7, 8]
_INNER_SIZES = [3, 5, 5, 7]
_MAX_RETRIES = 400

_ROTATION_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
_ANCHOR_BY_SIDE = {
    "top": (0, 1),
    "right": (1, 2),
    "bottom": (2, 1),
    "left": (1, 0),
}
_SIDES = ["top", "right", "bottom", "left"]


def _shape(rows: List[str]) -> np.ndarray:
    arr = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    assert arr.shape == (3, 3)
    assert arr[0, 1], "canonical templates must anchor at top-center"
    return arr


_TEMPLATE_POOL = [
    _shape(["010", "110", "010"]),
    _shape(["010", "011", "010"]),
    _shape(["010", "111", "100"]),
    _shape(["010", "110", "001"]),
    _shape(["011", "010", "110"]),
    _shape(["010", "101", "011"]),
    _shape(["011", "110", "010"]),
]


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> Tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def _is_legend_entry(sub: np.ndarray) -> bool:
    if sub.shape != (_LEGEND_H, _LEGEND_ENTRY_W):
        return False
    left = sub[:, :3]
    right = sub[:, 3:]
    color = int(left[0, 0])
    if color in (_BG, _GREY, _MAROON):
        return False
    if not np.all(left == color):
        return False
    right_colors = set(map(int, np.unique(right)))
    return right_colors <= {_BG, _MAROON} and _MAROON in right_colors


def _read_legend(grid: np.ndarray) -> Tuple[Dict[int, np.ndarray], int]:
    h, w = grid.shape
    legend_row = None
    for r in range(h - _LEGEND_H, -1, -1):
        found = False
        for c in range(w - _LEGEND_ENTRY_W + 1):
            if _is_legend_entry(grid[r : r + _LEGEND_H, c : c + _LEGEND_ENTRY_W]):
                found = True
                break
        if found:
            legend_row = r
            break
    assert legend_row is not None, "legend strip not found"

    mapping: Dict[int, np.ndarray] = {}
    c = 0
    while c <= w - _LEGEND_ENTRY_W:
        sub = grid[legend_row : legend_row + _LEGEND_H, c : c + _LEGEND_ENTRY_W]
        if _is_legend_entry(sub):
            color = int(sub[0, 0])
            assert color not in mapping, "seed colors in the legend must be unique"
            mapping[color] = sub[:, 3:] == _MAROON
            c += _LEGEND_ENTRY_W
            while c < w and np.all(grid[legend_row : legend_row + _LEGEND_H, c] == _BG):
                c += 1
        else:
            c += 1

    assert mapping, "legend must contain at least one color-template entry"
    return mapping, legend_row


def _find_rectangles(grid: np.ndarray, legend_row: int) -> List[Tuple[int, int, int, int]]:
    grey_mask = grid == _GREY
    rects: List[Tuple[int, int, int, int]] = []
    for comp in _components(grey_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 >= legend_row:
            continue
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        sub = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(sub, border):
            continue
        rects.append((r0, c0, r1, c1))
    assert rects, "expected at least one closed grey rectangle"
    return rects


def _seed_side(rect: Tuple[int, int, int, int], seed: Tuple[int, int]) -> str:
    r0, c0, r1, c1 = rect
    sr, sc = seed
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if (sr, sc) == (r0 + 1, mid_c):
        return "top"
    if (sr, sc) == (mid_r, c1 - 1):
        return "right"
    if (sr, sc) == (r1 - 1, mid_c):
        return "bottom"
    if (sr, sc) == (mid_r, c0 + 1):
        return "left"
    raise AssertionError(f"seed is not on an interior side midpoint: {(sr, sc)}")


def _stamp_template(
    out: np.ndarray,
    template: np.ndarray,
    color: int,
    seed: Tuple[int, int],
    side: str,
    rect: Tuple[int, int, int, int],
) -> None:
    rotated = np.rot90(template, _ROTATION_K[side])
    anchor_r, anchor_c = _ANCHOR_BY_SIDE[side]
    seed_r, seed_c = seed
    r0, c0, r1, c1 = rect

    out[seed_r, seed_c] = _BG
    for tr, tc in np.argwhere(rotated):
        rr = int(seed_r + tr - anchor_r)
        cc = int(seed_c + tc - anchor_c)
        assert r0 < rr < r1 and c0 < cc < c1, "stamped template must stay inside the rectangle"
        out[rr, cc] = color


def _apply_rule_from_rectangles(
    grid: np.ndarray,
    mapping: Dict[int, np.ndarray],
    rects: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((interior != _BG) & (interior != _GREY))
        assert len(seeds) == 1, "each rectangle must contain exactly one seed pixel"
        sr, sc = map(int, seeds[0])
        seed = (r0 + 1 + sr, c0 + 1 + sc)
        color = int(grid[seed])
        assert color in mapping, f"seed color {color} is missing from the legend"
        side = _seed_side(rect, seed)
        _stamp_template(out, mapping[color], color, seed, side, rect)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp legend-selected 3x3 templates into grey-bounded rectangles based on seed color and side.

    Input structure:
    - Background is black. The main field contains several closed grey rectangular borders.
    - Inside each rectangle there is exactly one colored seed pixel, positioned at the midpoint of one interior side.
    - A legend strip sits at the bottom: each entry has a solid color block on the left and a 3x3 maroon-on-black template on the right.

    Rule:
    - Read the bottom legend as a mapping from seed color to a 3x3 template.
    - For each rectangle, locate its unique seed color and determine the seed side: top, right, bottom, or left.
    - Rotate the matched template according to that side and stamp it inside the rectangle so the template's facing edge-center cell aligns with the seed.
    - Draw stamped cells in the seed color over black interior cells, replacing the seed pixel itself.

    Key insight:
    - The seed color selects which template to use, while the seed's side midpoint selects the template rotation and anchor.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_row = _read_legend(g)
    rects = _find_rectangles(g, legend_row)
    return _apply_rule_from_rectangles(g, mapping, rects)


def _legend_width(n_entries: int) -> int:
    return n_entries * _LEGEND_ENTRY_W + (n_entries - 1) * _LEGEND_GAP


def _draw_rectangle(grid: np.ndarray, rect: Tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = _GREY
    grid[r1, c0 : c1 + 1] = _GREY
    grid[r0 : r1 + 1, c0] = _GREY
    grid[r0 : r1 + 1, c1] = _GREY


def _pick_seed(rect: Tuple[int, int, int, int], side: str) -> Tuple[int, int]:
    r0, c0, r1, c1 = rect
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "top":
        return (r0 + 1, mid_c)
    if side == "right":
        return (mid_r, c1 - 1)
    if side == "bottom":
        return (r1 - 1, mid_c)
    return (mid_r, c0 + 1)


def _find_positions(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    max_row: int,
) -> List[Tuple[int, int]]:
    h, w = occupied.shape
    positions: List[Tuple[int, int]] = []
    for r0 in range(1, max_row - box_h + 1):
        for c0 in range(1, w - box_w):
            r1 = r0 + box_h
            c1 = c0 + box_w
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r1 + 1)
            cc1 = min(w, c1 + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_legend(grid: np.ndarray, mapping: Dict[int, np.ndarray], rng: random.Random) -> int:
    h, w = grid.shape
    legend_row = h - _LEGEND_H
    order = list(mapping)
    rng.shuffle(order)
    total_w = _legend_width(len(order))
    start_c = rng.randint(0, w - total_w)
    c = start_c
    for color in order:
        grid[legend_row : legend_row + _LEGEND_H, c : c + 3] = color
        patch = np.where(mapping[color], _MAROON, _BG)
        grid[legend_row : legend_row + _LEGEND_H, c + 3 : c + 6] = patch
        c += _LEGEND_ENTRY_W + _LEGEND_GAP
    return legend_row


def _expected_output(
    inp: np.ndarray,
    specs: List[dict],
    mapping: Dict[int, np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _stamp_template(
            out,
            mapping[spec["color"]],
            spec["color"],
            spec["seed"],
            spec["side"],
            spec["rect"],
        )
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(2, 4)
    legend_count = rng.randint(2, min(4, region_count))

    min_w = max(18, _legend_width(legend_count) + 1)
    h = rng.randint(max(16, 10 + region_count * 3), 29)
    w = rng.randint(min_w, 30)
    legend_row = h - _LEGEND_H
    max_rect_row = legend_row - 1
    if max_rect_row < 6:
        return None

    colors = rng.sample(_SEED_COLORS, legend_count)
    template_ids = rng.sample(range(len(_TEMPLATE_POOL)), legend_count)
    mapping = {color: _TEMPLATE_POOL[idx] for color, idx in zip(colors, template_ids)}

    region_colors = colors[:]
    while len(region_colors) < region_count:
        region_colors.append(rng.choice(colors))
    rng.shuffle(region_colors)

    directions = _SIDES[:]
    rng.shuffle(directions)
    sides = directions[:region_count]

    specs: List[dict] = []
    for color, side in zip(region_colors, sides):
        inner_h = rng.choice(_INNER_SIZES)
        inner_w = rng.choice(_INNER_SIZES)
        rect_h = inner_h + 2
        rect_w = inner_w + 2
        specs.append(
            {
                "color": color,
                "side": side,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "area": rect_h * rect_w,
            }
        )
    specs.sort(key=lambda spec: spec["area"], reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, legend_row - 1) :, :] = True
    for spec in specs:
        positions = _find_positions(occupied, spec["rect_h"], spec["rect_w"], max_rect_row)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        rect = (r0, c0, r0 + spec["rect_h"] - 1, c0 + spec["rect_w"] - 1)
        spec["rect"] = rect
        spec["seed"] = _pick_seed(rect, spec["side"])
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + spec["rect_h"] + 1)
        cc1 = min(w, c0 + spec["rect_w"] + 1)
        occupied[rr0:rr1, cc0:cc1] = True

    inp = np.full((h, w), _BG, dtype=int)
    for spec in specs:
        _draw_rectangle(inp, spec["rect"])
        seed_r, seed_c = spec["seed"]
        inp[seed_r, seed_c] = spec["color"]
    _place_legend(inp, mapping, rng)

    expected = _expected_output(inp, specs, mapping)
    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
