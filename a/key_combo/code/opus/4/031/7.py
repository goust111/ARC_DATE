from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, bbox_coords, components_mask


_BAR_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
_HEIGHT_POOL = (2, 3, 4, 5)
_MAX_GEN_ATTEMPTS = 500

_CUP_TEMPLATES = (
    {
        "name": "u",
        "shape": np.array([[5, 5, 5], [5, 0, 5]], dtype=int),
        "cavity": ((1, 1),),
        "opening": (1, 1),
    },
    {
        "name": "c_left",
        "shape": np.array([[5, 5], [5, 0], [5, 0]], dtype=int),
        "cavity": ((1, 1), (2, 1)),
        "opening": (2, 1),
    },
    {
        "name": "c_right",
        "shape": np.array([[5, 5], [0, 5], [0, 5]], dtype=int),
        "cavity": ((1, 0), (2, 0)),
        "opening": (2, 0),
    },
)


def _flood_cavity(cup_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = cup_mask.shape
    seeds = [(h - 1, c) for c in range(w) if not cup_mask[h - 1, c]]
    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not cup_mask[nr, nc] and (nr, nc) not in visited:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _extract_bars(grid: np.ndarray) -> list[dict]:
    h, _w = grid.shape
    bars: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {Color.BLACK, Color.GREY}):
        for coords in components_mask(grid == int(color), connectivity=4):
            if coords.size == 0:
                continue
            rows = coords[:, 0].astype(int)
            cols = coords[:, 1].astype(int)
            unique_cols = sorted(set(map(int, cols)))
            if len(unique_cols) != 1 or int(rows.max()) != h - 1:
                continue
            height = int(coords.shape[0])
            if height != int(rows.max() - rows.min() + 1):
                continue
            bars.append(
                {
                    "color": int(color),
                    "height": height,
                    "col": unique_cols[0],
                    "coords": [(int(r), unique_cols[0]) for r in sorted(rows.tolist())],
                }
            )
    return bars


def _extract_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    for coords in components_mask(grid == Color.GREY, connectivity=4):
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True
        cavity_rel = _flood_cavity(mask)
        if not cavity_rel:
            continue
        bottom_r = max(r for r, _c in cavity_rel)
        bottom_cols = sorted(c for r, c in cavity_rel if r == bottom_r)
        opening_rel = (bottom_r, bottom_cols[len(bottom_cols) // 2])
        cups.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center2": int(c0 + c1),
                "cavity_global": [(int(r0 + r), int(c0 + c)) for r, c in cavity_rel],
                "opening_global": (int(r0 + opening_rel[0]), int(c0 + opening_rel[1])),
            }
        )
    return cups


def _pair_bars_and_cups(grid: np.ndarray) -> tuple[list[dict], list[dict], list[tuple[dict, dict]]]:
    bars = _extract_bars(grid)
    cups = _extract_cups(grid)
    assert len(bars) == len(cups) and 3 <= len(bars) <= 4, "expected matched bars and cups"
    ranked_bars = sorted(bars, key=lambda item: (item["height"], item["col"]))
    ranked_cups = sorted(cups, key=lambda item: (item["center2"], item["bbox"][1], item["bbox"][0]))
    return bars, cups, list(zip(ranked_bars, ranked_cups))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair cup rays with ranked bars so each cup inherits one bar's height and color.

    Input structure:
    - Black background with 3-4 single-column colored bars touching the bottom edge.
    - Bar colors are chosen from blue/red/green/yellow, and bar heights are all distinct.
    - The upper area contains the same number of gray cup-shaped components that open downward.

    Rule:
    - Rank the bars from shortest to tallest.
    - Rank the cups from left to right using the horizontal center of each cup bbox.
    - Pair equal ranks; for each pair, fill that cup cavity with the bar color and emit a vertical
      downward ray from the cup opening whose target length equals the paired bar height.
    - While emitting, stop early at the first non-black cell or at the grid boundary.
    - After all cups are processed, erase every original bar pixel to black and leave the gray cup walls unchanged.

    Key insight:
    - Cup-bar matching is indirect: cups are ordered by horizontal position, while bars are ordered
      by height, so the correct color and ray length appear only after doing both rankings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    bars, _cups, pairs = _pair_bars_and_cups(g)

    for bar, cup in pairs:
        for r, c in cup["cavity_global"]:
            out[r, c] = int(bar["color"])

        start_r, start_c = cup["opening_global"]
        actual_len = 0
        for step in range(int(bar["height"])):
            rr = start_r + step
            if rr >= out.shape[0]:
                break
            if step > 0 and int(out[rr, start_c]) != Color.BLACK:
                break
            out[rr, start_c] = int(bar["color"])
            actual_len = step + 1
        assert actual_len >= 1

    for bar in bars:
        for r, c in bar["coords"]:
            out[r, c] = Color.BLACK

    return out


def _oracle_from_scene(grid: np.ndarray, scene: dict) -> tuple[np.ndarray, list[int]]:
    out = np.array(grid, copy=True)
    ranked_bars = sorted(scene["bars"], key=lambda item: (item["height"], item["col"]))
    ranked_cups = sorted(scene["cups"], key=lambda item: (item["center2"], item["bbox"][1], item["bbox"][0]))
    ray_lengths: list[int] = []

    for bar, cup in zip(ranked_bars, ranked_cups):
        for r, c in cup["cavity_global"]:
            out[r, c] = int(bar["color"])

        start_r, start_c = cup["opening_global"]
        actual_len = 0
        for step in range(int(bar["height"])):
            rr = start_r + step
            if rr >= out.shape[0]:
                break
            if step > 0 and int(out[rr, start_c]) != Color.BLACK:
                break
            out[rr, start_c] = int(bar["color"])
            actual_len = step + 1
        ray_lengths.append(actual_len)

    for bar in scene["bars"]:
        for r, c in bar["coords"]:
            out[r, c] = Color.BLACK

    return out, ray_lengths


def _sample_nontrivial_heights(n: int) -> list[int]:
    sampled = sorted(random.sample(_HEIGHT_POOL, n))
    while True:
        perm = random.sample(sampled, n)
        if perm != sorted(perm) and perm != sorted(perm, reverse=True):
            return perm


def _sample_spaced_columns(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    if len(candidates) < count:
        return None
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _choose_bar_columns(width: int, count: int) -> list[int] | None:
    return _sample_spaced_columns(list(range(1, width - 1)), count, min_gap=3)


def _choose_cup_openings(width: int, count: int, bar_cols: list[int], blocked_count: int) -> list[int] | None:
    blocked_cols = random.sample(bar_cols, blocked_count) if blocked_count else []
    free_pool = [c for c in range(1, width - 1) if c not in bar_cols]
    for _ in range(200):
        extra = random.sample(free_pool, count - blocked_count)
        cols = sorted(blocked_cols + extra)
        if all(b - a >= 3 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _build_scene() -> tuple[np.ndarray, dict] | None:
    count = random.randint(3, 4)
    height = random.randint(10, 16)
    width = random.randint(max(12, count * 4), 18)

    bar_cols = _choose_bar_columns(width, count)
    if bar_cols is None:
        return None

    heights_in_x_order = _sample_nontrivial_heights(count)
    colors_in_x_order = random.sample(_BAR_COLORS, count)
    want_block = random.random() < 0.7
    blocked_count = random.randint(1, max(1, count - 2)) if want_block else 0

    templates = [random.choice(_CUP_TEMPLATES) for _ in range(count)]
    if all(t["name"] == "u" for t in templates) and random.random() < 0.75:
        templates[random.randrange(count)] = random.choice(_CUP_TEMPLATES[1:])

    cup_open_cols = _choose_cup_openings(width, count, bar_cols, blocked_count)
    if cup_open_cols is None:
        return None

    max_bar_height = max(heights_in_x_order)
    cup_rows: list[int] = []
    cup_lefts: list[int] = []
    occupied: set[tuple[int, int]] = set()
    padded_occupied: set[tuple[int, int]] = set()

    for template, open_col in zip(templates, cup_open_cols):
        patch = template["shape"]
        patch_h, patch_w = patch.shape
        left = int(open_col - template["opening"][1])
        if left < 0 or left + patch_w > width:
            return None
        max_top = height - max_bar_height - patch_h - 1
        if max_top < 0:
            return None
        candidate_rows = list(range(0, max_top + 1))
        random.shuffle(candidate_rows)
        placed = False
        for top in candidate_rows:
            coords = {
                (top + r, left + c)
                for r in range(patch_h)
                for c in range(patch_w)
                if int(patch[r, c]) == Color.GREY
            }
            if coords & padded_occupied:
                continue
            occupied |= coords
            for r, c in coords:
                padded_occupied.add((r, c))
                padded_occupied.add((r - 1, c))
                padded_occupied.add((r + 1, c))
                padded_occupied.add((r, c - 1))
                padded_occupied.add((r, c + 1))
            cup_rows.append(top)
            cup_lefts.append(left)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    bars: list[dict] = []
    for col, bar_height, color in zip(bar_cols, heights_in_x_order, colors_in_x_order):
        coords = [(height - bar_height + k, col) for k in range(bar_height)]
        for r, c in coords:
            grid[r, c] = int(color)
        bars.append({"color": int(color), "height": int(bar_height), "col": int(col), "coords": coords})

    cups: list[dict] = []
    for template, top, left in zip(templates, cup_rows, cup_lefts):
        patch = template["shape"]
        patch_h, patch_w = patch.shape
        for r in range(patch_h):
            for c in range(patch_w):
                if int(patch[r, c]) == Color.GREY:
                    if grid[top + r, left + c] != Color.BLACK:
                        return None
                    grid[top + r, left + c] = Color.GREY
        cavity_global = [(top + r, left + c) for r, c in template["cavity"]]
        opening_global = (top + template["opening"][0], left + template["opening"][1])
        cups.append(
            {
                "bbox": (top, left, top + patch_h - 1, left + patch_w - 1),
                "center2": int(2 * left + patch_w - 1),
                "cavity_global": cavity_global,
                "opening_global": opening_global,
            }
        )

    scene = {
        "bars": bars,
        "cups": cups,
        "want_block": want_block,
        "bar_cols": bar_cols,
        "heights_in_x_order": heights_in_x_order,
    }
    return grid, scene


def _validate_scene(grid: np.ndarray, scene: dict) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if len(scene["bars"]) != len(scene["cups"]) or len(scene["bars"]) not in (3, 4):
        return False

    heights_in_x_order = scene["heights_in_x_order"]
    if heights_in_x_order == sorted(heights_in_x_order):
        return False
    if heights_in_x_order == sorted(heights_in_x_order, reverse=True):
        return False

    opening_cols = [cup["opening_global"][1] for cup in scene["cups"]]
    if len(set(opening_cols)) != len(opening_cols):
        return False

    detected_bars, detected_cups, _pairs = _pair_bars_and_cups(grid)
    if len(detected_bars) != len(scene["bars"]) or len(detected_cups) != len(scene["cups"]):
        return False

    oracle_out, ray_lengths = _oracle_from_scene(grid, scene)
    main_out = main(grid)
    if not np.array_equal(main_out, oracle_out):
        return False

    ranked_bars = sorted(scene["bars"], key=lambda item: (item["height"], item["col"]))
    blocked = [ray_len < bar["height"] for ray_len, bar in zip(ray_lengths, ranked_bars)]
    if not any(ray_len == bar["height"] for ray_len, bar in zip(ray_lengths, ranked_bars)):
        return False
    if scene["want_block"] and not any(blocked):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        grid, scene = built
        if _validate_scene(grid, scene):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
