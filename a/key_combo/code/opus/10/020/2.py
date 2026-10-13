from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
YELLOW = 4
ORANGE = 7
DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
BORDER_COLORS = (1, 2, 3, 5, 6, 8, 9)
GRID_RANGE = (12, 15)
COUNT_OPTIONS = (2, 3)
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    seen = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen or (nr, nc) in border_set:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _detect_cups(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BACKGROUND, YELLOW, ORANGE)]
    cups: list[dict] = []
    for color in colors:
        for coords in _components(grid == color, connectivity=8):
            if coords.size == 0:
                continue
            bbox = _bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            border_set = {(int(r), int(c)) for r, c in coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (tail_dir, cells[0])
                for tail_dir, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            valid = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                content_cells = [
                    (r, c)
                    for r, c in mouth
                    if int(grid[r, c]) == YELLOW
                ]
                foreign = [
                    (r, c)
                    for r, c in mouth
                    if int(grid[r, c]) not in (BACKGROUND, YELLOW, color)
                ]
                if content_cells and not foreign:
                    valid.append(
                        {
                            "border_color": color,
                            "tail_dir": tail_dir,
                            "spike": spike,
                            "content_cells": sorted(content_cells),
                        }
                    )
            if len(valid) == 1:
                cups.append(valid[0])
    return cups


def _path_hits(shape: tuple[int, int], cups: list[dict], counts: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    vertical = np.zeros((h, w), dtype=int)
    horizontal = np.zeros((h, w), dtype=int)
    for idx, cup in enumerate(cups):
        steps = int(counts[idx]) if counts is not None else len(cup["content_cells"])
        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        target = vertical if cup["tail_dir"] in ("U", "D") else horizontal
        for step in range(1, steps + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            target[rr, cc] += 1
    return vertical, horizontal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject the yellow cup contents from top and left cups, then mark vertical/horizontal path crossings in orange.

    Input structure:
    - The background is black.
    - Near the top edge sit several cup frames whose tails point downward; near the left edge sit several cup frames whose tails point rightward.
    - Every cup frame has its own border color, and each cup contains one or more yellow content pixels in the mouth side opposite the tail.

    Rule:
    - For each cup, count its yellow content pixels and erase those interior yellow cells back to black.
    - Starting from the tail spike, draw a straight ray in the tail direction with one yellow pixel per erased content pixel, clipping anything beyond the grid edge.
    - Any cell reached by at least one vertical ejection ray and at least one horizontal ejection ray becomes orange; cells reached by only one orientation stay yellow.
    - Keep all cup-border pixels unchanged.

    Key insight:
    - The orange pixels are not emitted directly by any one cup: they are exactly the Cartesian intersections of count-limited vertical and horizontal ejection segments.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cups = _detect_cups(g)

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = BACKGROUND

    vertical, horizontal = _path_hits(g.shape, cups)
    collisions = (vertical > 0) & (horizontal > 0)
    single_axis = ((vertical > 0) | (horizontal > 0)) & ~collisions
    paintable = out == BACKGROUND
    out[collisions & paintable] = ORANGE
    out[single_axis & paintable] = YELLOW
    return out


def _rotate_point_ccw(point: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = point
    return w - 1 - c, r


def _rotate_points(points: list[tuple[int, int]], h: int, w: int, turns: int) -> tuple[list[tuple[int, int]], int, int]:
    pts = list(points)
    hh, ww = h, w
    for _ in range(turns % 4):
        pts = [_rotate_point_ccw(p, hh, ww) for p in pts]
        hh, ww = ww, hh
    return pts, hh, ww


def _build_cup_patch(interior_h: int, interior_w: int, border_color: int, orientation: str, tail_col: int) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    base_h = interior_h + 2
    base_w = interior_w + 2
    base = np.zeros((base_h, base_w), dtype=int)
    for r in range(interior_h + 1):
        base[r, 0] = border_color
        base[r, base_w - 1] = border_color
    base[interior_h, :] = border_color
    base[base_h - 1, tail_col] = border_color

    interior = [(r, c) for r in range(interior_h) for c in range(1, interior_w + 1)]
    spike = (base_h - 1, tail_col)
    turns = {"D": 0, "R": 1, "U": 2, "L": 3}[orientation]
    patch = np.rot90(base, turns)
    interior, _, _ = _rotate_points(interior, base_h, base_w, turns)
    spike_list, _, _ = _rotate_points([spike], base_h, base_w, turns)
    return patch, interior, spike_list[0]


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BACKGROUND
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _place_band(length: int, sizes: list[int], start_min: int) -> list[int] | None:
    order = list(range(len(sizes)))
    random.shuffle(order)
    starts = [0] * len(sizes)
    used: list[tuple[int, int]] = []
    for idx in order:
        size = sizes[idx]
        choices = []
        for start in range(start_min, length - size + 1):
            end = start + size - 1
            if all(end < s or start > e for s, e in used):
                choices.append(start)
        if not choices:
            return None
        chosen = random.choice(choices)
        starts[idx] = chosen
        used.append((chosen, chosen + size - 1))
    return starts


def _make_cup_defs(count: int, orientation: str, colors: list[int]) -> list[dict]:
    defs = []
    for color in colors[:count]:
        interior_h = random.randint(1, 2)
        interior_w = random.randint(1, 2)
        tail_col = random.randint(1, interior_w)
        patch, interior, spike = _build_cup_patch(interior_h, interior_w, color, orientation, tail_col)
        defs.append(
            {
                "orientation": orientation,
                "patch": patch,
                "interior_local": interior,
                "spike_local": spike,
                "capacity": len(interior),
            }
        )
    return defs


def _globalize(defn: dict, top: int, left: int) -> dict:
    patch = defn["patch"]
    interiors = [(top + r, left + c) for r, c in defn["interior_local"]]
    spike = (top + defn["spike_local"][0], left + defn["spike_local"][1])
    return {
        "orientation": defn["orientation"],
        "patch": patch,
        "top": top,
        "left": left,
        "spike": spike,
        "interiors": interiors,
        "capacity": defn["capacity"],
    }


def _cup_bbox(cup: dict) -> tuple[int, int, int, int]:
    h, w = cup["patch"].shape
    return cup["top"], cup["left"], cup["top"] + h - 1, cup["left"] + w - 1


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _frame_cells(cup: dict) -> set[tuple[int, int]]:
    patch = cup["patch"]
    top = cup["top"]
    left = cup["left"]
    return {
        (top + r, left + c)
        for r, c in np.argwhere(patch != BACKGROUND)
    }


def _layout_is_clear(shape: tuple[int, int], top_cups: list[dict], left_cups: list[dict]) -> bool:
    all_cups = top_cups + left_cups
    bboxes = [_cup_bbox(cup) for cup in all_cups]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _bboxes_overlap(bboxes[i], bboxes[j]):
                return False

    frame_by_index = [_frame_cells(cup) for cup in all_cups]
    frame_union = set().union(*frame_by_index) if frame_by_index else set()
    h, w = shape
    for idx, cup in enumerate(all_cups):
        own = frame_by_index[idx]
        others = frame_union - own
        dr, dc = DIR4[cup["orientation"]]
        sr, sc = cup["spike"]
        for step in range(1, cup["capacity"] + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if (rr, cc) in others:
                return False
    return True


def _pair_intersections(top_cups: list[dict], left_cups: list[dict]) -> set[tuple[int, int]]:
    pairs = set()
    for tc in top_cups:
        tr, tc_col = tc["spike"]
        for lc in left_cups:
            lr, lc_col = lc["spike"]
            if lr > tr and tc_col > lc_col:
                pairs.add((lr, tc_col))
    return pairs


def _analyze_choice(shape: tuple[int, int], top_cups: list[dict], left_cups: list[dict], top_counts: tuple[int, ...], left_counts: tuple[int, ...]) -> dict:
    h, w = shape
    vertical = np.zeros((h, w), dtype=int)
    for cup, count in zip(top_cups, top_counts):
        sr, sc = cup["spike"]
        for step in range(1, count + 1):
            rr = sr + step
            if rr >= h:
                break
            vertical[rr, sc] += 1

    horizontal = np.zeros((h, w), dtype=int)
    for cup, count in zip(left_cups, left_counts):
        sr, sc = cup["spike"]
        for step in range(1, count + 1):
            cc = sc + step
            if cc >= w:
                break
            horizontal[sr, cc] += 1

    collisions_mask = (vertical > 0) & (horizontal > 0)
    yellow_mask = ((vertical > 0) | (horizontal > 0)) & ~collisions_mask
    collisions = {tuple(p) for p in np.argwhere(collisions_mask)}
    all_pairs = _pair_intersections(top_cups, left_cups)
    return {
        "vertical": vertical,
        "horizontal": horizontal,
        "collisions": collisions,
        "yellow_count": int(np.count_nonzero(yellow_mask)),
        "all_pairs": all_pairs,
        "valid": bool(collisions) and len(collisions) < len(all_pairs) and int(np.count_nonzero(yellow_mask)) > 0,
    }


def _choose_counts(shape: tuple[int, int], top_cups: list[dict], left_cups: list[dict]) -> tuple[tuple[int, ...], tuple[int, ...], dict] | None:
    top_ranges = [range(1, cup["capacity"] + 1) for cup in top_cups]
    left_ranges = [range(1, cup["capacity"] + 1) for cup in left_cups]
    candidates: list[tuple[tuple[int, ...], tuple[int, ...], dict, tuple[int, int, int]]] = []

    for top_counts in itertools.product(*top_ranges):
        for left_counts in itertools.product(*left_ranges):
            analysis = _analyze_choice(shape, top_cups, left_cups, top_counts, left_counts)
            if not analysis["valid"]:
                continue
            distinct = len(set(top_counts + left_counts))
            score = (
                distinct,
                analysis["yellow_count"],
                len(analysis["all_pairs"]) - len(analysis["collisions"]),
            )
            candidates.append((top_counts, left_counts, analysis, score))

    if not candidates:
        return None

    best_score = max(item[3] for item in candidates)
    best = [item for item in candidates if item[3] == best_score]
    top_counts, left_counts, analysis, _ = random.choice(best)
    return top_counts, left_counts, analysis


def _render_input(shape: tuple[int, int], top_cups: list[dict], left_cups: list[dict], top_counts: tuple[int, ...], left_counts: tuple[int, ...]) -> tuple[np.ndarray, dict]:
    grid = np.full(shape, BACKGROUND, dtype=int)
    chosen_content: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for cup in top_cups + left_cups:
        _paste_patch(grid, cup["patch"], cup["top"], cup["left"])

    for cup, count in zip(top_cups, top_counts):
        cells = random.sample(cup["interiors"], count)
        for r, c in cells:
            grid[r, c] = YELLOW
        chosen_content[cup["spike"]] = cells

    for cup, count in zip(left_cups, left_counts):
        cells = random.sample(cup["interiors"], count)
        for r, c in cells:
            grid[r, c] = YELLOW
        chosen_content[cup["spike"]] = cells

    return grid, chosen_content


def _expected_output(inp: np.ndarray, top_cups: list[dict], left_cups: list[dict], top_counts: tuple[int, ...], left_counts: tuple[int, ...], chosen_content: dict[tuple[int, int], list[tuple[int, int]]]) -> np.ndarray:
    out = inp.copy()
    for cells in chosen_content.values():
        for r, c in cells:
            out[r, c] = BACKGROUND

    vertical = np.zeros(inp.shape, dtype=int)
    horizontal = np.zeros(inp.shape, dtype=int)
    for cup, count in zip(top_cups, top_counts):
        sr, sc = cup["spike"]
        for step in range(1, count + 1):
            rr = sr + step
            if rr >= inp.shape[0]:
                break
            vertical[rr, sc] += 1

    for cup, count in zip(left_cups, left_counts):
        sr, sc = cup["spike"]
        for step in range(1, count + 1):
            cc = sc + step
            if cc >= inp.shape[1]:
                break
            horizontal[sr, cc] += 1

    collisions = (vertical > 0) & (horizontal > 0)
    yellow = ((vertical > 0) | (horizontal > 0)) & ~collisions
    out[collisions & (out == BACKGROUND)] = ORANGE
    out[yellow & (out == BACKGROUND)] = YELLOW
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    top_count = random.choice(COUNT_OPTIONS)
    left_count = random.choice(COUNT_OPTIONS)
    total = top_count + left_count
    if total > len(BORDER_COLORS):
        return None

    colors = random.sample(BORDER_COLORS, total)
    top_defs = _make_cup_defs(top_count, "D", colors[:top_count])
    left_defs = _make_cup_defs(left_count, "R", colors[top_count:])

    top_widths = [int(defn["patch"].shape[1]) for defn in top_defs]
    left_heights = [int(defn["patch"].shape[0]) for defn in left_defs]
    top_cols = _place_band(w, top_widths, start_min=4)
    left_rows = _place_band(h, left_heights, start_min=4)
    if top_cols is None or left_rows is None:
        return None

    top_cups = []
    for defn, col in zip(top_defs, top_cols):
        row = random.randint(0, 1)
        top_cups.append(_globalize(defn, row, col))

    left_cups = []
    for defn, row in zip(left_defs, left_rows):
        col = random.randint(0, 1)
        left_cups.append(_globalize(defn, row, col))

    if len({cup["spike"][1] for cup in top_cups}) != len(top_cups):
        return None
    if len({cup["spike"][0] for cup in left_cups}) != len(left_cups):
        return None

    if any(cup["spike"][1] <= 3 for cup in top_cups):
        return None
    if any(cup["spike"][0] <= 3 for cup in left_cups):
        return None
    if not _layout_is_clear((h, w), top_cups, left_cups):
        return None

    choice = _choose_counts((h, w), top_cups, left_cups)
    if choice is None:
        return None
    top_counts, left_counts, analysis = choice

    inp, chosen_content = _render_input((h, w), top_cups, left_cups, top_counts, left_counts)
    expected = _expected_output(inp, top_cups, left_cups, top_counts, left_counts, chosen_content)

    if not np.any(expected == ORANGE):
        return None
    if not np.any(expected == YELLOW):
        return None
    if len(analysis["collisions"]) >= len(analysis["all_pairs"]):
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("generate_input: failed to build a valid dual-axis ejection instance")
