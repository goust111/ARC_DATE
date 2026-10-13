from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, paint_coords


POS_COLOR = 8  # teal / cyan bars
NEG_COLOR = 2  # red bars
BAR_COLORS = {POS_COLOR, NEG_COLOR}
GEN_ATTEMPTS = 400
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (20, 28)
MAX_BAR_HEIGHT = 7
TARGET_N_RANGE = (6, 11)
BAR_COUNT_RANGE = (4, 7)
OUTLINE_GAP = 1


def _row_span_interior(
    grid: np.ndarray | None,
    coords: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...],
    bg: int | None = None,
) -> tuple[tuple[int, int], ...]:
    cells = {(int(r), int(c)) for r, c in coords}
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(int(r), []).append(int(c))

    interior: list[tuple[int, int]] = []
    for r in sorted(by_row):
        cols = sorted(by_row[r])
        if len(cols) < 2:
            continue
        for c in range(cols[0] + 1, cols[-1]):
            if (r, c) in cells:
                continue
            if grid is not None:
                assert bg is not None and int(grid[r, c]) == int(bg)
            interior.append((r, c))
    return tuple(interior)


def _bar_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, _w = grid.shape
    bars: list[dict] = []
    for color, sign in ((POS_COLOR, 1), (NEG_COLOR, -1)):
        components = find_components(
            grid,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        for comp in components:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            area = int(comp["area"])
            if r1 != h - 1:
                continue
            if c0 != c1:
                continue
            if area != r1 - r0 + 1:
                continue
            bars.append(
                {
                    "color": int(color),
                    "sign": int(sign),
                    "height": area,
                    "coords": comp["coords"],
                    "col": c0,
                }
            )

    bar_pixel_count = sum(int(bar["height"]) for bar in bars)
    assert bar_pixel_count == int(np.count_nonzero(np.isin(grid, [POS_COLOR, NEG_COLOR])))
    assert bars and any(bar["color"] == POS_COLOR for bar in bars)
    assert any(bar["color"] == NEG_COLOR for bar in bars)
    return sorted(bars, key=lambda bar: int(bar["col"]))


def _outline_components(grid: np.ndarray, bg: int) -> list[dict]:
    outline_colors = sorted(set(map(int, np.unique(grid))) - {int(bg), POS_COLOR, NEG_COLOR})
    components: list[dict] = []
    for color in outline_colors:
        comps = find_components(
            grid,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[color],
        )
        assert len(comps) == 1, "each outline color should form exactly one object"
        comp = comps[0]
        interior = _row_span_interior(grid, comp["coords"], bg)
        components.append(
            {
                "color": int(color),
                "coords": comp["coords"],
                "bbox": tuple(map(int, comp["bbox"])),
                "area": int(comp["area"]),
                "interior": interior,
                "cavity": len(interior),
            }
        )
    assert len(components) >= 3
    return components


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    bars = _bar_components(g, bg)
    n_value = sum(int(bar["sign"]) * int(bar["height"]) for bar in bars)
    assert n_value > 0

    outlines = _outline_components(g, bg)
    candidates = [outline for outline in outlines if int(outline["cavity"]) == n_value]
    assert len(candidates) == 1, "expected exactly one cavity-size match"
    return {
        "grid": g,
        "bg": bg,
        "bars": bars,
        "outlines": outlines,
        "N": n_value,
        "target": candidates[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the unique outline cavity whose row-span area matches the signed bottom-bar equation.

    Input structure:
    - A dominant background fills the grid.
    - On the bottom edge stand several width-1 vertical bars in teal and red, all bottom-aligned.
    - Above them sit several single-color hollow outline objects in distinct colors.

    Rule:
    - Compute N as the sum of teal bar heights minus the sum of red bar heights.
    - For each outline object, compute its row-span interior: on every occupied row, take the
      background cells strictly between the leftmost and rightmost outline pixels.
    - Find the unique outline whose interior area equals N, fill that interior with the outline's
      own color, and erase all bottom bars.

    Key insight:
    - The bars encode an arithmetic value, and that value matches an invisible cavity area rather
      than the outline's visible pixel count, so a same-count outline border can be a decoy.
    """
    parsed = _parse_instance(np.asarray(grid, dtype=int))
    out = parsed["grid"].copy()
    bg = int(parsed["bg"])

    for bar in parsed["bars"]:
        for r, c in bar["coords"]:
            out[int(r), int(c)] = bg

    paint_coords(out, parsed["target"]["interior"], int(parsed["target"]["color"]))
    return out


def _build_u_outline(
    gap: int,
    arm_height: int,
    left_stub: int,
    right_stub: int,
    flip: bool,
) -> tuple[tuple[int, int], ...]:
    width = int(gap + 2 + left_stub + right_stub)
    arm_left = int(left_stub)
    arm_right = int(left_stub + gap + 1)
    coords: set[tuple[int, int]] = set()

    for r in range(int(arm_height)):
        coords.add((r, arm_left))
        coords.add((r, arm_right))
    cap_row = int(arm_height)
    for c in range(width):
        coords.add((cap_row, c))

    if flip:
        coords = {(cap_row - r, c) for r, c in coords}
    return tuple(sorted(coords))


def _spec_from_coords(coords: tuple[tuple[int, int], ...]) -> dict:
    arr = np.asarray(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    box_h = max(r for r, _ in norm) + 1
    box_w = max(c for _, c in norm) + 1
    cavity = len(_row_span_interior(None, norm))
    return {
        "coords": norm,
        "height": int(box_h),
        "width": int(box_w),
        "area": len(norm),
        "cavity": int(cavity),
    }


def _outline_catalog() -> list[dict]:
    catalog: list[dict] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for gap in range(1, 9):
        for arm_height in range(1, 5):
            for left_stub in (0, 1):
                for right_stub in (0, 1):
                    for flip in (False, True):
                        spec = _spec_from_coords(
                            _build_u_outline(gap, arm_height, left_stub, right_stub, flip)
                        )
                        if spec["coords"] in seen:
                            continue
                        if spec["height"] > 6 or spec["width"] > 10:
                            continue
                        if spec["cavity"] <= 0:
                            continue
                        seen.add(spec["coords"])
                        catalog.append(spec)
    return catalog


OUTLINE_CATALOG = _outline_catalog()


def _choose_outline_specs(target_n: int) -> list[dict] | None:
    targets = [spec for spec in OUTLINE_CATALOG if spec["cavity"] == target_n and spec["area"] != target_n]
    decoys = [spec for spec in OUTLINE_CATALOG if spec["area"] == target_n and spec["cavity"] != target_n]
    larger = [spec for spec in OUTLINE_CATALOG if spec["cavity"] > target_n]
    others = [spec for spec in OUTLINE_CATALOG if spec["cavity"] != target_n]
    random.shuffle(targets)
    random.shuffle(decoys)
    random.shuffle(larger)
    random.shuffle(others)

    for target in targets:
        used_areas = {int(target["area"])}
        used_cavities = {int(target["cavity"])}
        for decoy in decoys:
            if int(decoy["area"]) in used_areas or int(decoy["cavity"]) in used_cavities:
                continue
            used_areas_2 = used_areas | {int(decoy["area"])}
            used_cavities_2 = used_cavities | {int(decoy["cavity"])}
            for high in larger:
                if int(high["area"]) in used_areas_2 or int(high["cavity"]) in used_cavities_2:
                    continue
                used_areas_3 = used_areas_2 | {int(high["area"])}
                used_cavities_3 = used_cavities_2 | {int(high["cavity"])}
                for other in others:
                    if int(other["area"]) in used_areas_3 or int(other["cavity"]) in used_cavities_3:
                        continue
                    return [
                        {**target, "role": "target"},
                        {**decoy, "role": "decoy"},
                        {**high, "role": "high"},
                        {**other, "role": "other"},
                    ]
    return None


def _bbox_with_gap(bbox: tuple[int, int, int, int], gap: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 - gap, c0 - gap, r1 + gap, c1 + gap)


def _bboxes_overlap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def _place_outlines(h: int, w: int, top_limit: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    order = sorted(specs, key=lambda spec: (spec["width"] * spec["height"], spec["width"]), reverse=True)
    for spec in order:
        ok = False
        row_choices = list(range(1, top_limit - int(spec["height"]) + 1))
        col_choices = list(range(1, w - int(spec["width"])))
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        for r0 in row_choices:
            for c0 in col_choices:
                bbox = (r0, c0, r0 + int(spec["height"]) - 1, c0 + int(spec["width"]) - 1)
                grown = _bbox_with_gap(bbox, OUTLINE_GAP)
                if any(_bboxes_overlap(grown, _bbox_with_gap(item["bbox"], OUTLINE_GAP)) for item in placed):
                    continue
                placed.append({**spec, "top": r0, "left": c0, "bbox": bbox})
                ok = True
                break
            if ok:
                break
        if not ok:
            return None
    random.shuffle(placed)
    return placed


def _random_partition(total: int, parts: int, max_part: int) -> list[int] | None:
    if parts <= 0 or total < parts or total > parts * max_part:
        return None
    out: list[int] = []
    remaining = int(total)
    for i in range(parts):
        slots = parts - i - 1
        lo = max(1, remaining - max_part * slots)
        hi = min(max_part, remaining - slots)
        if lo > hi:
            return None
        out.append(random.randint(lo, hi))
        remaining -= out[-1]
    random.shuffle(out)
    return out


def _mixed_bar_sequence(bars: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    need_transitions = 2 if sum(color == POS_COLOR for color, _ in bars) > 1 and sum(color == NEG_COLOR for color, _ in bars) > 1 else 1
    for _ in range(80):
        seq = list(bars)
        random.shuffle(seq)
        transitions = sum(int(seq[i][0] != seq[i - 1][0]) for i in range(1, len(seq)))
        if transitions >= need_transitions:
            return seq
    return None


def _build_bar_specs(target_n: int) -> list[tuple[int, int]] | None:
    for _ in range(160):
        n_neg = random.randint(1, 3)
        n_pos = random.randint(2, 4)
        if not (BAR_COUNT_RANGE[0] <= n_neg + n_pos <= BAR_COUNT_RANGE[1]):
            continue
        neg_total = random.randint(n_neg, min(n_neg * MAX_BAR_HEIGHT, 6))
        pos_total = target_n + neg_total
        pos_parts = _random_partition(pos_total, n_pos, MAX_BAR_HEIGHT)
        neg_parts = _random_partition(neg_total, n_neg, MAX_BAR_HEIGHT)
        if pos_parts is None or neg_parts is None:
            continue
        seq = _mixed_bar_sequence([(POS_COLOR, h) for h in pos_parts] + [(NEG_COLOR, h) for h in neg_parts])
        if seq is not None:
            return seq
    return None


def _paint_outline(grid: np.ndarray, spec: dict, color: int) -> tuple[tuple[int, int], ...]:
    coords = tuple((int(spec["top"]) + r, int(spec["left"]) + c) for r, c in spec["coords"])
    paint_coords(grid, coords, color)
    return coords


def _paint_bars(grid: np.ndarray, bars: list[tuple[int, int]], columns: list[int]) -> None:
    h = grid.shape[0]
    for (color, height), col in zip(bars, columns):
        for r in range(h - int(height), h):
            grid[r, int(col)] = int(color)


def _expected_output(grid: np.ndarray, bg: int, target_color: int, target_interior: tuple[tuple[int, int], ...]) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[np.isin(out, [POS_COLOR, NEG_COLOR])] = int(bg)
    paint_coords(out, target_interior, int(target_color))
    return out


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        parsed = _parse_instance(grid)
    except AssertionError:
        return False

    outlines = parsed["outlines"]
    cavities = [int(item["cavity"]) for item in outlines]
    areas = [int(item["area"]) for item in outlines]
    if len(set(cavities)) != len(cavities):
        return False
    if len(set(areas)) != len(areas):
        return False
    if not any(int(item["area"]) == int(parsed["N"]) and int(item["cavity"]) != int(parsed["N"]) for item in outlines):
        return False
    if min(cavities) >= int(parsed["N"]) or max(cavities) <= int(parsed["N"]):
        return False
    color_seq = [int(bar["color"]) for bar in parsed["bars"]]
    if sum(int(color_seq[i] != color_seq[i - 1]) for i in range(1, len(color_seq))) < 1:
        return False
    try:
        if not np.array_equal(main(grid), expected):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    target_n = random.randint(*TARGET_N_RANGE)
    specs = _choose_outline_specs(target_n)
    if specs is None:
        return None
    bars = _build_bar_specs(target_n)
    if bars is None:
        return None

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if max(spec["width"] for spec in specs) >= w - 2:
        return None
    if len(bars) > len(range(1, w - 1, 2)):
        return None

    max_bar_height = max(height for _color, height in bars)
    top_limit = h - max_bar_height - 3
    if top_limit < max(spec["height"] for spec in specs) + 1:
        return None

    placements = _place_outlines(h, w, top_limit, specs)
    if placements is None:
        return None

    bg_choices = [color for color in range(10) if color not in BAR_COLORS]
    bg = random.choice(bg_choices)
    outline_colors = random.sample([color for color in range(10) if color not in BAR_COLORS | {bg}], len(placements))
    grid = np.full((h, w), int(bg), dtype=int)

    target_color = -1
    target_interior: tuple[tuple[int, int], ...] = ()
    for spec, color in zip(placements, outline_colors):
        coords = _paint_outline(grid, spec, color)
        interior = tuple((int(spec["top"]) + r, int(spec["left"]) + c) for r, c in _row_span_interior(None, spec["coords"]))
        if spec["role"] == "target":
            target_color = int(color)
            target_interior = interior
        assert len(interior) == int(spec["cavity"])
        assert len(coords) == int(spec["area"])

    bar_columns = sorted(random.sample(list(range(1, w - 1, 2)), len(bars)))
    _paint_bars(grid, bars, bar_columns)

    expected = _expected_output(grid, bg, target_color, target_interior)
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
