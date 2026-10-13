from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MIN_GRID = 18
MAX_GRID = 24
MIN_SIZE = 6
MAX_SIZE = 8
MIN_FRAMES = 2
MAX_FRAMES = 3
MAX_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _gap_side_and_dir(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    gap_r, gap_c = gap
    r0, c0, r1, c1 = bbox
    if gap_r == r0 and c0 < gap_c < c1:
        return "top", DIRS["top"]
    if gap_r == r1 and c0 < gap_c < c1:
        return "bottom", DIRS["bottom"]
    if gap_c == c0 and r0 < gap_r < r1:
        return "left", DIRS["left"]
    if gap_c == c1 and r0 < gap_r < r1:
        return "right", DIRS["right"]
    raise AssertionError("gap must be a non-corner border cell")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components_of_color(grid, FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < MIN_SIZE or c1 - c0 + 1 < MIN_SIZE:
            continue
        border = _border_cells(r0, c0, r1, c1)
        coords_set = set(comp)
        if not coords_set <= border:
            continue
        missing = list(border - coords_set)
        if len(missing) != 1:
            continue
        gap = missing[0]
        side, direction = _gap_side_and_dir(gap, (r0, c0, r1, c1))
        if len(coords_set) != len(border) - 1:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "direction": direction,
            }
        )
    return frames


def _row_majority(row: np.ndarray) -> int:
    values = [int(v) for v in row if int(v) != BG]
    assert values, "interior row must contain non-background colors"
    counts = Counter(values)
    best = max(counts.values())
    winners = {color for color, count in counts.items() if count == best}
    for value in map(int, row):
        if value in winners:
            return value
    raise AssertionError("unreachable majority resolution")


def _compress(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each gap-frame's compressed row-majority source color to flood-fill the box and spill through its gap.

    Input structure:
    - The background is black and the special objects are 1-pixel-thick grey rectangular frames.
    - Each grey frame has exactly one missing non-corner border cell acting as a gap.
    - Every frame interior is fully occupied by a multicolor pattern whose rows encode a row-majority sequence.

    Rule:
    - Detect every grey one-gap rectangle and read its interior rows from top to bottom.
    - For each row, take the most frequent non-background color; if several colors tie, choose the one whose first occurrence is leftmost in that row.
    - Compress consecutive duplicate row-majority colors, take the first compressed color, repaint the whole interior with it, and paint the gap plus the outward background ray with that same color until blocked or out of bounds.

    Key insight:
    - The noisy interior is both the information source and the target: the frame's own row-majority compression decides the fill color that then replaces the pattern and extends outward through the unique gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    frames = _extract_frames(g)
    assert frames, "expected at least one gap frame"

    h, w = g.shape
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0
        majorities = [_row_majority(inner[r]) for r in range(inner.shape[0])]
        compressed = _compress(majorities)
        assert compressed
        fill = compressed[0]

        out[r0 + 1 : r1, c0 + 1 : c1] = fill

        dr, dc = frame["direction"]
        rr, cc = frame["gap"]
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BG:
            out[rr, cc] = fill
            rr += dr
            cc += dc

    return out


def _gap_from_side(r0: int, c0: int, height: int, width: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r0 + height - 1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c0 + width - 1


def _ray_cells(shape: tuple[int, int], gap: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIRS[side]
    rr, cc = gap
    cells: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap >= 1 or col_gap >= 1


def _partition_positive(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _build_row(width: int, majority: int, others: list[int], *, force_tie: bool, force_first_not_majority: bool) -> tuple[np.ndarray, bool, bool]:
    assert width >= 4
    tie_used = False
    first_not_majority_used = False

    if force_tie:
        alt = others[0]
        row: list[int]
        if width % 2 == 1:
            assert len(others) > 1
            filler = others[1] if len(others) > 1 else others[0]
            row = [filler]
            slots = width - 1
        else:
            row = []
            slots = width
        for i in range(slots):
            row.append(majority if i % 2 == 0 else alt)
        tie_used = True
        first_not_majority_used = row[0] != majority
        counts = Counter(row)
        assert counts[majority] == counts[alt]
        assert _row_majority(np.array(row, dtype=int)) == majority
        return np.array(row, dtype=int), tie_used, first_not_majority_used

    if force_first_not_majority:
        row = [others[0]] + [majority] * (width - 1)
        if width >= 6 and len(others) > 1:
            row[2] = others[1]
        first_not_majority_used = True
    else:
        row = [majority] * width
        minority_slots = 1 if width == 4 else random.randint(1, 2)
        positions = list(range(1, width))
        random.shuffle(positions)
        positions = positions[:minority_slots]
        for index, pos in enumerate(positions):
            row[pos] = others[index % len(others)]

    if row[0] != majority:
        first_not_majority_used = True

    counts = Counter(row)
    best = max(counts.values())
    winners = {color for color, count in counts.items() if count == best}
    chosen = next(value for value in row if value in winners)
    assert chosen == majority
    return np.array(row, dtype=int), tie_used, first_not_majority_used


def _build_pattern(inner_h: int, inner_w: int, palette: list[int], require_tie: bool, require_first_not_majority: bool) -> tuple[np.ndarray, list[int], dict]:
    compressed_len = random.randint(2, min(4, inner_h))
    compressed = [palette[0]]
    while len(compressed) < compressed_len:
        color = random.choice(palette)
        if color != compressed[-1]:
            compressed.append(color)

    segment_lengths = _partition_positive(inner_h, compressed_len)
    majorities: list[int] = []
    for color, length in zip(compressed, segment_lengths):
        majorities.extend([color] * length)

    pattern = np.zeros((inner_h, inner_w), dtype=int)
    tie_done = False
    first_not_majority_done = False
    for r, majority in enumerate(majorities):
        others = [color for color in palette if color != majority]
        if not others:
            others = [majority]
        force_tie = require_tie and not tie_done and r == 0
        force_first_not_majority = require_first_not_majority and not first_not_majority_done and (
            r == 0 or (r == inner_h - 1 and not force_tie)
        )
        row, used_tie, used_first_not_majority = _build_row(
            inner_w,
            majority,
            others,
            force_tie=force_tie,
            force_first_not_majority=force_first_not_majority,
        )
        pattern[r] = row
        tie_done = tie_done or used_tie
        first_not_majority_done = first_not_majority_done or used_first_not_majority

    actual_majorities = [_row_majority(pattern[r]) for r in range(inner_h)]
    actual_compressed = _compress(actual_majorities)
    assert actual_majorities == majorities
    assert actual_compressed == compressed
    assert len(actual_compressed) >= 2

    info = {
        "fill": compressed[0],
        "majorities": actual_majorities,
        "compressed": actual_compressed,
        "has_tie_row": tie_done,
        "has_first_not_majority_row": first_not_majority_done,
    }
    return pattern, compressed, info


def _manual_output(input_grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    h, w = input_grid.shape
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        fill = spec["fill"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
        dr, dc = DIRS[spec["side"]]
        rr, cc = spec["gap"]
        while 0 <= rr < h and 0 <= cc < w and int(input_grid[rr, cc]) == BG:
            out[rr, cc] = fill
            rr += dr
            cc += dc
    return out


def _validate_scene(grid: np.ndarray, specs: list[dict], *, require_tie: bool, require_first_not_majority: bool) -> None:
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    seen_boxes: list[tuple[int, int, int, int]] = []
    seen_rays: set[tuple[int, int]] = set()
    tie_seen = False
    first_not_majority_seen = False

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert MIN_SIZE <= h <= MAX_SIZE
        assert MIN_SIZE <= w <= MAX_SIZE
        for other in seen_boxes:
            assert _boxes_separated(spec["bbox"], other)
        seen_boxes.append(spec["bbox"])

        border = _border_cells(r0, c0, r1, c1)
        gap = spec["gap"]
        assert gap in border
        side, direction = _gap_side_and_dir(gap, spec["bbox"])
        assert side == spec["side"]
        assert direction == DIRS[spec["side"]]

        grey_cells = {(rr, cc) for rr, cc in border if (rr, cc) != gap}
        for rr, cc in grey_cells:
            assert int(grid[rr, cc]) == FRAME
        assert int(grid[gap]) == BG

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        majorities = [_row_majority(inner[r]) for r in range(inner.shape[0])]
        compressed = _compress(majorities)
        assert compressed == spec["compressed"]
        assert compressed[0] == spec["fill"]
        assert len(compressed) >= 2

        for row in inner:
            assert np.all(row != BG)
            assert np.all(row != FRAME)

        ray = _ray_cells(grid.shape, gap, side)
        for rr, cc in ray:
            assert (rr, cc) not in seen_rays
            seen_rays.add((rr, cc))
            assert int(grid[rr, cc]) == BG

        tie_seen = tie_seen or bool(spec["has_tie_row"])
        first_not_majority_seen = first_not_majority_seen or bool(spec["has_first_not_majority_row"])

    detected = _extract_frames(grid)
    assert len(detected) == len(specs)
    expected_keys = {(spec["bbox"], spec["gap"]) for spec in specs}
    detected_keys = {(frame["bbox"], frame["gap"]) for frame in detected}
    assert detected_keys == expected_keys

    if require_tie:
        assert tie_seen
    if require_first_not_majority:
        assert first_not_majority_seen

    expected = _manual_output(grid, specs)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        fill = spec["fill"]
        assert np.all(actual[r0 + 1 : r1, c0 + 1 : c1] == fill)
        for rr, cc in _border_cells(r0, c0, r1, c1):
            if (rr, cc) != spec["gap"]:
                assert int(actual[rr, cc]) == FRAME


def _sample_frame_spec(shape: tuple[int, int], side: str) -> dict:
    h, w = shape
    box_h = random.randint(MIN_SIZE, MAX_SIZE)
    box_w = random.randint(MIN_SIZE, MAX_SIZE)
    if side == "top":
        r0 = random.randint(1, max(1, h // 3))
        c0 = random.randint(1, w - box_w - 1)
        offset = random.randint(1, box_w - 2)
    elif side == "bottom":
        r0 = random.randint(max(1, h - box_h - max(1, h // 3)), h - box_h - 1)
        c0 = random.randint(1, w - box_w - 1)
        offset = random.randint(1, box_w - 2)
    elif side == "left":
        r0 = random.randint(1, h - box_h - 1)
        c0 = random.randint(1, max(1, w // 3))
        offset = random.randint(1, box_h - 2)
    else:
        r0 = random.randint(1, h - box_h - 1)
        c0 = random.randint(max(1, w - box_w - max(1, w // 3)), w - box_w - 1)
        offset = random.randint(1, box_h - 2)

    gap = _gap_from_side(r0, c0, box_h, box_w, side, offset)
    ray = _ray_cells(shape, gap, side)
    bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
    return {
        "bbox": bbox,
        "gap": gap,
        "side": side,
        "ray": set(ray),
        "box_cells": {(rr, cc) for rr in range(bbox[0], bbox[2] + 1) for cc in range(bbox[1], bbox[3] + 1)},
    }


def _compatible(candidate: dict, selected: list[dict]) -> bool:
    for other in selected:
        if not _boxes_separated(candidate["bbox"], other["bbox"]):
            return False
        if candidate["ray"] & other["box_cells"]:
            return False
        if other["ray"] & candidate["box_cells"]:
            return False
        if candidate["ray"] & other["ray"]:
            return False
    return True


def _attempt_input() -> np.ndarray | None:
    height = random.randint(MIN_GRID, MAX_GRID)
    width = random.randint(MIN_GRID, MAX_GRID)
    grid = np.full((height, width), BG, dtype=int)

    frame_count = random.randint(MIN_FRAMES, MAX_FRAMES)
    sides = random.sample(SIDES, frame_count)
    require_tie = True
    require_first_not_majority = True

    placed: list[dict] = []
    for side in sides:
        for _ in range(80):
            candidate = _sample_frame_spec((height, width), side)
            if _compatible(candidate, placed):
                placed.append(candidate)
                break
        else:
            return None

    all_colors = list(COLOR_POOL)
    random.shuffle(all_colors)

    for index, spec in enumerate(placed):
        r0, c0, r1, c1 = spec["bbox"]
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        inner_h = box_h - 2
        inner_w = box_w - 2

        need_tie_here = require_tie and index == 0
        need_first_not_majority_here = require_first_not_majority and (index == 0 or frame_count == 2 and index == 1)
        palette_size = 3 if (index == 0 and frame_count == 3) or random.random() < 0.45 else 2
        if need_tie_here and inner_w % 2 == 1:
            palette_size = max(palette_size, 3)
        if len(all_colors) < palette_size:
            all_colors = list(COLOR_POOL)
            random.shuffle(all_colors)
        palette = [all_colors.pop() for _ in range(palette_size)]

        pattern, compressed, info = _build_pattern(
            inner_h,
            inner_w,
            palette,
            require_tie=need_tie_here,
            require_first_not_majority=need_first_not_majority_here,
        )

        grid[r0:r1 + 1, c0:c1 + 1] = BG
        for rr, cc in _border_cells(r0, c0, r1, c1):
            if (rr, cc) != spec["gap"]:
                grid[rr, cc] = FRAME
        grid[r0 + 1 : r1, c0 + 1 : c1] = pattern

        spec["compressed"] = compressed
        spec["fill"] = info["fill"]
        spec["has_tie_row"] = info["has_tie_row"]
        spec["has_first_not_majority_row"] = info["has_first_not_majority_row"]

    try:
        _validate_scene(
            grid,
            placed,
            require_tie=require_tie,
            require_first_not_majority=require_first_not_majority,
        )
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
