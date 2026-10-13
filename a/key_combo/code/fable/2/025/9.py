from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
WALL_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
START_GAP = 2
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_by_color(grid: np.ndarray, skip_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in skip_colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return out


def intervals_where(line: np.ndarray, predicate) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(map(int, line)):
        if predicate(value):
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(line) - 1))
    return out


def detect_wall(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == WALL_COLOR
    box = bbox(mask)
    assert box is not None, "expected a gray wall"
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "wall must enclose an interior"
    wall_patch = crop_bbox(mask.astype(int), box).astype(bool)
    border = np.zeros_like(wall_patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.all(~wall_patch | border), "gray cells must stay on the rectangle border"
    assert wall_patch[0, 0] and wall_patch[0, -1] and wall_patch[-1, 0] and wall_patch[-1, -1], "corners must exist"
    return box


def detect_gaps(grid: np.ndarray, wall_box: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = wall_box
    top = intervals_where(grid[r0, c0 + 1 : c1], lambda v: v != WALL_COLOR)
    bottom = intervals_where(grid[r1, c0 + 1 : c1], lambda v: v != WALL_COLOR)
    left = intervals_where(grid[r0 + 1 : r1, c0], lambda v: v != WALL_COLOR)
    right = intervals_where(grid[r0 + 1 : r1, c1], lambda v: v != WALL_COLOR)
    return {
        "top": [(c0 + 1 + a, c0 + 1 + b) for a, b in top],
        "bottom": [(c0 + 1 + a, c0 + 1 + b) for a, b in bottom],
        "left": [(r0 + 1 + a, r0 + 1 + b) for a, b in left],
        "right": [(r0 + 1 + a, r0 + 1 + b) for a, b in right],
    }


def classify_side(block_box: tuple[int, int, int, int], wall_box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = block_box
    wr0, wc0, wr1, wc1 = wall_box
    candidates: list[str] = []
    if r1 < wr0 and c0 <= wc1 and c1 >= wc0:
        candidates.append("top")
    if r0 > wr1 and c0 <= wc1 and c1 >= wc0:
        candidates.append("bottom")
    if c1 < wc0 and r0 <= wr1 and r1 >= wr0:
        candidates.append("left")
    if c0 > wc1 and r0 <= wr1 and r1 >= wr0:
        candidates.append("right")
    assert len(candidates) == 1, "every block must face exactly one wall side"
    return candidates[0]


def match_gap(
    side: str,
    block_box: tuple[int, int, int, int],
    gaps: dict[str, list[tuple[int, int]]],
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = block_box
    if side in ("top", "bottom"):
        span0, span1 = c0, c1
        block_span = c1 - c0 + 1
    else:
        span0, span1 = r0, r1
        block_span = r1 - r0 + 1
    matches = []
    for g0, g1 in gaps[side]:
        o0 = max(span0, g0)
        o1 = min(span1, g1)
        if o0 <= o1:
            matches.append((g0, g1, o0, o1))
    assert len(matches) == 1, "each block must overlap exactly one gap on its side"
    g0, g1, o0, o1 = matches[0]
    assert block_span > (g1 - g0 + 1), "the block span must exceed the gap span"
    return g0, g1, o0, o1


def move_component_cells(
    coords: np.ndarray,
    block_box: tuple[int, int, int, int],
    side: str,
    overlap: tuple[int, int],
    wall_box: tuple[int, int, int, int],
) -> np.ndarray:
    wr0, wc0, wr1, wc1 = wall_box
    r0, c0, r1, c1 = block_box
    o0, o1 = overlap
    out: list[tuple[int, int]] = []
    for rr, cc in map(tuple, coords):
        if side == "top":
            dr = (wr1 - 1 - r1) if o0 <= cc <= o1 else (wr0 - 1 - r1)
            out.append((int(rr + dr), int(cc)))
        elif side == "bottom":
            dr = (wr0 + 1 - r0) if o0 <= cc <= o1 else (wr1 + 1 - r0)
            out.append((int(rr + dr), int(cc)))
        elif side == "left":
            dc = (wc1 - 1 - c1) if o0 <= rr <= o1 else (wc0 - 1 - c1)
            out.append((int(rr), int(cc + dc)))
        else:
            dc = (wc0 + 1 - c0) if o0 <= rr <= o1 else (wc1 + 1 - c0)
            out.append((int(rr), int(cc + dc)))
    return np.array(out, dtype=int)


def render_output(
    grid: np.ndarray,
    components: list[dict],
    wall_box: tuple[int, int, int, int],
    gaps: dict[str, list[tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for comp in components:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = BACKGROUND

    occupied = (out != BACKGROUND)
    for comp in components:
        side = classify_side(comp["bbox"], wall_box)
        _g0, _g1, o0, o1 = match_gap(side, comp["bbox"], gaps)
        moved = move_component_cells(comp["coords"], comp["bbox"], side, (o0, o1), wall_box)
        color = int(comp["color"])
        for rr, cc in moved:
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "moved block left the grid"
            assert not occupied[rr, cc], "moved blocks must not overlap"
            out[rr, cc] = color
            occupied[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each exterior rectangle at the wall gap and slide the two parts to different stopping depths.

    Input structure:
    - The background is black.
    - A large gray hollow rectangle sits near the center, with one or more straight non-corner gaps on its border.
    - Outside the wall are several solid single-color rectangles, each entirely on one side of the wall and facing it.

    Rule:
    - For every exterior rectangle, find the wall side it faces and the unique gap whose span overlaps the rectangle's span.
    - Move the overlap-aligned columns or rows straight through that gap until their front edge reaches the opposite inner wall.
    - Move the non-overlapping remainder straight toward the same wall only until it is flush with the outer wall.
    - Clear the original rectangle positions and keep all colors unchanged; the wall itself stays fixed except where passing colors occupy gap cells.

    Key insight:
    - The rectangle does not stop as one rigid piece: the gap cuts out only the aligned slice, so the same block is partitioned by span and the two resulting parts travel different distances along the wall normal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    wall_box = detect_wall(g)
    gaps = detect_gaps(g, wall_box)
    components = components_by_color(g, {get_bg(g), WALL_COLOR})
    assert components, "expected at least one colored block"
    for comp in components:
        box = comp["bbox"]
        patch = crop_bbox(g, box)
        color = int(comp["color"])
        assert np.all(patch == color), "every foreground block must be a solid rectangle"
        side = classify_side(box, wall_box)
        match_gap(side, box, gaps)
    return render_output(g, components, wall_box, gaps)


def classify_interval(side_total: int, gap: tuple[int, int], interval: tuple[int, int]) -> str | None:
    gs, ge = gap
    bs, be = interval
    gap_len = ge - gs + 1
    block_len = be - bs + 1
    if block_len <= gap_len:
        return None
    os = max(bs, gs)
    oe = min(be, ge)
    if os > oe:
        return None
    overlap_len = oe - os + 1
    if overlap_len == gap_len:
        if bs < gs and be > ge:
            return "full_center"
        if bs <= gs and be >= ge:
            return "full_edge"
        return None
    if bs < gs and be < ge:
        return "partial_left"
    if bs > gs and be > ge:
        return "partial_right"
    return None


def choose_side_set(rng: random.Random, sides: list[str]) -> list[str]:
    k = rng.choices([2, 3, 4], weights=[2, 3, 3], k=1)[0]
    if k == 2:
        pairs = [
            ["top", "left"],
            ["top", "right"],
            ["bottom", "left"],
            ["bottom", "right"],
            ["top", "bottom"],
            ["left", "right"],
        ]
        rng.shuffle(pairs)
        for pair in pairs:
            if all(side in sides for side in pair):
                return pair
    return rng.sample(sides, k=min(k, len(sides)))


def side_capacity(side: str, wall_box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int]:
    wr0, wc0, wr1, wc1 = wall_box
    h, w = shape
    if side == "top":
        return wr1 - wr0 + 1, wr0 - START_GAP
    if side == "bottom":
        return wr1 - wr0 + 1, (h - wr1 - 1) - START_GAP
    if side == "left":
        return wc1 - wc0 + 1, wc0 - START_GAP
    return wc1 - wc0 + 1, (w - wc1 - 1) - START_GAP


def side_global_span(side: str, wall_box: tuple[int, int, int, int]) -> tuple[int, int]:
    wr0, wc0, wr1, wc1 = wall_box
    if side in ("top", "bottom"):
        return wc0, wc1
    return wr0, wr1


def side_gap_candidates(
    side: str,
    wall_box: tuple[int, int, int, int],
    max_gap_len: int,
) -> list[tuple[int, int]]:
    wr0, wc0, wr1, wc1 = wall_box
    side_total = (wc1 - wc0 + 1) if side in ("top", "bottom") else (wr1 - wr0 + 1)
    out: list[tuple[int, int]] = []
    for gap_len in range(1, min(max_gap_len, side_total - 2) + 1):
        for gs in range(1, side_total - gap_len):
            ge = gs + gap_len - 1
            out.append((gs, ge))
    return out


def choose_interval_for_gap(
    rng: random.Random,
    side_total: int,
    gap_local: tuple[int, int],
    want_class: str,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for bs in range(side_total):
        for be in range(bs, side_total - 1 + 1):
            if classify_interval(side_total, gap_local, (bs, be)) == want_class:
                candidates.append((bs, be))
    if not candidates:
        return None
    rng.shuffle(candidates)
    return candidates[0]


def build_component_from_spec(
    side: str,
    span_global: tuple[int, int],
    motion_len: int,
    wall_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    wr0, wc0, wr1, wc1 = wall_box
    span0, span1 = span_global
    if side == "top":
        r1 = wr0 - START_GAP - 1
        r0 = r1 - motion_len + 1
        box = (r0, span0, r1, span1)
    elif side == "bottom":
        r0 = wr1 + START_GAP + 1
        r1 = r0 + motion_len - 1
        box = (r0, span0, r1, span1)
    elif side == "left":
        c1 = wc0 - START_GAP - 1
        c0 = c1 - motion_len + 1
        box = (span0, c0, span1, c1)
    else:
        c0 = wc1 + START_GAP + 1
        c1 = c0 + motion_len - 1
        box = (span0, c0, span1, c1)

    r0, c0, r1, c1 = box
    coords = np.array([(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)], dtype=int)
    return coords, box


def motion_overlap_global(
    wall_box: tuple[int, int, int, int],
    side: str,
    gap_local: tuple[int, int],
    interval_local: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    global_start, _global_end = side_global_span(side, wall_box)
    gs, ge = gap_local
    bs, be = interval_local
    gap_global = (global_start + gs, global_start + ge)
    overlap_global = (global_start + max(gs, bs), global_start + min(ge, be))
    span_global = (global_start + bs, global_start + be)
    return gap_global, overlap_global, span_global


def expected_output_from_specs(
    grid: np.ndarray,
    specs: list[dict],
    wall_box: tuple[int, int, int, int],
) -> np.ndarray | None:
    out = np.array(grid, copy=True)
    for spec in specs:
        out[spec["coords"][:, 0], spec["coords"][:, 1]] = BACKGROUND
    occupied = (out != BACKGROUND)
    for spec in specs:
        moved = move_component_cells(spec["coords"], spec["bbox"], spec["side"], spec["overlap_global"], wall_box)
        for rr, cc in moved:
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                return None
            if occupied[rr, cc]:
                return None
            out[rr, cc] = int(spec["color"])
            occupied[rr, cc] = True
    return out


def validate_specs(grid: np.ndarray, specs: list[dict], wall_box: tuple[int, int, int, int]) -> bool:
    wall_area = (wall_box[2] - wall_box[0] - 1) * (wall_box[3] - wall_box[1] - 1)
    interior_occupied = 0
    overlap_classes = {spec["interval_class"] for spec in specs}
    covers_gap_modes = {spec["covers_gap"] for spec in specs}
    for spec in specs:
        patch = crop_bbox(grid, spec["bbox"])
        if not np.all(patch == spec["color"]):
            return False
        if spec["overlap_global"][0] > spec["overlap_global"][1]:
            return False
        if spec["gap_global"][0] > spec["gap_global"][1]:
            return False
        moved = move_component_cells(spec["coords"], spec["bbox"], spec["side"], spec["overlap_global"], wall_box)
        wr0, wc0, wr1, wc1 = wall_box
        interior_mask = (wr0 < moved[:, 0]) & (moved[:, 0] < wr1) & (wc0 < moved[:, 1]) & (moved[:, 1] < wc1)
        interior_occupied += int(interior_mask.sum())
    if interior_occupied == 0 or interior_occupied >= wall_area * 0.55:
        return False
    if len(overlap_classes) < 2:
        return False
    if len(covers_gap_modes) < 2:
        return False

    expected = expected_output_from_specs(grid, specs, wall_box)
    if expected is None:
        return False
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    found = components_by_color(grid, {BACKGROUND, WALL_COLOR})
    if len(found) != len(specs):
        return False
    return True


def generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    wall_h = rng.randint(8, min(14, h - 8))
    wall_w = rng.randint(8, min(14, w - 8))
    wr0 = rng.randint(4, h - wall_h - 4)
    wc0 = rng.randint(4, w - wall_w - 4)
    wr1 = wr0 + wall_h - 1
    wc1 = wc0 + wall_w - 1
    wall_box = (wr0, wc0, wr1, wc1)

    grid[wr0, wc0 : wc1 + 1] = WALL_COLOR
    grid[wr1, wc0 : wc1 + 1] = WALL_COLOR
    grid[wr0 : wr1 + 1, wc0] = WALL_COLOR
    grid[wr0 : wr1 + 1, wc1] = WALL_COLOR

    all_sides = ["top", "bottom", "left", "right"]
    chosen_sides = choose_side_set(rng, all_sides)
    colors = rng.sample(PALETTE, len(chosen_sides))

    desired_classes = ["full_center", "partial_left", "partial_right", "full_edge"]
    rng.shuffle(desired_classes)
    long_flags = [True, False, True, False]
    rng.shuffle(long_flags)

    specs: list[dict] = []
    gaps: dict[str, list[tuple[int, int]]] = {side: [] for side in all_sides}

    for idx, side in enumerate(chosen_sides):
        side_total, max_motion_len = side_capacity(side, wall_box, grid.shape)
        if max_motion_len < 2:
            return None

        wanted_classes = desired_classes[:]
        wanted_classes = sorted(wanted_classes, key=lambda cls: cls != desired_classes[idx % len(desired_classes)])
        chosen_gap = None
        interval_local = None
        interval_class = None
        for want_class in wanted_classes:
            gap_candidates = side_gap_candidates(side, wall_box, max_gap_len=min(4, side_total - 2))
            rng.shuffle(gap_candidates)
            for gap_local in gap_candidates:
                candidate_interval = choose_interval_for_gap(rng, side_total, gap_local, want_class)
                if candidate_interval is not None:
                    chosen_gap = gap_local
                    interval_local = candidate_interval
                    interval_class = want_class
                    break
            if chosen_gap is not None:
                break
        if chosen_gap is None or interval_local is None or interval_class is None:
            return None

        gap_global, overlap_global, span_global = motion_overlap_global(wall_box, side, chosen_gap, interval_local)
        gap_len = gap_global[1] - gap_global[0] + 1
        span_len = span_global[1] - span_global[0] + 1
        if span_len <= gap_len:
            return None

        wall_depth, max_len = side_capacity(side, wall_box, grid.shape)
        prefer_long = long_flags[idx % len(long_flags)]
        long_threshold = wall_depth - 1
        if prefer_long and max_len >= long_threshold:
            motion_len = rng.randint(long_threshold, max_len)
        else:
            short_high = min(max_len, max(2, long_threshold - 1))
            if short_high < 2:
                motion_len = max_len
            else:
                motion_len = rng.randint(2, short_high)

        coords, block_box = build_component_from_spec(side, span_global, motion_len, wall_box)
        r0, c0, r1, c1 = block_box
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return None

        spec = {
            "side": side,
            "color": colors[idx],
            "coords": coords,
            "bbox": block_box,
            "gap_global": gap_global,
            "overlap_global": overlap_global,
            "interval_class": interval_class,
            "covers_gap": motion_len >= long_threshold,
        }
        specs.append(spec)
        gaps[side].append(gap_global)

    expected = np.full_like(grid, False, dtype=bool)
    for spec in specs:
        rr = spec["coords"][:, 0]
        cc = spec["coords"][:, 1]
        if expected[rr, cc].any():
            return None
        expected[rr, cc] = True
        grid[rr, cc] = int(spec["color"])

    for side, spans in gaps.items():
        if not spans:
            continue
        if len(spans) != 1:
            return None
        start, end = spans[0]
        if side == "top":
            grid[wr0, start : end + 1] = BACKGROUND
        elif side == "bottom":
            grid[wr1, start : end + 1] = BACKGROUND
        elif side == "left":
            grid[start : end + 1, wc0] = BACKGROUND
        else:
            grid[start : end + 1, wc1] = BACKGROUND

    out = expected_output_from_specs(grid, specs, wall_box)
    if out is None:
        return None
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[grid == WALL_COLOR] = True
    for spec in specs:
        moved = move_component_cells(spec["coords"], spec["bbox"], spec["side"], spec["overlap_global"], wall_box)
        for rr, cc in moved:
            if occupied[rr, cc]:
                return None
            occupied[rr, cc] = True

    if np.array_equal(grid, out):
        return None
    if not validate_specs(grid, specs, wall_box):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid screening-wall diversion sample")
