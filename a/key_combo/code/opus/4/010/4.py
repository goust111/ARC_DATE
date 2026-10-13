from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ROD_COLORS = (1, 2, 3, 4)
GREY = 5
ENDPOINT = 6

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
OPEN_SIDES = ("top", "bottom", "left", "right")
VERTICAL_OPEN_SIDES = ("top", "bottom")
HORIZONTAL_OPEN_SIDES = ("left", "right")
VARIANTS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("left", "top"),
    ("left", "bottom"),
    ("right", "top"),
    ("right", "bottom"),
)
MAX_GENERATE_ATTEMPTS = 400
GROUP_PLACEMENT_ATTEMPTS = 200
GRID_SIZE_RANGE = (18, 26)
GAP_LEN_RANGE = (3, 6)
RECT_SPAN_RANGE = (3, 6)
FIXED_LEN_RANGE = (2, 4)
GROUP_COUNTS = (2, 3)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _line_cells(start: tuple[int, int], step: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = step
    return [(int(r + dr * k), int(c + dc * k)) for k in range(length)]


def _ray_from(origin: tuple[int, int], step: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = origin
    dr, dc = step
    return [(int(r + dr * k), int(c + dc * k)) for k in range(1, length + 1)]


def _choose_fold_step(shape: tuple[int, int], hinge: tuple[int, int], rod_step: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if rod_step[0] == 0:
        up = int(hinge[0])
        down = int(h - 1 - hinge[0])
        assert up != down, "hinge cannot sit at the horizontal centerline"
        return (-1, 0) if up > down else (1, 0)
    left = int(hinge[1])
    right = int(w - 1 - hinge[1])
    assert left != right, "hinge cannot sit at the vertical centerline"
    return (0, -1) if left > right else (0, 1)


def _missing_side_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["missing_side"]
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    return [(r, c1) for r in range(r0, r1 + 1)]


def _frame_expected_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["missing_side"]
    cells: set[tuple[int, int]] = set()
    if side != "top":
        cells.update((r0, c) for c in range(c0, c1 + 1))
    if side != "bottom":
        cells.update((r1, c) for c in range(c0, c1 + 1))
    if side != "left":
        cells.update((r, c0) for r in range(r0, r1 + 1))
    if side != "right":
        cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _parse_frame(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    coord_set = {tuple(map(int, rc)) for rc in coords}
    top = all((r0, c) in coord_set for c in range(c0, c1 + 1))
    bottom = all((r1, c) in coord_set for c in range(c0, c1 + 1))
    left = all((r, c0) in coord_set for r in range(r0, r1 + 1))
    right = all((r, c1) in coord_set for r in range(r0, r1 + 1))
    sides = {"top": top, "bottom": bottom, "left": left, "right": right}
    missing = [name for name, present in sides.items() if not present]
    assert len(missing) == 1, "each grey component must be a three-sided frame"
    frame = {"bbox": (r0, c0, r1, c1), "missing_side": missing[0]}
    assert coord_set == _frame_expected_cells(frame), "grey component must be a width-1 rectangular U"
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"
    return frame


def _parse_rod(grid: np.ndarray, endpoint: tuple[int, int]) -> dict:
    h, w = grid.shape
    r, c = endpoint
    rod_neighbors = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in ROD_COLORS:
            rod_neighbors.append((dr, dc))
    assert len(rod_neighbors) == 1, "endpoint marker must touch exactly one rod cell"
    rod_step = rod_neighbors[0]
    color = int(grid[r + rod_step[0], c + rod_step[1]])

    fixed_cells: list[tuple[int, int]] = []
    k = 1
    while True:
        rr = r + rod_step[0] * k
        cc = c + rod_step[1] * k
        assert in_bounds(grid.shape, rr, cc), "rod must contain an interior hinge"
        value = int(grid[rr, cc])
        if value == color:
            fixed_cells.append((rr, cc))
            k += 1
            continue
        assert value == BG, "rod hinge must be a black cell"
        hinge = (rr, cc)
        break
    assert fixed_cells, "fixed segment must contain at least one rod-colored pixel"

    tail_cells: list[tuple[int, int]] = []
    k = 1
    while True:
        rr = hinge[0] + rod_step[0] * k
        cc = hinge[1] + rod_step[1] * k
        if not in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != color:
            break
        tail_cells.append((rr, cc))
        k += 1
    assert tail_cells, "rod must have a movable tail"

    fold_step = _choose_fold_step((h, w), hinge, rod_step)
    folded = _ray_from(hinge, fold_step, len(tail_cells))
    assert all(in_bounds(grid.shape, rr, cc) for rr, cc in folded), "folded tail must stay in bounds"

    return {
        "endpoint": endpoint,
        "color": color,
        "rod_step": rod_step,
        "hinge": hinge,
        "fixed_cells": fixed_cells,
        "tail_cells": tail_cells,
        "fold_step": fold_step,
        "folded_cells": folded,
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    endpoints = [tuple(map(int, rc)) for rc in np.argwhere(g == ENDPOINT)]
    assert 2 <= len(endpoints) <= 3, "expected 2-3 endpoint markers"
    rods = [_parse_rod(g, endpoint) for endpoint in endpoints]
    rod_colors = [rod["color"] for rod in rods]
    assert len(set(rod_colors)) == len(rod_colors), "rod colors must be distinct"

    frame_comps = components_mask(g == GREY, connectivity=4)
    assert len(frame_comps) == len(rods), "expected one frame per rod"
    frames = [_parse_frame(comp) for comp in frame_comps]
    return rods, frames


def main(grid: np.ndarray) -> np.ndarray:
    """Fold each marked rod so its tail closes a grey U-frame, then fill the newly enclosed rectangle.

    Input structure:
    - The grid has black background, 2-3 disjoint grey width-1 U-shaped rectangular frames, and one colored straight rod near each frame.
    - Every rod uses one color from {1,2,3,4}, contains a magenta endpoint marker at one end, and contains a black hinge cell in its interior.
    - The rod is perpendicular to the frame's missing side, and the hinge-side tail length matches that side exactly.

    Rule:
    - For each rod, keep the segment from the magenta endpoint up to the black hinge fixed, and treat the opposite same-color segment as the movable tail.
    - Rotate the movable tail 90 degrees around the hinge toward the side with more space to the grid edge; clear its old cells and redraw the tail at the rotated positions, while keeping the hinge black and the endpoint magenta.
    - Match each folded tail to the unique grey U-frame whose missing side it exactly covers, then fill the rectangle's black interior cells with that rod color.

    Key insight:
    - The endpoint tells which side of the hinge stays fixed, the hinge plus grid-edge clearance determines the fold direction, and only after that fold does a complete rectangular boundary appear that can be filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    rods, frames = _parse_scene(g)

    for rod in rods:
        for rr, cc in rod["tail_cells"]:
            out[rr, cc] = BG
        for rr, cc in rod["folded_cells"]:
            assert int(out[rr, cc]) in (BG, GREY), "folded tail may only overlap the grey frame corners"
            if int(out[rr, cc]) == BG:
                out[rr, cc] = rod["color"]
        out[rod["hinge"]] = BG
        out[rod["endpoint"]] = ENDPOINT

    used_rods: set[int] = set()
    for frame in frames:
        missing = _missing_side_cells(frame)
        matches = [idx for idx, rod in enumerate(rods) if rod["folded_cells"] == missing]
        assert len(matches) == 1, "each frame must match exactly one folded rod"
        idx = matches[0]
        assert idx not in used_rods, "each rod must close exactly one frame"
        used_rods.add(idx)
        color = int(rods[idx]["color"])
        r0, c0, r1, c1 = frame["bbox"]
        border = _frame_expected_cells(frame) | set(missing)
        assert all(int(out[r, c]) != BG for r, c in border), "folded scene must form a closed rectangle"
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if int(out[rr, cc]) == BG:
                    out[rr, cc] = color
    assert len(used_rods) == len(rods)
    return out


def _variant_fold_step(open_side: str, anchor: str) -> tuple[int, int]:
    if open_side in VERTICAL_OPEN_SIDES:
        return (0, 1) if anchor == "left" else (0, -1)
    return (1, 0) if anchor == "top" else (-1, 0)


def _candidate_from_variant(
    shape: tuple[int, int],
    color: int,
    open_side: str,
    anchor: str,
    gap_len: int,
    rect_span: int,
    fixed_len: int,
    fixed_sign: int,
    hinge: tuple[int, int],
) -> dict | None:
    h, w = map(int, shape)
    hinge_r, hinge_c = hinge

    if open_side == "top":
        r0 = hinge_r
        r1 = r0 + rect_span - 1
        if anchor == "left":
            c0 = hinge_c + 1
            c1 = c0 + gap_len - 1
        else:
            c1 = hinge_c - 1
            c0 = c1 - gap_len + 1
    elif open_side == "bottom":
        r1 = hinge_r
        r0 = r1 - rect_span + 1
        if anchor == "left":
            c0 = hinge_c + 1
            c1 = c0 + gap_len - 1
        else:
            c1 = hinge_c - 1
            c0 = c1 - gap_len + 1
    elif open_side == "left":
        c0 = hinge_c
        c1 = c0 + rect_span - 1
        if anchor == "top":
            r0 = hinge_r + 1
            r1 = r0 + gap_len - 1
        else:
            r1 = hinge_r - 1
            r0 = r1 - gap_len + 1
    else:
        c1 = hinge_c
        c0 = c1 - rect_span + 1
        if anchor == "top":
            r0 = hinge_r + 1
            r1 = r0 + gap_len - 1
        else:
            r1 = hinge_r - 1
            r0 = r1 - gap_len + 1

    if min(r0, c0) < 0 or r1 >= h or c1 >= w:
        return None

    rod_step = (fixed_sign, 0) if open_side in VERTICAL_OPEN_SIDES else (0, fixed_sign)
    fold_step = _variant_fold_step(open_side, anchor)
    if _choose_fold_step(shape, hinge, rod_step) != fold_step:
        return None

    endpoint = (hinge_r + rod_step[0] * fixed_len, hinge_c + rod_step[1] * fixed_len)
    fixed_cells = _ray_from(hinge, rod_step, fixed_len - 1)
    tail_cells = _ray_from(hinge, (-rod_step[0], -rod_step[1]), gap_len)
    folded_cells = _ray_from(hinge, fold_step, gap_len)
    all_cells = [endpoint, hinge, *fixed_cells, *tail_cells, *folded_cells]
    if not all(in_bounds(shape, rr, cc) for rr, cc in all_cells):
        return None

    frame = {"bbox": (r0, c0, r1, c1), "missing_side": open_side}
    missing = _missing_side_cells(frame)
    if folded_cells != missing:
        return None

    grey_cells = sorted(_frame_expected_cells(frame))
    input_nonblack = set(grey_cells) | set(fixed_cells) | set(tail_cells) | {endpoint}
    if len(input_nonblack) != len(grey_cells) + len(fixed_cells) + len(tail_cells) + 1:
        return None
    if hinge in input_nonblack:
        return None

    border = set(grey_cells) | set(folded_cells)
    interior = {(rr, cc) for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)}
    envelope = set(input_nonblack) | set(folded_cells) | interior | {hinge}

    return {
        "color": color,
        "frame": frame,
        "hinge": hinge,
        "endpoint": endpoint,
        "rod_step": rod_step,
        "fixed_cells": fixed_cells,
        "tail_cells": tail_cells,
        "folded_cells": folded_cells,
        "grey_cells": grey_cells,
        "border_cells": sorted(border),
        "interior_cells": sorted(interior),
        "envelope_bbox": bbox_coords(np.asarray(sorted(envelope), dtype=int)),
        "input_nonblack": input_nonblack,
    }


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _pick_variants(group_count: int) -> list[tuple[str, str]]:
    vertical = [v for v in VARIANTS if v[0] in VERTICAL_OPEN_SIDES]
    horizontal = [v for v in VARIANTS if v[0] in HORIZONTAL_OPEN_SIDES]
    chosen = [random.choice(vertical), random.choice(horizontal)]
    remaining = [v for v in VARIANTS if v not in chosen]
    random.shuffle(remaining)
    while len(chosen) < group_count:
        chosen.append(remaining.pop())
    random.shuffle(chosen)
    return chosen


def _sample_group(shape: tuple[int, int], color: int, variant: tuple[str, str], placed: list[dict]) -> dict | None:
    h, w = shape
    open_side, anchor = variant
    for _ in range(GROUP_PLACEMENT_ATTEMPTS):
        gap_len = random.randint(*GAP_LEN_RANGE)
        rect_span = random.randint(*RECT_SPAN_RANGE)
        fixed_len = random.randint(*FIXED_LEN_RANGE)
        fixed_sign = random.choice((-1, 1))

        if open_side in VERTICAL_OPEN_SIDES:
            hinge_r_min = max(rect_span - 1 if open_side == "bottom" else 0, gap_len if fixed_sign == 1 else fixed_len)
            hinge_r_max = min(
                h - rect_span if open_side == "top" else h - 1,
                h - 1 - fixed_len if fixed_sign == 1 else h - 1 - gap_len,
            )
            if anchor == "left":
                hinge_c_min = 0
                hinge_c_max = min(w - gap_len - 1, (w - 2) // 2)
            else:
                hinge_c_min = max(gap_len, (w + 1) // 2)
                hinge_c_max = w - 1
        else:
            hinge_c_min = max(rect_span - 1 if open_side == "right" else 0, gap_len if fixed_sign == 1 else fixed_len)
            hinge_c_max = min(
                w - rect_span if open_side == "left" else w - 1,
                w - 1 - fixed_len if fixed_sign == 1 else w - 1 - gap_len,
            )
            if anchor == "top":
                hinge_r_min = 0
                hinge_r_max = min(h - gap_len - 1, (h - 2) // 2)
            else:
                hinge_r_min = max(gap_len, (h + 1) // 2)
                hinge_r_max = h - 1

        if hinge_r_min > hinge_r_max or hinge_c_min > hinge_c_max:
            continue

        hinge = (
            random.randint(int(hinge_r_min), int(hinge_r_max)),
            random.randint(int(hinge_c_min), int(hinge_c_max)),
        )
        candidate = _candidate_from_variant(
            shape, color, open_side, anchor, gap_len, rect_span, fixed_len, fixed_sign, hinge
        )
        if candidate is None:
            continue
        if any(not _boxes_separated(candidate["envelope_bbox"], other["envelope_bbox"]) for other in placed):
            continue
        return candidate
    return None


def _render_input(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for group in groups:
        for rr, cc in group["grey_cells"]:
            grid[rr, cc] = GREY
        for rr, cc in group["fixed_cells"]:
            grid[rr, cc] = group["color"]
        for rr, cc in group["tail_cells"]:
            grid[rr, cc] = group["color"]
        grid[group["endpoint"]] = ENDPOINT
        grid[group["hinge"]] = BG
    return grid


def _render_expected(inp: np.ndarray, groups: list[dict]) -> np.ndarray:
    out = inp.copy()
    for group in groups:
        for rr, cc in group["tail_cells"]:
            out[rr, cc] = BG
        for rr, cc in group["folded_cells"]:
            if out[rr, cc] == BG:
                out[rr, cc] = group["color"]
        for rr, cc in group["interior_cells"]:
            if out[rr, cc] == BG:
                out[rr, cc] = group["color"]
        out[group["hinge"]] = BG
        out[group["endpoint"]] = ENDPOINT
    return out


def _validate_generated_scene(inp: np.ndarray, expected: np.ndarray, groups: list[dict]) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.count_nonzero(inp == ENDPOINT) != len(groups):
        return False
    rods, frames = _parse_scene(inp)
    if len(rods) != len(groups) or len(frames) != len(groups):
        return False
    if {tuple(rod["folded_cells"]) for rod in rods} != {tuple(group["folded_cells"]) for group in groups}:
        return False
    return bool(np.array_equal(main(inp), expected))


def _attempt_generate() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIZE_RANGE),
        random.randint(*GRID_SIZE_RANGE),
    )
    group_count = random.choice(GROUP_COUNTS)
    colors = random.sample(list(ROD_COLORS), group_count)
    variants = _pick_variants(group_count)

    groups: list[dict] = []
    for color, variant in zip(colors, variants):
        group = _sample_group(shape, color, variant, groups)
        if group is None:
            return None
        groups.append(group)

    inp = _render_input(shape, groups)
    expected = _render_expected(inp, groups)
    if not _validate_generated_scene(inp, expected, groups):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-and-fill scene")
