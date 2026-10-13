from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_REGION_SIDE = 2
MIN_INTERIOR = 10
MAX_INTERIOR = 16
MIN_REGIONS = 4
MAX_REGIONS = 6
MAX_LAYOUT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _black_regions(
    g: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> list[dict[str, object]]:
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    h, w = inner.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict[str, object]] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(inner[sr, sc]) != BG:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            min_r = r0 + 1 + sr
            min_c = c0 + 1 + sc
            while stack:
                r, c = stack.pop()
                ar = r0 + 1 + r
                ac = c0 + 1 + c
                coords.append((ar, ac))
                if ar < min_r:
                    min_r = ar
                if ac < min_c:
                    min_c = ac
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(inner[nr, nc]) == BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            regions.append(
                {
                    "coords": coords,
                    "area": len(coords),
                    "top": min_r,
                    "left": min_c,
                }
            )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect border seed colors clockwise, sort enclosed black regions by area,
    and fill them cyclically with the seed sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mask = g != BG
    assert bool(mask.any())
    r0, c0, r1, c1 = _bbox(mask)
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    assert np.all(g[r0, c0 : c1 + 1] != BG)
    assert np.all(g[r1, c0 : c1 + 1] != BG)
    assert np.all(g[r0 : r1 + 1, c0] != BG)
    assert np.all(g[r0 : r1 + 1, c1] != BG)

    seed_sequence = [int(g[r, c]) for r, c in _perimeter_coords(r0, c0, r1, c1) if int(g[r, c]) != WALL]
    assert seed_sequence

    regions = _black_regions(g, r0, c0, r1, c1)
    assert regions
    regions.sort(key=lambda region: (int(region["area"]), int(region["top"]), int(region["left"])))

    out = g.copy()
    for idx, region in enumerate(regions):
        color = seed_sequence[idx % len(seed_sequence)]
        for r, c in region["coords"]:
            out[r, c] = color
    return out


def _rect_h(rect: tuple[int, int, int, int]) -> int:
    return rect[2] - rect[0] + 1


def _rect_w(rect: tuple[int, int, int, int]) -> int:
    return rect[3] - rect[1] + 1


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    return _rect_h(rect) * _rect_w(rect)


def _split_options(rect: tuple[int, int, int, int]) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = rect
    opts: list[tuple[str, int]] = []
    for wall_c in range(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE + 1):
        opts.append(("v", wall_c))
    for wall_r in range(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE + 1):
        opts.append(("h", wall_r))
    return opts


def _apply_split(
    rect: tuple[int, int, int, int], axis: str, pos: int
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    if axis == "v":
        left = (r0, c0, r1, pos - 1)
        right = (r0, pos + 1, r1, c1)
        assert _rect_w(left) >= MIN_REGION_SIDE and _rect_w(right) >= MIN_REGION_SIDE
        wall = [(r, pos) for r in range(r0, r1 + 1)]
        return [left, right], wall

    top = (r0, c0, pos - 1, c1)
    bottom = (pos + 1, c0, r1, c1)
    assert _rect_h(top) >= MIN_REGION_SIDE and _rect_h(bottom) >= MIN_REGION_SIDE
    wall = [(pos, c) for c in range(c0, c1 + 1)]
    return [top, bottom], wall


def _is_balanced_split(rect: tuple[int, int, int, int], axis: str, pos: int) -> bool:
    children, _wall = _apply_split(rect, axis, pos)
    return _rect_area(children[0]) == _rect_area(children[1])


def _balanced_split_options(rect: tuple[int, int, int, int]) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = rect
    opts: list[tuple[str, int]] = []
    width = _rect_w(rect)
    height = _rect_h(rect)
    if width % 2 == 1 and width >= 2 * MIN_REGION_SIDE + 1:
        opts.append(("v", c0 + width // 2))
    if height % 2 == 1 and height >= 2 * MIN_REGION_SIDE + 1:
        opts.append(("h", r0 + height // 2))
    return opts


def _grow_partition(
    leaves: list[dict[str, object]], walls: list[list[tuple[int, int]]], target_regions: int
) -> tuple[list[dict[str, object]], list[list[tuple[int, int]]]] | None:
    leaves = [{"rect": tuple(leaf["rect"]), "locked": bool(leaf["locked"])} for leaf in leaves]
    walls = [list(segment) for segment in walls]

    while len(leaves) < target_regions:
        candidates = [i for i, leaf in enumerate(leaves) if not leaf["locked"] and _split_options(leaf["rect"])]
        if not candidates:
            return None
        weights = [_rect_area(leaves[i]["rect"]) for i in candidates]
        idx = random.choices(candidates, weights=weights, k=1)[0]
        rect = leaves[idx]["rect"]
        opts = _split_options(rect)
        non_balanced = [opt for opt in opts if not _is_balanced_split(rect, *opt)]
        if non_balanced:
            opts = non_balanced
        axis, pos = random.choice(opts)
        children, wall = _apply_split(rect, axis, pos)
        child_leaves = [{"rect": child, "locked": False} for child in sorted(children)]
        leaves[idx : idx + 1] = child_leaves
        walls.append(wall)

    return leaves, walls


def _has_area_tie(rects: list[tuple[int, int, int, int]]) -> bool:
    areas = [_rect_area(rect) for rect in rects]
    return len(set(areas)) != len(areas)


def _generate_unique_partition(target_regions: int) -> tuple[int, int, list[tuple[int, int, int, int]], list[list[tuple[int, int]]]] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        h = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        w = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        root = (0, 0, h - 1, w - 1)
        grown = _grow_partition([{"rect": root, "locked": False}], [], target_regions)
        if grown is None:
            continue
        leaves, walls = grown
        rects = [leaf["rect"] for leaf in leaves]
        if _has_area_tie(rects):
            continue
        return h, w, rects, walls
    return None


def _generate_tied_partition(target_regions: int) -> tuple[int, int, list[tuple[int, int, int, int]], list[list[tuple[int, int]]]] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        h = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        w = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        root = (0, 0, h - 1, w - 1)
        root_opts = _split_options(root)
        random.shuffle(root_opts)

        for axis1, pos1 in root_opts:
            first_children, wall1 = _apply_split(root, axis1, pos1)
            child_indices = [0, 1]
            random.shuffle(child_indices)
            for tie_idx in child_indices:
                tie_parent = first_children[tie_idx]
                other = first_children[1 - tie_idx]
                tie_opts = _balanced_split_options(tie_parent)
                if not tie_opts:
                    continue
                random.shuffle(tie_opts)
                for axis2, pos2 in tie_opts:
                    tie_children, wall2 = _apply_split(tie_parent, axis2, pos2)
                    initial_leaves = [
                        {"rect": other, "locked": False},
                        {"rect": tie_children[0], "locked": True},
                        {"rect": tie_children[1], "locked": True},
                    ]
                    grown = _grow_partition(initial_leaves, [wall1, wall2], target_regions)
                    if grown is None:
                        continue
                    leaves, walls = grown
                    rects = [leaf["rect"] for leaf in leaves]
                    if not _has_area_tie(rects):
                        continue
                    return h, w, rects, walls
    return None


def _choose_seed_count(region_count: int, wrap_mode: bool) -> int:
    if wrap_mode:
        return random.randint(4, region_count - 1)
    return random.randint(region_count, min(6, len(SEED_COLORS)))


def _cyclic_distance(a: int, b: int, n: int) -> int:
    diff = abs(a - b)
    return min(diff, n - diff)


def _sample_seed_positions(frame_h: int, frame_w: int, seed_count: int) -> list[tuple[int, int]] | None:
    perimeter = _perimeter_coords(0, 0, frame_h - 1, frame_w - 1)
    index_of = {coord: idx for idx, coord in enumerate(perimeter)}
    by_side = {
        "top": [(0, c) for c in range(1, frame_w - 1)],
        "right": [(r, frame_w - 1) for r in range(1, frame_h - 1)],
        "bottom": [(frame_h - 1, c) for c in range(1, frame_w - 1)],
        "left": [(r, 0) for r in range(1, frame_h - 1)],
    }
    if any(not cells for cells in by_side.values()):
        return None

    chosen = [random.choice(by_side[side]) for side in ("top", "right", "bottom", "left")]
    chosen_set = set(chosen)

    extras = [coord for coord in perimeter if coord not in chosen_set and coord not in {(0, 0), (0, frame_w - 1), (frame_h - 1, 0), (frame_h - 1, frame_w - 1)}]
    random.shuffle(extras)
    for coord in extras:
        if len(chosen) == seed_count:
            break
        idx = index_of[coord]
        if all(_cyclic_distance(idx, index_of[other], len(perimeter)) > 1 for other in chosen):
            chosen.append(coord)

    if len(chosen) != seed_count:
        return None
    chosen.sort(key=lambda coord: index_of[coord])
    return chosen


def _expected_output(
    inp: np.ndarray,
    frame_top: int,
    frame_left: int,
    rects: list[tuple[int, int, int, int]],
    seed_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    ordered = sorted(
        rects,
        key=lambda rect: (
            _rect_area(rect),
            frame_top + 1 + rect[0],
            frame_left + 1 + rect[1],
        ),
    )
    for idx, (r0, c0, r1, c1) in enumerate(ordered):
        color = seed_colors[idx % len(seed_colors)]
        out[frame_top + 1 + r0 : frame_top + 2 + r1, frame_left + 1 + c0 : frame_left + 2 + c1] = color
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    tie_mode = random.random() < 0.5
    wrap_mode = random.random() < 0.5
    min_regions = 5 if wrap_mode else MIN_REGIONS
    target_regions = random.randint(min_regions, MAX_REGIONS)

    if tie_mode:
        layout = _generate_tied_partition(target_regions)
    else:
        layout = _generate_unique_partition(target_regions)
    if layout is None:
        return None

    interior_h, interior_w, rects, walls = layout
    seed_count = _choose_seed_count(len(rects), wrap_mode)
    frame_h = interior_h + 2
    frame_w = interior_w + 2
    seed_positions = _sample_seed_positions(frame_h, frame_w, seed_count)
    if seed_positions is None:
        return None

    top_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    grid_h = frame_h + top_margin + bottom_margin
    grid_w = frame_w + left_margin + right_margin

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    frame_top = top_margin
    frame_left = left_margin
    frame_bottom = frame_top + frame_h - 1
    frame_right = frame_left + frame_w - 1

    inp[frame_top, frame_left : frame_right + 1] = WALL
    inp[frame_bottom, frame_left : frame_right + 1] = WALL
    inp[frame_top : frame_bottom + 1, frame_left] = WALL
    inp[frame_top : frame_bottom + 1, frame_right] = WALL

    for segment in walls:
        for r, c in segment:
            inp[frame_top + 1 + r, frame_left + 1 + c] = WALL

    seed_colors = random.sample(SEED_COLORS, seed_count)
    for color, (r, c) in zip(seed_colors, seed_positions):
        inp[frame_top + r, frame_left + c] = color

    expected = _expected_output(inp, frame_top, frame_left, rects, seed_colors)

    areas = [_rect_area(rect) for rect in rects]
    if tie_mode != (len(set(areas)) != len(areas)):
        return None
    if wrap_mode != (len(rects) > seed_count):
        return None
    if len(set(seed_colors)) != len(seed_colors):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
