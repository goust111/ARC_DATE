import random

import numpy as np


BG = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _inside_interior(pos: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    r, c = pos
    r0, c0, r1, c1 = frame_bbox
    return r0 + 1 <= r <= r1 - 1 and c0 + 1 <= c <= c1 - 1


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    candidates: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(patch[border] == color):
            continue
        if np.any(patch[~border] == color):
            continue
        if int(np.count_nonzero(grid == color)) != int(border.sum()):
            continue
        candidates.append((h * w, int(border.sum()), color, (r0, c0, r1, c1)))

    assert candidates, "expected one hollow rectangular frame"
    candidates.sort(reverse=True)
    _area, _perim, frame_color, frame_bbox = candidates[0]
    return frame_color, *frame_bbox


def _seed_side(
    bbox: tuple[int, int, int, int],
    frame_bbox: tuple[int, int, int, int],
) -> str:
    r0, c0, r1, c1 = bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    matches: list[str] = []

    if r0 == r1 and r0 == fr0 - 1 and fc0 + 1 <= c0 and c1 <= fc1 - 1:
        matches.append("top")
    if r0 == r1 and r0 == fr1 + 1 and fc0 + 1 <= c0 and c1 <= fc1 - 1:
        matches.append("bottom")
    if c0 == c1 and c0 == fc0 - 1 and fr0 + 1 <= r0 and r1 <= fr1 - 1:
        matches.append("left")
    if c0 == c1 and c0 == fc1 + 1 and fr0 + 1 <= r0 and r1 <= fr1 - 1:
        matches.append("right")

    assert len(matches) == 1, "seed must attach to exactly one frame side"
    return matches[0]


def _find_seeds(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> list[dict]:
    seeds: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, frame_color}:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "seed colors must be present"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h, w = r1 - r0 + 1, c1 - c0 + 1
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "seed must be a solid rectangle"
        assert h == 1 or w == 1, "seed must be a 1xK or Kx1 bar"
        length = max(h, w)
        assert 1 <= length <= 4, "seed length must be between 1 and 4"
        assert int(np.count_nonzero(grid == color)) == h * w, "each seed color must be unique"

        bbox = (r0, c0, r1, c1)
        seeds.append(
            {
                "color": color,
                "bbox": bbox,
                "side": _seed_side(bbox, frame_bbox),
                "length": length,
            }
        )

    assert seeds, "expected at least one seed"
    seeds.sort(key=lambda seed: (seed["bbox"][1], seed["bbox"][0]))
    return seeds


def _branch_geometry(
    seed: dict,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = seed["bbox"]
    fr0, fc0, fr1, fc1 = frame_bbox
    side = seed["side"]

    if side == "top":
        initial = [(fr0 + 1, c) for c in range(c0, c1 + 1)]
        grow = (1, 0)
        perp = (0, 1)
    elif side == "bottom":
        initial = [(fr1 - 1, c) for c in range(c0, c1 + 1)]
        grow = (-1, 0)
        perp = (0, 1)
    elif side == "left":
        initial = [(r, fc0 + 1) for r in range(r0, r1 + 1)]
        grow = (0, 1)
        perp = (1, 0)
    else:
        initial = [(r, fc1 - 1) for r in range(r0, r1 + 1)]
        grow = (0, -1)
        perp = (1, 0)
    return initial, grow, perp


def _branch_pixels(seed: dict, frame_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    initial, grow, perp = _branch_geometry(seed, frame_bbox)
    pixels: set[tuple[int, int]] = set()
    for pos in initial:
        if _inside_interior(pos, frame_bbox):
            pixels.add(pos)

    endpoints = {initial[0], initial[-1]}
    for _ in range(2, seed["length"] + 1):
        next_endpoints: set[tuple[int, int]] = set()
        for r, c in endpoints:
            center = (r + grow[0], c + grow[1])
            tip_a = (center[0] + perp[0], center[1] + perp[1])
            tip_b = (center[0] - perp[0], center[1] - perp[1])
            for pos in (center, tip_a, tip_b):
                if _inside_interior(pos, frame_bbox):
                    pixels.add(pos)
            next_endpoints.add(tip_a)
            next_endpoints.add(tip_b)
        endpoints = next_endpoints
    return pixels


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frame_color, fr0, fc0, fr1, fc1 = _find_frame(g, bg)
    frame_bbox = (fr0, fc0, fr1, fc1)
    assert frame_color != bg
    assert np.all(g[fr0 + 1 : fr1, fc0 + 1 : fc1] == bg), "frame interior must start empty"

    out = g.copy()
    for seed in _find_seeds(g, bg, frame_color, frame_bbox):
        for r, c in _branch_pixels(seed, frame_bbox):
            out[r, c] = seed["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow inward binary branch trees from side-attached seed bars on a hollow frame.

    Input structure:
    - bg = most common color.
    - One large grey hollow rectangular frame sits in the grid, with a 1-cell border and an empty background interior.
    - Several small monochrome seed bars of distinct colors are attached to the outside of the frame on its top, bottom, left, or right side.
    - Each seed is a 1xK or Kx1 rectangle whose long axis is parallel to the touched frame side.

    Rule:
    - Preserve the original frame and seeds, and only draw new pixels inside the frame.
    - For each seed, infer the inward growth direction from the side it touches. Its area K is both the first-layer segment length and the total number of growth layers.
    - Draw layer 1 one cell inside the frame, parallel to the touched side. For every later layer, each current branch tip advances one cell inward and then sprouts one cell to both perpendicular sides; pixels outside the frame interior are clipped away.
    - Process seeds in left-to-right, then top-to-bottom order, so later branches overwrite earlier overlaps.

    Key insight:
    - The side of attachment selects the inward direction, the seed area selects the depth, and every deeper layer follows the same endpoint-splitting T-branch pattern.
    """
    return _transform(grid)


def _paint_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _available_starts(track_len: int, length: int, occupied: list[tuple[int, int]]) -> list[int]:
    starts: list[int] = []
    for start in range(0, track_len - length + 1):
        end = start + length - 1
        ok = True
        for occ0, occ1 in occupied:
            if not (end + 1 < occ0 or start - 1 > occ1):
                ok = False
                break
        if ok:
            starts.append(start)
    return starts


def _sample_side_targets(seed_count: int) -> list[str]:
    side_targets: list[str]
    if random.random() < 0.7:
        side_targets = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
        while len(side_targets) < seed_count:
            side_targets.append(random.choice(SIDES))
    else:
        distinct_count = random.randint(2, min(4, seed_count))
        side_targets = random.sample(SIDES, distinct_count)
        while len(side_targets) < seed_count:
            side_targets.append(random.choice(side_targets))
    random.shuffle(side_targets)
    return side_targets


def _seed_from_track(
    side: str,
    start: int,
    length: int,
    color: int,
    frame_bbox: tuple[int, int, int, int],
) -> dict:
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "top":
        bbox = (fr0 - 1, fc0 + 1 + start, fr0 - 1, fc0 + start + length)
    elif side == "bottom":
        bbox = (fr1 + 1, fc0 + 1 + start, fr1 + 1, fc0 + start + length)
    elif side == "left":
        bbox = (fr0 + 1 + start, fc0 - 1, fr0 + start + length, fc0 - 1)
    else:
        bbox = (fr0 + 1 + start, fc1 + 1, fr0 + start + length, fc1 + 1)
    return {"color": color, "side": side, "bbox": bbox, "length": length}


def _paint_seed(grid: np.ndarray, seed: dict) -> None:
    r0, c0, r1, c1 = seed["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = seed["color"]


def _build_input(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    seeds: list[dict],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _paint_frame(grid, frame_bbox)
    for seed in seeds:
        _paint_seed(grid, seed)
    return grid


def _expected_from_seeds(
    input_grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    seeds: list[dict],
) -> np.ndarray:
    out = input_grid.copy()
    ordered = sorted(seeds, key=lambda seed: (seed["bbox"][1], seed["bbox"][0]))
    for seed in ordered:
        for r, c in _branch_pixels(seed, frame_bbox):
            out[r, c] = seed["color"]
    return out


def _overlap_count(branches: list[set[tuple[int, int]]]) -> int:
    total = sum(len(branch) for branch in branches)
    union: set[tuple[int, int]] = set()
    for branch in branches:
        union |= branch
    return total - len(union)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)

    frame_h = random.randint(9, min(22, height - 2))
    frame_w = random.randint(9, min(22, width - 2))
    fr0 = random.randint(1, height - frame_h - 1)
    fc0 = random.randint(1, width - frame_w - 1)
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    seed_count = random.randint(2, 5)
    colors = random.sample(SEED_COLORS, seed_count)
    side_targets = _sample_side_targets(seed_count)
    occupied = {side: [] for side in SIDES}
    seeds: list[dict] = []

    for color, side in zip(colors, side_targets):
        track_len = inner_w if side in {"top", "bottom"} else inner_h
        placed = False
        for _ in range(20):
            length = random.randint(1, min(4, track_len))
            starts = _available_starts(track_len, length, occupied[side])
            if not starts:
                continue
            start = random.choice(starts)
            occupied[side].append((start, start + length - 1))
            seeds.append(_seed_from_track(side, start, length, color, frame_bbox))
            placed = True
            break
        if not placed:
            return None

    lengths = {seed["length"] for seed in seeds}
    if len(lengths) < 2:
        return None

    input_grid = _build_input((height, width), frame_bbox, seeds)
    branches = [_branch_pixels(seed, frame_bbox) for seed in seeds]
    interior_area = inner_h * inner_w
    branch_union: set[tuple[int, int]] = set()
    for branch in branches:
        branch_union |= branch
    overlap = _overlap_count(branches)
    allow_overlap = random.random() < 0.35

    if len(branch_union) > int(0.58 * interior_area):
        return None
    if overlap > 0 and not allow_overlap:
        return None
    if overlap > max(4, interior_area // 8):
        return None

    frame_color, fr0_, fc0_, fr1_, fc1_ = _find_frame(input_grid, BG)
    if frame_color != FRAME_COLOR or (fr0_, fc0_, fr1_, fc1_) != frame_bbox:
        return None
    expected = _expected_from_seeds(input_grid, frame_bbox, seeds)
    if not np.array_equal(main(input_grid), expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-branch ARC input")
