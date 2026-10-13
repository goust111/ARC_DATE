from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
FRAME_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_SIDES = (5, 7, 9, 11, 13)
DIRECTIONS = ("up", "down", "left", "right")
DIR_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MIN_FRAMES = 3
MAX_FRAMES = 5
MAX_GENERATION_ATTEMPTS = 300
RESERVE_PAD = 1
_RNG = np.random.default_rng()

_BASE_SHAPES = {
    1: [[(0, 0)]],
    2: [[(0, 0), (0, 1)]],
    3: [[(0, 0), (0, 1), (1, 0)]],
    4: [
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
    5: [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    ],
}


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(list(coords))
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _r0, _c0, r1, _c1 = _bbox(list(coords))
    height = r1 + 1
    rotated = [(c, height - 1 - r) for r, c in coords]
    return _normalize_shape(rotated)


def _mirror_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _r0, _c0, _r1, c1 = _bbox(list(coords))
    mirrored = [(r, c1 - c) for r, c in coords]
    return _normalize_shape(mirrored)


def _shape_variants(count: int) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for base in _BASE_SHAPES[count]:
        cur = _normalize_shape(base)
        for _ in range(4):
            variants.add(cur)
            variants.add(_mirror_shape(cur))
            cur = _rotate_shape(cur)
    return sorted(variants)


_SHAPE_VARIANTS = {count: _shape_variants(count) for count in _BASE_SHAPES}


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_border_coords(r0: int, c0: int, side: int) -> set[tuple[int, int]]:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    coords: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0 + 1, r1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _extract_frame(comp: list[tuple[int, int]]) -> dict[str, object]:
    coord_set = set(comp)
    matches: list[dict[str, object]] = []
    for bud in comp:
        body = coord_set - {bud}
        if not body:
            continue
        r0, c0, r1, c1 = _bbox(body)
        side = r1 - r0 + 1
        if side < 5 or side != c1 - c0 + 1 or side % 2 == 0:
            continue
        center_r = r0 + side // 2
        center_c = c0 + side // 2
        direction = None
        if bud == (r0 - 1, center_c):
            direction = "up"
        elif bud == (r1 + 1, center_c):
            direction = "down"
        elif bud == (center_r, c0 - 1):
            direction = "left"
        elif bud == (center_r, c1 + 1):
            direction = "right"
        if direction is None:
            continue
        if body == _frame_border_coords(r0, c0, side):
            matches.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "direction": direction,
                    "bud": bud,
                }
            )
    assert len(matches) == 1, "each colored component must be one square frame with one bud"
    return matches[0]


def _seed_offset(seed: list[tuple[int, int]], frame: dict[str, object]) -> float:
    rows = np.array([r for r, _ in seed], dtype=float)
    cols = np.array([c for _, c in seed], dtype=float)
    r0, c0, r1, c1 = frame["bbox"]
    center_r = 0.5 * (r0 + r1)
    center_c = 0.5 * (c0 + c1)
    direction = frame["direction"]
    if direction == "up":
        return float(center_r - rows.mean())
    if direction == "down":
        return float(rows.mean() - center_r)
    if direction == "left":
        return float(center_c - cols.mean())
    return float(cols.mean() - center_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect hollow square frames with one outward bud, then extend an arm from the bud.

    The bud direction sets the arm direction. The arm length is the seed centroid
    offset along that normal direction. The arm color comes from the frame-color
    table ordered by frame size from smallest to largest, indexed by the number of
    grey seed pixels (1-based).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames: list[dict[str, object]] = []
    grey_seen: set[tuple[int, int]] = set()
    for color in np.unique(g):
        color = int(color)
        if color in {BG, GREY}:
            continue
        comps = _components_of_color(g, color)
        assert len(comps) == 1, "frame colors must be unique"
        frame = _extract_frame(comps[0])
        r0, c0, r1, c1 = frame["bbox"]
        seed = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(g[r, c]) == GREY]
        assert seed, "each frame must contain grey seed pixels"
        grey_seen.update(seed)
        frame["color"] = color
        frame["seed"] = seed
        frames.append(frame)

    total_grey = int(np.count_nonzero(g == GREY))
    assert total_grey == len(grey_seen), "grey pixels must belong to exactly one frame interior"

    palette = [int(frame["color"]) for frame in sorted(frames, key=lambda item: (int(item["side"]), item["bbox"][0], item["bbox"][1]))]
    out = g.copy()

    for frame in frames:
        seed = frame["seed"]
        count = len(seed)
        assert 1 <= count <= len(palette), "seed count must index the global color table"
        offset = _seed_offset(seed, frame)
        length = _round_half_up(offset)
        assert length >= 1, "seed must be visibly biased toward the bud side"
        arm_color = palette[count - 1]
        bud_r, bud_c = frame["bud"]
        dr, dc = DIR_VEC[frame["direction"]]
        for step in range(1, length + 1):
            rr = bud_r + dr * step
            cc = bud_c + dc * step
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            assert int(out[rr, cc]) == BG
            out[rr, cc] = arm_color

    return out


def _local_offset(cells: list[tuple[int, int]], inner: int, direction: str) -> float:
    center = 0.5 * (inner - 1)
    rows = np.array([r for r, _ in cells], dtype=float)
    cols = np.array([c for _, c in cells], dtype=float)
    if direction == "up":
        return float(center - rows.mean())
    if direction == "down":
        return float(rows.mean() - center)
    if direction == "left":
        return float(center - cols.mean())
    return float(cols.mean() - center)


def _half_check(cells: list[tuple[int, int]], inner: int, direction: str) -> bool:
    center = 0.5 * (inner - 1)
    if direction == "up":
        return max(r for r, _ in cells) <= center
    if direction == "down":
        return min(r for r, _ in cells) >= center
    if direction == "left":
        return max(c for _, c in cells) <= center
    return min(c for _, c in cells) >= center


def _choose_seed(inner: int, direction: str, count: int, length: int) -> list[tuple[int, int]] | None:
    candidates: list[list[tuple[int, int]]] = []
    for shape in _SHAPE_VARIANTS[count]:
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        row_starts = _RNG.permutation(inner - h + 1)
        col_starts = _RNG.permutation(inner - w + 1)
        for r0 in row_starts:
            for c0 in col_starts:
                cells = [(r0 + r, c0 + c) for r, c in shape]
                offset = _local_offset(cells, inner, direction)
                if _round_half_up(offset) != int(length) or offset < 0.55:
                    continue
                if not _half_check(cells, inner, direction):
                    continue
                candidates.append(cells)
    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _reserve_rel(side: int, direction: str, length: int, pad: int = RESERVE_PAD) -> tuple[int, int, int, int]:
    if direction == "up":
        raw = (-length - 1, 0, side - 1, side - 1)
    elif direction == "down":
        raw = (0, 0, side + length, side - 1)
    elif direction == "left":
        raw = (0, -length - 1, side - 1, side - 1)
    else:
        raw = (0, 0, side - 1, side + length)
    return raw[0] - pad, raw[1] - pad, raw[2] + pad, raw[3] + pad


def _sample_directions(count: int) -> list[str]:
    base = list(DIRECTIONS)
    _RNG.shuffle(base)
    if count <= 4:
        return base[:count]
    extra = [base[int(_RNG.integers(4))] for _ in range(count - 4)]
    result = base + extra
    _RNG.shuffle(result)
    return result


def _place_frames(specs: list[dict[str, object]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    dims = []
    total_area = 0
    for spec in specs:
        rr0, cc0, rr1, cc1 = spec["reserve_rel"]
        height = rr1 - rr0 + 1
        width = cc1 - cc0 + 1
        dims.append((height, width))
        total_area += height * width
    base = max(
        max(height for height, _ in dims) + 2,
        max(width for _, width in dims) + 2,
        int(np.ceil(np.sqrt(total_area * 1.35))),
    )
    h = int(base + _RNG.integers(0, 6))
    w = int(base + _RNG.integers(0, 6))

    occupancy = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)

    for idx in order:
        rr0, cc0, rr1, cc1 = specs[idx]["reserve_rel"]
        row_min = -rr0
        row_max = h - 1 - rr1
        col_min = -cc0
        col_max = w - 1 - cc1
        candidates: list[tuple[int, int]] = []
        for r0 in range(row_min, row_max + 1):
            for c0 in range(col_min, col_max + 1):
                ar0 = r0 + rr0
                ac0 = c0 + cc0
                ar1 = r0 + rr1
                ac1 = c0 + cc1
                if occupancy[ar0 : ar1 + 1, ac0 : ac1 + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        choice = candidates[int(_RNG.integers(len(candidates)))]
        positions[idx] = choice
        ar0 = choice[0] + rr0
        ac0 = choice[1] + cc0
        ar1 = choice[0] + rr1
        ac1 = choice[1] + cc1
        occupancy[ar0 : ar1 + 1, ac0 : ac1 + 1] = True

    return h, w, [pos for pos in positions if pos is not None]


def _paint_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int, direction: str, seed: list[tuple[int, int]]) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    mid = side // 2
    if direction == "up":
        grid[r0 - 1, c0 + mid] = color
    elif direction == "down":
        grid[r1 + 1, c0 + mid] = color
    elif direction == "left":
        grid[r0 + mid, c0 - 1] = color
    else:
        grid[r0 + mid, c1 + 1] = color
    for rr, cc in seed:
        grid[r0 + 1 + rr, c0 + 1 + cc] = GREY


def _gen_attempt() -> np.ndarray | None:
    frame_count = int(_RNG.integers(MIN_FRAMES, MAX_FRAMES + 1))
    sides = sorted(int(v) for v in _RNG.choice(FRAME_SIDES, size=frame_count, replace=False))
    colors = [int(v) for v in _RNG.choice(FRAME_COLOR_POOL, size=frame_count, replace=False)]
    directions = _sample_directions(frame_count)
    counts = [int(v) for v in _RNG.permutation(np.arange(1, frame_count + 1))]

    specs: list[dict[str, object]] = []
    for side, color, direction, count in zip(sides, colors, directions, counts):
        max_length = min(3, (side - 3) // 2)
        length = int(_RNG.integers(1, max_length + 1))
        seed = _choose_seed(side - 2, direction, count, length)
        if seed is None:
            return None
        specs.append(
            {
                "side": side,
                "color": color,
                "direction": direction,
                "count": count,
                "length": length,
                "seed": seed,
                "reserve_rel": _reserve_rel(side, direction, length),
            }
        )

    placed = _place_frames(specs)
    if placed is None:
        return None
    h, w, positions = placed
    grid = np.full((h, w), BG, dtype=int)
    for spec, (r0, c0) in zip(specs, positions):
        _paint_frame(grid, r0, c0, spec["side"], spec["color"], spec["direction"], spec["seed"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
