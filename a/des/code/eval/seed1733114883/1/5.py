from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
ANCHOR = 9
SEED_COLORS = (1, 3, 4, 5, 6, 7, 8)
DIRS = ("up", "right", "down", "left")
CW = {"up": "right", "right": "down", "down": "left", "left": "up"}
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
CORNERS = ("tl", "tr", "bl", "br")
PATTERN_KINDS = ("rows", "cols", "checker", "frame", "quadrants")

GEN_ATTEMPTS = 400
GRID_MIN = 12
GRID_MAX = 30
TARGET_STEPS = (1, 2, 3, 4)


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1) + 1, int(c1) + 1


def _fits_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= int(r0) and 0 <= int(c0) and int(r1) <= int(shape[0]) and int(c1) <= int(shape[1])


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    )


def _next_bbox(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0)
    w = int(c1 - c0)
    if direction == "right":
        return int(r0), int(c1), int(r1), int(c1 + w)
    if direction == "down":
        return int(r1), int(c0), int(r1 + h), int(c1)
    if direction == "left":
        return int(r0), int(c0 - w), int(r1), int(c0)
    return int(r0 - h), int(c0), int(r0), int(c1)


def _is_l_arrow(coords: np.ndarray) -> bool:
    pts = {tuple(map(int, xy)) for xy in coords}
    if len(pts) != 3:
        return False
    degrees = []
    for r, c in pts:
        degree = 0
        for dr, dc in ORTHO:
            degree += int((int(r + dr), int(c + dc)) in pts)
        degrees.append(degree)
    return sorted(degrees) == [1, 1, 2]


def _anchor_color(seed_patch: np.ndarray, ar: int, ac: int) -> int:
    h, w = seed_patch.shape
    vals: list[int] = []
    for rr in range(max(0, int(ar) - 1), min(h, int(ar) + 2)):
        for cc in range(max(0, int(ac) - 1), min(w, int(ac) + 2)):
            if rr == int(ar) and cc == int(ac):
                continue
            v = int(seed_patch[rr, cc])
            if v not in {BG, RED, ANCHOR}:
                vals.append(v)
    counts = Counter(vals)
    assert counts, "anchor neighborhood should reveal the hidden color"
    return int(max(counts, key=lambda color: (counts[color], color)))


def _arrow_direction(red_coords: np.ndarray, seed_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = seed_bbox
    rows = red_coords[:, 0]
    cols = red_coords[:, 1]
    candidates: list[str] = []
    if int(cols.min()) == int(c1) and bool(np.all(cols >= int(c1))):
        candidates.append("right")
    if int(cols.max()) == int(c0 - 1) and bool(np.all(cols <= int(c0 - 1))):
        candidates.append("left")
    if int(rows.min()) == int(r1) and bool(np.all(rows >= int(r1))):
        candidates.append("down")
    if int(rows.max()) == int(r0 - 1) and bool(np.all(rows <= int(r0 - 1))):
        candidates.append("up")
    assert len(candidates) == 1, "expected a unique side-attached arrow"
    return candidates[0]


def _extract_seed_and_direction(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], str]:
    g = np.asarray(grid, dtype=int)
    red_coords = np.argwhere(g == RED)
    assert red_coords.shape[0] == 3, "expected one 3-pixel red arrow"
    assert _is_l_arrow(red_coords), "red marker should be an L arrow"

    seed_mask = (g != BG) & (g != RED)
    seed_bbox = _bbox_of(seed_mask)
    direction = _arrow_direction(red_coords, seed_bbox)

    r0, c0, r1, c1 = seed_bbox
    seed_patch = g[r0:r1, c0:c1].copy()
    anchor_coords = np.argwhere(seed_patch == ANCHOR)
    assert anchor_coords.shape[0] == 1, "expected one maroon anchor"
    ar, ac = map(int, anchor_coords[0])
    h, w = seed_patch.shape
    assert (ar, ac) in {(0, 0), (0, int(w - 1)), (int(h - 1), 0), (int(h - 1), int(w - 1))}
    seed_patch[ar, ac] = _anchor_color(seed_patch, ar, ac)
    assert not bool(np.any(seed_patch == RED)), "seed colors must be distinct from the arrow color"
    return seed_patch, seed_bbox, direction


def _grow(seed_patch: np.ndarray, seed_bbox: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = seed_bbox
    out[r0:r1, c0:c1] = seed_patch
    current_bbox = tuple(map(int, seed_bbox))
    current_dir = direction
    while True:
        next_bbox = _next_bbox(current_bbox, current_dir)
        if not _fits_bbox(next_bbox, shape):
            break
        cr0, cc0, cr1, cc1 = current_bbox
        nr0, nc0, nr1, nc1 = next_bbox
        patch = out[cr0:cr1, cc0:cc1].copy()
        out[nr0:nr1, nc0:nc1] = patch
        current_bbox = _union_bbox(current_bbox, next_bbox)
        current_dir = CW[current_dir]
    return out


def _max_steps(seed_bbox: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> int:
    steps = 0
    current_bbox = tuple(map(int, seed_bbox))
    current_dir = direction
    while True:
        next_bbox = _next_bbox(current_bbox, current_dir)
        if not _fits_bbox(next_bbox, shape):
            return int(steps)
        steps += 1
        current_bbox = _union_bbox(current_bbox, next_bbox)
        current_dir = CW[current_dir]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the seed pattern by repeatedly copying the current assembled block.

    The input contains one small seed, one red L-shaped direction arrow attached
    to one side of the seed, and one maroon anchor hiding a corner color inside
    the seed. The arrow decides the first growth side; later steps rotate that
    direction clockwise. Each step duplicates the whole current assembly and
    concatenates it flush to the chosen side. Growth stops just before the next
    duplication would leave the grid, and the output contains only the cleaned
    repeated seed pattern.
    """
    g = np.asarray(grid, dtype=int)
    seed_patch, seed_bbox, direction = _extract_seed_and_direction(g)
    return _grow(seed_patch, seed_bbox, direction, g.shape)


def _corner_rc(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, int(w - 1)
    if corner == "bl":
        return int(h - 1), 0
    return int(h - 1), int(w - 1)


def _corner_slices(corner: str, h: int, w: int) -> tuple[slice, slice]:
    if corner in {"tl", "tr"}:
        rs = slice(0, 2)
    else:
        rs = slice(int(h - 2), int(h))
    if corner in {"tl", "bl"}:
        cs = slice(0, 2)
    else:
        cs = slice(int(w - 2), int(w))
    return rs, cs


def _seed_pattern(h: int, w: int, corner: str) -> np.ndarray:
    palette = random.sample(SEED_COLORS, k=random.randint(2, 4))
    kind = random.choice(PATTERN_KINDS)
    seed = np.empty((h, w), dtype=int)

    if kind == "rows":
        while True:
            row_colors = [random.choice(palette) for _ in range(h)]
            if len(set(row_colors)) >= 2:
                break
        seed[:] = np.array(row_colors, dtype=int)[:, None]
    elif kind == "cols":
        while True:
            col_colors = [random.choice(palette) for _ in range(w)]
            if len(set(col_colors)) >= 2:
                break
        seed[:] = np.array(col_colors, dtype=int)[None, :]
    elif kind == "checker":
        offset = random.randrange(len(palette))
        twist = random.choice((1, 2, 3))
        for r in range(h):
            for c in range(w):
                seed[r, c] = int(palette[(offset + r + twist * c) % len(palette)])
    elif kind == "frame":
        border = palette[0]
        inner = palette[1]
        seed.fill(int(border))
        seed[1:-1, 1:-1] = int(inner)
        if len(palette) >= 3:
            if random.random() < 0.5:
                seed[random.randrange(h), :] = int(palette[2])
            else:
                seed[:, random.randrange(w)] = int(palette[2])
    else:
        rcut = random.randint(1, h - 1)
        ccut = random.randint(1, w - 1)
        quad_colors = [random.choice(palette) for _ in range(4)]
        if len(set(quad_colors)) == 1:
            quad_colors[3] = palette[1]
        seed[:rcut, :ccut] = int(quad_colors[0])
        seed[:rcut, ccut:] = int(quad_colors[1])
        seed[rcut:, :ccut] = int(quad_colors[2])
        seed[rcut:, ccut:] = int(quad_colors[3])

    rs, cs = _corner_slices(corner, h, w)
    corner_color = random.choice(palette)
    seed[rs, cs] = int(corner_color)

    if len(set(map(int, seed.ravel()))) == 1:
        ar, ac = _corner_rc(corner, h, w)
        rr = int(h - 1) if ar == 0 else 0
        cc = int(w - 1) if ac == 0 else 0
        alternatives = [color for color in SEED_COLORS if color != int(seed[ar, ac])]
        seed[rr, cc] = int(random.choice(alternatives))

    return seed


def _seed_with_anchor(seed: np.ndarray, corner: str) -> np.ndarray:
    out = seed.copy()
    ar, ac = _corner_rc(corner, *seed.shape)
    out[ar, ac] = ANCHOR
    return out


def _arrow_cells(r0: int, c0: int, h: int, w: int, direction: str) -> list[tuple[int, int]]:
    if direction in {"right", "left"}:
        t = random.randint(0, h - 2)
        tip_end = random.randint(0, 1)
        if direction == "right":
            return [
                (int(r0 + t), int(c0 + w)),
                (int(r0 + t + 1), int(c0 + w)),
                (int(r0 + t + tip_end), int(c0 + w + 1)),
            ]
        return [
            (int(r0 + t), int(c0 - 1)),
            (int(r0 + t + 1), int(c0 - 1)),
            (int(r0 + t + tip_end), int(c0 - 2)),
        ]

    t = random.randint(0, w - 2)
    tip_end = random.randint(0, 1)
    if direction == "down":
        return [
            (int(r0 + h), int(c0 + t)),
            (int(r0 + h), int(c0 + t + 1)),
            (int(r0 + h + 1), int(c0 + t + tip_end)),
        ]
    return [
        (int(r0 - 1), int(c0 + t)),
        (int(r0 - 1), int(c0 + t + 1)),
        (int(r0 - 2), int(c0 + t + tip_end)),
    ]


def _arrow_fits(r0: int, c0: int, h: int, w: int, direction: str, shape: tuple[int, int]) -> bool:
    H, W = shape
    if direction == "right":
        return int(c0 + w + 1) < int(W)
    if direction == "left":
        return int(c0) >= 2
    if direction == "down":
        return int(r0 + h + 1) < int(H)
    return int(r0) >= 2


def _candidate_positions(
    H: int, W: int, h: int, w: int, direction: str, target_steps: int
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            bbox = (int(r0), int(c0), int(r0 + h), int(c0 + w))
            if not _arrow_fits(r0, c0, h, w, direction, (H, W)):
                continue
            if _max_steps(bbox, direction, (H, W)) == int(target_steps):
                positions.append((int(r0), int(c0)))
    return positions


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_ATTEMPTS):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        H = random.randint(max(GRID_MIN, h + 4), GRID_MAX)
        W = random.randint(max(GRID_MIN, w + 4), GRID_MAX)
        direction = random.choice(DIRS)
        corner = random.choice(CORNERS)
        target_steps = random.choice(TARGET_STEPS)

        positions = _candidate_positions(H, W, h, w, direction, target_steps)
        if not positions:
            continue

        center_r = (H - h) / 2.0
        center_c = (W - w) / 2.0
        positions.sort(key=lambda rc: abs(rc[0] - center_r) + abs(rc[1] - center_c))
        keep = positions[: max(1, len(positions) // 3)]
        r0, c0 = random.choice(keep)

        seed = _seed_pattern(h, w, corner)
        grid = np.full((H, W), BG, dtype=int)
        grid[r0 : r0 + h, c0 : c0 + w] = _seed_with_anchor(seed, corner)
        for rr, cc in _arrow_cells(r0, c0, h, w, direction):
            grid[rr, cc] = RED

        assert int(np.sum(grid == RED)) == 3
        assert int(np.sum(grid == ANCHOR)) == 1
        expected = _grow(seed, (r0, c0, r0 + h, c0 + w), direction, grid.shape)
        actual = main(grid)
        assert np.array_equal(actual, expected), "generator should only emit solvable instances"
        return grid

    raise RuntimeError("failed to generate a valid spiral-growth instance")
