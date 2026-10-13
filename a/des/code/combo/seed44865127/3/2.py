from __future__ import annotations

import random
import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns) % 4)


def _perimeter_coords(size: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(size)]
    coords.extend((r, size - 1) for r in range(1, size - 1))
    coords.extend((size - 1, c) for c in range(size - 1, -1, -1))
    coords.extend((r, 0) for r in range(size - 2, 0, -1))
    return coords


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _analyze_square(
    grid: np.ndarray, bg: int, r0: int, c0: int, size: int
) -> dict | None:
    perimeter = _perimeter_coords(size)
    values = [int(grid[r0 + dr, c0 + dc]) for dr, dc in perimeter]
    gap_count = sum(v == bg for v in values)
    if gap_count > 3:
        return None

    frame_values = [v for v in values if v != bg]
    if not frame_values or len(set(frame_values)) != 1:
        return None
    frame_color = int(frame_values[0])

    inner = grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
    if inner.size == 0 or not bool((inner == bg).any()):
        return None
    if bool((inner == frame_color).any()):
        return None
    if not bool((inner != bg).any()):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)),
        "size": int(size),
        "gap_count": int(gap_count),
    }


def _find_regions(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    regions: list[dict] = []
    for size in range(min(h, w), 4, -1):
        for r0 in range(0, h - size + 1):
            for c0 in range(0, w - size + 1):
                candidate = _analyze_square(grid, bg, r0, c0, size)
                if candidate is None:
                    continue
                if any(_boxes_overlap(candidate["bbox"], reg["bbox"]) for reg in regions):
                    continue
                regions.append(candidate)
    regions.sort(key=lambda reg: (reg["bbox"][0], reg["bbox"][1]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed region's interior by 90 degrees per border gap.

    Input structure:
    - Background is the dominant color.
    - Square one-pixel frames use one foreground color.
    - Each frame border has 0-3 missing border pixels.
    - Each interior contains a small asymmetric pattern in another color.

    Rule:
    - Count the number of border gaps for each frame.
    - Rotate the entire interior subgrid clockwise by gap_count * 90 degrees.
    - Keep the border unchanged, including its gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = g.copy()
    for region in _find_regions(g, bg):
        r0, c0, r1, c1 = map(int, region["bbox"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _rotate_cw(inner, region["gap_count"])
    return out


# --------------- generator ---------------

MIN_GRID = 18
MAX_GRID = 30
MIN_REGIONS = 3
MAX_REGIONS = 5
MIN_SIZE = 7
PADDING = 2
MAX_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _balanced_gap_counts(n_regions: int, rng: random.Random) -> list[int]:
    result: list[int] = []
    while len(result) < n_regions:
        batch = [0, 1, 2, 3]
        rng.shuffle(batch)
        result.extend(batch)
    return result[:n_regions]


def _sample_gap_cells(size: int, gap_count: int, rng: random.Random) -> set[tuple[int, int]]:
    if gap_count == 0:
        return set()

    coords = _perimeter_coords(size)
    cycle_len = len(coords)
    valid = [
        idx
        for idx, (r, c) in enumerate(coords)
        if not (r in (0, size - 1) and c in (0, size - 1))
    ]

    for _ in range(200):
        chosen = rng.sample(valid, gap_count)
        ok = True
        for i, idx_a in enumerate(chosen):
            for idx_b in chosen[i + 1 :]:
                dist = min((idx_a - idx_b) % cycle_len, (idx_b - idx_a) % cycle_len)
                if dist <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return {coords[idx] for idx in chosen}
    raise RuntimeError("failed to sample non-adjacent gaps")


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _all_rotations_distinct(arr: np.ndarray) -> bool:
    return all(not np.array_equal(arr, _rotate_cw(arr, k)) for k in (1, 2, 3))


def _sample_pattern_canvas(inner_size: int, color: int, bg: int, rng: random.Random) -> np.ndarray:
    max_patch = min(4, inner_size - 2)
    dims = [
        (h, w)
        for h in range(2, max_patch + 1)
        for w in range(2, max_patch + 1)
        if h * w >= 4
    ]
    assert dims

    for _ in range(300):
        ph, pw = rng.choice(dims)
        area_hi = min(ph * pw - 1, 6)
        if area_hi < 3:
            continue
        area = rng.randint(3, area_hi)
        mask = _grow_connected_mask(ph, pw, area, rng)

        canvas = np.full((inner_size, inner_size), bg, dtype=int)
        pr = rng.randrange(1, inner_size - ph)
        pc = rng.randrange(1, inner_size - pw)
        sub = canvas[pr : pr + ph, pc : pc + pw]
        sub[mask] = color
        canvas = _rotate_cw(canvas, rng.randrange(4))
        if _all_rotations_distinct(canvas):
            return canvas

    raise RuntimeError("failed to sample asymmetric pattern")


def _find_positions(occupied: np.ndarray, size: int) -> list[tuple[int, int]]:
    h, w = map(int, occupied.shape)
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - size):
        for c0 in range(1, w - size):
            rr0 = max(0, r0 - PADDING)
            rr1 = min(h, r0 + size + PADDING)
            cc0 = max(0, c0 - PADDING)
            cc1 = min(w, c0 + size + PADDING)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                positions.append((r0, c0))
    return positions


def _draw_region(
    grid: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    frame_color: int,
    inner_canvas: np.ndarray,
    gaps: set[tuple[int, int]],
) -> None:
    for dr, dc in _perimeter_coords(size):
        if (dr, dc) not in gaps:
            grid[r0 + dr, c0 + dc] = frame_color
    grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = inner_canvas


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg, frame_color, pattern_color = rng.sample(PALETTE, 3)

    n_regions = rng.randint(MIN_REGIONS, MAX_REGIONS)
    max_size = {3: 11, 4: 10, 5: 9}[n_regions]
    sizes = [rng.randint(MIN_SIZE, max_size) for _ in range(n_regions)]

    min_side = max(MIN_GRID, max(sizes) + 2, 18 + max(0, n_regions - 4) * 2)
    if min_side > MAX_GRID:
        return None
    height = rng.randint(min_side, MAX_GRID)
    width = rng.randint(min_side, MAX_GRID)

    gap_counts = _balanced_gap_counts(n_regions, rng)
    occupied = np.zeros((height, width), dtype=bool)
    specs: list[dict | None] = [None] * n_regions

    for idx in sorted(range(n_regions), key=lambda i: sizes[i], reverse=True):
        size = int(sizes[idx])
        positions = _find_positions(occupied, size)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        inner = _sample_pattern_canvas(size - 2, pattern_color, bg, rng)
        gaps = _sample_gap_cells(size, gap_counts[idx], rng)
        specs[idx] = {
            "r0": int(r0),
            "c0": int(c0),
            "size": size,
            "gap_count": int(gap_counts[idx]),
            "inner": inner,
            "gaps": gaps,
        }
        rr0 = max(0, r0 - PADDING)
        rr1 = min(height, r0 + size + PADDING)
        cc0 = max(0, c0 - PADDING)
        cc1 = min(width, c0 + size + PADDING)
        occupied[rr0:rr1, cc0:cc1] = True

    specs = [spec for spec in specs if spec is not None]
    inp = np.full((height, width), bg, dtype=int)
    for spec in specs:
        _draw_region(
            inp,
            spec["r0"],
            spec["c0"],
            spec["size"],
            frame_color,
            spec["inner"],
            spec["gaps"],
        )

    expected = inp.copy()
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        size = int(spec["size"])
        expected[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = _rotate_cw(
            spec["inner"], spec["gap_count"]
        )

    if _get_bg(inp) != bg:
        return None
    if len(_find_regions(inp, bg)) != n_regions:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
