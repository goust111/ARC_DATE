from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400
RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False

    rel = np.zeros((h, w), dtype=bool)
    rel[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(rel, border))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components4(grid == color):
            if _is_rectangular_ring(coords):
                r0, c0, r1, c1 = _bbox(coords)
                frames.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                    }
                )
    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "expected at least one rectangular fence"
    return frames


def _frame_seeds(grid: np.ndarray, frame: dict, bg: int) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(interior != bg)
    seeds = [(int(r0 + 1 + rr), int(c0 + 1 + cc), int(interior[rr, cc])) for rr, cc in coords]
    seeds.sort(key=lambda seed: (seed[0], seed[1]))
    assert 1 <= len(seeds) <= 4, "each fence must contain between one and four seeds"
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each rectangular fence, every seed grows K concentric Chebyshev-square layers,
    where K is the total number of seeds inside that same fence. Overlaps keep the
    nearest seed's color, breaking ties by topmost then leftmost seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        seeds = _frame_seeds(g, frame, bg)
        k = len(seeds)
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                best_key = None
                best_color = None
                for sr, sc, color in seeds:
                    dist = max(abs(r - sr), abs(c - sc))
                    if dist > k:
                        continue
                    key = (dist, sr, sc)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_color = color
                if best_color is not None:
                    out[r, c] = best_color
    return out


def _spread_positions(inner_h: int, inner_w: int, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    positions = [cells[int(rng.integers(len(cells)))]]
    while len(positions) < k:
        best_key = None
        best_cells: list[tuple[int, int]] = []
        for cell in cells:
            if cell in positions:
                continue
            dist = min(max(abs(cell[0] - rr), abs(cell[1] - cc)) for rr, cc in positions)
            edge = min(cell[0], inner_h - 1 - cell[0], cell[1], inner_w - 1 - cell[1])
            key = (dist, edge)
            if best_key is None or key > best_key:
                best_key = key
                best_cells = [cell]
            elif key == best_key:
                best_cells.append(cell)
        positions.append(best_cells[int(rng.integers(len(best_cells)))])
    positions.sort()
    return positions


def _line_indices(length: int, k: int, rng: np.random.Generator) -> list[int] | None:
    if length < 2 * k - 1:
        return None
    for _ in range(80):
        pts = sorted(int(x) for x in rng.choice(length, size=k, replace=False))
        if all(b - a >= 2 for a, b in zip(pts, pts[1:])):
            return pts
    return list(range(0, 2 * k - 1, 2))


def _seed_positions(inner_h: int, inner_w: int, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if k == 1:
        return [(int(rng.integers(inner_h)), int(rng.integers(inner_w)))]

    use_line = rng.random() < 0.45
    if use_line:
        choices: list[list[tuple[int, int]]] = []
        cols = _line_indices(inner_w, k, rng)
        if cols is not None:
            row = int(rng.integers(inner_h))
            choices.append([(row, c) for c in cols])
        rows = _line_indices(inner_h, k, rng)
        if rows is not None:
            col = int(rng.integers(inner_w))
            choices.append([(r, col) for r in rows])
        if choices:
            picked = choices[int(rng.integers(len(choices)))]
            picked.sort()
            return picked

    return _spread_positions(inner_h, inner_w, k, rng)


def _layout_shape(n_frames: int, rng: np.random.Generator) -> tuple[int, int]:
    if n_frames == 1:
        return (1, 1)
    if n_frames == 2:
        return (1, 2) if rng.random() < 0.5 else (2, 1)
    return (2, 2)


def _max_inner_size(rows: int, cols: int) -> int:
    cells = rows * cols
    if cells == 1:
        return 11
    if cells == 2:
        return 9
    return 7


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(1, 5))
    rows, cols = _layout_shape(n_frames, rng)
    max_inner = _max_inner_size(rows, cols)

    slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    rng.shuffle(slots)
    active_slots = slots[:n_frames]

    frame_specs: dict[tuple[int, int], dict] = {}
    for slot in active_slots:
        k = int(rng.integers(1, 5))
        min_inner = max(3, k + 2)
        if min_inner > max_inner:
            return None
        inner_h = int(rng.integers(min_inner, max_inner + 1))
        inner_w = int(rng.integers(min_inner, max_inner + 1))
        frame_specs[slot] = {
            "k": k,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": inner_h + 2,
            "outer_w": inner_w + 2,
        }

    row_heights = []
    for rr in range(rows):
        need = max((spec["outer_h"] for (row, _), spec in frame_specs.items() if row == rr), default=0)
        row_heights.append(need + int(rng.integers(2, 6)))
    col_widths = []
    for cc in range(cols):
        need = max((spec["outer_w"] for (_, col), spec in frame_specs.items() if col == cc), default=0)
        col_widths.append(need + int(rng.integers(2, 6)))

    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))
    gap_h = int(rng.integers(2, 5))
    gap_w = int(rng.integers(2, 5))

    height = top + bottom + sum(row_heights) + gap_h * (rows - 1)
    width = left + right + sum(col_widths) + gap_w * (cols - 1)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)

    row_starts = []
    cursor = top
    for size in row_heights:
        row_starts.append(cursor)
        cursor += size + gap_h
    col_starts = []
    cursor = left
    for size in col_widths:
        col_starts.append(cursor)
        cursor += size + gap_w

    for (rr, cc), spec in frame_specs.items():
        slot_r0 = row_starts[rr]
        slot_c0 = col_starts[cc]
        slot_h = row_heights[rr]
        slot_w = col_widths[cc]
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]

        frame_r0 = int(slot_r0 + rng.integers(slot_h - outer_h + 1))
        frame_c0 = int(slot_c0 + rng.integers(slot_w - outer_w + 1))
        frame_r1 = frame_r0 + outer_h - 1
        frame_c1 = frame_c0 + outer_w - 1

        grid[frame_r0, frame_c0 : frame_c1 + 1] = FRAME_COLOR
        grid[frame_r1, frame_c0 : frame_c1 + 1] = FRAME_COLOR
        grid[frame_r0 : frame_r1 + 1, frame_c0] = FRAME_COLOR
        grid[frame_r0 : frame_r1 + 1, frame_c1] = FRAME_COLOR

        positions = _seed_positions(spec["inner_h"], spec["inner_w"], spec["k"], rng)
        colors = rng.choice(PALETTE, size=spec["k"], replace=False)
        for (ir, ic), color in zip(positions, colors):
            grid[frame_r0 + 1 + ir, frame_c0 + 1 + ic] = int(color)

    try:
        frames = _find_frames(grid, BG)
        if len(frames) != n_frames:
            return None
        for frame in frames:
            if frame["color"] != FRAME_COLOR:
                return None
            seeds = _frame_seeds(grid, frame, BG)
            if len(seeds) != len({(r, c) for r, c, _ in seeds}):
                return None
            if len({color for _, _, color in seeds}) != len(seeds):
                return None
        non_bg = set(map(tuple, np.argwhere(grid != BG)))
        claimed = set()
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            for c in range(c0, c1 + 1):
                claimed.add((r0, c))
                claimed.add((r1, c))
            for r in range(r0, r1 + 1):
                claimed.add((r, c0))
                claimed.add((r, c1))
            claimed.update((r, c) for r, c, _ in _frame_seeds(grid, frame, BG))
        if claimed != non_bg:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
