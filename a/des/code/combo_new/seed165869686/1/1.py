from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE_BY_DIR = {
    "left": "tl",
    "right": "tr",
    "down": "bl",
    "up": "br",
}

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_H_RANGE = (5, 8)
FRAME_W_RANGE = (5, 8)
FRAME_MARGIN = 2
GAP_RANGE = (2, 5)
BAR_LEN_RANGE = (2, 4)
GEN_RETRIES = 400
LAYOUT_RETRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _corner_bars(grid: np.ndarray) -> dict[str, list[int]]:
    mask = (grid != BLACK) & (grid != GREY)
    bars: dict[str, list[int]] = {}
    h, w = grid.shape
    for coords in _components(mask):
        area = int(coords.shape[0])
        if area == 1:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        assert r0 == r1, "palette bars must be horizontal"
        assert area == c1 - c0 + 1, "palette bar must be contiguous"
        assert BAR_LEN_RANGE[0] <= area <= BAR_LEN_RANGE[1], "palette bar length must be 2-4"

        seq = [int(grid[r0, c]) for c in range(c0, c1 + 1)]
        assert len(set(seq)) == 2, "palette bar must alternate exactly two colors"
        assert all(a != b for a, b in zip(seq, seq[1:])), "palette bar must alternate"

        key = ("t" if r0 < h // 2 else "b") + ("l" if c0 < w // 2 else "r")
        assert key not in bars, "each corner must contain exactly one palette bar"
        bars[key] = seq

    assert set(bars) == {"tl", "tr", "bl", "br"}, "expected four corner palette bars"
    return bars


def _frames(grid: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != BLACK) & (grid != GREY)
    color_components = _components(mask)
    seeds = [tuple(map(int, coords[0])) for coords in color_components if int(coords.shape[0]) == 1]
    assert seeds, "expected at least one seed"

    frames: list[dict[str, object]] = []
    used_boxes: set[tuple[int, int, int, int]] = set()
    h, w = grid.shape
    for sr, sc in seeds:
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(0, sr):
            for r1 in range(sr + 1, h):
                if r1 - r0 < 3:
                    continue
                if int(grid[r0, sc]) != BLACK or int(grid[r1, sc]) != BLACK:
                    continue
                for c0 in range(0, sc):
                    for c1 in range(sc + 1, w):
                        if c1 - c0 < 3:
                            continue
                        if int(grid[sr, c0]) != BLACK or int(grid[sr, c1]) != BLACK:
                            continue

                        ok = True
                        for c in range(c0, c1 + 1):
                            if int(grid[r0, c]) != (BLACK if c == sc else GREY):
                                ok = False
                                break
                            if int(grid[r1, c]) != (BLACK if c == sc else GREY):
                                ok = False
                                break
                        if not ok:
                            continue

                        for r in range(r0, r1 + 1):
                            if int(grid[r, c0]) != (BLACK if r == sr else GREY):
                                ok = False
                                break
                            if int(grid[r, c1]) != (BLACK if r == sr else GREY):
                                ok = False
                                break
                        if not ok:
                            continue

                        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                        if np.any(interior == GREY):
                            continue
                        inner_colors = np.argwhere((interior != BLACK) & (interior != GREY))
                        if inner_colors.shape[0] != 1:
                            continue
                        rr, cc = map(int, inner_colors[0] + np.array([r0 + 1, c0 + 1], dtype=int))
                        if (rr, cc) != (sr, sc):
                            continue
                        candidates.append((r0, c0, r1, c1))

        assert len(candidates) == 1, "each seed must belong to exactly one gapped rectangle"
        box = candidates[0]
        assert box not in used_boxes, "two seeds cannot share the same frame"
        used_boxes.add(box)
        frames.append({"bbox": box, "seed": (sr, sc), "seed_color": int(grid[sr, sc])})

    return frames


def _neighbor_color(seq: list[int], seed_color: int) -> int:
    neighbors: set[int] = set()
    for i, color in enumerate(seq):
        if color != seed_color:
            continue
        if i > 0:
            neighbors.add(int(seq[i - 1]))
        if i + 1 < len(seq):
            neighbors.add(int(seq[i + 1]))
    neighbors.discard(seed_color)
    assert len(neighbors) == 1, "seed color must appear in the palette bar with one alternating neighbor color"
    return next(iter(neighbors))


def _ray_cells(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], str]:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells, "boundary"
        if int(grid[r, c]) == GREY:
            return cells, "grey"
        cells.append((r, c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four rays out of each gapped gray frame, coloring each ray from the corner palette bar cross-mapped to its gap direction.

    Input structure:
    - The background is black.
    - One or more gray rectangular one-cell-thick frames appear on the black background, each with exactly one black gap on every side.
    - Each frame encloses exactly one colored seed pixel, and the seed row/column align with the left-right and top-bottom gaps.
    - Four short horizontal two-color bars sit in the corners and act as the palette keys.

    Rule:
    - Detect every gray frame together with its four side gaps and its seed.
    - From the seed, paint one straight ray in each cardinal direction; the ray passes through the matching gap and continues until it leaves the grid or would hit a gray pixel.
    - Use the cross-mapped corner palette bar to choose the ray color: top-left for the left ray, top-right for the right ray, bottom-left for the down ray, and bottom-right for the up ray.
    - Inside that palette bar, find the seed color and replace the ray with the alternating neighbor color; keep the gray frame and seed unchanged.

    Key insight:
    - The palette lookup is indirect and cross-wired: the useful bar is not on the intuitive same side, and the emitted color is the seed color's alternating partner rather than the seed color itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bars = _corner_bars(g)
    frames = _frames(g)

    out = g.copy()
    for frame in frames:
        sr, sc = frame["seed"]  # type: ignore[index]
        seed_color = int(frame["seed_color"])  # type: ignore[arg-type]
        for direction_name, delta in DIRS.items():
            palette_key = PALETTE_BY_DIR[direction_name]
            paint_color = _neighbor_color(bars[palette_key], seed_color)
            cells, _stop = _ray_cells(g, (sr, sc), delta)
            for r, c in cells:
                out[r, c] = paint_color
    return out


def _alternating_bar(seed_color: int, partner_color: int, length: int) -> list[int]:
    start_with_seed = random.choice([False, True])
    seq = []
    for i in range(length):
        use_seed = (i % 2 == 0) == start_with_seed
        seq.append(seed_color if use_seed else partner_color)
    return seq


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], seed: tuple[int, int], seed_color: int) -> None:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, sc] = BLACK
    grid[r1, sc] = BLACK
    grid[sr, c0] = BLACK
    grid[sr, c1] = BLACK
    grid[sr, sc] = seed_color


def _flip_frames(
    frames: list[dict[str, tuple[int, int] | tuple[int, int, int, int]]],
    height: int,
    width: int,
    flip_h: bool,
    flip_v: bool,
) -> list[dict[str, tuple[int, int] | tuple[int, int, int, int]]]:
    out: list[dict[str, tuple[int, int] | tuple[int, int, int, int]]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        sr, sc = frame["seed"]  # type: ignore[index]
        if flip_h:
            c0, c1 = width - 1 - c1, width - 1 - c0
            sc = width - 1 - sc
        if flip_v:
            r0, r1 = height - 1 - r1, height - 1 - r0
            sr = height - 1 - sr
        out.append({"bbox": (r0, c0, r1, c1), "seed": (sr, sc)})
    return out


def _build_pair_horizontal(height: int, width: int, safe_cols: set[int]) -> list[dict[str, tuple[int, int] | tuple[int, int, int, int]]] | None:
    for _ in range(LAYOUT_RETRIES):
        h1 = random.randint(*FRAME_H_RANGE)
        h2 = random.randint(*FRAME_H_RANGE)
        w1 = random.randint(*FRAME_W_RANGE)
        w2 = random.randint(*FRAME_W_RANGE)
        gap = random.randint(*GAP_RANGE)
        max_h = max(h1, h2)
        total_w = w1 + gap + w2
        if height < 2 * FRAME_MARGIN + max_h or width < 2 * FRAME_MARGIN + total_w:
            continue

        r0 = random.randint(FRAME_MARGIN, height - FRAME_MARGIN - max_h)
        c0 = random.randint(FRAME_MARGIN, width - FRAME_MARGIN - total_w)
        rows_overlap = list(range(r0 + 1, r0 + min(h1, h2) - 1))
        if len(rows_overlap) < 2:
            continue

        row1, row2 = random.sample(rows_overlap, 2)
        cols1 = [c for c in range(c0 + 1, c0 + w1 - 1) if c in safe_cols]
        c0b = c0 + w1 + gap
        cols2 = [c for c in range(c0b + 1, c0b + w2 - 1) if c in safe_cols]
        if not cols1 or not cols2:
            continue

        col1 = random.choice(cols1)
        col2 = random.choice(cols2)
        return [
            {"bbox": (r0, c0, r0 + h1 - 1, c0 + w1 - 1), "seed": (row1, col1)},
            {"bbox": (r0, c0b, r0 + h2 - 1, c0b + w2 - 1), "seed": (row2, col2)},
        ]
    return None


def _build_pair_vertical(height: int, width: int, safe_cols: set[int]) -> list[dict[str, tuple[int, int] | tuple[int, int, int, int]]] | None:
    for _ in range(LAYOUT_RETRIES):
        h1 = random.randint(*FRAME_H_RANGE)
        h2 = random.randint(*FRAME_H_RANGE)
        w1 = random.randint(*FRAME_W_RANGE)
        w2 = random.randint(*FRAME_W_RANGE)
        gap = random.randint(*GAP_RANGE)
        max_w = max(w1, w2)
        total_h = h1 + gap + h2
        if width < 2 * FRAME_MARGIN + max_w or height < 2 * FRAME_MARGIN + total_h:
            continue

        c0 = random.randint(FRAME_MARGIN, width - FRAME_MARGIN - max_w)
        r0 = random.randint(FRAME_MARGIN, height - FRAME_MARGIN - total_h)
        cols_overlap = [c for c in range(c0 + 1, c0 + min(w1, w2) - 1) if c in safe_cols]
        if len(cols_overlap) < 2:
            continue

        col1, col2 = random.sample(cols_overlap, 2)
        rows1 = list(range(r0 + 1, r0 + h1 - 1))
        r0b = r0 + h1 + gap
        rows2 = list(range(r0b + 1, r0b + h2 - 1))
        if not rows1 or not rows2:
            continue

        row1 = random.choice(rows1)
        row2 = random.choice(rows2)
        return [
            {"bbox": (r0, c0, r0 + h1 - 1, c0 + w1 - 1), "seed": (row1, col1)},
            {"bbox": (r0b, c0, r0b + h2 - 1, c0 + w2 - 1), "seed": (row2, col2)},
        ]
    return None


def _build_triple_l(height: int, width: int, safe_cols: set[int]) -> list[dict[str, tuple[int, int] | tuple[int, int, int, int]]] | None:
    for _ in range(LAYOUT_RETRIES):
        h1 = random.randint(*FRAME_H_RANGE)
        h2 = random.randint(*FRAME_H_RANGE)
        h3 = random.randint(*FRAME_H_RANGE)
        w1 = random.randint(*FRAME_W_RANGE)
        w2 = random.randint(*FRAME_W_RANGE)
        w3 = random.randint(*FRAME_W_RANGE)
        gap_h = random.randint(*GAP_RANGE)
        gap_v = random.randint(*GAP_RANGE)

        top_band = max(h1, h2)
        left_band = max(w1, w3)
        total_h = top_band + gap_v + h3
        total_w = left_band + gap_h + w2
        if height < 2 * FRAME_MARGIN + total_h or width < 2 * FRAME_MARGIN + total_w:
            continue

        r0 = random.randint(FRAME_MARGIN, height - FRAME_MARGIN - total_h)
        c0 = random.randint(FRAME_MARGIN, width - FRAME_MARGIN - total_w)
        rows_overlap = list(range(r0 + 1, r0 + min(h1, h2) - 1))
        cols_overlap = [c for c in range(c0 + 1, c0 + min(w1, w3) - 1) if c in safe_cols]
        if len(rows_overlap) < 2 or len(cols_overlap) < 2:
            continue

        row1, row2 = random.sample(rows_overlap, 2)
        col1, col3 = random.sample(cols_overlap, 2)

        c0b = c0 + left_band + gap_h
        r0c = r0 + top_band + gap_v
        cols2 = [c for c in range(c0b + 1, c0b + w2 - 1) if c in safe_cols]
        rows3 = list(range(r0c + 1, r0c + h3 - 1))
        if not cols2 or not rows3:
            continue

        col2 = random.choice(cols2)
        row3 = random.choice(rows3)
        return [
            {"bbox": (r0, c0, r0 + h1 - 1, c0 + w1 - 1), "seed": (row1, col1)},
            {"bbox": (r0, c0b, r0 + h2 - 1, c0b + w2 - 1), "seed": (row2, col2)},
            {"bbox": (r0c, c0, r0c + h3 - 1, c0 + w3 - 1), "seed": (row3, col3)},
        ]
    return None


def _analyze_rays(grid: np.ndarray) -> tuple[int, int] | None:
    seen: set[tuple[int, int]] = set()
    boundary_count = 0
    grey_count = 0
    for frame in _frames(grid):
        sr, sc = frame["seed"]  # type: ignore[index]
        for delta in DIRS.values():
            cells, stop_kind = _ray_cells(grid, (sr, sc), delta)
            if len(cells) < 2:
                return None
            for r, c in cells:
                if int(grid[r, c]) != BLACK:
                    return None
                if (r, c) in seen:
                    return None
                seen.add((r, c))
            if stop_kind == "boundary":
                boundary_count += 1
            else:
                grey_count += 1
    return boundary_count, grey_count


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    seed_color, left_color, right_color, down_color, up_color = random.sample(COLORS, 5)

    tl = _alternating_bar(seed_color, left_color, random.randint(*BAR_LEN_RANGE))
    tr = _alternating_bar(seed_color, right_color, random.randint(*BAR_LEN_RANGE))
    bl = _alternating_bar(seed_color, down_color, random.randint(*BAR_LEN_RANGE))
    br = _alternating_bar(seed_color, up_color, random.randint(*BAR_LEN_RANGE))

    edge_forbidden = max(len(tl), len(tr), len(bl), len(br))
    safe_cols = set(range(edge_forbidden, width - edge_forbidden))
    if len(safe_cols) < 8:
        return None

    layout = random.choice(["pair_h", "pair_v", "triple"])
    if layout == "pair_h":
        frames = _build_pair_horizontal(height, width, safe_cols)
        flip_h = random.choice([False, True])
        flip_v = random.choice([False, True])
    elif layout == "pair_v":
        frames = _build_pair_vertical(height, width, safe_cols)
        flip_h = random.choice([False, True])
        flip_v = random.choice([False, True])
    else:
        frames = _build_triple_l(height, width, safe_cols)
        flip_h = random.choice([False, True])
        flip_v = random.choice([False, True])
    if frames is None:
        return None

    frames = _flip_frames(frames, height, width, flip_h=flip_h, flip_v=flip_v)
    for frame in frames:
        _r0, c0, _r1, c1 = frame["bbox"]  # type: ignore[index]
        sr, sc = frame["seed"]  # type: ignore[index]
        if sc not in safe_cols:
            return None
        if not (c0 + 1 <= sc <= c1 - 1):
            return None
        if not (_r0 + 1 <= sr <= _r1 - 1):
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, : len(tl)] = tl
    grid[0, width - len(tr) :] = tr
    grid[height - 1, : len(bl)] = bl
    grid[height - 1, width - len(br) :] = br

    for frame in frames:
        _draw_frame(
            grid,
            frame["bbox"],  # type: ignore[arg-type]
            frame["seed"],  # type: ignore[arg-type]
            seed_color,
        )

    analysis = _analyze_rays(grid)
    if analysis is None:
        return None
    boundary_count, grey_count = analysis
    total = boundary_count + grey_count
    if boundary_count == 0 or grey_count == 0:
        return None
    if not (0.25 <= grey_count / total <= 0.75):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
