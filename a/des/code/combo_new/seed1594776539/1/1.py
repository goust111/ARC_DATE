from __future__ import annotations

import random

import numpy as np


BLACK = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 250


Point = tuple[int, int]


def _missing_run(arr: np.ndarray) -> tuple[int, int] | None:
    missing = np.flatnonzero(~arr)
    if missing.size < 2:
        return None
    if missing[0] == 0 or missing[-1] == arr.size - 1:
        return None
    if not np.all(np.diff(missing) == 1):
        return None
    return int(missing[0]), int(missing[-1])


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], str]:
    candidates: list[tuple[int, tuple[int, int, int, int], str]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == BLACK:
            continue

        coords = np.argwhere(grid == color)
        if len(coords) < 8:
            continue

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 4 or width < 4:
            continue

        mask = np.zeros((height, width), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if bool(mask[1:-1, 1:-1].any()):
            continue
        if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
            continue

        sides = {
            "top": mask[0, 1:-1],
            "bottom": mask[-1, 1:-1],
            "left": mask[1:-1, 0],
            "right": mask[1:-1, -1],
        }
        gap_side = None
        gap_len = None
        valid = True
        for side, arr in sides.items():
            if bool(arr.all()):
                continue
            run = _missing_run(arr)
            if run is None or gap_side is not None:
                valid = False
                break
            gap_side = side
            gap_len = run[1] - run[0] + 1

        if not valid or gap_side is None or gap_len is None:
            continue

        expected = 2 * height + 2 * width - 4 - gap_len
        if int(coords.shape[0]) != expected:
            continue
        candidates.append((color, (int(r0), int(c0), int(r1), int(c1)), gap_side))

    assert len(candidates) == 1, "expected exactly one gapped rectangular frame"
    return candidates[0]


def _path_cells(a: Point, b: Point, gap_side: str) -> list[Point]:
    (r1, c1), (r2, c2) = a, b

    if gap_side in ("top", "bottom"):
        assert r1 != r2, "vertical projection requires different rows"
        if c1 == c2:
            lo, hi = sorted((r1, r2))
            return [(r, c1) for r in range(lo, hi + 1)]

        if gap_side == "top":
            start, end = (a, b) if r1 < r2 else (b, a)
        else:
            start, end = (a, b) if r1 > r2 else (b, a)

        (sr, sc), (er, ec) = start, end
        cells = [(r, sc) for r in range(min(sr, er), max(sr, er) + 1)]
        cells.extend((er, c) for c in range(min(sc, ec), max(sc, ec) + 1) if c != sc)
        return cells

    assert c1 != c2, "horizontal projection requires different columns"
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return [(r1, c) for c in range(lo, hi + 1)]

    if gap_side == "left":
        start, end = (a, b) if c1 < c2 else (b, a)
    else:
        start, end = (a, b) if c1 > c2 else (b, a)

    (sr, sc), (er, ec) = start, end
    cells = [(sr, c) for c in range(min(sc, ec), max(sc, ec) + 1)]
    cells.extend((r, ec) for r in range(min(sr, er), max(sr, er) + 1) if r != sr)
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-color seed pair using the inward projection implied by the frame gap.

    Input structure:
    - The background is black.
    - One non-black color forms a one-pixel-thick rectangular frame with exactly one continuous gap on a single side.
    - Inside the frame, each remaining color appears exactly twice as isolated seed pixels.

    Rule:
    - Detect which frame side has the gap.
    - If the gap is on the top or bottom, connect each color pair with a vertical-first path; if it is on the left or right, connect each pair with a horizontal-first path.
    - When the seeds are already aligned on that axis, draw a straight segment; otherwise start from the seed nearer the gap side, move along the implied axis to the other seed's row or column, then turn once to finish the connection.

    Key insight:
    - The gap encodes not just the relevant axis but also which seed should be projected inward first, so top/bottom and left/right gaps yield mirrored L-shaped connections.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, (r0, c0, r1, c1), gap_side = _find_frame(g)

    pairs: list[tuple[int, Point, Point]] = []
    for color0 in np.unique(g):
        color = int(color0)
        if color in (BLACK, frame_color):
            continue
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each seed color must appear exactly twice"
        a = (int(coords[0, 0]), int(coords[0, 1]))
        b = (int(coords[1, 0]), int(coords[1, 1]))
        for r, c in (a, b):
            assert r0 < r < r1 and c0 < c < c1, "seed points must lie inside the frame"
        pairs.append((color, a, b))

    assert pairs, "expected at least one seed pair"

    out = g.copy()
    for color, a, b in pairs:
        cells = _path_cells(a, b, gap_side)
        for r, c in cells:
            assert r0 < r < r1 and c0 < c < c1, "paths must stay inside the frame"
            assert int(out[r, c]) in (BLACK, color), "different paths must not overlap"
        for r, c in cells:
            out[r, c] = color
    return out


def _split_margin(extra: int) -> tuple[int, int]:
    before = random.randint(1, extra - 1)
    return before, extra - before


def _build_frame(
    height: int,
    width: int,
    side: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    extra_rows = random.randint(2, min(8, 30 - height))
    extra_cols = random.randint(2, min(8, 30 - width))
    top_margin, bottom_margin = _split_margin(extra_rows)
    left_margin, right_margin = _split_margin(extra_cols)

    grid = np.zeros((height + extra_rows, width + extra_cols), dtype=int)
    r0, c0 = top_margin, left_margin
    r1, c1 = r0 + height - 1, c0 + width - 1

    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    interior_h = height - 2
    interior_w = width - 2
    if side in ("top", "bottom"):
        gap_w = random.randint(2, min(6, interior_w - 2))
        gap_offset = random.randint(1, interior_w - gap_w - 1)
        gc0 = c0 + 1 + gap_offset
        gc1 = gc0 + gap_w - 1
        row = r0 if side == "top" else r1
        grid[row, gc0 : gc1 + 1] = BLACK
    else:
        gap_h = random.randint(2, min(6, interior_h - 2))
        gap_offset = random.randint(1, interior_h - gap_h - 1)
        gr0 = r0 + 1 + gap_offset
        gr1 = gr0 + gap_h - 1
        col = c0 if side == "left" else c1
        grid[gr0 : gr1 + 1, col] = BLACK

    return grid, (r0, c0, r1, c1)


def _buffered(cells: set[Point], shape: tuple[int, int]) -> set[Point]:
    h, w = shape
    out: set[Point] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                out.add((rr, cc))
    return out


def _sample_candidate(side: str, rows: list[int], cols: list[int], kind: str) -> tuple[Point, Point, set[Point]] | None:
    if side in ("top", "bottom"):
        r_lo, r_hi = sorted(random.sample(rows, 2))
        if r_hi - r_lo < 2:
            return None
        if kind == "straight":
            col = random.choice(cols)
            a = (r_lo, col)
            b = (r_hi, col)
        else:
            c1, c2 = random.sample(cols, 2)
            if abs(c1 - c2) < 2:
                return None
            if side == "top":
                a = (r_lo, c1)
                b = (r_hi, c2)
            else:
                a = (r_hi, c1)
                b = (r_lo, c2)
    else:
        c_lo, c_hi = sorted(random.sample(cols, 2))
        if c_hi - c_lo < 2:
            return None
        if kind == "straight":
            row = random.choice(rows)
            a = (row, c_lo)
            b = (row, c_hi)
        else:
            r1, r2 = random.sample(rows, 2)
            if side == "left":
                a = (r1, c_lo)
                b = (r2, c_hi)
            else:
                a = (r1, c_hi)
                b = (r2, c_lo)

    return a, b, set(_path_cells(a, b, side))


def _sample_kinds(pair_count: int) -> list[str]:
    if pair_count == 1:
        return [random.choice(("straight", "elbow"))]
    straight_count = random.choice((pair_count // 2, (pair_count + 1) // 2))
    kinds = ["straight"] * straight_count + ["elbow"] * (pair_count - straight_count)
    random.shuffle(kinds)
    return kinds


def _place_pairs(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    pair_count: int,
) -> list[tuple[int, Point, Point]] | None:
    r0, c0, r1, c1 = frame_bbox
    rows = list(range(r0 + 2, r1 - 1))
    cols = list(range(c0 + 2, c1 - 1))
    if len(rows) < 2 or len(cols) < 2:
        return None

    colors = random.sample(SEED_COLORS, pair_count)
    kinds = _sample_kinds(pair_count)
    jobs = list(zip(colors, kinds))
    jobs.sort(key=lambda item: item[1] == "straight")

    blocked: set[Point] = set()
    placed: list[tuple[int, Point, Point]] = []
    for color, kind in jobs:
        for _ in range(PLACEMENT_RETRIES):
            candidate = _sample_candidate(side, rows, cols, kind)
            if candidate is None:
                continue
            a, b, cells = candidate
            if any(cell in blocked for cell in cells):
                continue
            placed.append((color, a, b))
            blocked |= _buffered(cells, grid.shape)
            break
        else:
            return None
    return placed


def _render_pair(
    side: str,
    interior_h: int,
    interior_w: int,
    pair_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    frame_h = interior_h + 2
    frame_w = interior_w + 2
    inp, frame_bbox = _build_frame(frame_h, frame_w, side)
    pairs = _place_pairs(inp, frame_bbox, side, pair_count)
    if pairs is None:
        return None

    out = inp.copy()
    for color, a, b in pairs:
        out_cells = _path_cells(a, b, side)
        for r, c in out_cells:
            out[r, c] = color
        for r, c in (a, b):
            inp[r, c] = color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDES)
    pair_count = random.randint(1, 4)
    min_span = 7 + 2 * pair_count
    max_span = min(20, min_span + 5)
    interior_h = random.randint(min_span, max_span)
    interior_w = random.randint(min_span, max_span)

    rendered = _render_pair(side, interior_h, interior_w, pair_count)
    if rendered is None:
        return None
    inp, out = rendered
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
