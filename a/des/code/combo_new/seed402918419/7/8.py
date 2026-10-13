import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 300
BRANCHES = ("empty", "mono", "mixed")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden rectangular partition from opposite-edge seed pairs, then compress each region to one summary pixel.

    Input structure:
    - The background is black.
    - Non-black seed pixels appear only on the outer border, with matching colors on
      top/bottom edges defining vertical cuts and matching colors on left/right edges
      defining horizontal cuts.
    - The interior contains zero or more small foreground objects whose colors never
      reuse any seed color.

    Rule:
    - Match equal-colored top/bottom seeds to recover every vertical cut column, and
      equal-colored left/right seeds to recover every horizontal cut row.
    - Treat the outer border together with those hidden cut lines as the rectangle
      boundaries, and inspect each resulting subregion independently.
    - If a subregion contains exactly one non-black non-seed color, output that color
      for the region; if it contains multiple such colors or none at all, output black.

    Key insight:
    - The border colors are not ordinary objects: they are coordinate markers for an
      invisible grid, and the final answer is the compressed region-by-region summary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "grid must have an interior"

    cut_rows, cut_cols, seed_colors = _recover_cuts(g)
    row_spans = _region_spans_from_cuts(cut_rows, h)
    col_spans = _region_spans_from_cuts(cut_cols, w)

    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for out_r, (r0, r1) in enumerate(row_spans):
        for out_c, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = {
                int(color)
                for color in np.unique(patch)
                if int(color) != BG and int(color) not in seed_colors
            }
            if len(colors) == 1:
                out[out_r, out_c] = next(iter(colors))
    return out


def _recover_cuts(grid: np.ndarray) -> tuple[list[int], list[int], set[int]]:
    h, w = grid.shape
    assert int(grid[0, 0]) == BG and int(grid[0, w - 1]) == BG
    assert int(grid[h - 1, 0]) == BG and int(grid[h - 1, w - 1]) == BG

    top = {int(grid[0, c]): int(c) for c in range(1, w - 1) if int(grid[0, c]) != BG}
    bottom = {int(grid[h - 1, c]): int(c) for c in range(1, w - 1) if int(grid[h - 1, c]) != BG}
    left = {int(grid[r, 0]): int(r) for r in range(1, h - 1) if int(grid[r, 0]) != BG}
    right = {int(grid[r, w - 1]): int(r) for r in range(1, h - 1) if int(grid[r, w - 1]) != BG}

    cut_cols = []
    for color in sorted(set(top) & set(bottom), key=lambda value: top[value]):
        assert top[color] == bottom[color], "top/bottom seed pair must align vertically"
        cut_cols.append(int(top[color]))

    cut_rows = []
    for color in sorted(set(left) & set(right), key=lambda value: left[value]):
        assert left[color] == right[color], "left/right seed pair must align horizontally"
        cut_rows.append(int(left[color]))

    seed_colors = set(top) | set(bottom) | set(left) | set(right)
    return cut_rows, cut_cols, seed_colors


def _region_spans_from_cuts(cuts: list[int], axis_len: int) -> list[tuple[int, int]]:
    cuts = sorted(int(cut) for cut in cuts)
    assert len(cuts) == len(set(cuts)), "cut positions must be unique"

    spans = []
    prev = 0
    for cut in cuts:
        start = prev + 1
        end = cut - 1
        assert start <= end, "every region span must be non-empty"
        spans.append((start, end))
        prev = cut

    start = prev + 1
    end = axis_len - 2
    assert start <= end, "final region span must be non-empty"
    spans.append((start, end))
    return spans


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid, expected = _gen_attempt(rng)
        if grid is None:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray | None, np.ndarray | None]:
    out_h = rng.randint(2, 4)
    out_w = rng.randint(2, 4)

    region_heights = [rng.randint(2, 5) for _ in range(out_h)]
    region_widths = [rng.randint(2, 5) for _ in range(out_w)]

    height = sum(region_heights) + (out_h - 1) + 2
    width = sum(region_widths) + (out_w - 1) + 2
    if height > 30 or width > 30:
        return None, None

    grid = np.zeros((height, width), dtype=int)
    row_spans, cut_rows = _spans_and_cuts(region_heights)
    col_spans, cut_cols = _spans_and_cuts(region_widths)

    n_seed_colors = len(cut_rows) + len(cut_cols)
    if n_seed_colors > len(PALETTE) - 3:
        return None, None

    shuffled = list(PALETTE)
    rng.shuffle(shuffled)
    seed_palette = shuffled[:n_seed_colors]
    object_palette = shuffled[n_seed_colors:]
    if len(object_palette) < 3:
        return None, None

    vertical_seed_colors = seed_palette[: len(cut_cols)]
    horizontal_seed_colors = seed_palette[len(cut_cols) :]
    for color, col in zip(vertical_seed_colors, cut_cols, strict=True):
        grid[0, col] = color
        grid[height - 1, col] = color
    for color, row in zip(horizontal_seed_colors, cut_rows, strict=True):
        grid[row, 0] = color
        grid[row, width - 1] = color

    branches = _branch_layout(out_h * out_w, rng)
    expected = np.zeros((out_h, out_w), dtype=int)
    index = 0
    for out_r, (r0, r1) in enumerate(row_spans):
        for out_c, (c0, c1) in enumerate(col_spans):
            branch = branches[index]
            index += 1
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            expected[out_r, out_c] = _paint_region(region, branch, object_palette, rng)

    if np.count_nonzero(expected) == 0:
        return None, None
    return grid, expected


def _spans_and_cuts(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans = []
    cuts = []
    pos = 1
    for idx, size in enumerate(sizes):
        spans.append((pos, pos + size - 1))
        pos += size
        if idx != len(sizes) - 1:
            cuts.append(pos)
            pos += 1
    return spans, cuts


def _branch_layout(count: int, rng: random.Random) -> list[str]:
    branches = []
    if count >= 3:
        branches.extend(BRANCHES)
    while len(branches) < count:
        branches.append(rng.choice(BRANCHES))
    rng.shuffle(branches)
    return branches


def _paint_region(region: np.ndarray, branch: str, object_palette: list[int], rng: random.Random) -> int:
    h, w = region.shape
    area = h * w
    assert area >= 4, "generator expects at least 2x2 regions"

    if branch == "empty":
        return BG

    if branch == "mono":
        color = int(rng.choice(object_palette))
        blobs = rng.randint(1, min(3, area))
        _paint_blobs(region, [color] * blobs, rng)
        return color

    n_colors = rng.randint(2, min(3, len(object_palette)))
    colors = [int(color) for color in rng.sample(object_palette, n_colors)]
    extra = rng.randint(0, min(2, area - n_colors))
    plan = colors + [int(rng.choice(colors)) for _ in range(extra)]
    _paint_blobs(region, plan, rng)
    return BG


def _paint_blobs(region: np.ndarray, colors: list[int], rng: random.Random) -> None:
    h, w = region.shape
    reserved = len(colors)
    for idx, color in enumerate(colors):
        remaining = reserved - idx - 1
        free = int(np.count_nonzero(region == BG))
        assert free >= remaining + 1, "not enough free cells to place remaining blobs"
        max_blob = min(4, free - remaining)
        target = rng.randint(1, max_blob)
        cells = _grow_blob(region, target, rng)
        for r, c in cells:
            region[r, c] = color


def _grow_blob(region: np.ndarray, target: int, rng: random.Random) -> list[tuple[int, int]]:
    h, w = region.shape
    free = [(r, c) for r in range(h) for c in range(w) if int(region[r, c]) == BG]
    assert free, "blob needs at least one free cell"
    start = rng.choice(free)
    blob = [start]
    used = {start}

    while len(blob) < target:
        frontier = []
        for r, c in blob:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(region[nr, nc]) == BG and (nr, nc) not in used:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nxt = rng.choice(frontier)
        used.add(nxt)
        blob.append(nxt)
    return blob


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
