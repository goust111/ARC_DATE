from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
FRAME_PLACEMENT_ATTEMPTS = 200

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _is_rectangular_frame(frame_mask: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))

    actual = {tuple(map(int, cell)) for cell in coords}
    if actual != expected:
        return False

    if bool(frame_mask[r0 + 1 : r1, c0 + 1 : c1].any()):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangular fence from the palette index given by its interior marker count.

    Input structure:
    - The background is black.
    - The main area contains several disjoint one-pixel-thick grey rectangular fences.
    - Each fence surrounds a black interior with several colored single-pixel markers;
      marker colors are non-black and non-grey.
    - The bottom row contains a left-to-right palette of colored single pixels that
      defines an ordered sequence of output colors.

    Rule:
    - Read the palette colors from the bottom row in left-to-right order.
    - For each grey fence, count how many non-black marker pixels lie strictly inside it.
    - Use that count N as a 1-based palette index and recolor the entire fence with
      the N-th palette color.
    - Keep the marker pixels, background, and palette row unchanged.

    Key insight:
    - The fence color is determined by a two-step count-then-lookup rule: only the
      number of enclosed markers matters, not their specific colors or positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 4, "expected room for fences and a palette row"

    palette = [int(v) for v in g[-1, :] if int(v) != BLACK]
    assert palette, "expected at least one palette color on the bottom row"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert all(color not in (BLACK, GREY) for color in palette), "palette cannot use black or grey"

    frame_mask = g[:-1, :] == GREY
    frame_components = _components(frame_mask)
    assert frame_components, "expected at least one grey fence"

    out = g.copy()
    for coords in frame_components:
        assert _is_rectangular_frame(frame_mask, coords), "grey component must be a hollow rectangle"
        r0, c0, r1, c1 = _bbox(coords)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "frame interior must be nonempty"
        assert not bool((interior == GREY).any()), "frame interior cannot contain grey pixels"

        marker_count = int(((interior != BLACK) & (interior != GREY)).sum())
        assert 1 <= marker_count <= len(palette), "marker count must index the palette"
        out[coords[:, 0], coords[:, 1]] = int(palette[marker_count - 1])

    return out


def _random_palette() -> list[int]:
    palette_len = random.randint(3, len(COLOR_POOL))
    return random.sample(COLOR_POOL, palette_len)


def _candidate_frame_sizes(max_rows: int, max_cols: int, marker_count: int) -> list[tuple[int, int]]:
    options = []
    for outer_h in range(3, min(10, max_rows) + 1):
        for outer_w in range(3, min(10, max_cols) + 1):
            if (outer_h - 2) * (outer_w - 2) >= marker_count:
                options.append((outer_h, outer_w))
    return options


def _sample_marker_counts(fence_count: int, palette_len: int) -> list[int]:
    values = list(range(1, palette_len + 1))
    if fence_count <= palette_len and random.random() < 0.75:
        counts = random.sample(values, fence_count)
    else:
        counts = [random.randint(1, palette_len) for _ in range(fence_count)]
    random.shuffle(counts)
    return counts


def _paint_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _try_place_fences(height: int, width: int, counts: list[int]) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height - 1, width), dtype=bool)

    for marker_count in counts:
        placed = False
        sizes = _candidate_frame_sizes(height - 3, width - 2, marker_count)
        if not sizes:
            return None

        for _ in range(FRAME_PLACEMENT_ATTEMPTS):
            outer_h, outer_w = random.choice(sizes)
            max_r0 = height - outer_h - 2
            max_c0 = width - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue

            r0 = random.randint(1, max_r0)
            c0 = random.randint(1, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            pad_r0 = r0 - 1
            pad_c0 = c0 - 1
            pad_r1 = r1 + 1
            pad_c1 = c1 + 1
            if bool(occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any()):
                continue

            _paint_frame(grid, (r0, c0, r1, c1), GREY)
            occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True

            inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            marker_cells = random.sample(inner_cells, marker_count)
            marker_colors = random.sample(COLOR_POOL, marker_count)
            for (mr, mc), color in zip(marker_cells, marker_colors):
                grid[mr, mc] = int(color)

            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    palette = _random_palette()
    fence_count = random.randint(1, 4)
    counts = _sample_marker_counts(fence_count, len(palette))

    width_min = max(len(palette) + 2, 9)
    width = random.randint(width_min, 30)
    height = random.randint(8, 24)

    grid = _try_place_fences(height, width, counts)
    if grid is None:
        return None

    palette_start = random.randint(0, width - len(palette))
    grid[-1, palette_start : palette_start + len(palette)] = np.array(palette, dtype=int)

    if np.any(grid[-1, :] == GREY):
        return None

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
