import random

import numpy as np


BG = 0
GEN_RETRIES = 500
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (16, 26)
REGION_COUNT_RANGE = (3, 5)
RECT_H_RANGE = (3, 6)
RECT_W_RANGE = (4, 8)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _same_color_components(grid: np.ndarray, *, skip_top_row: bool) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    start_row = 1 if skip_top_row else 0
    for r in range(start_row, h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, np.asarray(cells, dtype=int)))
    return comps


def _ring_bbox(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    if int(coords.shape[0]) != 2 * height + 2 * width - 4:
        return None
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(region[0, :] == color):
        return None
    if not np.all(region[-1, :] == color):
        return None
    if not np.all(region[:, 0] == color):
        return None
    if not np.all(region[:, -1] == color):
        return None
    if np.any(region[1:-1, 1:-1] == color):
        return None
    return r0, c0, r1, c1


def _uniform_interior_color(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    values = np.unique(interior)
    assert values.size == 1, "region interior must be a single color"
    return int(values[0])


def _chain_colors(grid: np.ndarray) -> list[int]:
    cols = np.flatnonzero(grid[0] != BG)
    colors = [int(grid[0, c]) for c in cols]
    assert len(colors) >= 2, "top row must contain at least two chain colors"
    assert len(colors) == len(set(colors)), "chain colors must be unique"
    return colors


def _extract_regions(grid: np.ndarray) -> dict[int, dict[str, tuple[int, int, int, int] | int]]:
    regions: dict[int, dict[str, tuple[int, int, int, int] | int]] = {}
    for color, coords in _same_color_components(grid, skip_top_row=True):
        bbox = _ring_bbox(grid, color, coords)
        if bbox is None:
            continue
        assert color not in regions, "each border color must identify one region"
        regions[color] = {
            "bbox": bbox,
            "fill": _uniform_interior_color(grid, bbox),
        }
    assert regions, "expected at least one framed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate interior fill colors along the top-row border-color chain.

    Input structure:
    - The background is black.
    - Below the top row, each object is a one-pixel-thick rectangular border ring.
    - The top row contains one pixel of each border color in left-to-right chain order.

    Rule:
    - For each adjacent pair A→B in the chain, if region A is filled and region B
      is empty, copy A's fill color into B.
    - Process the chain from left to right so newly filled regions can propagate further.
    - Keep the borders and the chain row unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    chain = _chain_colors(g)
    regions = _extract_regions(g)
    assert set(chain) == set(regions), "chain colors must match region border colors"

    fills = {color: int(regions[color]["fill"]) for color in chain}
    out = g.copy()
    for src, dst in zip(chain, chain[1:]):
        if fills[src] != BG and fills[dst] == BG:
            fills[dst] = fills[src]

    for color in chain:
        r0, c0, r1, c1 = regions[color]["bbox"]  # type: ignore[index]
        out[r0 + 1 : r1, c0 + 1 : c1] = fills[color]
    return out


def _sample_fill_mask(rng: random.Random, n: int) -> list[int]:
    states = [rng.randint(0, 1) for _ in range(n)]
    mode = rng.choice(("cascade", "blocked", "empty_source"))
    start = rng.randint(0, n - 3)
    if mode == "cascade":
        states[start : start + 3] = [1, 0, 0]
    elif mode == "blocked":
        states[start : start + 3] = [1, 1, 0]
    else:
        states[start : start + 3] = [0, 1, 0]
    return states


def _sample_sizes(rng: random.Random, n: int) -> list[tuple[int, int]]:
    if n >= 5:
        h_hi, w_hi = 5, 6
    elif n == 4:
        h_hi, w_hi = 5, 7
    else:
        h_hi, w_hi = RECT_H_RANGE[1], RECT_W_RANGE[1]
    sizes: list[tuple[int, int]] = []
    for _ in range(n):
        height = rng.randint(RECT_H_RANGE[0], h_hi)
        width = rng.randint(RECT_W_RANGE[0], w_hi)
        sizes.append((height, width))
    return sizes


def _place_regions(
    rng: random.Random, shape: tuple[int, int], sizes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        rh, rw = sizes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(2, h - rh + 1):
            for c0 in range(0, w - rw + 1):
                r1 = r0 + rh - 1
                c1 = c0 + rw - 1
                rr0 = max(1, r0 - 1)
                rr1 = min(h - 1, r1 + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w - 1, c1 + 1)
                if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        positions[idx] = (r0, c0)
        blocked[r0 : r0 + rh, c0 : c0 + rw] = True
    return [pos for pos in positions if pos is not None]


def _draw_region(
    grid: np.ndarray, top: int, left: int, height: int, width: int, border: int, fill: int
) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top : bottom + 1, left] = border
    grid[top : bottom + 1, right] = border
    grid[top, left : right + 1] = border
    grid[bottom, left : right + 1] = border
    grid[top + 1 : bottom, left + 1 : right] = fill


def _expected_output(
    grid: np.ndarray,
    chain: list[int],
    placements: list[tuple[int, int]],
    sizes: list[tuple[int, int]],
    fills: list[int],
) -> np.ndarray:
    final_fills = fills[:]
    for i in range(len(chain) - 1):
        if final_fills[i] != BG and final_fills[i + 1] == BG:
            final_fills[i + 1] = final_fills[i]
    out = grid.copy()
    for (top, left), (height, width), fill in zip(placements, sizes, final_fills):
        out[top + 1 : top + height - 1, left + 1 : left + width - 1] = fill
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(*REGION_COUNT_RANGE)
    h = rng.randint(max(GRID_H_RANGE[0], n + 6), GRID_H_RANGE[1])
    w = rng.randint(max(GRID_W_RANGE[0], n + 6), GRID_W_RANGE[1])

    sizes = _sample_sizes(rng, n)
    placements = _place_regions(rng, (h, w), sizes)
    if placements is None:
        return None

    border_colors = rng.sample(list(range(1, 10)), n)
    chain = border_colors[:]
    rng.shuffle(chain)

    fill_mask = _sample_fill_mask(rng, n)
    fill_palette = [color for color in range(1, 10) if color not in chain]
    rng.shuffle(fill_palette)
    fills = [fill_palette.pop() if bit else BG for bit in fill_mask]

    grid = np.zeros((h, w), dtype=int)
    chain_start = rng.randint(0, w - n)
    grid[0, chain_start : chain_start + n] = np.asarray(chain, dtype=int)

    size_by_color = {color: size for color, size in zip(chain, sizes)}
    pos_by_color = {color: pos for color, pos in zip(chain, placements)}
    fill_by_color = {color: fill for color, fill in zip(chain, fills)}
    ordered_sizes = [size_by_color[color] for color in chain]
    ordered_positions = [pos_by_color[color] for color in chain]

    for color in chain:
        top, left = pos_by_color[color]
        height, width = size_by_color[color]
        _draw_region(grid, top, left, height, width, color, fill_by_color[color])

    expected = _expected_output(grid, chain, ordered_positions, ordered_sizes, fills)
    solved = main(grid)
    if np.array_equal(grid, solved):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
