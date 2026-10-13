from __future__ import annotations

import numpy as np


BACKGROUND = 0
TOP_COLOR = 4
BOTTOM_COLOR = 1
SEPARATOR_COLOR = 5
MARKER_COLOR = 2
OUTPUT_COLOR = 3

GENERATOR_RETRIES = 500
COMPONENT_RETRIES = 80
_GEN_RNG = np.random.default_rng()


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h):
        row = grid[r]
        marker_cols = np.where(row == MARKER_COLOR)[0]
        if marker_cols.size != 1:
            continue
        if np.count_nonzero(row == SEPARATOR_COLOR) != w - 1:
            continue
        if np.all((row == SEPARATOR_COLOR) | (row == MARKER_COLOR)):
            candidates.append((r, int(marker_cols[0])))
    assert len(candidates) == 1, "expected exactly one marked separator row"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a shift amount from the separator row, roll the lower mask, and keep only shared background.

    Input structure:
    - The grid uses black as background.
    - A single full-width separator row sits between two equal-sized masks.
    - The separator row is grey except for one red marker whose column encodes a shift amount.
    - The upper mask has yellow foreground pixels; the lower mask has blue foreground pixels.

    Rule:
    - Find the unique separator row and read the red marker column index s.
    - Split the grid into the upper mask and lower mask around that row.
    - Cyclically shift the lower mask right by s columns.
    - Remove the separator row and compare the two masks cell by cell.
    - Output a grid the size of one mask, coloring a cell green only when both aligned cells are background; otherwise color it black.

    Key insight:
    - The separator row has a double role: it is both the physical boundary between masks and the control strip whose marker position encodes the lower mask's horizontal roll.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row, shift = _find_separator(g)
    upper = g[:separator_row, :]
    lower = g[separator_row + 1 :, :]
    assert upper.size > 0 and lower.size > 0, "separator must split two non-empty masks"
    assert upper.shape == lower.shape, "upper and lower masks must have the same shape"

    shifted_lower = np.roll(lower, shift=shift, axis=1)
    out = np.full(upper.shape, BACKGROUND, dtype=int)
    out[(upper == BACKGROUND) & (shifted_lower == BACKGROUND)] = OUTPUT_COLOR
    return out


def _sample_partition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    assert parts >= 1 and total >= parts
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _pick_weighted(cells: list[tuple[int, int]], scores: list[int], rng: np.random.Generator) -> tuple[int, int]:
    weights = np.asarray(scores, dtype=float)
    weights /= weights.sum()
    index = int(rng.choice(len(cells), p=weights))
    return cells[index]


def _grow_component(
    shape: tuple[int, int],
    size: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
    style: str,
) -> set[tuple[int, int]] | None:
    h, w = shape
    free_cells = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    if len(free_cells) < size:
        return None

    for _ in range(COMPONENT_RETRIES):
        seed = free_cells[int(rng.integers(len(free_cells)))]
        comp = {seed}
        while len(comp) < size:
            frontier: list[tuple[int, int]] = []
            scores: list[int] = []
            for r, c in comp:
                for nr, nc in _neighbors4(shape, r, c):
                    if blocked[nr, nc] or (nr, nc) in comp:
                        continue
                    if (nr, nc) in frontier:
                        continue
                    frontier.append((nr, nc))
                    adjacency = sum((rr, cc) in comp for rr, cc in _neighbors4(shape, nr, nc))
                    if style == "blob":
                        scores.append(adjacency + 2)
                    elif style == "path":
                        scores.append(5 - adjacency)
                    else:
                        scores.append(2 if adjacency == 1 else 3)
            if not frontier:
                break
            comp.add(_pick_weighted(frontier, scores, rng))
        if len(comp) == size:
            return comp
    return None


def _rectangle_component(
    shape: tuple[int, int],
    size: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    h, w = shape
    if size <= 0:
        return None
    for _ in range(COMPONENT_RETRIES):
        max_rh = min(h, size)
        rh = int(rng.integers(1, max_rh + 1))
        max_rw = min(w, max(1, size // rh))
        rw = int(rng.integers(1, max_rw + 1))
        area = rh * rw
        if area < max(1, size // 2):
            continue
        r0 = int(rng.integers(0, h - rh + 1))
        c0 = int(rng.integers(0, w - rw + 1))
        if np.any(blocked[r0 : r0 + rh, c0 : c0 + rw]):
            continue
        comp = {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}
        if area == size:
            return comp
        extra = _grow_component(shape, size - area, blocked | _mask_from_cells(shape, comp), rng, "mixed")
        if extra is None:
            continue
        return comp | extra
    return None


def _mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_mask(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray | None:
    h, w = shape
    area = h * w
    target = int(rng.integers(max(3, area // 5), min(area - 1, (2 * area) // 3) + 1))
    n_components = int(rng.integers(1, min(3, target) + 1))
    while target < n_components:
        n_components -= 1
    sizes = _sample_partition(target, n_components, rng)

    mask = np.zeros(shape, dtype=bool)
    for size in sizes:
        styles = ["blob", "path", "rect"]
        rng.shuffle(styles)
        placed = None
        for style in styles:
            if style == "rect":
                placed = _rectangle_component(shape, size, mask, rng)
            else:
                placed = _grow_component(shape, size, mask, rng, style)
            if placed is not None:
                break
        if placed is None:
            return None
        for r, c in placed:
            mask[r, c] = True
    return mask


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.full(mask.shape, BACKGROUND, dtype=int)
    out[mask] = color
    return out


def _accept_mask_pair(top_mask: np.ndarray, shifted_bottom_mask: np.ndarray, shift: int) -> bool:
    lower_input_mask = np.roll(shifted_bottom_mask, shift=-shift, axis=1)
    shifted_green = (~top_mask) & (~shifted_bottom_mask)
    unshifted_green = (~top_mask) & (~lower_input_mask)

    area = int(top_mask.size)
    green_count = int(np.count_nonzero(shifted_green))
    diff_count = int(np.count_nonzero(shifted_green != unshifted_green))

    if green_count == 0 or green_count == area:
        return False
    if green_count < max(1, area // 8):
        return False
    if green_count > area - max(2, area // 6):
        return False
    if diff_count < max(2, top_mask.shape[1] // 2):
        return False
    return True


def _build_input(shape: tuple[int, int], shift: int, rng: np.random.Generator) -> np.ndarray | None:
    top_mask = _make_mask(shape, rng)
    shifted_bottom_mask = _make_mask(shape, rng)
    if top_mask is None or shifted_bottom_mask is None:
        return None
    if not _accept_mask_pair(top_mask, shifted_bottom_mask, shift):
        return None

    lower_input_mask = np.roll(shifted_bottom_mask, shift=-shift, axis=1)
    top = _render_mask(top_mask, TOP_COLOR)
    bottom = _render_mask(lower_input_mask, BOTTOM_COLOR)
    separator = np.full((1, shape[1]), SEPARATOR_COLOR, dtype=int)
    separator[0, shift] = MARKER_COLOR
    grid = np.vstack([top, separator, bottom])

    expected = np.full(shape, BACKGROUND, dtype=int)
    expected[(~top_mask) & (~shifted_bottom_mask)] = OUTPUT_COLOR
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        mask_height = int(_GEN_RNG.integers(3, 11))
        mask_width = int(_GEN_RNG.integers(4, 13))
        shift = int(_GEN_RNG.integers(1, mask_width))
        grid = _build_input((mask_height, mask_width), shift, _GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
