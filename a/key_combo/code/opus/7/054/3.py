from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
MOTIF_COLORS = (1, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _count_full_green_squares(grid: np.ndarray) -> int:
    count = 0
    for comp in _components(grid == GREEN):
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) == 4 and r1 - r0 == 1 and c1 - c0 == 1:
            count += 1
    return count


def _first_occupied_band(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0
    start = int(indices[0])
    end = start
    for value in indices[1:]:
        value = int(value)
        if value == end + 1:
            end = value
        else:
            break
    return start, end


def _extract_motif(grid: np.ndarray) -> np.ndarray:
    occupied = grid != BG
    rows = np.flatnonzero(np.any(occupied, axis=1))
    cols = np.flatnonzero(np.any(occupied, axis=0))
    assert rows.size >= 3 and cols.size >= 3, "expected a visible source motif"

    r0, r1 = _first_occupied_band(rows)
    c0, c1 = _first_occupied_band(cols)
    assert r1 - r0 == 2 and c1 - c0 == 2, "source motif must occupy a 3x3 bbox"

    next_rows = rows[rows > r1]
    next_cols = cols[cols > c1]
    if next_rows.size:
        assert int(next_rows[0]) >= r1 + 3, "motif must be separated from the other objects"
    if next_cols.size:
        assert int(next_cols[0]) >= c1 + 3, "motif must be separated from the other objects"

    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count complete green squares to choose a motif scaling factor and odd-parity horizontal reflection.

    Input structure:
    - Black is the background.
    - A 3x3 colored source motif sits in the upper-left region and may contain black holes.
    - Elsewhere the grid contains green foreground components and red distractor components separated from the motif.

    Rule:
    - Count only the green 4-connected components whose bounding box is exactly 2x2 and whose area is 4.
    - Crop the upper-left 3x3 motif including its internal black cells.
    - Scale the motif by that count using nearest-neighbor pixel replication.
    - If the count is odd, horizontally flip the scaled motif; if it is even, keep it as-is.

    Key insight:
    - The same filtered square count controls both the output size and whether the enlarged motif must be mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    motif = _extract_motif(g)
    n = _count_full_green_squares(g)
    assert 1 <= n <= 3, "the number of complete green squares must be between 1 and 3"

    out = np.repeat(np.repeat(motif, n, axis=0), n, axis=1)
    if n % 2 == 1:
        out = np.fliplr(out)
    return out


def _grow_connected_mask_without_center(size: int) -> np.ndarray | None:
    allowed = {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}
    cells = {random.choice(tuple(allowed))}
    while len(cells) < size:
        frontier = {
            (nr, nc)
            for r, c in cells
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) in allowed and (nr, nc) not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if rows != {0, 1, 2} or cols != {0, 1, 2}:
        return None

    mask = np.zeros((3, 3), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if np.array_equal(mask, np.fliplr(mask)):
        return None
    return mask


def _make_motif() -> np.ndarray | None:
    for _ in range(80):
        mask = _grow_connected_mask_without_center(random.randint(5, 8))
        if mask is None:
            continue

        motif = np.zeros((3, 3), dtype=int)
        n_colors = random.randint(2, 3)
        colors = random.sample(list(MOTIF_COLORS), n_colors)
        filled = [(r, c) for r in range(3) for c in range(3) if mask[r, c]]
        random.shuffle(filled)

        assignments = colors[:]
        while len(assignments) < len(filled):
            assignments.append(random.choice(colors))
        random.shuffle(assignments)

        for (r, c), color in zip(filled, assignments, strict=True):
            motif[r, c] = int(color)

        if np.array_equal(motif, np.fliplr(motif)):
            continue
        return motif
    return None


def _rotations(patch: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen: set[bytes] = set()
    for k in range(4):
        rotated = np.rot90(patch, k)
        key = rotated.tobytes()
        if key not in seen:
            seen.add(key)
            variants.append(rotated.copy())
    return variants


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    pad: int,
    row_min: int = 0,
    col_min: int = 0,
) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(row_min, H - h + 1):
        for c0 in range(col_min, W - w + 1):
            if np.any(blocked[r0 : r0 + h, c0 : c0 + w]):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    mask = patch != BG
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    _mark_blocked(blocked, r0, c0, h, w, pad)
    return True


def _green_square() -> np.ndarray:
    return np.full((2, 2), GREEN, dtype=int)


def _red_distractor() -> np.ndarray:
    base_options = [
        np.array([[RED]], dtype=int),
        np.array([[RED, RED]], dtype=int),
        np.array([[RED], [RED]], dtype=int),
        np.array([[RED, RED], [RED, BG]], dtype=int),
        np.array([[RED, RED], [BG, RED]], dtype=int),
        np.array([[RED, BG], [RED, RED]], dtype=int),
        np.array([[BG, RED], [RED, RED]], dtype=int),
    ]
    return random.choice(base_options).copy()


def _green_distractor() -> np.ndarray:
    base_options = [
        np.array([[GREEN]], dtype=int),
        np.array([[GREEN, GREEN, GREEN]], dtype=int),
        np.array([[GREEN], [GREEN], [GREEN]], dtype=int),
        np.array([[GREEN, GREEN], [GREEN, BG]], dtype=int),
        np.array([[GREEN, GREEN], [BG, GREEN]], dtype=int),
    ]
    return random.choice(base_options).copy()


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 18)
    width = random.randint(14, 18)
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    motif = _make_motif()
    if motif is None:
        return None
    motif_r = random.randint(0, 2)
    motif_c = random.randint(0, 2)
    grid[motif_r : motif_r + 3, motif_c : motif_c + 3] = motif
    _mark_blocked(blocked, motif_r, motif_c, 3, 3, pad=2)

    n_squares = random.randint(1, 3)
    for _ in range(n_squares):
        if not _place_patch(grid, blocked, _green_square(), pad=1, row_min=motif_r + 5, col_min=motif_c + 5):
            return None

    red_count = random.randint(1, 4)
    for _ in range(red_count):
        patch = random.choice(_rotations(_red_distractor()))
        if not _place_patch(grid, blocked, patch, pad=1, row_min=motif_r + 5, col_min=motif_c + 5):
            return None

    for _ in range(random.randint(0, 2)):
        patch = random.choice(_rotations(_green_distractor()))
        if not _place_patch(grid, blocked, patch, pad=1, row_min=motif_r + 5, col_min=motif_c + 5):
            return None

    out = main(grid)
    expected = np.repeat(np.repeat(motif, n_squares, axis=0), n_squares, axis=1)
    if n_squares % 2 == 1:
        expected = np.fliplr(expected)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
