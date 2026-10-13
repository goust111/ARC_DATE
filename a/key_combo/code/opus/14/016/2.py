from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEAL = 8

TEMPLATE_MASKS: dict[str, np.ndarray] = {
    "rect_h": np.array([[1, 1, 1]], dtype=bool),
    "rect_v": np.array([[1], [1], [1]], dtype=bool),
    "rr_el": np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    "sr_compact": np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    "sr_long": np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    "ss_domino": np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
}

TEMPLATE_SQUARES = {
    "rect_h": 0,
    "rect_v": 0,
    "rr_el": 0,
    "sr_compact": 1,
    "sr_long": 1,
    "ss_domino": 2,
}

GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
            out.append(sorted(comp))
    return out


def _normalize_cells(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return r0, c0, norm


@lru_cache(maxsize=None)
def _solve_normalized_component(norm_cells: tuple[tuple[int, int], ...]) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    cell_set = set(norm_cells)
    index = {cell: i for i, cell in enumerate(norm_cells)}
    placements: list[tuple[str, tuple[tuple[int, int], ...], int]] = []

    for r, c in norm_cells:
        square = ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
        if all(cell in cell_set for cell in square):
            bits = 0
            for cell in square:
                bits |= 1 << index[cell]
            placements.append(("square", square, bits))

        horiz = ((r, c), (r, c + 1), (r, c + 2))
        if all(cell in cell_set for cell in horiz):
            bits = 0
            for cell in horiz:
                bits |= 1 << index[cell]
            placements.append(("rect", horiz, bits))

        vert = ((r, c), (r + 1, c), (r + 2, c))
        if all(cell in cell_set for cell in vert):
            bits = 0
            for cell in vert:
                bits |= 1 << index[cell]
            placements.append(("rect", vert, bits))

    by_cell: list[list[int]] = [[] for _ in norm_cells]
    for pid, (_kind, cells, _bits) in enumerate(placements):
        for cell in cells:
            by_cell[index[cell]].append(pid)

    full = (1 << len(norm_cells)) - 1
    solutions: list[tuple[int, ...]] = []

    def search(covered: int, chosen: list[int]) -> None:
        if len(solutions) >= 2:
            return
        if covered == full:
            solutions.append(tuple(chosen))
            return
        remaining = full & ~covered
        bit = remaining & -remaining
        cell_idx = bit.bit_length() - 1
        for pid in by_cell[cell_idx]:
            bits = placements[pid][2]
            if bits & covered:
                continue
            chosen.append(pid)
            search(covered | bits, chosen)
            chosen.pop()

    search(0, [])
    if len(solutions) != 1:
        raise AssertionError("component does not admit a unique valid decomposition")

    return tuple((placements[pid][0], tuple(sorted(placements[pid][1]))) for pid in solutions[0])


def _decompose_component(cells: list[tuple[int, int]]) -> list[tuple[str, list[tuple[int, int]]]]:
    key = tuple(sorted(cells))
    r0, c0, norm = _normalize_cells(key)
    solved = _solve_normalized_component(norm)
    out: list[tuple[str, list[tuple[int, int]]]] = []
    for kind, piece_cells in solved:
        shifted = [(r + r0, c + c0) for r, c in piece_cells]
        out.append((kind, shifted))
    return out


def _quadrant_index(r: int, c: int, mid: int) -> int:
    return (0 if r < mid else 2) + (0 if c < mid else 1)


def _quadrant_slices(size: int, idx: int) -> tuple[slice, slice]:
    half = size // 2
    row = 0 if idx < 2 else 1
    col = idx % 2
    return slice(row * half, (row + 1) * half), slice(col * half, (col + 1) * half)


def _recolored_grid_and_counts(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] % 2 != 0:
        raise AssertionError("expected an even-sized square grid")
    if not set(np.unique(g)).issubset({BLACK, GREY}):
        raise AssertionError("input should only use black and gray")

    half = g.shape[0] // 2
    mask = g == GREY
    comps = _components(mask)
    out = np.full_like(g, BLACK)
    square_counts = [0, 0, 0, 0]

    for comp in comps:
        if not (3 <= len(comp) <= 12):
            raise AssertionError("each gray component must contain 3-12 cells")
        pieces = _decompose_component(comp)
        for kind, cells in pieces:
            q = _quadrant_index(cells[0][0], cells[0][1], half)
            if any(_quadrant_index(r, c, half) != q for r, c in cells):
                raise AssertionError("pieces must not cross quadrant boundaries")
            color = TEAL if kind == "square" else RED
            if kind == "square":
                square_counts[q] += 1
            for r, c in cells:
                out[r, c] = color
    return out, square_counts, comps


def _mirror_from_source(tile: np.ndarray, source_idx: int) -> np.ndarray:
    half = tile.shape[0]
    out = np.full((half * 2, half * 2), BLACK, dtype=int)
    src_row = 0 if source_idx < 2 else 1
    src_col = source_idx % 2
    for row in range(2):
        for col in range(2):
            if row == src_row and col == src_col:
                patch = tile
            elif row == src_row:
                patch = np.fliplr(tile)
            elif col == src_col:
                patch = np.flipud(tile)
            else:
                patch = np.rot90(tile, 2)
            rs = slice(row * half, (row + 1) * half)
            cs = slice(col * half, (col + 1) * half)
            out[rs, cs] = patch
    return out


def _naive_visible_square_counts(grid: np.ndarray, comps: list[list[tuple[int, int]]]) -> list[int]:
    half = grid.shape[0] // 2
    counts = [0, 0, 0, 0]
    for comp in comps:
        if len(comp) != 4:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        is_square = (r1 - r0 == 1) and (c1 - c0 == 1) and set(comp) == {
            (r0, c0),
            (r0, c1),
            (r1, c0),
            (r1, c1),
        }
        if is_square:
            counts[_quadrant_index(r0, c0, half)] += 1
    return counts


def _source_outputs(recolored: np.ndarray) -> list[np.ndarray]:
    outputs = []
    for idx in range(4):
        rs, cs = _quadrant_slices(recolored.shape[0], idx)
        outputs.append(_mirror_from_source(recolored[rs, cs], idx))
    return outputs


def _semantic_report(grid: np.ndarray) -> dict | None:
    try:
        recolored, counts, comps = _recolored_grid_and_counts(grid)
    except AssertionError:
        return None

    if sorted(counts) != [0, 1, 2, 3]:
        return None

    source_idx = counts.index(3)
    outputs = _source_outputs(recolored)
    expected = outputs[source_idx]

    if any(np.array_equal(expected, other) for idx, other in enumerate(outputs) if idx != source_idx):
        return None

    naive_counts = _naive_visible_square_counts(grid, comps)
    naive_max = max(naive_counts)
    naive_argmax = [idx for idx, value in enumerate(naive_counts) if value == naive_max]
    if naive_argmax == [source_idx]:
        return None

    half = grid.shape[0] // 2
    for idx in range(4):
        rs, cs = _quadrant_slices(grid.shape[0], idx)
        if not np.any(grid[rs, cs] == GREY):
            return None

    source_tile = recolored[_quadrant_slices(grid.shape[0], source_idx)]
    if not (np.any(source_tile == RED) and np.any(source_tile == TEAL)):
        return None
    if np.array_equal(source_tile, np.fliplr(source_tile)):
        return None
    if np.array_equal(source_tile, np.flipud(source_tile)):
        return None
    if np.array_equal(source_tile, np.rot90(source_tile, 2)):
        return None

    return {
        "recolored": recolored,
        "counts": counts,
        "source_idx": source_idx,
        "expected": expected,
        "naive_counts": naive_counts,
    }


def _template_variants(name: str) -> list[np.ndarray]:
    base = TEMPLATE_MASKS[name]
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        for flipped in (rotated, np.fliplr(rotated)):
            candidate = flipped.astype(bool, copy=True)
            if not any(np.array_equal(candidate, seen) for seen in variants):
                variants.append(candidate)
    return variants


TEMPLATE_VARIANTS = {name: _template_variants(name) for name in TEMPLATE_MASKS}


def _candidate_bundles(square_count: int, half: int) -> list[list[str]]:
    bundles = {
        0: [
            ["rr_el"],
            ["rr_el", random.choice(["rect_h", "rect_v"])],
            ["rect_h", "rect_v"],
        ],
        1: [
            ["sr_compact"],
            ["sr_long"],
            ["sr_compact", random.choice(["rect_h", "rect_v"])],
        ],
        2: [
            ["ss_domino"],
            ["ss_domino", random.choice(["rect_h", "rect_v"])],
        ],
        3: [
            ["ss_domino", "sr_compact"],
        ],
    }
    if half >= 7:
        bundles[3].append(["ss_domino", "sr_long"])
        bundles[2].append(["sr_compact", "sr_long"])
    return bundles[square_count]


def _can_place(mask: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    coords = np.argwhere(patch)
    for pr, pc in coords:
        r = int(r0 + pr)
        c = int(c0 + pc)
        r1 = max(0, r - padding)
        r2 = min(H, r + padding + 1)
        c1 = max(0, c - padding)
        c2 = min(W, c + padding + 1)
        if np.any(mask[r1:r2, c1:c2]):
            return False
    return True


def _place_patch(mask: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    region = mask[r0 : r0 + h, c0 : c0 + w]
    region[patch] = True


def _generate_quadrant(half: int, square_count: int) -> np.ndarray | None:
    bundle = random.choice(_candidate_bundles(square_count, half))
    transformed = [random.choice(TEMPLATE_VARIANTS[name]) for name in bundle]
    order = sorted(transformed, key=lambda arr: int(arr.sum()), reverse=True)
    mask = np.zeros((half, half), dtype=bool)

    for patch in order:
        h, w = patch.shape
        positions = [(r, c) for r in range(half - h + 1) for c in range(half - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _can_place(mask, patch, r0, c0, padding=1):
                _place_patch(mask, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    grid = np.full((half, half), BLACK, dtype=int)
    grid[mask] = GREY
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose gray regions into squares and length-three rectangles, recolor them, then mirror the quadrant with the most squares across both axes.

    Input structure:
    - The input is an even-sized square grid split implicitly into four equal quadrants by the central row gap and central column gap.
    - Black is background and gray foreground components stay inside a single quadrant.
    - Every gray component has a unique non-overlapping decomposition into 2x2 squares and 1x3 or 3x1 rectangles.

    Rule:
    - Uniquely decompose every gray component into the allowed primitives.
    - Recolor each 2x2 square teal and each length-three rectangle red.
    - Count the teal square primitives in each quadrant and choose the unique quadrant with the largest count as the source.
    - Keep the recolored source quadrant in place, mirror it left-right into the horizontal neighbor, up-down into the vertical neighbor, and rotate it 180 degrees into the diagonal quadrant.

    Key insight:
    - The source quadrant is determined only after primitive decomposition, so both the counting step and the final mirroring depend on the recovered recolored parts rather than on the original gray blobs.
    """
    recolored, counts, _comps = _recolored_grid_and_counts(grid)
    max_count = max(counts)
    if counts.count(max_count) != 1:
        raise AssertionError("expected a unique source quadrant")
    source_idx = counts.index(max_count)
    rs, cs = _quadrant_slices(recolored.shape[0], source_idx)
    return _mirror_from_source(recolored[rs, cs], source_idx)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        size = random.choice([12, 14])
        half = size // 2
        counts = [0, 1, 2, 3]
        random.shuffle(counts)

        quadrants: list[np.ndarray] = []
        ok = True
        for square_count in counts:
            tile = _generate_quadrant(half, square_count)
            if tile is None:
                ok = False
                break
            quadrants.append(tile)
        if not ok:
            continue

        grid = np.full((size, size), BLACK, dtype=int)
        for idx, tile in enumerate(quadrants):
            rs, cs = _quadrant_slices(size, idx)
            grid[rs, cs] = tile

        report = _semantic_report(grid)
        if report is None:
            continue

        actual = main(grid)
        if not np.array_equal(actual, report["expected"]):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
