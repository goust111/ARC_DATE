from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GRAY = 5
MAGENTA = 6
SAFE_COLORS = (1, 2, 3, 4, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
MOTIF_BUILD_ATTEMPTS = 120


def _bbox_non_black(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BLACK)
    assert coords.size, "expected at least one non-black cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _content_spans(limit: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    boundaries = [-1, *sorted(set(map(int, separators))), int(limit)]
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left <= 1:
            continue
        spans.append((int(left + 1), int(right)))
    assert spans, "expected at least one content span"
    return spans


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected non-empty values"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely corrupted tile copy and broadcast its error coordinates to every intact copy as magenta highlights.

    Input structure:
    - Black is the outer background, and grey full rows/columns split a repeated-lattice region into equal rectangular modules.
    - Every module is a copy of the same multicolor motif except for exactly one module, where some local pixels were recolored.
    - The damaged module stays inside the same separator lattice and has the same size as the intact modules.

    Rule:
    - Crop to the repeated-lattice region, read the module rows and columns from the grey separator lines, and extract every module patch.
    - Reconstruct the correct motif by taking the unique per-coordinate majority color across all module copies.
    - Find the single module that differs from that consensus motif and record the local coordinates where it disagrees.
    - Leave the damaged module unchanged, but recolor those same local coordinates to magenta in every other module; preserve separators and black margins.

    Key insight:
    - The lone bad module is identified by cross-copy majority voting, and its wrong local coordinates are not repaired locally; they are propagated as position markers onto every correct copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _bbox_non_black(g)
    core = g[r0 : r1 + 1, c0 : c1 + 1]

    sep_rows = [r for r in range(core.shape[0]) if np.all(core[r, :] == GRAY)]
    sep_cols = [c for c in range(core.shape[1]) if np.all(core[:, c] == GRAY)]
    row_spans = _content_spans(core.shape[0], sep_rows)
    col_spans = _content_spans(core.shape[1], sep_cols)

    heights = {r1_ - r0_ for r0_, r1_ in row_spans}
    widths = {c1_ - c0_ for c0_, c1_ in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "modules must all have the same size"

    module_h = heights.pop()
    module_w = widths.pop()

    modules: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for rr0, rr1 in row_spans:
        for cc0, cc1 in col_spans:
            patch = core[rr0:rr1, cc0:cc1]
            assert patch.shape == (module_h, module_w)
            modules.append(patch.copy())
            origins.append((int(rr0), int(cc0)))

    assert len(modules) >= 3, "expected at least three repeated modules for majority voting"

    blocks = np.stack(modules, axis=0)
    motif = np.empty((module_h, module_w), dtype=int)
    for rr in range(module_h):
        for cc in range(module_w):
            motif[rr, cc] = _unique_mode(blocks[:, rr, cc])

    diff_masks = blocks != motif
    diff_counts = diff_masks.reshape(len(modules), module_h * module_w).sum(axis=1)
    damaged = np.flatnonzero(diff_counts > 0)
    assert damaged.size == 1, "expected exactly one modified module"

    damaged_idx = int(damaged[0])
    changed_mask = diff_masks[damaged_idx]
    assert changed_mask.any(), "damaged module must differ from the consensus motif"
    assert np.all(diff_counts[np.arange(len(modules)) != damaged_idx] == 0), "all other modules must match the consensus"

    out = g.copy()
    for idx, (rr0, cc0) in enumerate(origins):
        if idx == damaged_idx:
            continue
        patch = out[r0 + rr0 : r0 + rr0 + module_h, c0 + cc0 : c0 + cc0 + module_w]
        patch[changed_mask] = MAGENTA

    return out


def _random_shape_cells(height: int, width: int) -> list[tuple[int, int]]:
    kind = random.choice(("rect", "hline", "vline", "dot", "plus"))
    if kind == "rect":
        h = random.randint(1, height)
        w = random.randint(1, width)
        r0 = random.randint(0, height - h)
        c0 = random.randint(0, width - w)
        return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    if kind == "hline":
        r = random.randrange(height)
        c0 = random.randrange(width)
        c1 = random.randint(c0, width - 1)
        return [(r, c) for c in range(c0, c1 + 1)]
    if kind == "vline":
        c = random.randrange(width)
        r0 = random.randrange(height)
        r1 = random.randint(r0, height - 1)
        return [(r, c) for r in range(r0, r1 + 1)]
    if kind == "plus":
        r = random.randrange(height)
        c = random.randrange(width)
        cells = {(r, c)}
        if r > 0:
            cells.add((r - 1, c))
        if r + 1 < height:
            cells.add((r + 1, c))
        if c > 0:
            cells.add((r, c - 1))
        if c + 1 < width:
            cells.add((r, c + 1))
        return sorted(cells)
    return [(random.randrange(height), random.randrange(width))]


def _build_motif(height: int, width: int) -> np.ndarray | None:
    for _ in range(MOTIF_BUILD_ATTEMPTS):
        accent_count = random.randint(2, min(4, len(SAFE_COLORS)))
        accent_colors = random.sample(SAFE_COLORS, accent_count)
        base = random.choice([BLACK, accent_colors[0], accent_colors[1]])
        motif = np.full((height, width), base, dtype=int)

        for _ in range(random.randint(4, 8)):
            color = random.choice(accent_colors)
            for r, c in _random_shape_cells(height, width):
                motif[r, c] = color

        if base == BLACK:
            for r in range(height):
                if np.all(motif[r, :] == BLACK):
                    c = random.randrange(width)
                    motif[r, c] = random.choice(accent_colors)
            for c in range(width):
                if np.all(motif[:, c] == BLACK):
                    r = random.randrange(height)
                    motif[r, c] = random.choice(accent_colors)

        colors = set(map(int, np.unique(motif)))
        non_black = colors - {BLACK}
        if len(non_black) < 2:
            continue
        if base == BLACK and (
            any(np.all(motif[r, :] == BLACK) for r in range(height))
            or any(np.all(motif[:, c] == BLACK) for c in range(width))
        ):
            continue
        return motif
    return None


def _build_structure(
    motif: np.ndarray,
    block_rows: int,
    block_cols: int,
    outer_border: bool,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    tile_h, tile_w = map(int, motif.shape)
    border = 1 if outer_border else 0
    struct_h = block_rows * tile_h + max(0, block_rows - 1) + 2 * border
    struct_w = block_cols * tile_w + max(0, block_cols - 1) + 2 * border

    structure = np.full((struct_h, struct_w), BLACK, dtype=int)
    origins: list[tuple[int, int]] = []

    if outer_border:
        structure[0, :] = GRAY
        structure[-1, :] = GRAY
        structure[:, 0] = GRAY
        structure[:, -1] = GRAY

    row = border
    for br in range(block_rows):
        col = border
        for bc in range(block_cols):
            structure[row : row + tile_h, col : col + tile_w] = motif
            origins.append((int(row), int(col)))
            col += tile_w
            if bc != block_cols - 1:
                structure[:, col] = GRAY
                col += 1
        row += tile_h
        if br != block_rows - 1:
            structure[row, :] = GRAY
            row += 1

    return structure, origins


def _expected_output(
    inp: np.ndarray,
    structure_top: int,
    structure_left: int,
    origins: list[tuple[int, int]],
    motif_shape: tuple[int, int],
    damaged_index: int,
    changed_cells: list[tuple[int, int]],
) -> np.ndarray:
    tile_h, tile_w = map(int, motif_shape)
    out = inp.copy()
    for idx, (r0, c0) in enumerate(origins):
        if idx == damaged_index:
            continue
        patch = out[
            structure_top + r0 : structure_top + r0 + tile_h,
            structure_left + c0 : structure_left + c0 + tile_w,
        ]
        for rr, cc in changed_cells:
            patch[rr, cc] = MAGENTA
    return out


def _gen_attempt() -> np.ndarray | None:
    block_rows, block_cols = random.choice(
        [
            (1, 3),
            (1, 4),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
            (4, 1),
        ]
    )
    tile_h = random.randint(3, 7)
    tile_w = random.randint(3, 7)
    outer_border = random.choice((False, True))
    margins = [0, 0, 0, 0]
    if outer_border:
        margins = [random.randint(0, 2) for _ in range(4)]

    motif = _build_motif(tile_h, tile_w)
    if motif is None:
        return None

    structure, origins = _build_structure(motif, block_rows, block_cols, outer_border)
    total_h = structure.shape[0] + margins[0] + margins[1]
    total_w = structure.shape[1] + margins[2] + margins[3]
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    top, left = margins[0], margins[2]
    grid[top : top + structure.shape[0], left : left + structure.shape[1]] = structure

    damage_cap = min(6, max(1, (tile_h * tile_w) // 3))
    damage_count = random.randint(1, damage_cap)
    changed_cells = random.sample(
        [(r, c) for r in range(tile_h) for c in range(tile_w)],
        damage_count,
    )
    damaged_index = random.randrange(len(origins))
    damage_palette = [BLACK, *SAFE_COLORS]

    dr0, dc0 = origins[damaged_index]
    for rr, cc in changed_cells:
        original = int(grid[top + dr0 + rr, left + dc0 + cc])
        choices = [color for color in damage_palette if color != original]
        grid[top + dr0 + rr, left + dc0 + cc] = random.choice(choices)

    expected = _expected_output(
        grid,
        top,
        left,
        origins,
        motif.shape,
        damaged_index,
        changed_cells,
    )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
