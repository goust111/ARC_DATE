import random
from collections import Counter

import numpy as np


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _full_line_colors(grid: np.ndarray, axis: int) -> list[int]:
    out: list[int] = []
    n = grid.shape[axis]
    for idx in range(n):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        out.append(int(line[0]) if np.all(line == line[0]) else -1)
    return out


def _find_divider(grid: np.ndarray) -> int:
    row_colors = {color for color in _full_line_colors(grid, axis=0) if color >= 0}
    col_colors = {color for color in _full_line_colors(grid, axis=1) if color >= 0}
    common = row_colors & col_colors
    assert len(common) == 1, "expected exactly one divider color"
    return next(iter(common))


def _partition_blocks(grid: np.ndarray, divider: int):
    row_runs = _runs([r for r in range(grid.shape[0]) if not np.all(grid[r, :] == divider)])
    col_runs = _runs([c for c in range(grid.shape[1]) if not np.all(grid[:, c] == divider)])
    assert row_runs and col_runs, "expected a divider lattice"
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for br, (r0, r1) in enumerate(row_runs):
        for bc, (c0, c1) in enumerate(col_runs):
            blocks[(br, bc)] = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    return row_runs, col_runs, blocks


def _unique_mode(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    order = np.argsort(-counts)
    colors = colors[order]
    counts = counts[order]
    assert len(colors) == 1 or counts[0] > counts[1], "expected a unique majority color"
    return int(colors[0])


def _symmetries(grid: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("id", grid.copy()),
        ("rot90", np.rot90(grid, 1).copy()),
        ("rot180", np.rot90(grid, 2).copy()),
        ("rot270", np.rot90(grid, 3).copy()),
        ("flip_lr", np.fliplr(grid).copy()),
        ("flip_ud", np.flipud(grid).copy()),
        ("transpose", grid.T.copy()),
        ("anti_transpose", np.flipud(np.fliplr(grid)).T.copy()),
    ]


def _find_thumbnail_block(blocks: dict[tuple[int, int], np.ndarray]) -> tuple[int, int]:
    candidates = [pos for pos, block in blocks.items() if len(np.unique(block)) > 2]
    assert len(candidates) == 1, "expected exactly one multicolor thumbnail block"
    return candidates[0]


def _infer_grey(blocks: dict[tuple[int, int], np.ndarray], thumb_pos: tuple[int, int]) -> int:
    majority_colors: set[int] = set()
    minority_colors: Counter[int] = Counter()
    for pos, block in blocks.items():
        if pos == thumb_pos:
            continue
        colors, counts = np.unique(block, return_counts=True)
        order = np.argsort(-counts)
        colors = colors[order]
        counts = counts[order]
        assert len(colors) <= 2, "non-thumbnail blocks should contain only paint and grey noise"
        majority_colors.add(int(colors[0]))
        if len(colors) == 2:
            assert counts[0] > counts[1], "non-thumbnail blocks need a strict majority color"
            minority_colors[int(colors[1])] += 1
    candidates = [color for color, count in minority_colors.items() if color not in majority_colors and count >= 2]
    assert len(candidates) == 1, "expected a unique grey noise color"
    return candidates[0]


def _coarse_map(blocks: dict[tuple[int, int], np.ndarray], shape: tuple[int, int], thumb_pos: tuple[int, int], grey: int) -> np.ndarray:
    out = np.full(shape, -1, dtype=int)
    for pos, block in blocks.items():
        if pos == thumb_pos:
            continue
        data = block[block != grey]
        assert data.size > 0, "pure blocks must contain at least one non-grey cell"
        out[pos] = _unique_mode(data)
    return out


def _recover_layout(thumb: np.ndarray, coarse: np.ndarray, thumb_pos: tuple[int, int]) -> np.ndarray:
    mask = np.ones(coarse.shape, dtype=bool)
    mask[thumb_pos] = False
    matches: list[np.ndarray] = []
    for _name, variant in _symmetries(thumb):
        if variant.shape != coarse.shape:
            continue
        if np.array_equal(variant[mask], coarse[mask]):
            matches.append(variant)
    assert len(matches) == 1, "expected a unique pose consistent with the other blocks"
    return matches[0]


def _render_layout(row_runs, col_runs, divider: int, coarse: np.ndarray) -> np.ndarray:
    height = row_runs[-1][1] + 2
    width = col_runs[-1][1] + 2
    out = np.full((height, width), divider, dtype=int)
    for br, (r0, r1) in enumerate(row_runs):
        for bc, (c0, c1) in enumerate(col_runs):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(coarse[br, bc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate or mirror the embedded self-thumbnail back into place, then repaint every block with its declared solid color.

    Input structure:
    - Complete separator rows and columns of one divider color split the grid into a rectangular block lattice.
    - All but one blocks are nearly solid: one real block color fills the block, with a few grey noise cells scattered inside.
    - The remaining block is a multicolor thumbnail of the whole block layout in some rotated or mirrored pose.

    Rule:
    - Detect the divider lattice and locate the unique multicolor thumbnail block.
    - Infer the grey noise color from the minority speckles in the ordinary blocks, then take each ordinary block's non-grey majority color.
    - Try all eight rotations and reflections of the thumbnail; keep the unique pose whose coarse pixels agree with every known block position except the thumbnail's own location.
    - Rebuild the full grid by filling every block rectangle with the recovered thumbnail color for that lattice position, while keeping all divider lines unchanged.

    Key insight:
    - The anomalous block is not damage; it is a self-map of the entire lattice, and the correct pose is the only one that is globally consistent once the noisy solid blocks are collapsed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider(g)
    row_runs, col_runs, blocks = _partition_blocks(g, divider)
    thumb_pos = _find_thumbnail_block(blocks)
    grey = _infer_grey(blocks, thumb_pos)
    coarse = _coarse_map(blocks, (len(row_runs), len(col_runs)), thumb_pos, grey)
    layout = _recover_layout(blocks[thumb_pos], coarse, thumb_pos)
    return _render_layout(row_runs, col_runs, divider, layout)


SYMMETRY_NAMES = [name for name, _grid in _symmetries(np.arange(4).reshape(2, 2))]
GEN_ATTEMPTS = 500


def _apply_named_symmetry(grid: np.ndarray, name: str) -> np.ndarray:
    for sym_name, variant in _symmetries(grid):
        if sym_name == name:
            return variant
    raise ValueError(f"unknown symmetry: {name}")


def _sample_layout(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    n_cells = rows * cols
    n_colors = random.randint(3, min(5, len(palette), n_cells))
    chosen = random.sample(palette, n_colors)
    values = chosen[:] + random.choices(chosen, k=n_cells - n_colors)
    random.shuffle(values)
    layout = np.array(values, dtype=int).reshape(rows, cols)
    assert len(set(map(int, layout.flat))) >= 3
    return layout


def _layout_unique_for_thumb(layout: np.ndarray, shown_thumb: np.ndarray, thumb_pos: tuple[int, int]) -> bool:
    mask = np.ones(layout.shape, dtype=bool)
    mask[thumb_pos] = False
    matches = 0
    for _name, variant in _symmetries(shown_thumb):
        if variant.shape != layout.shape:
            continue
        if np.array_equal(variant[mask], layout[mask]):
            matches += 1
    return matches == 1


def _paint_noisy_block(color: int, grey: int, shape: tuple[int, int], noise_count: int) -> np.ndarray:
    block = np.full(shape, color, dtype=int)
    if noise_count <= 0:
        return block
    cells = random.sample(range(shape[0] * shape[1]), noise_count)
    for cell in cells:
        block[cell // shape[1], cell % shape[1]] = grey
    return block


def _render_input(layout: np.ndarray, shown_thumb: np.ndarray, thumb_pos: tuple[int, int], divider: int, grey: int) -> np.ndarray:
    rows, cols = layout.shape
    block_h, block_w = shown_thumb.shape
    grid = np.full((rows * block_h + rows + 1, cols * block_w + cols + 1), divider, dtype=int)
    pure_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != thumb_pos]
    area = block_h * block_w
    max_noise = min(3, (area - 1) // 2)
    assert max_noise >= 1, "blocks must be large enough for visible majority noise"

    noisy_count = min(len(pure_positions), 2 + random.randint(0, max(0, len(pure_positions) - 2)))
    noisy_positions = set(random.sample(pure_positions, noisy_count))

    for r in range(rows):
        for c in range(cols):
            r0 = 1 + r * (block_h + 1)
            c0 = 1 + c * (block_w + 1)
            if (r, c) == thumb_pos:
                grid[r0 : r0 + block_h, c0 : c0 + block_w] = shown_thumb
                continue
            noise_count = random.randint(1, max_noise) if (r, c) in noisy_positions else 0
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = _paint_noisy_block(
                int(layout[r, c]), grey, (block_h, block_w), noise_count
            )
    return grid


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    all_colors = list(range(10))
    divider, grey = random.sample(all_colors, 2)
    palette = [color for color in all_colors if color not in {divider, grey}]
    layout = _sample_layout(rows, cols, palette)
    thumb_pos = (random.randrange(rows), random.randrange(cols))
    pose = random.choice(SYMMETRY_NAMES)
    shown_thumb = _apply_named_symmetry(layout, pose)
    if max(shown_thumb.shape) < 2:
        return None
    height = rows * shown_thumb.shape[0] + rows + 1
    width = cols * shown_thumb.shape[1] + cols + 1
    if height > 30 or width > 30:
        return None
    if len(np.unique(shown_thumb)) < 3:
        return None
    if not _layout_unique_for_thumb(layout, shown_thumb, thumb_pos):
        return None

    grid = _render_input(layout, shown_thumb, thumb_pos, divider, grey)
    expected = _render_layout(
        [(1 + r * (shown_thumb.shape[0] + 1), r * (shown_thumb.shape[0] + 1) + shown_thumb.shape[0]) for r in range(rows)],
        [(1 + c * (shown_thumb.shape[1] + 1), c * (shown_thumb.shape[1] + 1) + shown_thumb.shape[1]) for c in range(cols)],
        divider,
        layout,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
