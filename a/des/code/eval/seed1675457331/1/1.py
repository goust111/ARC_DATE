import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TYPES = ("h", "v", "rot", "hv")
GEN_ATTEMPTS = 400


def _classify_mask(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == (3, 3)
    assert bool(mask.any())

    has_h = np.array_equal(mask, mask[::-1, :])
    has_v = np.array_equal(mask, mask[:, ::-1])
    has_rot = np.array_equal(mask, mask[::-1, ::-1])

    if has_h and has_v:
        return "hv"
    if has_h and not has_v and not has_rot:
        return "h"
    if has_v and not has_h and not has_rot:
        return "v"
    if has_rot and not has_h and not has_v:
        return "rot"
    raise AssertionError("seed must match exactly one supported symmetry type")


def _build_mask_catalog() -> dict[str, list[np.ndarray]]:
    catalog = {kind: [] for kind in TYPES}
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3)
        area = int(mask.sum())
        if area < 2 or area > 7:
            continue
        # Keep the pattern visibly anchored in the upper-left 3x3 seed block.
        if not mask[0, :].any() or not mask[:, 0].any():
            continue
        try:
            kind = _classify_mask(mask)
        except AssertionError:
            continue
        catalog[kind].append(mask)

    for kind in TYPES:
        assert catalog[kind]
    return catalog


MASK_CATALOG = _build_mask_catalog()


def _segments_between(boundaries: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left > 1:
            segments.append((left + 1, right))
    return segments


def _seed_kind(seed: np.ndarray) -> str:
    assert seed.shape == (3, 3)
    colors = {int(v) for v in np.unique(seed) if int(v) != BG}
    assert colors and GREY not in colors and len(colors) == 1
    return _classify_mask(seed != BG)


def _tile_seed(seed: np.ndarray, shape: tuple[int, int], fill_color: int) -> np.ndarray:
    mask = seed != BG
    rr, cc = np.indices(shape)
    out = np.zeros(shape, dtype=int)
    out[mask[rr % 3, cc % 3]] = int(fill_color)
    return out


def _parse_legend(legend: np.ndarray) -> dict[str, int]:
    h, w = map(int, legend.shape)
    row_lines = [r for r in range(h) if np.all(legend[r, :] == GREY)]
    col_lines = [c for c in range(w) if np.all(legend[:, c] == GREY)]
    row_segments = _segments_between([-1] + row_lines + [h])
    col_segments = _segments_between([-1] + col_lines + [w])
    assert len(row_segments) == 2 and len(col_segments) == 2

    mapping: dict[str, int] = {}
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            sub = legend[r0:r1, c0:c1]
            assert sub.shape[0] >= 3 and sub.shape[1] >= 5
            pattern = sub[:3, :3]
            block = sub[:2, 3:5]
            assert np.all(block == block[0, 0])
            block_color = int(block[0, 0])
            assert block_color not in (BG, GREY)
            kind = _seed_kind(pattern)
            assert kind not in mapping
            mapping[kind] = block_color

    assert set(mapping) == set(TYPES)
    assert len(set(mapping.values())) == 4
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine the legend's symmetry->color mapping, classify each region's 3x3 seed,
    and tile that seed across the whole region using the mapped color.
    The legend region in the top-right stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    row_lines = [r for r in range(h) if np.all(g[r, :] == GREY)]
    col_lines = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert row_lines and col_lines
    assert row_lines[0] == 0 and row_lines[-1] == h - 1
    assert col_lines[0] == 0 and col_lines[-1] == w - 1

    row_segments = _segments_between(row_lines)
    col_segments = _segments_between(col_lines)
    assert len(row_segments) >= 2 and len(col_segments) >= 2

    legend_r0, legend_r1 = row_segments[0]
    legend_c0, legend_c1 = col_segments[-1]
    legend = g[legend_r0:legend_r1, legend_c0:legend_c1]
    mapping = _parse_legend(legend)

    out = g.copy()
    for ri, (r0, r1) in enumerate(row_segments):
        for ci, (c0, c1) in enumerate(col_segments):
            if ri == 0 and ci == len(col_segments) - 1:
                continue
            region = g[r0:r1, c0:c1]
            assert region.shape[0] >= 3 and region.shape[1] >= 3
            seed = region[:3, :3]
            kind = _seed_kind(seed)
            out[r0:r1, c0:c1] = _tile_seed(seed, region.shape, mapping[kind])

    return out


def _paint_mask(arr: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    sub = arr[top : top + 3, left : left + 3]
    sub[mask.astype(bool)] = int(color)


def _choose_layout() -> tuple[list[int], list[int]]:
    shape_choices = [
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    n_rows, n_cols = random.choice(shape_choices)
    row_heights = [random.randint(7, 10)] + [random.randint(4, 9) for _ in range(n_rows - 1)]
    col_widths = [random.randint(4, 9) for _ in range(n_cols - 1)] + [random.randint(11, 15)]
    return row_heights, col_widths


def _make_empty_grid(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_heights) + len(row_heights) + 1
    width = sum(col_widths) + len(col_widths) + 1
    grid = np.full((height, width), GREY, dtype=int)

    row_segments: list[tuple[int, int]] = []
    cursor = 1
    for size in row_heights:
        row_segments.append((cursor, cursor + size))
        cursor += size + 1

    col_segments: list[tuple[int, int]] = []
    cursor = 1
    for size in col_widths:
        col_segments.append((cursor, cursor + size))
        cursor += size + 1

    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            grid[r0:r1, c0:c1] = BG

    return grid, row_segments, col_segments


def _render_legend(legend: np.ndarray, mapping: dict[str, int], seed_colors: list[int]) -> None:
    h, w = map(int, legend.shape)
    split_r = random.randint(3, h - 4)
    split_c = random.randint(5, w - 6)
    legend[split_r, :] = GREY
    legend[:, split_c] = GREY

    row_segments = [(0, split_r), (split_r + 1, h)]
    col_segments = [(0, split_c), (split_c + 1, w)]
    panel_types = random.sample(list(TYPES), len(TYPES))

    idx = 0
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            sub = legend[r0:r1, c0:c1]
            kind = panel_types[idx]
            idx += 1
            mask = random.choice(MASK_CATALOG[kind])
            color = random.choice(seed_colors)
            _paint_mask(sub, 0, 0, mask, color)
            sub[:2, 3:5] = mapping[kind]


def _render_region(region: np.ndarray, kind: str, color: int) -> None:
    mask = random.choice(MASK_CATALOG[kind])
    _paint_mask(region, 0, 0, mask, color)


def _generate_once() -> np.ndarray:
    row_heights, col_widths = _choose_layout()
    grid, row_segments, col_segments = _make_empty_grid(row_heights, col_widths)

    fill_colors = random.sample(PALETTE, 4)
    mapping = {kind: fill for kind, fill in zip(TYPES, fill_colors)}
    seed_colors = [color for color in PALETTE if color not in fill_colors]
    assert len(seed_colors) >= 4

    legend_r0, legend_r1 = row_segments[0]
    legend_c0, legend_c1 = col_segments[-1]
    _render_legend(grid[legend_r0:legend_r1, legend_c0:legend_c1], mapping, seed_colors)

    region_positions = [
        (r0, r1, c0, c1)
        for ri, (r0, r1) in enumerate(row_segments)
        for ci, (c0, c1) in enumerate(col_segments)
        if not (ri == 0 and ci == len(col_segments) - 1)
    ]
    kinds = list(TYPES)
    while len(kinds) < len(region_positions):
        kinds.append(random.choice(TYPES))
    random.shuffle(kinds)

    for (r0, r1, c0, c1), kind in zip(region_positions, kinds):
        _render_region(grid[r0:r1, c0:c1], kind, random.choice(seed_colors))

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        out = main(grid)
        assert isinstance(out, np.ndarray)
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
