import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

CATEGORIES = ("horizontal", "vertical", "rotation", "asymmetric")
PALETTE_COLORS = (BLUE, GREEN, YELLOW, MAGENTA, ORANGE, CYAN, MAROON)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _symmetry_flags(mask: np.ndarray) -> Tuple[bool, bool, bool]:
    tight = _tight(mask)
    if tight.size == 0:
        return False, False, False
    horizontal = np.array_equal(tight, np.fliplr(tight))
    vertical = np.array_equal(tight, np.flipud(tight))
    rotation = np.array_equal(tight, np.rot90(tight, 2))
    return horizontal, vertical, rotation


def _classify(mask: np.ndarray) -> str:
    horizontal, vertical, rotation = _symmetry_flags(mask)
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    if rotation:
        return "rotation"
    return "asymmetric"


def _is_connected(mask: np.ndarray) -> bool:
    tight = _tight(mask)
    if tight.size == 0 or not tight.any():
        return False
    return len(_components(tight)) == 1


def _mask_signature(mask: np.ndarray) -> Tuple[str, ...]:
    tight = _tight(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _find_grid(g: np.ndarray) -> Dict[str, int]:
    row_scores = np.count_nonzero(g == GREY, axis=1)
    col_scores = np.count_nonzero(g == GREY, axis=0)
    assert row_scores.max() > 0 and col_scores.max() > 0, "missing grey separators"

    sep_rows = np.flatnonzero(row_scores == row_scores.max())
    sep_cols = np.flatnonzero(col_scores == col_scores.max())
    nr = len(sep_rows) + 1
    nc = len(sep_cols) + 1
    assert nr >= 2 and nc >= 2, "expected at least a 2x2 cell grid"

    width = int(row_scores.max())
    height = int(col_scores.max())
    assert (width - (nc - 1)) % nc == 0
    assert (height - (nr - 1)) % nr == 0
    cell_size_w = (width - (nc - 1)) // nc
    cell_size_h = (height - (nr - 1)) // nr
    assert cell_size_w == cell_size_h
    cell_size = int(cell_size_w)
    assert cell_size >= 3

    if len(sep_rows) > 1:
        assert np.all(np.diff(sep_rows) == cell_size + 1)
    if len(sep_cols) > 1:
        assert np.all(np.diff(sep_cols) == cell_size + 1)

    r0 = int(sep_rows[0] - cell_size)
    c0 = int(sep_cols[0] - cell_size)
    r1 = int(sep_rows[-1] + cell_size)
    c1 = int(sep_cols[-1] + cell_size)
    assert 0 <= r0 <= r1 < g.shape[0]
    assert 0 <= c0 <= c1 < g.shape[1]

    for r in sep_rows:
        assert np.all(g[r, c0 : c1 + 1] == GREY)
    for c in sep_cols:
        assert np.all(g[r0 : r1 + 1, c] == GREY)

    return {
        "cell_size": cell_size,
        "rows": nr,
        "cols": nc,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
    }


def _extract_example_patch(g: np.ndarray, block_bbox: Tuple[int, int, int, int], cell_size: int) -> np.ndarray:
    br0, bc0, br1, bc1 = block_bbox
    block_h = br1 - br0 + 1
    top = br0 - (cell_size - block_h) // 2
    best_patch = None
    best_score = -1
    for gap in range(0, 3):
        for side in (1, -1):
            if side == 1:
                left = bc1 + 1 + gap
            else:
                left = bc0 - gap - cell_size
            if top < 0 or left < 0 or top + cell_size > g.shape[0] or left + cell_size > g.shape[1]:
                continue
            patch = g[top : top + cell_size, left : left + cell_size]
            if np.any((patch != BG) & (patch != RED)):
                continue
            red_count = int(np.count_nonzero(patch == RED))
            if red_count > best_score:
                best_score = red_count
                best_patch = patch
    assert best_patch is not None and np.count_nonzero(best_patch == RED) > 0, "failed to locate legend example"
    return best_patch


def _legend_mapping(g: np.ndarray, grid: Dict[str, int]) -> Dict[str, int]:
    outside = np.ones_like(g, dtype=bool)
    outside[grid["r0"] : grid["r1"] + 1, grid["c0"] : grid["c1"] + 1] = False
    color_blocks = outside & (g != BG) & (g != RED) & (g != GREY)
    mapping: Dict[str, int] = {}
    for comp in _components(color_blocks):
        vals = {int(g[r, c]) for r, c in comp}
        assert len(vals) == 1
        color = vals.pop()
        block_bbox = _bbox(comp)
        example = _extract_example_patch(g, block_bbox, grid["cell_size"])
        mapping[_classify(example == RED)] = color
    assert set(mapping) == set(CATEGORIES), "legend must provide all four symmetry classes"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grid cell background according to the symmetry class of its red seed pattern.

    The grey separator lattice defines the cells. Outside the lattice, four legend labels pair
    a color block with a red example pattern; the example's own symmetry class identifies which
    category that color stands for.
    """
    g = np.asarray(grid, dtype=int)
    grid_info = _find_grid(g)
    mapping = _legend_mapping(g, grid_info)

    out = g.copy()
    s = grid_info["cell_size"]
    for rr in range(grid_info["rows"]):
        for cc in range(grid_info["cols"]):
            r0 = grid_info["r0"] + rr * (s + 1)
            c0 = grid_info["c0"] + cc * (s + 1)
            cell = g[r0 : r0 + s, c0 : c0 + s]
            category = _classify(cell == RED)
            color = mapping[category]
            fill_mask = cell == BG
            out[r0 : r0 + s, c0 : c0 + s][fill_mask] = color
    return out


def _make_lr_symmetric(h: int, w: int, rng: random.Random) -> np.ndarray:
    prob = rng.uniform(0.25, 0.55)
    left = rng.random() < prob
    seed = np.array(
        [[rng.random() < prob for _ in range((w + 1) // 2)] for _ in range(h)],
        dtype=bool,
    )
    mask = np.zeros((h, w), dtype=bool)
    mask[:, : (w + 1) // 2] = seed
    mask |= np.fliplr(mask)
    if left:
        mask[rng.randrange(h), 0] = True
        mask[rng.randrange(h), -1] = True
    return mask


def _make_ud_symmetric(h: int, w: int, rng: random.Random) -> np.ndarray:
    prob = rng.uniform(0.25, 0.55)
    seed = np.array(
        [[rng.random() < prob for _ in range(w)] for _ in range((h + 1) // 2)],
        dtype=bool,
    )
    mask = np.zeros((h, w), dtype=bool)
    mask[: (h + 1) // 2, :] = seed
    mask |= np.flipud(mask)
    return mask


def _make_rot_symmetric(h: int, w: int, rng: random.Random) -> np.ndarray:
    prob = rng.uniform(0.22, 0.5)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            rr, cc = h - 1 - r, w - 1 - c
            if (r > rr) or (r == rr and c > cc):
                continue
            if rng.random() < prob:
                mask[r, c] = True
                mask[rr, cc] = True
    return mask


def _make_both_symmetric(h: int, w: int, rng: random.Random) -> np.ndarray:
    prob = rng.uniform(0.2, 0.45)
    seed = np.array(
        [[rng.random() < prob for _ in range((w + 1) // 2)] for _ in range((h + 1) // 2)],
        dtype=bool,
    )
    mask = np.zeros((h, w), dtype=bool)
    mask[: (h + 1) // 2, : (w + 1) // 2] = seed
    mask |= np.fliplr(mask)
    mask |= np.flipud(mask)
    return mask


def _make_asymmetric(h: int, w: int, rng: random.Random) -> np.ndarray:
    prob = rng.uniform(0.22, 0.5)
    return np.array([[rng.random() < prob for _ in range(w)] for _ in range(h)], dtype=bool)


def _target_ok(mask: np.ndarray, target: str, exact: bool) -> bool:
    tight = _tight(mask)
    if tight.size == 0 or tight.shape[0] < 2 or tight.shape[1] < 2:
        return False
    if tight.sum() < 3 or tight.sum() >= tight.size:
        return False
    if not _is_connected(tight):
        return False
    horizontal, vertical, rotation = _symmetry_flags(tight)
    if exact:
        if target == "horizontal":
            return horizontal and not vertical
        if target == "vertical":
            return vertical and not horizontal
        if target == "rotation":
            return rotation and not horizontal and not vertical
        return not horizontal and not vertical and not rotation
    return _classify(tight) == target


def _generate_shape(
    cell_size: int,
    target: str,
    rng: random.Random,
    *,
    exact: bool,
    allow_priority_double: bool = False,
) -> np.ndarray:
    for _ in range(500):
        h = rng.randint(3, cell_size)
        w = rng.randint(3, cell_size)
        if allow_priority_double and target == "horizontal" and rng.random() < 0.35:
            mask = _make_both_symmetric(h, w, rng)
        elif target == "horizontal":
            mask = _make_lr_symmetric(h, w, rng)
        elif target == "vertical":
            mask = _make_ud_symmetric(h, w, rng)
        elif target == "rotation":
            mask = _make_rot_symmetric(h, w, rng)
        else:
            mask = _make_asymmetric(h, w, rng)
        if _target_ok(mask, target, exact):
            return _tight(mask)
    raise RuntimeError(f"failed to generate {target} shape")


def _place_in_square(mask: np.ndarray, side: int, rng: random.Random) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    h, w = mask.shape
    top = rng.randint(0, side - h)
    left = rng.randint(0, side - w)
    out[top : top + h, left : left + w][mask] = RED
    return out


def _draw_grid(rows: int, cols: int, cell_size: int, cell_patterns: Sequence[np.ndarray]) -> np.ndarray:
    h = rows * cell_size + (rows - 1)
    w = cols * cell_size + (cols - 1)
    grid = np.zeros((h, w), dtype=int)
    for r in range(1, rows):
        grid[r * cell_size + (r - 1), :] = GREY
    for c in range(1, cols):
        grid[:, c * cell_size + (c - 1)] = GREY
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (cell_size + 1)
            c0 = cc * (cell_size + 1)
            patch = cell_patterns[idx]
            idx += 1
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch
    return grid


def _paint_expected(
    grid: np.ndarray,
    rows: int,
    cols: int,
    cell_size: int,
    categories: Sequence[str],
    category_colors: Dict[str, int],
    grid_r0: int,
    grid_c0: int,
) -> np.ndarray:
    out = grid.copy()
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = grid_r0 + rr * (cell_size + 1)
            c0 = grid_c0 + cc * (cell_size + 1)
            cell = out[r0 : r0 + cell_size, c0 : c0 + cell_size]
            cell[cell == BG] = category_colors[categories[idx]]
            idx += 1
    return out


def _legend_stack(
    cell_size: int,
    block_size: int,
    legend_order: Sequence[str],
    category_colors: Dict[str, int],
    legend_examples: Dict[str, np.ndarray],
    vgap: int,
) -> np.ndarray:
    patch_w = block_size + 1 + cell_size
    patch_h = cell_size
    total_h = len(legend_order) * patch_h + (len(legend_order) - 1) * vgap
    canvas = np.zeros((total_h, patch_w), dtype=int)
    for idx, category in enumerate(legend_order):
        r0 = idx * (patch_h + vgap)
        block_top = r0 + (cell_size - block_size) // 2
        canvas[block_top : block_top + block_size, 0:block_size] = category_colors[category]
        canvas[r0 : r0 + cell_size, block_size + 1 : block_size + 1 + cell_size] = legend_examples[category]
    return canvas


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    cell_size = rng.randint(5, 7)
    block_size = rng.randint(2, min(3, cell_size - 2))
    side = rng.choice(["right", "bottom"])
    vgap = rng.randint(1, 2)

    total_cells = rows * cols
    cell_categories = list(CATEGORIES)
    while len(cell_categories) < total_cells:
        cell_categories.append(rng.choice(CATEGORIES))
    rng.shuffle(cell_categories)

    used = set()
    cell_patterns: List[np.ndarray] = []
    for category in cell_categories:
        for _ in range(200):
            shape = _generate_shape(
                cell_size,
                category,
                rng,
                exact=False,
                allow_priority_double=True,
            )
            sig = _mask_signature(shape)
            if sig in used:
                continue
            used.add(sig)
            cell_patterns.append(_place_in_square(shape, cell_size, rng))
            break
        else:
            return None

    legend_examples: Dict[str, np.ndarray] = {}
    for category in CATEGORIES:
        for _ in range(200):
            shape = _generate_shape(cell_size, category, rng, exact=True)
            sig = _mask_signature(shape)
            if sig in used:
                continue
            used.add(sig)
            legend_examples[category] = _place_in_square(shape, cell_size, rng)
            break
        else:
            return None

    legend_colors = rng.sample(PALETTE_COLORS, 4)
    category_colors = {category: legend_colors[idx] for idx, category in enumerate(CATEGORIES)}
    legend_order = list(CATEGORIES)
    rng.shuffle(legend_order)

    core_grid = _draw_grid(rows, cols, cell_size, cell_patterns)
    legend = _legend_stack(cell_size, block_size, legend_order, category_colors, legend_examples, vgap)

    top_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    gap = rng.randint(2, 4)

    gh, gw = core_grid.shape
    lh, lw = legend.shape
    if side == "right":
        legend_top = rng.randint(1, 3)
        canvas_h = max(top_margin + gh + bottom_margin, legend_top + lh + rng.randint(1, 3))
        canvas_w = left_margin + gw + gap + lw + right_margin
        canvas = np.zeros((canvas_h, canvas_w), dtype=int)
        grid_r0 = top_margin
        grid_c0 = left_margin
        legend_r0 = legend_top
        legend_c0 = grid_c0 + gw + gap
    else:
        legend_left = rng.randint(1, 3)
        canvas_h = top_margin + gh + gap + lh + bottom_margin
        canvas_w = max(left_margin + gw + right_margin, legend_left + lw + rng.randint(1, 3))
        canvas = np.zeros((canvas_h, canvas_w), dtype=int)
        grid_r0 = top_margin
        grid_c0 = left_margin
        legend_r0 = grid_r0 + gh + gap
        legend_c0 = legend_left

    canvas[grid_r0 : grid_r0 + gh, grid_c0 : grid_c0 + gw] = core_grid
    canvas[legend_r0 : legend_r0 + lh, legend_c0 : legend_c0 + lw] = legend

    expected = _paint_expected(
        canvas,
        rows,
        cols,
        cell_size,
        cell_categories,
        category_colors,
        grid_r0,
        grid_c0,
    )
    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-palette task")
