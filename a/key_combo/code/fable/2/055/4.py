from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
CELL_SIZE = 3
CELL_GAP = 1
CELL_STRIDE = CELL_SIZE + CELL_GAP
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STATE_NONE = 0
STATE_LEFT = 1
STATE_RIGHT = 2
STATE_BOTH = 3
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox(arr), "area": int(arr.shape[0])})
    return components


def _is_mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _enumerate_template_masks() -> tuple[list[np.ndarray], list[np.ndarray]]:
    asymmetric: list[np.ndarray] = []
    symmetric: list[np.ndarray] = []
    for bits in range(1, 1 << (CELL_SIZE * CELL_SIZE)):
        flat = np.array([(bits >> i) & 1 for i in range(CELL_SIZE * CELL_SIZE)], dtype=bool)
        mask = flat.reshape(CELL_SIZE, CELL_SIZE)
        if int(mask.sum()) < 4 or int(mask.sum()) > 7:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_mask_connected(mask):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            symmetric.append(mask)
        else:
            asymmetric.append(mask)
    return asymmetric, symmetric


ASYM_MASKS, SYM_MASKS = _enumerate_template_masks()


def _regular_lattice(
    components: list[dict],
) -> tuple[list[int], list[int], dict[tuple[int, int], dict], tuple[int, int, int, int]] | None:
    if len(components) < 4:
        return None

    by_anchor: dict[tuple[int, int], dict] = {}
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (CELL_SIZE, CELL_SIZE):
            return None
        by_anchor[(r0, c0)] = comp

    if len(by_anchor) != len(components):
        return None

    rows = sorted({r for r, _ in by_anchor})
    cols = sorted({c for _, c in by_anchor})
    if len(rows) * len(cols) != len(components):
        return None
    if any(rows[i + 1] - rows[i] != CELL_STRIDE for i in range(len(rows) - 1)):
        return None
    if any(cols[i + 1] - cols[i] != CELL_STRIDE for i in range(len(cols) - 1)):
        return None
    if any((r, c) not in by_anchor for r in rows for c in cols):
        return None

    lattice_bbox = (rows[0], cols[0], rows[-1] + CELL_SIZE - 1, cols[-1] + CELL_SIZE - 1)
    return rows, cols, by_anchor, lattice_bbox


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _all_color_inside(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox
    return bool(np.all((coords[:, 0] >= r0) & (coords[:, 0] <= r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1)))


def _detect_lattice(grid: np.ndarray, bg: int) -> tuple[list[int], list[int], tuple[int, int, int, int]]:
    assert GREY != bg, "background cannot be gray in this puzzle family"
    components = _find_components_of_color(grid, GREY)
    lattice = _regular_lattice(components)
    assert lattice is not None, "expected one regular gray 3x3 template lattice"
    rows, cols, _by_anchor, lattice_bbox = lattice

    r0, c0, r1, c1 = lattice_bbox
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert set(map(int, np.unique(crop))).issubset({bg, GREY}), "lattice crop must use only background and gray"
    return rows, cols, lattice_bbox


def _detect_panels(
    grid: np.ndarray,
    bg: int,
    rows_count: int,
    cols_count: int,
    lattice_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    h, w = grid.shape
    other_colors = sorted(set(map(int, np.unique(grid))) - {bg, GREY})
    candidates: list[tuple[np.ndarray, np.ndarray, int, int]] = []

    for sep_color in other_colors:
        coords = np.argwhere(grid == int(sep_color))
        if coords.shape[0] != rows_count:
            continue
        if len(set(map(int, coords[:, 1]))) != 1:
            continue
        sep_col = int(coords[0, 1])
        row_values = sorted(map(int, coords[:, 0]))
        row_start = row_values[0]
        if row_values != list(range(row_start, row_start + rows_count)):
            continue

        left_c0 = sep_col - cols_count
        right_c0 = sep_col + 1
        if left_c0 < 0 or right_c0 + cols_count > w:
            continue

        left_rect = grid[row_start : row_start + rows_count, left_c0:sep_col]
        right_rect = grid[row_start : row_start + rows_count, right_c0 : right_c0 + cols_count]
        left_colors = set(map(int, np.unique(left_rect)))
        right_colors = set(map(int, np.unique(right_rect)))
        if left_colors == {bg} or right_colors == {bg}:
            continue
        if bg not in left_colors or bg not in right_colors:
            continue
        if len(left_colors) != 2 or len(right_colors) != 2:
            continue

        left_fg = next(iter(left_colors - {bg}))
        right_fg = next(iter(right_colors - {bg}))
        if len({left_fg, right_fg, sep_color}) != 3:
            continue

        left_bbox = (row_start, left_c0, row_start + rows_count - 1, sep_col - 1)
        right_bbox = (row_start, right_c0, row_start + rows_count - 1, right_c0 + cols_count - 1)
        panel_bbox = (row_start, left_c0, row_start + rows_count - 1, right_c0 + cols_count - 1)
        if _bboxes_overlap(panel_bbox, lattice_bbox):
            continue
        if not _all_color_inside(grid, left_fg, left_bbox):
            continue
        if not _all_color_inside(grid, right_fg, right_bbox):
            continue

        candidates.append((left_rect.copy(), right_rect.copy(), left_fg, right_fg))

    assert len(candidates) == 1, "expected one valid separator column with two matching control panels"
    return candidates[0]


def _apply_controls(
    lattice_crop: np.ndarray,
    row_offsets: list[int],
    col_offsets: list[int],
    left_panel: np.ndarray,
    right_panel: np.ndarray,
    left_fg: int,
    right_fg: int,
    bg: int,
) -> np.ndarray:
    out = lattice_crop.copy()
    row0 = row_offsets[0]
    col0 = col_offsets[0]
    for i, row in enumerate(row_offsets):
        for j, col in enumerate(col_offsets):
            rr = row - row0
            cc = col - col0
            patch = out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE].copy()
            left_on = int(left_panel[i, j]) == int(left_fg)
            right_on = int(right_panel[i, j]) == int(right_fg)
            if left_on and right_on:
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = np.rot90(patch, 2)
            elif left_on:
                patch[patch == GREY] = left_fg
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = patch
            elif right_on:
                patch[patch == GREY] = right_fg
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = patch
            else:
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = bg
    return out


def _parse_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int], np.ndarray, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    rows, cols, lattice_bbox = _detect_lattice(g, bg)
    left_panel, right_panel, left_fg, right_fg = _detect_panels(g, bg, len(rows), len(cols), lattice_bbox)
    r0, c0, r1, c1 = lattice_bbox
    lattice_crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    return lattice_crop, rows, cols, left_panel, right_panel, left_fg, right_fg, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two control panels cell by cell to recolor, rotate, or erase the matching gray template lattice.

    Input structure:
    - One side contains an R by C lattice of gray 3x3 template masks, with one background row and column between neighboring templates.
    - Elsewhere, one separator-colored column splits two R by C panels that share a background color and use two different foreground colors.
    - The lattice templates all live inside fixed 3x3 cells, but their gray masks vary from slot to slot.

    Rule:
    - Detect the gray lattice and crop that whole lattice region to become the output canvas.
    - Compare the left and right panels at matching row and column indices.
    - Left-only foreground recolors that template to the left foreground color; right-only foreground recolors it to the right foreground color.
    - Both foreground keeps the template gray but rotates its 3x3 cell by 180 degrees, while both background erases that cell to background.

    Key insight:
    - The real control matrix is invisible in the input itself; it emerges only after comparing the two panels, and those four comparison outcomes drive color change, geometric change, and deletion across the lattice.
    """
    lattice_crop, rows, cols, left_panel, right_panel, left_fg, right_fg, bg = _parse_scene(grid)
    return _apply_controls(lattice_crop, rows, cols, left_panel, right_panel, left_fg, right_fg, bg)


def _choose_dimensions() -> tuple[int, int]:
    return random.randint(2, 4), random.randint(2, 4)


def _sample_states(rows: int, cols: int) -> np.ndarray:
    flat = [STATE_NONE, STATE_LEFT, STATE_RIGHT, STATE_BOTH]
    flat.extend(random.randrange(4) for _ in range(rows * cols - 4))
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(rows, cols)


def _full_vertical_column(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.shape[0] != mask.shape[0]:
        return False
    return len(set(map(int, coords[:, 1]))) == 1


def _sample_masks(rows: int, cols: int, states: np.ndarray) -> list[list[np.ndarray]]:
    total = rows * cols
    assert total <= len(ASYM_MASKS), "not enough asymmetric masks for a unique lattice"
    chosen = random.sample(ASYM_MASKS, total)
    masks: list[list[np.ndarray]] = []
    k = 0
    for i in range(rows):
        row_masks: list[np.ndarray] = []
        for j in range(cols):
            mask = chosen[k].copy()
            k += 1
            if int(states[i, j]) != STATE_BOTH and SYM_MASKS and random.random() < 0.2:
                mask = random.choice(SYM_MASKS).copy()
            row_masks.append(mask)
        masks.append(row_masks)

    if len({tuple(mask.ravel().tolist()) for row in masks for mask in row}) < 2:
        raise AssertionError("lattice must contain multiple template shapes")
    return masks


def _render_lattice(masks: list[list[np.ndarray]], bg: int) -> np.ndarray:
    rows = len(masks)
    cols = len(masks[0])
    lattice = np.full((rows * CELL_STRIDE - CELL_GAP, cols * CELL_STRIDE - CELL_GAP), bg, dtype=int)
    for i in range(rows):
        for j in range(cols):
            rr = i * CELL_STRIDE
            cc = j * CELL_STRIDE
            patch = lattice[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE]
            patch[masks[i][j]] = GREY
    return lattice


def _states_to_panels(states: np.ndarray, left_fg: int, right_fg: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.where(np.isin(states, [STATE_LEFT, STATE_BOTH]), left_fg, bg)
    right = np.where(np.isin(states, [STATE_RIGHT, STATE_BOTH]), right_fg, bg)
    return left.astype(int), right.astype(int)


def _apply_from_states(
    lattice_crop: np.ndarray,
    states: np.ndarray,
    left_fg: int,
    right_fg: int,
    bg: int,
    *,
    use_left: bool = True,
    use_right: bool = True,
    rotate_both: bool = True,
) -> np.ndarray:
    out = lattice_crop.copy()
    rows, cols = states.shape
    for i in range(rows):
        for j in range(cols):
            rr = i * CELL_STRIDE
            cc = j * CELL_STRIDE
            patch = out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE].copy()
            state = int(states[i, j])
            left_on = use_left and state in (STATE_LEFT, STATE_BOTH)
            right_on = use_right and state in (STATE_RIGHT, STATE_BOTH)
            if left_on and right_on:
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = np.rot90(patch, 2) if rotate_both else patch
            elif left_on:
                patch[patch == GREY] = left_fg
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = patch
            elif right_on:
                patch[patch == GREY] = right_fg
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = patch
            else:
                out[rr : rr + CELL_SIZE, cc : cc + CELL_SIZE] = bg
    return out


def _render_scene(
    lattice_crop: np.ndarray,
    left_panel: np.ndarray,
    right_panel: np.ndarray,
    sep_color: int,
    bg: int,
    *,
    lattice_left: bool,
) -> np.ndarray:
    lattice_h, lattice_w = lattice_crop.shape
    rows, cols = left_panel.shape
    panel_w = 2 * cols + 1
    gap = random.randint(1, 2)
    outer_left = random.randint(0, 1)
    outer_right = random.randint(0, 1)
    width = lattice_w + panel_w + gap + outer_left + outer_right

    height = max(lattice_h, rows) + random.randint(1, 4)
    canvas = np.full((height, width), bg, dtype=int)
    lattice_top = random.randint(0, height - lattice_h)
    panel_top = random.randint(0, height - rows)

    if lattice_left:
        lattice_left_c = outer_left
        panel_left_c = lattice_left_c + lattice_w + gap
    else:
        panel_left_c = outer_left
        lattice_left_c = panel_left_c + panel_w + gap

    canvas[lattice_top : lattice_top + lattice_h, lattice_left_c : lattice_left_c + lattice_w] = lattice_crop
    canvas[panel_top : panel_top + rows, panel_left_c : panel_left_c + cols] = left_panel
    canvas[panel_top : panel_top + rows, panel_left_c + cols] = sep_color
    canvas[panel_top : panel_top + rows, panel_left_c + cols + 1 : panel_left_c + 2 * cols + 1] = right_panel
    return canvas


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    states: np.ndarray,
    lattice_crop: np.ndarray,
    left_panel: np.ndarray,
    right_panel: np.ndarray,
    left_fg: int,
    right_fg: int,
    bg: int,
) -> bool:
    unique_states = set(map(int, states.ravel()))
    if unique_states != {STATE_NONE, STATE_LEFT, STATE_RIGHT, STATE_BOTH}:
        return False
    if _full_vertical_column(left_panel == left_fg) or _full_vertical_column(right_panel == right_fg):
        return False

    both_positions = np.argwhere(states == STATE_BOTH)
    if both_positions.size == 0:
        return False
    if not any(
        not np.array_equal(
            lattice_crop[r * CELL_STRIDE : r * CELL_STRIDE + CELL_SIZE, c * CELL_STRIDE : c * CELL_STRIDE + CELL_SIZE],
            np.rot90(
                lattice_crop[r * CELL_STRIDE : r * CELL_STRIDE + CELL_SIZE, c * CELL_STRIDE : c * CELL_STRIDE + CELL_SIZE],
                2,
            ),
        )
        for r, c in both_positions
    ):
        return False

    if np.array_equal(expected, lattice_crop):
        return False
    if np.array_equal(expected, _apply_from_states(lattice_crop, states, left_fg, right_fg, bg, rotate_both=False)):
        return False
    if np.array_equal(expected, _apply_from_states(lattice_crop, states, left_fg, right_fg, bg, use_left=False, use_right=True, rotate_both=False)):
        return False
    if np.array_equal(expected, _apply_from_states(lattice_crop, states, left_fg, right_fg, bg, use_left=True, use_right=False, rotate_both=False)):
        return False
    if set(map(int, np.unique(expected))) != {bg, GREY, left_fg, right_fg}:
        return False

    try:
        parsed = _parse_scene(inp)
    except AssertionError:
        return False
    parsed_lattice, parsed_rows, parsed_cols, parsed_left, parsed_right, parsed_left_fg, parsed_right_fg, parsed_bg = parsed
    if parsed_bg != bg or parsed_left_fg != left_fg or parsed_right_fg != right_fg:
        return False
    if parsed_left.shape != left_panel.shape or parsed_right.shape != right_panel.shape:
        return False
    if not np.array_equal(parsed_lattice, lattice_crop):
        return False
    if not np.array_equal(parsed_left, left_panel) or not np.array_equal(parsed_right, right_panel):
        return False
    if len(parsed_rows) != states.shape[0] or len(parsed_cols) != states.shape[1]:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _attempt_generate() -> np.ndarray | None:
    rows, cols = _choose_dimensions()
    states = _sample_states(rows, cols)

    palette = [c for c in range(10) if c != GREY]
    bg = random.choice(palette)
    remaining = [c for c in palette if c != bg]
    left_fg, right_fg, sep_color = random.sample(remaining, 3)

    left_panel, right_panel = _states_to_panels(states, left_fg, right_fg, bg)
    if _full_vertical_column(left_panel == left_fg) or _full_vertical_column(right_panel == right_fg):
        return None

    masks = _sample_masks(rows, cols, states)
    lattice_crop = _render_lattice(masks, bg)
    expected = _apply_from_states(lattice_crop, states, left_fg, right_fg, bg)
    inp = _render_scene(
        lattice_crop,
        left_panel,
        right_panel,
        sep_color,
        bg,
        lattice_left=bool(random.getrandbits(1)),
    )

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(inp, expected, states, lattice_crop, left_panel, right_panel, left_fg, right_fg, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-control-matrix ARC input")
