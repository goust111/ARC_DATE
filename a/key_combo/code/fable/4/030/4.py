from __future__ import annotations

import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    dst[r0 : r0 + h, c0 : c0 + w] = src


def _corner_origins(shape: tuple[int, int]) -> list[tuple[str, int, int]]:
    h, w = shape
    return [
        ("tl", 0, 0),
        ("tr", 0, w - 4),
        ("bl", h - 4, 0),
        ("br", h - 4, w - 4),
    ]


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray, int]:
    matches = []
    for _name, r0, c0 in _corner_origins(grid.shape):
        patch = grid[r0 : r0 + 4, c0 : c0 + 4]
        if patch.shape != (4, 4):
            continue
        border = np.concatenate(
            [patch[0, :], patch[3, :], patch[1:3, 0], patch[1:3, 3]]
        )
        border_colors = set(map(int, border))
        if len(border_colors) != 1:
            continue
        border_color = next(iter(border_colors))
        if border_color == bg:
            continue
        interior = patch[1:3, 1:3]
        interior_colors = set(map(int, interior.ravel())) - {bg}
        if len(interior_colors) != 1:
            continue
        output_color = next(iter(interior_colors))
        if output_color == border_color:
            continue
        table = interior == output_color
        matches.append(((r0, c0, r0 + 3, c0 + 3), table, int(output_color)))

    assert len(matches) == 1, "expected exactly one corner truth-table legend"
    return matches[0]


def _find_panels(
    grid: np.ndarray, bg: int, legend_box: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    work = grid.copy()
    r0, c0, r1, c1 = legend_box
    work[r0 : r1 + 1, c0 : c1 + 1] = bg

    main_box = bbox(work != bg)
    assert main_box is not None, "expected a main panel area besides the legend"
    mr0, mc0, mr1, mc1 = main_box
    patch = work[mr0 : mr1 + 1, mc0 : mc1 + 1]
    h, w = patch.shape

    candidates = []
    c = 0
    while c < w:
        col_color = int(patch[0, c])
        if col_color != bg and np.all(patch[:, c] == col_color):
            c_end = c
            while c_end + 1 < w and np.all(patch[:, c_end + 1] == col_color):
                c_end += 1
            left_w = c
            right_w = w - c_end - 1
            if left_w == right_w and left_w > 0:
                left = patch[:, :c]
                right = patch[:, c_end + 1 :]
                fg_colors = (set(map(int, np.unique(left))) | set(map(int, np.unique(right)))) - {bg}
                if len(fg_colors) == 1:
                    mask_color = next(iter(fg_colors))
                    if mask_color != col_color:
                        if (
                            np.any(left == mask_color)
                            and np.any(right == mask_color)
                            and np.all((left == bg) | (left == mask_color))
                            and np.all((right == bg) | (right == mask_color))
                        ):
                            candidates.append((left.copy(), right.copy()))
            c = c_end + 1
        else:
            c += 1

    assert len(candidates) == 1, "expected one equal-width separator between two masks"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner truth table and use it to combine the two masks on either side of the separator bar.

    Input structure:
    - The background is the dominant color.
    - A vertical single-color separator bar splits the main area into equal-size left and right binary masks.
    - A framed 2x2 legend sits in one grid corner; lit cells use the output color and unlit cells stay background.

    Rule:
    - Interpret each legend cell at row r and column c as the Boolean output for left value r and right value c.
    - Treat each mask cell as false on background and true on its foreground color.
    - Apply that 2-input Boolean function pixelwise to the aligned left and right masks.
    - Return only one mask-sized output grid, painting true cells with the legend's lit color.

    Key insight:
    - The corner block is not an example picture; its four interior cells are the full truth table of the operator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    legend_box, table, output_color = _find_legend(g, bg)
    left, right = _find_panels(g, bg, legend_box)

    mask_color = int(next(iter(set(map(int, np.unique(left))) - {bg})))
    left_true = left == mask_color
    right_true = right == mask_color

    out = np.full(left.shape, bg, dtype=int)
    truth = table[left_true.astype(int), right_true.astype(int)]
    out[truth] = output_color
    return out


# --------------- generator (gpt-5) ---------------

BACKGROUND = 0
NON_BG_COLORS = tuple(range(1, 10))
MASK_H_RANGE = (6, 11)
MASK_W_RANGE = (5, 10)
SEP_W_RANGE = (1, 2)
ROW_GAP_RANGE = (1, 3)
EXTRA_W_RANGE = (0, 6)
EXTRA_H_RANGE = (0, 4)
MAX_ATTEMPTS = 300

TRUTH_TABLES = (
    np.array([[0, 0], [0, 1]], dtype=bool),  # and
    np.array([[0, 1], [1, 1]], dtype=bool),  # or
    np.array([[0, 1], [1, 0]], dtype=bool),  # xor
    np.array([[1, 0], [0, 1]], dtype=bool),  # xnor
    np.array([[0, 0], [1, 1]], dtype=bool),  # left
    np.array([[0, 1], [0, 1]], dtype=bool),  # right
    np.array([[0, 0], [1, 0]], dtype=bool),  # left and not right
    np.array([[0, 1], [0, 0]], dtype=bool),  # right and not left
    np.array([[1, 1], [0, 1]], dtype=bool),  # left -> right
    np.array([[1, 0], [1, 1]], dtype=bool),  # right -> left
)


def _draw_legend(table: np.ndarray, border_color: int, output_color: int) -> np.ndarray:
    patch = np.full((4, 4), border_color, dtype=int)
    patch[1:3, 1:3] = np.where(table, output_color, BACKGROUND)
    return patch


def _paint_random_rect(mask: np.ndarray) -> None:
    h, w = mask.shape
    r0 = random.randint(0, h - 1)
    c0 = random.randint(0, w - 1)
    r1 = random.randint(r0, h - 1)
    c1 = random.randint(c0, w - 1)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint_random_line(mask: np.ndarray) -> None:
    h, w = mask.shape
    if random.random() < 0.5:
        r = random.randint(0, h - 1)
        c0 = random.randint(0, w - 1)
        c1 = random.randint(c0, w - 1)
        mask[r, c0 : c1 + 1] = True
    else:
        c = random.randint(0, w - 1)
        r0 = random.randint(0, h - 1)
        r1 = random.randint(r0, h - 1)
        mask[r0 : r1 + 1, c] = True


def _random_mask(h: int, w: int, *, anchor_side: str) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(random.randint(2, 4)):
            _paint_random_rect(mask)
        for _ in range(random.randint(1, 3)):
            _paint_random_line(mask)

        if anchor_side == "left":
            mask[random.randrange(h), 0] = True
        else:
            mask[random.randrange(h), w - 1] = True

        if random.random() < 0.6:
            mask[random.randrange(h), random.randrange(w)] = True
            mask[random.randrange(h), random.randrange(w)] = False

        density = float(mask.mean())
        if not (0.18 <= density <= 0.68):
            continue
        if mask.all(axis=0).any():
            continue
        if np.array_equal(mask, mask[0, 0]):
            continue
        return mask
    return None


def _build_input(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    table: np.ndarray,
    mask_color: int,
    sep_color: int,
    border_color: int,
    output_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask_h, mask_w = left_mask.shape
    sep_w = random.randint(*SEP_W_RANGE)
    main_w = 2 * mask_w + sep_w
    main_patch = np.full((mask_h, main_w), BACKGROUND, dtype=int)
    left_view = main_patch[:, :mask_w]
    left_view[left_mask] = mask_color
    main_patch[:, mask_w : mask_w + sep_w] = sep_color
    right_view = main_patch[:, mask_w + sep_w :]
    right_view[right_mask] = mask_color

    output = np.full((mask_h, mask_w), BACKGROUND, dtype=int)
    truth = table[left_mask.astype(int), right_mask.astype(int)]
    output[truth] = output_color

    row_gap = random.randint(*ROW_GAP_RANGE)
    extra_h = random.randint(*EXTRA_H_RANGE)
    extra_w = random.randint(*EXTRA_W_RANGE)
    total_h = 4 + row_gap + mask_h + extra_h
    total_w = max(4, main_w) + extra_w
    grid = np.full((total_h, total_w), BACKGROUND, dtype=int)

    main_c = random.randint(0, total_w - main_w)
    legend = _draw_legend(table, border_color, output_color)
    corner = random.choice(("tl", "tr", "bl", "br"))

    if corner in ("tl", "tr"):
        legend_r = 0
        main_r = 4 + row_gap
    else:
        main_r = random.randint(0, extra_h)
        legend_r = total_h - 4
    legend_c = 0 if corner in ("tl", "bl") else total_w - 4

    paste(grid, legend, legend_r, legend_c)
    paste(grid, main_patch, main_r, main_c)
    return grid, output


def _sample_truth_table() -> np.ndarray:
    # Favor asymmetric operators so row/column semantics are visible in a few examples.
    weighted = [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]
    return TRUTH_TABLES[random.choice(weighted)].copy()


def _gen_attempt() -> np.ndarray | None:
    mask_h = random.randint(*MASK_H_RANGE)
    mask_w = random.randint(*MASK_W_RANGE)
    left_mask = _random_mask(mask_h, mask_w, anchor_side="left")
    right_mask = _random_mask(mask_h, mask_w, anchor_side="right")
    if left_mask is None or right_mask is None:
        return None

    table = _sample_truth_table()
    colors = random.sample(NON_BG_COLORS, 4)
    mask_color, sep_color, border_color, output_color = colors

    grid, expected = _build_input(
        left_mask, right_mask, table, mask_color, sep_color, border_color, output_color
    )

    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    if np.all(expected == BACKGROUND) or np.all(expected == output_color):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
