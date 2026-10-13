import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

CONTROL_COLORS = (RED, BLUE, YELLOW)
NON_CONTROL_INPUT_COLORS = (BLACK, 5, 6, 7, 8, 9)

HEIGHT_RANGE = (3, 9)
PANEL_WIDTH_RANGE = (4, 10)
MAX_GENERATION_ATTEMPTS = 400


def _unique_colors(grid: np.ndarray) -> set[int]:
    return set(map(int, np.unique(grid)))


def _split_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert width % 2 == 1 and width >= 3, "expected two equal-width panels plus one separator column"

    sep_col = width // 2
    left = g[:, :sep_col]
    seam = g[:, sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape == (height, sep_col), "expected aligned equal-size side panels"
    return left, seam, right


def _extract_panel_palette(left: np.ndarray, right: np.ndarray) -> tuple[int, int, int]:
    left_colors = _unique_colors(left)
    right_colors = _unique_colors(right)
    shared = left_colors & right_colors
    assert len(shared) == 1, "the panels must share exactly one background color"

    bg = shared.pop()
    left_fg = left_colors - {bg}
    right_fg = right_colors - {bg}
    assert bg != GREEN, "green is reserved for the output marker"
    assert len(left_fg) == 1 and len(right_fg) == 1, "each panel must contain exactly one foreground color"

    left_color = next(iter(left_fg))
    right_color = next(iter(right_fg))
    assert left_color != right_color, "the two panels must use different foreground colors"
    assert left_color not in CONTROL_COLORS and right_color not in CONTROL_COLORS, "panel colors must stay separate from seam control colors"
    return int(bg), int(left_color), int(right_color)


def _apply_operator(left_mask: np.ndarray, right_mask: np.ndarray, control_color: int) -> np.ndarray:
    if control_color == RED:
        return left_mask & right_mask
    if control_color == BLUE:
        return left_mask | right_mask
    if control_color == YELLOW:
        return left_mask ^ right_mask
    raise AssertionError("separator cells must use only red, blue, or yellow")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge the two side panels row by row using the separator column as a stream of boolean opcodes.

    Input structure:
    - The grid is split into an equal-width left panel, a single separator column, and an equal-width right panel.
    - Each side panel uses one shared background color plus its own single foreground color.
    - Every separator cell is a control color, and different rows may use different control colors.

    Rule:
    - Remove the separator column and keep the single-panel size.
    - For each row, read that row's separator color: red means intersection of the two foreground masks, blue means union, and yellow means exclusive-or.
    - Paint green on the cells selected by that row's boolean operator and paint the shared background color elsewhere.

    Key insight:
    - The middle column is not a border; it is a per-row instruction stream that changes how the aligned panel masks must be combined.
    """
    left, seam, right = _split_grid(grid)
    bg, _left_fg, _right_fg = _extract_panel_palette(left, right)
    assert set(map(int, seam)).issubset(CONTROL_COLORS), "separator column must contain only the three control colors"

    out = np.full(left.shape, bg, dtype=int)
    for r, control_color in enumerate(seam):
        keep = _apply_operator(left[r] != bg, right[r] != bg, int(control_color))
        out[r, keep] = GREEN
    return out


def _sample_row_masks(panel_w: int) -> tuple[np.ndarray, np.ndarray]:
    while True:
        both_count = random.randint(1, panel_w - 2)
        remaining = panel_w - both_count
        bg_count = random.randint(1, remaining - 1)
        one_sided_count = remaining - bg_count

        slots = list(range(panel_w))
        random.shuffle(slots)
        both_slots = slots[:both_count]
        one_slots = slots[both_count : both_count + one_sided_count]

        if one_sided_count >= 2 and random.random() < 0.65:
            left_only_count = random.randint(1, one_sided_count - 1)
        else:
            left_only_count = random.randint(0, one_sided_count)
        right_only_count = one_sided_count - left_only_count

        left_mask = np.zeros(panel_w, dtype=bool)
        right_mask = np.zeros(panel_w, dtype=bool)
        left_mask[both_slots] = True
        right_mask[both_slots] = True
        left_mask[one_slots[:left_only_count]] = True
        right_mask[one_slots[left_only_count:]] = True

        if _row_is_disambiguating(left_mask, right_mask):
            return left_mask, right_mask


def _row_is_disambiguating(left_mask: np.ndarray, right_mask: np.ndarray) -> bool:
    outputs = [
        _apply_operator(left_mask, right_mask, RED),
        _apply_operator(left_mask, right_mask, BLUE),
        _apply_operator(left_mask, right_mask, YELLOW),
    ]
    return (
        not np.array_equal(outputs[0], outputs[1])
        and not np.array_equal(outputs[0], outputs[2])
        and not np.array_equal(outputs[1], outputs[2])
    )


def _global_output(left: np.ndarray, right: np.ndarray, bg: int, control_color: int) -> np.ndarray:
    out = np.full(left.shape, bg, dtype=int)
    for r in range(left.shape[0]):
        keep = _apply_operator(left[r] != bg, right[r] != bg, control_color)
        out[r, keep] = GREEN
    return out


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        left, seam, right = _split_grid(inp)
        bg, _left_fg, _right_fg = _extract_panel_palette(left, right)
        expected = main(inp)
    except Exception:
        return False

    if not set(map(int, seam)) == set(CONTROL_COLORS):
        return False

    seen_left_only = False
    seen_right_only = False
    for r, control_color in enumerate(seam):
        left_mask = left[r] != bg
        right_mask = right[r] != bg
        if not _row_is_disambiguating(left_mask, right_mask):
            return False

        row_outputs = {
            RED: _apply_operator(left_mask, right_mask, RED),
            BLUE: _apply_operator(left_mask, right_mask, BLUE),
            YELLOW: _apply_operator(left_mask, right_mask, YELLOW),
        }
        if not np.array_equal(expected[r] == GREEN, row_outputs[int(control_color)]):
            return False

        seen_left_only |= bool(np.any(left_mask & ~right_mask))
        seen_right_only |= bool(np.any(right_mask & ~left_mask))

    if not seen_left_only or not seen_right_only:
        return False

    area = expected.size
    green_count = int(np.sum(expected == GREEN))
    if green_count < max(2, area // 6) or green_count > area - max(2, area // 6):
        return False

    if len({tuple(row.tolist()) for row in expected}) < 2:
        return False

    for control_color in CONTROL_COLORS:
        if np.array_equal(expected, _global_output(left, right, bg, control_color)):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    bg, left_fg, right_fg = random.sample(list(NON_CONTROL_INPUT_COLORS), 3)

    seam_colors = [RED, BLUE, YELLOW]
    seam_colors.extend(random.choice(CONTROL_COLORS) for _ in range(height - 3))
    random.shuffle(seam_colors)

    left = np.full((height, panel_w), bg, dtype=int)
    right = np.full((height, panel_w), bg, dtype=int)

    for r in range(height):
        left_mask, right_mask = _sample_row_masks(panel_w)
        left[r, left_mask] = left_fg
        right[r, right_mask] = right_fg

    inp = np.full((height, panel_w * 2 + 1), bg, dtype=int)
    inp[:, :panel_w] = left
    inp[:, panel_w] = np.asarray(seam_colors, dtype=int)
    inp[:, panel_w + 1 :] = right

    if not _semantic_checks(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid per-row boolean-seam input")
