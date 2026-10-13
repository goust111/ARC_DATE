import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint green in each output cell where both side panels are background.

    Input structure:
    - The grid has a centered solid non-background separator column.
    - The separator splits the grid into equal-size left and right panels.
    - Each panel is a same-size mask using the shared background and one foreground color.

    Rule:
    - Remove the separator column.
    - Compare the two side panels cell by cell.
    - Output background where either panel has foreground, and green where both panels have background.

    Key insight:
    - The output is the intersection of the two panels' empty space, not a copy of either colored mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 1 and width >= 3, "expected two equal panels plus a separator"

    bg = get_bg(g)
    sep_col = width // 2
    sep_color = int(g[0, sep_col])
    assert sep_color != bg, "separator must be non-background"
    assert np.all(g[:, sep_col] == sep_color), "expected a solid separator column"

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape == (height, sep_col), "expected equal-size side panels"

    left_fg = colors_present(left, exclude={bg})
    right_fg = colors_present(right, exclude={bg})
    assert len(left_fg) == 1 and len(right_fg) == 1, "each panel must have one foreground color"
    assert sep_color not in left_fg and sep_color not in right_fg, "separator color must not appear in panels"

    out = np.full(left.shape, bg, dtype=int)
    out[(left == bg) & (right == bg)] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------

# Fixed colors.
_GEN_BG, _GEN_SEP, _GEN_GREEN = Color.BLACK, Color.BLUE, Color.GREEN
_GEN_RIGHT, _GEN_LEFT = Color.GREY, Color.ORANGE

# Fixed geometry and retry budget.
_GEN_HEIGHT = 4
_GEN_PANEL_W = 3
_GEN_PANEL_SHAPE = (_GEN_HEIGHT, _GEN_PANEL_W)
_GEN_PANEL_CELLS = tuple((r, c) for r in range(_GEN_HEIGHT) for c in range(_GEN_PANEL_W))
_GEN_MAX_ATTEMPTS = 300

# Inclusive output, foreground, and component-count bounds.
_GEN_GREEN_COUNT_RANGE = (2, 5)
_GEN_LEFT_FG_COUNT_RANGE = (3, 6)
_GEN_RIGHT_FG_COUNT_RANGE = (4, 8)
_GEN_COMPONENT_COUNT_RANGE = (1, 3)


def _gen_component_count(panel: np.ndarray) -> int:
    components = find_components(
        panel, bg=_GEN_BG, connectivity=4, monochromatic=False, return_format="dict"
    )
    return len(components)


def _sample_output(cells: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return the target output and non-green cells that need panel foreground."""
    out = np.full(_GEN_PANEL_SHAPE, _GEN_BG, dtype=int)
    green_count = random.randint(*_GEN_GREEN_COUNT_RANGE)
    for r, c in cells[:green_count]:
        out[r, c] = _GEN_GREEN
    return out, cells[green_count:]


def _sample_panel_counts(active_count: int) -> tuple[int, int, int] | None:
    """Choose left/right/both foreground counts compatible with active cells."""
    left_min, left_max = _GEN_LEFT_FG_COUNT_RANGE
    right_min, right_max = _GEN_RIGHT_FG_COUNT_RANGE
    feasible_counts = []
    for left_count in range(left_min, left_max + 1):
        for right_count in range(right_min, right_max + 1):
            both_count = left_count + right_count - active_count
            if (
                left_count <= active_count
                and right_count <= active_count
                and 0 <= both_count <= min(left_count, right_count)
            ):
                feasible_counts.append((left_count, right_count, both_count))
    return random.choice(feasible_counts) if feasible_counts else None


def _build_panels(active, counts) -> tuple[np.ndarray, np.ndarray]:
    left_count, _, both_count = counts
    left = np.full(_GEN_PANEL_SHAPE, _GEN_BG, dtype=int)
    right = np.full(_GEN_PANEL_SHAPE, _GEN_BG, dtype=int)

    split_a = both_count
    split_b = left_count

    for r, c in active[:split_a]:
        left[r, c] = _GEN_LEFT
        right[r, c] = _GEN_RIGHT
    for r, c in active[split_a:split_b]:
        left[r, c] = _GEN_LEFT
    for r, c in active[split_b:]:
        right[r, c] = _GEN_RIGHT
    return left, right


def _gen_attempt() -> np.ndarray | None:
    cells = list(_GEN_PANEL_CELLS)
    random.shuffle(cells)

    out, active = _sample_output(cells)
    counts = _sample_panel_counts(len(active))
    if counts is None:
        return None

    left, right = _build_panels(active, counts)

    comp_min, comp_max = _GEN_COMPONENT_COUNT_RANGE
    if not (comp_min <= _gen_component_count(left) <= comp_max):
        return None
    if not (comp_min <= _gen_component_count(right) <= comp_max):
        return None

    inp = np.full((_GEN_HEIGHT, _GEN_PANEL_W * 2 + 1), _GEN_BG, dtype=int)
    inp[:, :_GEN_PANEL_W] = left
    inp[:, _GEN_PANEL_W] = _GEN_SEP
    inp[:, _GEN_PANEL_W + 1 :] = right
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
