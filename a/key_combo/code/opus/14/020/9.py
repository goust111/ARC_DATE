from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
NOR_COLOR = 2
OR_COLOR = 3
SEPARATOR_COLOR = 5
LOWER_COLOR = 7
UPPER_COLOR = 9

PANEL_HEIGHT_RANGE = (3, 6)
PANEL_WIDTH_RANGE = (4, 8)
ACTIVE_RATIO_RANGE = (0.22, 0.55)
SHAPE_COUNT_RANGE = (2, 4)
GENERATION_ATTEMPTS = 1200
HISTORY_LIMIT = 256

_GEN_COUNTER = 0
_SEEN_INPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_INPUT_HISTORY: deque[tuple[tuple[int, int], bytes]] = deque()
_OUTPUT_HISTORY: deque[tuple[tuple[int, int], bytes]] = deque()


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.astype(np.uint8, copy=False).tobytes()


def _remember_signature(
    sig: tuple[tuple[int, int], bytes],
    seen: set[tuple[tuple[int, int], bytes]],
    history: deque[tuple[tuple[int, int], bytes]],
) -> None:
    if sig in seen:
        return
    seen.add(sig)
    history.append(sig)
    if len(history) > HISTORY_LIMIT:
        seen.discard(history.popleft())


def _full_color_rows(grid: np.ndarray, color: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_rows = _full_color_rows(g, SEPARATOR_COLOR)
    assert len(separator_rows) == 2, "expected exactly two separator rows"
    assert np.count_nonzero(g == SEPARATOR_COLOR) == 2 * g.shape[1], "separator color must appear only in the two rows"

    top_h = separator_rows[0]
    middle_h = separator_rows[1] - separator_rows[0] - 1
    bottom_h = g.shape[0] - separator_rows[1] - 1
    assert top_h == middle_h == bottom_h >= 3, "separator rows must split the grid into three equal-height panels"

    upper = g[: separator_rows[0], :]
    middle = g[separator_rows[0] + 1 : separator_rows[1], :]
    lower = g[separator_rows[1] + 1 :, :]
    assert upper.shape == middle.shape == lower.shape, "all three panels must align"

    assert set(map(int, np.unique(upper))) <= {BACKGROUND, UPPER_COLOR}, "upper panel must use only background and maroon"
    assert set(map(int, np.unique(middle))) <= {BACKGROUND}, "middle panel must start empty"
    assert set(map(int, np.unique(lower))) <= {BACKGROUND, LOWER_COLOR}, "lower panel must use only background and orange"
    return upper, middle, lower, separator_rows


def _solve_masks(upper_mask: np.ndarray, lower_mask: np.ndarray) -> tuple[bool, np.ndarray, int]:
    mirrored_upper = np.fliplr(upper_mask)
    is_match = np.array_equal(mirrored_upper, lower_mask)
    if is_match:
        return True, upper_mask | lower_mask, OR_COLOR
    return False, ~(upper_mask | lower_mask), NOR_COLOR


def _build_output(shape: tuple[int, int], separator_rows: list[int], active_mask: np.ndarray, color: int) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    out[separator_rows[0], :] = SEPARATOR_COLOR
    out[separator_rows[1], :] = SEPARATOR_COLOR
    middle = np.full(active_mask.shape, BACKGROUND, dtype=int)
    middle[active_mask] = color
    out[separator_rows[0] + 1 : separator_rows[1], :] = middle
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use horizontal mirror agreement between the outer masks to choose whether the middle panel shows OR or NOR.

    Input structure:
    - Two full-width grey rows split the grid into equal-height top, middle, and bottom panels.
    - The top panel contains a maroon-on-black mask, the bottom panel contains an orange-on-black mask, and the middle panel is all black.
    - The top and bottom panels have the same dimensions and are aligned cell by cell.

    Rule:
    - Read the maroon and orange cells as binary foreground masks.
    - Mirror the top mask left-to-right and compare that mirrored mask with the bottom mask.
    - If they match exactly, compute the cellwise OR of the original top and bottom masks and draw the active cells green in the middle panel.
    - Otherwise compute the cellwise NOR of the original top and bottom masks and draw the active cells red in the middle panel.
    - Clear the top and bottom panels to black and keep both grey separator rows unchanged.

    Key insight:
    - Mirror symmetry is the hidden decision node: it does not directly produce the answer, but it decides which logical operation must be applied.
    """
    g = np.asarray(grid, dtype=int)
    upper, _middle, lower, separator_rows = _extract_panels(g)
    upper_mask = upper == UPPER_COLOR
    lower_mask = lower == LOWER_COLOR

    _is_match, active_mask, color = _solve_masks(upper_mask, lower_mask)
    return _build_output(g.shape, separator_rows, active_mask, color)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _active_bounds(h: int, w: int) -> tuple[int, int]:
    cells = h * w
    lo = max(2, int(np.ceil(ACTIVE_RATIO_RANGE[0] * cells)))
    hi = min(cells - 2, int(np.floor(ACTIVE_RATIO_RANGE[1] * cells)))
    return lo, max(lo, hi)


def _apply_shape(mask: np.ndarray) -> None:
    h, w = mask.shape
    kind = random.choice(("rect", "hline", "vline", "cell"))
    if kind == "rect":
        sh = random.randint(1, min(3, h))
        sw = random.randint(1, min(3, w))
    elif kind == "hline":
        sh = 1
        sw = random.randint(2, min(4, w))
    elif kind == "vline":
        sh = random.randint(2, min(4, h))
        sw = 1
    else:
        sh = sw = 1

    r0 = random.randint(0, h - sh)
    c0 = random.randint(0, w - sw)
    mask[r0 : r0 + sh, c0 : c0 + sw] = True


def _toggle_random_cells(mask: np.ndarray, count: int) -> np.ndarray:
    out = mask.copy()
    positions = [(r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1])]
    random.shuffle(positions)
    for r, c in positions[:count]:
        out[r, c] = ~out[r, c]
    return out


def _sample_upper_mask(h: int, w: int) -> np.ndarray | None:
    min_active, max_active = _active_bounds(h, w)
    for _ in range(300):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(random.randint(*SHAPE_COUNT_RANGE)):
            _apply_shape(mask)
        if random.random() < 0.5:
            _apply_shape(mask)
        active = int(mask.sum())
        if not (min_active <= active <= max_active):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        comp_count = len(_components(mask))
        if not (1 <= comp_count <= 4):
            continue
        return mask
    return None


def _pair_uses_both_masks(upper_mask: np.ndarray, lower_mask: np.ndarray) -> bool:
    return bool(np.any(upper_mask & ~lower_mask) and np.any(lower_mask & ~upper_mask))


def _pair_is_interesting(upper_mask: np.ndarray, lower_mask: np.ndarray, expect_match: bool) -> bool:
    if int(upper_mask.sum()) < 2 or int(lower_mask.sum()) < 2:
        return False
    if np.array_equal(upper_mask, lower_mask):
        return False
    if not _pair_uses_both_masks(upper_mask, lower_mask):
        return False

    is_match, active_mask, _color = _solve_masks(upper_mask, lower_mask)
    if is_match != expect_match:
        return False

    cells = upper_mask.size
    active_count = int(active_mask.sum())
    if active_count < 2 or active_count > cells - 2:
        return False
    if expect_match and np.array_equal(active_mask, upper_mask):
        return False
    if expect_match and np.array_equal(active_mask, lower_mask):
        return False
    return True


def _make_near_mismatch(upper_mask: np.ndarray) -> np.ndarray | None:
    base = np.fliplr(upper_mask)
    for _ in range(200):
        toggle_count = random.randint(1, 3)
        lower = _toggle_random_cells(base, toggle_count)
        if np.array_equal(lower, base):
            continue
        comp_count = len(_components(lower))
        if not (1 <= comp_count <= 5):
            continue
        if _pair_is_interesting(upper_mask, lower, expect_match=False):
            return lower
    return None


def _make_far_mismatch(upper_mask: np.ndarray) -> np.ndarray | None:
    h, w = upper_mask.shape
    for _ in range(300):
        lower = _sample_upper_mask(h, w)
        if lower is None:
            continue
        if np.array_equal(lower, np.fliplr(upper_mask)):
            continue
        if _pair_is_interesting(upper_mask, lower, expect_match=False):
            return lower
    return None


def _render_input(upper_mask: np.ndarray, lower_mask: np.ndarray) -> np.ndarray:
    h, w = upper_mask.shape
    grid = np.full((3 * h + 2, w), BACKGROUND, dtype=int)
    upper_panel = np.full((h, w), BACKGROUND, dtype=int)
    lower_panel = np.full((h, w), BACKGROUND, dtype=int)
    upper_panel[upper_mask] = UPPER_COLOR
    lower_panel[lower_mask] = LOWER_COLOR
    grid[:h, :] = upper_panel
    grid[h, :] = SEPARATOR_COLOR
    grid[2 * h + 1, :] = SEPARATOR_COLOR
    grid[2 * h + 2 :, :] = lower_panel
    return grid


def _expected_output_from_input(grid: np.ndarray) -> tuple[np.ndarray, bool]:
    upper, _middle, lower, separator_rows = _extract_panels(grid)
    is_match, active_mask, color = _solve_masks(upper == UPPER_COLOR, lower == LOWER_COLOR)
    return _build_output(grid.shape, separator_rows, active_mask, color), is_match


def _validate_generated_case(grid: np.ndarray, expect_match: bool) -> bool:
    try:
        upper, middle, lower, _separator_rows = _extract_panels(grid)
        if np.any(middle != BACKGROUND):
            return False
        upper_mask = upper == UPPER_COLOR
        lower_mask = lower == LOWER_COLOR
        if np.array_equal(upper_mask, np.fliplr(upper_mask)):
            return False
        if not _pair_is_interesting(upper_mask, lower_mask, expect_match):
            return False
        expected, is_match = _expected_output_from_input(grid)
        if is_match != expect_match:
            return False
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False
        output_colors = set(map(int, np.unique(solved)))
        if expect_match and OR_COLOR not in output_colors:
            return False
        if (not expect_match) and NOR_COLOR not in output_colors:
            return False
        return True
    except AssertionError:
        return False


def _branch_mode() -> tuple[bool, str]:
    global _GEN_COUNTER
    expect_match = (_GEN_COUNTER % 2) == 0
    if expect_match:
        return True, "match"
    return False, "near" if (_GEN_COUNTER % 4) == 1 else "far"


def _gen_attempt() -> np.ndarray | None:
    expect_match, mismatch_mode = _branch_mode()
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)

    upper_mask = _sample_upper_mask(panel_h, panel_w)
    if upper_mask is None:
        return None

    if expect_match:
        lower_mask = np.fliplr(upper_mask)
        if not _pair_is_interesting(upper_mask, lower_mask, expect_match=True):
            return None
    elif mismatch_mode == "near":
        lower_mask = _make_near_mismatch(upper_mask)
        if lower_mask is None:
            return None
    else:
        lower_mask = _make_far_mismatch(upper_mask)
        if lower_mask is None:
            return None

    grid = _render_input(upper_mask, lower_mask)
    if not _validate_generated_case(grid, expect_match):
        return None

    out = main(grid)
    input_sig = _grid_signature(grid)
    output_sig = _grid_signature(out)
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        out = main(grid)
        _remember_signature(_grid_signature(grid), _SEEN_INPUT_SIGNATURES, _INPUT_HISTORY)
        _remember_signature(_grid_signature(out), _SEEN_OUTPUT_SIGNATURES, _OUTPUT_HISTORY)
        _GEN_COUNTER += 1
        return grid
    raise RuntimeError("failed to generate a valid mirror-branch logical-mask input")
