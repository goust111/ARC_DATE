from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

_BOOL_OP_BY_COLOR = {
    GREY: "and",
    YELLOW: "xor",
}
_INPUT_COLOR_POOL = (BLACK, BLUE, RED, PINK, ORANGE, TEAL, MAROON)
_BRANCHES = (
    (True, "and", "xor"),
    (True, "xor", "and"),
    (False, "and", "xor"),
    (False, "xor", "and"),
)
_PATTERN_BITS = tuple(itertools.product((0, 1), repeat=3))
_RNG = np.random.default_rng()
_CASE_COUNTER = 0
_MAX_GENERATION_ATTEMPTS = 500


def _solid_band_color(band: np.ndarray) -> int | None:
    if band.size == 0:
        return None
    color = int(band[0, 0])
    if np.all(band == color):
        return color
    return None


def _apply_bool(left: np.ndarray, right: np.ndarray, op: str) -> np.ndarray:
    if op == "and":
        return left & right
    if op == "xor":
        return np.logical_xor(left, right)
    raise ValueError(f"unknown op {op}")


def _apply_scalar(left: int, right: int, op: str) -> int:
    if op == "and":
        return left & right
    if op == "xor":
        return left ^ right
    raise ValueError(f"unknown op {op}")


def _flip_op(op: str) -> str:
    return "xor" if op == "and" else "and"


def _final_mask(
    top: np.ndarray,
    middle: np.ndarray,
    bottom: np.ndarray,
    top_op: str,
    bottom_op: str,
    top_is_thick: bool,
) -> np.ndarray:
    if top_is_thick:
        return _apply_bool(_apply_bool(top, middle, top_op), bottom, bottom_op)
    return _apply_bool(top, _apply_bool(middle, bottom, bottom_op), top_op)


def _final_scalar(pattern_id: int, top_op: str, bottom_op: str, top_is_thick: bool) -> int:
    a = (pattern_id >> 2) & 1
    b = (pattern_id >> 1) & 1
    c = pattern_id & 1
    if top_is_thick:
        return _apply_scalar(_apply_scalar(a, b, top_op), c, bottom_op)
    return _apply_scalar(a, _apply_scalar(b, c, bottom_op), top_op)


def _parse_layers(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, str, str, bool]:
    height, width = map(int, grid.shape)
    assert width > 0 and height >= 6
    assert (height - 3) % 3 == 0, "grid height must be three equal layers plus 3 divider rows"
    layer_h = (height - 3) // 3
    assert layer_h >= 1

    candidates = []
    for top_thickness, bottom_thickness in ((1, 2), (2, 1)):
        top_band = grid[layer_h : layer_h + top_thickness, :]
        middle_start = layer_h + top_thickness
        middle_end = middle_start + layer_h
        bottom_band = grid[middle_end : middle_end + bottom_thickness, :]
        bottom_start = middle_end + bottom_thickness
        if bottom_start + layer_h != height:
            continue
        top_color = _solid_band_color(top_band)
        bottom_color = _solid_band_color(bottom_band)
        if top_color not in _BOOL_OP_BY_COLOR or bottom_color not in _BOOL_OP_BY_COLOR:
            continue
        top = grid[:layer_h, :]
        middle = grid[middle_start:middle_end, :]
        bottom = grid[bottom_start:, :]
        shared_colors = set(map(int, np.unique(top)))
        shared_colors &= set(map(int, np.unique(middle)))
        shared_colors &= set(map(int, np.unique(bottom)))
        if len(shared_colors) != 1:
            continue
        bg = int(next(iter(shared_colors)))
        if bg in (GREEN, GREY, YELLOW):
            continue
        layer_colors = []
        ok = True
        for layer in (top, middle, bottom):
            fg_colors = set(map(int, np.unique(layer))) - {bg}
            if len(fg_colors) != 1:
                ok = False
                break
            layer_colors.append(int(next(iter(fg_colors))))
        if not ok:
            continue
        if GREEN in layer_colors:
            continue
        candidates.append((top, middle, bottom, bg, _BOOL_OP_BY_COLOR[top_color], _BOOL_OP_BY_COLOR[bottom_color], top_thickness == 2))

    assert len(candidates) == 1, "expected a unique valid divider layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold three vertically stacked masks into one, using divider color as the boolean operator and divider thickness as precedence.

    Input structure:
    - The grid contains three equal-sized mask layers stacked vertically.
    - Between the layers are two full-width solid divider bands: one is 1 row thick and the other is 2 rows thick.
    - Divider colors are grey and yellow, encoding intersection and symmetric difference respectively.
    - Each mask uses one shared background color plus exactly one layer-specific foreground color.

    Rule:
    - Interpret grey as boolean intersection and yellow as boolean symmetric difference on foreground membership.
    - First combine the two masks adjacent to the thicker divider band.
    - Then combine that intermediate mask with the remaining layer using the operator encoded by the thinner divider.
    - Output one mask-sized grid: paint green where the final boolean mask is foreground, and keep the shared background color elsewhere.

    Key insight:
    - The separator bands are the expression itself: color says which operation to apply, while thickness says which pair must be evaluated first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, middle, bottom, bg, top_op, bottom_op, top_is_thick = _parse_layers(g)
    top_mask = top != bg
    middle_mask = middle != bg
    bottom_mask = bottom != bg
    final = _final_mask(top_mask, middle_mask, bottom_mask, top_op, bottom_op, top_is_thick)

    out = np.full(top.shape, bg, dtype=int)
    out[final] = GREEN
    return out


def _pick_colors(rng: np.random.Generator) -> tuple[int, int, int, int]:
    chosen = rng.choice(_INPUT_COLOR_POOL, size=4, replace=False)
    return tuple(int(x) for x in chosen)


def _pattern_mask(pattern_grid: np.ndarray, bit_index: int) -> np.ndarray:
    return ((pattern_grid >> bit_index) & 1).astype(bool)


def _difference_patterns(
    top_op: str,
    bottom_op: str,
    top_is_thick: bool,
    alt_top_op: str,
    alt_bottom_op: str,
    alt_top_is_thick: bool,
) -> set[int]:
    return {
        pid
        for pid in range(8)
        if _final_scalar(pid, top_op, bottom_op, top_is_thick)
        != _final_scalar(pid, alt_top_op, alt_bottom_op, alt_top_is_thick)
    }


def _required_pattern_ids(top_op: str, bottom_op: str, top_is_thick: bool, rng: np.random.Generator) -> list[int]:
    diff_sets = [
        _difference_patterns(top_op, bottom_op, top_is_thick, top_op, bottom_op, not top_is_thick),
        _difference_patterns(top_op, bottom_op, top_is_thick, _flip_op(top_op), bottom_op, top_is_thick),
        _difference_patterns(top_op, bottom_op, top_is_thick, top_op, _flip_op(bottom_op), top_is_thick),
    ]
    required = {4, 2, 1}
    overlap_choices = [3, 5, 6, 7]
    required.add(int(rng.choice(overlap_choices)))
    for diff in diff_sets:
        choices = sorted(diff)
        required.add(int(rng.choice(choices)))
    return list(required)


def _place_block(
    pattern_grid: np.ndarray,
    pattern_id: int,
    rng: np.random.Generator,
    *,
    max_h: int,
    max_w: int,
    min_h: int = 1,
    min_w: int = 1,
) -> bool:
    height, width = pattern_grid.shape
    max_h = min(max_h, height)
    max_w = min(max_w, width)
    min_h = min(min_h, max_h)
    min_w = min(min_w, max_w)
    for _ in range(80):
        block_h = int(rng.integers(min_h, max_h + 1))
        block_w = int(rng.integers(min_w, max_w + 1))
        r0 = int(rng.integers(0, height - block_h + 1))
        c0 = int(rng.integers(0, width - block_w + 1))
        region = pattern_grid[r0 : r0 + block_h, c0 : c0 + block_w]
        if np.any(region != 0):
            continue
        region[:] = pattern_id
        return True
    return False


def _pairwise_shortcuts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Iterable[np.ndarray]:
    yield a
    yield b
    yield c
    yield a & b
    yield a & c
    yield b & c
    yield np.logical_xor(a, b)
    yield np.logical_xor(a, c)
    yield np.logical_xor(b, c)
    yield a | b
    yield a | c
    yield b | c


def _semantic_ok(pattern_grid: np.ndarray, top_op: str, bottom_op: str, top_is_thick: bool) -> bool:
    a = _pattern_mask(pattern_grid, 2)
    b = _pattern_mask(pattern_grid, 1)
    c = _pattern_mask(pattern_grid, 0)
    area = int(pattern_grid.size)
    fills = [int(mask.sum()) for mask in (a, b, c)]
    min_fill = max(3, area // 7)
    max_fill = int(area * 0.7)
    if any(fill < min_fill or fill > max_fill for fill in fills):
        return False

    final = _final_mask(a, b, c, top_op, bottom_op, top_is_thick)
    output_fill = int(final.sum())
    if output_fill < max(2, area // 10) or output_fill > int(area * 0.6):
        return False

    wrong_group = _final_mask(a, b, c, top_op, bottom_op, not top_is_thick)
    wrong_top = _final_mask(a, b, c, _flip_op(top_op), bottom_op, top_is_thick)
    wrong_bottom = _final_mask(a, b, c, top_op, _flip_op(bottom_op), top_is_thick)
    if min(
        int(np.count_nonzero(final != wrong_group)),
        int(np.count_nonzero(final != wrong_top)),
        int(np.count_nonzero(final != wrong_bottom)),
    ) < 2:
        return False

    if any(np.array_equal(final, shortcut) for shortcut in _pairwise_shortcuts(a, b, c)):
        return False

    if np.array_equal(a, b) or np.array_equal(a, c) or np.array_equal(b, c):
        return False

    if any(
        np.array_equal(
            final,
            _final_mask(
                a if idx != 0 else np.zeros_like(a),
                b if idx != 1 else np.zeros_like(b),
                c if idx != 2 else np.zeros_like(c),
                top_op,
                bottom_op,
                top_is_thick,
            ),
        )
        for idx in range(3)
    ):
        return False

    pattern_ids = set(map(int, pattern_grid.ravel()))
    if len(pattern_ids - {0}) < 5:
        return False
    if not pattern_ids & {3, 5, 6, 7}:
        return False
    if not pattern_ids & {1, 2, 4}:
        return False
    return True


def _build_pattern_grid(height: int, width: int, top_op: str, bottom_op: str, top_is_thick: bool, rng: np.random.Generator) -> np.ndarray | None:
    pattern_grid = np.zeros((height, width), dtype=int)
    required = _required_pattern_ids(top_op, bottom_op, top_is_thick, rng)
    rng.shuffle(required)

    for pattern_id in required:
        if not _place_block(pattern_grid, pattern_id, rng, max_h=min(2, height), max_w=min(3, width)):
            return None

    extra_blocks = int(rng.integers(3, 7))
    extra_ids = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
    weights = np.array([1.0, 1.0, 1.3, 1.0, 1.3, 1.3, 0.8], dtype=float)
    weights /= weights.sum()
    for _ in range(extra_blocks):
        pattern_id = int(rng.choice(extra_ids, p=weights))
        if not _place_block(
            pattern_grid,
            pattern_id,
            rng,
            min_h=1,
            min_w=1,
            max_h=min(3, height),
            max_w=min(4, width),
        ):
            break

    if np.count_nonzero(pattern_grid) > int(pattern_grid.size * 0.72):
        return None
    if not _semantic_ok(pattern_grid, top_op, bottom_op, top_is_thick):
        return None
    return pattern_grid


def _render_layer(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    layer = np.full(mask.shape, bg, dtype=int)
    layer[mask] = fg
    return layer


def _assemble_input(
    pattern_grid: np.ndarray,
    bg: int,
    top_fg: int,
    middle_fg: int,
    bottom_fg: int,
    top_op: str,
    bottom_op: str,
    top_is_thick: bool,
) -> np.ndarray:
    top_mask = _pattern_mask(pattern_grid, 2)
    middle_mask = _pattern_mask(pattern_grid, 1)
    bottom_mask = _pattern_mask(pattern_grid, 0)
    top_layer = _render_layer(top_mask, bg, top_fg)
    middle_layer = _render_layer(middle_mask, bg, middle_fg)
    bottom_layer = _render_layer(bottom_mask, bg, bottom_fg)

    top_divider_h = 2 if top_is_thick else 1
    bottom_divider_h = 1 if top_is_thick else 2
    top_divider_color = GREY if top_op == "and" else YELLOW
    bottom_divider_color = GREY if bottom_op == "and" else YELLOW
    top_divider = np.full((top_divider_h, pattern_grid.shape[1]), top_divider_color, dtype=int)
    bottom_divider = np.full((bottom_divider_h, pattern_grid.shape[1]), bottom_divider_color, dtype=int)
    return np.vstack([top_layer, top_divider, middle_layer, bottom_divider, bottom_layer])


def _build_case(branch_index: int, rng: np.random.Generator) -> np.ndarray | None:
    top_is_thick, top_op, bottom_op = _BRANCHES[branch_index]
    layer_h = int(rng.integers(4, 10))
    width = int(rng.integers(5, 11))
    pattern_grid = _build_pattern_grid(layer_h, width, top_op, bottom_op, top_is_thick, rng)
    if pattern_grid is None:
        return None

    bg, top_fg, middle_fg, bottom_fg = _pick_colors(rng)
    inp = _assemble_input(pattern_grid, bg, top_fg, middle_fg, bottom_fg, top_op, bottom_op, top_is_thick)
    expected = main(inp)
    if expected.shape != pattern_grid.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    global _CASE_COUNTER

    branch_index = _CASE_COUNTER % len(_BRANCHES)
    _CASE_COUNTER += 1
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        inp = _build_case(branch_index, _RNG)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a non-degenerate divider-expression puzzle")
