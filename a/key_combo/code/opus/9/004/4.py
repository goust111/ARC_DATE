from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, is_contiguous, paste  # noqa: E402


BG = 0
PLACEHOLDER_COLOR = 1
TEMPLATE_COLORS = (3, 4, 7, 8, 9)
CONTROL_COLORS = (2, 3, 4, 6, 8)
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (15, 22)
GEN_ATTEMPTS = 400
CELL_PADDING = 1


def _normalized_shape(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return int(mask.shape[0]), int(mask.shape[1]), tuple(int(x) for x in mask.astype(int).ravel())


def _top_row(comp: dict) -> int:
    return int(comp["coords"][:, 0].min())


def _control_components(components: list[dict]) -> list[dict]:
    singletons = [comp for comp in components if int(comp["color"]) != PLACEHOLDER_COLOR and int(comp["area"]) == 1]
    assert singletons, "expected at least one non-blue singleton control pixel"
    by_col: dict[int, list[dict]] = {}
    for comp in singletons:
        _r, c = map(int, comp["coords"][0])
        by_col.setdefault(c, []).append(comp)
    max_count = max(len(items) for items in by_col.values())
    candidate_cols = [c for c, items in by_col.items() if len(items) == max_count]
    control_col = max(candidate_cols)
    return by_col[control_col]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Verify each blue placeholder by normalized shape matching, then recolor it from the control pixel on its top row.

    Input structure:
    - The background is black.
    - Blue 4-connected placeholder objects appear in the middle of the grid.
    - Non-blue multi-pixel template objects appear elsewhere, and non-blue singleton control pixels sit in one right-side control column.

    Rule:
    - Extract every object's normalized 4-connected shape mask from its bounding box.
    - For each blue placeholder, check whether exactly one non-blue multi-pixel template has the same normalized shape.
    - If the match is unique, recolor the whole placeholder with the color of the control-column pixel on the placeholder's highest row; otherwise erase the placeholder to black.
    - Leave all templates and control pixels unchanged.

    Key insight:
    - Shape matching is only a legitimacy test: it decides whether a placeholder may receive color, but the actual output color comes from the independent row-aligned control pixel, not from the matched template.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    controls = _control_components(components)
    control_by_row: dict[int, int] = {}
    for comp in controls:
        r, _c = map(int, comp["coords"][0])
        assert r not in control_by_row, "control rows must be unique"
        control_by_row[r] = int(comp["color"])

    placeholders = [comp for comp in components if int(comp["color"]) == PLACEHOLDER_COLOR]
    templates = [
        comp
        for comp in components
        if int(comp["color"]) != PLACEHOLDER_COLOR and int(comp["area"]) > 1
    ]

    template_keys = [_mask_key(_normalized_shape(comp)) for comp in templates]
    out = g.copy()
    for comp in placeholders:
        key = _mask_key(_normalized_shape(comp))
        matched = sum(template_key == key for template_key in template_keys)
        row = _top_row(comp)
        color = control_by_row.get(row)
        if matched == 1 and color is not None:
            out[comp["coords"][:, 0], comp["coords"][:, 1]] = int(color)
        else:
            out[comp["coords"][:, 0], comp["coords"][:, 1]] = bg
    return out


def _build_shape_pool() -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3)):
        for area in range(3, min(8, h * w) + 1):
            for cells in itertools.combinations(range(h * w), area):
                mask = np.zeros((h, w), dtype=int)
                for idx in cells:
                    mask[idx // w, idx % w] = 1
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if not is_contiguous(mask, background=0, connectivity=4):
                    continue
                key = _mask_key(mask.astype(bool))
                if key not in seen:
                    seen.add(key)
                    pool.append(mask.astype(bool))
    assert len(pool) >= 8, "shape pool should contain enough distinct shapes"
    return pool


SHAPE_POOL = _build_shape_pool()


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(reserved: np.ndarray, mask: np.ndarray, r0: int, c0: int, *, padding: int = CELL_PADDING) -> bool:
    h, w = mask.shape
    H, W = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, mask: np.ndarray, r0: int, c0: int, *, padding: int = CELL_PADDING) -> None:
    h, w = mask.shape
    H, W = reserved.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    reserved[rr0:rr1, cc0:cc1] = True


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    *,
    row_values: list[int],
    col_range: range,
) -> tuple[int, int] | None:
    choices = [
        (r0, c0)
        for r0 in row_values
        for c0 in col_range
        if _can_place(reserved, mask, r0, c0)
    ]
    if not choices:
        return None
    r0, c0 = random.choice(choices)
    paste(grid, _shape_patch(mask, color), r0, c0, transparent=BG)
    _reserve(reserved, mask, r0, c0)
    return int(r0), int(c0)


def _choose_template_colors(control_colors: list[int]) -> list[int]:
    chosen: list[int] = []
    used: set[int] = set()
    for control_color in control_colors:
        options = [color for color in TEMPLATE_COLORS if color != control_color and color not in used]
        if not options:
            options = [color for color in TEMPLATE_COLORS if color != control_color]
        color = random.choice(options)
        chosen.append(int(color))
        used.add(int(color))
    return chosen


def _example_semantics(
    shapes: list[np.ndarray],
    valid_count: int,
    has_invalid: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    valid_shapes = shapes[:valid_count]
    invalid_shapes = shapes[valid_count:] if has_invalid else []
    return valid_shapes, invalid_shapes


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((H, W), dtype=int)
    expected = np.zeros_like(grid)
    reserved = np.zeros((H, W), dtype=bool)

    valid_count = random.choice((2, 3))
    has_invalid = valid_count == 2 and random.random() < 0.55
    total_placeholders = valid_count + int(has_invalid)
    shapes = random.sample(SHAPE_POOL, total_placeholders)
    valid_shapes, invalid_shapes = _example_semantics(shapes, valid_count, has_invalid)

    placeholder_rows: list[int] = []
    placeholder_infos: list[tuple[np.ndarray, tuple[int, int], int | None]] = []

    control_col = W - 2
    left_cols = range(1, 4)
    middle_start = 6
    middle_end = control_col - 3
    if middle_start > middle_end:
        return None
    middle_cols = range(middle_start, middle_end + 1)

    control_colors = random.sample(CONTROL_COLORS, total_placeholders)
    template_colors = _choose_template_colors(control_colors[:valid_count])

    for template_mask, template_color in zip(valid_shapes, template_colors):
        template_rows = list(range(1, H - template_mask.shape[0]))
        pos = _place_object(
            grid,
            reserved,
            template_mask,
            template_color,
            row_values=template_rows,
            col_range=left_cols,
        )
        if pos is None:
            return None
        paste(expected, _shape_patch(template_mask, template_color), pos[0], pos[1], transparent=BG)

    all_placeholder_shapes = valid_shapes + invalid_shapes
    for idx, placeholder_mask in enumerate(all_placeholder_shapes):
        candidate_rows = [
            r0
            for r0 in range(1, H - placeholder_mask.shape[0])
            if r0 not in placeholder_rows and _can_place(reserved, placeholder_mask, r0, middle_start)
        ]
        if not candidate_rows:
            return None
        pos = _place_object(
            grid,
            reserved,
            placeholder_mask,
            PLACEHOLDER_COLOR,
            row_values=candidate_rows,
            col_range=middle_cols,
        )
        if pos is None:
            return None
        row = int(pos[0])
        placeholder_rows.append(row)
        match_index = idx if idx < valid_count else None
        placeholder_infos.append((placeholder_mask, pos, match_index))
        paste(expected, _shape_patch(placeholder_mask, PLACEHOLDER_COLOR), pos[0], pos[1], transparent=BG)

    for color, row in zip(control_colors, placeholder_rows):
        if reserved[row, control_col]:
            return None
        grid[row, control_col] = int(color)
        expected[row, control_col] = int(color)
        reserved[row, control_col] = True

    for idx, (_mask, (r0, c0), match_index) in enumerate(placeholder_infos):
        color = control_colors[idx]
        comp_patch = _shape_patch(_mask, color if match_index is not None else BG)
        paste(expected, comp_patch, r0, c0, transparent=BG)

    if has_invalid:
        invalid_mask, (invalid_r, invalid_c), invalid_match = placeholder_infos[-1]
        assert invalid_match is None
        invalid_patch = np.zeros(invalid_mask.shape, dtype=int)
        paste(expected, invalid_patch, invalid_r, invalid_c, transparent=None)

    if not _validate_generated_example(
        grid,
        expected,
        valid_count=valid_count,
        total_placeholders=total_placeholders,
        has_invalid=has_invalid,
    ):
        return None
    return grid, expected


def _validate_generated_example(
    inp: np.ndarray,
    expected: np.ndarray,
    *,
    valid_count: int,
    total_placeholders: int,
    has_invalid: bool,
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = int(get_bg(inp))
    components = find_components(inp, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    placeholders = [comp for comp in components if int(comp["color"]) == PLACEHOLDER_COLOR]
    templates = [comp for comp in components if int(comp["color"]) != PLACEHOLDER_COLOR and int(comp["area"]) > 1]
    controls = _control_components(components)

    if len(placeholders) != total_placeholders or len(templates) != valid_count or len(controls) != total_placeholders:
        return False

    control_by_row = {int(comp["coords"][0, 0]): int(comp["color"]) for comp in controls}
    template_keys = [_mask_key(_normalized_shape(comp)) for comp in templates]
    matched_placeholders = 0
    erased_placeholders = 0
    color_mismatch_seen = False

    template_color_by_key = {
        _mask_key(_normalized_shape(comp)): int(comp["color"])
        for comp in templates
    }

    for comp in placeholders:
        row = _top_row(comp)
        matches = sum(_mask_key(_normalized_shape(comp)) == key for key in template_keys)
        out_vals = set(int(v) for v in actual[comp["coords"][:, 0], comp["coords"][:, 1]])
        if matches == 1:
            matched_placeholders += 1
            if len(out_vals) != 1 or next(iter(out_vals)) != control_by_row[row]:
                return False
            if template_color_by_key[_mask_key(_normalized_shape(comp))] != control_by_row[row]:
                color_mismatch_seen = True
        else:
            erased_placeholders += 1
            if out_vals != {BG}:
                return False

    if matched_placeholders != valid_count:
        return False
    if has_invalid != (erased_placeholders == 1):
        return False
    return color_mismatch_seen


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _generate_case()
        if result is None:
            continue
        inp, _expected = result
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
