from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, find_components, paste


def _frame_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rectangular_frame(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    return np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], _frame_mask((h, w)))


def _find_two_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_components = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        min_area=1,
    )
    if len(grey_components) != 2:
        raise AssertionError("expected exactly two grey components")

    out: list[tuple[int, int, int, int]] = []
    for comp in grey_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        mask = np.zeros_like(grid, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        if not _is_rectangular_frame(mask):
            raise AssertionError("grey component must be a single-pixel rectangular frame")
        out.append((r0, c0, r1, c1))
    out.sort(key=lambda box: box[0])
    return out


def _wave_shift_right(interior: np.ndarray) -> np.ndarray:
    out = interior.copy()
    h = interior.shape[0]
    for r in range(h):
        if (h - 1 - r) % 2 == 1:
            out[r] = np.roll(interior[r], 1)
    return out


def _wave_shift_left(interior: np.ndarray) -> np.ndarray:
    out = interior.copy()
    h = interior.shape[0]
    for r in range(h):
        if (h - 1 - r) % 2 == 1:
            out[r] = np.roll(interior[r], -1)
    return out


def _swap_two_colors(patch: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = patch.copy()
    out[patch == color_a] = color_b
    out[patch == color_b] = color_a
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a bottom-anchored row shift in the upper frame and a two-color swap in the lower frame, then mark same-color overlaps orange.

    Input structure:
    - A black background contains exactly two grey single-pixel rectangular frames, one above the other.
    - The two frame interiors have the same size and contain black gaps plus exactly two non-black, non-grey colors.
    - Colors outside the frames remain only black background and grey borders.

    Rule:
    - Identify the upper and lower grey frames by vertical position and extract their interiors.
    - In the upper interior, keep the bottom row fixed and shift every odd-numbered row above it one cell to the right with wraparound.
    - In the lower interior, swap the two non-black interior colors while leaving black cells unchanged.
    - Compare the two transformed interiors cell by cell; whenever both cells have the same non-black color, recolor that position to orange in both frames.
    - Keep the grey borders and everything outside the frames unchanged.

    Key insight:
    - The orange marks are determined only after combining two independent transforms: a positional shift in the upper frame and a color-value swap in the lower frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    top_box, bottom_box = _find_two_frames(g)
    top_inner = g[top_box[0] + 1 : top_box[2], top_box[1] + 1 : top_box[3]].copy()
    bottom_inner = g[bottom_box[0] + 1 : bottom_box[2], bottom_box[1] + 1 : bottom_box[3]].copy()
    assert top_inner.shape == bottom_inner.shape

    palette = sorted((set(map(int, top_inner.flatten())) | set(map(int, bottom_inner.flatten()))) - {Color.BLACK, Color.GREY})
    assert len(palette) == 2
    color_a, color_b = palette

    shifted_top = _wave_shift_right(top_inner)
    swapped_bottom = _swap_two_colors(bottom_inner, color_a, color_b)

    match_mask = (shifted_top == swapped_bottom) & (shifted_top != Color.BLACK)
    shifted_top[match_mask] = Color.ORANGE
    swapped_bottom[match_mask] = Color.ORANGE

    out = g.copy()
    out[top_box[0] + 1 : top_box[2], top_box[1] + 1 : top_box[3]] = shifted_top
    out[bottom_box[0] + 1 : bottom_box[2], bottom_box[1] + 1 : bottom_box[3]] = swapped_bottom
    return out


_INTERIOR_RANGE = (5, 7)
_FRAME_COLOR = Color.GREY
_BACKGROUND = Color.BLACK
_MATCH_COLOR = Color.ORANGE
_PALETTE_CHOICES = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.TEAL, Color.MAROON]
_MAX_ATTEMPTS = 400


def _make_structured_pattern(h: int, w: int, color_a: int, color_b: int) -> np.ndarray:
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for _ in range(60):
        kind = random.choice(("checker", "bands", "blocks", "diagonal"))
        phase = random.randint(0, 4)
        pattern = np.zeros((h, w), dtype=int)

        if kind == "checker":
            idx = (rows + cols + phase) % 3
            pattern[idx == 1] = color_a
            pattern[idx == 2] = color_b
        elif kind == "bands":
            motifs = (
                [color_a, 0, color_b],
                [color_a, color_b, 0],
                [0, color_a, color_b, color_b],
                [color_b, 0, color_a, color_a],
            )
            motif = np.array(random.choice(motifs), dtype=int)
            row_step = random.choice((0, 1, 2))
            for r in range(h):
                pattern[r] = motif[(np.arange(w) + phase + row_step * r) % len(motif)]
        elif kind == "blocks":
            idx = ((rows // 2) + (cols // 2) + phase) % 3
            pattern[idx == 1] = color_a
            pattern[idx == 2] = color_b
        else:
            idx = (2 * rows + cols + phase) % 4
            pattern[idx == 1] = color_a
            pattern[idx >= 2] = color_b

        for _ in range(random.randint(1, 3)):
            rr = random.randrange(h)
            cc = random.randrange(w)
            pattern[rr, cc] = random.choice((0, color_a, color_b))

        values = set(map(int, pattern.flatten()))
        if color_a not in values or color_b not in values:
            continue
        if int(np.count_nonzero(pattern)) < max(6, (h * w) // 3):
            continue
        return pattern
    raise RuntimeError("failed to build structured pattern")


def _ensure_palette_presence(pattern: np.ndarray, color_a: int, color_b: int) -> None:
    values = set(map(int, pattern.flatten()))
    if color_a not in values:
        rr, cc = random.randrange(pattern.shape[0]), random.randrange(pattern.shape[1])
        pattern[rr, cc] = color_a
    values = set(map(int, pattern.flatten()))
    if color_b not in values:
        free = [(r, c) for r in range(pattern.shape[0]) for c in range(pattern.shape[1]) if pattern[r, c] != color_a]
        rr, cc = random.choice(free)
        pattern[rr, cc] = color_b


def _force_match_structure(top_final: np.ndarray, bottom_final: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    h, w = top_final.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    min_matches = max(1, (h * w) // 8)
    max_matches = max(min_matches, (h * w) // 3)
    match_count = random.randint(min_matches, max_matches)
    match_mask = np.zeros((h, w), dtype=bool)

    for r, c in cells[:match_count]:
        if top_final[r, c] == 0:
            top_final[r, c] = random.choice((color_a, color_b))
        bottom_final[r, c] = top_final[r, c]
        match_mask[r, c] = True

    for r, c in cells[match_count:]:
        if top_final[r, c] != 0 and bottom_final[r, c] == top_final[r, c]:
            choices = [0, color_a, color_b]
            choices.remove(int(top_final[r, c]))
            bottom_final[r, c] = random.choice(choices)

    _ensure_palette_presence(top_final, color_a, color_b)
    _ensure_palette_presence(bottom_final, color_a, color_b)
    return match_mask


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    color_a, color_b = random.sample(_PALETTE_CHOICES, 2)
    h = random.randint(*_INTERIOR_RANGE)
    w = random.randint(*_INTERIOR_RANGE)

    top_final = _make_structured_pattern(h, w, color_a, color_b)
    bottom_final = _make_structured_pattern(h, w, color_a, color_b)
    match_mask = _force_match_structure(top_final, bottom_final, color_a, color_b)

    top_input = _wave_shift_left(top_final)
    bottom_input = _swap_two_colors(bottom_final, color_a, color_b)

    if np.array_equal(top_input, bottom_input):
        return None
    if np.array_equal(top_input, top_final):
        return None
    if np.array_equal(bottom_input, bottom_final):
        return None
    if not np.any(match_mask):
        return None
    if int(match_mask.sum()) >= h * w - 1:
        return None

    outer_h = h + 2
    outer_w = w + 2
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    gap = random.randint(2, 5)

    top_col = random.randint(left_margin, left_margin + 3)
    bottom_col = random.randint(left_margin, left_margin + 3)
    grid_w = max(top_col, bottom_col) + outer_w + right_margin
    bottom_row = top_margin + outer_h + gap
    grid_h = bottom_row + outer_h + bottom_margin
    if grid_h > 30 or grid_w > 30:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    top_patch = np.full((outer_h, outer_w), _BACKGROUND, dtype=int)
    bottom_patch = np.full((outer_h, outer_w), _BACKGROUND, dtype=int)
    top_patch[_frame_mask((outer_h, outer_w))] = _FRAME_COLOR
    bottom_patch[_frame_mask((outer_h, outer_w))] = _FRAME_COLOR
    top_patch[1:-1, 1:-1] = top_input
    bottom_patch[1:-1, 1:-1] = bottom_input

    paste(inp, top_patch, top_margin, top_col)
    paste(inp, bottom_patch, bottom_row, bottom_col)

    expected = inp.copy()
    expected[top_margin + 1 : top_margin + 1 + h, top_col + 1 : top_col + 1 + w] = top_final
    expected[bottom_row + 1 : bottom_row + 1 + h, bottom_col + 1 : bottom_col + 1 + w] = bottom_final
    expected[top_margin + 1 : top_margin + 1 + h, top_col + 1 : top_col + 1 + w][match_mask] = _MATCH_COLOR
    expected[bottom_row + 1 : bottom_row + 1 + h, bottom_col + 1 : bottom_col + 1 + w][match_mask] = _MATCH_COLOR

    try:
        if np.array_equal(main(inp), expected):
            return inp, expected
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        built = _build_candidate()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input")
