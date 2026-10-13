from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, get_bg, paste


BG = Color.BLACK
RED = Color.RED
DIVIDER = Color.GREY
TEMPLATE_PALETTE = tuple(c for c in Color.NOT_BLACK if c not in {RED, DIVIDER})
GEN_ATTEMPTS = 500


def _trace_right_hits(height: int, width: int, start_col: int) -> list[int]:
    """Simulate the invisible up-right ray inside a width-wide corridor."""
    assert height >= 1 and width >= 2
    assert 0 <= start_col < width - 1, "generator/solver disallows starting on the right wall"

    r = height - 1
    c = int(start_col)
    dc = 1
    hit_rows: list[int] = []

    while r >= 0:
        if c == width - 1:
            hit_rows.append(int(r))

        nr = r - 1
        if nr < 0:
            break

        nc = c + dc
        if nc < 0 or nc >= width:
            dc = -dc
            nc = c + dc
        r, c = nr, nc

    return hit_rows


def _signature(height: int, width: int, slot_h: int, start_col: int) -> tuple[int, ...]:
    return tuple(sorted({row // slot_h for row in _trace_right_hits(height, width, start_col)}))


def _extract_objects(grid: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    assert bg == BG, "expected black background"

    divider_cols = np.where(np.all(g == DIVIDER, axis=0))[0]
    assert divider_cols.size == 1, "expected exactly one full-height grey divider column"
    divider_col = int(divider_cols[0])
    assert 0 < divider_col < width - 2, "divider must leave room for template and right corridor"

    template_mask = (g != BG) & (g != DIVIDER) & (g != RED)
    template_box = bbox(template_mask)
    assert template_box is not None, "expected one colored template left of the divider"
    tr0, tc0, tr1, tc1 = map(int, template_box)
    template = crop_bbox(g, template_box)
    assert tc1 == divider_col - 1, "template must touch the divider on its right edge"
    assert np.all(template != BG), "template bbox must be fully occupied"
    assert np.all(template != DIVIDER), "template cannot contain divider color"
    assert np.all(template != RED), "template cannot contain red"
    assert len(set(map(int, template.ravel()))) >= 2, "template must be multi-colored"

    corridor = g[:, divider_col + 1 :]
    assert corridor.shape[1] == template.shape[1], "slot width must match template width"
    assert np.all((corridor == BG) | (corridor == RED)), "right corridor may only contain bg plus one red control"

    red_coords = np.argwhere(g == RED)
    assert red_coords.shape[0] == 1, "expected exactly one red control pixel"
    rr, rc = map(int, red_coords[0])
    assert rr == height - 1, "red control must be on the bottom row"
    assert rc >= divider_col + 1, "red control must be in the right corridor"

    local_start = rc - (divider_col + 1)
    assert 0 <= local_start < corridor.shape[1] - 1, "red control cannot start on the right wall"

    return divider_col, template, (tr0, tc0), local_start


def _activated_slots(height: int, slot_h: int, corridor_w: int, start_col: int) -> tuple[int, ...]:
    assert height % slot_h == 0, "grid height must split evenly into slots"
    slot_count = height // slot_h
    active = _signature(height, corridor_w, slot_h, start_col)
    assert active, "ray must hit the right wall at least once"
    assert all(0 <= idx < slot_count for idx in active)
    return active


def _assert_unique_signature(height: int, slot_h: int, corridor_w: int, actual_start: int) -> tuple[int, ...]:
    signatures: dict[tuple[int, ...], list[int]] = {}
    for start_col in range(corridor_w - 1):
        sig = _signature(height, corridor_w, slot_h, start_col)
        signatures.setdefault(sig, []).append(start_col)

    actual_sig = _activated_slots(height, slot_h, corridor_w, actual_start)
    assert signatures.get(actual_sig) == [actual_start], "red control must uniquely determine the activated slots"
    return actual_sig


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bounce the bottom red control off the divider and right wall so each right-wall hit selects a slot to receive the left template.

    Input structure:
    - The background is black, with exactly one full-height grey divider column.
    - A fully occupied multi-color rectangular template touches the divider on its left side.
    - The region to the right of the divider is template-width wide, splits evenly into vertical slots whose height equals the template height, and contains only background plus one red bottom control pixel.

    Rule:
    - Starting from the red control, move one cell up and one cell sideways each step, initially toward the upper right.
    - When the next step would leave the corridor through the divider side or the right edge, reverse only the horizontal direction and continue upward; the path itself remains invisible.
    - Every visited cell on the rightmost column activates the slot containing that row; remove the red control and paste the left template into every activated slot.

    Key insight:
    - The output depends on an invisible reflected diagonal trajectory: only its right-wall impact rows matter, because those rows identify which empty slot rectangles receive the unchanged template.
    """
    divider_col, template, _template_pos, local_start = _extract_objects(grid)
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    slot_h, slot_w = map(int, template.shape)
    assert width - divider_col - 1 == slot_w

    active_slots = _assert_unique_signature(height, slot_h, slot_w, local_start)

    out = g.copy()
    out[out == RED] = BG
    out[:, divider_col + 1 :] = BG
    for slot_idx in active_slots:
        paste(out, template, slot_idx * slot_h, divider_col + 1)
    return out


def _random_template(rng: random.Random, h: int, w: int) -> np.ndarray:
    n_colors = rng.randint(2, min(4, len(TEMPLATE_PALETTE)))
    colors = rng.sample(TEMPLATE_PALETTE, n_colors)
    flat = [rng.choice(colors) for _ in range(h * w)]
    for idx, color in enumerate(colors):
        flat[idx] = color
    rng.shuffle(flat)
    patch = np.array(flat, dtype=int).reshape(h, w)

    if len(set(map(int, patch.ravel()))) < 2:
        patch[0, 0] = colors[0]
        patch[-1, -1] = colors[1]

    if np.all(patch == patch[0, 0]):
        patch[-1, -1] = colors[1]
    return patch


def _reference_output(
    template: np.ndarray,
    height: int,
    left_margin: int,
    start_col: int,
    template_top: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    slot_h, slot_w = map(int, template.shape)
    divider_col = left_margin + slot_w
    width = divider_col + 1 + slot_w

    inp = np.full((height, width), BG, dtype=int)
    inp[:, divider_col] = DIVIDER
    paste(inp, template, template_top, left_margin)
    inp[height - 1, divider_col + 1 + start_col] = RED

    active_slots = _activated_slots(height, slot_h, slot_w, start_col)

    out = inp.copy()
    out[out == RED] = BG
    out[:, divider_col + 1 :] = BG
    for slot_idx in active_slots:
        paste(out, template, slot_idx * slot_h, divider_col + 1)
    return inp, out, active_slots


def _choose_start(rng: random.Random, height: int, slot_h: int, slot_w: int) -> tuple[int, tuple[int, ...]] | None:
    slot_count = height // slot_h
    candidates: list[tuple[int, tuple[int, ...]]] = []
    signatures: dict[tuple[int, ...], list[int]] = {}
    for start_col in range(slot_w - 1):
        sig = _signature(height, slot_w, slot_h, start_col)
        signatures.setdefault(sig, []).append(start_col)

    for start_col in range(slot_w - 1):
        actual_sig = _signature(height, slot_w, slot_h, start_col)
        if signatures[actual_sig] != [start_col]:
            continue
        if not actual_sig or len(actual_sig) == slot_count:
            continue
        candidates.append((start_col, actual_sig))

    if not candidates:
        return None

    weights = []
    for start_col, sig in candidates:
        weight = 1
        if start_col == 0 or start_col == slot_w - 2:
            weight += 2
        if len(sig) >= 2:
            weight += 1
        weights.append(weight)
    return rng.choices(candidates, weights=weights, k=1)[0]


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, active_slots: tuple[int, ...]) -> None:
    divider_col, template, (template_top, template_left), local_start = _extract_objects(inp)
    slot_h, slot_w = map(int, template.shape)
    height, width = map(int, inp.shape)

    assert template_left + slot_w == divider_col
    assert height % slot_h == 0 and width <= 30 and height <= 30
    assert local_start < slot_w - 1
    assert active_slots == _assert_unique_signature(height, slot_h, slot_w, local_start)

    actual = main(inp)
    assert np.array_equal(actual, expected)

    corridor = expected[:, divider_col + 1 :]
    for slot_idx in range(height // slot_h):
        block = corridor[slot_idx * slot_h : (slot_idx + 1) * slot_h, :]
        if slot_idx in active_slots:
            assert np.array_equal(block, template)
        else:
            assert np.all(block == BG)

    assert np.array_equal(expected[template_top : template_top + slot_h, template_left : template_left + slot_w], template)
    assert not np.any(expected == RED)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    slot_h = rng.randint(2, 5)
    slot_w = rng.randint(2, 6)
    slot_count = rng.randint(3, min(6, 30 // slot_h))
    height = slot_h * slot_count
    left_margin = rng.randint(0, min(4, 29 - (2 * slot_w + 1)))
    width = left_margin + 2 * slot_w + 1
    if width > 30:
        return None

    template_top = rng.randint(0, height - slot_h)
    template = _random_template(rng, slot_h, slot_w)

    choice = _choose_start(rng, height, slot_h, slot_w)
    if choice is None:
        return None
    start_col, active_slots = choice

    inp, expected, reference_slots = _reference_output(template, height, left_margin, start_col, template_top)
    if active_slots != reference_slots:
        return None

    _semantic_checks(inp, expected, active_slots)
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-bounce slot puzzle")
