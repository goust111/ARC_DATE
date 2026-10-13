from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors4


PAIR_MAP = {
    Color.TEAL: Color.RED,
    Color.GREEN: Color.BLUE,
    Color.GREY: Color.YELLOW,
}

PANEL_BITS = (1, 2, 4)  # A, B, C
PANEL_SHAPE = (6, 6)
INPUT_SHAPE = (18, 6)
MAX_PANEL_AREA = 9
MAX_GENERATION_ATTEMPTS = 400
EXTRA_STEPS_RANGE = (0, 4)
MAX_OVERLAP_EXTRAS = 2

ROLE_A = 1
ROLE_B = 2
ROLE_C = 4
ROLE_AB = 3
ROLE_AC = 5
ROLE_BC = 6
ROLE_ABC = 7
OVERLAP_ROLES = {ROLE_AB, ROLE_AC, ROLE_BC, ROLE_ABC}
ROLE_WEIGHTS = {
    ROLE_A: 3,
    ROLE_B: 3,
    ROLE_C: 3,
    ROLE_AB: 2,
    ROLE_AC: 2,
    ROLE_BC: 2,
    ROLE_ABC: 1,
}

BASE_TEMPLATES = [
    np.array(
        [
            [0, ROLE_B, 0],
            [0, ROLE_BC, ROLE_C],
            [ROLE_AB, ROLE_ABC, ROLE_AC],
            [ROLE_A, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [ROLE_A, 0, ROLE_B],
            [ROLE_AB, ROLE_ABC, ROLE_BC],
            [0, ROLE_AC, 0],
            [0, ROLE_C, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, ROLE_A, 0],
            [ROLE_B, ROLE_AB, ROLE_BC],
            [0, ROLE_ABC, ROLE_AC],
            [0, 0, ROLE_C],
        ],
        dtype=int,
    ),
]

COLOR_ORDERS = [
    (Color.TEAL, Color.GREEN, Color.GREY),
    (Color.GREEN, Color.GREY, Color.TEAL),
    (Color.GREY, Color.TEAL, Color.GREEN),
]

_GEN_CALL_COUNT = 0
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    assert h % 3 == 0 and w > 0, "expected three equal-height stacked panels"
    panel_h = h // 3
    top = grid[:panel_h, :]
    middle = grid[panel_h : 2 * panel_h, :]
    bottom = grid[2 * panel_h :, :]
    return top, middle, bottom


def _panel_masks_from_roles(roles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple((roles & bit) != 0 for bit in PANEL_BITS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge three stacked monochrome panels, branching on whether each output cell is claimed by one, two, or three layers.

    Input structure
    - The input is three equal-size panels stacked vertically on a black background.
    - Each panel uses exactly one foreground color, and the three panel colors are teal, green, and grey in some order.
    - Within each panel, foreground pixels form one 4-connected shape occupying the same 6x6 coordinate frame as the other panels.

    Rule
    - Split the input into top panel A, middle panel B, and bottom panel C, then compare corresponding cells across the three panels.
    - If exactly one panel has foreground at a cell, keep that panel's original color.
    - If exactly two panels have foreground, choose the lower of those two panels as the winner and remap its color: teal -> red, green -> blue, grey -> yellow.
    - If all three panels have foreground, erase that cell to black; empty cells also stay black.

    Key insight
    - This is not ordinary overwrite compositing: the decisive feature is the conflict multiplicity itself, because one-layer, two-layer, and three-layer collisions trigger three different output behaviors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    top, middle, bottom = _split_panels(g)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    panel_colors: list[int] = []
    panel_masks: list[np.ndarray] = []
    for panel in (top, middle, bottom):
        colors = colors_present(panel, exclude={bg})
        assert len(colors) == 1, "each panel must use exactly one foreground color"
        color = colors[0]
        assert color in PAIR_MAP, "panel foreground color must be teal, green, or grey"
        mask = panel == color
        assert 3 <= int(mask.sum()) <= 12, "panel foreground area out of range"
        assert len(components_mask(mask, connectivity=4)) == 1, "each panel shape must be 4-connected"
        panel_colors.append(color)
        panel_masks.append(mask)

    assert set(panel_colors) == {Color.TEAL, Color.GREEN, Color.GREY}, "expected one teal, one green, and one grey panel"

    a_mask, b_mask, c_mask = panel_masks
    counts = a_mask.astype(int) + b_mask.astype(int) + c_mask.astype(int)
    assert np.any(counts == 1) and np.any(counts == 2) and np.any(counts == 3), "expected single, double, and triple occupancy"

    out = np.zeros_like(top)
    out[a_mask & ~b_mask & ~c_mask] = panel_colors[0]
    out[~a_mask & b_mask & ~c_mask] = panel_colors[1]
    out[~a_mask & ~b_mask & c_mask] = panel_colors[2]
    out[a_mask & b_mask & ~c_mask] = PAIR_MAP[panel_colors[1]]
    out[a_mask & ~b_mask & c_mask] = PAIR_MAP[panel_colors[2]]
    out[~a_mask & b_mask & c_mask] = PAIR_MAP[panel_colors[2]]
    return out


def _array_sig(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + repr(arr.shape).encode()).hexdigest()


def _panel_count(roles: np.ndarray, bit: int) -> int:
    return int(np.count_nonzero((roles & bit) != 0))


def _render_panels_from_roles(roles: np.ndarray, colors: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panels = []
    for bit, color in zip(PANEL_BITS, colors):
        panel = np.zeros(PANEL_SHAPE, dtype=int)
        panel[(roles & bit) != 0] = color
        panels.append(panel)
    return tuple(panels)


def _expected_from_roles(roles: np.ndarray, colors: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(PANEL_SHAPE, dtype=int)
    out[roles == ROLE_A] = colors[0]
    out[roles == ROLE_B] = colors[1]
    out[roles == ROLE_C] = colors[2]
    out[roles == ROLE_AB] = PAIR_MAP[colors[1]]
    out[roles == ROLE_AC] = PAIR_MAP[colors[2]]
    out[roles == ROLE_BC] = PAIR_MAP[colors[2]]
    return out


def _overlay_baseline(panels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    top, middle, bottom = panels
    out = middle.copy()
    out[top != 0] = top[top != 0]
    out[bottom != 0] = bottom[bottom != 0]
    return out


def _single_only_baseline(roles: np.ndarray, colors: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(PANEL_SHAPE, dtype=int)
    out[roles == ROLE_A] = colors[0]
    out[roles == ROLE_B] = colors[1]
    out[roles == ROLE_C] = colors[2]
    return out


def _map_all_conflicts_baseline(roles: np.ndarray, colors: tuple[int, int, int]) -> np.ndarray:
    out = _expected_from_roles(roles, colors)
    out[roles == ROLE_ABC] = PAIR_MAP[colors[2]]
    return out


def _bottom_biased_double_baseline(roles: np.ndarray, colors: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(PANEL_SHAPE, dtype=int)
    out[roles == ROLE_A] = colors[0]
    out[roles == ROLE_B] = colors[1]
    out[roles == ROLE_C] = colors[2]
    out[np.isin(roles, [ROLE_AB, ROLE_AC, ROLE_BC])] = PAIR_MAP[colors[2]]
    return out


def _transformed_template() -> np.ndarray:
    template = random.choice(BASE_TEMPLATES)
    rotated = np.rot90(template, random.randrange(4))
    if random.getrandbits(1):
        rotated = np.fliplr(rotated)
    h, w = rotated.shape
    roles = np.zeros(PANEL_SHAPE, dtype=int)
    r0 = random.randint(0, PANEL_SHAPE[0] - h)
    c0 = random.randint(0, PANEL_SHAPE[1] - w)
    roles[r0 : r0 + h, c0 : c0 + w] = rotated
    return roles


def _role_candidates(roles: np.ndarray, role: int) -> list[tuple[int, int]]:
    bits = [bit for bit in PANEL_BITS if role & bit]
    masks = _panel_masks_from_roles(roles)
    candidates: list[tuple[int, int]] = []

    for r in range(PANEL_SHAPE[0]):
        for c in range(PANEL_SHAPE[1]):
            if roles[r, c] != 0:
                continue
            ok = True
            for bit, mask in zip(PANEL_BITS, masks):
                if bit not in bits:
                    continue
                if not any(
                    0 <= nr < PANEL_SHAPE[0] and 0 <= nc < PANEL_SHAPE[1] and mask[nr, nc]
                    for nr, nc in neighbors4(r, c)
                ):
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    return candidates


def _grow_roles(seed_roles: np.ndarray) -> np.ndarray:
    roles = seed_roles.copy()
    overlap_extras = 0
    triple_extras = 0

    for _ in range(random.randint(*EXTRA_STEPS_RANGE)):
        weighted_moves: list[tuple[int, int, int]] = []
        counts = {bit: _panel_count(roles, bit) for bit in PANEL_BITS}
        for role, weight in ROLE_WEIGHTS.items():
            if role == ROLE_ABC and triple_extras >= 1:
                continue
            if role in OVERLAP_ROLES and overlap_extras >= MAX_OVERLAP_EXTRAS:
                continue
            if any(counts[bit] >= MAX_PANEL_AREA for bit in PANEL_BITS if role & bit):
                continue
            for r, c in _role_candidates(roles, role):
                weighted_moves.extend([(role, r, c)] * weight)
        if not weighted_moves:
            break
        role, r, c = random.choice(weighted_moves)
        roles[r, c] = role
        if role == ROLE_ABC:
            triple_extras += 1
        if role in OVERLAP_ROLES:
            overlap_extras += 1
    return roles


def _roles_are_valid(roles: np.ndarray) -> bool:
    if roles.shape != PANEL_SHAPE:
        return False
    if not {ROLE_A, ROLE_B, ROLE_C, ROLE_AB, ROLE_AC, ROLE_BC, ROLE_ABC}.issubset(set(map(int, np.unique(roles)))):
        return False

    masks = _panel_masks_from_roles(roles)
    counts = masks[0].astype(int) + masks[1].astype(int) + masks[2].astype(int)
    if not np.any(counts == 1) or not np.any(counts == 2) or not np.any(counts == 3):
        return False
    if int(np.count_nonzero(counts >= 2)) > 8:
        return False

    for mask in masks:
        area = int(mask.sum())
        if area < 3 or area > 12:
            return False
        if len(components_mask(mask, connectivity=4)) != 1:
            return False
    return True


def _example_is_semantic(roles: np.ndarray, colors: tuple[int, int, int], panels: tuple[np.ndarray, np.ndarray, np.ndarray], out: np.ndarray) -> bool:
    if not _roles_are_valid(roles):
        return False

    expected = _expected_from_roles(roles, colors)
    if not np.array_equal(out, expected):
        return False

    direct = main(np.vstack(panels))
    if not np.array_equal(direct, expected):
        return False

    needed_non_black = {colors[0], colors[1], colors[2], PAIR_MAP[colors[1]], PAIR_MAP[colors[2]]}
    if set(colors_present(expected, exclude={Color.BLACK})) != needed_non_black:
        return False

    if np.array_equal(expected, _overlay_baseline(panels)):
        return False
    if np.array_equal(expected, _single_only_baseline(roles, colors)):
        return False
    if np.array_equal(expected, _map_all_conflicts_baseline(roles, colors)):
        return False
    if np.array_equal(expected, _bottom_biased_double_baseline(roles, colors)):
        return False

    return True


def _gen_attempt(colors: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    roles = _grow_roles(_transformed_template())
    if not _roles_are_valid(roles):
        return None

    panels = _render_panels_from_roles(roles, colors)
    inp = np.vstack(panels)
    out = _expected_from_roles(roles, colors)
    if not _example_is_semantic(roles, colors, panels, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(MAX_GENERATION_ATTEMPTS):
        colors = COLOR_ORDERS[_GEN_CALL_COUNT % len(COLOR_ORDERS)]
        generated = _gen_attempt(colors)
        if generated is None:
            continue
        inp, out = generated
        in_sig = _array_sig(inp)
        out_sig = _array_sig(out)
        if in_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(in_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        _GEN_CALL_COUNT += 1
        return inp

    raise RuntimeError("failed to generate a valid input")
