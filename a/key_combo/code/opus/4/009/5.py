from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, paste


BLACK = 0
RED = 2
GREY = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _mask(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=bool)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask.astype(bool))
    assert box is not None
    return crop_bbox(mask.astype(int), box).astype(bool)


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    box = bbox(patch != BLACK)
    assert box is not None
    return crop_bbox(patch, box)


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            rot = np.rot90(base, k)
            norm = _trim_mask(rot)
            key = (norm.shape, norm.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(norm)
    return variants


def _rand_oriented(mask_bank: list[np.ndarray]) -> np.ndarray:
    return random.choice(_unique_orientations(random.choice(mask_bank)))


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


DOWN_T = np.array(
    [
        [RED, RED, RED],
        [BLACK, RED, BLACK],
        [BLACK, RED, BLACK],
    ],
    dtype=int,
)

UP_T = np.array(
    [
        [BLACK, RED, BLACK],
        [BLACK, RED, BLACK],
        [RED, RED, RED],
    ],
    dtype=int,
)


COMPONENT_BANK = [
    _mask([[1, 1]]),
    _mask([[1], [1]]),
    _mask([[1, 1, 1]]),
    _mask([[1], [1], [1]]),
    _mask([[1, 1], [1, 0]]),
    _mask([[1, 1], [0, 1]]),
    _mask([[1, 1], [1, 1]]),
    _mask([[1, 1, 1], [1, 0, 0]]),
    _mask([[1, 1, 1], [0, 1, 0]]),
]

SPRITE_BANK = [
    _mask([[1, 1, 1], [1, 0, 0], [1, 1, 0]]),
    _mask([[1, 1, 1], [0, 1, 0], [0, 1, 1]]),
    _mask([[1, 1, 1], [1, 1, 0], [0, 1, 0]]),
    _mask([[1, 1, 0], [0, 1, 1], [0, 1, 0]]),
    _mask([[1, 1, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]]),
    _mask([[1, 1, 1], [1, 0, 1], [1, 0, 1]]),
    _mask([[1, 1, 1], [1, 0, 0], [1, 1, 1]]),
    _mask([[1, 1, 1, 1], [1, 0, 1, 0]]),
]

LOWER_OBJECT_BANK = [
    _mask([[1, 1, 1], [1, 0, 0], [1, 1, 0]]),
    _mask([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
    _mask([[1, 1, 0], [1, 1, 1]]),
    _mask([[1, 1, 1], [1, 0, 1]]),
    _mask([[1, 1], [1, 1]]),
    _mask([[1, 0], [1, 1], [0, 1]]),
]

TEMPLATE_LAYOUTS = [
    {"shape": (7, 11), "slots": [(0, 0), (4, 0), (1, 4), (3, 8)], "same_col_pair": (0, 1)},
    {"shape": (7, 11), "slots": [(1, 0), (0, 4), (4, 4), (2, 8)], "same_col_pair": (1, 2)},
    {"shape": (7, 11), "slots": [(0, 1), (4, 1), (0, 6), (4, 7)], "same_col_pair": (0, 1)},
    {"shape": (7, 11), "slots": [(2, 0), (0, 4), (4, 4), (2, 8)], "same_col_pair": (1, 2)},
]


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1
    return int(rows[0])


def _detect_t_direction(upper: np.ndarray) -> str:
    t_components = find_components(upper, bg=BLACK, colors=[RED], connectivity=4)
    matches: list[str] = []
    for comp in t_components:
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1, comp["area"]) != (3, 3, 5):
            continue
        crop = crop_bbox(upper, comp["bbox"])
        if np.array_equal(crop, DOWN_T):
            matches.append("down")
        elif np.array_equal(crop, UP_T):
            matches.append("up")
    assert len(matches) == 1
    return matches[0]


def _parse_instance(grid: np.ndarray) -> tuple[np.ndarray, str, np.ndarray, list[int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    upper = g[:sep, :]
    lower = g[sep + 1 :, :]

    direction = _detect_t_direction(upper)
    allowed = [int(c) for c in np.unique(upper) if int(c) not in (BLACK, RED, GREY)]
    components = find_components(upper, bg=BLACK, colors=allowed, connectivity=4)
    assert 3 <= len(components) <= 5

    max_area = max(comp["area"] for comp in components)
    sprite_candidates = [comp for comp in components if comp["area"] == max_area]
    assert len(sprite_candidates) == 1
    sprite = sprite_candidates[0]
    sprite_crop = crop_bbox(upper, sprite["bbox"])

    template_components = [comp for comp in components if comp is not sprite]
    ordered = sorted(template_components, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    color_sequence = [int(comp["color"]) for comp in ordered]

    lower_components = find_components(lower, bg=BLACK, colors=[GREY], connectivity=4)
    assert len(lower_components) == 1
    template_bbox = tuple(int(v) for v in lower_components[0]["bbox"])

    return lower, direction, sprite_crop, color_sequence, template_bbox


def _stamp_sequence(
    lower: np.ndarray,
    direction: str,
    sprite_crop: np.ndarray,
    color_sequence: list[int],
    template_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = lower.copy()
    sprite_mask = sprite_crop != BLACK
    sprite_h, _sprite_w = sprite_crop.shape
    r0, c0, r1, _c1 = template_bbox
    for idx, color in enumerate(color_sequence):
        patch = np.zeros_like(sprite_crop)
        patch[sprite_mask] = int(color)
        if direction == "down":
            top = r1 + 1 + idx * sprite_h
        else:
            top = r0 - (idx + 1) * sprite_h
        paste(out, patch, top, c0, transparent=BLACK, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack recolored copies of the cropped upper sprite above or below the lower gray template.

    Input structure:
    - A black grid is split by one full-width gray separator row into an upper zone and a lower zone.
    - The upper zone contains exactly one red 3x3 T marker, one monochrome non-red/non-gray 4-connected sprite, and 2 to 4 separated monochrome template components in distinct non-red/non-gray colors.
    - The lower zone contains one gray 4-connected template object on black background.

    Rule:
    - Read the red T marker: a top horizontal arm means stack downward, while a bottom horizontal arm means stack upward.
    - Among upper pixels that are not black, red, or gray, the unique largest 4-connected component is the sprite; crop it to its bounding box.
    - Sort the remaining upper components by the column and then row of their bounding-box top-left corners, use their colors as the recolor sequence, and stamp one recolored sprite copy per color with left edges aligned to the lower gray template object's left edge.
    - The first copy touches the gray object's chosen side, later copies continue the same stack, clipped pixels are discarded, and only the lower zone is returned.

    Key insight:
    - The red T, the count of small upper components, and their spatial order each encode different parts of the same stamping instruction.
    """
    lower, direction, sprite_crop, color_sequence, template_bbox = _parse_instance(grid)
    return _stamp_sequence(lower, direction, sprite_crop, color_sequence, template_bbox)


def _choose_template_count() -> int:
    return random.choices([2, 3, 4], weights=[2, 4, 4], k=1)[0]


def _nontrivial_color_order(colors: list[int]) -> list[int] | None:
    asc = tuple(sorted(colors))
    desc = tuple(sorted(colors, reverse=True))
    perms = list(itertools.permutations(colors))
    if len(colors) == 2:
        valid = [perm for perm in perms if perm != asc]
    else:
        valid = [perm for perm in perms if perm != asc and perm != desc]
    if not valid:
        return None
    return list(random.choice(valid))


def _build_template_patch(n: int, available_colors: list[int]) -> tuple[np.ndarray, list[int]]:
    for _ in range(200):
        layout = random.choice(TEMPLATE_LAYOUTS)
        slots = list(range(len(layout["slots"])))
        tie_mode = n >= 2 and random.random() < 0.5
        if tie_mode:
            a, b = layout["same_col_pair"]
            rest = [idx for idx in slots if idx not in (a, b)]
            chosen = [a, b] + random.sample(rest, n - 2)
        else:
            chosen = random.sample(slots, n)

        chosen_slots = [layout["slots"][idx] for idx in chosen]
        order = sorted(range(n), key=lambda idx: (chosen_slots[idx][1], chosen_slots[idx][0]))
        masks = [_rand_oriented(COMPONENT_BANK) for _ in range(n)]
        ordered_areas = [int(masks[idx].sum()) for idx in order]
        if n >= 3:
            if len(set(ordered_areas)) < 2:
                continue
            if ordered_areas == sorted(ordered_areas) or ordered_areas == sorted(ordered_areas, reverse=True):
                continue

        picked_colors = random.sample(available_colors, n)
        ordered_colors = _nontrivial_color_order(picked_colors)
        if ordered_colors is None:
            continue

        per_component_colors = [0] * n
        for rank, idx in enumerate(order):
            per_component_colors[idx] = ordered_colors[rank]

        patch = np.zeros(layout["shape"], dtype=int)
        for idx, (top, left) in enumerate(chosen_slots):
            comp = _colorize(masks[idx], per_component_colors[idx])
            h, w = comp.shape
            view = patch[top : top + h, left : left + w]
            assert view.shape == comp.shape
            assert not np.any((view != BLACK) & (comp != BLACK))
            view[comp != BLACK] = comp[comp != BLACK]

        return _trim_patch(patch), ordered_colors
    raise RuntimeError("failed to build a non-degenerate template patch")


def _random_place(occupied: np.ndarray, patch: np.ndarray, pad: int = 1) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = patch.shape
    if h > H or w > W:
        return None
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        r0 = max(0, top - pad)
        c0 = max(0, left - pad)
        r1 = min(H, top + h + pad)
        c1 = min(W, left + w + pad)
        if occupied[r0:r1, c0:c1].any():
            continue
        return int(top), int(left)
    return None


def _pick_lower_row(lower_h: int, obj_h: int, sprite_h: int, copies: int, direction: str) -> tuple[int, str] | None:
    modes = ["clear", "clip"]
    random.shuffle(modes)
    for mode in modes:
        if direction == "down":
            if mode == "clear":
                hi = lower_h - obj_h - copies * sprite_h
                lo = 0
            else:
                lo = max(0, lower_h - obj_h - copies * sprite_h + 1)
                hi = lower_h - obj_h - 1
        else:
            if mode == "clear":
                lo = copies * sprite_h
                hi = lower_h - obj_h
            else:
                lo = 1
                hi = min(copies * sprite_h - 1, lower_h - obj_h)
        if lo <= hi:
            return random.randint(lo, hi), mode
    return None


def _pick_lower_col(width: int, obj_w: int, sprite_w: int) -> tuple[int, str] | None:
    modes = ["clear", "clip"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "clear":
            lo = 0
            hi = min(width - obj_w, width - sprite_w)
        else:
            lo = max(0, width - sprite_w + 1)
            hi = width - obj_w
        if lo <= hi:
            return random.randint(lo, hi), mode
    return None


def _validate_generated_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    direction: str,
    sprite_patch: np.ndarray,
    ordered_colors: list[int],
    lower_bbox: tuple[int, int, int, int],
) -> None:
    sep = _separator_row(grid)
    upper = grid[:sep, :]
    lower = grid[sep + 1 :, :]

    assert _detect_t_direction(upper) == direction
    allowed = [int(c) for c in np.unique(upper) if int(c) not in (BLACK, RED, GREY)]
    components = find_components(upper, bg=BLACK, colors=allowed, connectivity=4)
    assert len(components) == len(ordered_colors) + 1

    areas = [int(comp["area"]) for comp in components]
    assert areas.count(max(areas)) == 1
    sprite = max(components, key=lambda comp: comp["area"])
    assert np.array_equal(crop_bbox(upper, sprite["bbox"]), sprite_patch)

    template_components = [comp for comp in components if comp is not sprite]
    parsed_order = sorted(template_components, key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    parsed_colors = [int(comp["color"]) for comp in parsed_order]
    assert parsed_colors == ordered_colors

    if len(ordered_colors) >= 2:
        assert tuple(ordered_colors) != tuple(sorted(ordered_colors))
    if len(ordered_colors) >= 3:
        assert tuple(ordered_colors) != tuple(sorted(ordered_colors, reverse=True))
        parsed_areas = [int(comp["area"]) for comp in parsed_order]
        assert parsed_areas != sorted(parsed_areas)
        assert parsed_areas != sorted(parsed_areas, reverse=True)

    lower_components = find_components(lower, bg=BLACK, colors=[GREY], connectivity=4)
    assert len(lower_components) == 1
    assert tuple(int(v) for v in lower_components[0]["bbox"]) == tuple(lower_bbox)

    produced = main(grid)
    assert np.array_equal(produced, expected)
    assert produced.shape == expected.shape
    assert np.count_nonzero(produced != lower) > 0


def _gen_attempt() -> np.ndarray | None:
    direction = random.choice(["down", "up"])
    sprite_color = random.choice(PALETTE)
    sprite_patch = _colorize(_rand_oriented(SPRITE_BANK), sprite_color)
    sprite_h, sprite_w = sprite_patch.shape

    template_count = _choose_template_count()
    template_colors_pool = [color for color in PALETTE if color != sprite_color]
    template_patch, ordered_colors = _build_template_patch(template_count, template_colors_pool)
    t_patch = DOWN_T if direction == "down" else UP_T

    upper_h = random.randint(max(8, template_patch.shape[0]), 11)
    max_lower_h = 30 - upper_h - 1
    if max_lower_h < 8:
        return None

    lower_object_patch = _colorize(_rand_oriented(LOWER_OBJECT_BANK), GREY)
    obj_h, obj_w = lower_object_patch.shape

    width = random.randint(18, 22)
    if max(template_patch.shape[1], sprite_w, obj_w, 3) > width:
        return None

    upper = np.zeros((upper_h, width), dtype=int)
    occupied = np.zeros_like(upper, dtype=bool)
    placements: dict[str, tuple[int, int]] = {}
    for name, patch in random.sample(
        [("t", t_patch), ("sprite", sprite_patch), ("template", template_patch)],
        k=3,
    ):
        pos = _random_place(occupied, patch, pad=1)
        if pos is None:
            return None
        placements[name] = pos
        paste(upper, patch, pos[0], pos[1], transparent=BLACK)
        occupied |= upper != BLACK

    min_clear_lower_h = obj_h + template_count * sprite_h
    if random.random() < 0.5 and min_clear_lower_h <= min(18, max_lower_h):
        lower_h = random.randint(min_clear_lower_h, min(18, max_lower_h))
    else:
        lower_h = random.randint(max(8, obj_h + 2), min(18, max_lower_h))

    lower = np.zeros((lower_h, width), dtype=int)
    row_pick = _pick_lower_row(lower_h, obj_h, sprite_h, template_count, direction)
    col_pick = _pick_lower_col(width, obj_w, sprite_w)
    if row_pick is None or col_pick is None:
        return None
    obj_r, _vertical_mode = row_pick
    obj_c, _horizontal_mode = col_pick
    paste(lower, lower_object_patch, obj_r, obj_c, transparent=BLACK)
    lower_bbox = (obj_r, obj_c, obj_r + obj_h - 1, obj_c + obj_w - 1)

    grid = np.zeros((upper_h + 1 + lower_h, width), dtype=int)
    grid[:upper_h, :] = upper
    grid[upper_h, :] = GREY
    grid[upper_h + 1 :, :] = lower

    expected = _stamp_sequence(lower, direction, sprite_patch, ordered_colors, lower_bbox)
    if np.count_nonzero(expected != lower) == 0:
        return None
    _validate_generated_instance(grid, expected, direction, sprite_patch, ordered_colors, lower_bbox)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 5")
