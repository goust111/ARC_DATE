from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
SEP = 5
ORANGE = 7
PALETTE = (BLUE, ORANGE)
POSITIONS = ("tl", "tr", "bl", "br")
MUTATION_MODES = (
    "shift_same_sprite",
    "same_box_same_count",
    "same_box_changed_count",
    "independent_any",
)
MAX_GENERATION_ATTEMPTS = 400

_POSITION_ORDER = list(range(len(POSITIONS)))
random.shuffle(_POSITION_ORDER)
_POSITION_CURSOR = 0

_MODE_ORDER = list(MUTATION_MODES)
random.shuffle(_MODE_ORDER)
_MODE_CURSOR = 0


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    height, width = mask.shape
    stack = [tuple(map(int, coords[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, height, width):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 1 and g.shape[1] % 2 == 1
    mid_r = g.shape[0] // 2
    mid_c = g.shape[1] // 2
    assert np.all(g[mid_r, :] == SEP), "expected a grey separator row"
    assert np.all(g[:, mid_c] == SEP), "expected a grey separator column"
    quadrants = [
        g[:mid_r, :mid_c],
        g[:mid_r, mid_c + 1 :],
        g[mid_r + 1 :, :mid_c],
        g[mid_r + 1 :, mid_c + 1 :],
    ]
    assert len({q.shape for q in quadrants}) == 1, "quadrants must have equal size"
    assert quadrants[0].size > 0, "quadrants must be non-empty"
    return quadrants


def _quadrant_keys(quadrants: list[np.ndarray]) -> list[tuple[tuple[int, ...], ...]]:
    return [tuple(tuple(int(v) for v in row) for row in q.tolist()) for q in quadrants]


def _unique_position(quadrants: list[np.ndarray]) -> tuple[int, np.ndarray]:
    keys = _quadrant_keys(quadrants)
    counts = Counter(keys)
    assert sorted(counts.values()) == [1, 3], "expected one unique quadrant and three identical quadrants"
    unique_index = next(i for i, key in enumerate(keys) if counts[key] == 1)
    shared_index = next(i for i, key in enumerate(keys) if counts[key] == 3)
    return unique_index, quadrants[shared_index].copy()


def _row_start(row: int, height: int, unique_index: int) -> int:
    max_row = height - 1
    if unique_index == 0:
        left_offset = row
        return max_row - left_offset
    if unique_index == 1:
        right_offset = row
        return right_offset
    if unique_index == 2:
        left_offset = max_row - row
        return max_row - left_offset
    if unique_index == 3:
        right_offset = max_row - row
        return right_offset
    raise AssertionError(f"unknown quadrant index: {unique_index}")


def _shear(quad: np.ndarray, unique_index: int) -> np.ndarray:
    quad = np.asarray(quad, dtype=int)
    height, width = quad.shape
    out = np.full((height, width + height - 1), BG, dtype=int)
    for r in range(height):
        start = _row_start(r, height, unique_index)
        src_c0 = max(0, -start)
        dst_c0 = max(0, start)
        run = min(width - src_c0, out.shape[1] - dst_c0)
        if run > 0:
            out[r, dst_c0 : dst_c0 + run] = quad[r, src_c0 : src_c0 + run]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the odd quadrant and shear the repeated quadrant pattern in the direction encoded by that odd quadrant's position.

    Input structure:
    - The grid has odd height and width, with a grey middle row and grey middle column forming a separator cross.
    - The cross splits the grid into four equal rectangular quadrants on a black background.
    - Three quadrants contain the same blue/orange pattern at the same relative position, and the fourth quadrant contains a different pattern.

    Rule:
    - Compare the four complete quadrant arrays to find the unique quadrant and the repeated shared quadrant.
    - Use the repeated quadrant as the shear input.
    - Map the unique quadrant position to the shear direction: TL -> down-left, TR -> down-right, BL -> up-left, BR -> up-right.
    - Output the sheared repeated quadrant on a black canvas of height H and width W + H - 1.

    Key insight:
    - The odd quadrant is only a control signal: its position determines how to transform the repeated pattern that appears in the other three quadrants.
    """
    quadrants = _extract_quadrants(np.asarray(grid, dtype=int))
    unique_index, shared = _unique_position(quadrants)
    return _shear(shared, unique_index)


def _cycled_pick_position() -> int:
    global _POSITION_CURSOR
    if _POSITION_CURSOR % len(_POSITION_ORDER) == 0:
        random.shuffle(_POSITION_ORDER)
    choice = _POSITION_ORDER[_POSITION_CURSOR % len(_POSITION_ORDER)]
    _POSITION_CURSOR += 1
    return int(choice)


def _cycled_pick_mode() -> str:
    global _MODE_CURSOR
    if _MODE_CURSOR % len(_MODE_ORDER) == 0:
        random.shuffle(_MODE_ORDER)
    choice = _MODE_ORDER[_MODE_CURSOR % len(_MODE_ORDER)]
    _MODE_CURSOR += 1
    return str(choice)


def _connected_mask(height: int, width: int, cells: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        frontier = set(_neighbors4(start[0], start[1], height, width))
        while mask.sum() < cells and frontier:
            r, c = random.choice(tuple(frontier))
            frontier.remove((r, c))
            if mask[r, c]:
                continue
            mask[r, c] = True
            for nr, nc in _neighbors4(r, c, height, width):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))
        box = _bbox(mask)
        if mask.sum() != cells or box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 == r0 or c1 == c0:
            continue
        if np.all(mask):
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    blue_count = random.randint(1, len(coords) - 1)
    sprite = np.zeros(mask.shape, dtype=int)
    for i, (r, c) in enumerate(coords):
        sprite[r, c] = BLUE if i < blue_count else ORANGE
    return sprite


def _anchor_choices(limit: int, span: int) -> list[int]:
    slack = limit - span
    if slack >= 2:
        return list(range(1, slack))
    return list(range(slack + 1))


def _make_sprite(quad_h: int, quad_w: int) -> dict[str, object] | None:
    box_h_max = min(5, quad_h - 1)
    box_w_max = min(5, quad_w - 1)
    if box_h_max < 3 or box_w_max < 3:
        return None
    for _ in range(80):
        box_h = random.randint(3, box_h_max)
        box_w = random.randint(3, box_w_max)
        max_cells = min(10, box_h * box_w - 1)
        if max_cells < 6:
            continue
        cells = random.randint(6, max_cells)
        mask = _connected_mask(box_h, box_w, cells)
        if mask is None:
            continue
        sprite = _colorize_mask(mask)
        anchor_r = random.choice(_anchor_choices(quad_h, box_h))
        anchor_c = random.choice(_anchor_choices(quad_w, box_w))
        quad = np.zeros((quad_h, quad_w), dtype=int)
        quad[anchor_r : anchor_r + box_h, anchor_c : anchor_c + box_w] = sprite
        return {
            "quad": quad,
            "sprite": sprite,
            "mask": mask,
            "anchor": (anchor_r, anchor_c),
        }
    return None


def _occupancy_difference(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero((a != BG) != (b != BG)))


def _same_box_variant(shared_spec: dict[str, object], quad_h: int, quad_w: int, same_count: bool) -> np.ndarray | None:
    sprite = np.asarray(shared_spec["sprite"], dtype=int)
    mask = np.asarray(shared_spec["mask"], dtype=bool)
    anchor_r, anchor_c = map(int, shared_spec["anchor"])
    box_h, box_w = sprite.shape
    base_cells = int(mask.sum())
    counts = [base_cells] if same_count else [n for n in range(6, min(10, box_h * box_w - 1) + 1) if n != base_cells]
    random.shuffle(counts)
    for cells in counts:
        for _ in range(40):
            new_mask = _connected_mask(box_h, box_w, cells)
            if new_mask is None or np.array_equal(new_mask, mask):
                continue
            new_sprite = _colorize_mask(new_mask)
            quad = np.zeros((quad_h, quad_w), dtype=int)
            quad[anchor_r : anchor_r + box_h, anchor_c : anchor_c + box_w] = new_sprite
            if _occupancy_difference(quad, np.asarray(shared_spec["quad"], dtype=int)) >= 2:
                return quad
    return None


def _shift_variant(shared_spec: dict[str, object], quad_h: int, quad_w: int) -> np.ndarray | None:
    sprite = np.asarray(shared_spec["sprite"], dtype=int)
    anchor_r, anchor_c = map(int, shared_spec["anchor"])
    box_h, box_w = sprite.shape
    candidates = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = anchor_r + dr
        nc = anchor_c + dc
        if 0 <= nr <= quad_h - box_h and 0 <= nc <= quad_w - box_w:
            candidates.append((nr, nc))
    random.shuffle(candidates)
    for nr, nc in candidates:
        quad = np.zeros((quad_h, quad_w), dtype=int)
        quad[nr : nr + box_h, nc : nc + box_w] = sprite
        if _occupancy_difference(quad, np.asarray(shared_spec["quad"], dtype=int)) >= 2:
            return quad
    return None


def _make_unique_quad(shared_spec: dict[str, object], quad_h: int, quad_w: int, mode: str) -> np.ndarray | None:
    if mode == "shift_same_sprite":
        return _shift_variant(shared_spec, quad_h, quad_w)
    if mode == "same_box_same_count":
        return _same_box_variant(shared_spec, quad_h, quad_w, same_count=True)
    if mode == "same_box_changed_count":
        return _same_box_variant(shared_spec, quad_h, quad_w, same_count=False)
    if mode == "independent_any":
        for _ in range(40):
            spec = _make_sprite(quad_h, quad_w)
            if spec is None:
                continue
            quad = np.asarray(spec["quad"], dtype=int)
            if _occupancy_difference(quad, np.asarray(shared_spec["quad"], dtype=int)) >= 2:
                return quad
        return None
    raise AssertionError(f"unknown mutation mode: {mode}")


def _assemble_input(quad_h: int, quad_w: int, unique_index: int, shared: np.ndarray, unique: np.ndarray) -> np.ndarray:
    grid = np.full((2 * quad_h + 1, 2 * quad_w + 1), BG, dtype=int)
    grid[quad_h, :] = SEP
    grid[:, quad_w] = SEP
    placements = (
        (0, 0),
        (0, quad_w + 1),
        (quad_h + 1, 0),
        (quad_h + 1, quad_w + 1),
    )
    for idx, (r0, c0) in enumerate(placements):
        grid[r0 : r0 + quad_h, c0 : c0 + quad_w] = unique if idx == unique_index else shared
    return grid


def _semantic_checks(grid: np.ndarray, unique_index: int, shared: np.ndarray, unique: np.ndarray) -> np.ndarray | None:
    if _occupancy_difference(shared, unique) < 2:
        return None
    left_family = _shear(shared, 0)
    right_family = _shear(shared, 1)
    if np.array_equal(left_family, right_family):
        return None

    quadrants = _extract_quadrants(grid)
    found_unique_index, found_shared = _unique_position(quadrants)
    if found_unique_index != unique_index:
        return None
    if not np.array_equal(found_shared, shared):
        return None
    if not np.array_equal(quadrants[unique_index], unique):
        return None
    if not all(np.array_equal(quadrants[i], shared) for i in range(4) if i != unique_index):
        return None

    expected = _shear(shared, unique_index)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    if unique_index in (0, 3) and not np.array_equal(actual, left_family):
        return None
    if unique_index in (1, 2) and not np.array_equal(actual, right_family):
        return None
    return expected


def _generate_one() -> np.ndarray | None:
    quad_h = random.randint(6, 10)
    quad_w = random.randint(6, 10)
    shared_spec = _make_sprite(quad_h, quad_w)
    if shared_spec is None:
        return None
    shared = np.asarray(shared_spec["quad"], dtype=int)

    unique_index = _cycled_pick_position()
    modes = [m for m in MUTATION_MODES if m != _cycled_pick_mode()]
    modes.insert(0, _MODE_ORDER[(_MODE_CURSOR - 1) % len(_MODE_ORDER)])
    seen_modes = set()
    unique = None
    for mode in modes:
        if mode in seen_modes:
            continue
        seen_modes.add(mode)
        unique = _make_unique_quad(shared_spec, quad_h, quad_w, mode)
        if unique is not None:
            break
    if unique is None:
        return None

    grid = _assemble_input(quad_h, quad_w, unique_index, shared, unique)
    if max(grid.shape) > 30:
        return None
    if _semantic_checks(grid, unique_index, shared, unique) is None:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
