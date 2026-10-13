from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8
MAROON = 9

SLOT_COUNT = 5
SLOT_W = 4
SLOT_H = 6
GRID_H = SLOT_H + 2
DIVIDER_W = 1
MAX_GEN_ATTEMPTS = 400


def _cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _ccw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _slot_bounds(divider_cols: list[int], width: int) -> list[tuple[int, int]]:
    bounds = [-1, *divider_cols, width]
    slots = []
    for i in range(len(bounds) - 1):
        c0 = bounds[i] + 1
        c1 = bounds[i + 1]
        if c0 < c1:
            slots.append((c0, c1))
    return slots


def _infer_layout(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    grey_rows = np.where(np.any(grid == GREY, axis=1))[0]
    assert grey_rows.size > 0, "expected a divider band"
    band_r0 = int(grey_rows.min())
    band_r1 = int(grey_rows.max())
    divider_cols = [
        c
        for c in range(grid.shape[1])
        if np.all(grid[band_r0 : band_r1 + 1, c] == GREY)
    ]
    slots = _slot_bounds(divider_cols, grid.shape[1])
    assert len(slots) >= 2, "expected multiple slots"
    widths = {c1 - c0 for c0, c1 in slots}
    assert len(widths) == 1, "slot widths must match"
    return band_r0, band_r1, slots


def _find_template_slot(
    grid: np.ndarray,
    band_r0: int,
    band_r1: int,
    slots: list[tuple[int, int]],
) -> tuple[int, np.ndarray, tuple[int, int]]:
    found: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    for idx, (c0, c1) in enumerate(slots):
        band = grid[band_r0 : band_r1 + 1, c0:c1]
        mask = np.isin(band, [BLUE, RED])
        if not mask.any():
            continue
        box = _bbox(mask)
        assert box is not None
        r0, cc0, r1, cc1 = box
        found.append((idx, band[r0 : r1 + 1, cc0 : cc1 + 1].copy(), (r0, cc0)))
    assert len(found) == 1, "expected exactly one template slot"
    return found[0]


def _slot_index(slots: list[tuple[int, int]], col: int) -> int | None:
    for idx, (c0, c1) in enumerate(slots):
        if c0 <= col < c1:
            return idx
    return None


def _rotate_cw_times(patch: np.ndarray, times: int) -> np.ndarray:
    t = times % 4
    if t == 0:
        return patch.copy()
    return np.rot90(patch, k=-t)


def _simulate_path(
    grid: np.ndarray,
    band_r0: int,
    band_r1: int,
    slots: list[tuple[int, int]],
    template_slot: int,
) -> list[int]:
    start = np.argwhere(grid == GREEN)
    end = np.argwhere(grid == MAROON)
    assert start.shape == (2, 2), "expected a 2-cell start line"
    assert end.shape == (2, 2), "expected a 2-cell end line"
    assert len(set(map(int, start[:, 1]))) == 1, "start must be vertical"
    assert len(set(map(int, end[:, 1]))) == 1, "end must be vertical"

    pos = tuple(map(int, start[np.argmax(start[:, 0])]))
    direction = (1, 0)
    seen_slots: list[int] = []
    seen_slot_set: set[int] = set()
    visited_positions: set[tuple[int, int]] = {pos}
    seen_states: set[tuple[int, int, int, int]] = set()

    def is_goal(cell: tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and int(grid[r, c]) == MAROON

    def is_blocked(cell: tuple[int, int]) -> bool:
        r, c = cell
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return True
        if cell in visited_positions:
            return False
        color = int(grid[r, c])
        return color not in (BLACK, GREEN)

    limit = grid.shape[0] * grid.shape[1] * 8
    for _ in range(limit):
        r, c = pos
        if band_r0 <= r <= band_r1:
            slot_idx = _slot_index(slots, c)
            if slot_idx is not None and slot_idx != template_slot and slot_idx not in seen_slot_set:
                seen_slot_set.add(slot_idx)
                seen_slots.append(slot_idx)

        state = (r, c, direction[0], direction[1])
        assert state not in seen_states, "path loop detected"
        seen_states.add(state)

        nxt = (r + direction[0], c + direction[1])
        if is_goal(nxt):
            return seen_slots
        if not is_blocked(nxt):
            pos = nxt
            visited_positions.add(pos)
            continue

        turned = False
        for new_dir in (_cw(direction), _ccw(direction)):
            cand = (r + new_dir[0], c + new_dir[1])
            if is_goal(cand) or not is_blocked(cand):
                direction = new_dir
                turned = True
                break
        if not turned:
            break

    raise AssertionError("path did not reach the end marker")


def _render_output_from_visits(grid: np.ndarray, visits: list[int]) -> np.ndarray:
    band_r0, band_r1, slots = _infer_layout(grid)
    template_slot, template_crop, template_offset = _find_template_slot(grid, band_r0, band_r1, slots)

    out = grid.copy()
    out[out == TEAL] = BLACK
    offset_r, offset_c = template_offset

    for k, slot_idx in enumerate(visits, start=1):
        rotated = _rotate_cw_times(template_crop, k)
        c0, c1 = slots[slot_idx]
        r1 = band_r0 + offset_r + rotated.shape[0]
        c1_abs = c0 + offset_c + rotated.shape[1]
        out[band_r0 + offset_r : r1, c0 + offset_c : c1_abs] = rotated

    # Keep the original template slot untouched.
    c0, c1 = slots[template_slot]
    template_band = grid[band_r0 : band_r1 + 1, c0:c1]
    out[band_r0 : band_r1 + 1, c0:c1] = template_band
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each empty slot with the template rotation determined by path visitation order.

    Input structure:
    - A horizontal band is split into equal-width slots by grey vertical divider columns.
    - One slot contains a blue/red template sprite; the other slots are empty except for teal
      turn-trigger pixels and top/bottom edge start/end markers.
    - A 2-cell green vertical line marks the entry slot from above, and a 2-cell maroon
      vertical line marks the final destination from below.

    Rule:
    - Infer the slot slices from the grey divider columns and extract the template crop plus
      its offset inside the template slot.
    - Simulate the particle from the lower green cell, initially moving downward. Moving
      straight is allowed only through black cells; when the next cell is blocked by a teal
      trigger, divider, non-path content, or the grid edge, first try a clockwise turn and
      then a counterclockwise turn.
    - Record the empty slots in the order the particle first enters their interior band. The
      1st visited slot gets the template rotated 90 degrees clockwise, the 2nd gets 180
      degrees, the 3rd gets 270 degrees, the 4th gets 0 degrees, and so on modulo 4.
      Teal trigger pixels are erased in the output while dividers and markers stay.

    Key insight:
    - The crucial order is not left-to-right slot order but the hidden traversal order encoded
      by obstacle-triggered turns along the path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    band_r0, band_r1, slots = _infer_layout(g)
    template_slot, _template_crop, _template_offset = _find_template_slot(g, band_r0, band_r1, slots)
    visits = _simulate_path(g, band_r0, band_r1, slots, template_slot)
    assert len(visits) == len(slots) - 1, "path must visit every empty slot exactly once"
    return _render_output_from_visits(g, visits)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    all_cells = set(cells)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in all_cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(all_cells)


def _random_template_sprite() -> np.ndarray:
    for _ in range(300):
        target_area = random.randint(6, 8)
        mask = np.zeros((3, 3), dtype=bool)
        seed = (random.randrange(3), random.randrange(3))
        mask[seed] = True
        cells = {seed}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            chosen = random.choice(sorted(frontier))
            cells.add(chosen)
            mask[chosen] = True
        if len(cells) != target_area or not _is_connected(mask):
            continue
        if np.array_equal(mask, np.rot90(mask, 1)) or np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue

        sprite = np.zeros((3, 3), dtype=int)
        for r, c in cells:
            sprite[r, c] = random.choice([BLUE, RED])
        if BLUE not in sprite or RED not in sprite:
            continue
        variants = {_rotate_cw_times(sprite, k).tobytes() for k in range(4)}
        if len(variants) != 4:
            continue
        return sprite
    raise RuntimeError("failed to generate a distinct template sprite")


def _slot_start(slot_idx: int) -> int:
    return slot_idx * (SLOT_W + DIVIDER_W)


def _base_grid() -> np.ndarray:
    grid = np.full((GRID_H, SLOT_COUNT * SLOT_W + (SLOT_COUNT - 1) * DIVIDER_W), BLACK, dtype=int)
    for slot_idx in range(SLOT_COUNT - 1):
        divider_c = _slot_start(slot_idx) + SLOT_W
        grid[1 : GRID_H - 1, divider_c] = GREY
    return grid


def _place_template(
    grid: np.ndarray,
    template_slot: int,
    sprite: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> None:
    c0 = _slot_start(template_slot)
    grid[1 + row_offset : 1 + row_offset + sprite.shape[0], c0 + col_offset : c0 + col_offset + sprite.shape[1]] = sprite


def _entry_col(slot_idx: int, from_left: bool) -> int:
    c0 = _slot_start(slot_idx)
    return c0 + (1 if from_left else 2)


def _turn_blocker_col(slot_idx: int, moving_right: bool) -> int:
    c0 = _slot_start(slot_idx)
    return c0 + (2 if moving_right else 1)


def _add_decoy_obstacles(grid: np.ndarray, slot_idx: int, reserved_cols: set[int]) -> None:
    c0 = _slot_start(slot_idx)
    candidates = [
        (r, c0 + dc)
        for r in range(2, GRID_H - 2)
        for dc in range(SLOT_W)
        if c0 + dc not in reserved_cols
    ]
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(1, 3)]:
        grid[r, c] = TEAL


def _expected_visit_order(template_slot: int) -> list[int]:
    empties = [idx for idx in range(SLOT_COUNT) if idx != template_slot]
    return [empties[1], empties[0], empties[2], empties[3]]


def _plan_output(grid: np.ndarray, visit_order: list[int]) -> np.ndarray:
    return _render_output_from_visits(grid, visit_order)


def _validate_instance(grid: np.ndarray, template_slot: int, visit_order: list[int]) -> bool:
    try:
        band_r0, band_r1, slots = _infer_layout(grid)
        found_slot, template_crop, _offset = _find_template_slot(grid, band_r0, band_r1, slots)
        if found_slot != template_slot:
            return False
        if len({_rotate_cw_times(template_crop, k).tobytes() for k in range(4)}) != 4:
            return False
        recovered = _simulate_path(grid, band_r0, band_r1, slots, template_slot)
        if recovered != visit_order:
            return False
        if recovered == sorted(recovered):
            return False
        if len(set(recovered)) != len(slots) - 1:
            return False
        oracle = _plan_output(grid, visit_order)
        solved = main(grid)
        if not np.array_equal(solved, oracle):
            return False
        for slot_idx in visit_order:
            c0, c1 = slots[slot_idx]
            band = solved[band_r0 : band_r1 + 1, c0:c1]
            if not np.isin(band, [BLUE, RED, BLACK, GREEN, MAROON]).all():
                return False
        return True
    except Exception:
        return False


def _generate_candidate() -> np.ndarray | None:
    grid = _base_grid()
    template_slot = random.randrange(SLOT_COUNT)
    visit_order = _expected_visit_order(template_slot)

    sprite = _random_template_sprite()
    row_offset = random.choice([1, 2])
    col_offset = random.choice([0, 1])
    _place_template(grid, template_slot, sprite, row_offset, col_offset)

    first_slot = visit_order[0]
    first_track = _slot_start(first_slot) + random.choice([1, 2])
    grid[0, first_track] = GREEN
    grid[1, first_track] = GREEN

    last_slot = visit_order[-1]
    reserved: dict[int, set[int]] = {slot_idx: set() for slot_idx in range(SLOT_COUNT)}
    reserved[first_slot].add(first_track)

    # Transition 1: first slot down -> bottom row -> second slot up (default left turn).
    second_slot = visit_order[1]
    second_track = _entry_col(second_slot, from_left=False)
    grid[GRID_H - 1, _turn_blocker_col(second_slot, moving_right=False)] = TEAL
    reserved[second_slot].add(second_track)

    # Transition 2: second slot up -> top row -> third slot down (default right turn).
    third_slot = visit_order[2]
    third_track = _entry_col(third_slot, from_left=True)
    grid[0, _turn_blocker_col(third_slot, moving_right=True)] = TEAL
    reserved[third_slot].add(third_track)

    # Transition 3: third slot down -> bottom row -> force right, then enter last slot upward.
    grid[GRID_H - 1, third_track - 1] = TEAL
    last_c0 = _slot_start(last_slot)
    last_up_track = last_c0
    grid[GRID_H - 1, last_c0 + 1] = TEAL
    reserved[last_slot].add(last_up_track)

    # Final slot: internal U-turn from upward column to a different downward column that ends at the marker.
    final_down_track = last_c0 + 2
    grid[3, last_up_track] = TEAL
    grid[4, last_c0 + 3] = TEAL
    grid[GRID_H - 2, final_down_track] = MAROON
    grid[GRID_H - 1, final_down_track] = MAROON
    reserved[last_slot].add(final_down_track)
    reserved[last_slot].update({last_c0 + 3})

    for slot_idx in visit_order:
        if slot_idx == template_slot:
            continue
        _add_decoy_obstacles(grid, slot_idx, reserved[slot_idx])

    if not _validate_instance(grid, template_slot, visit_order):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
