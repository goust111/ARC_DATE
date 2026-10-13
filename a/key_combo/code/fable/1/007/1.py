from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

REPAIR_ONLY = "repair"
DOOR_ONLY = "door"
BOTH = "both"

ROOM_COUNT_CHOICES = (2, 3)
HEIGHT_RANGE = (4, 7)
TOP_BOTTOM_MARGIN_RANGE = (1, 3)
VERTICAL_GAP_RANGE = (2, 4)
WIDTH_RANGE = (16, 29)
MIN_SPAN = 3
MAX_GENERATION_ATTEMPTS = 400


def _mirror_mask(mask: np.ndarray, axis2: int) -> np.ndarray | None:
    h, w = mask.shape
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    mirrored_cols = int(axis2) - coords[:, 1]
    if np.any((mirrored_cols < 0) | (mirrored_cols >= w)):
        return None

    mirrored = np.zeros((h, w), dtype=bool)
    mirrored[coords[:, 0], mirrored_cols] = True
    return mirrored


def _axis_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    if h == 0 or w == 0 or not np.any(mask):
        return []

    scores: list[tuple[int, int]] = []
    for axis2 in range(2 * w - 1):
        mirrored = _mirror_mask(mask, axis2)
        if mirrored is None:
            continue
        added = int(np.count_nonzero(mirrored & ~mask))
        scores.append((added, int(axis2)))
    return scores


def _infer_vertical_axis(mask: np.ndarray) -> int:
    scores = _axis_candidates(mask)
    assert scores, "expected at least one valid vertical symmetry axis"
    best = min(score for score, _axis2 in scores)
    winners = [axis2 for score, axis2 in scores if score == best]
    assert len(winners) == 1, "expected a unique best vertical symmetry axis"
    return winners[0]


def _repair_walls(mask: np.ndarray, axis2: int) -> np.ndarray:
    mirrored = _mirror_mask(mask, axis2)
    assert mirrored is not None, "axis must keep mirrored wall cells inside the grid"
    return mask | mirrored


def _exterior_background(walls: np.ndarray) -> np.ndarray:
    h, w = walls.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if not walls[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w:
                add(rr, cc)

    return exterior


def _render_from_walls(walls: np.ndarray) -> np.ndarray:
    out = np.full(walls.shape, BLACK, dtype=int)
    out[walls] = RED
    enclosed = (~walls) & (~_exterior_background(walls))
    out[enclosed] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repair one-sided wall damage by vertical symmetry, then fill only truly closed rooms blue.

    Input structure:
    - A black background contains one-cell-thick red wall segments.
    - The underlying red wall pattern is symmetric about one vertical axis, but
      some wall cells have been erased to black as gaps.
    - Blue is absent from the input.

    Rule:
    - Infer the unique vertical symmetry axis that minimizes one-sided wall
      mismatches.
    - Mirror-complete the observed red walls across that axis, which restores
      every single-sided damage while leaving paired gaps open.
    - Treat the repaired red cells as 4-connected walls, flood-fill background
      from the grid border, and recolor only the unreachable background cells
      blue.

    Key insight:
    - A gap whose mirror mate is still red is damage and must be repaired, while
      a gap whose whole mirror orbit is missing is a true door; enclosed blue
      regions can only be determined after that asymmetric damage is restored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED}, "input should contain only black background and red walls"

    red = g == RED
    assert np.any(red), "expected at least one red wall cell"

    axis2 = _infer_vertical_axis(red)
    repaired = _repair_walls(red, axis2)
    return _render_from_walls(repaired)


def _axis_max_span(width: int, axis2: int) -> int:
    if axis2 % 2 == 0:
        center = axis2 // 2
        return min(center - 1, width - 2 - center)

    left_center = axis2 // 2
    right_center = left_center + 1
    return min(left_center - 1, width - 2 - right_center)


def _centered_room(axis2: int, top: int, height: int, span: int) -> tuple[int, int, int, int]:
    if axis2 % 2 == 0:
        center = axis2 // 2
        left = center - span
        right = center + span
    else:
        left_center = axis2 // 2
        right_center = left_center + 1
        left = left_center - span
        right = right_center + span
    return int(top), int(left), int(top + height - 1), int(right)


def _draw_outline(mask: np.ndarray, room: tuple[int, int, int, int]) -> None:
    top, left, bottom, right = room
    mask[top, left : right + 1] = True
    mask[bottom, left : right + 1] = True
    mask[top : bottom + 1, left] = True
    mask[top : bottom + 1, right] = True


def _room_orbits(room: tuple[int, int, int, int], axis2: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    top, left, bottom, right = room
    orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for c in range(left + 1, right):
        mc = int(axis2) - int(c)
        if left < mc < right and c < mc:
            orbits.append(((top, c), (top, mc)))
            orbits.append(((bottom, c), (bottom, mc)))

    for r in range(top + 1, bottom):
        orbits.append(((r, left), (r, right)))

    return orbits


def _orbit_gap_distance(
    orbit_a: tuple[tuple[int, int], tuple[int, int]],
    orbit_b: tuple[tuple[int, int], tuple[int, int]],
) -> int:
    return min(
        abs(ar - br) + abs(ac - bc)
        for ar, ac in orbit_a
        for br, bc in orbit_b
    )


def _pick_spaced_orbits(
    orbits: list[tuple[tuple[int, int], tuple[int, int]]],
    count: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    pool = list(orbits)
    random.shuffle(pool)
    picked: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for orbit in pool:
        if all(_orbit_gap_distance(orbit, other) > 1 for other in picked):
            picked.append(orbit)
            if len(picked) == count:
                return picked
    return None


def _remove_damage(raw: np.ndarray, orbit: tuple[tuple[int, int], tuple[int, int]]) -> None:
    raw[random.choice(orbit)] = False


def _remove_door(raw: np.ndarray, orbit: tuple[tuple[int, int], tuple[int, int]]) -> None:
    for r, c in orbit:
        raw[r, c] = False


def _room_probe(room: tuple[int, int, int, int]) -> tuple[int, int]:
    top, left, _bottom, _right = room
    return top + 1, left + 1


def _build_geometry() -> tuple[int, int, int, list[tuple[int, int, int, int]], list[str]] | None:
    room_count = random.choice(ROOM_COUNT_CHOICES)
    categories = [REPAIR_ONLY, DOOR_ONLY]
    if room_count == 3:
        categories.append(random.choice((REPAIR_ONLY, DOOR_ONLY, BOTH)))
    random.shuffle(categories)

    widths = list(range(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    random.shuffle(widths)
    for width in widths:
        axes = [axis2 for axis2 in range(2 * width - 1) if _axis_max_span(width, axis2) >= MIN_SPAN]
        random.shuffle(axes)
        for axis2 in axes:
            max_span = _axis_max_span(width, axis2)
            heights = [random.randint(*HEIGHT_RANGE) for _ in range(room_count)]
            gaps = [random.randint(*VERTICAL_GAP_RANGE) for _ in range(room_count - 1)]
            top_margin = random.randint(*TOP_BOTTOM_MARGIN_RANGE)
            bottom_margin = random.randint(*TOP_BOTTOM_MARGIN_RANGE)
            height = top_margin + sum(heights) + sum(gaps) + bottom_margin
            if height > 30:
                continue

            spans = [random.randint(MIN_SPAN, max_span) for _ in range(room_count)]
            top = top_margin
            rooms: list[tuple[int, int, int, int]] = []
            for room_height, span in zip(heights, spans):
                rooms.append(_centered_room(axis2, top, room_height, span))
                top += room_height
                if len(rooms) < room_count:
                    top += gaps[len(rooms) - 1]

            return int(height), int(width), int(axis2), rooms, categories
    return None


def _carve_room(
    raw: np.ndarray,
    room: tuple[int, int, int, int],
    axis2: int,
    category: str,
) -> bool:
    orbits = _room_orbits(room, axis2)
    if category == REPAIR_ONLY:
        damage_count = random.choice((1, 2))
        picked = _pick_spaced_orbits(orbits, damage_count)
        if picked is None:
            return False
        for orbit in picked:
            _remove_damage(raw, orbit)
        return True

    if category == DOOR_ONLY:
        door_count = random.choice((1, 2))
        picked = _pick_spaced_orbits(orbits, door_count)
        if picked is None:
            return False
        for orbit in picked:
            _remove_door(raw, orbit)
        return True

    picked = _pick_spaced_orbits(orbits, 2)
    if picked is None:
        return False
    _remove_door(raw, picked[0])
    _remove_damage(raw, picked[1])
    return True


def _semantic_checks(
    base_walls: np.ndarray,
    raw_walls: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    categories: list[str],
    axis2: int,
) -> bool:
    raw_grid = np.where(raw_walls, RED, BLACK).astype(int)
    raw_output = _render_from_walls(raw_walls)
    repaired_walls = _repair_walls(raw_walls, axis2)
    final_output = _render_from_walls(repaired_walls)
    closed_output = _render_from_walls(base_walls)

    try:
        inferred_axis = _infer_vertical_axis(raw_walls)
    except AssertionError:
        return False
    if inferred_axis != axis2:
        return False

    if not np.array_equal(main(raw_grid), final_output):
        return False
    if np.array_equal(raw_output, final_output):
        return False
    if np.array_equal(closed_output, final_output):
        return False
    if not np.any(final_output == BLUE):
        return False
    if not np.any(final_output == BLACK):
        return False

    matching_axes: list[int] = []
    for _score, cand_axis in _axis_candidates(raw_walls):
        cand_output = _render_from_walls(_repair_walls(raw_walls, cand_axis))
        if np.array_equal(cand_output, final_output):
            matching_axes.append(cand_axis)
    if matching_axes != [axis2]:
        return False

    for room, category in zip(rooms, categories):
        probe = _room_probe(room)
        if closed_output[probe] != BLUE:
            return False
        raw_blue = raw_output[probe] == BLUE
        final_blue = final_output[probe] == BLUE
        if category == REPAIR_ONLY:
            if raw_blue or not final_blue:
                return False
        else:
            if final_blue:
                return False

    return True


def _attempt_generate() -> np.ndarray | None:
    built = _build_geometry()
    if built is None:
        return None

    height, width, axis2, rooms, categories = built
    base_walls = np.zeros((height, width), dtype=bool)
    for room in rooms:
        _draw_outline(base_walls, room)

    raw_walls = base_walls.copy()
    for room, category in zip(rooms, categories):
        if not _carve_room(raw_walls, room, axis2, category):
            return None

    if np.array_equal(raw_walls, base_walls):
        return None
    if not _semantic_checks(base_walls, raw_walls, rooms, categories, axis2):
        return None
    return np.where(raw_walls, RED, BLACK).astype(int)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-wall ARC input")
