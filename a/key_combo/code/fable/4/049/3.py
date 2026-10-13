from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIRS = ("up", "right", "down", "left")
STEP_TO_SOURCE = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
FACE_TARGET = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

SPRITE_SHAPES = (
    (
        "00100",
        "01110",
        "11111",
        "01010",
        "00100",
    ),
    (
        "00100",
        "01110",
        "11111",
        "00110",
        "00100",
    ),
    (
        "00100",
        "01110",
        "11111",
        "01100",
        "00100",
    ),
)
CANONICAL_EYE = (2, 4)
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _unique_eye_color(sprite: np.ndarray) -> int:
    counts = Counter(int(v) for v in sprite.flatten() if int(v) != BG)
    singletons = [color for color, count in counts.items() if count == 1]
    assert len(singletons) == 1, "expected exactly one unique eye color in the main sprite"
    return int(singletons[0])


def _eye_side(sprite: np.ndarray, eye_color: int) -> str:
    h, w = map(int, sprite.shape)
    eye_coords = np.argwhere(sprite == eye_color)
    assert eye_coords.shape == (1, 2), "expected exactly one eye pixel"
    er, ec = map(int, eye_coords[0])
    distances = {
        "up": er,
        "down": h - 1 - er,
        "left": ec,
        "right": w - 1 - ec,
    }
    best = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == best]
    assert len(sides) == 1, "eye side must be unambiguous"
    return sides[0]


def _rotate_to_face(sprite: np.ndarray, eye_color: int, target_side: str) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(sprite, k)
        if _eye_side(rotated, eye_color) == target_side:
            return rotated.copy()
    raise AssertionError("failed to rotate sprite to the requested facing")


def _sprite_entries(sprite: np.ndarray, center: tuple[int, int]) -> list[tuple[int, int, int]]:
    cr, cc = center
    h, w = map(int, sprite.shape)
    ar, ac = h // 2, w // 2
    entries: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(sprite[r, c])
            if color == BG:
                continue
            entries.append((cr + r - ar, cc + c - ac, color))
    return entries


def _touches(entries: list[tuple[int, int, int]], occupied: np.ndarray) -> bool:
    for r, c, _color in entries:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(occupied.shape, nr, nc) and occupied[nr, nc]:
                return True
    return False


def _overlaps(entries: list[tuple[int, int, int]], occupied: np.ndarray) -> bool:
    return any(occupied[r, c] for r, c, _color in entries)


def _paint_entries(grid: np.ndarray, entries: list[tuple[int, int, int]]) -> None:
    for r, c, color in entries:
        grid[r, c] = int(color)


def _direction_from_marker(marker: tuple[int, int], center: tuple[int, int]) -> str:
    mr, mc = marker
    cr, cc = center
    if mc == cc and mr < cr:
        return "up"
    if mc == cc and mr > cr:
        return "down"
    if mr == cr and mc < cc:
        return "left"
    if mr == cr and mc > cc:
        return "right"
    raise AssertionError("marker must lie on a cardinal line through the main center")


def _recolor_copy(sprite: np.ndarray, eye_color: int, marker_color: int) -> np.ndarray:
    out = np.where(sprite != BG, int(marker_color), BG).astype(int)
    eye_coords = np.argwhere(sprite == eye_color)
    assert eye_coords.shape == (1, 2)
    er, ec = map(int, eye_coords[0])
    out[er, ec] = int(eye_color)
    return out


def _dock_on_line(
    grid_shape: tuple[int, int],
    sprite: np.ndarray,
    start_center: tuple[int, int],
    direction: str,
    occupied: np.ndarray,
) -> list[tuple[int, int, int]]:
    center = tuple(map(int, start_center))
    step = STEP_TO_SOURCE[direction]
    entries = _sprite_entries(sprite, center)
    assert all(_in_bounds(grid_shape, r, c) for r, c, _color in entries), "copy must stay within the grid"
    assert not _overlaps(entries, occupied), "copy start position overlaps existing foreground"

    for _ in range(max(grid_shape) + 5):
        if _touches(entries, occupied):
            return entries
        next_center = (center[0] + step[0], center[1] + step[1])
        next_entries = _sprite_entries(sprite, next_center)
        assert all(_in_bounds(grid_shape, r, c) for r, c, _color in next_entries), "copy walked out of bounds"
        assert not _overlaps(next_entries, occupied), "copy should stop before overlap"
        center = next_center
        entries = next_entries
    raise AssertionError("failed to dock a copy")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate and recolor a copy of the eyed main sprite for each marker, then dock those copies toward the source sprite.

    Input structure:
    - The grid has black background, one multi-pixel main sprite near the center, and several single-pixel summon markers.
    - The main sprite contains exactly one unique-color eye pixel whose side indicates the sprite's facing.
    - Every marker lies on the main sprite center's row or column, and marker colors are distinct from the sprite colors.

    Rule:
    - Keep the main sprite fixed and erase all markers.
    - For each marker, make a rotated copy of the main sprite whose eye side faces the main sprite, recolor the whole copy body to the marker color, and keep the eye color unchanged.
    - Slide each copy straight toward the main sprite along its marker line until it first becomes 4-neighbor adjacent to the already docked foreground on that same line; on each line, nearer markers dock before farther markers.

    Key insight:
    - Marker positions only encode direction, color, and per-line ordering: the required orientation is inferred by rotating the eye side to face the source sprite, then docking along that axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g != BG)
    assert comps, "expected foreground objects"
    main_candidates = [comp for comp in comps if len(comp) > 1]
    assert len(main_candidates) == 1, "expected exactly one non-singleton main sprite"
    main_coords = main_candidates[0]
    marker_coords = [comp[0] for comp in comps if len(comp) == 1]

    main_sprite, main_box = _crop_component(g, main_coords)
    r0, c0, r1, c1 = main_box
    h, w = map(int, main_sprite.shape)
    assert h % 2 == 1 and w % 2 == 1, "main sprite bbox must have an odd center"
    center = ((r0 + r1) // 2, (c0 + c1) // 2)

    eye_color = _unique_eye_color(main_sprite)

    out = np.full_like(g, BG)
    for r, c in main_coords:
        out[r, c] = int(g[r, c])
    final_occupied = out != BG

    line_groups = {direction: [] for direction in DIRS}
    for marker in marker_coords:
        line_groups[_direction_from_marker(marker, center)].append(marker)

    for direction in DIRS:
        markers = line_groups[direction]
        if not markers:
            continue
        if direction in ("up", "down"):
            markers.sort(key=lambda pos: abs(pos[0] - center[0]))
        else:
            markers.sort(key=lambda pos: abs(pos[1] - center[1]))

        line_occupied = out != BG
        target_sprite = _rotate_to_face(main_sprite, eye_color, FACE_TARGET[direction])
        for marker in markers:
            marker_color = int(g[marker])
            copy_sprite = _recolor_copy(target_sprite, eye_color, marker_color)
            entries = _dock_on_line(g.shape, copy_sprite, marker, direction, line_occupied)
            assert not _overlaps(entries, final_occupied), "copies from different directions must not overlap"
            _paint_entries(out, entries)
            for r, c, _color in entries:
                line_occupied[r, c] = True
                final_occupied[r, c] = True

    return out


def _make_canonical_sprite(body_color: int, eye_color: int) -> np.ndarray:
    mask = random.choice(SPRITE_SHAPES)
    sprite = np.zeros((5, 5), dtype=int)
    for r, row in enumerate(mask):
        for c, ch in enumerate(row):
            if ch == "1":
                sprite[r, c] = int(body_color)
    sprite[CANONICAL_EYE] = int(eye_color)
    return sprite


def _sample_spaced(values: list[int], count: int, min_gap: int = 3) -> list[int] | None:
    if count == 1:
        return [random.choice(values)] if values else None
    candidates = values[:]
    random.shuffle(candidates)
    picked: list[int] = []
    for value in candidates:
        if all(abs(value - other) >= min_gap for other in picked):
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return None


def _place_centered(grid: np.ndarray, sprite: np.ndarray, center: tuple[int, int]) -> None:
    entries = _sprite_entries(sprite, center)
    assert all(_in_bounds(grid.shape, r, c) for r, c, _color in entries)
    for r, c, color in entries:
        assert grid[r, c] == BG
        grid[r, c] = int(color)


def _choose_counts(directions: list[str]) -> dict[str, int]:
    remaining = 6
    counts: dict[str, int] = {}
    doubled = False
    for i, direction in enumerate(directions):
        slots_left = len(directions) - i - 1
        max_for_this = min(2, remaining - slots_left)
        count = 2 if max_for_this >= 2 and random.random() < 0.6 else 1
        count = min(count, max_for_this)
        counts[direction] = count
        remaining -= count
        doubled = doubled or count == 2
    if not doubled:
        bumpable = [direction for direction in directions if counts[direction] == 1 and remaining >= 1]
        if bumpable:
            counts[random.choice(bumpable)] = 2
    return counts


def _validate_generated(inp: np.ndarray, out: np.ndarray, eye_color: int, marker_colors: list[int]) -> bool:
    if inp.shape != out.shape:
        return False
    if int(np.count_nonzero(out == eye_color)) != 1 + len(marker_colors):
        return False
    for color in marker_colors:
        if int(np.count_nonzero(out == color)) <= 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)

    body_color, eye_color = random.sample(COLORS, 2)
    main_sprite = _make_canonical_sprite(body_color, eye_color)
    main_sprite = np.rot90(main_sprite, random.randrange(4)).copy()

    center_r = random.randint(8, h - 9)
    center_c = random.randint(8, w - 9)
    center = (center_r, center_c)

    up_vals = list(range(2, center_r - 5))
    down_vals = list(range(center_r + 6, h - 2))
    left_vals = list(range(2, center_c - 5))
    right_vals = list(range(center_c + 6, w - 2))
    pools = {
        "up": up_vals,
        "down": down_vals,
        "left": left_vals,
        "right": right_vals,
    }
    if sum(bool(values) for values in pools.values()) < 2:
        return None

    direction_count = random.randint(2, 4)
    directions = random.sample(list(DIRS), direction_count)
    counts = _choose_counts(directions)

    available_colors = [color for color in COLORS if color not in {body_color, eye_color}]
    if sum(counts.values()) > len(available_colors):
        return None
    marker_colors = random.sample(available_colors, sum(counts.values()))
    color_idx = 0

    placements: list[tuple[int, int, int]] = []
    for direction in directions:
        desired = counts[direction]
        if len(pools[direction]) < desired:
            return None
        picked = _sample_spaced(pools[direction], desired)
        if picked is None:
            return None
        for value in picked:
            color = marker_colors[color_idx]
            color_idx += 1
            if direction == "up":
                placements.append((value, center_c, color))
            elif direction == "down":
                placements.append((value, center_c, color))
            elif direction == "left":
                placements.append((center_r, value, color))
            else:
                placements.append((center_r, value, color))

    grid = np.zeros((h, w), dtype=int)
    try:
        _place_centered(grid, main_sprite, center)
    except AssertionError:
        return None

    for r, c, color in placements:
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _validate_generated(grid, out, eye_color, marker_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 049/3")
