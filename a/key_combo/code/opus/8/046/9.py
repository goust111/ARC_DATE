from __future__ import annotations

import random
from collections import deque

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

GRID_SHAPE = (11, 11)
CROSS_ROW = 5
CROSS_COL = 5
FRAME_COLORS = (PINK, ORANGE, MAROON)
RANK_TO_COLOR = {0: RED, 1: BLUE, 2: YELLOW}
QUADRANT_ORDER = ("tl", "tr", "bl", "br")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _count_enclosed_regions(walls: np.ndarray) -> int:
    h, w = walls.shape
    bg = ~walls
    count = 0
    seen = np.zeros_like(bg, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_edge = True
                        q.append((nr, nc))
            if not touches_edge:
                count += 1
    return count


def _outer_ring() -> np.ndarray:
    walls = np.zeros((5, 5), dtype=bool)
    walls[0, :] = True
    walls[4, :] = True
    walls[:, 0] = True
    walls[:, 4] = True
    return walls


def _build_templates() -> dict[str, np.ndarray]:
    outer = _outer_ring()
    templates: dict[str, np.ndarray] = {}

    walls = outer.copy()
    templates["rooms1"] = walls

    walls = outer.copy()
    walls[:, 2] = True
    templates["rooms2_v"] = walls

    walls = outer.copy()
    walls[2, :] = True
    templates["rooms2_h"] = walls

    walls = outer.copy()
    walls[:, 2] = True
    walls[2, 0:3] = True
    templates["rooms3_left"] = walls

    walls = outer.copy()
    walls[:, 2] = True
    walls[2, 2:5] = True
    templates["rooms3_right"] = walls

    walls = outer.copy()
    walls[2, :] = True
    walls[0:3, 2] = True
    templates["rooms3_top"] = walls

    walls = outer.copy()
    walls[2, :] = True
    walls[2:5, 2] = True
    templates["rooms3_bottom"] = walls

    walls = outer.copy()
    walls[:, 2] = True
    walls[2, :] = True
    templates["rooms4"] = walls
    return templates


TEMPLATES = _build_templates()
TEMPLATES_BY_COUNT: dict[int, list[np.ndarray]] = {}
for _name, _walls in TEMPLATES.items():
    TEMPLATES_BY_COUNT.setdefault(_count_enclosed_regions(_walls), []).append(_walls)


def _find_cross(g: np.ndarray) -> tuple[int, int]:
    teal_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == TEAL)]
    teal_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == TEAL)]
    assert len(teal_rows) == 1 and len(teal_cols) == 1
    return int(teal_rows[0]), int(teal_cols[0])


def _extract_quadrants(g: np.ndarray) -> list[np.ndarray]:
    cross_r, cross_c = _find_cross(g)
    return [
        g[:cross_r, :cross_c],
        g[:cross_r, cross_c + 1 :],
        g[cross_r + 1 :, :cross_c],
        g[cross_r + 1 :, cross_c + 1 :],
    ]


def _frame_color(subgrid: np.ndarray) -> int:
    colors = sorted(set(int(v) for v in np.unique(subgrid)) - {BLACK, GREY})
    assert len(colors) == 1
    return int(colors[0])


def _best_template_for_count(mask: np.ndarray) -> tuple[np.ndarray, int]:
    observed = int(mask.sum())
    assert observed > 0
    best_by_count: dict[int, tuple[float, int, np.ndarray]] = {}
    for count, templates in TEMPLATES_BY_COUNT.items():
        for walls in templates:
            if np.any(mask & ~walls):
                continue
            size = int(walls.sum())
            score = observed / size
            candidate = (score, -size, walls)
            prev = best_by_count.get(count)
            if prev is None or candidate[:2] > prev[:2]:
                best_by_count[count] = candidate
    assert best_by_count
    ranked = sorted(
        ((count, vals[0], vals[1], vals[2]) for count, vals in best_by_count.items()),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    best_count, best_score, best_size_key, best_walls = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        assert (best_score, best_size_key) != (second[1], second[2])
    return best_walls.copy(), int(best_count)


def _infer_room_count(subgrid: np.ndarray) -> int:
    color = _frame_color(subgrid)
    observed = subgrid == color
    walls, count = _best_template_for_count(observed)
    assert _count_enclosed_regions(walls) == count
    return int(count)


def _sprite_bbox(subgrid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(subgrid == GREEN)
    assert box is not None
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 2 and c1 - c0 + 1 >= 2
    assert (r1 - r0 + 1) % 2 == 0 and (c1 - c0 + 1) % 2 == 0
    return r0, c0, r1, c1


def _sprite_quadrant_slices(box: tuple[int, int, int, int]) -> list[tuple[slice, slice]]:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rm = r0 + h // 2
    cm = c0 + w // 2
    return [
        (slice(r0, rm), slice(c0, cm)),
        (slice(r0, rm), slice(cm, c1 + 1)),
        (slice(rm, r1 + 1), slice(c0, cm)),
        (slice(rm, r1 + 1), slice(cm, c1 + 1)),
    ]


def _apply_recolor(grid: np.ndarray, sprite_pos: int, pos_to_color: dict[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    quadrants = _extract_quadrants(out)
    sprite = quadrants[sprite_pos].copy()
    box = _sprite_bbox(sprite)
    for pos, (rs, cs) in enumerate(_sprite_quadrant_slices(box)):
        if pos == sprite_pos:
            continue
        color = pos_to_color[pos]
        patch = sprite[rs, cs]
        patch[patch == GREEN] = color
        sprite[rs, cs] = patch
    cross_r, cross_c = _find_cross(out)
    starts = [(0, 0), (0, cross_c + 1), (cross_r + 1, 0), (cross_r + 1, cross_c + 1)]
    r0, c0 = starts[sprite_pos]
    out[r0 : r0 + 5, c0 : c0 + 5] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the sprite quadrants by sorting the reconstructed frame room counts.

    Input structure:
    - An 11x11 black grid is split into four 5x5 quadrants by one full teal row and one full teal column.
    - Exactly one quadrant contains a green sprite, while the other three contain single-color partition frames
      whose wall pixels may be replaced by grey noise.
    - Each frame is an axis-aligned 1-pixel wall drawing whose complete form encloses 1 to 4 black regions.

    Rule:
    - Identify the sprite quadrant and the three frame quadrants from the teal cross.
    - For each frame quadrant, ignore grey pixels, reconstruct the most plausible 5x5 partition frame, and count
      its enclosed 4-connected black regions.
    - Sort the three frame quadrants by room count from small to large and map them to red, blue, and yellow.
    - Split the sprite's green bounding box into four equal local quadrants; recolor each local quadrant whose
      global counterpart holds a frame, while the local quadrant corresponding to the sprite's own global position stays green.

    Key insight:
    - The noisy frame geometry does not directly move anywhere; it first determines a room-count ranking, and that
      ranking becomes a color legend transferred to the corresponding sprite quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE
    quadrants = _extract_quadrants(g)
    sprite_positions = [idx for idx, sub in enumerate(quadrants) if np.any(sub == GREEN)]
    assert len(sprite_positions) == 1
    sprite_pos = int(sprite_positions[0])

    room_counts: dict[int, int] = {}
    for pos, subgrid in enumerate(quadrants):
        if pos == sprite_pos:
            continue
        room_counts[pos] = _infer_room_count(subgrid)

    ordered_positions = [pos for pos, _count in sorted(room_counts.items(), key=lambda item: item[1])]
    assert len(set(room_counts.values())) == 3
    pos_to_color = {pos: RANK_TO_COLOR[rank] for rank, pos in enumerate(ordered_positions)}
    return _apply_recolor(g, sprite_pos, pos_to_color)


def _random_connected_sprite(rng: random.Random) -> np.ndarray:
    cells = [(r, c) for r in range(4) for c in range(4)]
    quadrants = [
        {(r, c) for r in range(0, 2) for c in range(0, 2)},
        {(r, c) for r in range(0, 2) for c in range(2, 4)},
        {(r, c) for r in range(2, 4) for c in range(0, 2)},
        {(r, c) for r in range(2, 4) for c in range(2, 4)},
    ]
    while True:
        target = rng.randint(8, 14)
        start = rng.choice(cells)
        chosen = {start}
        while len(chosen) < target:
            frontier = []
            for r, c in sorted(chosen):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in chosen:
                        frontier.append((nr, nc))
            if not frontier:
                break
            chosen.add(rng.choice(frontier))
        mask = np.zeros((4, 4), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if len(_components(mask)) != 1:
            continue
        if not all(mask[0, :]) and not np.any(mask[0, :]):
            continue
        if not np.any(mask[3, :]) or not np.any(mask[:, 0]) or not np.any(mask[:, 3]):
            continue
        if not all(np.any(mask[list({r for r, _c in quad}), :][:, list({c for _r, c in quad})]) for quad in quadrants):
            continue
        if np.all(mask):
            continue
        return mask


def _place_sprite(quadrant: np.ndarray, mask4: np.ndarray, rng: random.Random) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = quadrant.copy()
    top = rng.randint(0, 1)
    left = rng.randint(0, 1)
    out[top : top + 4, left : left + 4][mask4] = GREEN
    return out, (top, left, top + 3, left + 3)


def _template_names_for_count(count: int) -> list[str]:
    return [name for name, walls in TEMPLATES.items() if _count_enclosed_regions(walls) == count]


def _damage_frame(template_name: str, color: int, rng: random.Random) -> np.ndarray:
    walls = TEMPLATES[template_name]
    wall_cells = [(int(r), int(c)) for r, c in np.argwhere(walls)]
    empty_cells = [(r, c) for r in range(5) for c in range(5) if not walls[r, c]]
    while True:
        sub = np.zeros((5, 5), dtype=int)
        sub[walls] = color
        max_damage = min(4, len(wall_cells) - 6)
        damage_count = rng.randint(1, max_damage)
        for r, c in rng.sample(wall_cells, damage_count):
            sub[r, c] = GREY
        extra_noise = rng.randint(0, 2)
        if extra_noise and empty_cells:
            for r, c in rng.sample(empty_cells, min(extra_noise, len(empty_cells))):
                sub[r, c] = GREY
        if _infer_room_count(sub) == _count_enclosed_regions(walls):
            return sub


def _quadrant_origin(pos: int) -> tuple[int, int]:
    if pos == 0:
        return 0, 0
    if pos == 1:
        return 0, CROSS_COL + 1
    if pos == 2:
        return CROSS_ROW + 1, 0
    return CROSS_ROW + 1, CROSS_COL + 1


def _expected_output_from_truth(inp: np.ndarray, sprite_pos: int, room_counts: dict[int, int]) -> np.ndarray:
    ordered_positions = [pos for pos, _count in sorted(room_counts.items(), key=lambda item: item[1])]
    pos_to_color = {pos: RANK_TO_COLOR[rank] for rank, pos in enumerate(ordered_positions)}
    return _apply_recolor(inp, sprite_pos, pos_to_color)


def _sprite_quadrant_has_green(subgrid: np.ndarray, box: tuple[int, int, int, int], pos: int) -> bool:
    rs, cs = _sprite_quadrant_slices(box)[pos]
    return bool(np.any(subgrid[rs, cs] == GREEN))


def _validate_example(inp: np.ndarray, sprite_pos: int, room_counts: dict[int, int]) -> bool:
    if inp.shape != GRID_SHAPE:
        return False
    quadrants = _extract_quadrants(inp)
    if sum(int(np.any(sub == GREEN)) for sub in quadrants) != 1:
        return False
    if len(set(room_counts.values())) != 3:
        return False
    try:
        solved_counts = {
            pos: _infer_room_count(sub)
            for pos, sub in enumerate(quadrants)
            if pos != sprite_pos
        }
    except AssertionError:
        return False
    if solved_counts != room_counts:
        return False
    sprite_sub = quadrants[sprite_pos]
    try:
        box = _sprite_bbox(sprite_sub)
    except AssertionError:
        return False
    if box[2] - box[0] != 3 or box[3] - box[1] != 3:
        return False
    if not all(_sprite_quadrant_has_green(sprite_sub, box, pos) for pos in range(4)):
        return False
    expected = _expected_output_from_truth(inp, sprite_pos, room_counts)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, actual):
        return False
    if not {RED, BLUE, YELLOW, GREEN}.issubset(set(int(v) for v in np.unique(actual))):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        grid = np.zeros(GRID_SHAPE, dtype=int)
        grid[CROSS_ROW, :] = TEAL
        grid[:, CROSS_COL] = TEAL

        sprite_pos = rng.randrange(4)
        frame_positions = [pos for pos in range(4) if pos != sprite_pos]
        frame_colors = list(FRAME_COLORS)
        rng.shuffle(frame_colors)

        counts = rng.sample([1, 2, 3, 4], 3)
        room_counts = {pos: count for pos, count in zip(frame_positions, counts)}

        for pos, color in zip(frame_positions, frame_colors):
            template_name = rng.choice(_template_names_for_count(room_counts[pos]))
            quadrant = _damage_frame(template_name, color, rng)
            r0, c0 = _quadrant_origin(pos)
            grid[r0 : r0 + 5, c0 : c0 + 5] = quadrant

        sprite_mask = _random_connected_sprite(rng)
        sprite_quadrant = np.zeros((5, 5), dtype=int)
        sprite_quadrant, _box = _place_sprite(sprite_quadrant, sprite_mask, rng)
        r0, c0 = _quadrant_origin(sprite_pos)
        grid[r0 : r0 + 5, c0 : c0 + 5] = sprite_quadrant

        if _validate_example(grid, sprite_pos, room_counts):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
