from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _cells_from_patch(patch: np.ndarray) -> list[tuple[int, int, int]]:
    rr, cc = np.nonzero(patch != BG)
    vals = patch[rr, cc]
    return [(int(r), int(c), int(v)) for r, c, v in zip(rr, cc, vals)]


def _placed_coords(sprite: dict, top: int | None = None, left: int | None = None) -> set[tuple[int, int]]:
    if top is None:
        top = int(sprite["top"])
    if left is None:
        left = int(sprite["left"])
    return {(top + r, left + c) for r, c, _ in sprite["cells"]}


def _touches_occupied(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in occupied:
            return True
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _parse_sprite(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    color_counts = Counter(int(grid[r, c]) for r, c in coords)
    assert len(color_counts) == 2, "each sprite must have one body color and one marker color"
    marker_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(marker_colors) == 1, "sprite marker must be a singleton pixel"
    marker_color = int(marker_colors[0])
    body_colors = [color for color in color_counts if color != marker_color]
    assert len(body_colors) == 1
    body_color = int(body_colors[0])

    marker_r, marker_c = next((r, c) for r, c in coords if int(grid[r, c]) == marker_color)
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])

    directions: list[tuple[int, int]] = []
    if w % 2 == 1 and (marker_r, marker_c) == (r0, c0 + w // 2):
        directions.append((-1, 0))
    if w % 2 == 1 and (marker_r, marker_c) == (r1, c0 + w // 2):
        directions.append((1, 0))
    if h % 2 == 1 and (marker_r, marker_c) == (r0 + h // 2, c0):
        directions.append((0, -1))
    if h % 2 == 1 and (marker_r, marker_c) == (r0 + h // 2, c1):
        directions.append((0, 1))
    assert len(directions) == 1, "marker must lie at a bbox edge midpoint"

    return {
        "patch": patch,
        "cells": _cells_from_patch(patch),
        "top": r0,
        "left": c0,
        "dir": directions[0],
        "body_color": body_color,
        "marker_color": marker_color,
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]], tuple[int, int], int]:
    moving_mask = (grid != BG) & (grid != WALL)
    components = _components(moving_mask)
    controls: list[tuple[int, int]] = []
    sprites: list[dict] = []
    for comp in components:
        if len(comp) == 1:
            controls.append(comp[0])
        else:
            sprites.append(_parse_sprite(grid, comp))
    assert len(controls) == 1, "expected exactly one isolated control pixel"
    assert sprites, "expected at least one sprite"
    wall_coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == WALL)}
    assert wall_coords, "expected at least one wall segment"

    control = controls[0]
    control_color = int(grid[control])
    starters = [idx for idx, sprite in enumerate(sprites) if int(sprite["marker_color"]) == control_color]
    assert len(starters) == 1, "control color must match exactly one sprite marker"
    return sprites, wall_coords, control, starters[0]


def _slide_sprite(idx: int, sprites: list[dict], wall_coords: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[int | None, str]:
    h, w = shape
    sprite = sprites[idx]
    dr, dc = sprite["dir"]
    top = int(sprite["top"])
    left = int(sprite["left"])
    moved = False

    while True:
        next_coords: list[tuple[int, int]] = []
        edge_block = False
        for rr, cc, _ in sprite["cells"]:
            nr = top + rr + dr
            nc = left + cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                edge_block = True
                break
            next_coords.append((nr, nc))
        if edge_block:
            assert moved, "triggered sprite must move at least one step"
            sprite["top"], sprite["left"] = top, left
            return None, "edge"

        next_set = set(next_coords)
        if next_set & wall_coords:
            assert moved, "triggered sprite must move at least one step"
            sprite["top"], sprite["left"] = top, left
            return None, "wall"

        colliders = []
        for other_idx, other in enumerate(sprites):
            if other_idx == idx:
                continue
            if next_set & _placed_coords(other):
                colliders.append(other_idx)
        if colliders:
            assert moved, "triggered sprite must move at least one step"
            assert len(colliders) == 1, "each slide may collide with at most one sprite"
            sprite["top"], sprite["left"] = top, left
            return colliders[0], "sprite"

        top += dr
        left += dc
        moved = True


def _transform(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites, wall_coords, control, current = _parse_scene(g)
    order: list[int] = []
    reasons: list[str] = []
    seen: set[int] = set()
    while True:
        assert current not in seen, "collision chain must not cycle"
        seen.add(current)
        order.append(current)
        blocker, reason = _slide_sprite(current, sprites, wall_coords, g.shape)
        reasons.append(reason)
        if blocker is None:
            break
        current = blocker

    out = np.full_like(g, BG)
    for r, c in wall_coords:
        out[r, c] = WALL
    for sprite in sprites:
        for rr, cc, color in sprite["cells"]:
            out[int(sprite["top"]) + rr, int(sprite["left"]) + cc] = int(color)

    if return_trace:
        return out, {"order": order, "reasons": reasons, "control": control}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trigger the matching sprite and propagate a sequential slide chain through sprite-to-sprite collisions.

    Input structure:
    - The background is black, and teal pixels form fixed wall segments.
    - Several 4-connected sprites each contain exactly one embedded marker pixel of a different color; the marker sits at the midpoint of one bbox edge and sets that sprite's slide direction.
    - Exactly one isolated control pixel appears elsewhere, and its color matches exactly one sprite marker.

    Rule:
    - The sprite whose embedded marker matches the control color moves first along its own marker direction until one more step would hit a wall, the grid edge, or another sprite.
    - If it stops because of another sprite, that contacted sprite is triggered next and slides along its own marker direction by the same rule; untriggered sprites never move.
    - Erase the control pixel, keep the wall segments fixed, and move each triggered sprite together with its embedded marker.

    Key insight:
    - The control pixel only chooses the first mover; the domino effect then follows collision order, while each awakened sprite uses its own embedded direction rather than inheriting the previous one.
    """
    return _transform(grid)


def _random_body_mask(h: int, w: int) -> np.ndarray:
    family = random.choice(["rect", "rounded", "cross"])
    if family == "rect":
        return np.ones((h, w), dtype=bool)
    if family == "rounded":
        mask = np.ones((h, w), dtype=bool)
        mask[0, 0] = False
        mask[0, -1] = False
        mask[-1, 0] = False
        mask[-1, -1] = False
        return mask

    mask = np.zeros((h, w), dtype=bool)
    row_half = 1 if h >= 5 else 0
    col_half = 1 if w >= 5 else 0
    r0 = max(0, h // 2 - row_half)
    r1 = min(h, h // 2 + row_half + 1)
    c0 = max(0, w // 2 - col_half)
    c1 = min(w, w // 2 + col_half + 1)
    mask[r0:r1, :] = True
    mask[:, c0:c1] = True
    return mask


def _marker_local(direction: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    if direction == (-1, 0):
        return 0, w // 2
    if direction == (1, 0):
        return h - 1, w // 2
    if direction == (0, -1):
        return h // 2, 0
    assert direction == (0, 1)
    return h // 2, w - 1


def _make_sprite(direction: tuple[int, int], body_color: int, marker_color: int) -> dict:
    h = random.choices([3, 5, 7], weights=[4, 4, 1])[0]
    w = random.choices([3, 5, 7], weights=[4, 4, 1])[0]
    patch = np.zeros((h, w), dtype=int)
    mask = _random_body_mask(h, w)
    mr, mc = _marker_local(direction, h, w)
    assert mask[mr, mc], "shape family must keep all edge midpoints occupied"
    patch[mask] = int(body_color)
    patch[mr, mc] = int(marker_color)
    return {
        "patch": patch,
        "cells": _cells_from_patch(patch),
        "top": 0,
        "left": 0,
        "dir": direction,
        "body_color": int(body_color),
        "marker_color": int(marker_color),
    }


def _coords_for(sprite: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c, _ in sprite["cells"]}


def _can_place_exact(sprite: dict, top: int, left: int, shape: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    h, w = shape
    ph, pw = sprite["patch"].shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    return not (_coords_for(sprite, top, left) & occupied)


def _max_back_steps(sprite: dict, final_top: int, final_left: int, direction: tuple[int, int], shape: tuple[int, int], occupied: set[tuple[int, int]]) -> int:
    dr, dc = direction
    steps = 0
    while _can_place_exact(sprite, final_top - (steps + 1) * dr, final_left - (steps + 1) * dc, shape, occupied):
        steps += 1
    return steps


def _possible_starts(
    sprite: dict,
    finals: list[tuple[int, int]],
    direction: tuple[int, int],
    shape: tuple[int, int],
    path_occupied: set[tuple[int, int]],
    input_occupied: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    dr, dc = direction
    starts: list[tuple[int, int, int]] = []
    for final_top, final_left in finals:
        max_steps = _max_back_steps(sprite, final_top, final_left, direction, shape, path_occupied)
        for steps in range(1, max_steps + 1):
            top = final_top - steps * dr
            left = final_left - steps * dc
            start_cells = _coords_for(sprite, top, left)
            if _touches_occupied(start_cells, input_occupied):
                continue
            starts.append((top, left, steps))
    return starts


def _contact_finals(
    sprite: dict,
    blocker: dict,
    later_sprites: list[dict],
    wall_coords: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = sprite["patch"].shape
    dr, dc = sprite["dir"]
    blocker_cells = _placed_coords(blocker)
    other_occupied = set(wall_coords)
    for later in later_sprites:
        other_occupied |= _placed_coords(later)

    finals: list[tuple[int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            here = _coords_for(sprite, top, left)
            if here & blocker_cells or here & other_occupied:
                continue
            next_top = top + dr
            next_left = left + dc
            if next_top < 0 or next_left < 0 or next_top + ph > h or next_left + pw > w:
                continue
            next_cells = _coords_for(sprite, next_top, next_left)
            if next_cells & other_occupied:
                continue
            if next_cells & blocker_cells:
                finals.append((top, left))
    return finals


def _face_wall(sprite: dict, final_top: int, final_left: int, shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    ph, pw = sprite["patch"].shape
    dr, dc = sprite["dir"]
    if dr == -1 and final_top == 0:
        return None
    if dr == 1 and final_top + ph >= h:
        return None
    if dc == -1 and final_left == 0:
        return None
    if dc == 1 and final_left + pw >= w:
        return None

    if dc == 1:
        wall_col = final_left + pw
        return {(final_top + r, wall_col) for r in range(ph)}
    if dc == -1:
        wall_col = final_left - 1
        return {(final_top + r, wall_col) for r in range(ph)}
    if dr == 1:
        wall_row = final_top + ph
        return {(wall_row, final_left + c) for c in range(pw)}
    wall_row = final_top - 1
    return {(wall_row, final_left + c) for c in range(pw)}


def _wall_stop_starts(sprite: dict, shape: tuple[int, int]) -> list[tuple[int, int, int, set[tuple[int, int]]]]:
    h, w = shape
    ph, pw = sprite["patch"].shape
    dr, dc = sprite["dir"]
    finals: list[tuple[int, int]] = []
    if dc == 1:
        cols = range(0, w - pw)
        rows = range(0, h - ph + 1)
    elif dc == -1:
        cols = range(1, w - pw + 1)
        rows = range(0, h - ph + 1)
    elif dr == 1:
        rows = range(0, h - ph)
        cols = range(0, w - pw + 1)
    else:
        rows = range(1, h - ph + 1)
        cols = range(0, w - pw + 1)

    for top in rows:
        for left in cols:
            wall = _face_wall(sprite, top, left, shape)
            if wall is None:
                continue
            next_cells = _coords_for(sprite, top + dr, left + dc)
            if next_cells & wall:
                finals.append((top, left))

    starts: list[tuple[int, int, int, set[tuple[int, int]]]] = []
    for final_top, final_left in finals:
        wall = _face_wall(sprite, final_top, final_left, shape)
        assert wall is not None
        input_occupied = set(wall)
        path_occupied = set(wall)
        choices = _possible_starts(sprite, [(final_top, final_left)], sprite["dir"], shape, path_occupied, input_occupied)
        for top, left, steps in choices:
            starts.append((top, left, steps, wall))
    return starts


def _edge_stop_starts(sprite: dict, shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = shape
    ph, pw = sprite["patch"].shape
    dr, dc = sprite["dir"]
    finals: list[tuple[int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            hit_edge = False
            for rr, cc, _ in sprite["cells"]:
                nr = top + rr + dr
                nc = left + cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    hit_edge = True
                    break
            if hit_edge:
                finals.append((top, left))
    return _possible_starts(sprite, finals, sprite["dir"], shape, set(), set())


def _random_free_wall(shape: tuple[int, int], occupied: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        horizontal = random.choice([False, True])
        length = random.randint(2, 6)
        if horizontal:
            if length > w:
                continue
            row = random.randint(0, h - 1)
            col = random.randint(0, w - length)
            cells = {(row, col + offset) for offset in range(length)}
        else:
            if length > h:
                continue
            row = random.randint(0, h - length)
            col = random.randint(0, w - 1)
            cells = {(row + offset, col) for offset in range(length)}
        if not cells & occupied and not _touches_occupied(cells, occupied):
            return cells
    return None


def _place_inert_sprite(sprite: dict, shape: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    h, w = shape
    ph, pw = sprite["patch"].shape
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        cells = _coords_for(sprite, top, left)
        if cells & occupied or _touches_occupied(cells, occupied):
            continue
        sprite["top"] = top
        sprite["left"] = left
        return True
    return False


def _place_control(shape: tuple[int, int], occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS * 2):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        cells = {(r, c)}
        if cells & occupied or _touches_occupied(cells, occupied):
            continue
        return r, c
    return None


def _render(shape: tuple[int, int], sprites: list[dict], wall_coords: set[tuple[int, int]], control: tuple[int, int], control_color: int) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in wall_coords:
        grid[r, c] = WALL
    for sprite in sprites:
        for rr, cc, color in sprite["cells"]:
            grid[int(sprite["top"]) + rr, int(sprite["left"]) + cc] = int(color)
    grid[control] = int(control_color)
    return grid


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 30), random.randint(18, 30))
    chain_len = random.choices([1, 2, 3, 4], weights=[2, 4, 4, 2])[0]
    inert_count = 1 if chain_len == 1 else random.choice([0, 1])
    total_sprites = chain_len + inert_count
    marker_colors = random.sample(COLOR_POOL, total_sprites)
    body_palette = [color for color in COLOR_POOL if color not in marker_colors]
    body_colors = [random.choice(body_palette) for _ in range(total_sprites)]

    chain = [
        _make_sprite(random.choice(DIRS4), body_colors[idx], marker_colors[idx])
        for idx in range(chain_len)
    ]
    inert = [
        _make_sprite(random.choice(DIRS4), body_colors[chain_len + idx], marker_colors[chain_len + idx])
        for idx in range(inert_count)
    ]

    stop_mode = random.choice(["edge", "wall"])
    wall_coords: set[tuple[int, int]] = set()

    last = chain[-1]
    if stop_mode == "wall":
        options = _wall_stop_starts(last, shape)
        if not options:
            return None
        preferred = [item for item in options if item[2] >= 2] or options
        top, left, _steps, wall = random.choice(preferred)
        last["top"] = top
        last["left"] = left
        wall_coords |= wall
    else:
        options = _edge_stop_starts(last, shape)
        if not options:
            return None
        preferred = [item for item in options if item[2] >= 2] or options
        top, left, _steps = random.choice(preferred)
        last["top"] = top
        last["left"] = left

    for idx in range(chain_len - 2, -1, -1):
        sprite = chain[idx]
        blocker = chain[idx + 1]
        later = chain[idx + 2 :]
        finals = _contact_finals(sprite, blocker, later, wall_coords, shape)
        if not finals:
            return None
        path_occupied = set(wall_coords)
        for placed in chain[idx + 1 :]:
            path_occupied |= _placed_coords(placed)
        starts = _possible_starts(sprite, finals, sprite["dir"], shape, path_occupied, path_occupied)
        if not starts:
            return None
        preferred = [item for item in starts if item[2] >= 2] or starts
        top, left, _steps = random.choice(preferred)
        sprite["top"] = top
        sprite["left"] = left

    occupied = set(wall_coords)
    for sprite in chain:
        occupied |= _placed_coords(sprite)

    if stop_mode == "edge" or random.random() < 0.5:
        extra_wall = _random_free_wall(shape, occupied)
        if extra_wall is None:
            return None
        wall_coords |= extra_wall
        occupied |= extra_wall

    for sprite in inert:
        if not _place_inert_sprite(sprite, shape, occupied):
            return None
        occupied |= _placed_coords(sprite)

    control = _place_control(shape, occupied)
    if control is None:
        return None

    sprites = chain + inert
    grid = _render(shape, sprites, wall_coords, control, chain[0]["marker_color"])

    try:
        out, trace = _transform(grid, return_trace=True)
    except AssertionError:
        return None
    if len(trace["order"]) != chain_len:
        return None
    if trace["reasons"][-1] != stop_mode:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
