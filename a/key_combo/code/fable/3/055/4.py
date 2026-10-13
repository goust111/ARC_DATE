from __future__ import annotations

import itertools
import random

import numpy as np


Coord = tuple[int, int]
DIRS4: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR_NAMES: tuple[str, ...] = ("up", "right", "down", "left")
NOSE_BY_DIR: dict[str, Coord] = {
    "up": (0, 1),
    "right": (1, 2),
    "down": (2, 1),
    "left": (1, 0),
}
CANONICAL_NOSE: Coord = NOSE_BY_DIR["up"]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _is_connected(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) > 0 and len(_components(mask)) == 1


def _direction_name(src: Coord, dst: Coord) -> str:
    sr, sc = src
    dr, dc = dst
    if sr == dr and sc < dc:
        return "right"
    if sr == dr and sc > dc:
        return "left"
    if sc == dc and sr < dr:
        return "down"
    if sc == dc and sr > dr:
        return "up"
    raise AssertionError("expected a rook move between consecutive path points")


def _rotate_to_direction(patch: np.ndarray, source_dir: str, target_dir: str) -> np.ndarray:
    source_idx = DIR_NAMES.index(source_dir)
    target_idx = DIR_NAMES.index(target_dir)
    turns = (source_idx - target_idx) % 4
    return np.rot90(patch, k=turns)


def _find_sprite(grid: np.ndarray, bg: int) -> tuple[np.ndarray, Coord, str, int]:
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for coords in _components(grid != bg):
        if int(coords.shape[0]) <= 1:
            continue
        box = _bbox_coords(coords)
        if box[2] - box[0] == 2 and box[3] - box[1] == 2:
            candidates.append((coords, box))

    assert len(candidates) == 1, "expected exactly one non-singleton 3x3 sprite"
    _coords, box = candidates[0]
    patch = _crop(grid, box)
    colors = [int(color) for color in np.unique(patch) if int(color) != int(bg)]
    assert len(colors) == 2, "sprite should use exactly two non-background colors"

    counts = {color: int(np.count_nonzero(patch == color)) for color in colors}
    nose_positions = []
    for pos in NOSE_BY_DIR.values():
        color = int(patch[pos])
        if color != int(bg) and counts[color] == 1:
            nose_positions.append(pos)

    assert len(nose_positions) == 1, "expected a unique edge-middle nose pixel"
    nose_rel = nose_positions[0]
    source_dir = next(name for name, pos in NOSE_BY_DIR.items() if pos == nose_rel)
    nose_color = int(patch[nose_rel])
    center = (box[0] + 1, box[1] + 1)
    return patch, center, source_dir, nose_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the unique rook-chain of waypoints and stamp the 3x3 sprite at each stop with travel-facing rotation.

    Input structure:
    - A black-dominant grid contains one connected 3x3 sprite made from a body color plus one different nose color.
    - The nose is a single edge-middle pixel inside the sprite, and additional singleton pixels of that same color act as waypoints.
    - Starting from the sprite center, at every stage exactly one unvisited waypoint shares the current row or column.

    Rule:
    - Read the input sprite as the stamp template and infer its current orientation from the nose position.
    - Traverse the waypoint chain by repeatedly moving to the unique aligned unvisited waypoint.
    - At each visited waypoint, paste a rotated copy of the sprite centered on that waypoint so the nose points along the travel direction; keep the original sprite and preserve background elsewhere.

    Key insight:
    - The singleton pixels are not copied individually: they encode an ordered rook path, and each step direction determines how the same 3x3 sprite must be rotated before stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    patch, current, source_dir, nose_color = _find_sprite(g, bg)

    marker_coords = {(int(r), int(c)) for r, c in np.argwhere(g == int(nose_color))}
    nose_pos = None
    for name, rel in NOSE_BY_DIR.items():
        if name == source_dir:
            nose_pos = (current[0] + rel[0] - 1, current[1] + rel[1] - 1)
            break
    assert nose_pos is not None
    marker_coords.remove(nose_pos)
    assert marker_coords, "expected at least one waypoint marker"

    out = g.copy()
    while marker_coords:
        aligned = [coord for coord in marker_coords if coord[0] == current[0] or coord[1] == current[1]]
        assert len(aligned) == 1, "waypoints must form a unique rook-chain"
        nxt = aligned[0]
        travel_dir = _direction_name(current, nxt)
        stamped = _rotate_to_direction(patch, source_dir, travel_dir)
        assert int(stamped[1, 1]) != int(bg), "stamps must cover their waypoint centers"
        _paste(out, stamped, nxt[0] - 1, nxt[1] - 1, transparent=bg)
        marker_coords.remove(nxt)
        current = nxt

    return out


def _sprite_bbox_ok(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return (r0, c0, r1, c1) == (0, 0, 2, 2)


def _body_is_nonsymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k=k)) for k in (1, 2, 3))


def _build_sprite_library() -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    body_cells = [(r, c) for r in range(3) for c in range(3) if (r, c) != CANONICAL_NOSE]
    for area in range(3, 7):
        for combo in itertools.combinations(body_cells, area):
            if (1, 1) not in combo:
                continue
            body = np.zeros((3, 3), dtype=bool)
            for cell in combo:
                body[cell] = True
            occupied = body.copy()
            occupied[CANONICAL_NOSE] = True
            if not _is_connected(occupied):
                continue
            if not _sprite_bbox_ok(occupied):
                continue
            if not _body_is_nonsymmetric(body):
                continue
            patch = np.zeros((3, 3), dtype=int)
            patch[body] = 1
            patch[CANONICAL_NOSE] = 2
            sprites.append(patch)
    assert sprites, "sprite library should not be empty"
    return sprites


SPRITE_LIBRARY = _build_sprite_library()


def _sample_spaced_values(count: int, low: int, high: int, gap: int) -> list[int]:
    values = list(range(low, high + 1))
    for _ in range(400):
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - prev) >= gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return chosen
    raise RuntimeError("failed to sample spaced coordinates")


def _build_path(step_count: int, height: int, width: int, horizontal_first: bool) -> list[Coord]:
    horizontal_steps = (step_count + int(horizontal_first)) // 2
    vertical_steps = step_count - horizontal_steps
    row_need = vertical_steps + 1
    col_need = horizontal_steps + 1

    rows = _sample_spaced_values(row_need, 2, height - 3, gap=4)
    cols = _sample_spaced_values(col_need, 2, width - 3, gap=4)

    r_idx = 0
    c_idx = 0
    points = [(rows[r_idx], cols[c_idx])]
    is_horizontal = horizontal_first
    for _ in range(step_count):
        if is_horizontal:
            c_idx += 1
        else:
            r_idx += 1
        points.append((rows[r_idx], cols[c_idx]))
        is_horizontal = not is_horizontal
    return points


def _colorize_sprite(proto: np.ndarray, body_color: int, nose_color: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[proto == 1] = int(body_color)
    patch[proto == 2] = int(nose_color)
    return patch


def _min_size_for_count(count: int) -> int:
    return 5 + 4 * (count - 1)


def _generate_case() -> np.ndarray:
    step_count = random.randint(2, 6)
    horizontal_first = bool(random.getrandbits(1))
    horizontal_steps = (step_count + int(horizontal_first)) // 2
    vertical_steps = step_count - horizontal_steps
    row_need = vertical_steps + 1
    col_need = horizontal_steps + 1

    min_h = _min_size_for_count(row_need)
    min_w = _min_size_for_count(col_need)
    height = random.randint(max(12, min_h), min(30, min_h + 8))
    width = random.randint(max(12, min_w), min(30, min_w + 8))

    points = _build_path(step_count, height, width, horizontal_first)
    proto = random.choice(SPRITE_LIBRARY)
    source_dir = random.choice(DIR_NAMES)
    sprite = _rotate_to_direction(proto, "up", source_dir)

    body_color, nose_color = random.sample(range(1, 10), 2)
    sprite = _colorize_sprite(sprite, body_color, nose_color)

    grid = np.zeros((height, width), dtype=int)
    start_r, start_c = points[0]
    grid[start_r - 1 : start_r + 2, start_c - 1 : start_c + 2] = sprite
    for r, c in points[1:]:
        grid[r, c] = int(nose_color)

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            return _generate_case()
        except (AssertionError, RuntimeError, ValueError, KeyError):
            continue
    raise RuntimeError("failed to generate a valid waypoint-stamping input")
