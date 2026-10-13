from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SEQUENCE_LEN = 4
TOP_RESERVED_ROWS = 2
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, GREY))

GRID_H_RANGE = (12, 31)
GRID_W_RANGE = (12, 31)
OBJECT_COUNT_RANGE = (4, 9)
POLYOMINO_AREA_RANGE = (4, 8)
POLYOMINO_SIDE_MAX = 5
PLACEMENT_PADDING = 1
PLACEMENT_RETRIES = 150
POLYOMINO_RETRIES = 200
GENERATION_RETRIES = 400

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for start_r, start_c in np.argwhere(mask):
        r0 = int(start_r)
        c0 = int(start_c)
        if visited[r0, c0]:
            continue

        visited[r0, c0] = True
        queue = deque([(r0, c0)])
        comp: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        components.append(comp)

    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -turns)


def _extract_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for coords in _connected_components(grid == color):
        mask, bbox = _coords_to_mask(coords)
        components.append({"coords": coords, "mask": mask, "bbox": bbox})
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))  # type: ignore[index]
    return components


def _sequence_colors(grid: np.ndarray) -> list[int]:
    seq: list[int] = []
    started = False
    for value in np.asarray(grid[0], dtype=int):
        color = int(value)
        if color != BG:
            seq.append(color)
            started = True
        elif started:
            break
    assert len(seq) == SEQUENCE_LEN
    assert len(set(seq)) == SEQUENCE_LEN
    return seq


def _dominant_body_color(grid: np.ndarray) -> int:
    body = np.asarray(grid[1:], dtype=int)
    values = body[body != BG]
    assert values.size > 0
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor rotated copies of one grey shape using the top instruction sequence.

    Input structure:
    - Background is black.
    - The top row begins with four distinct colored pixels, then stays black.
    - Below that row are several separate grey 4-connected objects.
    - Every grey object is the same asymmetric shape, but some copies are rotated.

    Rule:
    - Keep the instruction row unchanged.
    - Use the upper-left grey object as the 0-degree reference orientation.
    - For every other grey object, determine how many clockwise quarter-turns map the reference shape onto it.
    - Replace each grey object with the 1st, 2nd, 3rd, or 4th instruction color for 0, 1, 2, or 3 clockwise turns.

    Key insight:
    - The grey objects do not differ by identity; they are one shared shape viewed in different rotations, and the top color sequence encodes those rotations.
    """
    g = np.asarray(grid, dtype=int)
    colors = _sequence_colors(g)
    grey = _dominant_body_color(g)
    components = _extract_components(g[1:], grey)
    assert components

    reference_mask = components[0]["mask"]  # type: ignore[index]
    out = g.copy()

    for comp in components:
        comp_mask = comp["mask"]  # type: ignore[index]
        turns = None
        for candidate in range(SEQUENCE_LEN):
            if np.array_equal(_rotate_cw(reference_mask, candidate), comp_mask):
                turns = candidate
                break
        assert turns is not None, "every body object must be a rotation of the reference object"

        for r, c in comp["coords"]:  # type: ignore[index]
            out[r + 1, c] = colors[turns]

    return out


def _random_polyomino(rng: random.Random) -> np.ndarray:
    for _ in range(POLYOMINO_RETRIES):
        area = rng.randrange(*POLYOMINO_AREA_RANGE)
        cells = {(0, 0)}

        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(rng.choice(tuple(frontier)))

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2 or h > POLYOMINO_SIDE_MAX or w > POLYOMINO_SIDE_MAX:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        if int(mask.sum()) == h * w:
            continue

        keys = {
            (_trim_mask(_rotate_cw(mask, turns)).shape, _trim_mask(_rotate_cw(mask, turns)).tobytes())
            for turns in range(SEQUENCE_LEN)
        }
        if len(keys) != SEQUENCE_LEN:
            continue

        return mask

    raise RuntimeError("failed to generate an asymmetric reference shape")


def _placement_is_free(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = blocked.shape
    obj_h, obj_w = shape
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(h, top + obj_h + PLACEMENT_PADDING)
    c1 = min(w, left + obj_w + PLACEMENT_PADDING)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> None:
    h, w = blocked.shape
    obj_h, obj_w = shape
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(h, top + obj_h + PLACEMENT_PADDING)
    c1 = min(w, left + obj_w + PLACEMENT_PADDING)
    blocked[r0:r1, c0:c1] = True


def _place_objects(
    rng: random.Random,
    grid_shape: tuple[int, int],
    rotations: list[int],
    rotated_masks: list[np.ndarray],
) -> list[tuple[int, int, np.ndarray]]:
    h, w = grid_shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0:TOP_RESERVED_ROWS, :] = True
    placements: list[tuple[int, int, np.ndarray]] = []

    for turns in rotations:
        mask = rotated_masks[turns]
        obj_h, obj_w = mask.shape
        candidates = [
            (top, left)
            for top in range(TOP_RESERVED_ROWS, h - obj_h + 1)
            for left in range(0, w - obj_w + 1)
            if _placement_is_free(blocked, top, left, (obj_h, obj_w))
        ]
        if not candidates:
            return []

        top, left = rng.choice(candidates)
        _mark_blocked(blocked, top, left, (obj_h, obj_w))
        placements.append((top, left, mask))

    return placements


def _build_rotation_list(rng: random.Random, count: int) -> list[int]:
    rotations = list(range(SEQUENCE_LEN))
    while len(rotations) < count:
        rotations.append(rng.randrange(SEQUENCE_LEN))
    rng.shuffle(rotations)
    return rotations


def _render_grid(
    colors: list[int],
    placements: list[tuple[int, int, np.ndarray]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :SEQUENCE_LEN] = np.array(colors, dtype=int)
    for top, left, mask in placements:
        rows, cols = np.argwhere(mask).T
        grid[top + rows, left + cols] = GREY
    return grid


def _validate_generated_sample(grid: np.ndarray, object_count: int) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert len(_sequence_colors(grid)) == SEQUENCE_LEN

    comps = _extract_components(grid[1:], GREY)
    assert len(comps) == object_count

    out = main(grid)
    assert np.array_equal(out[0], grid[0])
    assert not np.any(out[1:] == GREY)
    painted = {int(value) for value in np.unique(out[1:]) if int(value) != BG}
    assert set(_sequence_colors(grid)).issubset(painted)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base_mask = _random_polyomino(rng)
    rotated_masks = [_trim_mask(_rotate_cw(base_mask, turns)) for turns in range(SEQUENCE_LEN)]

    object_count = rng.randrange(*OBJECT_COUNT_RANGE)
    rotations = _build_rotation_list(rng, object_count)

    min_h = TOP_RESERVED_ROWS + max(mask.shape[0] for mask in rotated_masks) + 1
    min_w = max(SEQUENCE_LEN, max(mask.shape[1] for mask in rotated_masks))
    grid_h = rng.randint(max(min_h, GRID_H_RANGE[0]), GRID_H_RANGE[1] - 1)
    grid_w = rng.randint(max(min_w, GRID_W_RANGE[0]), GRID_W_RANGE[1] - 1)

    placements = _place_objects(rng, (grid_h, grid_w), rotations, rotated_masks)
    if len(placements) != object_count:
        return None

    colors = rng.sample(COLOR_POOL, SEQUENCE_LEN)
    grid = _render_grid(colors, placements, (grid_h, grid_w))
    _validate_generated_sample(grid, object_count)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
