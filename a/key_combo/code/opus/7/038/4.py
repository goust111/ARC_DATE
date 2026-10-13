from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
TAIL = 5
BODY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

MOVE_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
TAIL_SIDES = {
    "up": "bottom",
    "down": "top",
    "left": "right",
    "right": "left",
}

BASE_SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
]

GRID_MIN = 10
GRID_MAX = 14
OBJECT_COUNT_CHOICES = (3, 4)
GEN_RETRIES = 400
PLACEMENT_TRIES = 120


@dataclass
class Sprite:
    color: int
    body: np.ndarray
    tail: tuple[int, int]
    move: tuple[int, int]
    distance: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return _bbox(self.body)


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    neighbors = DIRS4 if connectivity == 4 else DIRS8
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _is_connected4(coords: np.ndarray) -> bool:
    cells = {tuple(cell) for cell in coords.tolist()}
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == len(cells)


def _move_from_tail(
    tail: tuple[int, int], body_bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[tuple[int, int], int]:
    tr, tc = tail
    r0, c0, r1, c1 = body_bbox
    H, W = shape
    if tr < r0:
        return (1, 0), H - 1 - r1
    if tr > r1:
        return (-1, 0), r0
    if tc < c0:
        return (0, 1), W - 1 - c1
    if tc > c1:
        return (0, -1), c0
    raise AssertionError("tail must sit outside the body bbox on one side")


def _parse_sprites(grid: np.ndarray) -> tuple[int, list[Sprite]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    parts = _components(g != bg, connectivity=8)
    sprites: list[Sprite] = []
    for part in parts:
        colors = np.array([g[r, c] for r, c in part], dtype=int)
        tail_mask = colors == TAIL
        assert int(tail_mask.sum()) == 1, "each sprite must have exactly one gray tail"
        body = part[~tail_mask]
        assert 3 <= body.shape[0] <= 5, "body must have 3-5 cells"
        assert _is_connected4(body), "body must be 4-connected"
        body_colors = {int(g[r, c]) for r, c in body}
        assert len(body_colors) == 1 and TAIL not in body_colors, "body must be monochrome"
        tail = tuple(map(int, part[tail_mask][0]))
        move, distance = _move_from_tail(tail, _bbox(body), g.shape)
        sprites.append(
            Sprite(
                color=next(iter(body_colors)),
                body=body.astype(int),
                tail=tail,
                move=move,
                distance=int(distance),
            )
        )
    assert 3 <= len(sprites) <= 4, "expected 3-4 sprites"
    distances = [sprite.distance for sprite in sprites]
    assert len(set(distances)) == len(distances), "target-edge distances must be unique"
    return bg, sprites


def _shift(coords: np.ndarray, dr: int, dc: int, steps: int) -> np.ndarray:
    delta = np.array([dr * steps, dc * steps], dtype=int)
    return coords + delta


def _max_steps(body_bbox: tuple[int, int, int, int], move: tuple[int, int], shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = body_bbox
    H, W = shape
    dr, dc = move
    if dr == 1:
        return H - 1 - r1
    if dr == -1:
        return r0
    if dc == 1:
        return W - 1 - c1
    return c0


def _independent_edge_body(sprite: Sprite, shape: tuple[int, int]) -> np.ndarray:
    steps = _max_steps(sprite.bbox, sprite.move, shape)
    return _shift(sprite.body, sprite.move[0], sprite.move[1], steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each tailed monochrome sprite away from its gray tail, stopping for earlier arrivals, and leave a gray motion trail.

    Input structure:
    - The grid is mostly black background with 3-4 composite 8-connected sprites.
    - Each composite contains one gray tail pixel attached to a single-color 4-connected body of 3-5 cells.
    - The tail sits on exactly one side of the body and encodes the opposite slide direction.

    Rule:
    - Read each body's slide direction as away from its gray tail.
    - Order sprites by the body's front-edge distance to the target grid edge, with the smallest distance moving first.
    - Slide each body until its next step would leave the grid or overlap a body that has already finished moving.
    - Erase every tail, paint gray on originally black cells swept by the moving body, and finally draw each body in its new position.

    Key insight:
    - The gray tail is both a disposable direction marker and the key to the hidden move order: sprites nearer to their destination edge claim space first and block later sliders.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites = _parse_sprites(g)
    ordered = sorted(sprites, key=lambda sprite: sprite.distance)

    out = np.full_like(g, bg)
    occupied = np.zeros(g.shape, dtype=bool)
    finals: list[tuple[int, np.ndarray]] = []

    for sprite in ordered:
        dr, dc = sprite.move
        steps = 0
        limit = _max_steps(sprite.bbox, sprite.move, g.shape)
        while steps < limit:
            moved = _shift(sprite.body, dr, dc, steps + 1)
            if np.any(occupied[moved[:, 0], moved[:, 1]]):
                break
            steps += 1

        for step in range(1, steps + 1):
            swept = _shift(sprite.body, dr, dc, step)
            for r, c in swept:
                if int(g[r, c]) == bg:
                    out[r, c] = TAIL

        final_body = _shift(sprite.body, dr, dc, steps)
        occupied[final_body[:, 0], final_body[:, 1]] = True
        finals.append((sprite.color, final_body))

    for color, final_body in finals:
        out[final_body[:, 0], final_body[:, 1]] = color
    return out


def _normalize_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple(tuple(map(int, cell)) for cell in arr.tolist())


def _random_body_shape() -> tuple[tuple[int, int], ...]:
    shape = np.array(random.choice(BASE_SHAPES), dtype=int)
    if random.random() < 0.5:
        shape[:, 1] *= -1
    for _ in range(random.randrange(4)):
        shape = np.stack([shape[:, 1], -shape[:, 0]], axis=1)
    return _normalize_shape(shape.tolist())


def _shape_size(body: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.array(body, dtype=int)
    r1, c1 = arr.max(axis=0)
    return int(r1 + 1), int(c1 + 1)


def _tail_candidates(body: tuple[tuple[int, int], ...], side: str) -> list[tuple[int, int]]:
    arr = np.array(body, dtype=int)
    if side == "top":
        edge = int(arr[:, 0].min())
        cells = arr[arr[:, 0] == edge]
        return [(int(r - 1), int(c)) for r, c in cells]
    if side == "bottom":
        edge = int(arr[:, 0].max())
        cells = arr[arr[:, 0] == edge]
        return [(int(r + 1), int(c)) for r, c in cells]
    if side == "left":
        edge = int(arr[:, 1].min())
        cells = arr[arr[:, 1] == edge]
        return [(int(r), int(c - 1)) for r, c in cells]
    edge = int(arr[:, 1].max())
    cells = arr[arr[:, 1] == edge]
    return [(int(r), int(c + 1)) for r, c in cells]


def _origin_from_distance(
    move_name: str,
    distance: int,
    body_h: int,
    body_w: int,
    ortho: int,
    shape: tuple[int, int],
) -> tuple[int, int]:
    H, W = shape
    if move_name == "right":
        return ortho, W - body_w - distance
    if move_name == "left":
        return ortho, distance
    if move_name == "down":
        return H - body_h - distance, ortho
    return distance, ortho


def _component_cells(
    body: tuple[tuple[int, int], ...], tail: tuple[int, int], r0: int, c0: int
) -> list[tuple[int, int]]:
    cells = [(r0 + r, c0 + c) for r, c in body]
    cells.append((r0 + tail[0], c0 + tail[1]))
    return cells


def _can_place_component(shape: tuple[int, int], occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    H, W = shape
    seen = set()
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in seen:
            return False
        seen.add((r, c))
        r0 = max(0, r - 1)
        r1 = min(H, r + 2)
        c0 = max(0, c - 1)
        c1 = min(W, c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _paint_component(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    body: tuple[tuple[int, int], ...],
    tail: tuple[int, int],
    r0: int,
    c0: int,
) -> dict:
    body_cells = np.array([(r0 + r, c0 + c) for r, c in body], dtype=int)
    tail_cell = (r0 + tail[0], c0 + tail[1])
    grid[body_cells[:, 0], body_cells[:, 1]] = color
    grid[tail_cell[0], tail_cell[1]] = TAIL
    for r, c in _component_cells(body, tail, r0, c0):
        occupied[r, c] = True
    return {"color": color, "body": body_cells, "tail": tail_cell}


def _max_distance_for_body(move_name: str, body_h: int, body_w: int, shape: tuple[int, int]) -> int:
    H, W = shape
    if move_name in ("left", "right"):
        return W - body_w - 1
    return H - body_h - 1


def _choose_anchor_pair(
    shape: tuple[int, int], colors: list[int], grid: np.ndarray, occupied: np.ndarray
) -> list[dict] | None:
    H, W = shape
    for _ in range(PLACEMENT_TRIES):
        move_name = random.choice(tuple(MOVE_VECTORS))
        body = _random_body_shape()
        body_h, body_w = _shape_size(body)
        span = body_w if move_name in ("left", "right") else body_h
        max_dist = _max_distance_for_body(move_name, body_h, body_w, shape)
        min_gap = span + 2
        dist_pairs = [
            (near, far)
            for near in range(1, max_dist + 1)
            for far in range(near + min_gap, max_dist + 1)
        ]
        if not dist_pairs:
            continue
        near, far = random.choice(dist_pairs)
        side = TAIL_SIDES[move_name]
        tail_options = _tail_candidates(body, side)
        random.shuffle(tail_options)

        if move_name in ("left", "right"):
            ortho_positions = list(range(H - body_h + 1))
        else:
            ortho_positions = list(range(W - body_w + 1))
        random.shuffle(ortho_positions)

        for ortho in ortho_positions:
            r_near, c_near = _origin_from_distance(move_name, near, body_h, body_w, ortho, shape)
            r_far, c_far = _origin_from_distance(move_name, far, body_h, body_w, ortho, shape)
            for tail in tail_options:
                near_cells = _component_cells(body, tail, r_near, c_near)
                far_cells = _component_cells(body, tail, r_far, c_far)
                if not _can_place_component(shape, occupied, near_cells):
                    continue
                temp_occupied = occupied.copy()
                for r, c in near_cells:
                    temp_occupied[r, c] = True
                if not _can_place_component(shape, temp_occupied, far_cells):
                    continue
                first = _paint_component(grid, occupied, colors[0], body, tail, r_near, c_near)
                second = _paint_component(grid, occupied, colors[1], body, tail, r_far, c_far)
                first["move_name"] = move_name
                second["move_name"] = move_name
                return [first, second]
    return None


def _place_extra_object(
    shape: tuple[int, int],
    color: int,
    used_distances: set[int],
    grid: np.ndarray,
    occupied: np.ndarray,
) -> dict | None:
    H, W = shape
    for _ in range(PLACEMENT_TRIES):
        move_name = random.choice(tuple(MOVE_VECTORS))
        body = _random_body_shape()
        body_h, body_w = _shape_size(body)
        max_dist = _max_distance_for_body(move_name, body_h, body_w, shape)
        distances = [dist for dist in range(1, max_dist + 1) if dist not in used_distances]
        if not distances:
            continue
        random.shuffle(distances)
        side = TAIL_SIDES[move_name]
        tail_options = _tail_candidates(body, side)
        random.shuffle(tail_options)

        if move_name in ("left", "right"):
            orthos = list(range(H - body_h + 1))
        else:
            orthos = list(range(W - body_w + 1))
        random.shuffle(orthos)

        for dist in distances:
            for ortho in orthos:
                r0, c0 = _origin_from_distance(move_name, dist, body_h, body_w, ortho, shape)
                for tail in tail_options:
                    cells = _component_cells(body, tail, r0, c0)
                    if not _can_place_component(shape, occupied, cells):
                        continue
                    item = _paint_component(grid, occupied, color, body, tail, r0, c0)
                    item["move_name"] = move_name
                    return item
    return None


def _naive_overlap_exists(grid: np.ndarray) -> bool:
    _, sprites = _parse_sprites(grid)
    placed: list[set[tuple[int, int]]] = []
    for sprite in sprites:
        dest = _independent_edge_body(sprite, grid.shape)
        cell_set = {tuple(cell) for cell in dest.tolist()}
        for other in placed:
            if cell_set & other:
                return True
        placed.append(cell_set)
    return False


def _generated_input_is_valid(grid: np.ndarray) -> bool:
    try:
        bg, sprites = _parse_sprites(grid)
    except AssertionError:
        return False
    if bg != BG or not _naive_overlap_exists(grid):
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not any(np.any(out == sprite.color) for sprite in sprites):
        return False
    return bool(np.any(out == TAIL)) and not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    count = random.choice(OBJECT_COUNT_CHOICES)
    colors = random.sample(BODY_COLORS, count)

    anchored = _choose_anchor_pair((H, W), colors[:2], grid, occupied)
    if anchored is None:
        return None

    used_distances = set()
    for body_info in anchored:
        sprite = Sprite(
            color=body_info["color"],
            body=body_info["body"],
            tail=body_info["tail"],
            move=MOVE_VECTORS[body_info["move_name"]],
            distance=0,
        )
        sprite.distance = _move_from_tail(sprite.tail, sprite.bbox, grid.shape)[1]
        used_distances.add(sprite.distance)

    for color in colors[2:]:
        item = _place_extra_object((H, W), color, used_distances, grid, occupied)
        if item is None:
            return None
        sprite = Sprite(
            color=item["color"],
            body=item["body"],
            tail=item["tail"],
            move=MOVE_VECTORS[item["move_name"]],
            distance=0,
        )
        sprite.distance = _move_from_tail(sprite.tail, sprite.bbox, grid.shape)[1]
        if sprite.distance in used_distances:
            return None
        used_distances.add(sprite.distance)

    return grid if _generated_input_is_valid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
