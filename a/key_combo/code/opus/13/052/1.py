from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

WALL_COLORS = {BLUE, GREEN}
SPRITE_COLORS = (RED, YELLOW, ORANGE)
SPRITE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)

GRID_MIN = 13
GRID_MAX = 19
ROOM_COUNT_OPTIONS = (2, 3)
ROOM_OUTER_MIN = 5
ROOM_OUTER_MAX = 8
RETRY_LIMIT = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _sprite_components(grid: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for color in SPRITE_COLORS:
        comps = _components(grid == color)
        for coords in comps:
            sprites.append({"color": color, "coords": coords})
    return sprites


def _chambers(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    traversable = ~np.isin(grid, list(WALL_COLORS))
    comps = _components(traversable)
    chamber_id = np.full(grid.shape, -1, dtype=int)
    chambers = []
    for idx, coords in enumerate(comps):
        chamber_id[coords[:, 0], coords[:, 1]] = idx
        chambers.append(
            {
                "id": idx,
                "coords": coords,
                "open": _touches_edge(coords, grid.shape),
            }
        )
    return chambers, chamber_id


def _drop_distance(
    coords: np.ndarray,
    obstacle_mask: np.ndarray,
    own_mask: np.ndarray | None = None,
) -> tuple[int, str]:
    h, _ = obstacle_mask.shape
    if own_mask is None:
        own_mask = np.zeros_like(obstacle_mask, dtype=bool)

    distance = 0
    while True:
        next_cells = []
        hit_wall = False
        for r, c in coords:
            nr = int(r) + distance + 1
            nc = int(c)
            if nr >= h:
                return distance, "bottom"
            if obstacle_mask[nr, nc] and not own_mask[nr, nc]:
                hit_wall = True
                break
            next_cells.append((nr, nc))
        if hit_wall:
            return distance, "obstacle"
        distance += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let only sprites in edge-reachable chambers fall, while enclosed chambers become grey zero-gravity rooms.

    Input structure:
    - The grid uses black background, blue/green wall pixels, and up to three small monochrome sprites in red, yellow, and orange.
    - Blue and green walls partition the non-wall cells into 4-connected chambers; some chambers touch the outer border and some are fully enclosed.
    - Each sprite is a 4-connected 2-4 cell object that lies wholly inside one chamber.

    Rule:
    - Classify every chamber of non-wall cells by whether it touches the grid edge.
    - Sprites inside open chambers translate straight downward as rigid bodies until the next downward step would hit a wall, another sprite, or the bottom border.
    - Sprites inside enclosed chambers do not move.
    - After movement, recolor every black cell belonging to an enclosed chamber to grey, while keeping open-chamber black cells black and preserving all wall pixels.

    Key insight:
    - The decisive feature is not sprite color or absolute position but the topology of the surrounding chamber: edge-connected space has gravity, enclosed space is a zero-gravity room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BLACK, BLUE, GREEN, RED, YELLOW, ORANGE})

    chambers, chamber_id = _chambers(g)
    sprites = _sprite_components(g)

    for sprite in sprites:
        ids = {int(chamber_id[r, c]) for r, c in sprite["coords"]}
        assert len(ids) == 1 and next(iter(ids)) >= 0
        sprite["chamber_id"] = next(iter(ids))
        sprite["open"] = bool(chambers[sprite["chamber_id"]]["open"])
        area = int(sprite["coords"].shape[0])
        assert 2 <= area <= 4

    out = g.copy()
    wall_mask = np.isin(g, list(WALL_COLORS))
    all_sprite_mask = np.isin(g, list(SPRITE_COLORS))

    moved_positions: list[tuple[np.ndarray, int]] = []
    for sprite in sprites:
        coords = sprite["coords"]
        if not sprite["open"]:
            moved_positions.append((coords.copy(), int(sprite["color"])))
            continue
        own_mask = np.zeros_like(g, dtype=bool)
        own_mask[coords[:, 0], coords[:, 1]] = True
        obstacle_mask = wall_mask | all_sprite_mask
        distance, _ = _drop_distance(coords, obstacle_mask, own_mask)
        moved = coords.copy()
        moved[:, 0] += distance
        moved_positions.append((moved, int(sprite["color"])))
        out[coords[:, 0], coords[:, 1]] = BLACK

    placed = np.zeros_like(g, dtype=bool)
    for coords, color in moved_positions:
        assert not np.any(placed[coords[:, 0], coords[:, 1]])
        placed[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = color

    for chamber in chambers:
        if chamber["open"]:
            continue
        for r, c in chamber["coords"]:
            if out[r, c] == BLACK:
                out[r, c] = GREY

    return out


def _paint_room(mask: np.ndarray, room: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = room
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _room_fits(existing: np.ndarray, room: tuple[int, int, int, int]) -> bool:
    h, w = existing.shape
    r0, c0, r1, c1 = room
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    rr0 = max(0, r0 - 2)
    cc0 = max(0, c0 - 2)
    rr1 = min(h - 1, r1 + 2)
    cc1 = min(w - 1, c1 + 2)
    return not np.any(existing[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _add_spur(mask: np.ndarray, room: tuple[int, int, int, int], rng: random.Random) -> None:
    r0, c0, r1, c1 = room
    options = []
    if r0 >= 2:
        options.append(("up", r0, rng.randint(c0 + 1, c1 - 1)))
    if r1 + 2 < mask.shape[0]:
        options.append(("down", r1, rng.randint(c0 + 1, c1 - 1)))
    if c0 >= 2:
        options.append(("left", rng.randint(r0 + 1, r1 - 1), c0))
    if c1 + 2 < mask.shape[1]:
        options.append(("right", rng.randint(r0 + 1, r1 - 1), c1))
    rng.shuffle(options)
    for direction, a, b in options:
        length = rng.randint(1, 3)
        cells = []
        if direction == "up":
            cells = [(a - step, b) for step in range(1, length + 1)]
        elif direction == "down":
            cells = [(a + step, b) for step in range(1, length + 1)]
        elif direction == "left":
            cells = [(a, b - step) for step in range(1, length + 1)]
        else:
            cells = [(a, b + step) for step in range(1, length + 1)]
        if all(
            0 <= r < mask.shape[0]
            and 0 <= c < mask.shape[1]
            and not mask[r, c]
            for r, c in cells
        ):
            for r, c in cells:
                mask[r, c] = True
            return


def _add_border_divider(mask: np.ndarray, rng: random.Random) -> bool:
    h, w = mask.shape
    for _ in range(40):
        side = rng.choice(("top", "bottom", "left", "right"))
        length = rng.randint(3, 7)
        if side in ("top", "bottom"):
            c = rng.randint(1, w - 2)
            start_r = 0 if side == "top" else h - 1
            step = 1 if side == "top" else -1
            cells = [(start_r + step * i, c) for i in range(length)]
        else:
            r = rng.randint(1, h - 2)
            start_c = 0 if side == "left" else w - 1
            step = 1 if side == "left" else -1
            cells = [(r, start_c + step * i) for i in range(length)]
        if all(
            0 <= r < h and 0 <= c < w and not mask[r, c]
            for r, c in cells
        ):
            for r, c in cells:
                mask[r, c] = True
            return True
    return False


def _add_shelf(mask: np.ndarray, rng: random.Random) -> bool:
    h, w = mask.shape
    for _ in range(60):
        horizontal = rng.random() < 0.6
        length = rng.randint(3, 7)
        if horizontal:
            r = rng.randint(2, h - 3)
            c0 = rng.randint(1, w - length - 1)
            cells = [(r, c0 + dc) for dc in range(length)]
        else:
            c = rng.randint(2, w - 3)
            r0 = rng.randint(1, h - length - 1)
            cells = [(r0 + dr, c) for dr in range(length)]
        if all(not mask[r, c] for r, c in cells):
            for r, c in cells:
                mask[r, c] = True
            return True
    return False


def _candidate_positions(
    chamber_mask: np.ndarray,
    wall_mask: np.ndarray,
    occupied: np.ndarray,
    require_move: bool,
    desired_reason: str | None = None,
) -> list[dict]:
    h, w = chamber_mask.shape
    candidates = []
    obstacle_mask = wall_mask | occupied
    for shape in SPRITE_SHAPES:
        for anchor_r in range(1, h - 1):
            for anchor_c in range(1, w - 1):
                cells = np.array(
                    [(anchor_r + dr, anchor_c + dc) for dr, dc in shape],
                    dtype=int,
                )
                if np.any(cells[:, 0] <= 0) or np.any(cells[:, 0] >= h - 1):
                    continue
                if np.any(cells[:, 1] <= 0) or np.any(cells[:, 1] >= w - 1):
                    continue
                if not np.all(chamber_mask[cells[:, 0], cells[:, 1]]):
                    continue
                if np.any(occupied[cells[:, 0], cells[:, 1]]):
                    continue
                own = np.zeros_like(chamber_mask, dtype=bool)
                own[cells[:, 0], cells[:, 1]] = True
                distance, reason = _drop_distance(cells, obstacle_mask | own, own)
                if require_move and distance <= 0:
                    continue
                if desired_reason is not None and reason != desired_reason:
                    continue
                candidates.append(
                    {
                        "coords": cells,
                        "distance": distance,
                        "reason": reason,
                    }
                )
    return candidates


def _build_layout(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    wall_mask = np.zeros((h, w), dtype=bool)

    rooms = []
    for _ in range(rng.choice(ROOM_COUNT_OPTIONS)):
        placed = False
        for _ in range(120):
            room_h = rng.randint(ROOM_OUTER_MIN, min(ROOM_OUTER_MAX, h - 2))
            room_w = rng.randint(ROOM_OUTER_MIN, min(ROOM_OUTER_MAX, w - 2))
            r0 = rng.randint(1, h - room_h)
            c0 = rng.randint(1, w - room_w)
            room = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
            if not _room_fits(wall_mask, room):
                continue
            _paint_room(wall_mask, room)
            rooms.append(room)
            placed = True
            break
        if not placed:
            return None

    for room in rooms:
        if rng.random() < 0.7:
            _add_spur(wall_mask, room, rng)
        if rng.random() < 0.35:
            _add_spur(wall_mask, room, rng)

    divider_count = 1 if rng.random() < 0.55 else 0
    divider_count += 1 if rng.random() < 0.2 else 0
    for _ in range(divider_count):
        _add_border_divider(wall_mask, rng)

    shelf_count = rng.randint(1, 3)
    for _ in range(shelf_count):
        _add_shelf(wall_mask, rng)

    grid = np.zeros((h, w), dtype=int)
    wall_coords = np.argwhere(wall_mask)
    if wall_coords.size == 0:
        return None
    wall_colors = np.array([rng.choice((BLUE, GREEN)) for _ in range(len(wall_coords))], dtype=int)
    if BLUE not in wall_colors:
        wall_colors[0] = BLUE
    if GREEN not in wall_colors:
        wall_colors[-1] = GREEN
    grid[wall_coords[:, 0], wall_coords[:, 1]] = wall_colors
    return grid


def _choose_sprite_chambers(
    grid: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, bool]]:
    chambers, _ = _chambers(grid)
    open_ids = [c["id"] for c in chambers if c["open"]]
    closed_ids = [c["id"] for c in chambers if not c["open"]]
    if not open_ids or len(closed_ids) < 1:
        return []
    if len(open_ids) >= 2 and rng.random() < 0.45:
        if len(closed_ids) < 1:
            return []
        return [(open_ids[0], True), (open_ids[1], True), (closed_ids[0], False)]
    if len(closed_ids) >= 2:
        return [(open_ids[0], True), (closed_ids[0], False), (closed_ids[1], False)]
    return []


def _place_sprites(grid: np.ndarray, rng: random.Random) -> np.ndarray | None:
    chambers, chamber_id = _chambers(grid)
    chamber_map = {ch["id"]: ch for ch in chambers}
    targets = _choose_sprite_chambers(grid, rng)
    if len(targets) != 3:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    wall_mask = np.isin(grid, list(WALL_COLORS))
    placements: list[dict] = []

    open_indices = [idx for idx, (_, is_open) in enumerate(targets) if is_open]
    desired_open_reason = "obstacle" if rng.random() < 0.6 else "bottom"

    for idx, (cid, is_open) in enumerate(targets):
        chamber = chamber_map[cid]
        chamber_mask = np.zeros_like(grid, dtype=bool)
        chamber_mask[chamber["coords"][:, 0], chamber["coords"][:, 1]] = True
        desired_reason = None
        if is_open and idx == open_indices[0]:
            desired_reason = desired_open_reason
        candidates = _candidate_positions(
            chamber_mask,
            wall_mask,
            occupied,
            require_move=is_open,
            desired_reason=desired_reason,
        )
        if not candidates and is_open and desired_reason is not None:
            candidates = _candidate_positions(
                chamber_mask,
                wall_mask,
                occupied,
                require_move=True,
                desired_reason=None,
            )
        if not candidates:
            return None
        candidate = rng.choice(candidates)
        occupied[candidate["coords"][:, 0], candidate["coords"][:, 1]] = True
        candidate["chamber_id"] = cid
        candidate["open"] = is_open
        placements.append(candidate)

    colors = list(SPRITE_COLORS)
    rng.shuffle(colors)
    out = grid.copy()
    for color, placement in zip(colors, placements):
        coords = placement["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    sprite_by_chamber = {}
    for color, placement in zip(colors, placements):
        sprite_by_chamber[placement["chamber_id"]] = {
            "color": color,
            "coords": placement["coords"],
            "distance": placement["distance"],
            "reason": placement["reason"],
            "open": placement["open"],
        }

    for ch in chambers:
        sid = sprite_by_chamber.get(ch["id"])
        if sid is None:
            continue
        ids_here = {int(chamber_id[r, c]) for r, c in sid["coords"]}
        if ids_here != {ch["id"]}:
            return None

    return out


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    present = set(map(int, np.unique(grid)))
    if not {BLUE, GREEN}.issubset(present):
        return False
    sprites = _sprite_components(grid)
    if len(sprites) != 3:
        return False
    if {int(s["color"]) for s in sprites} != set(SPRITE_COLORS):
        return False
    for sprite in sprites:
        if not 2 <= len(sprite["coords"]) <= 4:
            return False

    chambers, chamber_id = _chambers(grid)
    chamber_sprite_counts = {ch["id"]: 0 for ch in chambers}
    open_with_sprite = 0
    closed_with_sprite = 0
    wall_mask = np.isin(grid, list(WALL_COLORS))
    all_sprite_mask = np.isin(grid, list(SPRITE_COLORS))
    for sprite in sprites:
        ids = {int(chamber_id[r, c]) for r, c in sprite["coords"]}
        if len(ids) != 1:
            return False
        cid = next(iter(ids))
        chamber_sprite_counts[cid] += 1
        own_mask = np.zeros_like(grid, dtype=bool)
        own_mask[sprite["coords"][:, 0], sprite["coords"][:, 1]] = True
        distance, reason = _drop_distance(sprite["coords"], wall_mask | all_sprite_mask, own_mask)
        if chambers[cid]["open"]:
            open_with_sprite += 1
            if distance <= 0:
                return False
        else:
            closed_with_sprite += 1
    if open_with_sprite < 1 or closed_with_sprite < 1:
        return False
    if any(count > 1 for count in chamber_sprite_counts.values()):
        return False

    result = main(grid)
    if result.shape != grid.shape:
        return False

    grey_count = 0
    for chamber in chambers:
        black_cells = chamber["coords"][grid[chamber["coords"][:, 0], chamber["coords"][:, 1]] == BLACK]
        if chamber["open"]:
            if len(black_cells) and np.any(result[black_cells[:, 0], black_cells[:, 1]] == GREY):
                return False
        else:
            if len(black_cells):
                if not np.all(result[black_cells[:, 0], black_cells[:, 1]] == GREY):
                    return False
                grey_count += len(black_cells)
    if grey_count == 0:
        return False
    if np.array_equal(result, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(RETRY_LIMIT):
        base = _build_layout(rng)
        if base is None:
            continue
        grid = _place_sprites(base, rng)
        if grid is None:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid zero-gravity chamber input")
