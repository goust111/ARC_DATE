import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

UTILS_ROOT = Path("/home/lixinyu/snap/new")
if str(UTILS_ROOT) not in sys.path:
    sys.path.append(str(UTILS_ROOT))

from solvers.arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
WALL = Color.PINK
SPRITE_COLOR_SET = {BLUE, GREEN}

SPRITE_PROTOTYPES = [
    np.array(
        [
            [BLUE, BLUE, GREEN],
            [GREEN, BLUE, GREEN],
            [GREEN, GREEN, BLUE],
        ],
        dtype=int,
    ),
    np.array(
        [
            [BLUE, GREEN, BLUE],
            [BLUE, GREEN, GREEN],
            [GREEN, BLUE, GREEN],
        ],
        dtype=int,
    ),
    np.array(
        [
            [GREEN, BLUE, BLUE],
            [BLUE, GREEN, GREEN],
            [GREEN, BLUE, GREEN],
        ],
        dtype=int,
    ),
]

BASE_COUNT_BY_MOD = {0: 4, 1: 1, 2: 2, 3: 3}
CHAMBER_KEYS = ("TL", "TR", "BL", "BR")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rotation_k(clockwise_steps: int) -> int:
    return (-int(clockwise_steps)) % 4


def _sprite_records(grid: np.ndarray) -> list[dict]:
    comps = components_mask(np.isin(grid, list(SPRITE_COLOR_SET)), connectivity=4)
    sprites = []
    for coords in comps:
        if int(coords.shape[0]) != 9:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(np.isin(patch, list(SPRITE_COLOR_SET))):
            sprites.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch.copy(),
                }
            )
    sprites.sort(key=lambda item: item["bbox"])
    assert len(sprites) == 3, "expected exactly three filled 3x3 sprites"
    return sprites


def _reachable_markers(grid: np.ndarray, sprite: dict, sprites: list[dict]) -> set[tuple[int, int]]:
    blocked = np.zeros_like(grid, dtype=bool)
    blocked[grid == WALL] = True
    for other in sprites:
        if other is sprite:
            continue
        for r, c in other["coords"]:
            blocked[int(r), int(c)] = True

    visited = np.zeros_like(grid, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r, c in sprite["coords"]:
        rr = int(r)
        cc = int(c)
        visited[rr, cc] = True
        queue.append((rr, cc))

    reachable: set[tuple[int, int]] = set()
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if visited[nr, nc] or blocked[nr, nc]:
                continue
            if int(grid[nr, nc]) not in (BG, RED):
                continue
            visited[nr, nc] = True
            if int(grid[nr, nc]) == RED:
                reachable.add((nr, nc))
            queue.append((nr, nc))
    return reachable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each 3x3 sprite in place by the quarter-turn count encoded by its wall-reachable red markers, then erase those markers.

    Input structure
    - The background is black, magenta wall pixels split the board into separated floor regions, and red singleton markers sit on the floor.
    - Exactly three filled 3x3 sprites are present; each sprite is a 4-connected blue/green two-color patch with no background holes.

    Rule
    - For each sprite, start a 4-connected flood fill from all of its pixels and allow movement only through black background and red marker cells.
    - Magenta walls and the other two sprites are blocked, so the reachable red markers are exactly the markers in that sprite's accessible floor region.
    - Let K be the number of reachable red markers; rotate the sprite in place by K mod 4 quarter-turns clockwise and erase those reachable markers to black.

    Key insight
    - The controlling number is not based on geometric proximity: a nearby red marker can be separated by a wall while a farther marker in the same region is reachable, so wall topology determines the rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert int(get_bg(g)) == BG, "expected black background"

    sprites = _sprite_records(g)
    reachable_sets = [_reachable_markers(g, sprite, sprites) for sprite in sprites]

    out = g.copy()
    to_erase: set[tuple[int, int]] = set()
    for sprite, markers in zip(sprites, reachable_sets, strict=False):
        r0, c0, r1, c1 = sprite["bbox"]
        turns = len(markers) % 4
        out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(sprite["patch"], k=_rotation_k(turns))
        to_erase.update(markers)

    for r, c in to_erase:
        out[r, c] = BG
    return out


def _manhattan_to_sprite(cell: tuple[int, int], sprite_bbox: tuple[int, int, int, int]) -> int:
    r, c = cell
    r0, c0, r1, c1 = sprite_bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _chamber_boxes(h: int, w: int, wall_r: int, wall_c: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "TL": (0, 0, wall_r - 1, wall_c - 1),
        "TR": (0, wall_c + 1, wall_r - 1, w - 1),
        "BL": (wall_r + 1, 0, h - 1, wall_c - 1),
        "BR": (wall_r + 1, wall_c + 1, h - 1, w - 1),
    }


def _shared_side(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> str:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ac1 < bc0:
        return "left_to_right"
    if bc1 < ac0:
        return "right_to_left"
    if ar1 < br0:
        return "top_to_bottom"
    if br1 < ar0:
        return "bottom_to_top"
    raise ValueError("boxes must be disjoint chambers")


def _sprite_bbox_for_side(
    chamber: tuple[int, int, int, int],
    side: str | None,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = chamber
    row_choices = list(range(r0, r1 - 1))
    col_choices = list(range(c0, c1 - 1))
    if side == "left":
        c = c0
        r = rng.choice(row_choices)
    elif side == "right":
        c = c1 - 2
        r = rng.choice(row_choices)
    elif side == "top":
        r = r0
        c = rng.choice(col_choices)
    elif side == "bottom":
        r = r1 - 2
        c = rng.choice(col_choices)
    else:
        r = rng.choice(row_choices)
        c = rng.choice(col_choices)
    return (r, c, r + 2, c + 2)


def _cells_in_box(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _marker_ok(
    cell: tuple[int, int],
    chosen: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    if cell in reserved:
        return False
    r, c = cell
    for rr, cc in chosen:
        if abs(r - rr) + abs(c - cc) <= 1:
            return False
    return True


def _pick_marker_cells(
    chamber: tuple[int, int, int, int],
    count: int,
    sprite_bbox: tuple[int, int, int, int] | None,
    rng: random.Random,
    *,
    required: list[tuple[int, int]] | None = None,
    first_far: bool = False,
    reserved: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    required = list(required or [])
    reserved = set(reserved or set())
    occupied = set(required)
    if sprite_bbox is not None:
        occupied.update(_cells_in_box(sprite_bbox))

    chosen: list[tuple[int, int]] = []
    for cell in required:
        if not _marker_ok(cell, chosen, reserved):
            raise ValueError("required marker cells are not mutually valid")
        chosen.append(cell)

    candidates = [cell for cell in _cells_in_box(chamber) if cell not in occupied]
    rng.shuffle(candidates)
    if sprite_bbox is not None:
        candidates.sort(key=lambda cell: _manhattan_to_sprite(cell, sprite_bbox), reverse=True)

    need = count - len(chosen)
    if need < 0:
        raise ValueError("too many required marker cells")

    if first_far and need > 0 and sprite_bbox is not None:
        far = [cell for cell in candidates if _manhattan_to_sprite(cell, sprite_bbox) >= 4 and _marker_ok(cell, chosen, reserved)]
        if far:
            chosen.append(far[0])
            candidates.remove(far[0])
            need -= 1

    for cell in candidates:
        if need == 0:
            break
        if _marker_ok(cell, chosen, reserved):
            chosen.append(cell)
            need -= 1
    if need != 0:
        raise ValueError("not enough marker cells")
    return chosen


def _side_hint_from_relation(relation: str) -> str:
    return {
        "left_to_right": "left",
        "right_to_left": "right",
        "top_to_bottom": "top",
        "bottom_to_top": "bottom",
    }[relation]


def _reserved_blocked_marker(
    empty_box: tuple[int, int, int, int],
    sprite_box: tuple[int, int, int, int],
    relation: str,
    rng: random.Random,
) -> tuple[int, int]:
    er0, ec0, er1, ec1 = empty_box
    sr0, sc0, sr1, sc1 = sprite_box
    if relation == "left_to_right":
        return rng.choice(list(range(sr0, sr1 + 1))), ec1
    if relation == "right_to_left":
        return rng.choice(list(range(sr0, sr1 + 1))), ec0
    if relation == "top_to_bottom":
        return er1, rng.choice(list(range(sc0, sc1 + 1)))
    return er0, rng.choice(list(range(sc0, sc1 + 1)))


def _paint_sprite(grid: np.ndarray, sprite_box: tuple[int, int, int, int], sprite_patch: np.ndarray) -> None:
    r0, c0, r1, c1 = sprite_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = sprite_patch


def _place_markers(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = RED


def _near_unreachable_exists(grid: np.ndarray) -> bool:
    sprites = _sprite_records(grid)
    all_markers = set(map(tuple, np.argwhere(grid == RED)))
    for sprite in sprites:
        reachable = _reachable_markers(grid, sprite, sprites)
        unreachable = all_markers - reachable
        if not unreachable:
            continue
        for cell in unreachable:
            if _manhattan_to_sprite(cell, sprite["bbox"]) <= 2:
                return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if int(np.count_nonzero(grid == RED)) < 8 or int(np.count_nonzero(grid == RED)) > 12:
        return False
    if int(np.count_nonzero(grid == WALL)) == 0:
        return False

    sprites = _sprite_records(grid)
    reachable_sets = [_reachable_markers(grid, sprite, sprites) for sprite in sprites]
    counts = [len(markers) for markers in reachable_sets]
    mods = [count % 4 for count in counts]
    if any(count < 1 for count in counts):
        return False
    if len(set(mods)) != 3:
        return False

    marker_owner_count: dict[tuple[int, int], int] = {}
    for markers in reachable_sets:
        for cell in markers:
            marker_owner_count[cell] = marker_owner_count.get(cell, 0) + 1
    if any(v > 1 for v in marker_owner_count.values()):
        return False

    if not _near_unreachable_exists(grid):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out == RED):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 19)
    w = rng.randint(15, 19)
    wall_r = rng.randint(5, h - 6)
    wall_c = rng.randint(5, w - 6)

    grid = np.full((h, w), BG, dtype=int)
    grid[wall_r, :] = WALL
    grid[:, wall_c] = WALL

    chambers = _chamber_boxes(h, w, wall_r, wall_c)
    empty_key = rng.choice(CHAMBER_KEYS)
    sprite_keys = [key for key in CHAMBER_KEYS if key != empty_key]
    rng.shuffle(sprite_keys)

    empty_box = chambers[empty_key]
    blocked_sprite_key = rng.choice(
        [key for key in sprite_keys if key[0] == empty_key[0] or key[1] == empty_key[1]]
    )
    blocked_box = chambers[blocked_sprite_key]
    relation = _shared_side(empty_box, blocked_box)

    sprite_patch = rng.choice(SPRITE_PROTOTYPES)
    sprite_boxes: dict[str, tuple[int, int, int, int]] = {}
    for key in sprite_keys:
        chamber = chambers[key]
        if key == blocked_sprite_key:
            side = _side_hint_from_relation(relation)
        else:
            side = None
        sprite_boxes[key] = _sprite_bbox_for_side(chamber, side, rng)
        _paint_sprite(grid, sprite_boxes[key], sprite_patch)

    missing_mod = rng.randrange(4)
    mods = [mod for mod in range(4) if mod != missing_mod]
    rng.shuffle(mods)
    counts_by_key = {
        key: BASE_COUNT_BY_MOD[mod]
        for key, mod in zip(sprite_keys, mods, strict=False)
    }
    reachable_total = sum(counts_by_key.values())
    extra_low = max(1, 8 - reachable_total)
    extra_high = min(4, 12 - reachable_total)
    if extra_low > extra_high:
        return None
    extra_markers = rng.randint(extra_low, extra_high)

    blocked_marker = _reserved_blocked_marker(empty_box, sprite_boxes[blocked_sprite_key], relation, rng)
    empty_cells = _pick_marker_cells(
        empty_box,
        extra_markers,
        None,
        rng,
        required=[blocked_marker],
    )
    _place_markers(grid, empty_cells)

    for key in sprite_keys:
        chamber = chambers[key]
        sprite_box = sprite_boxes[key]
        try:
            marker_cells = _pick_marker_cells(
                chamber,
                counts_by_key[key],
                sprite_box,
                rng,
                first_far=(key == blocked_sprite_key),
            )
        except ValueError:
            return None
        _place_markers(grid, marker_cells)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
