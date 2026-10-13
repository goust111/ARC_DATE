import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

EMITTER_COLORS = [BLUE, RED, GREEN, MAGENTA, ORANGE, AZURE]
SPECIAL_INPUT_COLORS = {BLACK, YELLOW, GREY, MAROON}

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
REFLECT = {
    "\\": {
        UP: LEFT,
        RIGHT: DOWN,
        DOWN: RIGHT,
        LEFT: UP,
    },
    "/": {
        UP: RIGHT,
        RIGHT: UP,
        DOWN: LEFT,
        LEFT: DOWN,
    },
}

MIN_H = 10
MAX_H = 18
MIN_W = 10
MAX_W = 18
MAX_RETRIES = 400


def _is_boundary(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _direction_from_edge(r: int, c: int, h: int, w: int) -> int:
    if r == 0 and 0 < c < w - 1:
        return DOWN
    if r == h - 1 and 0 < c < w - 1:
        return UP
    if c == 0 and 0 < r < h - 1:
        return RIGHT
    if c == w - 1 and 0 < r < h - 1:
        return LEFT
    raise AssertionError("emitters must sit on a single non-corner edge cell")


def _mirror_kind_for_turn(in_dir: int, out_dir: int) -> str:
    for kind, mapping in REFLECT.items():
        if mapping[in_dir] == out_dir:
            return kind
    raise AssertionError("invalid turn")


def _contact_top_left_options(r: int, c: int, d: int) -> list[tuple[int, int]]:
    if d == DOWN:
        return [(r + 1, c), (r + 1, c - 1)]
    if d == UP:
        return [(r - 2, c), (r - 2, c - 1)]
    if d == RIGHT:
        return [(r, c + 1), (r - 1, c + 1)]
    if d == LEFT:
        return [(r, c - 2), (r - 1, c - 2)]
    raise AssertionError("bad direction")


def _mirror_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _mirror_moat(top_left: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    cells = _mirror_cells(top_left)
    moat = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    moat.add((rr, cc))
    return moat


def _detect_mirrors(grid: np.ndarray) -> tuple[list[tuple[int, int, str]], dict[tuple[int, int], str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    occupied: set[tuple[int, int]] = set()
    mirrors: list[tuple[int, int, str]] = []
    mirror_map: dict[tuple[int, int], str] = {}

    yellow_cells = sorted(tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW))
    for yr, yc in yellow_cells:
        candidates = set()
        for r0 in (yr, yr - 1):
            for c0 in (yc, yc - 1):
                if not (0 <= r0 < h - 1 and 0 <= c0 < w - 1):
                    continue
                block = g[r0 : r0 + 2, c0 : c0 + 2]
                if np.count_nonzero(block == YELLOW) == 1 and np.count_nonzero(block == GREY) == 3:
                    candidates.add((r0, c0))

        free_candidates = []
        for r0, c0 in sorted(candidates):
            cells = _mirror_cells((r0, c0))
            if cells & occupied:
                continue
            free_candidates.append((r0, c0))

        assert len(free_candidates) == 1, "each yellow cell must identify exactly one mirror"
        r0, c0 = free_candidates[0]
        assert 1 <= r0 <= h - 3 and 1 <= c0 <= w - 3, "mirrors must be internal 2x2 blocks"
        rel = (yr - r0, yc - c0)
        kind = "\\" if rel in {(0, 0), (1, 1)} else "/"
        cells = _mirror_cells((r0, c0))
        occupied |= cells
        mirrors.append((r0, c0, kind))
        for cell in cells:
            mirror_map[cell] = kind

    mirror_mask = (g == GREY) | (g == YELLOW)
    assert np.count_nonzero(mirror_mask) == len(occupied), "grey/yellow cells must belong to mirrors"
    return mirrors, mirror_map


def _detect_emitters(grid: np.ndarray, mirror_map: dict[tuple[int, int], str]) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    emitters = []
    seen = set()
    boundary_cells = []
    for c in range(w):
        boundary_cells.append((0, c))
        boundary_cells.append((h - 1, c))
    for r in range(1, h - 1):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, w - 1))

    for r, c in boundary_cells:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        color = int(g[r, c])
        if color in SPECIAL_INPUT_COLORS:
            continue
        assert (r, c) not in mirror_map, "mirror cells cannot sit on the edge"
        direction = _direction_from_edge(r, c, h, w)
        emitters.append((r, c, color, direction))

    assert emitters, "at least one emitter is required"
    colors = [color for _, _, color, _ in emitters]
    assert len(colors) == len(set(colors)), "emitters must have distinct colors"
    return emitters


def _trace_rays(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    mirrors, mirror_map = _detect_mirrors(g)
    absorbers = {tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)}
    for r, c in absorbers:
        assert 0 < r < h - 1 and 0 < c < w - 1, "absorbers must be internal"

    emitters = _detect_emitters(g, mirror_map)
    painted: dict[tuple[int, int], set[int]] = defaultdict(set)
    ray_info = []

    for er, ec, color, direction in emitters:
        cur_r, cur_c = er, ec
        d = direction
        seen_states = set()
        path_cells = []
        reflections = 0

        while True:
            state = (cur_r, cur_c, d)
            assert state not in seen_states, "ray path must not cycle"
            seen_states.add(state)

            dr, dc = DIRS[d]
            nr = cur_r + dr
            nc = cur_c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                termination = "boundary"
                break

            if (nr, nc) in absorbers:
                termination = "absorber"
                break

            if (nr, nc) in mirror_map:
                d = REFLECT[mirror_map[(nr, nc)]][d]
                reflections += 1
                continue

            cur_r, cur_c = nr, nc
            if int(g[cur_r, cur_c]) == BLACK:
                painted[(cur_r, cur_c)].add(color)
                path_cells.append((cur_r, cur_c))

            if _is_boundary(cur_r, cur_c, h, w):
                termination = "boundary"
                break

        ray_info.append(
            {
                "emitter": (er, ec, color),
                "termination": termination,
                "reflections": reflections,
                "path_cells": path_cells,
            }
        )

    out = g.copy()
    for (r, c), colors in painted.items():
        out[r, c] = YELLOW if len(colors) >= 2 else next(iter(colors))

    meta = {
        "mirrors": mirrors,
        "emitters": emitters,
        "absorbers": absorbers,
        "ray_info": ray_info,
        "painted": painted,
        "intersection_cells": [cell for cell, colors in painted.items() if len(colors) >= 2],
        "reflection_count": sum(info["reflections"] for info in ray_info),
        "termination_types": [info["termination"] for info in ray_info],
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge-launched rays, reflect them off 2x2 coded mirrors, and paint the traversed black cells.
    """
    out, _meta = _trace_rays(grid)
    return out


def _choose_primary_layout(
    h: int, w: int, need_intersection: bool
) -> tuple[tuple[int, int], int, tuple[int, int], int, int, int]:
    vertical_side = random.choice(["top", "bottom"])
    horizontal_side = random.choice(["left", "right"])
    turn_dir = random.choice([LEFT, RIGHT])

    cc_min = 4
    cc_max = w - 5
    if cc_min > cc_max:
        raise AssertionError("grid too narrow")
    cc = random.randint(cc_min, cc_max)

    tr = random.randint(3, h - 4)
    if vertical_side == "top":
        if need_intersection:
            cr = random.randint(2, tr - 2)
        else:
            cr = random.randint(tr + 2, h - 3)
        primary_emitter = (0, cc)
        primary_dir = DOWN
    else:
        if need_intersection:
            cr = random.randint(tr + 2, h - 3)
        else:
            cr = random.randint(2, tr - 2)
        primary_emitter = (h - 1, cc)
        primary_dir = UP

    secondary_emitter = (cr, 0) if horizontal_side == "left" else (cr, w - 1)
    secondary_dir = RIGHT if horizontal_side == "left" else LEFT
    return primary_emitter, primary_dir, secondary_emitter, secondary_dir, tr, turn_dir


def _segment_between(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _choose_absorber_on_segment(segment: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(segment) < 4:
        return None
    idx = random.randint(2, len(segment) - 2)
    return segment[idx]


def _place_mirror(
    mirrors: list[tuple[int, int, str]],
    object_cells: set[tuple[int, int]],
    mirror_moat: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
    h: int,
    w: int,
    top_left: tuple[int, int],
    kind: str,
) -> bool:
    r0, c0 = top_left
    if not (1 <= r0 <= h - 3 and 1 <= c0 <= w - 3):
        return False
    cells = _mirror_cells(top_left)
    if cells & object_cells:
        return False
    if cells & reserved_path:
        return False
    if cells & mirror_moat:
        return False
    mirrors.append((r0, c0, kind))
    object_cells |= cells
    mirror_moat |= _mirror_moat(top_left, h, w)
    return True


def _render_input(
    h: int,
    w: int,
    emitters: list[tuple[int, int, int]],
    mirrors: list[tuple[int, int, str]],
    absorbers: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in absorbers:
        grid[r, c] = MAROON

    for r0, c0, kind in mirrors:
        grid[r0 : r0 + 2, c0 : c0 + 2] = GREY
        if kind == "\\":
            yr, yc = random.choice([(r0, c0), (r0 + 1, c0 + 1)])
        else:
            yr, yc = random.choice([(r0, c0 + 1), (r0 + 1, c0)])
        grid[yr, yc] = YELLOW

    for r, c, color in emitters:
        grid[r, c] = color
    return grid


def _free_edge_positions(
    h: int,
    w: int,
    used: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    positions = []
    for c in range(1, w - 1):
        positions.append((0, c))
        positions.append((h - 1, c))
    for r in range(1, h - 1):
        positions.append((r, 0))
        positions.append((r, w - 1))
    return [pos for pos in positions if pos not in used]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_H, MAX_H)
    w = random.randint(MIN_W, MAX_W)
    need_intersection = random.random() < 0.5
    need_absorber_hit = random.random() < 0.5
    emitter_count = random.randint(2, 4)
    color_pool = random.sample(EMITTER_COLORS, emitter_count)

    try:
        primary_emitter, primary_dir, secondary_emitter, _secondary_dir, tr, turn_dir = _choose_primary_layout(
            h, w, need_intersection
        )
    except ValueError:
        return None
    except AssertionError:
        return None

    emitters = [
        (primary_emitter[0], primary_emitter[1], color_pool[0]),
        (secondary_emitter[0], secondary_emitter[1], color_pool[1]),
    ]

    object_cells = {(r, c) for r, c, _ in emitters}
    reserved_path: set[tuple[int, int]] = set()
    mirrors: list[tuple[int, int, str]] = []
    absorbers: set[tuple[int, int]] = set()
    mirror_moat: set[tuple[int, int]] = set()

    turn_cell = (tr, primary_emitter[1])
    primary_vertical = _segment_between(
        (primary_emitter[0] + DIRS[primary_dir][0], primary_emitter[1]),
        turn_cell,
    )
    reserved_path |= set(primary_vertical)

    secondary_row = secondary_emitter[0]
    if secondary_emitter[1] == 0:
        secondary_full = _segment_between((secondary_row, 1), (secondary_row, w - 1))
    else:
        secondary_full = _segment_between((secondary_row, w - 2), (secondary_row, 0))
    reserved_path |= set(secondary_full)

    primary_out = turn_dir
    primary_kind = _mirror_kind_for_turn(primary_dir, primary_out)
    options = _contact_top_left_options(turn_cell[0], turn_cell[1], primary_dir)
    random.shuffle(options)
    placed = False
    for top_left in options:
        if _place_mirror(mirrors, object_cells, mirror_moat, reserved_path, h, w, top_left, primary_kind):
            placed = True
            break
    if not placed:
        return None

    if primary_out == RIGHT:
        primary_tail = _segment_between((tr, primary_emitter[1] + 1), (tr, w - 1))
    else:
        primary_tail = _segment_between((tr, primary_emitter[1] - 1), (tr, 0))

    if need_absorber_hit:
        absorber_cell = _choose_absorber_on_segment(primary_tail)
        if absorber_cell is None:
            return None
        stop_index = primary_tail.index(absorber_cell)
        reserved_path |= set(primary_tail[:stop_index])
        absorbers.add(absorber_cell)
        object_cells.add(absorber_cell)
    else:
        reserved_path |= set(primary_tail)

    used_edges = {(r, c) for r, c, _ in emitters}
    extra_edges = _free_edge_positions(h, w, used_edges | reserved_path)
    random.shuffle(extra_edges)
    for color in color_pool[2:]:
        if not extra_edges:
            return None
        pos = extra_edges.pop()
        emitters.append((pos[0], pos[1], color))
        used_edges.add(pos)
        object_cells.add(pos)

    opposite_kind = "/" if primary_kind == "\\" else "\\"
    extra_kinds = [opposite_kind]
    extra_kinds.extend(random.choice(["\\", "/"]) for _ in range(random.randint(0, 2)))
    for kind in extra_kinds:
        placed = False
        candidates = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
        random.shuffle(candidates)
        for top_left in candidates:
            if _place_mirror(mirrors, object_cells, mirror_moat, reserved_path, h, w, top_left, kind):
                placed = True
                break
        if not placed:
            return None

    for _ in range(random.randint(0, 2)):
        free_cells = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if (r, c) not in object_cells and (r, c) not in reserved_path
        ]
        if not free_cells:
            break
        cell = random.choice(free_cells)
        absorbers.add(cell)
        object_cells.add(cell)

    grid = _render_input(h, w, emitters, mirrors, absorbers)

    try:
        out, meta = _trace_rays(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if meta["reflection_count"] < 1:
        return None
    if need_intersection and not meta["intersection_cells"]:
        return None
    if need_absorber_hit and "absorber" not in meta["termination_types"]:
        return None
    if len(meta["mirrors"]) < 2:
        return None

    kinds = {kind for _, _, kind in meta["mirrors"]}
    if kinds != {"\\", "/"}:
        return None

    painted_count = len(meta["painted"])
    if painted_count < 8:
        return None

    unchanged_mask = grid != BLACK
    if not np.array_equal(out[unchanged_mask], grid[unchanged_mask]):
        return None

    if len(meta["emitters"]) != emitter_count:
        return None
    if len(set(meta["termination_types"])) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
