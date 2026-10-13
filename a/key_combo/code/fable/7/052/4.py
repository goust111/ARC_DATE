from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, find_components, get_bg, in_bounds
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color, find_components, get_bg, in_bounds


BG = Color.BLACK
WALL = Color.GREY
RED = Color.RED
GREEN = Color.GREEN
MOBILE_COLORS = [Color.BLUE, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STOP_TYPES = ("edge", "wall", "sprite")

BASE_SHAPES = [
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
]

_RECENT_INPUT_SIGS: list[str] = []
_RECENT_OUTPUT_SIGS: list[str] = []
_RECENT_LIMIT = 256


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(cells, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    coords = sorted((int(r), int(c)) for r, c in arr)
    return tuple(coords)


def _rotate_once(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(cells, dtype=int)
    h = int(arr[:, 0].max()) + 1
    rot = np.stack([arr[:, 1], h - 1 - arr[:, 0]], axis=1)
    return _normalize(tuple(map(tuple, rot)))


def _all_shapes() -> list[tuple[tuple[int, int], ...]]:
    out: set[tuple[tuple[int, int], ...]] = set()
    for shape in BASE_SHAPES:
        cur = _normalize(shape)
        for _ in range(4):
            out.add(cur)
            cur = _rotate_once(cur)
    return sorted(out)


SHAPES = _all_shapes()


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shift(shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + r, ac + c) for r, c in shape}


def _padded(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _grid_sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _reserve_signature(sig_list: list[str], sig: str) -> None:
    sig_list.append(sig)
    if len(sig_list) > _RECENT_LIMIT:
        del sig_list[: len(sig_list) - _RECENT_LIMIT]


def _boundary_cells(shape: tuple[tuple[int, int], ...], direction: tuple[int, int], *, front: bool) -> list[tuple[int, int]]:
    dr, dc = direction if front else (-direction[0], -direction[1])
    shape_set = set(shape)
    return [(r, c) for r, c in shape if (r + dr, c + dc) not in shape_set]


def _sweep_cells(cells: set[tuple[int, int]], direction: tuple[int, int], steps: int) -> set[tuple[int, int]]:
    dr, dc = direction
    out: set[tuple[int, int]] = set()
    for k in range(steps + 1):
        out.update({(r + dr * k, c + dc * k) for r, c in cells})
    return out


def _slide(cells: set[tuple[int, int]], direction: tuple[int, int], blocked: set[tuple[int, int]], shape: tuple[int, int]) -> tuple[set[tuple[int, int]], int]:
    h, w = shape
    dr, dc = direction
    cur = set(cells)
    steps = 0
    while True:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if any(not in_bounds((h, w), r, c) for r, c in nxt):
            break
        if nxt & blocked:
            break
        cur = nxt
        steps += 1
    return cur, steps


def _render_sprite(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _parse_dominoes(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]]]:
    dominoes: list[dict] = []
    all_cells: set[tuple[int, int]] = set()
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        colors=[RED, GREEN],
        return_format="dict",
    ):
        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(coords) == 2 and colors == {RED, GREEN}, "red/green cells must form disjoint 2-cell dominoes"
        red = next((r, c) for r, c in coords if int(grid[r, c]) == RED)
        green = next((r, c) for r, c in coords if int(grid[r, c]) == GREEN)
        dr = green[0] - red[0]
        dc = green[1] - red[1]
        assert abs(dr) + abs(dc) == 1, "each instruction must be an orthogonally adjacent red-green pair"
        dominoes.append({"red": red, "green": green, "direction": (dr, dc)})
        all_cells.update(coords)
    dominoes.sort(key=lambda d: (d["green"], d["red"]))
    assert dominoes, "expected at least one red-green instruction domino"
    return dominoes, all_cells


def _parse_sprites(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    colors = [col for col in MOBILE_COLORS if int(col) in set(map(int, np.unique(grid)))]
    sprites: list[dict] = []
    by_cell: dict[tuple[int, int], int] = {}
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        return_format="dict",
    ):
        cells = {tuple(map(int, rc)) for rc in comp["coords"]}
        sid = len(sprites)
        sprites.append(
            {
                "id": sid,
                "color": int(comp["color"]),
                "cells": cells,
                "area": int(comp["area"]),
            }
        )
        for cell in cells:
            by_cell[cell] = sid
    assert sprites, "expected at least one movable sprite"
    return sprites, by_cell


def _ray_target(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    cell_to_sprite: dict[tuple[int, int], int],
    domino_cells: set[tuple[int, int]],
) -> int | None:
    h, w = grid.shape
    dr, dc = direction
    r, c = start
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        if (r, c) in domino_cells:
            continue
        if int(grid[r, c]) == WALL:
            return None
        sid = cell_to_sprite.get((r, c))
        if sid is not None:
            return sid


def _stop_reason(final_cells: set[tuple[int, int]], direction: tuple[int, int], walls: set[tuple[int, int]], other_sprites: set[tuple[int, int]], shape: tuple[int, int]) -> str:
    h, w = shape
    dr, dc = direction
    shifted = {(r + dr, c + dc) for r, c in final_cells}
    if any(not in_bounds((h, w), r, c) for r, c in shifted):
        return "edge"
    if shifted & walls:
        return "wall"
    assert shifted & other_sprites, "a non-edge stop must come from a wall or another sprite"
    return "sprite"


def _solve(grid: np.ndarray, *, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "grid must be a 2D ARC-sized array"
    assert int(get_bg(g)) == BG, "background must be black"

    dominoes, domino_cells = _parse_dominoes(g)
    sprites, cell_to_sprite = _parse_sprites(g)
    walls = {tuple(map(int, rc)) for rc in np.argwhere(g == WALL)}

    target_map: dict[int, dict] = {}
    for domino in dominoes:
        sid = _ray_target(g, domino["green"], domino["direction"], cell_to_sprite, domino_cells)
        if sid is None:
            continue
        assert sid not in target_map, "instructions must target distinct sprites so execution order is irrelevant"
        target_map[sid] = domino

    all_sprite_cells = set().union(*(sprite["cells"] for sprite in sprites))
    final_positions: dict[int, set[tuple[int, int]]] = {}
    moves: list[dict] = []
    for sprite in sprites:
        sid = int(sprite["id"])
        if sid not in target_map:
            continue
        blocked = walls | (all_sprite_cells - sprite["cells"])
        final_cells, steps = _slide(sprite["cells"], target_map[sid]["direction"], blocked, g.shape)
        final_positions[sid] = final_cells
        moves.append(
            {
                "sprite_id": sid,
                "color": int(sprite["color"]),
                "area": int(sprite["area"]),
                "direction": tuple(target_map[sid]["direction"]),
                "steps": int(steps),
                "stop_reason": _stop_reason(final_cells, target_map[sid]["direction"], walls, all_sprite_cells - sprite["cells"], g.shape),
            }
        )

    out = g.copy()
    for domino in dominoes:
        out[domino["red"]] = BG
        out[domino["green"]] = BG
    for sid, final_cells in final_positions.items():
        sprite = sprites[sid]
        for r, c in sprite["cells"]:
            out[r, c] = BG
        _render_sprite(out, final_cells, sprite["color"])

    if not with_meta:
        return out

    meta = {
        "dominoes": dominoes,
        "sprites": sprites,
        "moves": sorted(moves, key=lambda m: m["sprite_id"]),
        "targeted_ids": {m["sprite_id"] for m in moves},
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each red-green domino as a push instruction, ray-select its sprite, then slide that sprite to the stop.

    Input structure:
    - A black background contains grey wall pixels, several monochrome movable sprites, and disjoint 2-cell red-green dominoes.
    - Every domino is orthogonally adjacent; the direction is from its red cell to its green cell.
    - Sprites use colors other than red, green, grey, and black.

    Rule:
    - For each domino, start from the green cell and cast a ray in the domino's direction; the first sprite hit by that ray is the target.
    - Move the entire target sprite rigidly along that same direction until one more step would hit the grid edge, a grey wall, or another sprite.
    - Erase every domino from the scene; any sprite that is never hit by a ray stays where it started.

    Key insight:
    - The red-green pair is not a block to match or remove: its color order encodes a vector, and the ray from the green cell chooses which sprite receives the slide command.
    """
    return _solve(grid, with_meta=False)


def _make_wall_segment(
    stopper: tuple[int, int],
    direction: tuple[int, int],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    if direction[0] == 0:
        perp = (1, 0)
    else:
        perp = (0, 1)
    candidates: list[set[tuple[int, int]]] = []
    for length in (3, 2, 1):
        for offset in range(length):
            cells: set[tuple[int, int]] = set()
            ok = True
            for k in range(length):
                rr = stopper[0] + (k - offset) * perp[0]
                cc = stopper[1] + (k - offset) * perp[1]
                if not in_bounds((h, w), rr, cc) or (rr, cc) in forbidden:
                    ok = False
                    break
                cells.add((rr, cc))
            if ok:
                candidates.append(cells)
    if not candidates:
        return None
    return random.choice(candidates)


def _make_blocker(
    stopper: tuple[int, int],
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    blocker_shapes = [shape for shape in SHAPES if 1 <= len(shape) <= 2]
    random.shuffle(blocker_shapes)
    for shape in blocker_shapes:
        for touch in random.sample(list(shape), k=len(shape)):
            anchor = (stopper[0] - touch[0], stopper[1] - touch[1])
            cells = _shift(shape, anchor)
            if all(in_bounds((h, w), r, c) for r, c in cells) and not (cells & forbidden):
                return cells
    return None


def _try_place_instruction(
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    direction: tuple[int, int],
    stop_type: str,
    target_color: int,
    blocker_color: int | None,
) -> dict | None:
    for _ in range(250):
        shape = random.choice([s for s in SHAPES if 2 <= len(s) <= 4] + [((0, 0),)])
        shape_h, shape_w = _shape_dims(shape)
        trail = _boundary_cells(shape, direction, front=False)
        front = _boundary_cells(shape, direction, front=True)
        aim_rel = random.choice(trail)
        lead_rel = random.choice(front)
        gap = random.randint(1, 3)
        steps = random.randint(1, 5)

        if stop_type == "edge":
            if direction == (0, 1):
                final_c = w - shape_w
                if final_c - steps - gap - 1 < 0:
                    continue
                final_r = random.randint(0, h - shape_h)
            elif direction == (0, -1):
                final_c = 0
                if shape_w - 1 + steps + gap + 1 >= w:
                    continue
                final_r = random.randint(0, h - shape_h)
            elif direction == (1, 0):
                final_r = h - shape_h
                if final_r - steps - gap - 1 < 0:
                    continue
                final_c = random.randint(0, w - shape_w)
            else:
                final_r = 0
                if shape_h - 1 + steps + gap + 1 >= h:
                    continue
                final_c = random.randint(0, w - shape_w)
            stopper_cells: set[tuple[int, int]] = set()
            blocker_cells: set[tuple[int, int]] = set()
        else:
            if direction == (0, 1):
                if w - shape_w - 1 < 0:
                    continue
                final_r = random.randint(0, h - shape_h)
                final_c = random.randint(0, w - shape_w - 1)
            elif direction == (0, -1):
                if 1 > w - shape_w:
                    continue
                final_r = random.randint(0, h - shape_h)
                final_c = random.randint(1, w - shape_w)
            elif direction == (1, 0):
                if h - shape_h - 1 < 0:
                    continue
                final_r = random.randint(0, h - shape_h - 1)
                final_c = random.randint(0, w - shape_w)
            else:
                if 1 > h - shape_h:
                    continue
                final_r = random.randint(1, h - shape_h)
                final_c = random.randint(0, w - shape_w)

        input_anchor = (final_r - steps * direction[0], final_c - steps * direction[1])
        input_cells = _shift(shape, input_anchor)
        final_cells = _shift(shape, (final_r, final_c))
        if not all(in_bounds((h, w), r, c) for r, c in input_cells):
            continue

        aim_abs = (input_anchor[0] + aim_rel[0], input_anchor[1] + aim_rel[1])
        green = (aim_abs[0] - gap * direction[0], aim_abs[1] - gap * direction[1])
        red = (green[0] - direction[0], green[1] - direction[1])
        if not (in_bounds((h, w), *green) and in_bounds((h, w), *red)):
            continue
        if green in input_cells or red in input_cells:
            continue

        corridor = {
            (green[0] + k * direction[0], green[1] + k * direction[1])
            for k in range(1, gap + 1)
        }
        if aim_abs not in corridor:
            continue
        corridor.remove(aim_abs)

        stopper = (final_r + lead_rel[0] + direction[0], final_c + lead_rel[1] + direction[1])
        base_forbidden = occupied | input_cells | {red, green} | corridor | _sweep_cells(input_cells, direction, steps)
        blocker_cells = set()
        if stop_type == "wall":
            stopper_cells = _make_wall_segment(stopper, direction, h, w, base_forbidden)
            if stopper_cells is None:
                continue
        elif stop_type == "sprite":
            stopper_cells = set()
            blocker_cells = _make_blocker(stopper, h, w, base_forbidden)
            if blocker_cells is None or blocker_color is None:
                continue
        else:
            stopper_cells = set()

        core = input_cells | {red, green} | stopper_cells | blocker_cells
        scene = core | corridor | _sweep_cells(input_cells, direction, steps)
        if scene & reserved:
            continue
        if core & occupied:
            continue
        if stop_type != "edge" and stopper not in (stopper_cells | blocker_cells):
            continue

        blocked = (occupied | stopper_cells | blocker_cells) - input_cells
        solved_final, solved_steps = _slide(input_cells, direction, blocked, (h, w))
        if solved_final != final_cells or solved_steps != steps:
            continue

        return {
            "direction": direction,
            "stop_type": stop_type,
            "shape": shape,
            "target_color": target_color,
            "blocker_color": blocker_color,
            "input_cells": input_cells,
            "final_cells": final_cells,
            "domino": {"red": red, "green": green},
            "wall_cells": stopper_cells,
            "blocker_cells": blocker_cells,
            "reserved": _padded(scene, h, w, pad=1),
            "occupied": core,
        }
    return None


def _try_place_extra_sprite(
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    color: int,
) -> dict | None:
    for _ in range(120):
        shape = random.choice([s for s in SHAPES if 1 <= len(s) <= 4])
        sh, sw = _shape_dims(shape)
        anchor = (random.randint(0, h - sh), random.randint(0, w - sw))
        cells = _shift(shape, anchor)
        scene = _padded(cells, h, w, pad=1)
        if cells & occupied or scene & reserved:
            continue
        return {"color": color, "cells": cells, "reserved": scene}
    return None


def _build_scene() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    n_instr = random.choice([2, 3, 3])

    if n_instr == 3:
        stop_types = random.sample(list(STOP_TYPES), k=3)
    else:
        stop_types = random.sample(list(STOP_TYPES), k=2)

    directions = [random.choice(DIRS) for _ in range(n_instr)]
    while len(set(directions)) < 2:
        directions = [random.choice(DIRS) for _ in range(n_instr)]

    needed_colors = n_instr + stop_types.count("sprite") + 1
    if needed_colors > len(MOBILE_COLORS):
        return None
    palette = random.sample(MOBILE_COLORS, k=needed_colors)
    target_colors = palette[:n_instr]
    blocker_colors = palette[n_instr : n_instr + stop_types.count("sprite")]
    extra_color = palette[-1]

    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()
    blocker_idx = 0
    placements: list[dict] = []

    ordering = list(range(n_instr))
    random.shuffle(ordering)
    for idx in ordering:
        blocker_color = None
        if stop_types[idx] == "sprite":
            blocker_color = blocker_colors[blocker_idx]
            blocker_idx += 1
        placed = _try_place_instruction(
            h,
            w,
            reserved,
            occupied,
            directions[idx],
            stop_types[idx],
            target_colors[idx],
            blocker_color,
        )
        if placed is None:
            return None
        placements.append(placed)
        occupied |= placed["occupied"]
        reserved |= placed["reserved"]

    extra = _try_place_extra_sprite(h, w, reserved, occupied, extra_color)
    if extra is None:
        return None

    for placed in placements:
        _render_sprite(grid, placed["input_cells"], placed["target_color"])
        if placed["blocker_cells"]:
            _render_sprite(grid, placed["blocker_cells"], int(placed["blocker_color"]))
        if placed["wall_cells"]:
            _render_sprite(grid, placed["wall_cells"], WALL)
        grid[placed["domino"]["red"]] = RED
        grid[placed["domino"]["green"]] = GREEN
    _render_sprite(grid, extra["cells"], extra["color"])

    return grid


def _reverse_dominoes(grid: np.ndarray) -> np.ndarray:
    rev = np.array(grid, copy=True)
    dominoes, _ = _parse_dominoes(rev)
    for domino in dominoes:
        rev[domino["red"]] = GREEN
        rev[domino["green"]] = RED
    return rev


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid, with_meta=True)
    except AssertionError:
        return False

    moves = meta["moves"]
    if len(meta["dominoes"]) < 2:
        return False
    if len(moves) != len(meta["dominoes"]):
        return False
    if any(move["steps"] <= 0 for move in moves):
        return False
    if len(meta["sprites"]) <= len(moves):
        return False
    if len({move["direction"] for move in moves}) < 2:
        return False
    if len({move["stop_reason"] for move in moves}) < min(2, len(moves)):
        return False
    if not any(move["area"] >= 2 for move in moves):
        return False
    if np.array_equal(grid, out):
        return False
    for domino in meta["dominoes"]:
        if int(out[domino["red"]]) != BG or int(out[domino["green"]]) != BG:
            return False

    try:
        reversed_out = main(_reverse_dominoes(grid))
    except AssertionError:
        reversed_out = None
    if reversed_out is not None and np.array_equal(reversed_out, out):
        return False

    out_sig = _grid_sig(out)
    if out_sig in _RECENT_OUTPUT_SIGS:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _build_scene()
        if grid is None:
            continue
        if not _validate_scene(grid):
            continue
        inp_sig = _grid_sig(grid)
        out_sig = _grid_sig(main(grid))
        if inp_sig in _RECENT_INPUT_SIGS or out_sig in _RECENT_OUTPUT_SIGS:
            continue
        _reserve_signature(_RECENT_INPUT_SIGS, inp_sig)
        _reserve_signature(_RECENT_OUTPUT_SIGS, out_sig)
        return grid
    raise RuntimeError("failed to generate a valid non-degenerate domino-slide scene")
