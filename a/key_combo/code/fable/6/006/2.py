from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


DIR_TOP = (-1, 0)
DIR_BOTTOM = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)

START_DIR_BY_SIDE = {
    "top": DIR_BOTTOM,
    "bottom": DIR_TOP,
    "left": DIR_RIGHT,
    "right": DIR_LEFT,
}

TURN_LEFT = {
    DIR_TOP: DIR_LEFT,
    DIR_LEFT: DIR_BOTTOM,
    DIR_BOTTOM: DIR_RIGHT,
    DIR_RIGHT: DIR_TOP,
}

TURN_RIGHT = {
    DIR_TOP: DIR_RIGHT,
    DIR_RIGHT: DIR_BOTTOM,
    DIR_BOTTOM: DIR_LEFT,
    DIR_LEFT: DIR_TOP,
}

REFLECT = {
    "/": {
        DIR_RIGHT: DIR_TOP,
        DIR_TOP: DIR_RIGHT,
        DIR_LEFT: DIR_BOTTOM,
        DIR_BOTTOM: DIR_LEFT,
    },
    "\\": {
        DIR_RIGHT: DIR_BOTTOM,
        DIR_BOTTOM: DIR_RIGHT,
        DIR_LEFT: DIR_TOP,
        DIR_TOP: DIR_LEFT,
    },
}

SEGMENT_ARM_CHOICES = [
    (1, 1),
    (0, 1),
    (1, 0),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 0),
    (0, 3),
]

GEN_RETRIES = 600


def _border_positions(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(w):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    return cells


def _is_border_cell(coord: tuple[int, int], h: int, w: int) -> bool:
    r, c = coord
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _is_inner_ring(coord: tuple[int, int], h: int, w: int) -> bool:
    r, c = coord
    return r in {1, h - 2} or c in {1, w - 2}


def _in_core(coord: tuple[int, int], h: int, w: int) -> bool:
    r, c = coord
    return 2 <= r <= h - 3 and 2 <= c <= w - 3


def _line_from(coord: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = coord
    dr, dc = direction
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if _is_border_cell((r, c), h, w):
            break
        cells.append((r, c))
    return cells


def _orientation_for_reflection(
    incoming: tuple[int, int], outgoing: tuple[int, int]
) -> str:
    for orientation, mapping in REFLECT.items():
        if mapping[incoming] == outgoing:
            return orientation
    raise AssertionError("invalid reflection pair")


def _segment_orientation(coords: np.ndarray) -> str:
    pts = [tuple(map(int, p)) for p in coords]
    assert len(pts) >= 2, "segments must contain at least 2 pixels"

    if len({r - c for r, c in pts}) == 1:
        ordered = sorted(pts)
        assert all(
            b[0] - a[0] == 1 and b[1] - a[1] == 1 for a, b in zip(ordered, ordered[1:])
        ), "expected consecutive \\ diagonal cells"
        return "\\"

    if len({r + c for r, c in pts}) == 1:
        ordered = sorted(pts, key=lambda p: (p[0], -p[1]))
        assert all(
            b[0] - a[0] == 1 and b[1] - a[1] == -1 for a, b in zip(ordered, ordered[1:])
        ), "expected consecutive / diagonal cells"
        return "/"

    raise AssertionError("component is not a straight 45-degree segment")


def _neighbor8(coord: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = coord
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    }


def _segment_cells(
    hit: tuple[int, int], orientation: str, arm_a: int, arm_b: int
) -> set[tuple[int, int]]:
    if orientation == "\\":
        step_a, step_b = (-1, -1), (1, 1)
    else:
        assert orientation == "/"
        step_a, step_b = (-1, 1), (1, -1)

    cells = {hit}
    for k in range(1, int(arm_a) + 1):
        cells.add((hit[0] + k * step_a[0], hit[1] + k * step_a[1]))
    for k in range(1, int(arm_b) + 1):
        cells.add((hit[0] + k * step_b[0], hit[1] + k * step_b[1]))
    return cells


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    border_vals = [int(g[r, c]) for r, c in _border_positions(h, w)]
    frame_color = int(Counter(border_vals).most_common(1)[0][0])
    rivet_colors = set(border_vals) - {frame_color}
    assert len(rivet_colors) == 1, "expected exactly one rivet color on the frame"
    mirror_color = int(next(iter(rivet_colors)))

    lamp_color = None
    lamp_coord = None
    segment_by_cell: dict[tuple[int, int], dict[str, object]] = {}
    segment_colors: set[int] = set()

    for comp in find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    ):
        color = int(comp["color"])
        coords = comp["coords"]
        pts = [tuple(map(int, p)) for p in coords]
        touches_border = any(_is_border_cell(pt, h, w) for pt in pts)
        if touches_border:
            continue

        if int(comp["area"]) == 1:
            (r, c) = pts[0]
            if _is_inner_ring((r, c), h, w):
                assert lamp_coord is None, "expected a unique lamp pixel"
                lamp_coord = (r, c)
                lamp_color = color
                continue

        orientation = _segment_orientation(coords)
        segment_colors.add(color)
        for pt in pts:
            assert pt not in segment_by_cell, "segments must not overlap"
            segment_by_cell[pt] = {"color": color, "orientation": orientation}

    assert lamp_coord is not None and lamp_color is not None, "missing lamp pixel"
    assert int(np.sum(g == lamp_color)) == 1, "lamp color must be unique in the input"
    assert mirror_color in segment_colors and len(segment_colors) == 2, "expected two segment colors"
    absorber_color = int(next(color for color in segment_colors if color != mirror_color))

    r, c = lamp_coord
    sides = []
    if r == 1:
        sides.append("top")
    if r == h - 2:
        sides.append("bottom")
    if c == 1:
        sides.append("left")
    if c == w - 2:
        sides.append("right")
    assert len(sides) == 1, "lamp must touch exactly one frame side"

    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": bg,
        "frame_color": frame_color,
        "mirror_color": mirror_color,
        "absorber_color": absorber_color,
        "lamp_color": lamp_color,
        "lamp_coord": lamp_coord,
        "start_dir": START_DIR_BY_SIDE[sides[0]],
        "segment_by_cell": segment_by_cell,
    }


def _trace_with_roles(
    scene: dict,
    *,
    mirror_color: int | None = None,
    absorber_color: int | None = None,
) -> np.ndarray:
    g = scene["grid"]
    h, w = scene["h"], scene["w"]
    mirror_color = int(scene["mirror_color"] if mirror_color is None else mirror_color)
    absorber_color = int(scene["absorber_color"] if absorber_color is None else absorber_color)
    lamp_color = int(scene["lamp_color"])

    out = g.copy()
    pos = tuple(scene["lamp_coord"])
    direction = tuple(scene["start_dir"])
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while True:
        state = (pos, direction)
        assert state not in seen, "beam entered a loop"
        seen.add(state)

        nr = pos[0] + direction[0]
        nc = pos[1] + direction[1]
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if _is_border_cell((nr, nc), h, w):
            break

        nxt = (nr, nc)
        value = int(g[nr, nc])
        if value == int(scene["bg"]):
            out[nr, nc] = lamp_color
            pos = nxt
            continue

        seg = scene["segment_by_cell"].get(nxt)
        assert seg is not None, "beam may only interact with diagonal segments"
        seg_color = int(seg["color"])
        if seg_color == absorber_color:
            break
        assert seg_color == mirror_color, "unexpected segment color role"
        direction = REFLECT[str(seg["orientation"])][direction]
        pos = nxt

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the lamp beam through diagonal segments, using frame rivets to decide which color reflects.

    Input structure:
    - A one-pixel-thick rectangular frame surrounds the grid, with a few rivets of a second color embedded on the border.
    - Inside the frame, the background is black except for one unique lamp pixel on the inner ring and several straight 45-degree diagonal segments in two colors.
    - One of the two segment colors matches the rivet color; both "/" and "\" orientations may appear.

    Rule:
    - The lamp emits a beam straight inward, perpendicular to the frame side it touches, and every traversed background cell becomes the lamp color.
    - Hitting a segment whose color matches the rivets causes a 90-degree mirror reflection determined by the segment orientation, while hitting the other segment color stops the beam on the previous cell.
    - The frame, rivets, lamp, and diagonal segments stay unchanged; the beam also stops when its next step would enter the border.

    Key insight:
    - Segment roles are not fixed by color alone: the border rivets identify the mirror color, so the frame must be read before tracing the light path.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _trace_with_roles(scene)


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _random_lamp(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 1, random.randint(2, w - 3)
    if side == "bottom":
        return h - 2, random.randint(2, w - 3)
    if side == "left":
        return random.randint(2, h - 3), 1
    assert side == "right"
    return random.randint(2, h - 3), w - 2


def _segment_candidates(
    hit: tuple[int, int],
    orientation: str,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    choices: list[set[tuple[int, int]]] = []
    pairs = SEGMENT_ARM_CHOICES[:]
    random.shuffle(pairs)
    for arm_a, arm_b in pairs:
        cells = _segment_cells(hit, orientation, arm_a, arm_b)
        if any(not _in_core(cell, h, w) for cell in cells):
            continue
        if cells & forbidden:
            continue
        choices.append(cells)
    return choices


def _mark_forbidden(cells: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> None:
    for cell in cells:
        forbidden.add(cell)
        forbidden.update(_neighbor8(cell))


def _approach_clear(
    cells: list[tuple[int, int]],
    used_route: set[tuple[int, int]],
    occupied_objects: set[tuple[int, int]],
) -> bool:
    return not any(cell in used_route or cell in occupied_objects for cell in cells)


def _build_route_plan(
    h: int, w: int, lamp: tuple[int, int], mirror_count: int, final_mode: str
) -> dict | None:
    current = lamp
    direction = START_DIR_BY_SIDE[
        "top" if lamp[0] == 1 else "bottom" if lamp[0] == h - 2 else "left" if lamp[1] == 1 else "right"
    ]
    route_cells = {lamp}
    occupied_objects = {lamp}
    object_forbidden = set(_neighbor8(lamp)) | {lamp}
    mirrors: list[dict] = []
    mirror_orientations: list[str] = []

    for index in range(mirror_count):
        line = _line_from(current, direction, h, w)
        options = []
        for step_idx, hit in enumerate(line, start=1):
            if step_idx < 2 or not _in_core(hit, h, w):
                continue
            approach = line[: step_idx - 1]
            if not _approach_clear(approach, route_cells, occupied_objects):
                break

            for turn_map in (TURN_LEFT, TURN_RIGHT):
                new_dir = turn_map[direction]
                after_line = _line_from(hit, new_dir, h, w)
                core_after = [cell for cell in after_line if _in_core(cell, h, w)]
                min_needed = 2 if index < mirror_count - 1 or final_mode == "absorb" else 1
                if len(core_after) < min_needed:
                    continue

                orientation = _orientation_for_reflection(direction, new_dir)
                for segment_cells in _segment_candidates(hit, orientation, h, w, object_forbidden | route_cells):
                    options.append(
                        {
                            "hit": hit,
                            "approach": approach,
                            "out_dir": new_dir,
                            "orientation": orientation,
                            "segment_cells": segment_cells,
                        }
                    )

        if not options:
            return None

        random.shuffle(options)
        options.sort(key=lambda item: (-len(item["approach"]), random.random()))
        choice = options[0]
        route_cells.update(choice["approach"])
        occupied_objects.update(choice["segment_cells"])
        _mark_forbidden(choice["segment_cells"], object_forbidden)
        mirrors.append(choice)
        mirror_orientations.append(str(choice["orientation"]))
        current = tuple(choice["hit"])
        direction = tuple(choice["out_dir"])

    if len(set(mirror_orientations)) < 2:
        return None

    if final_mode == "absorb":
        line = _line_from(current, direction, h, w)
        options = []
        for step_idx, hit in enumerate(line, start=1):
            if step_idx < 2 or not _in_core(hit, h, w):
                continue
            approach = line[: step_idx - 1]
            if not _approach_clear(approach, route_cells, occupied_objects):
                break
            for orientation in ("/", "\\"):
                for segment_cells in _segment_candidates(hit, orientation, h, w, object_forbidden | route_cells):
                    options.append(
                        {
                            "hit": hit,
                            "approach": approach,
                            "orientation": orientation,
                            "segment_cells": segment_cells,
                        }
                    )
        if not options:
            return None
        choice = random.choice(options)
        route_cells.update(choice["approach"])
        occupied_objects.update(choice["segment_cells"])
        _mark_forbidden(choice["segment_cells"], object_forbidden)
        absorber = choice
        exit_path: list[tuple[int, int]] = []
    else:
        line = _line_from(current, direction, h, w)
        if len(line) < 2:
            return None
        if not _approach_clear(line, route_cells, occupied_objects):
            return None
        route_cells.update(line)
        absorber = None
        exit_path = line

    painted_count = len(route_cells) - 1
    if painted_count < 6:
        return None

    return {
        "route_cells": route_cells,
        "occupied_objects": occupied_objects,
        "object_forbidden": object_forbidden,
        "mirrors": mirrors,
        "absorber": absorber,
        "exit_path": exit_path,
        "final_mode": final_mode,
    }


def _add_distractors(
    grid: np.ndarray,
    *,
    h: int,
    w: int,
    color: int,
    count: int,
    occupied_objects: set[tuple[int, int]],
    object_forbidden: set[tuple[int, int]],
    route_cells: set[tuple[int, int]],
) -> bool:
    for _ in range(count):
        placed = False
        for _attempt in range(80):
            hit = (random.randint(2, h - 3), random.randint(2, w - 3))
            orientation = random.choice(["/", "\\"])
            candidates = _segment_candidates(hit, orientation, h, w, object_forbidden | route_cells)
            if not candidates:
                continue
            cells = random.choice(candidates)
            _paint_cells(grid, cells, color)
            occupied_objects.update(cells)
            _mark_forbidden(cells, object_forbidden)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_rivets(
    grid: np.ndarray, mirror_color: int, lamp: tuple[int, int], count: int
) -> None:
    h, w = grid.shape
    forbidden = {
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (lamp[0] - 1, lamp[1]) if lamp[0] == 1 else (lamp[0] + 1, lamp[1]) if lamp[0] == h - 2 else (lamp[0], lamp[1] - 1) if lamp[1] == 1 else (lamp[0], lamp[1] + 1),
    }
    candidates = [cell for cell in _border_positions(h, w) if cell not in forbidden]
    random.shuffle(candidates)
    for cell in candidates[:count]:
        grid[cell] = int(mirror_color)


def _render_instance(
    h: int,
    w: int,
    frame_color: int,
    mirror_color: int,
    absorber_color: int,
    lamp_color: int,
    lamp: tuple[int, int],
    plan: dict,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    inp[0, :] = frame_color
    inp[h - 1, :] = frame_color
    inp[:, 0] = frame_color
    inp[:, w - 1] = frame_color
    inp[lamp] = lamp_color

    for mirror in plan["mirrors"]:
        _paint_cells(inp, set(mirror["segment_cells"]), mirror_color)
    if plan["absorber"] is not None:
        _paint_cells(inp, set(plan["absorber"]["segment_cells"]), absorber_color)

    occupied_objects = set(plan["occupied_objects"])
    object_forbidden = set(plan["object_forbidden"])

    extra_mirror = random.randint(1, 2)
    extra_absorber = random.randint(1, 2)
    if plan["final_mode"] == "exit":
        extra_absorber = max(extra_absorber, 1)

    if not _add_distractors(
        inp,
        h=h,
        w=w,
        color=mirror_color,
        count=extra_mirror,
        occupied_objects=occupied_objects,
        object_forbidden=object_forbidden,
        route_cells=set(plan["route_cells"]),
    ):
        raise AssertionError("failed to place mirror distractors")

    if not _add_distractors(
        inp,
        h=h,
        w=w,
        color=absorber_color,
        count=extra_absorber,
        occupied_objects=occupied_objects,
        object_forbidden=object_forbidden,
        route_cells=set(plan["route_cells"]),
    ):
        raise AssertionError("failed to place absorber distractors")

    _place_rivets(inp, mirror_color, lamp, count=random.randint(2, 5))

    out = inp.copy()
    for cell in plan["route_cells"]:
        if cell != lamp and int(out[cell]) == 0:
            out[cell] = lamp_color
    return inp, out


def _semantic_checks(inp: np.ndarray, out: np.ndarray, plan: dict) -> bool:
    if np.array_equal(inp, out):
        return False

    try:
        scene = _parse_scene(inp)
    except Exception:
        return False

    if not np.array_equal(_trace_with_roles(scene), out):
        return False

    swapped = _trace_with_roles(
        scene,
        mirror_color=int(scene["absorber_color"]),
        absorber_color=int(scene["mirror_color"]),
    )
    if np.array_equal(swapped, out):
        return False

    all_absorb = _trace_with_roles(
        scene,
        mirror_color=-1,
        absorber_color=int(scene["mirror_color"]),
    )
    if np.array_equal(all_absorb, out):
        return False

    if len(plan["mirrors"]) < 2:
        return False

    orientations = {str(item["orientation"]) for item in plan["mirrors"]}
    if orientations != {"/", "\\"}:
        return False

    if int(np.sum(out == scene["lamp_color"])) <= 3:
        return False

    segment_colors = {
        int(value["color"])
        for value in scene["segment_by_cell"].values()
    }
    if segment_colors != {int(scene["mirror_color"]), int(scene["absorber_color"])}:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    frame_color, lamp_color, color_a, color_b = random.sample(range(1, 10), 4)
    mirror_color, absorber_color = random.choice([(color_a, color_b), (color_b, color_a)])
    side = random.choice(["top", "bottom", "left", "right"])
    lamp = _random_lamp(side, h, w)
    mirror_count = random.randint(2, 4)
    final_mode = random.choice(["absorb", "exit"])

    plan = _build_route_plan(h, w, lamp, mirror_count, final_mode)
    if plan is None:
        return None

    try:
        inp, out = _render_instance(
            h,
            w,
            frame_color,
            mirror_color,
            absorber_color,
            lamp_color,
            lamp,
            plan,
        )
    except AssertionError:
        return None

    if not _semantic_checks(inp, out, plan):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
