from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

RIGHT_MASK = np.array(
    [
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
    ],
    dtype=int,
)

ARROWS = {
    "right": {
        "mask": RIGHT_MASK,
        "delta": (0, 1),
        "tail": (1, 0),
        "axis": "row",
        "back_side": "left",
        "front_side": "right",
    },
    "up": {
        "mask": np.rot90(RIGHT_MASK, 1),
        "delta": (-1, 0),
        "tail": (2, 1),
        "axis": "col",
        "back_side": "bottom",
        "front_side": "top",
    },
    "left": {
        "mask": np.rot90(RIGHT_MASK, 2),
        "delta": (0, -1),
        "tail": (1, 2),
        "axis": "row",
        "back_side": "right",
        "front_side": "left",
    },
    "down": {
        "mask": np.rot90(RIGHT_MASK, 3),
        "delta": (1, 0),
        "tail": (0, 1),
        "axis": "col",
        "back_side": "top",
        "front_side": "bottom",
    },
}

MIN_SIDE_BY_COUNT = {2: 9, 3: 11, 4: 13}
MAX_SIDE = 20


def _components4(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _classify_arrow(component: dict) -> dict | None:
    coords = component["coords"]
    if len(coords) != 5:
        return None

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return None

    patch = np.zeros((3, 3), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = 1

    for direction, info in ARROWS.items():
        if np.array_equal(patch, info["mask"]):
            tail_r = r0 + int(info["tail"][0])
            tail_c = c0 + int(info["tail"][1])
            return {
                "color": int(component["color"]),
                "coords": {(int(r), int(c)) for r, c in coords},
                "direction": direction,
                "tail_center": (tail_r, tail_c),
                "delta": tuple(info["delta"]),
                "axis": info["axis"],
            }
    return None


def _boundary_count(static_grid: np.ndarray, color: int, axis: str, tail_center: tuple[int, int]) -> int:
    h, w = static_grid.shape
    r, c = tail_center
    if axis == "row":
        positions = {(r, 0), (r, w - 1)}
    else:
        positions = {(0, c), (h - 1, c)}
    return sum(1 for rr, cc in positions if int(static_grid[rr, cc]) == color)


def _translated_cells(coords: set[tuple[int, int]], dr: int, dc: int, step: int, h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        nr = r + dr * step
        nc = c + dc * step
        if 0 <= nr < h and 0 <= nc < w:
            out.add((nr, nc))
    return out


def _compute_motion(arrow: dict, static_grid: np.ndarray) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = static_grid.shape
    color = int(arrow["color"])
    dr, dc = arrow["delta"]
    distance = _boundary_count(static_grid, color, arrow["axis"], arrow["tail_center"])

    final_cells = _translated_cells(arrow["coords"], dr, dc, distance, h, w)
    trail_cells: set[tuple[int, int]] = set()
    for step in range(distance):
        trail_cells |= _translated_cells(arrow["coords"], dr, dc, step, h, w)
    trail_cells -= final_cells
    return distance, final_cells, trail_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each chevron arrow by the number of same-color boundary markers aligned with its tail row or tail column.

    Input structure:
    - The background is black.
    - Foreground objects include several monochrome 3x3 chevron arrows pointing up, down, left, or right.
    - Additional same-color single-pixel markers may appear on the outer border, separated from the arrows.

    Rule:
    - Detect each 3x3 chevron and infer its pointing direction from the shape.
    - For a horizontal arrow, count same-color border markers on the left/right edge cells of the tail-center row; for a vertical arrow, count same-color border markers on the top/bottom edge cells of the tail-center column.
    - Translate the whole arrow forward by that count, clip any moved pixels that leave the grid, and color every swept-but-not-final cell grey while leaving border markers unchanged.

    Key insight:
    The border pixels are not generic decoration: the count of same-color markers on the tail-aligned border line encodes how far that arrow slides, while the path itself is repainted in a uniform grey trail.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    arrows = []
    for comp in _components4(g, bg=BG):
        arrow = _classify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)

    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = BG

    static_grid = out.copy()
    static_mask = static_grid != BG

    final_paints: dict[tuple[int, int], int] = {}
    all_final_cells: set[tuple[int, int]] = set()
    all_trail_cells: set[tuple[int, int]] = set()

    for arrow in arrows:
        _distance, final_cells, trail_cells = _compute_motion(arrow, static_grid)
        color = int(arrow["color"])
        for cell in final_cells:
            final_paints[cell] = color
        all_final_cells |= final_cells
        all_trail_cells |= trail_cells

    for r, c in sorted(all_trail_cells - all_final_cells):
        if not static_mask[r, c]:
            out[r, c] = GREY

    for (r, c), color in final_paints.items():
        if not static_mask[r, c]:
            out[r, c] = color

    return out


def _axis_sides(direction: str) -> tuple[str, str]:
    info = ARROWS[direction]
    if info["axis"] == "row":
        return ("left", "right")
    return ("top", "bottom")


def _sample_specs() -> list[dict]:
    n_arrows = random.choices([2, 3, 4], weights=[3, 4, 3])[0]
    directions = random.sample(list(ARROWS.keys()), n_arrows)
    colors = random.sample(ARROW_COLORS, n_arrows)

    specs = []
    special_mode = random.choice(["stationary", "distance2", "clip"])
    special_idx = random.randrange(n_arrows)

    for idx, (direction, color) in enumerate(zip(directions, colors)):
        clip = False
        if idx == special_idx and special_mode == "clip":
            distance = 1
            clip = True
        elif idx == special_idx and special_mode == "distance2":
            distance = 2
        elif idx == special_idx and special_mode == "stationary":
            distance = 0
        else:
            distance = random.choice([0, 1, 2])
            clip = distance == 1 and random.random() < 0.2

        if distance == 0:
            marker_sides: tuple[str, ...] = ()
            clip = False
        elif distance == 2:
            marker_sides = _axis_sides(direction)
            clip = False
        else:
            if clip:
                marker_sides = (ARROWS[direction]["back_side"],)
            else:
                marker_sides = (random.choice(_axis_sides(direction)),)

        specs.append(
            {
                "direction": direction,
                "color": color,
                "distance": distance,
                "marker_sides": marker_sides,
                "clip": clip,
            }
        )

    if all(spec["distance"] == 0 for spec in specs):
        idx = random.randrange(len(specs))
        direction = specs[idx]["direction"]
        specs[idx]["distance"] = 1
        specs[idx]["marker_sides"] = (random.choice(_axis_sides(direction)),)

    return specs


def _scene_shape(n_arrows: int) -> tuple[int, int]:
    min_side = MIN_SIDE_BY_COUNT[n_arrows]
    return random.randint(min_side, MAX_SIDE), random.randint(min_side, MAX_SIDE)


def _candidate_cells(spec: dict, top: int, left: int, h: int, w: int) -> dict | None:
    info = ARROWS[spec["direction"]]
    coords = set()
    for rr, cc in np.argwhere(info["mask"]):
        coords.add((top + int(rr), left + int(cc)))

    tail_r = top + int(info["tail"][0])
    tail_c = left + int(info["tail"][1])

    markers = set()
    for side in spec["marker_sides"]:
        if side == "left":
            markers.add((tail_r, 0))
        elif side == "right":
            markers.add((tail_r, w - 1))
        elif side == "top":
            markers.add((0, tail_c))
        else:
            markers.add((h - 1, tail_c))

    for mr, mc in markers:
        for ar, ac in coords:
            if max(abs(mr - ar), abs(mc - ac)) <= 1:
                return None

    dr, dc = info["delta"]
    final_cells = _translated_cells(coords, dr, dc, spec["distance"], h, w)
    trail_cells: set[tuple[int, int]] = set()
    for step in range(spec["distance"]):
        trail_cells |= _translated_cells(coords, dr, dc, step, h, w)
    trail_cells -= final_cells

    if (markers & final_cells) or (markers & trail_cells):
        return None

    return {
        "direction": spec["direction"],
        "color": spec["color"],
        "distance": spec["distance"],
        "marker_sides": spec["marker_sides"],
        "arrow_cells": coords,
        "marker_cells": markers,
        "final_cells": final_cells,
        "trail_cells": trail_cells,
    }


def _origin_ok(spec: dict, top: int, left: int, h: int, w: int) -> bool:
    if not (0 <= top <= h - 3 and 0 <= left <= w - 3):
        return False

    if spec["clip"]:
        front = ARROWS[spec["direction"]]["front_side"]
        if front == "right" and left != w - 3:
            return False
        if front == "left" and left != 0:
            return False
        if front == "top" and top != 0:
            return False
        if front == "bottom" and top != h - 3:
            return False

    if "left" in spec["marker_sides"] and left < 2:
        return False
    if "right" in spec["marker_sides"] and left > w - 5:
        return False
    if "top" in spec["marker_sides"] and top < 2:
        return False
    if "bottom" in spec["marker_sides"] and top > h - 5:
        return False

    return True


def _placeable(
    candidate: dict,
    input_blocked: np.ndarray,
    output_static: set[tuple[int, int]],
    output_dynamic: set[tuple[int, int]],
) -> bool:
    input_cells = candidate["arrow_cells"] | candidate["marker_cells"]
    if any(input_blocked[r, c] for r, c in input_cells):
        return False

    dynamic_cells = candidate["final_cells"] | candidate["trail_cells"]
    if dynamic_cells & output_static:
        return False
    if dynamic_cells & output_dynamic:
        return False
    if candidate["marker_cells"] & output_dynamic:
        return False
    return True


def _mark_blocked(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                mask[nr, nc] = True


def _render_scene(shape: tuple[int, int], placed: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    final_paints: dict[tuple[int, int], int] = {}
    all_final = set()
    all_trail = set()

    for obj in placed:
        color = int(obj["color"])
        for r, c in obj["arrow_cells"] | obj["marker_cells"]:
            inp[r, c] = color
        for r, c in obj["marker_cells"]:
            out[r, c] = color
        for cell in obj["final_cells"]:
            final_paints[cell] = color
        all_final |= obj["final_cells"]
        all_trail |= obj["trail_cells"]

    for r, c in sorted(all_trail - all_final):
        if out[r, c] == BG:
            out[r, c] = GREY
    for (r, c), color in final_paints.items():
        if out[r, c] == BG:
            out[r, c] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    h, w = _scene_shape(len(specs))

    placed: list[dict] = []
    input_blocked = np.zeros((h, w), dtype=bool)
    output_static: set[tuple[int, int]] = set()
    output_dynamic: set[tuple[int, int]] = set()

    shuffled_specs = specs[:]
    random.shuffle(shuffled_specs)
    shuffled_specs.sort(key=lambda spec: (spec["clip"], spec["distance"]), reverse=True)

    for spec in shuffled_specs:
        origins = [(top, left) for top in range(h - 2) for left in range(w - 2) if _origin_ok(spec, top, left, h, w)]
        random.shuffle(origins)

        chosen = None
        for top, left in origins:
            candidate = _candidate_cells(spec, top, left, h, w)
            if candidate is None:
                continue
            if not _placeable(candidate, input_blocked, output_static, output_dynamic):
                continue
            chosen = candidate
            break

        if chosen is None:
            return None

        placed.append(chosen)
        input_cells = chosen["arrow_cells"] | chosen["marker_cells"]
        _mark_blocked(input_blocked, input_cells)
        output_static |= chosen["marker_cells"]
        output_dynamic |= chosen["final_cells"] | chosen["trail_cells"]

    inp, expected = _render_scene((h, w), placed)

    if not np.any(inp == BG):
        return None
    if not np.any(expected == GREY):
        # Keep the stationary branch possible, but require at least one visible trail overall.
        return None
    if np.max(inp.shape) > 30 or np.max(expected.shape) > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chevron-slide input")
