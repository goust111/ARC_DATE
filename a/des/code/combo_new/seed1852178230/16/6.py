from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ("up", "down", "left", "right")

MISSING_TO_CELLS = {
    "TL": [(0, 1), (1, 0), (1, 1)],
    "TR": [(0, 0), (1, 0), (1, 1)],
    "BL": [(0, 0), (0, 1), (1, 1)],
    "BR": [(0, 0), (0, 1), (1, 0)],
}

DIR_OPTIONS = {
    "TL": ("up", "left"),
    "TR": ("up", "right"),
    "BL": ("down", "left"),
    "BR": ("down", "right"),
}

TIE_BREAK = {
    "TL": "up",
    "TR": "right",
    "BR": "down",
    "BL": "left",
}

MIRROR_LR = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL"}
MIRROR_UD = {"TL": "BL", "TR": "BR", "BL": "TL", "BR": "TR"}

MISSING_BY_DIR = {
    "up": ("TL", "TR"),
    "down": ("BL", "BR"),
    "left": ("TL", "BL"),
    "right": ("TR", "BR"),
}

GENERATE_RETRIES = 500
OBJECT_PLACE_RETRIES = 500
MARKER_PLACE_RETRIES = 500

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (2, 5)
MARKER_COUNT_RANGE = (1, 4)
CLIP_PROB = 0.4


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}, int(r0), int(c0)


def _missing_corner(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) != 3:
        return None
    local, _r0, _c0 = _normalize(coords)
    if max(r for r, _ in local) != 1 or max(c for _, c in local) != 1:
        return None
    missing = {(0, 0), (0, 1), (1, 0), (1, 1)} - local
    if len(missing) != 1:
        return None
    corner = next(iter(missing))
    mapping = {(0, 0): "TL", (0, 1): "TR", (1, 0): "BL", (1, 1): "BR"}
    return mapping[corner]


def _direction_spaces(r0: int, c0: int, h: int, w: int) -> dict[str, int]:
    return {
        "up": int(r0),
        "down": int(h - (r0 + 2)),
        "left": int(c0),
        "right": int(w - (c0 + 2)),
    }


def _resolve_direction(missing: str, r0: int, c0: int, h: int, w: int) -> str:
    spaces = _direction_spaces(r0, c0, h, w)
    vertical_or_down, horizontal = DIR_OPTIONS[missing]
    a = int(spaces[vertical_or_down])
    b = int(spaces[horizontal])
    if a > b:
        return vertical_or_down
    if b > a:
        return horizontal
    return TIE_BREAK[missing]


def _shift_bbox(r0: int, c0: int, direction: str, steps: int, h: int, w: int) -> tuple[int, int]:
    spaces = _direction_spaces(r0, c0, h, w)
    shift = min(int(steps), int(spaces[direction]))
    if direction == "up":
        return int(r0 - shift), int(c0)
    if direction == "down":
        return int(r0 + shift), int(c0)
    if direction == "left":
        return int(r0), int(c0 - shift)
    return int(r0), int(c0 + shift)


def _mirrored_missing(missing: str, direction: str) -> str:
    if direction in ("left", "right"):
        return MIRROR_LR[missing]
    return MIRROR_UD[missing]


def _l_cells(r0: int, c0: int, missing: str) -> set[tuple[int, int]]:
    return {(int(r0 + dr), int(c0 + dc)) for dr, dc in MISSING_TO_CELLS[missing]}


def _paint_l(grid: np.ndarray, r0: int, c0: int, missing: str, color: int) -> None:
    for dr, dc in MISSING_TO_CELLS[missing]:
        grid[r0 + dr, c0 + dc] = int(color)


def _with_margin(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _sweep_bbox_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
    dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
    cr, cc = int(r0), int(c0)
    while True:
        for rr in range(cr, cr + 2):
            for cc2 in range(cc, cc + 2):
                out.add((int(rr), int(cc2)))
        if cr == r1 and cc == c1:
            break
        cr += dr
        cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored 3-cell L block by its same-color marker count, then mirror the L across the motion axis.

    Input structure:
    - The background is the dominant color.
    - Several distinct colors each form one 3-pixel L block whose 2x2 bbox is missing exactly one corner.
    - The same colors also appear as isolated singleton marker pixels elsewhere in the grid, and those markers are not attached to the matching L block.

    Rule:
    - For each color, find the unique 3-pixel L component and count the same-color singleton markers; that count N is the slide distance.
    - Read the L's open side from the missing-corner side of its 2x2 box, resolve its cardinal slide direction, and translate the whole L by N cells along that direction.
    - If the translated 2x2 bbox would leave the grid, clamp it to the nearest in-bounds position on that edge.
    - After translating, mirror the L across the axis perpendicular to the motion direction, and erase all marker pixels from the output.

    Key insight:
    - The single-pixel same-color markers are controls, not objects to preserve: their multiplicity encodes distance, while the L geometry determines the slide direction and the mirrored orientation of the final piece.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = np.full((h, w), int(bg), dtype=int)

    for color0 in np.unique(g):
        color = int(color0)
        if color == int(bg):
            continue
        comps = _color_components(g, color)
        l_components = [comp for comp in comps if _missing_corner(comp) is not None]
        if not l_components:
            continue
        assert len(l_components) == 1, f"expected one L component for color {color}"
        l_comp = l_components[0]
        assert all(len(comp) == 1 or comp is l_comp for comp in comps), f"unexpected same-color structure for {color}"

        marker_count = sum(1 for comp in comps if len(comp) == 1)
        assert marker_count >= 1, f"expected at least one marker for color {color}"

        missing = _missing_corner(l_comp)
        assert missing is not None
        _local, r0, c0 = _normalize(l_comp)
        direction = _resolve_direction(missing, r0, c0, h, w)
        nr0, nc0 = _shift_bbox(r0, c0, direction, marker_count, h, w)
        out_missing = _mirrored_missing(missing, direction)
        _paint_l(out, nr0, nc0, out_missing, color)

    return out


def _sample_object_directions(count: int) -> list[str]:
    dirs = list(DIR_NAMES)
    random.shuffle(dirs)
    return dirs[:count] if count <= 4 else dirs + random.choices(DIR_NAMES, k=count - 4)


def _place_object(
    h: int,
    w: int,
    direction: str,
    prefer_clip: bool,
    input_blocked: set[tuple[int, int]],
    input_occupied: set[tuple[int, int]],
    corridor_blocked: set[tuple[int, int]],
    output_blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(OBJECT_PLACE_RETRIES):
        missing = random.choice(MISSING_BY_DIR[direction])
        r0 = random.randint(0, h - 2)
        c0 = random.randint(0, w - 2)

        spaces = _direction_spaces(r0, c0, h, w)
        dir_a, dir_b = DIR_OPTIONS[missing]
        if int(spaces[dir_a]) == int(spaces[dir_b]):
            continue
        if _resolve_direction(missing, r0, c0, h, w) != direction:
            continue

        free = int(spaces[direction])
        if free < 1:
            continue

        if prefer_clip:
            marker_options = [n for n in range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1) if n > free]
        else:
            marker_options = [n for n in range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1) if n <= free]
        if not marker_options:
            continue

        marker_count = random.choice(marker_options)
        nr0, nc0 = _shift_bbox(r0, c0, direction, marker_count, h, w)

        source_cells = _l_cells(r0, c0, missing)
        dest_missing = _mirrored_missing(missing, direction)
        dest_cells = _l_cells(nr0, nc0, dest_missing)
        sweep_cells = _sweep_bbox_cells(r0, c0, nr0, nc0)

        if source_cells & input_blocked:
            continue
        if source_cells & corridor_blocked:
            continue
        if sweep_cells & input_occupied:
            continue
        if dest_cells & output_blocked:
            continue

        return {
            "source_r0": int(r0),
            "source_c0": int(c0),
            "dest_r0": int(nr0),
            "dest_c0": int(nc0),
            "source_missing": missing,
            "dest_missing": dest_missing,
            "direction": direction,
            "marker_count": int(marker_count),
            "source_cells": source_cells,
            "dest_cells": dest_cells,
            "sweep_cells": sweep_cells,
        }
    return None


def _place_markers(
    h: int,
    w: int,
    objects: list[dict[str, object]],
    blocked: set[tuple[int, int]],
) -> bool:
    for obj in objects:
        markers: list[tuple[int, int]] = []
        count = int(obj["marker_count"])
        for _ in range(count):
            placed = False
            for _attempt in range(MARKER_PLACE_RETRIES):
                r = random.randint(0, h - 1)
                c = random.randint(0, w - 1)
                if (r, c) in blocked:
                    continue
                markers.append((int(r), int(c)))
                blocked |= _with_margin({(int(r), int(c))}, h, w)
                placed = True
                break
            if not placed:
                return False
        obj["markers"] = markers
    return True


def _render_input(h: int, w: int, bg: int, objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for obj in objects:
        color = int(obj["color"])
        _paint_l(grid, int(obj["source_r0"]), int(obj["source_c0"]), str(obj["source_missing"]), color)
        for r, c in obj["markers"]:
            grid[int(r), int(c)] = color
    return grid


def _render_output(h: int, w: int, bg: int, objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), int(bg), dtype=int)
    for obj in objects:
        _paint_l(grid, int(obj["dest_r0"]), int(obj["dest_c0"]), str(obj["dest_missing"]), int(obj["color"]))
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    bg = random.randint(0, 9)

    max_objects = min(OBJECT_COUNT_RANGE[1], 9)
    object_count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    colors = random.sample([c for c in range(10) if c != bg], object_count)
    directions = _sample_object_directions(object_count)

    input_occupied: set[tuple[int, int]] = set()
    input_blocked: set[tuple[int, int]] = set()
    corridor_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()
    objects: list[dict[str, object]] = []

    for color, direction in zip(colors, directions):
        prefer_clip = random.random() < CLIP_PROB
        placed = _place_object(
            h,
            w,
            direction,
            prefer_clip,
            input_blocked,
            input_occupied,
            corridor_blocked,
            output_blocked,
        )
        if placed is None:
            return None
        placed["color"] = int(color)
        objects.append(placed)

        source_cells = set(placed["source_cells"])
        dest_cells = set(placed["dest_cells"])
        sweep_cells = set(placed["sweep_cells"])

        input_occupied |= source_cells
        input_blocked |= _with_margin(source_cells, h, w)
        corridor_blocked |= sweep_cells
        output_blocked |= _with_margin(dest_cells, h, w)

    marker_blocked = set(input_blocked) | set(corridor_blocked)
    if not _place_markers(h, w, objects, marker_blocked):
        return None

    inp = _render_input(h, w, bg, objects)
    expected = _render_output(h, w, bg, objects)

    if _bg_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
