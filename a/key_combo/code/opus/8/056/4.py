from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_anchor, bbox_coords, find_components  # noqa: E402


BG = 0
LINE_BLUE = 1
LINE_RED = 2
OBJ = 3
WALL = 5

MAX_RETRIES = 300
GRID_W_RANGE = (18, 25)
PAIR_COUNT_CHOICES = (2, 3)
BAND_HEIGHT_CHOICES = (7, 8)
BAND_GAP = 2
OUTER_MARGIN = 1


def _shape(mask_rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in mask_rows], dtype=bool)


SHAPE_LIBRARY: list[np.ndarray] = [
    _shape(["###"]),
    _shape(["#", "#", "#"]),
    _shape([".#.", "###", ".#."]),
    _shape(["###", ".#.", "..."]),
    _shape(["...", ".#.", "###"]),
    _shape([".#.", "##.", ".#."]),
    _shape([".#.", ".##", ".#."]),
    _shape(["###", ".#.", ".#."]),
]


Coord = tuple[int, int]


def _normalized_shape(coords: np.ndarray) -> tuple[Coord, ...]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]

    points: list[Coord] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    rr, cc = r0, c0
    while True:
        points.append((rr, cc))
        if rr == r1 and cc == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            cc += sc
        if e2 < dc:
            err += dc
            rr += sr
    return points


def _door_cells(grid: np.ndarray) -> set[Coord]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out: set[Coord] = set()
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != BG:
                continue
            horiz = 0 < c < W - 1 and int(g[r, c - 1]) == WALL and int(g[r, c + 1]) == WALL
            vert = 0 < r < H - 1 and int(g[r - 1, c]) == WALL and int(g[r + 1, c]) == WALL
            if horiz or vert:
                out.add((r, c))
    return out


def _pair_objects(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[OBJ],
        return_format="dict",
    )
    assert 4 <= len(comps) <= 6

    by_shape: dict[tuple[Coord, ...], list[dict]] = defaultdict(list)
    for comp in comps:
        by_shape[_normalized_shape(comp["coords"])].append(comp)

    assert 2 <= len(by_shape) <= 3
    assert all(len(group) == 2 for group in by_shape.values())

    pairs = []
    for group in by_shape.values():
        ordered = sorted(group, key=lambda comp: bbox_anchor(comp["bbox"], anchor="center"))
        a, b = ordered
        ca = bbox_anchor(a["bbox"], anchor="center")
        cb = bbox_anchor(b["bbox"], anchor="center")
        pairs.append({"a": a, "b": b, "centers": (ca, cb)})
    pairs.sort(key=lambda item: item["centers"][0])
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shape objects, connect their bbox centers, and toggle the line color at obstacle doors.

    Input structure:
    - Black background contains 4-6 small green 4-connected objects, all in the same color.
    - The green objects partition into 2-3 pairs by exact normalized bbox mask equality.
    - Gray single-pixel horizontal or vertical obstacle segments may contain one-cell black gaps that act as doors.

    Rule:
    - Find each green object, normalize its bbox mask, and pair the two objects with the same shape.
    - For each pair, ordered by the upper-left member's bbox center, draw the center-to-center segment in blue:
      horizontal if centers share a row, vertical if they share a column, otherwise an 8-connected Bresenham line.
    - While stepping from the first center toward the second, every time the path visits a door cell, toggle the
      drawing color between blue and red starting at that door cell. Do not overwrite original green objects, gray
      obstacle cells, or earlier non-background line pixels.

    Key insight:
    - Color alone is useless because every object is green; the pairing is determined by geometric shape matching,
      and only after the correct pair is known can the obstacle-door toggles determine which line segments become red.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    doors = _door_cells(g)

    for pair in _pair_objects(g):
        a, b = pair["centers"]
        color = LINE_BLUE
        for cell in _segment_cells(a, b):
            if cell in doors:
                color = LINE_RED if color == LINE_BLUE else LINE_BLUE
            r, c = cell
            if int(g[r, c]) == BG and int(out[r, c]) == BG:
                out[r, c] = color
    return out


def _shape_cells(mask: np.ndarray, center: Coord) -> set[Coord]:
    h, w = mask.shape
    top = int(center[0]) - h // 2
    left = int(center[1]) - w // 2
    return {
        (top + r, left + c)
        for r, c in np.argwhere(mask)
    }


def _paint_shape(grid: np.ndarray, mask: np.ndarray, center: Coord, color: int) -> None:
    for r, c in _shape_cells(mask, center):
        grid[r, c] = int(color)


def _band_ranges(pair_count: int, band_h: int) -> tuple[int, list[tuple[int, int]]]:
    H = OUTER_MARGIN * 2 + pair_count * band_h + (pair_count - 1) * BAND_GAP
    bands = []
    cursor = OUTER_MARGIN
    for _ in range(pair_count):
        bands.append((cursor, cursor + band_h - 1))
        cursor += band_h + BAND_GAP
    return H, bands


def _sample_pair_kind(toggle_count: int, band_h: int) -> str:
    kinds = ["h", "v", "diag_shallow_pos", "diag_shallow_neg", "diag_steep_pos", "diag_steep_neg"]
    if toggle_count == 2 and band_h == 7:
        kinds.remove("v")
    return random.choice(kinds)


def _sample_endpoints(kind: str, band: tuple[int, int], W: int) -> tuple[Coord, Coord] | None:
    r0, r1 = band
    top = r0 + 1
    bottom = r1 - 1
    if kind == "h":
        row = random.randint(top, bottom)
        c1 = random.randint(1, max(1, W - 13))
        c2 = random.randint(c1 + 8, W - 2)
        return (row, c1), (row, c2)
    if kind == "v":
        col = random.randint(1, W - 2)
        return (top, col), (bottom, col)
    if kind == "diag_shallow_pos":
        dr = random.randint(2, min(3, bottom - top))
        dc = random.randint(max(6, dr + 2), min(10, W - 4))
        row = random.randint(top, bottom - dr)
        col = random.randint(1, W - 2 - dc)
        return (row, col), (row + dr, col + dc)
    if kind == "diag_shallow_neg":
        dr = random.randint(2, min(3, bottom - top))
        dc = random.randint(max(6, dr + 2), min(10, W - 4))
        row = random.randint(top, bottom - dr)
        col = random.randint(dc + 1, W - 2)
        return (row, col), (row + dr, col - dc)
    if kind == "diag_steep_pos":
        dc = random.randint(2, 4)
        col = random.randint(1, W - 2 - dc)
        return (top, col), (bottom, col + dc)
    if kind == "diag_steep_neg":
        dc = random.randint(2, 4)
        col = random.randint(dc + 1, W - 2)
        return (top, col), (bottom, col - dc)
    return None


def _door_orientation(a: Coord, b: Coord) -> str:
    dr = int(b[0]) - int(a[0])
    dc = int(b[1]) - int(a[1])
    if dr == 0:
        return "vertical"
    if dc == 0:
        return "horizontal"
    return "vertical" if abs(dc) >= abs(dr) else "horizontal"


def _segment_wall_cells(
    orientation: str,
    door: Coord,
    band: tuple[int, int],
    W: int,
    extra_gap: int | None,
) -> set[Coord]:
    r, c = door
    if orientation == "vertical":
        gaps = {int(r)}
        if extra_gap is not None:
            gaps.add(int(extra_gap))
        return {(rr, c) for rr in range(band[0], band[1] + 1) if rr not in gaps}
    gaps = {int(c)}
    if extra_gap is not None:
        gaps.add(int(extra_gap))
    return {(r, cc) for cc in range(W) if cc not in gaps}


def _valid_cells(shape_cells: set[Coord], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in shape_cells)


def _dilate_cells(cells: set[Coord], radius: int, H: int, W: int) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _choose_used_doors(
    path: list[Coord],
    object_cells: set[Coord],
    kind: str,
    band: tuple[int, int],
    W: int,
    toggle_count: int,
) -> tuple[str, list[Coord]] | None:
    orientation = _door_orientation(path[0], path[-1])
    start = 2 if len(path) >= 6 else 1
    end = len(path) - 2 if len(path) >= 6 else len(path) - 1
    interior = path[start:end]
    if len(interior) < toggle_count:
        return None

    grouped: dict[int, list[Coord]] = defaultdict(list)
    for cell in interior:
        key = cell[1] if orientation == "vertical" else cell[0]
        grouped[key].append(cell)

    candidates: list[Coord] = []
    for key in sorted(grouped):
        cells = grouped[key]
        if len(cells) != 1:
            continue
        door = cells[0]
        wall_cells = _segment_wall_cells(orientation, door, band, W, None)
        if wall_cells & object_cells:
            continue
        candidates.append(door)

    if kind == "v" and toggle_count == 2 and len(candidates) < 2:
        return None
    if len(candidates) < toggle_count:
        return None

    picks = []
    if toggle_count == 1:
        picks.append(random.choice(candidates))
    else:
        for door in candidates:
            if not picks:
                picks.append(door)
                continue
            prev = picks[-1]
            if abs((door[1] if orientation == "vertical" else door[0]) - (prev[1] if orientation == "vertical" else prev[0])) >= 1:
                picks.append(door)
            if len(picks) == toggle_count:
                break
        if len(picks) != toggle_count:
            return None
    return orientation, picks


def _pick_extra_gap(
    orientation: str,
    door: Coord,
    path_cells: set[Coord],
    band: tuple[int, int],
    W: int,
) -> int | None:
    if random.random() < 0.6:
        return None
    if orientation == "vertical":
        choices = [r for r in range(band[0] + 1, band[1]) if (r, door[1]) not in path_cells and r != door[0]]
    else:
        choices = [c for c in range(1, W - 1) if (door[0], c) not in path_cells and c != door[1]]
    if not choices:
        return None
    return random.choice(choices)


def _build_pair_spec(
    shape: np.ndarray,
    band: tuple[int, int],
    H: int,
    W: int,
    toggle_count: int,
    occupied_clearance: set[Coord],
) -> dict | None:
    for _ in range(120):
        kind = _sample_pair_kind(toggle_count, band[1] - band[0] + 1)
        endpoints = _sample_endpoints(kind, band, W)
        if endpoints is None:
            continue
        a, b = endpoints
        a_cells = _shape_cells(shape, a)
        b_cells = _shape_cells(shape, b)
        object_cells = a_cells | b_cells
        if not _valid_cells(object_cells, H, W):
            continue
        if object_cells & occupied_clearance:
            continue
        path = _segment_cells(a, b)
        used = _choose_used_doors(path, object_cells, kind, band, W, toggle_count)
        if used is None:
            continue
        orientation, door_cells = used
        path_cells = set(path)
        segments = []
        wall_cells: set[Coord] = set()
        ok = True
        for door in door_cells:
            extra_gap = _pick_extra_gap(orientation, door, path_cells, band, W)
            seg_cells = _segment_wall_cells(orientation, door, band, W, extra_gap)
            if seg_cells & object_cells or seg_cells & wall_cells:
                ok = False
                break
            wall_cells |= seg_cells
            segments.append({"door": door, "orientation": orientation, "extra_gap": extra_gap})
        if not ok:
            continue
        if any(cell in wall_cells for cell in path_cells if cell not in set(door_cells)):
            continue
        return {
            "shape": shape,
            "centers": (a, b),
            "path": path,
            "doors": door_cells,
            "segments": segments,
            "object_cells": object_cells,
            "wall_cells": wall_cells,
        }
    return None


def _render_input(specs: list[dict], H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for spec in specs:
        _paint_shape(grid, spec["shape"], spec["centers"][0], OBJ)
        _paint_shape(grid, spec["shape"], spec["centers"][1], OBJ)
    for spec in specs:
        for r, c in spec["wall_cells"]:
            grid[r, c] = WALL
    return grid


def _expected_output_from_specs(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    all_doors = {tuple(door) for spec in specs for door in spec["doors"]}
    ordered = sorted(specs, key=lambda spec: min(spec["centers"]))
    for spec in ordered:
        color = LINE_BLUE
        for cell in spec["path"]:
            if cell in all_doors:
                color = LINE_RED if color == LINE_BLUE else LINE_BLUE
            r, c = cell
            if int(inp[r, c]) == BG and int(out[r, c]) == BG:
                out[r, c] = color
    return out


def _validate_generated(inp: np.ndarray, specs: list[dict]) -> bool:
    H, W = inp.shape
    if not (inp.ndim == 2 and H <= 30 and W <= 30):
        return False
    if set(np.unique(inp)) - {BG, OBJ, WALL}:
        return False

    pairs = _pair_objects(inp)
    if len(pairs) != len(specs):
        return False

    actual_doors = _door_cells(inp)
    planned_gap_cells = set()
    for spec in specs:
        for seg in spec["segments"]:
            door = tuple(seg["door"])
            planned_gap_cells.add(door)
            if seg["extra_gap"] is not None:
                if seg["orientation"] == "vertical":
                    planned_gap_cells.add((int(seg["extra_gap"]), int(door[1])))
                else:
                    planned_gap_cells.add((int(door[0]), int(seg["extra_gap"])))
    if actual_doors != planned_gap_cells:
        return False

    seen_doors: set[Coord] = set()
    paintable_paths: set[Coord] = set()
    for spec in specs:
        path = spec["path"]
        used_doors = set(map(tuple, spec["doors"]))
        if len(used_doors) not in (1, 2):
            return False
        if seen_doors & used_doors:
            return False
        seen_doors |= used_doors
        if actual_doors & (set(path) - used_doors):
            return False
        wall_cells = {
            (r, c)
            for r, c in spec["wall_cells"]
            if (r, c) not in actual_doors
        }
        if wall_cells & set(path):
            return False
        draw_cells = {cell for cell in path if int(inp[cell]) == BG}
        if paintable_paths & draw_cells:
            return False
        paintable_paths |= draw_cells

    expected = _expected_output_from_specs(inp, specs)
    got = main(inp)
    return np.array_equal(got, expected)


def _toggle_pattern(pair_count: int) -> list[int]:
    toggles = [1] * pair_count
    indices = list(range(pair_count))
    random.shuffle(indices)
    if pair_count == 3 or random.random() < 0.6:
        toggles[indices[0]] = 2
    return toggles


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    band_h = random.choice(BAND_HEIGHT_CHOICES)
    W = random.randint(*GRID_W_RANGE)
    H, bands = _band_ranges(pair_count, band_h)

    shapes = random.sample(SHAPE_LIBRARY, pair_count)
    toggles = _toggle_pattern(pair_count)

    specs = []
    occupied_clearance: set[Coord] = set()
    for band, shape, toggle_count in zip(bands, shapes, toggles):
        spec = _build_pair_spec(shape, band, H, W, toggle_count, occupied_clearance)
        if spec is None:
            return None
        specs.append(spec)
        occupied_clearance |= _dilate_cells(spec["object_cells"], radius=2, H=H, W=W)

    inp = _render_input(specs, H, W)
    if not _validate_generated(inp, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 4")
