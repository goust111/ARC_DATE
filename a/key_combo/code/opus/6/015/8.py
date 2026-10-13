from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds  # noqa: E402


BG = int(Color.BLACK)
RED = int(Color.RED)
GREY = int(Color.GREY)
GREEN = int(Color.GREEN)
MAROON = int(Color.MAROON)
YELLOW = int(Color.YELLOW)
TEAL = int(Color.TEAL)

SOURCE_COLORS = (RED, GREY)
TARGET_COLORS = (YELLOW, TEAL)
RAY_COLOR_BY_SOURCE = {RED: GREEN, GREY: MAROON}
MISSINGS = ((0, 0), (0, 1), (1, 0), (1, 1))
DIR_TO_MISSING = {(-1, -1): (0, 0), (-1, 1): (0, 1), (1, -1): (1, 0), (1, 1): (1, 1)}
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Triomino:
    r0: int
    c0: int
    missing: tuple[int, int]
    color: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r0 + 1, self.c0 + 1)

    @property
    def direction(self) -> tuple[int, int]:
        mr, mc = self.missing
        return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)

    @property
    def cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.r0 + rr, self.c0 + cc)
            for rr in (0, 1)
            for cc in (0, 1)
            if (rr, cc) != self.missing
        )

    @property
    def emitter(self) -> tuple[int, int]:
        mr, mc = self.missing
        return (self.r0 + 1 - mr, self.c0 + 1 - mc)

    def shifted_cells(self) -> tuple[tuple[int, int], ...]:
        dr, dc = self.direction
        return tuple((r + dr, c + dc) for r, c in self.cells)


def _missing_from_component(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    assert r1 == r0 + 1 and c1 == c0 + 1, "source triomino must occupy a 2x2 bbox"
    rel = {(int(r - r0), int(c - c0)) for r, c in comp["coords"]}
    missing = set(MISSINGS) - rel
    assert len(missing) == 1, "expected exactly one missing corner"
    return next(iter(missing))


def _detect_objects(grid: np.ndarray) -> list[Triomino]:
    comps = find_components(
        grid,
        bg=BG,
        colors=SOURCE_COLORS,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected source triominoes"
    objects: list[Triomino] = []
    for comp in comps:
        assert int(comp["area"]) == 3, "each source object must have area 3"
        color = int(comp["color"])
        missing = _missing_from_component(comp)
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        objects.append(Triomino(r0, c0, missing, color))
    objects.sort(key=lambda obj: (obj.r0, obj.c0, obj.color, obj.missing))
    return objects


def _ray_result(
    grid: np.ndarray,
    solid_cells: set[tuple[int, int]],
    obj: Triomino,
    *,
    paint: bool,
) -> dict:
    h, w = grid.shape
    ray_color = RAY_COLOR_BY_SOURCE[obj.color]
    dr, dc = obj.direction
    rr, cc = obj.emitter
    painted: list[tuple[int, int]] = []

    while True:
        rr += dr
        cc += dc
        assert in_bounds((h, w), rr, cc), "emitter must have an in-bounds first step"
        val = int(grid[rr, cc])
        on_edge = rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1
        if val == BG:
            if paint:
                grid[rr, cc] = ray_color
            painted.append((rr, cc))
            if on_edge:
                return {
                    "success": True,
                    "stop_kind": "edge",
                    "stop_cell": (rr, cc),
                    "ray_cells": painted,
                }
            continue
        if val in TARGET_COLORS and on_edge:
            if paint:
                grid[rr, cc] = ray_color
            painted.append((rr, cc))
            return {
                "success": True,
                "stop_kind": "target",
                "stop_cell": (rr, cc),
                "ray_cells": painted,
            }
        return {
            "success": False,
            "stop_kind": "triomino" if (rr, cc) in solid_cells else "mark",
            "stop_cell": (rr, cc),
            "ray_cells": painted,
        }


def _simulate(
    grid: np.ndarray,
    *,
    dynamic_paths: bool = True,
    move_on_success: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    original = np.asarray(grid, dtype=int)
    assert original.ndim == 2, "expected a 2D grid"
    assert np.all(np.isin(original, [BG, RED, GREY, YELLOW, TEAL])), "unexpected input colors"
    objects = _detect_objects(original)
    out = original.copy()
    path_grid = out if dynamic_paths else original.copy()
    solid_cells = {cell for obj in objects for cell in obj.cells}
    metas: list[dict] = []

    for obj in objects:
        ray = _ray_result(path_grid, solid_cells, obj, paint=True)
        moved_cells: list[tuple[int, int]] = []
        if ray["success"] and move_on_success:
            dr, dc = obj.direction
            new_cells = obj.shifted_cells()
            for nr, nc in new_cells:
                assert in_bounds(out.shape, nr, nc), "successful move must remain in bounds"

            for r, c in obj.cells:
                solid_cells.remove((r, c))
                out[r, c] = BG
                if dynamic_paths:
                    path_grid[r, c] = BG

            dst_color = RAY_COLOR_BY_SOURCE[obj.color]
            for nr, nc in new_cells:
                assert (nr, nc) not in solid_cells, "moved copy collided with another triomino"
                cur = int(out[nr, nc])
                assert cur not in SOURCE_COLORS, "moved copy collided with an unprocessed source"
                if dynamic_paths:
                    path_cur = int(path_grid[nr, nc])
                    assert path_cur not in SOURCE_COLORS, "moved copy collided in path grid"
                out[nr, nc] = dst_color
                if dynamic_paths:
                    path_grid[nr, nc] = dst_color
                solid_cells.add((nr, nc))
                moved_cells.append((nr, nc))

        metas.append(
            {
                "bbox": obj.bbox,
                "r0": obj.r0,
                "c0": obj.c0,
                "color": obj.color,
                "missing": obj.missing,
                "direction": obj.direction,
                "success": ray["success"],
                "stop_kind": ray["stop_kind"],
                "stop_cell": ray["stop_cell"],
                "ray_cells": ray["ray_cells"],
                "moved_cells": moved_cells,
            }
        )
    return out, metas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire diagonal rays from L triomino gaps and move only the emitters whose rays reach the boundary.

    Input structure:
    - The background is black.
    - The foreground contains 5-8 red or grey L triominoes, each made of three
      cells inside a 2x2 box with one missing corner.
    - Yellow or cyan short segments may lie on the outer border as boundary targets.

    Rule:
    - For each input triomino, read the missing 2x2 corner to get one diagonal
      direction. The occupied corner opposite that gap acts as the emitter cell.
    - Cast a diagonal ray through background cells in that direction: red sources
      paint green, grey sources paint maroon, and the ray stops at the first
      non-background cell or when it reaches the border.
    - Hitting the border or a border target counts as success; then clear the
      original triomino and redraw it shifted one diagonal step in the same
      direction, recolored to the same ray color. If the first blocker is another
      triomino or an earlier mark, the triomino stays in place.
    - Process triominoes in ascending order of their 2x2 bounding-box top-left
      corner, so earlier rays and moves can block later ones.

    Key insight:
    - The missing corner couples three things at once: the ray direction, the
      possible diagonal move, and the exact success condition for that move.
    """
    out, _meta = _simulate(np.asarray(grid, dtype=int), dynamic_paths=True, move_on_success=True)
    return out


def _all_geom_candidates(h: int, w: int) -> list[tuple[int, int, tuple[int, int]]]:
    cands = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            for missing in MISSINGS:
                dr = -1 if missing[0] == 0 else 1
                dc = -1 if missing[1] == 0 else 1
                src = Triomino(r0, c0, missing, RED)
                if all(in_bounds((h, w), r + dr, c + dc) for r, c in src.cells):
                    cands.append((r0, c0, missing))
    random.shuffle(cands)
    return cands


def _source_compatible(new_obj: Triomino, objects: list[Triomino]) -> bool:
    cells = set(new_obj.cells)
    for old in objects:
        old_cells = set(old.cells)
        if cells & old_cells:
            return False
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                if (r + dr, c + dc) in old_cells:
                    return False
    return True


def _render_input(h: int, w: int, objects: list[Triomino], targets: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in obj.cells:
            assert grid[r, c] == BG
            grid[r, c] = obj.color
    for r, c, color in targets:
        assert r in {0, h - 1} or c in {0, w - 1}
        assert grid[r, c] == BG
        grid[r, c] = color
    return grid


def _pair_objects(
    anchor: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int, tuple[int, int]], tuple[int, int, tuple[int, int]]]:
    dr, dc = direction
    missing_a = DIR_TO_MISSING[(dr, dc)]
    missing_b = DIR_TO_MISSING[(-dr, -dc)]
    r0, c0 = anchor
    return (r0, c0, missing_a), (r0 + dr, c0 + dc, missing_b)


def _try_place_block_pair(
    h: int,
    w: int,
    objects: list[Triomino],
) -> tuple[Triomino, Triomino] | None:
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    random.shuffle(directions)
    for dr, dc in directions:
        row_range = range(1, h - 2) if dr == -1 else range(0, h - 3)
        col_range = range(1, w - 2) if dc == -1 else range(0, w - 3)
        anchors = [(r, c) for r in row_range for c in col_range]
        random.shuffle(anchors)
        for anchor in anchors:
            ga, gb = _pair_objects(anchor, (dr, dc))
            a = Triomino(ga[0], ga[1], ga[2], RED)
            b = Triomino(gb[0], gb[1], gb[2], RED)
            if not _source_compatible(a, objects):
                continue
            if not _source_compatible(b, objects + [a]):
                continue
            return a, b
    return None


def _meta_by_bbox(meta: list[dict], obj: Triomino) -> dict:
    for item in meta:
        if item["bbox"] == obj.bbox:
            return item
    raise AssertionError(f"missing metadata for {obj.bbox}")


def _candidate_success(
    h: int,
    w: int,
    objects: list[Triomino],
    candidates: list[tuple[int, int, tuple[int, int]]],
) -> Triomino | None:
    for r0, c0, missing in candidates:
        cand = Triomino(r0, c0, missing, RED)
        if not _source_compatible(cand, objects):
            continue
        grid = _render_input(h, w, objects + [cand], [])
        try:
            _out, meta = _simulate(grid)
        except AssertionError:
            continue
        if _meta_by_bbox(meta, cand)["success"]:
            return cand
    return None


def _candidate_order_dependent(
    h: int,
    w: int,
    objects: list[Triomino],
    candidates: list[tuple[int, int, tuple[int, int]]],
) -> Triomino | None:
    for r0, c0, missing in candidates:
        cand = Triomino(r0, c0, missing, RED)
        if not _source_compatible(cand, objects):
            continue
        grid = _render_input(h, w, objects + [cand], [])
        try:
            _dyn_out, dyn_meta = _simulate(grid, dynamic_paths=True, move_on_success=True)
            _fro_out, fro_meta = _simulate(grid, dynamic_paths=False, move_on_success=True)
        except AssertionError:
            continue
        dyn_item = _meta_by_bbox(dyn_meta, cand)
        fro_item = _meta_by_bbox(fro_meta, cand)
        if fro_item["success"] and not dyn_item["success"]:
            return cand
    return None


def _assign_colors(objects: list[Triomino], meta: list[dict]) -> list[Triomino]:
    successes = [m["bbox"] for m in meta if m["success"]]
    blocks = [m["bbox"] for m in meta if not m["success"]]
    if len(successes) < 2:
        raise ValueError("need at least two successful objects before color assignment")
    color_by_bbox = {bbox: RED for bbox in successes + blocks}
    color_by_bbox[successes[0]] = RED
    color_by_bbox[successes[1]] = GREY
    if blocks:
        color_by_bbox[blocks[0]] = GREY
    if len(blocks) > 1:
        color_by_bbox[blocks[1]] = RED
    for bbox in color_by_bbox:
        if bbox not in {successes[0], successes[1]} and random.random() < 0.5:
            color_by_bbox[bbox] = GREY if color_by_bbox[bbox] == RED else RED
    return [Triomino(obj.r0, obj.c0, obj.missing, color_by_bbox[obj.bbox]) for obj in objects]


def _segment_cells(side: str, offset: int, length: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(offset, offset + length)]
    if side == "bottom":
        return [(h - 1, c) for c in range(offset, offset + length)]
    if side == "left":
        return [(r, 0) for r in range(offset, offset + length)]
    if side == "right":
        return [(r, w - 1) for r in range(offset, offset + length)]
    raise AssertionError(side)


def _cell_sides(r: int, c: int, h: int, w: int) -> tuple[str, ...]:
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    return tuple(sides)


def _segment_clear(
    side: str,
    cells: list[tuple[int, int]],
    used_edges: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    if any(cell in used_edges for cell in cells):
        return False

    if side in {"top", "bottom"}:
        row = 0 if side == "top" else h - 1
        cols = {c for _r, c in cells}
        for c in range(min(cols) - 1, max(cols) + 2):
            if 0 <= c < w and c not in cols and (row, c) in used_edges:
                return False
    else:
        col = 0 if side == "left" else w - 1
        rows = {r for r, _c in cells}
        for r in range(min(rows) - 1, max(rows) + 2):
            if 0 <= r < h and r not in rows and (r, col) in used_edges:
                return False

    return True


def _place_targets(h: int, w: int, objects: list[Triomino], success_meta: list[dict]) -> list[tuple[int, int, int]]:
    occupied = {cell for obj in objects for cell in obj.cells}
    targets: list[tuple[int, int, int]] = []
    used_edges: set[tuple[int, int]] = set()
    side_counts = {side: 0 for side in ("top", "bottom", "left", "right")}

    success_choices = [m for m in success_meta if m["success"]]
    random.shuffle(success_choices)
    forced = False
    prioritized = sorted(success_choices, key=lambda item: len(_cell_sides(*item["stop_cell"], h, w)) > 1)
    for item in prioritized:
        r, c = item["stop_cell"]
        if (r, c) in occupied:
            continue
        sides = _cell_sides(r, c, h, w)
        if not sides:
            continue
        color = random.choice(TARGET_COLORS)
        targets.append((r, c, color))
        used_edges.add((r, c))
        side_counts[sides[0]] += 1
        forced = True
        break
    assert forced, "need at least one target-reaching success candidate"

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        remaining = 3 - side_counts[side]
        if remaining <= 0:
            continue
        seg_count = random.randint(0, remaining)
        attempts = 0
        placed = 0
        max_offset = w - 1 if side in {"top", "bottom"} else h - 1
        while placed < seg_count and attempts < 20:
            attempts += 1
            length = random.randint(1, 2)
            low = 1
            high = max_offset - length
            if high < low:
                break
            offset = random.randint(low, high)
            cells = _segment_cells(side, offset, length, h, w)
            if any(cell in occupied for cell in cells):
                continue
            if not _segment_clear(side, cells, used_edges, h, w):
                continue
            color = random.choice(TARGET_COLORS)
            for r, c in cells:
                targets.append((r, c, color))
                used_edges.add((r, c))
            placed += 1
            side_counts[side] += 1
    return targets


def _shortcut_outputs(grid: np.ndarray) -> list[np.ndarray]:
    outs = []
    outs.append(_simulate(grid, dynamic_paths=True, move_on_success=False)[0])
    outs.append(_simulate(grid, dynamic_paths=False, move_on_success=True)[0])
    outs.append(_simulate(grid, dynamic_paths=False, move_on_success=False)[0])
    return outs


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, meta = _simulate(grid, dynamic_paths=True, move_on_success=True)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    blocked_by_triomino = [m for m in meta if m["stop_kind"] == "triomino"]
    successes = [m for m in meta if m["success"]]
    if len(blocked_by_triomino) < 2 or len(successes) < 2:
        return False

    success_colors = {m["color"] for m in successes}
    if success_colors != {RED, GREY}:
        return False

    if not any(m["stop_kind"] == "target" for m in successes):
        return False

    frozen_out, frozen_meta = _simulate(grid, dynamic_paths=False, move_on_success=True)
    if np.array_equal(frozen_out, out):
        return False
    if not any(
        (_meta_by_bbox(frozen_meta, Triomino(m["r0"], m["c0"], m["missing"], m["color"]))["success"] != m["success"])
        or (_meta_by_bbox(frozen_meta, Triomino(m["r0"], m["c0"], m["missing"], m["color"]))["stop_kind"] != m["stop_kind"])
        for m in meta
    ):
        return False

    for shortcut in _shortcut_outputs(grid):
        if np.array_equal(shortcut, out):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    candidates = _all_geom_candidates(h, w)
    objects: list[Triomino] = []

    pair1 = _try_place_block_pair(h, w, objects)
    if pair1 is None:
        return None
    objects.extend(pair1)

    pair2 = _try_place_block_pair(h, w, objects)
    if pair2 is None:
        return None
    objects.extend(pair2)

    success1 = _candidate_success(h, w, objects, candidates)
    if success1 is None:
        return None
    objects.append(success1)

    success2 = _candidate_success(h, w, objects, candidates)
    if success2 is None:
        return None
    objects.append(success2)

    dependent = _candidate_order_dependent(h, w, objects, candidates)
    if dependent is not None:
        objects.append(dependent)

    if len(objects) < 5 or len(objects) > 8:
        return None

    base_grid = _render_input(h, w, objects, [])
    try:
        _base_out, base_meta = _simulate(base_grid)
    except AssertionError:
        return None
    if sum(1 for item in base_meta if item["success"]) < 2:
        return None

    try:
        colored_objects = _assign_colors(objects, base_meta)
    except ValueError:
        return None
    color_grid = _render_input(h, w, colored_objects, [])
    try:
        _color_out, color_meta = _simulate(color_grid)
    except AssertionError:
        return None

    targets = _place_targets(h, w, colored_objects, color_meta)
    grid = _render_input(h, w, colored_objects, targets)
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
