from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(10))
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ARROW_SPECS = {
    "up": {
        "local_cells": {(1, 0), (1, 1), (1, 2), (0, 1)},
        "tip_local": (0, 1),
        "direction": (-1, 0),
    },
    "right": {
        "local_cells": {(0, 0), (1, 0), (2, 0), (1, 1)},
        "tip_local": (1, 1),
        "direction": (0, 1),
    },
    "down": {
        "local_cells": {(0, 0), (0, 1), (0, 2), (1, 1)},
        "tip_local": (1, 1),
        "direction": (1, 0),
    },
    "left": {
        "local_cells": {(0, 1), (1, 0), (1, 1), (2, 1)},
        "tip_local": (1, 0),
        "direction": (0, -1),
    },
}

for _spec in ARROW_SPECS.values():
    tr, tc = _spec["tip_local"]
    _spec["tip_relative_cells"] = {(r - tr, c - tc) for r, c in _spec["local_cells"]}

DISTRACTOR_SHAPES = [
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
]

GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def bbox_coords(coords: list[tuple[int, int]] | np.ndarray | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [int(r) for r, _c in pts]
    cols = [int(c) for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in N4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def find_components(grid: np.ndarray, *, exclude_colors: set[int] | None = None) -> list[dict]:
    exclude = set() if exclude_colors is None else {int(v) for v in exclude_colors}
    out: list[dict] = []
    for color in sorted(set(map(int, grid.flatten())) - exclude):
        if color in exclude:
            continue
        for coords in components_mask(grid == int(color)):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _turn(direction: tuple[int, int], sense: str = "right") -> tuple[int, int]:
    dr, dc = map(int, direction)
    if sense == "right":
        return dc, -dr
    if sense == "left":
        return -dc, dr
    raise ValueError(f"invalid turn sense: {sense}")


def _overlap_bbox(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    gap: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _expand_cells(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    *,
    gap: int = 1,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    h, w = map(int, shape)
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _match_arrow_component(coords: np.ndarray) -> dict | None:
    bbox = bbox_coords(coords)
    r0, c0, _r1, _c1 = bbox
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    for name, spec in ARROW_SPECS.items():
        if rel == spec["local_cells"]:
            tr, tc = spec["tip_local"]
            tip = (int(r0 + tr), int(c0 + tc))
            return {
                "name": name,
                "tip": tip,
                "direction": tuple(map(int, spec["direction"])),
                "cells": {(int(r), int(c)) for r, c in coords},
            }
    return None


def _legend_candidates(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in find_components(grid, exclude_colors=set(PALETTE) - {BLACK}):
        if int(comp["color"]) != BLACK:
            continue
        coords = {(int(r), int(c)) for r, c in comp["coords"]}
        bbox = comp["bbox"]
        r0, c0, r1, c1 = bbox
        if (r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 5:
            continue
        border = {
            (int(r), int(c))
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if coords == border:
            candidates.append(tuple(map(int, bbox)))
    return candidates


def _find_legend_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = _legend_candidates(grid)
    assert len(candidates) == 1, f"expected exactly one legend box, got {len(candidates)}"
    return candidates[0]


def _read_legend(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    bg: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, legend_bbox)
    inner = crop_bbox(grid, (r0 + 1, c0 + 1, r1 - 1, c1 - 1))
    key_col = None
    for cc in range(inner.shape[1]):
        col_vals = {int(v) for v in inner[:, cc]}
        col_vals.discard(int(bg))
        col_vals.discard(BLACK)
        if col_vals:
            key_col = int(cc)
            break
    assert key_col is not None, "legend key column not found"

    program: list[tuple[int, int]] = []
    for rr in range(inner.shape[0]):
        key = int(inner[rr, key_col])
        if key in (int(bg), BLACK):
            continue
        dots = [
            int(cc)
            for cc in range(key_col + 1, inner.shape[1])
            if int(inner[rr, cc]) not in (int(bg), BLACK)
        ]
        assert dots, "legend row has key but no dots"
        dot_colors = {int(inner[rr, cc]) for cc in dots}
        assert dot_colors == {int(key)}, "legend row dots must share the key color"
        program.append((int(key), int(len(dots))))
    assert len(program) >= 1, "empty legend program"
    return program


def _arrow_candidates(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    bg: int,
) -> list[dict]:
    r0, c0, r1, c1 = map(int, legend_bbox)
    out: list[dict] = []
    for comp in find_components(grid, exclude_colors={int(bg), BLACK}):
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in coords):
            continue
        match = _match_arrow_component(comp["coords"])
        if match is not None:
            out.append(match)
    return out


def _find_arrow(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    bg: int,
) -> dict:
    candidates = _arrow_candidates(grid, legend_bbox, bg)
    assert len(candidates) == 1, f"expected exactly one arrow, got {len(candidates)}"
    return candidates[0]


def _trace_program(
    start: tuple[int, int],
    direction: tuple[int, int],
    program: list[tuple[int, int]],
    *,
    turn: str = "right",
    first_includes_tip: bool = True,
) -> list[tuple[int, int, int]]:
    if not program:
        return []
    pos = (int(start[0]), int(start[1]))
    dr, dc = map(int, direction)
    out: list[tuple[int, int, int]] = []
    for idx, (color, steps) in enumerate(program):
        if idx > 0:
            dr, dc = _turn((dr, dc), sense=turn)
        if idx == 0 and first_includes_tip:
            out.append((int(pos[0]), int(pos[1]), int(color)))
            lead = int(steps - 1)
        else:
            lead = int(steps)
        for _ in range(max(0, lead)):
            pos = (int(pos[0] + dr), int(pos[1] + dc))
            out.append((int(pos[0]), int(pos[1]), int(color)))
    return out


def _render_output(
    grid: np.ndarray,
    arrow_cells: set[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    program: list[tuple[int, int]],
    bg: int,
    *,
    turn: str = "right",
    first_includes_tip: bool = True,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in arrow_cells:
        if in_bounds(out.shape, r, c):
            out[int(r), int(c)] = int(bg)
    for r, c, color in _trace_program(
        start,
        direction,
        program,
        turn=turn,
        first_includes_tip=first_includes_tip,
    ):
        if in_bounds(out.shape, r, c):
            out[int(r), int(c)] = int(color)
    return out


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legend_bbox = _find_legend_box(g)
    program = _read_legend(g, legend_bbox, bg)
    arrow = _find_arrow(g, legend_bbox, bg)
    return {
        "bg": bg,
        "legend_bbox": legend_bbox,
        "program": program,
        "tip": tuple(map(int, arrow["tip"])),
        "direction": tuple(map(int, arrow["direction"])),
        "arrow_cells": set(arrow["cells"]),
    }


def _distractor_stats(
    grid: np.ndarray,
    legend_bbox: tuple[int, int, int, int],
    arrow_cells: set[tuple[int, int]],
    bg: int,
) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = {}
    r0, c0, r1, c1 = map(int, legend_bbox)
    for comp in find_components(grid, exclude_colors={int(bg), BLACK}):
        color = int(comp["color"])
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in coords):
            continue
        cellset = {(int(r), int(c)) for r, c in coords}
        if cellset == set(arrow_cells):
            continue
        bucket = stats.setdefault(color, {"pixels": 0, "components": 0})
        bucket["pixels"] += int(len(coords))
        bucket["components"] += 1
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the legend rows as a right-turn drawing program from the triangle tip.

    Input structure:
    - A black rectangular legend box contains several keyed rows: the leftmost
      non-background interior column holds one color key per used row, and the
      same row has a run of separated colored dots to its right.
    - Outside the box there is exactly one monochrome triangle-arrow sprite whose
      pointed tip gives the start cell and initial direction.
    - Other outside monochrome distractor objects reuse legend colors but do not
      encode the step lengths.

    Rule:
    - Read the used legend rows from top to bottom. For each row, take its key
      color and count the dots on that row.
    - Clear the triangle sprite. The tip cell is the first painted cell of the
      first segment; later segments start after turning 90 degrees clockwise.
    - Draw consecutive path segments with those row colors and dot counts, while
      leaving the legend box and the unrelated distractors unchanged.

    Key insight:
    - The legend is not an object-count summary: row order is execution order,
      dot count is segment length, and the arrow contributes only the start cell
      plus the initial heading.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _render_output(
        np.asarray(grid, dtype=int),
        scene["arrow_cells"],
        scene["tip"],
        scene["direction"],
        scene["program"],
        scene["bg"],
    )


def _shape_cells(shape: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in shape}


def _choose_distractor_shapes(step: int) -> list[set[tuple[int, int]]] | None:
    for _ in range(80):
        count = random.randint(1, 3)
        if count == int(step):
            continue
        shapes = [set(random.choice(DISTRACTOR_SHAPES)) for _ in range(count)]
        pixels = sum(len(shape) for shape in shapes)
        if pixels != int(step):
            return shapes
    return None


def _place_shape(
    blocked: set[tuple[int, int]],
    shape: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    h, w = map(int, grid_shape)
    sr0, sc0, sr1, sc1 = bbox_coords(shape)
    sh = int(sr1 - sr0 + 1)
    sw = int(sc1 - sc0 + 1)
    row_lo = 1 if h - sh >= 2 else 0
    row_hi = h - sh - 1 if h - sh >= 2 else h - sh
    col_lo = 1 if w - sw >= 2 else 0
    col_hi = w - sw - 1 if w - sw >= 2 else w - sw
    if row_lo > row_hi or col_lo > col_hi:
        return None
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(int(row_lo), int(row_hi))
        left = random.randint(int(col_lo), int(col_hi))
        placed = _shape_cells(shape, top - sr0, left - sc0)
        if all((int(r), int(c)) not in blocked for r, c in placed):
            return placed
    return None


def _program_differs(program: list[tuple[int, int]]) -> bool:
    return len({int(steps) for _color, steps in program}) > 1


def _validate_instance(grid: np.ndarray, spec: dict) -> bool:
    try:
        scene = _extract_scene(grid)
    except Exception:
        return False

    if scene["legend_bbox"] != tuple(spec["legend_bbox"]):
        return False
    if scene["program"] != list(spec["program"]):
        return False
    if scene["tip"] != tuple(spec["tip"]):
        return False
    if scene["direction"] != tuple(spec["direction"]):
        return False
    if scene["arrow_cells"] != set(spec["arrow_cells"]):
        return False
    if len(_legend_candidates(grid)) != 1:
        return False
    if len(_arrow_candidates(grid, scene["legend_bbox"], scene["bg"])) != 1:
        return False

    truth = _render_output(
        grid,
        scene["arrow_cells"],
        scene["tip"],
        scene["direction"],
        scene["program"],
        scene["bg"],
    )
    if not np.array_equal(truth, main(grid)):
        return False

    stats = _distractor_stats(grid, scene["legend_bbox"], scene["arrow_cells"], scene["bg"])
    for color, steps in scene["program"]:
        if color not in stats:
            return False
        if stats[color]["components"] == int(steps):
            return False
        if stats[color]["pixels"] == int(steps):
            return False

    if not _program_differs(scene["program"]):
        return False

    reversed_program = list(reversed(scene["program"]))
    if np.array_equal(
        truth,
        _render_output(
            grid,
            scene["arrow_cells"],
            scene["tip"],
            scene["direction"],
            reversed_program,
            scene["bg"],
        ),
    ):
        return False

    if np.array_equal(
        truth,
        _render_output(
            grid,
            scene["arrow_cells"],
            scene["tip"],
            scene["direction"],
            scene["program"],
            scene["bg"],
            turn="left",
        ),
    ):
        return False

    if np.array_equal(
        truth,
        _render_output(
            grid,
            scene["arrow_cells"],
            scene["tip"],
            scene["direction"],
            scene["program"],
            scene["bg"],
            first_includes_tip=False,
        ),
    ):
        return False

    distractor_program = [(int(color), int(stats[color]["components"])) for color, _steps in scene["program"]]
    if np.array_equal(
        truth,
        _render_output(
            grid,
            scene["arrow_cells"],
            scene["tip"],
            scene["direction"],
            distractor_program,
            scene["bg"],
        ),
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(3, 5)
    usable = list(range(1, 10))
    random.shuffle(usable)
    bg = int(usable.pop())
    arrow_color = int(usable.pop())
    key_colors = [int(usable.pop()) for _ in range(n_rows)]

    steps = [random.randint(2, 6) for _ in range(n_rows)]
    if len(set(steps)) == 1:
        idx = random.randrange(n_rows)
        choices = [v for v in range(2, 7) if v != steps[idx]]
        steps[idx] = int(random.choice(choices))
    program = list(zip(key_colors, steps))

    arrow_name = random.choice(sorted(ARROW_SPECS.keys()))
    arrow_spec = ARROW_SPECS[arrow_name]
    direction = tuple(map(int, arrow_spec["direction"]))

    rel_path = _trace_program((0, 0), direction, program)
    rel_path_cells = [(int(r), int(c)) for r, c, _color in rel_path]
    if len(set(rel_path_cells)) != len(rel_path_cells):
        return None

    arrow_rel_cells = {(int(r), int(c)) for r, c in arrow_spec["tip_relative_cells"]}
    all_rel = set(rel_path_cells) | set(arrow_rel_cells)
    rel_bbox = bbox_coords(all_rel)
    pr0, pc0, pr1, pc1 = rel_bbox

    inner_h = 2 * n_rows - 1
    inner_w = 2 * max(steps) + 1
    box_h = inner_h + 2
    box_w = inner_w + 2

    min_h = max(18, box_h + 4, (pr1 - pr0 + 1) + 4)
    min_w = max(18, box_w + 4, (pc1 - pc0 + 1) + 4)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(int(min_h), 30)
    w = random.randint(int(min_w), 30)
    grid = np.full((h, w), int(bg), dtype=int)

    box_top = random.randint(1, h - box_h - 1)
    box_left = random.randint(1, w - box_w - 1)
    legend_bbox = (int(box_top), int(box_left), int(box_top + box_h - 1), int(box_left + box_w - 1))

    tip_row_lo = -pr0
    tip_row_hi = h - 1 - pr1
    tip_col_lo = -pc0
    tip_col_hi = w - 1 - pc1
    if tip_row_lo > tip_row_hi or tip_col_lo > tip_col_hi:
        return None

    tip = None
    arrow_cells_abs: set[tuple[int, int]] = set()
    path_cells_abs: set[tuple[int, int]] = set()
    for _ in range(80):
        cand_tip = (
            random.randint(int(tip_row_lo), int(tip_row_hi)),
            random.randint(int(tip_col_lo), int(tip_col_hi)),
        )
        arrow_cells_abs = {
            (int(cand_tip[0] + rr), int(cand_tip[1] + cc))
            for rr, cc in arrow_rel_cells
        }
        path_cells_abs = {
            (int(cand_tip[0] + r), int(cand_tip[1] + c))
            for r, c in rel_path_cells
        }
        combined_bbox = bbox_coords(arrow_cells_abs | path_cells_abs)
        if _overlap_bbox(combined_bbox, legend_bbox, gap=2):
            continue
        tip = cand_tip
        break
    if tip is None:
        return None

    r0, c0, r1, c1 = legend_bbox
    grid[r0 : r1 + 1, c0] = BLACK
    grid[r0 : r1 + 1, c1] = BLACK
    grid[r0, c0 : c1 + 1] = BLACK
    grid[r1, c0 : c1 + 1] = BLACK

    for idx, (color, step) in enumerate(program):
        rr = int(r0 + 1 + 2 * idx)
        key_col = int(c0 + 1)
        grid[rr, key_col] = int(color)
        for k in range(int(step)):
            grid[rr, c0 + 3 + 2 * k] = int(color)

    for r, c in arrow_cells_abs:
        if not in_bounds(grid.shape, r, c):
            return None
        grid[int(r), int(c)] = int(arrow_color)

    reserved: set[tuple[int, int]] = set()
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            reserved.add((int(rr), int(cc)))
    reserved |= set(arrow_cells_abs)
    reserved |= set(path_cells_abs)
    blocked = _expand_cells(reserved, grid.shape, gap=1)

    placements: list[tuple[int, set[tuple[int, int]]]] = []
    color_order = list(program)
    random.shuffle(color_order)
    for color, step in color_order:
        shapes = _choose_distractor_shapes(int(step))
        if shapes is None:
            return None
        per_color_cells: list[set[tuple[int, int]]] = []
        for shape in shapes:
            placed = _place_shape(blocked, shape, grid.shape)
            if placed is None:
                return None
            per_color_cells.append(placed)
            blocked |= _expand_cells(placed, grid.shape, gap=1)
        for placed in per_color_cells:
            placements.append((int(color), placed))

    for color, cells in placements:
        for r, c in cells:
            grid[int(r), int(c)] = int(color)

    spec = {
        "legend_bbox": legend_bbox,
        "program": program,
        "tip": tip,
        "direction": direction,
        "arrow_cells": arrow_cells_abs,
    }
    if not _validate_instance(grid, spec):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
