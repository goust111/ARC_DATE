# Variant of 88e364bc
# Dimension: D16 — 对角线滑动的角切割规则 (path)
# Change: 禁止角切割——对角线移动时，两个正交中间格子都必须在区域内，否则无法通过... → 宽度约束（需要2格宽通道）

"""
Design analysis for the D16 variant
===================================

spec:
  original_core:
    - Border legends still map wall color -> slide direction.
    - Yellow pixels still slide inside enclosed background regions and stack.
    - Diagonal moves still forbid corner-cutting: both orthogonal intermediate cells must exist.
  variant_core:
    - Cardinal movement is no longer "point-like".
    - A horizontal/vertical step is legal only if the step can be covered by at least one 2x2
      window fully inside the region. Equivalently, the current cell and the destination cell
      must have a same-side perpendicular support pair.
    - Therefore a 1-cell-wide corridor blocks yellow even when the next cell in the direction
      exists. This is the direct adaptation of the width-constraint idea from 332f06d7.

behavior_trace:
  concrete_example:
    - East-moving yellow sits in a 5-row room connected to another room by a 1-row corridor.
    - Original logic:
      - from mouth cell (r, c) to corridor cell (r, c+1), destination is inside, so the yellow
        enters the corridor and keeps going to the far room.
    - Variant logic:
      - the same step also needs one full 2x2 window spanning columns c..c+1.
      - at the corridor mouth, neither the upper pair nor the lower pair stays inside because the
        corridor height is only 1.
      - the yellow stops at the near-side mouth instead of passing through.
  affected_pipeline_stage:
    - solver: step-validity inside _slide_points changes.
    - generator: must deliberately build at least one region where "next cell exists but width=1"
      occurs along a cardinal slide; otherwise the new rule degenerates to the old task.

complexity_calibration:
  - The new rule is intentionally a single extra local geometric test, not a new global search.
  - To keep the rule observable, every generated grid contains at least one cardinal-direction
    region with a 1-cell bottleneck aligned with the motion direction.
  - Additional regions may remain wide/open so the puzzle still shows normal sliding too.

implicit_assumption_audit:
  - Original _slide_points assumed "destination inside region" is sufficient for cardinal steps.
    That assumption is invalid now; the new solver adds a same-side support check.
  - Original diagonal check depended on a point-sized mover and is still valid for this variant;
    we keep it unchanged to avoid introducing an unrelated diagonal geometry.
  - main() identifies enclosed passable components by 4-connectivity and infers border color by
    majority adjacent walls. The generator therefore keeps each region surrounded by a single wall
    color and leaves 1-cell buffers between different structures.
  - main() assumes legends are detectable as solid border rectangles with exactly 3 colors. The
    generator preserves that format unchanged.

placement_legality:
  - Regions are generated as explicit interior masks first; walls are computed as the 4-neighbor
    ring outside that interior.
  - A region is accepted only if interior and wall ring fit strictly inside the 20x20 board, do
    not overlap legends, and do not overlap previous regions or their safety buffers.
  - Topology is validated by re-labeling bounded background components and checking they match the
    intended interiors exactly.

visual_observability:
  - The bottleneck is visible as a narrow 1-cell neck between two larger rooms.
  - At least one yellow is placed so its arrow points through that neck. Under the variant it
    piles up on the near side, which is visually distinct from the original far-wall behavior.
  - Legends still carry only a small set of repeated directions/colors, so the main inference is
    geometric rather than color-numeric.

cleanliness_and_colors:
  - Only the standard task colors are used: bg=0, center=1, indicator=2, yellow=4, wall colors
    chosen from {3,5,6,7,8,9}.
  - No hidden ordering by color value is used anywhere; all behavior comes from visible geometry.
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        if len(others) < 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _cardinal_width_ok(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    """Cardinal step requires one full 2x2 window spanning current and destination."""
    for sr, sc in ((-dc, dr), (dc, -dr)):
        if (r + sr, c + sc) in region_set and (r + dr + sr, c + dc + sc) in region_set:
            return True
    return False


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        return (r + dr, c) in region_set and (r, c + dc) in region_set
    return _cardinal_width_ok(region_set, r, c, dr, dc)


def _slide_points(
    region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            if not _slide_step_valid(region_set, r, c, dr, dc):
                break
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg

        for r, c in _slide_points(pset, movers, dr, dc):
            out[r, c] = mover

    return out


GRID_H, GRID_W = 20, 20
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.55
LEGEND_BUFFER = 1
MAX_SLIDE_STEPS = 6
GEN_RETRIES = 400


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _reserve_with_buffer(mask: np.ndarray, coords: Iterable[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in coords:
        mask[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _transform_point_cardinal(
    pt: tuple[int, int], h: int, w: int, dr: int, dc: int
) -> tuple[int, int]:
    r, c = pt
    if (dr, dc) == (0, 1):
        return r, c
    if (dr, dc) == (0, -1):
        return r, w - 1 - c
    if (dr, dc) == (1, 0):
        return c, h - 1 - r
    if (dr, dc) == (-1, 0):
        return w - 1 - c, r
    raise ValueError((dr, dc))


def _transform_cardinal_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    if (dr, dc) == (0, 1):
        return mask.copy()
    if (dr, dc) == (0, -1):
        return np.fliplr(mask)
    if (dr, dc) == (1, 0):
        return np.rot90(mask, 3)
    if (dr, dc) == (-1, 0):
        return np.rot90(mask, 1)
    raise ValueError((dr, dc))


def _make_chokepoint_region(rng: random.Random, color: int, direction: tuple[int, int]) -> dict:
    left_w = rng.randint(5, 6)
    right_w = rng.randint(3, 4)
    corridor_len = rng.randint(2, 4)
    base = np.zeros((5, left_w + corridor_len + right_w), dtype=bool)
    base[:, :left_w] = True
    base[2, left_w : left_w + corridor_len] = True
    base[:, left_w + corridor_len :] = True

    mask = _transform_cardinal_mask(base, *direction)
    h0, w0 = base.shape
    forced_outputs = [
        _transform_point_cardinal((2, left_w - 1), h0, w0, *direction),
        _transform_point_cardinal((2, left_w - 2), h0, w0, *direction),
    ]
    interior = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    return {
        "color": color,
        "dir": direction,
        "interior_local": interior,
        "shape": mask.shape,
        "forced_outputs_local": forced_outputs,
        "kind": "chokepoint",
    }


def _make_open_region(rng: random.Random, color: int, direction: tuple[int, int]) -> dict:
    h = rng.randint(4, 6)
    w = rng.randint(5, 7)
    mask = np.ones((h, w), dtype=bool)
    interior = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    return {
        "color": color,
        "dir": direction,
        "interior_local": interior,
        "shape": mask.shape,
        "forced_outputs_local": None,
        "kind": "open",
    }


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    forced_cardinal = rng.randrange(n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []

    for idx, (corner, wall_color) in enumerate(zip(corners, wall_colors)):
        if idx == forced_cardinal or rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wall_color, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"color": wall_color, "dir": (dr, dc), "kind": "chokepoint" if idx == forced_cardinal else "open"})

    return legend_occupied, legends


def _shift_points(points: Iterable[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _try_place_region(rng: random.Random, occupied: np.ndarray, spec: dict) -> dict | None:
    h, w = occupied.shape
    rh, rw = spec["shape"]
    row_choices = list(range(1, h - rh - 1))
    col_choices = list(range(1, w - rw - 1))
    rng.shuffle(row_choices)
    rng.shuffle(col_choices)

    for r0 in row_choices:
        for c0 in col_choices:
            interior = set(_shift_points(spec["interior_local"], r0, c0))
            walls = _blob_wall_cells(interior, h, w)
            if any(
                rr in (0, h - 1) or cc in (0, w - 1) or occupied[rr, cc] for rr, cc in interior | walls
            ):
                continue
            return {
                "color": spec["color"],
                "dir": spec["dir"],
                "interior": interior,
                "walls": walls,
                "kind": spec["kind"],
                "forced_outputs": None
                if spec["forced_outputs_local"] is None
                else _shift_points(spec["forced_outputs_local"], r0, c0),
            }
    return None


def _build_regions(rng: random.Random, legends: list[dict], legend_occupied: np.ndarray) -> list[dict] | None:
    occupied = legend_occupied.copy()
    regions = []

    for legend in legends:
        if legend["kind"] == "chokepoint":
            spec = _make_chokepoint_region(rng, legend["color"], legend["dir"])
        else:
            spec = _make_open_region(rng, legend["color"], legend["dir"])
        placed = _try_place_region(rng, occupied, spec)
        if placed is None:
            return None
        _reserve_with_buffer(occupied, placed["interior"] | placed["walls"])
        regions.append(placed)

    if len(legends) == 1 and rng.random() < 0.4:
        extra_spec = _make_open_region(rng, legends[0]["color"], legends[0]["dir"])
        extra = _try_place_region(rng, occupied, extra_spec)
        if extra is not None:
            _reserve_with_buffer(occupied, extra["interior"] | extra["walls"])
            regions.append(extra)

    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _forward_slide(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return r, c


def _find_reverse_slide(
    blob: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    already_stopped: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in blob or (pr, pc) in already_stopped or (pr, pc) in existing_inputs:
                path_ok = False
                break
            if not _slide_step_valid(blob, pr, pc, dr, dc):
                path_ok = False
                break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(blob, (r, c), dr, dc, already_stopped) == out_pos:
            return r, c
    return None


def _choose_open_outputs(
    rng: random.Random, blob: set[tuple[int, int]], dr: int, dc: int, n_yellows: int
) -> list[tuple[int, int]]:
    cells_sorted = sorted(blob, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    far_key = dr * cells_sorted[0][0] + dc * cells_sorted[0][1]
    tier1 = [p for p in cells_sorted if dr * p[0] + dc * p[1] == far_key]
    first = rng.choice(tier1)
    output_positions = [first]
    if n_yellows == 2:
        pr, pc = first[0] - dr, first[1] - dc
        cands = ([(pr, pc)] if (pr, pc) in blob else []) + [p for p in tier1 if p != first]
        if cands:
            second = rng.choice(cands)
            if second != first:
                output_positions.append(second)
    return output_positions


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    choke_seen = False
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        n_yellows = rng.choice([1, 2])
        if reg["forced_outputs"] is not None:
            choke_seen = True
            output_positions = reg["forced_outputs"][:n_yellows]
        else:
            output_positions = _choose_open_outputs(rng, blob, dr, dc, n_yellows)

        out_sorted = sorted(output_positions, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
        already_stopped: set[tuple[int, int]] = set()
        yellows_input: list[tuple[int, int]] = []

        for out_pos in out_sorted:
            candidate = _find_reverse_slide(blob, out_pos, dr, dc, already_stopped, yellows_input, rng)
            if candidate is None:
                return False
            yellows_input.append(candidate)
            already_stopped.add(out_pos)

        for r, c in yellows_input:
            grid[r, c] = 4

    return choke_seen


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == 4).sum()) != int((input_grid == 4).sum()):
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == 4]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _build_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
