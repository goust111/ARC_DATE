# Variant of 88e364bc
# Dimension: D17 — 区域边界颜色判定方式 (obstacle)
# Change: region_membership——区域边界颜色由相邻墙壁像素的多数颜色决定（border_counts取max）... → 每个黄色像素独立判定最近墙壁颜色

"""
Design analysis for the D17 variant
===================================

spec_variant:
  preserved:
    - grid is 20x20 with bg=0
    - legends still sit on the border; legend fill color still maps wall color -> direction
    - yellow pixels still slide inside enclosed passable regions and walls/legends stay fixed
  changed_core:
    - a region is no longer assigned one shared border color by majority count
    - each yellow pixel independently looks at the nearest wall pixel around its own enclosed region
    - that nearest wall pixel's color selects the legend, so yellows in the same region may move in different directions
  concrete_visualization:
    - the generator uses one large enclosed chamber whose boundary contains both legend colors
    - one wall side is painted with a minority color, the other three sides with the majority color
    - at least two yellows are placed in the same chamber, each inside a zone whose nearest wall color differs
    - output therefore shows same-region yellows separating according to different legends
  traced_example:
    - suppose a chamber has grey on top/left/bottom and green on the right side
    - a yellow near the left half sees grey as the nearest wall color and uses the grey legend direction
    - a yellow near the right half sees green as the nearest wall color and uses the green legend direction
    - original logic would count the whole boundary, pick grey as the regional majority, and move both yellows in the grey direction
    - new logic moves them differently, so the output visibly diverges from the original task
  complexity_calibration:
    - exactly two legend colors are used so the mapping is still learnable from 2-4 examples
    - the chamber is mixed-color but visually clean; colors repeat on long wall segments instead of noisy per-pixel alternation
    - the generator rejects placements where both yellows effectively behave like the original majority-based solver
  implicit_assumptions_audit:
    - original `border_counts[...] += 1` assumed a region can be summarized by one color; no longer valid
    - original `border_col = max(border_counts, key=...)` assumed one direction per region; no longer valid
    - original `_slide_points(pset, movers, dr, dc)` assumed all movers in the region share `(dr, dc)`; no longer valid
    - clearing all movers first is still safe, but directions must be computed from the input positions before any motion occurs
    - mixed directions introduce ordering risk, so the solver switches to repeated one-step simulation with fixed per-yellow directions
  legality_and_generator_guarantees:
    - legends are pasted first and the chamber is searched in the remaining free space with a one-cell buffer
    - the chamber is fully enclosed and does not touch the outer border
    - every chamber wall color is one of the legend fill colors, so every yellow can be resolved by the solver
    - candidate yellow cells are filtered so their nearest wall color is unambiguous under the solver's tie-break and they have at least two free slide steps
  observability_and_cleanliness:
    - the chamber boundary is a few long colored sides, so a human can see which yellow is closer to which wall color
    - the generator forces the new solver's output to differ from a legacy majority-color solver, preventing silent regression
  new_solver_assumptions_checked:
    - nearest-wall lookup assumes the enclosed region has at least one adjacent non-bg, non-yellow wall pixel; chamber construction guarantees this
    - simultaneous step simulation may deadlock on head-on conflicts; the generator avoids crowded placements by spacing yellows and validating the produced sample
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
GEN_RETRIES = 300
MIN_MOVE_STEPS = 2
CHAMBER_OUTER_H = range(8, 12)
CHAMBER_OUTER_W = range(8, 12)


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legend rectangles and decode fill-color -> movement direction."""
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
        others.sort()
        if len(others) != 2:
            continue
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            candidates = []
            for _cnt, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                candidates.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            candidates.sort()
            center_col = candidates[0][1]
            indicator_col = candidates[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

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


def _slide_points(region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    """Original same-direction slide, kept for legacy-vs-variant validation in the generator."""
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _boundary_wall_cells(
    g: np.ndarray, region_set: set[tuple[int, int]], bg: int, mover: int
) -> list[tuple[int, int]]:
    h, w = g.shape
    boundary = set()
    for r, c in region_set:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in region_set:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            boundary.add((nr, nc))
    return sorted(boundary)


def _nearest_wall_color(g: np.ndarray, boundary: list[tuple[int, int]], pos: tuple[int, int]) -> int | None:
    if not boundary:
        return None
    r0, c0 = pos
    _dist, wr, wc = min((abs(wr - r0) + abs(wc - c0), wr, wc) for wr, wc in boundary)
    return int(g[wr, wc])


def _simulate_individual_slides(
    region_set: set[tuple[int, int]],
    movers: list[tuple[int, int]],
    move_dirs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Repeated one-step simulation with a fixed direction per yellow."""
    positions = list(movers)
    max_iters = max(1, len(region_set) * 2)
    for _ in range(max_iters):
        current = set(positions)
        proposals: list[tuple[int, int]] = []
        target_counts: dict[tuple[int, int], int] = {}
        for (r, c), (dr, dc) in zip(positions, move_dirs):
            if not _slide_step_valid(region_set, r, c, dr, dc):
                proposals.append((r, c))
                continue
            nr, nc = r + dr, c + dc
            if (nr, nc) in current:
                proposals.append((r, c))
                continue
            proposals.append((nr, nc))
            target_counts[(nr, nc)] = target_counts.get((nr, nc), 0) + 1

        moved = False
        new_positions = []
        for pos, proposal in zip(positions, proposals):
            if proposal != pos and target_counts.get(proposal, 0) == 1:
                new_positions.append(proposal)
                moved = True
            else:
                new_positions.append(pos)
        positions = new_positions
        if not moved:
            break
    return positions


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original region-majority interpretation, used only to reject degenerate generated samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    out = g.copy()
    passable = (g == bg) | (g == mover)

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


def _assigned_mover_data(grid: np.ndarray) -> list[dict]:
    """Helper for validation/metrics: nearest-wall color and direction for each yellow."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    passable = (g == bg) | (g == mover)
    info = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        region_set = {(int(r), int(c)) for r, c in pts}
        boundary = _boundary_wall_cells(g, region_set, bg, mover)
        for r, c in pts:
            if int(g[r, c]) != mover:
                continue
            pos = (int(r), int(c))
            color = _nearest_wall_color(g, boundary, pos)
            if color is None or color not in dirs:
                continue
            info.append({"pos": pos, "color": color, "dir": dirs[color]})
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - legends still map wall colors to directions
    - enclosed passable regions are still found as before
    - but each yellow independently chooses its direction from the color of its nearest wall pixel
    - yellows then slide with fixed per-pixel directions until blocked
    """
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

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        boundary = _boundary_wall_cells(g, region_set, bg, mover)
        active_movers = []
        move_dirs = []
        for pos in movers:
            border_col = _nearest_wall_color(g, boundary, pos)
            if border_col is None or border_col not in dirs:
                continue
            active_movers.append(pos)
            move_dirs.append(dirs[border_col])

        for r, c in active_movers:
            out[r, c] = bg
        for r, c in _simulate_individual_slides(region_set, active_movers, move_dirs):
            out[r, c] = mover

    return out


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


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    """Always place two distinct legends so mixed wall colors are visible."""
    h, w = grid.shape
    legend_occupied = np.zeros((h, w), dtype=bool)
    corners = rng.sample(CORNERS, 2)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, 2)

    directions = rng.sample(ALL_DIRS, 2)
    legends = []
    for corner, wc, (dr, dc) in zip(corners, wall_colors, directions):
        if (dr, dc) in CARDINAL_DIRS:
            patch = _legend_5x5(wc, dr, dc)
        else:
            patch = _legend_6x7(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))
    return legend_occupied, legends


def _find_chamber_rect(rng: random.Random, legend_occupied: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = legend_occupied.shape
    candidates = []
    for outer_h in CHAMBER_OUTER_H:
        for outer_w in CHAMBER_OUTER_W:
            for r0 in range(1, h - outer_h):
                for c0 in range(1, w - outer_w):
                    r1 = r0 + outer_h - 1
                    c1 = c0 + outer_w - 1
                    rr0 = max(0, r0 - 1)
                    rr1 = min(h, r1 + 2)
                    cc0 = max(0, c0 - 1)
                    cc1 = min(w, c1 + 2)
                    if legend_occupied[rr0:rr1, cc0:cc1].any():
                        continue
                    candidates.append((r0, c0, r1, c1))
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


def _build_chamber(
    grid: np.ndarray, legends: list[tuple[int, tuple[int, int]]], rect: tuple[int, int, int, int], rng: random.Random
) -> dict:
    """Paint one enclosed rectangular chamber whose boundary mixes both legend colors."""
    r0, c0, r1, c1 = rect
    majority_idx = rng.randrange(2)
    minority_idx = 1 - majority_idx
    majority_color = legends[majority_idx][0]
    minority_color = legends[minority_idx][0]
    minority_side = rng.choice(["top", "bottom", "left", "right"])

    wall_color_at: dict[tuple[int, int], int] = {}
    for c in range(c0, c1 + 1):
        wall_color_at[(r0, c)] = majority_color
        wall_color_at[(r1, c)] = majority_color
    for r in range(r0 + 1, r1):
        wall_color_at[(r, c0)] = majority_color
        wall_color_at[(r, c1)] = majority_color

    if minority_side == "top":
        for c in range(c0, c1 + 1):
            wall_color_at[(r0, c)] = minority_color
    elif minority_side == "bottom":
        for c in range(c0, c1 + 1):
            wall_color_at[(r1, c)] = minority_color
    elif minority_side == "left":
        for r in range(r0 + 1, r1):
            wall_color_at[(r, c0)] = minority_color
    else:
        for r in range(r0 + 1, r1):
            wall_color_at[(r, c1)] = minority_color

    for (r, c), color in wall_color_at.items():
        grid[r, c] = color

    interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    boundary = sorted(wall_color_at)
    return {
        "interior": interior,
        "boundary": boundary,
        "majority_color": majority_color,
        "minority_color": minority_color,
        "minority_side": minority_side,
    }


def _move_distance(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int) -> int:
    r, c = start
    steps = 0
    while _slide_step_valid(region_set, r, c, dr, dc):
        r += dr
        c += dc
        steps += 1
    return steps


def _candidate_cells_for_color(
    grid: np.ndarray,
    region_set: set[tuple[int, int]],
    boundary: list[tuple[int, int]],
    dirs: dict[int, tuple[int, int]],
    color: int,
) -> list[tuple[int, int]]:
    candidates = []
    for cell in sorted(region_set):
        assigned = _nearest_wall_color(grid, boundary, cell)
        if assigned != color:
            continue
        dr, dc = dirs[color]
        if _move_distance(region_set, cell, dr, dc) < MIN_MOVE_STEPS:
            continue
        candidates.append(cell)
    return candidates


def _far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 4


def _valid_sample(input_grid: np.ndarray) -> bool:
    try:
        out = main(input_grid)
    except Exception:
        return False
    old_out = _legacy_main(input_grid)
    if np.array_equal(out, old_out):
        return False
    if (input_grid == 4).sum() != (out == 4).sum():
        return False
    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(input_grid[struct_mask], out[struct_mask]):
        return False
    mover_info = _assigned_mover_data(input_grid)
    if len({info["color"] for info in mover_info}) < 2:
        return False
    return True


def _choose_yellows(
    rng: random.Random,
    base_grid: np.ndarray,
    legends: list[tuple[int, tuple[int, int]]],
    chamber: dict,
) -> np.ndarray | None:
    dirs = {color: direction for color, direction in legends}
    colors = [color for color, _direction in legends]
    region_set = chamber["interior"]
    boundary = chamber["boundary"]
    candidates = {
        color: _candidate_cells_for_color(base_grid, region_set, boundary, dirs, color)
        for color in colors
    }
    if any(not candidates[color] for color in colors):
        return None

    for _ in range(200):
        first = rng.choice(candidates[colors[0]])
        second = rng.choice(candidates[colors[1]])
        if first == second or not _far_enough(first, second):
            continue

        chosen = [first, second]
        input_grid = base_grid.copy()
        for r, c in chosen:
            input_grid[r, c] = 4
        if _valid_sample(input_grid):
            return input_grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    rect = _find_chamber_rect(rng, legend_occupied)
    if rect is None:
        return None

    base_grid = grid.copy()
    chamber = _build_chamber(base_grid, legends, rect, rng)
    return _choose_yellows(rng, base_grid, legends, chamber)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
