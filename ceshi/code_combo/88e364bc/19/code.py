# Combo variant of 88e364bc (index 19)
# combo_spec: D17/1+D22/0
# Name: 独立墙壁颜色判定+多色移动像素

"""
Design analysis for combo D17/1 + D22/0
=======================================

spec_variant:
  preserved:
    - the grid is still 20x20 with black background
    - legend rectangles still live on the outer border and still decode wall-color -> direction
    - enclosed wall chambers still define the passable regions where colored mover pixels live
    - walls and legends remain unchanged in the output
  fused_core_rule:
    - each mover chooses its own direction from the color of its nearest wall pixel
    - mover color now also changes how far that pixel is allowed to travel
    - yellow (4): travels its full available corridor
    - orange (7): travels half of its available corridor, using floor(free_steps / 2)
    - pink (6): stays fixed as a static anchor
  concrete_visualization:
    - the generator builds one large rectangular chamber whose boundary uses two legend colors
    - one wall color occupies three sides of the chamber, and the second wall color occupies the remaining side
    - yellow is placed in the majority-color zone so it follows legend A
    - orange and pink are placed in the minority-color zone so they both match legend B
    - this makes one axis of variation visible across wall proximity, and the second axis visible across mover color
  traced_example:
    - suppose the chamber has grey on top/left/bottom and teal on the right wall
    - a yellow on the left half sees grey as nearest, so it follows the grey legend direction and slides fully
    - an orange near the right wall sees teal as nearest, so it follows the teal legend direction but only moves half way
    - a pink near the same right wall also sees teal, but it does not move at all
    - the source task would have assigned one shared region color, so all movers would have used the grey direction
    - the single-color D17 variant would still miss the orange-half-distance and pink-anchor behavior
    - the single-direction D22 variant would still miss that the orange and yellow can get different directions inside one chamber
  complexity_calibration:
    - exactly two legend colors are used, which is enough to expose independent wall matching without creating noisy color maps
    - exactly three mover colors are used, and every generated sample contains all three so their roles repeat across examples
    - legends are restricted to cardinal directions to keep the fusion readable; adding diagonals on top of both new axes would make 2-3 examples harder to parse
    - the chamber boundary uses long solid sides rather than per-cell alternation, so nearest-wall regions are visually inferable by humans
  implicit_assumptions_audit:
    - original `mover = 4` is invalid: passability and repainting must now include three mover colors
    - original region-level `border_counts -> max` is invalid: a single chamber can no longer be summarized by one wall color
    - original `_slide_points(pset, movers, dr, dc)` is invalid twice over:
      - movers no longer share one direction
      - movers no longer share one travel length rule
    - computing directions after clearing movers would be unsafe if a later rule depended on current pixel positions, so the solver resolves nearest walls and travel budgets before motion
    - mixed directions can make simple far-to-near ordering brittle, so the fused solver uses repeated one-step simulation with a fixed direction and a per-mover step budget
  placement_legality:
    - legends are pasted first, and the chamber search reserves a one-cell buffer around them
    - the chamber never touches the outer border, so its interior is a proper enclosed region
    - every wall color on the chamber boundary comes from one of the two legend fill colors, so every mover has a valid legend lookup
    - mover candidates are filtered to avoid ambiguous nearest-wall ties, out-of-bounds motion, and obvious path collisions
  visual_observability:
    - yellow and orange start in the same chamber but in different nearest-wall zones, so the independent wall-color rule is visible
    - orange and pink start in the same nearest-wall zone but behave differently, so the mover-color rule is also visible
    - generated samples are rejected unless the fused solver differs from a legacy region-majority multicolor solver
  cleanliness_and_inferability:
    - the scene contains one chamber, two legend colors, and three mover pixels; this is dense enough to show the rule but not visually cluttered
    - the generator avoids ambiguous centerline cells where a human would have trouble deciding which wall is "nearest"
    - orange is only accepted when it has a strictly partial move, and pink is only accepted when it has room available but still stays put
  new_solver_assumptions_checked:
    - nearest-wall lookup assumes a bounded region has at least one adjacent wall pixel; chamber construction guarantees that
    - the synchronous stepper assumes there are no pathological head-on cycles that never resolve; the generator rejects candidate layouts with intersecting active paths
    - orange travel uses a precomputed budget from its wall-determined direction and static corridor length; the generator avoids layouts where another moving pixel would need to redefine that budget to remain interpretable
"""

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 8, 9]
LEGEND_BUFFER = 1

YELLOW = 4
PINK = 6
ORANGE = 7
MOVER_COLORS = (YELLOW, PINK, ORANGE)
ACTIVE_MOVER_COLORS = (YELLOW, ORANGE)

GEN_RETRIES = 400
CHAMBER_OUTER_H = range(9, 12)
CHAMBER_OUTER_W = range(9, 12)
MIN_START_SEP = 3


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
        if len(others) != 2:
            continue
        others.sort()
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
        indicator = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indicator:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indicator, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


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


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _boundary_wall_cells(g: np.ndarray, region_set: set[tuple[int, int]], bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    boundary = set()
    for r, c in region_set:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in region_set:
                continue
            v = int(g[nr, nc])
            if v == bg or v in MOVER_COLORS:
                continue
            boundary.add((nr, nc))
    return sorted(boundary)


def _nearest_wall_details(
    g: np.ndarray, boundary: list[tuple[int, int]], pos: tuple[int, int]
) -> tuple[int | None, bool]:
    if not boundary:
        return None, False
    r0, c0 = pos
    scored = sorted((abs(wr - r0) + abs(wc - c0), wr, wc, int(g[wr, wc])) for wr, wc in boundary)
    best_dist = scored[0][0]
    tied_colors = {color for dist, _wr, _wc, color in scored if dist == best_dist}
    return scored[0][3], len(tied_colors) > 1


def _static_path(region_set: set[tuple[int, int]], start: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    r, c = start
    path: list[tuple[int, int]] = []
    while _slide_step_valid(region_set, r, c, dr, dc):
        r += dr
        c += dc
        path.append((r, c))
    return path


def _move_budget(color: int, free_steps: int) -> int:
    if color == YELLOW:
        return free_steps
    if color == ORANGE:
        return free_steps // 2
    if color == PINK:
        return 0
    raise ValueError(f"unexpected mover color {color}")


def _simulate_individual_colored_slides(
    region_set: set[tuple[int, int]], movers: list[dict]
) -> list[tuple[int, int, int]]:
    if not movers:
        return []

    positions = [m["pos"] for m in movers]
    dirs = [m["dir"] for m in movers]
    colors = [m["color"] for m in movers]
    remaining = [m["budget"] for m in movers]
    max_iters = max(1, sum(remaining) + len(region_set))

    for _ in range(max_iters):
        current = set(positions)
        proposals: list[tuple[int, int]] = []
        target_counts: dict[tuple[int, int], int] = {}

        for (r, c), (dr, dc), steps_left in zip(positions, dirs, remaining):
            if steps_left <= 0 or not _slide_step_valid(region_set, r, c, dr, dc):
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
        for i, (pos, proposal) in enumerate(zip(positions, proposals)):
            if proposal != pos and target_counts.get(proposal, 0) == 1:
                new_positions.append(proposal)
                remaining[i] -= 1
                moved = True
            else:
                new_positions.append(pos)
        positions = new_positions
        if not moved:
            break

    return [(r, c, color) for (r, c), color in zip(positions, colors)]


def _region_majority_color(
    g: np.ndarray, pts: np.ndarray, region_set: set[tuple[int, int]], bg: int
) -> int | None:
    h, w = g.shape
    border_counts: dict[int, int] = {}
    for r, c in pts:
        for nr, nc in neighbors4(int(r), int(c)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in region_set:
                continue
            v = int(g[nr, nc])
            if v == bg or v in MOVER_COLORS:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None
    return max(border_counts, key=border_counts.get)


def _legacy_region_majority_main(grid: np.ndarray) -> np.ndarray:
    """Baseline used only by the generator to reject D17-degenerate samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    out = g.copy()

    passable = g == bg
    for color in MOVER_COLORS:
        passable |= g == color

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c), int(g[r, c])) for r, c in pts if int(g[r, c]) in MOVER_COLORS]
        if not movers:
            continue

        border_col = _region_majority_color(g, pts, region_set, bg)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        active = []
        for r, c, color in movers:
            free_steps = len(_static_path(region_set, (r, c), dr, dc))
            active.append({"pos": (r, c), "color": color, "dir": (dr, dc), "budget": _move_budget(color, free_steps)})

        for r, c, _color in movers:
            out[r, c] = bg
        for r, c, color in _simulate_individual_colored_slides(region_set, active):
            out[r, c] = color

    return out


def _analyze_movers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)

    passable = g == bg
    for color in MOVER_COLORS:
        passable |= g == color

    info: list[dict] = []
    for region_id, pts in enumerate(components_mask(passable, connectivity=4)):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        region_set = {(int(r), int(c)) for r, c in pts}
        boundary = _boundary_wall_cells(g, region_set, bg)
        majority_color = _region_majority_color(g, pts, region_set, bg)
        for r, c in pts:
            color = int(g[r, c])
            if color not in MOVER_COLORS:
                continue
            pos = (int(r), int(c))
            wall_color, ambiguous = _nearest_wall_details(g, boundary, pos)
            direction = dirs.get(wall_color) if wall_color in dirs else None
            free_steps = 0
            budget = 0
            if direction is not None:
                free_steps = len(_static_path(region_set, pos, direction[0], direction[1]))
                budget = _move_budget(color, free_steps)
            info.append(
                {
                    "region_id": region_id,
                    "pos": pos,
                    "color": color,
                    "wall_color": wall_color,
                    "ambiguous": ambiguous,
                    "dir": direction,
                    "free_steps": free_steps,
                    "budget": budget,
                    "majority_color": majority_color,
                }
            )
    return info


def _fusion_metrics(input_grid: np.ndarray) -> dict:
    mover_info = _analyze_movers(input_grid)
    out = main(input_grid)
    legacy_out = _legacy_region_majority_main(input_grid)

    by_color = {info["color"]: info for info in mover_info}
    yellow_in = tuple(map(tuple, np.argwhere(input_grid == YELLOW)))
    yellow_out = tuple(map(tuple, np.argwhere(out == YELLOW)))
    orange_in = tuple(map(tuple, np.argwhere(input_grid == ORANGE)))
    orange_out = tuple(map(tuple, np.argwhere(out == ORANGE)))
    pink_in = tuple(map(tuple, np.argwhere(input_grid == PINK)))
    pink_out = tuple(map(tuple, np.argwhere(out == PINK)))

    distinct_wall_colors = {info["wall_color"] for info in mover_info if info["wall_color"] is not None}
    distinct_active_wall_colors = {info["wall_color"] for info in mover_info if info["color"] in ACTIVE_MOVER_COLORS}
    return {
        "mover_info": mover_info,
        "all_unambiguous": all(not info["ambiguous"] for info in mover_info),
        "distinct_wall_colors": len(distinct_wall_colors),
        "distinct_active_wall_colors": len(distinct_active_wall_colors),
        "orange_partial": ORANGE in by_color and 0 < by_color[ORANGE]["budget"] < by_color[ORANGE]["free_steps"],
        "pink_room_but_static": PINK in by_color and by_color[PINK]["free_steps"] >= 2 and pink_in == pink_out,
        "yellow_moved": yellow_in != yellow_out,
        "orange_moved": orange_in != orange_out,
        "legacy_diff": not np.array_equal(out, legacy_out),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-colored mover pixels inside walled regions using legend directions chosen per pixel.

    Input structure:
    - Black is background; enclosed chambers are bounded by colored walls that match border legends.
    - Legends are solid border rectangles whose fill color identifies a wall color, while the center and red indicator encode a direction.
    - Movers may be yellow, orange, or pink, and multiple mover colors can appear inside the same enclosed chamber.

    Rule:
    - For each mover pixel, inspect the nearest wall pixel around its own enclosed chamber and use that wall color to select a legend direction.
    - Yellow then spends its entire free corridor in that direction, orange spends half of that corridor, and pink stays where it is.
    - All movers advance by repeated one-cell steps with those fixed per-pixel directions until their own travel budget or blocking condition stops them.

    Key insight: Direction is no longer a property of the whole region; it is chosen independently per mover from the nearest wall color, while the mover's own color separately encodes how much of that direction-specific corridor it is allowed to use.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = g == bg
    for color in MOVER_COLORS:
        passable |= g == color

    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c), int(g[r, c])) for r, c in pts if int(g[r, c]) in MOVER_COLORS]
        if not movers:
            continue

        boundary = _boundary_wall_cells(g, region_set, bg)
        active = []
        for r, c, color in movers:
            wall_color, _ambiguous = _nearest_wall_details(g, boundary, (r, c))
            if wall_color is None or wall_color not in dirs:
                continue
            dr, dc = dirs[wall_color]
            free_steps = len(_static_path(region_set, (r, c), dr, dc))
            active.append(
                {
                    "pos": (r, c),
                    "color": color,
                    "dir": (dr, dc),
                    "budget": _move_budget(color, free_steps),
                }
            )

        for r, c, _color in movers:
            out[r, c] = bg
        for r, c, color in _simulate_individual_colored_slides(region_set, active):
            out[r, c] = color

    return out


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    corners = rng.sample(CORNERS, 2)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, 2)
    directions = rng.sample(CARDINAL_DIRS, 2)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wc, direction in zip(corners, wall_colors, directions):
        patch = _legend_5x5(wc, direction[0], direction[1])
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, direction))
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
    return {
        "interior": interior,
        "boundary": sorted(wall_color_at),
        "majority_color": majority_color,
        "minority_color": minority_color,
        "minority_side": minority_side,
    }


def _candidate_cells(
    base_grid: np.ndarray,
    chamber: dict,
    assigned_wall_color: int,
    mover_color: int,
    dirs: dict[int, tuple[int, int]],
) -> list[dict]:
    region_set = chamber["interior"]
    boundary = chamber["boundary"]
    direction = dirs[assigned_wall_color]
    out = []
    for cell in sorted(region_set):
        wall_color, ambiguous = _nearest_wall_details(base_grid, boundary, cell)
        if ambiguous or wall_color != assigned_wall_color:
            continue
        free_steps = len(_static_path(region_set, cell, direction[0], direction[1]))
        budget = _move_budget(mover_color, free_steps)
        if mover_color == YELLOW and free_steps < 3:
            continue
        if mover_color == ORANGE and not (0 < budget < free_steps):
            continue
        if mover_color == PINK and free_steps < 2:
            continue
        out.append({"pos": cell, "dir": direction, "free_steps": free_steps, "budget": budget})
    return out


def _far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) >= MIN_START_SEP


def _paths_compatible(
    region_set: set[tuple[int, int]],
    yellow: dict,
    orange: dict,
    pink: dict,
) -> bool:
    yellow_path = set(_static_path(region_set, yellow["pos"], yellow["dir"][0], yellow["dir"][1]))
    orange_path = set(_static_path(region_set, orange["pos"], orange["dir"][0], orange["dir"][1]))

    if yellow["pos"] in orange_path or orange["pos"] in yellow_path:
        return False
    if pink["pos"] in yellow_path or pink["pos"] in orange_path:
        return False
    if yellow_path & orange_path:
        return False
    return True


def _valid_sample(input_grid: np.ndarray, chamber: dict) -> bool:
    try:
        out = main(input_grid)
    except Exception:
        return False

    mover_mask = np.isin(input_grid, list(MOVER_COLORS))
    struct_mask = (~mover_mask) & (input_grid != 0)
    if not np.array_equal(input_grid[struct_mask], out[struct_mask]):
        return False

    for color in MOVER_COLORS:
        if int((input_grid == color).sum()) != int((out == color).sum()):
            return False

    regions_mask = np.zeros(input_grid.shape, dtype=bool)
    for r, c in chamber["interior"]:
        regions_mask[r, c] = True
    if not bool(np.all(regions_mask[np.isin(out, list(MOVER_COLORS))])):
        return False

    metrics = _fusion_metrics(input_grid)
    return (
        len(metrics["mover_info"]) == 3
        and metrics["all_unambiguous"]
        and metrics["distinct_wall_colors"] == 2
        and metrics["distinct_active_wall_colors"] == 2
        and metrics["yellow_moved"]
        and metrics["orange_moved"]
        and metrics["orange_partial"]
        and metrics["pink_room_but_static"]
        and metrics["legacy_diff"]
    )


def _choose_movers(
    rng: random.Random,
    base_grid: np.ndarray,
    legends: list[tuple[int, tuple[int, int]]],
    chamber: dict,
) -> np.ndarray | None:
    dirs = {color: direction for color, direction in legends}
    yellow_candidates = _candidate_cells(base_grid, chamber, chamber["majority_color"], YELLOW, dirs)
    orange_candidates = _candidate_cells(base_grid, chamber, chamber["minority_color"], ORANGE, dirs)
    pink_candidates = _candidate_cells(base_grid, chamber, chamber["minority_color"], PINK, dirs)

    if not yellow_candidates or not orange_candidates or not pink_candidates:
        return None

    rng.shuffle(yellow_candidates)
    rng.shuffle(orange_candidates)
    rng.shuffle(pink_candidates)
    region_set = chamber["interior"]

    for orange in orange_candidates[:40]:
        for pink in pink_candidates[:40]:
            if pink["pos"] == orange["pos"]:
                continue
            if not _far_enough(pink["pos"], orange["pos"]):
                continue
            for yellow in yellow_candidates[:60]:
                if yellow["pos"] in {orange["pos"], pink["pos"]}:
                    continue
                if not _far_enough(yellow["pos"], orange["pos"]):
                    continue
                if not _far_enough(yellow["pos"], pink["pos"]):
                    continue
                if not _paths_compatible(region_set, yellow, orange, pink):
                    continue

                input_grid = base_grid.copy()
                input_grid[yellow["pos"]] = YELLOW
                input_grid[orange["pos"]] = ORANGE
                input_grid[pink["pos"]] = PINK
                if _valid_sample(input_grid, chamber):
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
    return _choose_movers(rng, base_grid, legends, chamber)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
