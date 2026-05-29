# Variant of 88e364bc
# Dimension: D20 — 区域间的空间关系 (obstacle)
# Change: 不同颜色墙壁可以4-邻接形成连通的迷宫结构，区域之间通过共享墙壁相邻或由空间隔开... → 区域串联（一个区域的输出边是另一个的输入边）

"""
Design analysis for the D20 variant
===================================

[a] Concrete behavior and trace
- Original task: each yellow lives in one isolated enclosed region and slides only until that
  region's far wall.
- Variant: a "region" is now a serial chamber system. A same-colored wall structure contains
  2-3 rectangular chambers aligned with the legend direction. Consecutive chambers share a wall,
  but that shared wall has a one-cell doorway. Yellow can therefore pass through the doorway and
  continue sliding into the next chamber.
- Concrete trace for a rightward chain:
  chamber A |door| chamber B |door| chamber C
  If a yellow starts on the doorway row inside chamber A, the old mental model would stop at
  A's right wall. The variant instead treats the open doorway cells as passable, so the yellow
  crosses A→B→C and finally stacks against C's far-right wall.

[b] Complexity calibration
- I keep the legend-to-wall-color mapping from the source task so the main rule is still
  "color chooses direction".
- I reduce region-shape freedom on purpose: serial interaction is the novelty, so chambers are
  clean rectangles with repeated separating walls. This makes the new relation easy to see.
- To avoid underfitting into the original task, every chain has 2-3 chambers and every chain
  contains at least one yellow whose input chamber is not the final chamber.

[c] Hidden assumptions review for the original main()
- `components_mask(passable, connectivity=4)`: in the original task this found one isolated room.
  In the variant it must be allowed to find one multi-chamber passable system. This still works.
- `if any(border-touching): continue`: still required, because only enclosed systems should move.
- Border-color majority vote: originally it identified a single room's wall color. In the variant
  it must identify the chamber system's shared wall color. The generator guarantees that all
  walls/partitions in one system use one color, so the assumption remains valid.
- `_slide_points(pset, movers, dr, dc)`: this already supports sliding through any connected
  passable cells. Doorways are just extra cells inside `pset`, so no new step logic is needed.
- The old generator relied on "one enclosed component == one region". That is false now. The new
  generator instead validates "one enclosed component == one whole chamber chain".

[d] Placement legality
- Legends stay on grid corners with a one-cell buffer.
- Each chamber chain is a closed rectangle-with-partitions placed strictly away from the border.
- Different chains get a one-cell exterior buffer so they do not merge accidentally.
- Shared walls are same-colored partitions with exactly one doorway; this preserves enclosure while
  making cross-region motion possible.

[e] Visual observability
- The chained relation is visible because the interior contains full partition walls with a single
  aligned opening. In input/output pairs, at least one yellow clearly ends beyond one of those
  partition walls.
- I generate only cardinal directions. Diagonal serial doorways are much harder to read visually
  and would add ambiguity that is unrelated to D20.

[f] Visual cleanliness
- One chain uses one wall color, one legend, and one aligned travel lane. This avoids clutter.
- Yellows are placed on the lane only, so their long travel path is easy to inspect by eye.

[g] Color-rule inferability
- Colors still only serve the original purpose: wall color ↔ border legend matching. There is no
  extra color remapping.

[h] Quantified anti-degeneration target
- Each generated chain has `n_chambers >= 2`.
- Each generated chain has at least one yellow with `input_chamber != final_chamber`.
- Validation scripts below measure crossing counts from generation metadata; if those counts were
  zero, the variant would have collapsed back into the original single-region behavior.

[i] New solver assumptions and generator guarantees
- Assumption: all walls around one connected passable chamber system share one non-background
  color. Guarantee: generator uses one color per chain.
- Assumption: doorway cells keep the passable system 4-connected. Guarantee: all partitions use
  a single aligned doorway row/column.
- Assumption: stacking happens along a straight lane with no side ambiguity. Guarantee: yellows
  are placed on the aligned doorway lane only.
"""

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

CHAINS_PER_GRID = [1, 2]
CHAMBERS_PER_CHAIN = [2, 3]
CHAMBER_SPAN_RANGE = (3, 5)
CROSS_SPAN_RANGE = (3, 5)
CHAIN_PLACEMENT_RETRIES = 120
GEN_RETRIES = 300


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend blocks on the border.

    A legend is a border-touching non-background component whose most common color is the legend
    background `C` (and also the region border color it controls). The legend contains at least
    two additional colors: a "center" color (more frequent) and an "indicator" color (less
    frequent).

    The movement direction for regions with border color `C` is:
      sign(centroid(indicator) - centroid(center))
    """
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


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
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


def _component_border_color(
    g: np.ndarray,
    pts: list[tuple[int, int]],
    bg: int,
    mover: int,
) -> int | None:
    h, w = g.shape
    pset = {(int(r), int(c)) for r, c in pts}
    border_counts: dict[int, int] = {}
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
        return None
    return max(border_counts, key=border_counts.get)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels through enclosed chamber systems using border-legend directions.

    Variant-specific interpretation:
    - A color-coded wall structure may contain multiple chambers arranged serially in the slide
      direction.
    - Shared walls between chambers may have one-cell doorways, so a yellow can continue moving
      into the next chamber instead of stopping at the first partition.

    Solver rule:
    - Find each enclosed 4-connected passable component (`bg` + yellow). In this variant that
      component may span several chained chambers.
    - Determine the system's wall color from adjacent non-passable cells.
    - Use the matching border legend to obtain the slide direction.
    - Slide all yellows across the whole connected passable system, stacking at the far end.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        border_col = _component_border_color(g, pts, bg, MOVER)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        chain_set = {(int(r), int(c)) for r, c in pts}
        for r, c in movers:
            out[r, c] = bg
        for r, c in _slide_points(chain_set, movers, dr, dc):
            out[r, c] = MOVER

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


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice(CHAINS_PER_GRID)
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    directions = rng.sample(CARDINAL_DIRS, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wc, direction in zip(corners, wall_colors, directions):
        dr, dc = direction
        patch = _legend_5x5(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, direction))
    return legend_occupied, legends


def _sort_along_direction(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]))


def _bbox_with_pad(bbox: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bbox_is_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = _bbox_with_pad(bbox, pad, h, w)
    return not occupied[r0 : r1 + 1, c0 : c1 + 1].any()


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = _bbox_with_pad(bbox, pad, h, w)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _build_chain(
    color: int,
    direction: tuple[int, int],
    r0: int,
    c0: int,
    chamber_spans: list[int],
    cross_span: int,
    door_offset: int,
) -> dict:
    dr, dc = direction
    n_chambers = len(chamber_spans)
    cell_to_chamber: dict[tuple[int, int], int] = {}
    chambers: list[set[tuple[int, int]]] = []
    partitions: list[set[tuple[int, int]]] = []
    doorways: list[tuple[int, int]] = []
    walls: set[tuple[int, int]] = set()

    if dr == 0:
        total_h = cross_span + 2
        total_w = 2 + sum(chamber_spans) + (n_chambers - 1)
        for c in range(c0, c0 + total_w):
            walls.add((r0, c))
            walls.add((r0 + total_h - 1, c))
        for r in range(r0, r0 + total_h):
            walls.add((r, c0))
            walls.add((r, c0 + total_w - 1))

        cursor = c0 + 1
        door_r = r0 + 1 + door_offset
        for idx, span in enumerate(chamber_spans):
            chamber = set()
            for r in range(r0 + 1, r0 + total_h - 1):
                for c in range(cursor, cursor + span):
                    chamber.add((r, c))
                    cell_to_chamber[(r, c)] = idx
            chambers.append(chamber)
            cursor += span
            if idx < n_chambers - 1:
                part_cells = set()
                for r in range(r0, r0 + total_h):
                    if r == door_r:
                        continue
                    walls.add((r, cursor))
                    part_cells.add((r, cursor))
                partitions.append(part_cells)
                doorways.append((door_r, cursor))
                cursor += 1

    else:
        total_h = 2 + sum(chamber_spans) + (n_chambers - 1)
        total_w = cross_span + 2
        for c in range(c0, c0 + total_w):
            walls.add((r0, c))
            walls.add((r0 + total_h - 1, c))
        for r in range(r0, r0 + total_h):
            walls.add((r, c0))
            walls.add((r, c0 + total_w - 1))

        cursor = r0 + 1
        door_c = c0 + 1 + door_offset
        for idx, span in enumerate(chamber_spans):
            chamber = set()
            for r in range(cursor, cursor + span):
                for c in range(c0 + 1, c0 + total_w - 1):
                    chamber.add((r, c))
                    cell_to_chamber[(r, c)] = idx
            chambers.append(chamber)
            cursor += span
            if idx < n_chambers - 1:
                part_cells = set()
                for c in range(c0, c0 + total_w):
                    if c == door_c:
                        continue
                    walls.add((cursor, c))
                    part_cells.add((cursor, c))
                partitions.append(part_cells)
                doorways.append((cursor, door_c))
                cursor += 1

    bbox = (r0, c0, r0 + total_h - 1, c0 + total_w - 1)
    interior = set()
    for r in range(r0 + 1, r0 + total_h - 1):
        for c in range(c0 + 1, c0 + total_w - 1):
            if (r, c) not in walls:
                interior.add((r, c))

    lane_cells = []
    if dr == 0:
        lane_r = doorways[0][0]
        for c in range(c0 + 1, c0 + total_w - 1):
            if (lane_r, c) in interior:
                lane_cells.append((lane_r, c))
    else:
        lane_c = doorways[0][1]
        for r in range(r0 + 1, r0 + total_h - 1):
            if (r, lane_c) in interior:
                lane_cells.append((r, lane_c))

    lane_sorted = _sort_along_direction(lane_cells, dr, dc)
    final_chamber = len(chambers) - 1 if (dr > 0 or dc > 0) else 0
    chamber_lane_cells = [p for p in lane_sorted if p in cell_to_chamber]
    crossing_starts = [p for p in chamber_lane_cells if cell_to_chamber[p] != final_chamber]

    return {
        "color": color,
        "dir": direction,
        "bbox": bbox,
        "walls": walls,
        "interior": interior,
        "chambers": chambers,
        "partitions": partitions,
        "doorways": doorways,
        "lane": lane_sorted,
        "cell_to_chamber": cell_to_chamber,
        "crossing_starts": crossing_starts,
        "final_chamber": final_chamber,
    }


def _try_place_chain(
    rng: random.Random,
    occupied: np.ndarray,
    color: int,
    direction: tuple[int, int],
) -> dict | None:
    h, w = occupied.shape
    for _ in range(CHAIN_PLACEMENT_RETRIES):
        n_chambers = rng.choice(CHAMBERS_PER_CHAIN)
        chamber_spans = [rng.randint(*CHAMBER_SPAN_RANGE) for _ in range(n_chambers)]
        cross_span = rng.randint(*CROSS_SPAN_RANGE)
        door_offset = rng.randrange(cross_span)

        if direction[0] == 0:
            total_h = cross_span + 2
            total_w = 2 + sum(chamber_spans) + (n_chambers - 1)
        else:
            total_h = 2 + sum(chamber_spans) + (n_chambers - 1)
            total_w = cross_span + 2

        if total_h > h - 2 or total_w > w - 2:
            continue

        r0 = rng.randint(1, h - total_h - 1)
        c0 = rng.randint(1, w - total_w - 1)
        bbox = (r0, c0, r0 + total_h - 1, c0 + total_w - 1)
        if not _bbox_is_free(occupied, bbox, pad=1):
            continue

        chain = _build_chain(color, direction, r0, c0, chamber_spans, cross_span, door_offset)
        if len(chain["crossing_starts"]) == 0:
            continue
        _mark_bbox(occupied, bbox, pad=1)
        return chain
    return None


def _validate_topology(grid: np.ndarray, chains: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    chain_sets = [chain["interior"] for chain in chains]
    if len(bounded) != len(chain_sets):
        return False
    return all(comp in chain_sets for comp in bounded)


def _place_yellows(
    rng: random.Random,
    chains: list[dict],
    grid: np.ndarray,
) -> tuple[list[dict], set[tuple[int, int]]] | None:
    metrics = []
    outputs_all: set[tuple[int, int]] = set()
    for chain in chains:
        lane = chain["lane"]
        chamber_lane_cells = [p for p in lane if p in chain["cell_to_chamber"]]
        if len(chamber_lane_cells) < 4:
            return None

        n_yellows = 2 if len(chamber_lane_cells) >= 6 and rng.random() < 0.6 else 1
        outputs = lane[-n_yellows:]
        occupied_inputs = set(outputs)

        starts = []
        first = rng.choice(chain["crossing_starts"])
        if first in occupied_inputs:
            return None
        starts.append(first)
        occupied_inputs.add(first)

        if n_yellows == 2:
            pool = [p for p in chamber_lane_cells[:-n_yellows] if p not in occupied_inputs]
            if not pool:
                return None
            starts.append(rng.choice(pool))

        for r, c in starts:
            grid[r, c] = MOVER
        outputs_all.update(outputs)

        crossings = [abs(chain["cell_to_chamber"][p] - chain["final_chamber"]) for p in starts]
        metrics.append(
            {
                "n_chambers": len(chain["chambers"]),
                "n_yellows": len(starts),
                "crossings": crossings,
            }
        )
    return metrics, outputs_all


def _oracle_check(input_grid: np.ndarray, expected_output: np.ndarray, chains: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    if not np.array_equal(solver_out, expected_output):
        return False
    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == MOVER).sum()) != int((input_grid == MOVER).sum()):
        return False
    regions_mask = np.zeros((h, w), dtype=bool)
    for chain in chains:
        for r, c in chain["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == MOVER]))


def _generate_case(rng: random.Random) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    occupied = legend_occupied.copy()

    chains = []
    for color, direction in legends:
        chain = _try_place_chain(rng, occupied, color, direction)
        if chain is None:
            return None
        chains.append(chain)

    for chain in chains:
        for r, c in chain["walls"]:
            grid[r, c] = chain["color"]

    if not _validate_topology(grid, chains):
        return None

    input_grid = grid.copy()
    placed = _place_yellows(rng, chains, input_grid)
    if placed is None:
        return None
    metrics, outputs = placed

    expected_output = grid.copy()
    for r, c in outputs:
        expected_output[r, c] = MOVER

    if not _oracle_check(input_grid, expected_output, chains):
        return None
    return input_grid, chains, metrics


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("generate_input: exceeded retries")
