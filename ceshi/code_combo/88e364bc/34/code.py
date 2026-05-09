# Combo variant of 88e364bc (index 34)
# combo_spec: D2/0+D20/1
# Name: 区域串联+面积排序匹配

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
for _path in (_ROOT, _ROOT / "solvers"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.append(_path_str)

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


"""
Design analysis for the combo variant
=====================================

a_concretization_and_behavior_trace:
  preserved_shell:
    - 20x20 black background, border legends, colored wall chambers, yellow movers.
    - Legends still encode cardinal directions through the original center/indicator geometry.
  fused_change:
    - The movable empty space is no longer two isolated rooms. It is one enclosed serial system:
      a large chamber and a small chamber are linked by a tiny elbow corridor, so a yellow can
      leave the first room and continue inside the second.
    - The two chambers do not read legends by wall color. Instead, all chambers across the grid
      are sorted by area descending, and the kth largest chamber uses the kth legend in border
      reading order.
    - The serial system is therefore a two-stage pipe: chamber_big uses legend_1, chamber_small
      uses legend_2, and the yellow may visibly turn at the doorway.
  concrete_example:
    - Legend order by position: L1 = right, L2 = down.
    - Chamber areas: A = 24 cells, B = 12 cells, so A gets right and B gets down.
    - A yellow starts near the left side of A. Under the source task it would stay inside one room.
    - Under the pure serial-topology variant with one wall-color direction, it would use one
      direction for the whole connected component.
    - Under this combo it first moves right across A, passes through the elbow corridor, then
      switches to B's direction and continues downward inside B.
    - The chambers' wall colors are intentionally swapped against legend order, so color matching
      would predict the opposite assignment.

b_complexity_calibration:
  - I fix the serial structure to exactly two rectangular chambers plus one doorway. That is the
    smallest layout where both "cross-region continuation" and "area-ranked direction reassignment"
    are simultaneously visible.
  - I use only cardinal directions and require the two directions to be orthogonal. This keeps the
    chamber turn human-readable and avoids ambiguous diagonal doorway geometry.
  - I keep only one yellow per grid. The novelty here is the chamber-to-chamber pipeline, not
    multi-yellow collision bookkeeping.
  - The larger chamber is noticeably larger than the smaller one, so the area ranking is visually
    stable over just a few examples.

c_implicit_assumptions_review_of_main:
  - Original broken assumption:
    `components_mask(passable, connectivity=4)` used to mean "one enclosed component == one room".
    Here one enclosed component is a whole serial system, so main() must split that component at
    its articulation doorway before assigning directions.
  - Original broken assumption:
    `border_col -> legend direction` is no longer the control key. Chamber border colors remain
    visible but are now deliberately misleading.
  - Original retained assumption:
    border legends are still border-touching solid rectangles with the same 3-color centroid code.
  - New main() assumptions:
    1. each serial system contains exactly two full rectangular chambers plus a tiny connector path;
    2. each chamber is large enough that the doorway split is unambiguous;
    3. legend count equals chamber count on legal inputs;
    4. at most one yellow is placed in each serial system, so independent tracing is sufficient.

d_placement_legality:
  - Legends occupy two distinct corners and reserve a one-cell buffer, so they cannot merge with
    walls into the same border component.
    - Each chamber is an axis-aligned rectangle with a one-cell-thick wall ring. The connector
      path is the only missing wall opening.
  - The large chamber sits on the side opposite its assigned motion vector; the small chamber sits
    on the side reached after passing through the doorway along its own vector.
  - Generator checks that wall rings do not overlap, the whole serial system stays away from the
    outer border, and the bounded-background topology equals "big chamber + doorway + small chamber".

e_visual_observability:
  - Humans can see one large room and one small room joined by a single black doorway, so the
    serial relation is explicit in the input geometry.
  - The larger room always acts first because the legends are paired by room size, not wall color.
  - The yellow ends beyond the doorway in the second chamber, making the cross-region continuation
    visible in the output rather than hidden inside solver-only logic.

f_visual_cleanliness:
  - The serial system is limited to one turn and two chambers, which keeps the grid sparse.
  - Each chamber uses one wall color, the legends use the original compact 5x5 motif, and the
    yellow stays a single pixel, so the scene reads cleanly.
  - The connector is only two empty cells, which still reads as one tiny pipe elbow.

g_inferability_of_change:
  - No invisible quantity is used in the ranking. Area is directly visible from chamber size.
  - Wall colors remain few and structural. They distinguish chambers, but they do not solve the
    mapping problem by themselves because the generator swaps them against the legend order.
  - The combo can be inferred from 2-4 examples because the same visual pattern repeats:
    "bigger chamber uses the earlier legend, then the mover hands off through the doorway".

h_generator_non_degeneracy_metric:
  - Metric 1: the yellow must start in the larger chamber and finish in the smaller chamber, so the
    serial continuation is active rather than decorative.
  - Metric 2: compare combo output against a chamber-aware but color-matched baseline. Accepted
    samples require `old_new_diff_cells > 0`.
  - Metric 3: the chamber wall colors must be fully swapped against the area ranking, so the D2
    change is actually needed.

i_post_edit_assumption_audit:
  - main() assumes the two chambers are recoverable as the two largest full passable rectangles;
    generate_input() constructs exactly that pattern.
  - main() assumes orthogonal turn geometry from chamber 1 to chamber 2; generator samples only
    orthogonal direction pairs.
  - main() assumes chamber areas are strictly ordered; generator enforces an area gap threshold.
  - main() assumes a mover never starts on the doorway; generator only samples starts from the
    larger chamber interior.
"""


GRID_H, GRID_W = 20, 20
MOVER = 4

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

ORTHOGONAL_PAIRS = [
    (d1, d2) for d1 in CARDINAL_DIRS for d2 in CARDINAL_DIRS if d1[0] * d2[0] + d1[1] * d2[1] == 0
]

BIG_HEIGHT_RANGE = (4, 6)
BIG_WIDTH_RANGE = (4, 6)
SMALL_HEIGHT_RANGE = (3, 4)
SMALL_WIDTH_RANGE = (3, 4)
AREA_GAP_MIN = 8

SERIAL_PLACEMENT_RETRIES = 200
GEN_RETRIES = 500
CONNECTOR_LEN = 3


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


def _bbox_key(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def _extract_legends(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    legends = []
    for pts in components_mask(g != bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            candidates = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                candidates.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            candidates.sort()
            center_col = candidates[0][1]
            indicator_col = candidates[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue
        vr, vc = (indicator.mean(axis=0) - center.mean(axis=0)).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if (dr, dc) not in CARDINAL_DIRS:
            continue
        legends.append(
            {
                "fill": int(fill),
                "dir": (dr, dc),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    legends.sort(key=lambda item: _bbox_key(item["bbox"]))
    return legends


def _subset_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _largest_rectangles(component: set[tuple[int, int]], min_area: int = 6) -> list[set[tuple[int, int]]]:
    if not component:
        return []
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=bool)
    for r, c in component:
        mask[r - rmin, c - cmin] = True

    candidates: list[tuple[int, set[tuple[int, int]]]] = []
    h, w = mask.shape
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                row_slice = mask[r0 : r1 + 1]
                if not row_slice[:, c0].all():
                    break
                for c1 in range(c0, w):
                    rect_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
                    if not rect_mask.all():
                        break
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < min_area:
                        continue
                    cells = {
                        (rmin + rr, cmin + cc)
                        for rr in range(r0, r1 + 1)
                        for cc in range(c0, c1 + 1)
                    }
                    candidates.append((area, cells))

    candidates.sort(key=lambda item: item[0], reverse=True)
    picked: list[set[tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()
    for _, cells in candidates:
        if cells & used:
            continue
        picked.append(cells)
        used |= cells
        if len(picked) == 2:
            break
    return picked


def _border_color_for_cells(g: np.ndarray, cells: set[tuple[int, int]], bg: int) -> int | None:
    h, w = g.shape
    counts: dict[int, int] = {}
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cells:
                continue
            v = int(g[nr, nc])
            if v == bg or v == MOVER:
                continue
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _extract_serial_systems(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    passable = (g == bg) | (g == MOVER)
    systems = []

    for sys_idx, pts in enumerate(components_mask(passable, connectivity=4)):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        comp = {(int(r), int(c)) for r, c in pts}
        chamber_sets = _largest_rectangles(comp, min_area=6)
        if len(chamber_sets) != 2:
            continue
        corridor = set(comp)
        for cells in chamber_sets:
            corridor -= cells
        if not (1 <= len(corridor) <= CONNECTOR_LEN):
            continue
        corridor_comps = _subset_components(corridor)
        if len(corridor_comps) != 1:
            continue

        chamber_infos = []
        cell_to_chamber: dict[tuple[int, int], str] = {}
        for chamber_idx, cells in enumerate(chamber_sets):
            cells = set(cells)
            bbox = bbox_coords(np.array(sorted(cells), dtype=int))
            chamber_id = f"S{sys_idx}_C{chamber_idx}"
            info = {
                "id": chamber_id,
                "cells": cells,
                "area": len(cells),
                "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                "movers": [(r, c) for r, c in cells if int(g[r, c]) == MOVER],
                "border_color": _border_color_for_cells(g, cells, bg),
            }
            chamber_infos.append(info)
            for cell in cells:
                cell_to_chamber[cell] = chamber_id

        chamber_neighbors: dict[str, set[tuple[int, int]]] = {}
        for chamber in chamber_infos:
            neigh = set()
            for cell in chamber["cells"]:
                for nr, nc in neighbors4(*cell):
                    if (nr, nc) in corridor:
                        neigh.add((nr, nc))
            chamber_neighbors[chamber["id"]] = neigh
        if any(len(neigh) == 0 for neigh in chamber_neighbors.values()):
            continue

        systems.append(
            {
                "id": f"S{sys_idx}",
                "component": comp,
                "corridor": corridor,
                "chambers": chamber_infos,
                "cell_to_chamber": cell_to_chamber,
                "chamber_neighbors": chamber_neighbors,
            }
        )

    return systems


def _trace_serial_mover(
    system: dict,
    start: tuple[int, int],
    chamber_dirs: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    corridor = system["corridor"]
    cell_to_chamber = system["cell_to_chamber"]

    cur = start
    prev = None
    max_steps = len(system["component"]) + 5

    for _ in range(max_steps):
        if cur in corridor:
            options = []
            for nxt in neighbors4(*cur):
                if nxt == prev or nxt not in system["component"]:
                    continue
                options.append(nxt)
            if len(options) != 1:
                break
            prev, cur = cur, options[0]
            continue

        active = cell_to_chamber[cur]

        dr, dc = chamber_dirs.get(active, (0, 0))
        if (dr, dc) == (0, 0):
            break
        nxt = (cur[0] + dr, cur[1] + dc)
        if nxt in corridor:
            prev, cur = cur, nxt
            continue
        if cell_to_chamber.get(nxt) == active:
            prev, cur = cur, nxt
            continue
        break

    return cur


def _collect_chambers_by_area(systems: list[dict]) -> list[dict]:
    chambers = []
    for system in systems:
        chambers.extend(system["chambers"])
    chambers.sort(key=lambda item: (-item["area"],) + _bbox_key(item["bbox"]))
    return chambers


def _area_rank_dirs(systems: list[dict], legends: list[dict]) -> dict[str, tuple[int, int]]:
    dirs: dict[str, tuple[int, int]] = {}
    for chamber, legend in zip(_collect_chambers_by_area(systems), legends):
        dirs[chamber["id"]] = legend["dir"]
    return dirs


def _color_match_dirs(systems: list[dict], legends: list[dict]) -> dict[str, tuple[int, int]]:
    lookup = {legend["fill"]: legend["dir"] for legend in legends}
    dirs: dict[str, tuple[int, int]] = {}
    for system in systems:
        for chamber in system["chambers"]:
            border_color = chamber["border_color"]
            if border_color in lookup:
                dirs[chamber["id"]] = lookup[border_color]
    return dirs


def _run_solver_with_dirs(g: np.ndarray, bg: int, systems: list[dict], chamber_dirs: dict[str, tuple[int, int]]) -> np.ndarray:
    out = g.copy()
    for system in systems:
        movers = []
        for chamber in system["chambers"]:
            movers.extend(chamber["movers"])
        if not movers:
            continue

        for r, c in movers:
            out[r, c] = bg

        final_cells = []
        for start in movers:
            final_cells.append(_trace_serial_mover(system, start, chamber_dirs))
        for r, c in final_cells:
            out[r, c] = MOVER

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels through a serial pair of enclosed chambers whose directions come from ordered legends.

    Input structure:
    - Border legends are still small solid rectangles on the outer edge, each with a fill color and
      a center/indicator pattern that encodes one cardinal direction.
    - Inside the grid, the passable black space forms one enclosed serial system: a larger chamber
      and a smaller chamber linked by a tiny elbow corridor.
    - Chamber wall colors are visible but are not the matching key in this combo.

    Rule:
    - Detect the border legends and sort them by the reading order of their bounding boxes.
    - Recover the two chambers inside each enclosed serial system as the two largest full passable rectangles.
    - Sort all chambers by descending area; the largest chamber uses the first legend, the next chamber
      uses the second legend, and so on.
    - A yellow moves inside its current chamber using that chamber's direction; if it reaches the connector
      path, it traverses that path and then follows the next chamber's direction.

    Key insight:
    The connected empty space must be decomposed into serial chambers first; only after that does the
    visible size ranking tell you which legend controls each movement stage.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legends = _extract_legends(g, bg)
    assert legends, "expected at least one border legend"

    systems = _extract_serial_systems(g, bg)
    if not systems:
        return g.copy()

    chamber_dirs = _area_rank_dirs(systems, legends)
    return _run_solver_with_dirs(g, bg, systems, chamber_dirs)


def _old_color_chain_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legends = _extract_legends(g, bg)
    systems = _extract_serial_systems(g, bg)
    chamber_dirs = _color_match_dirs(systems, legends)
    return _run_solver_with_dirs(g, bg, systems, chamber_dirs)


def _rect_from_door(
    door: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
    offset: int,
    role: str,
) -> tuple[int, int, int, int]:
    r, c = door
    dr, dc = direction

    if dr == 0 and dc == 1:
        r0 = r - offset
        r1 = r0 + height - 1
        if role == "prev":
            c0, c1 = c - width, c - 1
        else:
            c0, c1 = c + 1, c + width
    elif dr == 0 and dc == -1:
        r0 = r - offset
        r1 = r0 + height - 1
        if role == "prev":
            c0, c1 = c + 1, c + width
        else:
            c0, c1 = c - width, c - 1
    elif dr == 1 and dc == 0:
        c0 = c - offset
        c1 = c0 + width - 1
        if role == "prev":
            r0, r1 = r - height, r - 1
        else:
            r0, r1 = r + 1, r + height
    elif dr == -1 and dc == 0:
        c0 = c - offset
        c1 = c0 + width - 1
        if role == "prev":
            r0, r1 = r + 1, r + height
        else:
            r0, r1 = r - height, r - 1
    else:
        raise ValueError(direction)

    return int(r0), int(c0), int(r1), int(c1)


def _rect_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _ring_cells(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = set()
    for c in range(c0 - 1, c1 + 2):
        cells.add((r0 - 1, c))
        cells.add((r1 + 1, c))
    for r in range(r0 - 1, r1 + 2):
        cells.add((r, c0 - 1))
        cells.add((r, c1 + 1))
    return cells


def _rect_fits(rect: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect
    return 1 < r0 and 1 < c0 and r1 < h - 2 and c1 < w - 2


def _build_serial_layout(
    elbow: tuple[int, int],
    big_dir: tuple[int, int],
    small_dir: tuple[int, int],
    big_hw: tuple[int, int],
    small_hw: tuple[int, int],
    big_offset: int,
    small_offset: int,
    colors: tuple[int, int],
    legend_mask: np.ndarray,
) -> dict | None:
    big_h, big_w = big_hw
    small_h, small_w = small_hw
    c1 = elbow
    corridor = [(c1[0] + k * small_dir[0], c1[1] + k * small_dir[1]) for k in range(CONNECTOR_LEN)]
    c_entry = corridor[-1]
    big_rect = _rect_from_door(c1, big_dir, big_h, big_w, big_offset, role="prev")
    small_rect = _rect_from_door(c_entry, small_dir, small_h, small_w, small_offset, role="next")

    if not _rect_fits(big_rect, GRID_H, GRID_W) or not _rect_fits(small_rect, GRID_H, GRID_W):
        return None

    big_cells = _rect_cells(big_rect)
    small_cells = _rect_cells(small_rect)
    if big_cells & small_cells:
        return None

    big_ring = _ring_cells(big_rect)
    small_ring = _ring_cells(small_rect)
    corridor_set = set(corridor)
    big_ring -= corridor_set
    small_ring -= corridor_set
    if corridor_set & big_cells or corridor_set & small_cells:
        return None
    if big_ring & big_cells or small_ring & small_cells:
        return None
    if big_ring & small_cells or small_ring & big_cells:
        return None
    if big_ring & small_ring:
        return None

    all_used = big_cells | small_cells | big_ring | small_ring | corridor_set
    if any(r < 0 or c < 0 or r >= GRID_H or c >= GRID_W for r, c in all_used):
        return None
    if any(legend_mask[r, c] for r, c in all_used):
        return None

    connector_walls = set()
    for cell in corridor_set:
        for nbr in neighbors4(*cell):
            if nbr in corridor_set or nbr in big_cells or nbr in small_cells:
                continue
            connector_walls.add(nbr)
    if connector_walls & big_cells or connector_walls & small_cells:
        return None
    if any(r < 0 or c < 0 or r >= GRID_H or c >= GRID_W for r, c in connector_walls):
        return None
    if any(legend_mask[r, c] for r, c in connector_walls):
        return None

    big_adj = [p for p in neighbors4(*c1) if p in big_cells]
    small_adj = [p for p in neighbors4(*c_entry) if p in small_cells]
    if len(big_adj) != 1 or len(small_adj) != 1:
        return None

    return {
        "corridor": corridor_set,
        "connector_walls": connector_walls,
        "big": {"cells": big_cells, "ring": big_ring, "rect": big_rect, "color": colors[0], "dir": big_dir},
        "small": {"cells": small_cells, "ring": small_ring, "rect": small_rect, "color": colors[1], "dir": small_dir},
    }


def _place_legends(rng: random.Random, grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    corners = rng.sample(CORNERS, 2)
    fills = rng.sample(WALL_COLOR_CHOICES, 2)
    dirs = list(rng.choice(ORTHOGONAL_PAIRS))

    legend_mask = np.zeros(grid.shape, dtype=bool)
    legends = []
    for corner, fill, direction in zip(corners, fills, dirs):
        patch = _legend_5x5(fill, *direction)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, GRID_H, GRID_W, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(GRID_H, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(GRID_W, c0 + pw + LEGEND_BUFFER)
        legend_mask[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"fill": fill, "dir": direction, "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1)})

    legends.sort(key=lambda item: _bbox_key(item["bbox"]))
    return legend_mask, legends


def _try_place_serial_system(rng: random.Random, legends: list[dict], legend_mask: np.ndarray) -> dict | None:
    big_dir = legends[0]["dir"]
    small_dir = legends[1]["dir"]
    if big_dir[0] * small_dir[0] + big_dir[1] * small_dir[1] != 0:
        return None

    # Swap the wall colors against the area rank to make color matching fail.
    colors = (legends[1]["fill"], legends[0]["fill"])

    for _ in range(SERIAL_PLACEMENT_RETRIES):
        big_hw = (rng.randint(*BIG_HEIGHT_RANGE), rng.randint(*BIG_WIDTH_RANGE))
        small_hw = (rng.randint(*SMALL_HEIGHT_RANGE), rng.randint(*SMALL_WIDTH_RANGE))
        if big_hw[0] * big_hw[1] - small_hw[0] * small_hw[1] < AREA_GAP_MIN:
            continue

        big_offset = rng.randrange(big_hw[0] if big_dir[0] == 0 else big_hw[1])
        small_offset = rng.randrange(small_hw[0] if small_dir[0] == 0 else small_hw[1])
        elbow = (rng.randint(4, GRID_H - 5), rng.randint(4, GRID_W - 5))

        layout = _build_serial_layout(
            elbow=elbow,
            big_dir=big_dir,
            small_dir=small_dir,
            big_hw=big_hw,
            small_hw=small_hw,
            big_offset=big_offset,
            small_offset=small_offset,
            colors=colors,
            legend_mask=legend_mask,
        )
        if layout is not None:
            return layout
    return None


def _validate_topology(grid: np.ndarray, layout: dict) -> bool:
    expected = set(layout["big"]["cells"]) | set(layout["small"]["cells"]) | set(layout["corridor"])
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, GRID_H - 1) or c in (0, GRID_W - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    return len(bounded) == 1 and bounded[0] == expected


def _place_yellow(rng: random.Random, grid: np.ndarray, layout: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
    big_cells = list(layout["big"]["cells"])
    rng.shuffle(big_cells)
    for start in big_cells:
        candidate = grid.copy()
        candidate[start] = MOVER
        new_out = main(candidate)
        old_out = _old_color_chain_main(candidate)
        new_pos = tuple(np.argwhere(new_out == MOVER)[0])
        if new_pos not in layout["small"]["cells"]:
            continue
        if new_pos == start:
            continue
        if np.array_equal(new_out, old_out):
            continue
        grid[start] = MOVER
        return start, new_pos
    return None


def _behavior_metrics(input_grid: np.ndarray) -> dict:
    bg = get_bg(input_grid)
    legends = _extract_legends(input_grid, bg)
    systems = _extract_serial_systems(input_grid, bg)
    if len(legends) != 2 or len(systems) != 1:
        return {
            "n_legends": len(legends),
            "n_systems": len(systems),
            "area_gap": -1,
            "crosses_door": False,
            "swapped_assignments": 0,
            "old_new_diff_cells": 0,
        }

    chambers = _collect_chambers_by_area(systems)
    start = tuple(np.argwhere(input_grid == MOVER)[0])
    new_out = main(input_grid)
    old_out = _old_color_chain_main(input_grid)
    final = tuple(np.argwhere(new_out == MOVER)[0])

    area_gap = chambers[0]["area"] - chambers[1]["area"]
    crosses_door = start in chambers[0]["cells"] and final in chambers[1]["cells"]
    swapped_assignments = sum(
        int(chamber["border_color"] != legends[idx]["fill"]) for idx, chamber in enumerate(chambers[:2])
    )
    return {
        "n_legends": len(legends),
        "n_systems": len(systems),
        "area_gap": area_gap,
        "crosses_door": crosses_door,
        "swapped_assignments": swapped_assignments,
        "old_new_diff_cells": int(np.sum(new_out != old_out)),
        "corridor_len": len(systems[0]["corridor"]),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_mask, legends = _place_legends(rng, grid)
    layout = _try_place_serial_system(rng, legends, legend_mask)
    if layout is None:
        return None

    for r, c in layout["big"]["ring"]:
        grid[r, c] = layout["big"]["color"]
    for r, c in layout["small"]["ring"]:
        grid[r, c] = layout["small"]["color"]
    for r, c in layout["connector_walls"]:
        if grid[r, c] == 0:
            grid[r, c] = layout["big"]["color"]

    if not _validate_topology(grid, layout):
        return None

    input_grid = grid.copy()
    placed = _place_yellow(rng, input_grid, layout)
    if placed is None:
        return None

    metrics = _behavior_metrics(input_grid)
    if metrics["n_legends"] != 2 or metrics["n_systems"] != 1:
        return None
    if metrics["area_gap"] < AREA_GAP_MIN:
        return None
    if not metrics["crosses_door"]:
        return None
    if metrics["swapped_assignments"] != 2:
        return None
    if metrics["old_new_diff_cells"] <= 0:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
