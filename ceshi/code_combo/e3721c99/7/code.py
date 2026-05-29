# Combo variant of e3721c99 (index 7)
# combo_spec: D6/0+D9/0
# Name: 精灵移动到图例旁+一对一严格匹配

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, label_mask


# -----------------------------------------------------------------------------
# Design analysis for the D6/0 + D9/0 combo
#
# a) Concrete behavior:
#    - The original task recolored every matching grey sprite in place, and it
#      allowed several sprites to share the same legend hole count.
#    - This combo removes both freedoms. Each legend hole count now appears on
#      exactly one grey sprite, and that sprite leaves its source location.
#    - Example trace:
#      original: legends encode holes {0,2,4}; two separate 2-hole sprites
#      could both become color 7 in place.
#      combo: there is exactly one 0-hole, one 2-hole, and one 4-hole grey
#      sprite. After matching, each source patch is erased, then the recolored
#      patch is moved next to the corresponding legend entry on the workspace
#      side. The output therefore shows a one-to-one legend/sprite lineup.
#
# b) Complexity calibration:
#    - To keep the movement visually obvious, the output uses only 2-4 matched
#      sprites and no distractors; every grey object participates.
#    - To keep the one-to-one rule inferable, legend hole counts are unique and
#      the grey sprites use the exact same unique hole set.
#    - Legends remain sorted by hole count, so the moved sprites can also be
#      read left-to-right or top-to-bottom as an ordered pairing rather than a
#      noisy cluster.
#
# c) Old assumptions that break:
#    - The original `for pts in grey_components` recolor loop assumed local,
#      independent updates. That fails because source cells must now become
#      background before any destination placements happen.
#    - The original generator assumed duplicate matches and distractors were
#      harmless. Under strict one-to-one matching, duplicates would make the
#      pairing ambiguous and therefore must be forbidden.
#    - The D6 single-queue idea is refined here: because D9 guarantees one grey
#      sprite per legend, the solver can anchor each moved sprite around its own
#      legend entry instead of treating all matches as an undifferentiated pool.
#
# d) Placement legality:
#    - The generator reserves a placement band on the workspace side of the
#      divider so input sprites never occupy their future destinations.
#    - The solver only places moved patches where they fit inside the grid, do
#      not overlap existing non-background cells, and keep a one-cell buffer.
#    - Greedy per-entry placement is safe because the generator uses only 2-4
#      moderately sized sprites and validates every sample by running `main()`.
#
# e) Visual observability:
#    - Hole count is still the only matching key, which is a visible topological
#      feature. No hidden ordering by color id or generation order is used.
#    - The combo is visually distinct from the base task because all grey source
#      positions vanish, and from plain D6 because the moved sprites align with
#      individual legend entries instead of representing repeated many-to-one
#      matches.
#
# f) Visual cleanliness:
#    - The legend panel stays isolated behind the divider, while the moved
#      sprites occupy a reserved strip on the workspace side.
#    - One sprite per legend keeps the destination area sparse enough that each
#      legend/sprite pair remains separable at a glance.
#
# g) Inferability of the change:
#    - Color assignment is explained structurally by the legend shapes.
#    - The extra movement step is also structural: after matching, each unique
#      sprite migrates beside the legend entry with the same hole count.
#
# New implicit assumptions and generator guarantees:
#    - `main()` assumes grey components are unique by hole count and their hole
#      set exactly matches the legend hole set; `generate_input()` constructs
#      inputs that satisfy this.
#    - `main()` assumes enough empty space exists beside the legend entries for
#      the moved sprites; the reserved queue band and sample validation enforce
#      this.
#    - `main()` assumes legend entries form a single horizontal or vertical line
#      flush with the divider; the layout builder guarantees that geometry.
# -----------------------------------------------------------------------------


# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 4)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
SPRITE_RETRIES = 35
SPRITE_AREA_0 = (5, 18)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.1
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.42
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 10
SPRITE_MAX_W = 12
SPRITE_MAX_ASPECT = 2.3

# -- generation --
GEN_RETRIES = 400
QUEUE_BAND_DEPTH = 10

_STRUCT8 = generate_binary_structure(2, 2)


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding box border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for rr, cc in neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _legend_entries(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """Detect one interior legend component per color and annotate it with hole count."""
    h, w = g.shape
    entries: list[dict] = []
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox_coords(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append((pts, (int(r0), int(c0), int(r1), int(c1))))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        pts, bbox = candidates[0]
        r0, c0, r1, c1 = bbox
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
        entries.append(
            {
                "color": int(color),
                "hole": int(_hole_count(sub)),
                "bbox": bbox,
                "pts": pts,
            }
        )
    holes = [int(entry["hole"]) for entry in entries]
    assert len(set(holes)) == len(holes), "legend hole counts must be unique"
    entries.sort(key=lambda entry: (int(entry["hole"]), int(entry["bbox"][0]), int(entry["bbox"][1])))
    assert entries, "legend mapping must be non-empty: grid must contain colored legend shapes"
    return entries


def _legend_mapping(entries: list[dict]) -> dict[int, int]:
    return {int(entry["hole"]): int(entry["color"]) for entry in entries}


def _legend_union_bbox(entries: list[dict]) -> tuple[int, int, int, int]:
    r0 = min(int(entry["bbox"][0]) for entry in entries)
    c0 = min(int(entry["bbox"][1]) for entry in entries)
    r1 = max(int(entry["bbox"][2]) for entry in entries)
    c1 = max(int(entry["bbox"][3]) for entry in entries)
    return int(r0), int(c0), int(r1), int(c1)


def _legend_arrangement(entries: list[dict]) -> str:
    """Legend entries form a single line; infer whether that line is horizontal or vertical."""
    if len(entries) == 1:
        r0, c0, r1, c1 = entries[0]["bbox"]
        return "horizontal" if (c1 - c0) >= (r1 - r0) else "vertical"
    row_centers = [0.5 * (int(entry["bbox"][0]) + int(entry["bbox"][2])) for entry in entries]
    col_centers = [0.5 * (int(entry["bbox"][1]) + int(entry["bbox"][3])) for entry in entries]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    return "horizontal" if col_span >= row_span else "vertical"


def _component_record(pts: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[int(r - r0), int(c - c0)] = True
    return {
        "pts": pts,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "hole": int(_hole_count(mask)),
    }


def _queue_side(entries: list[dict], arrangement: str, shape: tuple[int, int]) -> str:
    legend_box = _legend_union_bbox(entries)
    height, width = map(int, shape)
    if arrangement == "horizontal":
        legend_mid = 0.5 * (legend_box[0] + legend_box[2])
        return "below" if legend_mid < 0.5 * (height - 1) else "above"
    legend_mid = 0.5 * (legend_box[1] + legend_box[3])
    return "right" if legend_mid < 0.5 * (width - 1) else "left"


def _fits_with_gap(out: np.ndarray, patch: np.ndarray, r: int, c: int, bg: int) -> bool:
    """Rectangular 1-cell clearance around placed sprites for a clean legend-side lineup."""
    h, w = patch.shape
    height, width = out.shape
    if r < 0 or c < 0 or r + h > height or c + w > width:
        return False
    if np.any(out[r : r + h, c : c + w][patch != bg] != bg):
        return False
    rr0, cc0 = max(0, r - 1), max(0, c - 1)
    rr1, cc1 = min(height, r + h + 1), min(width, c + w + 1)
    return not np.any(out[rr0:rr1, cc0:cc1] != bg)


def _around_anchor(anchor: int, low: int, high: int):
    anchor = int(min(max(anchor, low), high))
    seen = set()
    for dist in range(high - low + 1):
        for pos in (anchor - dist, anchor + dist):
            if low <= pos <= high and pos not in seen:
                seen.add(pos)
                yield int(pos)


def _candidate_positions_for_entry(
    out: np.ndarray,
    patch: np.ndarray,
    entry_bbox: tuple[int, int, int, int],
    arrangement: str,
    side: str,
):
    height, width = out.shape
    h, w = patch.shape
    r0, c0, r1, c1 = map(int, entry_bbox)
    if arrangement == "horizontal":
        anchor_col = int(round(0.5 * (c0 + c1) - 0.5 * (w - 1)))
        cols = _around_anchor(anchor_col, 0, width - w)
        if side == "below":
            start = r1 + 3
            stop = min(height - h + 1, start + max(0, QUEUE_BAND_DEPTH - h + 1))
            rows = range(start, stop)
        else:
            start = r0 - h - 2
            stop = max(-1, start - max(0, QUEUE_BAND_DEPTH - h + 1))
            rows = range(start, stop, -1)
        for r in rows:
            for c in cols:
                yield int(r), int(c)
        return

    anchor_row = int(round(0.5 * (r0 + r1) - 0.5 * (h - 1)))
    rows = _around_anchor(anchor_row, 0, height - h)
    if side == "right":
        start = c1 + 3
        stop = min(width - w + 1, start + max(0, QUEUE_BAND_DEPTH - w + 1))
        cols = range(start, stop)
    else:
        start = c0 - w - 2
        stop = max(-1, start - max(0, QUEUE_BAND_DEPTH - w + 1))
        cols = range(start, stop, -1)
    for c in cols:
        for r in rows:
            yield int(r), int(c)


def _solve_variant(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    grey = GREY

    entries = _legend_entries(g, bg, grey)
    legend = _legend_mapping(entries)
    arrangement = _legend_arrangement(entries)
    side = _queue_side(entries, arrangement, g.shape)

    grey_components = [_component_record(pts) for pts in components_mask(g == grey, connectivity=4)]
    assert grey_components, "grid must contain grey sprites"
    assert len(grey_components) == len(entries), "strict one-to-one variant needs one grey sprite per legend"
    grey_holes = [int(comp["hole"]) for comp in grey_components]
    legend_holes = [int(entry["hole"]) for entry in entries]
    assert len(set(grey_holes)) == len(grey_holes), "grey sprite hole counts must be unique"
    assert set(grey_holes) == set(legend_holes), "grey sprites must match the legend hole set exactly"
    by_hole = {int(comp["hole"]): comp for comp in grey_components}

    out = g.copy()
    for comp in grey_components:
        for r, c in comp["pts"]:
            out[int(r), int(c)] = bg

    placements = []
    moved_cells = 0
    for entry in entries:
        hole = int(entry["hole"])
        comp = by_hole[hole]
        patch = np.where(comp["mask"], int(entry["color"]), bg).astype(int)
        placed = False
        for r, c in _candidate_positions_for_entry(out, patch, entry["bbox"], arrangement, side):
            if not _fits_with_gap(out, patch, r, c, bg):
                continue
            paste(out, patch, int(r), int(c), transparent=bg)
            placements.append(
                {
                    "hole": hole,
                    "entry_bbox": tuple(map(int, entry["bbox"])),
                    "source_bbox": tuple(map(int, comp["bbox"])),
                    "dest_bbox": (int(r), int(c), int(r + patch.shape[0] - 1), int(c + patch.shape[1] - 1)),
                    "patch": patch.copy(),
                }
            )
            moved_cells += int(np.sum(comp["mask"]))
            placed = True
            break
        assert placed, "legend-side placement failed"

    if return_trace:
        return {
            "out": out,
            "entries": entries,
            "arrangement": arrangement,
            "side": side,
            "legend_box": _legend_union_bbox(entries),
            "legend_count": len(entries),
            "grey_count": len(grey_components),
            "matched_count": len(placements),
            "moved_cells": int(moved_cells),
            "entry_holes": legend_holes,
            "source_holes": grey_holes,
            "placements": placements,
        }
    return out


def _baseline_in_place_output(grid: np.ndarray) -> np.ndarray:
    """Original-task style output on inputs that already satisfy strict one-to-one matching."""
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    legend = _legend_mapping(_legend_entries(g, bg, GREY))
    for comp in components_mask(g == GREY, connectivity=4):
        rec = _component_record(comp)
        color = int(legend[rec["hole"]])
        for r, c in rec["pts"]:
            out[int(r), int(c)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey sprite to a unique legend entry, then move the recolored result beside that legend.

    Input structure:
    - bg is the most common color; grey (5) sprites are scattered only in the workspace area.
    - A divider isolates a legend panel near an edge or corner, and the legend entries form a single ordered line.
    - Each legend entry has a distinct hole count, and there is exactly one grey sprite with that same hole count.

    Rule:
    - Read the colored legend as a hole-count → color mapping.
    - For each grey sprite, count holes and pair it with the unique legend entry having the same count.
    - Remove the grey sprite from its source position, recolor it with the legend color, and place it on the workspace side directly beside the corresponding legend entry.

    Key insight: The topology match is still determined by hole count, but strict one-to-one matching turns the output into an explicit aligned pairing where every successful sprite migrates to the legend area instead of staying in place.
    """
    return _solve_variant(grid, return_trace=False)


# ---------- legend helpers ----------

def _legend_span(hole_count: int) -> int:
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, color: int, arrangement: str) -> np.ndarray:
    span = _legend_span(hole_count)
    if arrangement == "horizontal":
        patch = np.full((3, span), int(color), dtype=int)
        if hole_count:
            patch[1, 1::2] = BG
    else:
        patch = np.full((span, 3), int(color), dtype=int)
        if hole_count:
            patch[1::2, 1] = BG
    return patch


def _line_span(legends: list[tuple[int, int]]) -> int:
    return sum(_legend_span(hole) for hole, _ in legends) + len(legends) - 1


def _choose_legends(arrangement: str, max_span: int) -> list[tuple[int, int]]:
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(hole) for hole in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(arrangement, max_span + 2)
    holes = list(random.choice(combos))
    holes.sort()
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray, legends: list[tuple[int, int]], arrangement: str, r: int, c: int
) -> list[dict]:
    entries: list[dict] = []
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        h, w = patch.shape
        entries.append(
            {
                "hole": int(hole_count),
                "color": int(color),
                "bbox": (int(rr), int(cc), int(rr + h - 1), int(cc + w - 1)),
            }
        )
        if arrangement == "horizontal":
            cc += w + 1
        else:
            rr += h + 1
    return entries


# ---------- layout helpers ----------

def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> dict:
    legends = _choose_legends("horizontal", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = div_r - 3
        side = "below"
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = div_r + 1
        side = "above"
    entries = _place_legend_line(grid, legends, "horizontal", r0, c0)
    return {"entries": entries, "arrangement": "horizontal", "side": side}


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> dict:
    legends = _choose_legends("vertical", FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = div_c - 3
        side = "right"
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = div_c + 1
        side = "left"
    entries = _place_legend_line(grid, legends, "vertical", r0, c0)
    return {"entries": entries, "arrangement": "vertical", "side": side}


def _l_corner_geometry(corner: str, panel_h: int, panel_w: int, arrangement: str, total: int) -> tuple:
    size = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, size - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, size - 1 - along_dim
    elif corner == "bl":
        dr, dc = size - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, size - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = size - 1 - panel_h, size - 1 - panel_w
        r_min, r_max = dr + 2, size - 1 - cross_dim
        c_min, c_max = dc + 2, size - 1 - along_dim

    return dr, dc, r_min, r_max, c_min, c_max


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> dict:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(arrangement, L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    if arrangement == "horizontal":
        panel_h, panel_w = cross, along
    else:
        panel_h, panel_w = along, cross
    corner = random.choice(["tl", "tr", "bl", "br"])

    dr, dc, r_min, r_max, c_min, c_max = _l_corner_geometry(corner, panel_h, panel_w, arrangement, total)

    if corner == "tl":
        grid[dr, : dc + 1] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "tr":
        grid[dr, dc:] = DIV
        grid[: dr + 1, dc] = DIV
        blocked[:dr, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1:, dc + 1:] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    if arrangement == "horizontal":
        c0 = random.randint(c_min, c_max)
        if corner in ("tl", "tr"):
            r0 = dr - 3
            side = "below"
        else:
            r0 = dr + 1
            side = "above"
    else:
        r0 = random.randint(r_min, r_max)
        if corner in ("tl", "bl"):
            c0 = dc - 3
            side = "right"
        else:
            c0 = dc + 1
            side = "left"

    entries = _place_legend_line(grid, legends, arrangement, r0, c0)
    return {"entries": entries, "arrangement": arrangement, "side": side}


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> dict:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _reserve_queue_band(blocked: np.ndarray, layout: dict) -> None:
    r0, c0, r1, c1 = _legend_union_bbox(layout["entries"])
    if layout["arrangement"] == "horizontal":
        if layout["side"] == "below":
            a = min(GRID_SIZE, r1 + 3)
            b = min(GRID_SIZE, a + QUEUE_BAND_DEPTH)
            blocked[a:b, :] = 9
        else:
            b = max(0, r0 - 2)
            a = max(0, b - QUEUE_BAND_DEPTH)
            blocked[a:b, :] = 9
        return
    if layout["side"] == "right":
        a = min(GRID_SIZE, c1 + 3)
        b = min(GRID_SIZE, a + QUEUE_BAND_DEPTH)
        blocked[:, a:b] = 9
    else:
        b = max(0, c0 - 2)
        a = max(0, b - QUEUE_BAND_DEPTH)
        blocked[:, a:b] = 9


# ---------- sprite generation ----------

def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 20 + 7 * hole_count)
    return random.randint(10 + 2 * hole_count, 16 + 3 * hole_count)


def _place_hole_seeds(hole_count: int, center: int, csz: int, area: int) -> list[tuple[int, int]] | None:
    margin = HOLE_SEED_MARGIN
    radius = max(HOLE_SEED_MIN_RADIUS, int(HOLE_SEED_RADIUS_FACTOR * area ** 0.5))
    positions = [
        (r, c)
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
        if margin <= r < csz - margin and margin <= c < csz - margin
    ]
    random.shuffle(positions)
    seeds = []
    for r, c in positions:
        if all(max(abs(r - sr), abs(c - sc)) >= HOLE_SEED_MIN_SEP for sr, sc in seeds):
            seeds.append((r, c))
            if len(seeds) == hole_count:
                return seeds
    return None


def _grow_hole_blobs(seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int) -> set[tuple[int, int]]:
    margin = HOLE_SEED_MARGIN
    max_hole = max(1, area // (hole_count + 2))
    forbidden = set()
    for sr, sc in seeds:
        forbidden.add((sr, sc))
        blob = {(sr, sc)}
        target = random.randint(1, max(1, max_hole))
        while len(blob) < target:
            cands = []
            for r, c in blob:
                for nr, nc in neighbors4(r, c):
                    if not (margin <= nr < csz - margin and margin <= nc < csz - margin):
                        continue
                    if (nr, nc) in forbidden:
                        continue
                    if any((nnr, nnc) in forbidden and (nnr, nnc) not in blob for nnr, nnc in neighbors4(nr, nc)):
                        continue
                    cands.append((nr, nc))
            if not cands:
                break
            choice = random.choice(cands)
            forbidden.add(choice)
            blob.add(choice)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    if (center, center) not in forbidden:
        return center, center
    for dist in range(1, csz):
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                nr, nc = center + dr, center + dc
                if 0 <= nr < csz and 0 <= nc < csz and (nr, nc) not in forbidden:
                    return nr, nc
    return center, center


def _grow_mask(csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((csz, csz), dtype=bool)
    sr, sc = start
    mask[sr, sc] = True
    frontier = list(neighbors4(sr, sc))
    filled = 1
    while filled < area and frontier:
        idx = random.randrange(len(frontier))
        frontier[idx], frontier[-1] = frontier[-1], frontier[idx]
        r, c = frontier.pop()
        if not (0 <= r < csz and 0 <= c < csz) or mask[r, c] or (r, c) in forbidden:
            continue
        mask[r, c] = True
        filled += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc]:
                frontier.append((nr, nc))
    return mask


def _seal_and_connect(mask: np.ndarray, forbidden: set[tuple[int, int]]) -> np.ndarray:
    csz = mask.shape[0]
    for _ in range(SEAL_ITERS):
        sealed = False
        for r, c in sorted(forbidden):
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < csz and 0 <= nc < csz and not mask[nr, nc] and (nr, nc) not in forbidden:
                    mask[nr, nc] = True
                    sealed = True
        if not sealed:
            break
    if np.any(mask):
        labeled, n_comp = label_mask(mask)
        if n_comp > 1:
            best = max(range(1, n_comp + 1), key=lambda lab: int(np.sum(labeled == lab)))
            mask = labeled == best
    return mask


def _crop_and_validate(mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int) -> np.ndarray | None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    rmin, cmin = pts.min(axis=0)
    rmax, cmax = pts.max(axis=0)
    for r, c in forbidden:
        rmin, cmin = min(rmin, r), min(cmin, c)
        rmax, cmax = max(rmax, r), max(cmax, c)
    h, w = int(rmax - rmin + 1), int(cmax - cmin + 1)
    if h > SPRITE_MAX_H or w > SPRITE_MAX_W or max(h, w) > SPRITE_MAX_ASPECT * min(h, w):
        return None
    cropped = mask[int(rmin) : int(rmax) + 1, int(cmin) : int(cmax) + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _crop_color_patch(arr: np.ndarray) -> np.ndarray:
    pts = np.argwhere(arr != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _template_sprite(hole_count: int) -> np.ndarray | None:
    if hole_count == 0:
        choices = [
            np.array([[GREY, GREY, GREY], [GREY, BG, GREY], [GREY, GREY, BG]], dtype=int),
            np.array([[GREY, GREY, BG], [GREY, GREY, GREY], [BG, GREY, GREY]], dtype=int),
            np.array([[GREY, GREY, GREY, BG], [BG, GREY, GREY, GREY]], dtype=int),
        ]
        patch = random.choice(choices).copy()
        return np.rot90(patch, random.randint(0, 3)).copy()

    arrangement = random.choice(["horizontal", "vertical"])
    base = _legend_patch(hole_count, GREY, arrangement)
    canvas = np.pad(np.where(base == GREY, GREY, BG).astype(int), 1, constant_values=BG)
    h, w = canvas.shape
    if arrangement == "horizontal":
        tabs = [(0, 1), (h - 1, w - 2), (h - 1, 1)]
    else:
        tabs = [(1, 0), (h - 2, w - 1), (1, w - 1)]
    random.shuffle(tabs)
    for r, c in tabs[: random.randint(1, min(2, len(tabs)))]:
        canvas[int(r), int(c)] = GREY
    canvas = _crop_color_patch(canvas)
    mask = canvas == GREY
    if _hole_count(mask) != hole_count:
        canvas = np.where(base == GREY, GREY, BG).astype(int)
        if _hole_count(canvas == GREY) != hole_count:
            return None
    return canvas


def _make_sprite(hole_count: int) -> np.ndarray | None:
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2

        forbidden = set()
        if hole_count > 0:
            seeds = _place_hole_seeds(hole_count, center, csz, area)
            if seeds is None:
                continue
            forbidden = _grow_hole_blobs(seeds, hole_count, area, csz)

        start = _find_start(center, csz, forbidden)
        mask = _grow_mask(csz, area, start, forbidden)
        mask = _seal_and_connect(mask, forbidden)
        result = _crop_and_validate(mask, forbidden, hole_count)
        if result is not None:
            return result
    return _template_sprite(hole_count)


# ---------- sprite placement ----------

def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _random_sprite_location(static_blocked: np.ndarray, occupied: np.ndarray, sprite: np.ndarray) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    occupied_pad = _dilate_mask(occupied != BG)
    candidates = [(r, c) for r in range(GRID_SIZE - h + 1) for c in range(GRID_SIZE - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if np.any((static_blocked[r : r + h, c : c + w] != BG) & sprite_mask):
            continue
        if np.any(occupied_pad[r : r + h, c : c + w] & sprite_mask):
            continue
        return int(r), int(c)
    return None


def _place_sprites(grid: np.ndarray, blocked: np.ndarray, legends: list[tuple[int, int]]) -> bool:
    sprites = []
    for hole_count, _color in legends:
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        sprites.append((sprite.size, sprite))
    sprites.sort(key=lambda item: item[0], reverse=True)

    occupied = np.zeros_like(grid)
    for _, sprite in sprites:
        loc = _random_sprite_location(blocked, occupied, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)
    return True


# ---------- top-level generation ----------

def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    layout = _add_layout(grid, blocked)
    _reserve_queue_band(blocked, layout)

    legends = [(int(entry["hole"]), int(entry["color"])) for entry in layout["entries"]]
    if not _place_sprites(grid, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None

    try:
        trace = _solve_variant(grid, return_trace=True)
    except Exception:
        return None

    if trace["legend_count"] != len(legends):
        return None
    if trace["grey_count"] != len(legends) or trace["matched_count"] != len(legends):
        return None
    if trace["moved_cells"] <= 0:
        return None
    if trace["entry_holes"] != sorted(trace["entry_holes"]):
        return None
    if sorted(trace["source_holes"]) != trace["entry_holes"]:
        return None
    for item in trace["placements"]:
        r0, c0, r1, c1 = item["source_bbox"]
        src_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == GREY
        if np.any(trace["out"][r0 : r1 + 1, c0 : c1 + 1][src_mask] != BG):
            return None
    if np.array_equal(trace["out"], _baseline_in_place_output(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
