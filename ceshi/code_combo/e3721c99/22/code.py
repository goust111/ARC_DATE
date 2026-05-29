# Combo variant of e3721c99 (index 22)
# combo_spec: D6/0+D13/2
# Name: 紧密拼接精灵+移动到图例旁

from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, label_mask


# -----------------------------------------------------------------------------
# Design analysis for the D6/0 + D13/2 combo
#
# a) Concrete behavior:
#    - Original e3721c99 shows grey sprites scattered across the workspace. Each
#      sprite is already isolated, so the solver can directly count holes and
#      recolor in place.
#    - D6 changes the last step: successful matches no longer stay put. They are
#      erased from the source field and reassembled beside the legend in
#      hole-count order.
#    - D13 changes the input geometry: the grey sprites are no longer dispersed.
#      They are packed into one dense cluster with only 1-cell seams between
#      neighbors, so a human first has to visually split the cluster into
#      distinct sprites before counting holes.
#    - Example trace:
#      input: a dense pack contains one 0-hole blob, one 2-hole blob, one
#      distractor 4-hole blob, and the legend maps 0->red, 2->green.
#      original task: if these sprites were scattered, the 0-hole and 2-hole
#      blobs would simply become red and green in their original locations.
#      combo task: the dense pack is first segmented along its 1-cell seams;
#      then the 0-hole and 2-hole blobs are recolored; finally they disappear
#      from the pack and reappear near the legend as a sorted queue, while the
#      4-hole distractor vanishes entirely.
#
# b) Complexity calibration:
#    - The packed cluster must be visually unmistakable, so generator enforces a
#      single 8-dilation-connected grey pack instead of a loose scatter.
#    - The movement must also be unmistakable, so the destination queue uses a
#      reserved strip next to the legend and the source pack becomes empty.
#    - Legend entries remain few (2-3) and hole counts remain in a short visible
#      range, keeping
#      the mapping inferable while still supporting repeated evidence.
#
# c) Old assumptions that break:
#    - In the base solver, `components_mask(g == grey, 4)` was both detection
#      and segmentation. Under D13 this is no longer the whole story: the input
#      visually behaves like one dense mass, so the solver should reason in two
#      layers: packed cluster detection, then per-sprite extraction inside that
#      cluster.
#    - In the base solver, modifying grey cells in place was correct. Under D6
#      that becomes wrong because matched sprites must vacate the source area.
#    - The output order is now global. We cannot process sprites independently;
#      we must collect all matches, sort them by visible topology, and place
#      them into the destination strip without overlap.
#
# d) Placement legality:
#    - Generator reserves the queue strip next to the legend so the input pack
#      never occupies the output destination.
#    - The packed source cluster is built from a shelf layout with exact 1-cell
#      seams between neighboring sprite bounding boxes, so sprites stay
#      disconnected in 4-connectivity but merge into one visual pack after
#      8-dilation.
#    - Solver places moved sprites greedily into the arrival side, forbidding
#      overlap with the legend/divider and with earlier queued sprites.
#
# e) Visual observability:
#    - The split cue is human-visible: grey sub-shapes sit inside one compact
#      mass and are separated only by thin black seams.
#    - The matching cue is still the hole-count legend, which is directly
#      inspectable.
#    - The trajectory cue is also visible: the dense grey pack partially or
#      fully disappears, and a clean colored queue appears beside the legend.
#
# f) Visual cleanliness:
#    - Only one dense grey pack is generated, not several competing clusters.
#    - The output queue lives in a dedicated band next to the legend instead of
#      mixing with leftover workspace clutter.
#    - Distractors disappear, which keeps the output simpler than the input.
#
# g) Change inferability:
#    - Color does not depend on hidden ids or arbitrary ordering; it still comes
#      only from hole-count matching against the legend.
#    - Queue ordering also stays visible: sprites are arranged by ascending hole
#      count, with source-bbox order used only as a stable tie-breaker between
#      identical hole counts.
#
# New implicit assumptions and generator guarantees:
#    - `main()` assumes legend components are the only interior non-bg/non-grey
#      components; generator preserves that.
#    - `main()` assumes all grey sprites can be grouped into one or more tight
#      packs by 1-step 8-dilation; generator intentionally creates exactly one
#      such pack and validates it.
#    - `main()` assumes enough room exists in the queue band for all matches;
#      generator reserves that band and rejects samples where queue placement
#      fails under the oracle.
# -----------------------------------------------------------------------------


# -- palette & grid --
BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- legend panel --
LEGEND_COUNT_RANGE = (2, 3)
MAX_HOLE_COUNT = 5
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

# -- sprites --
TOTAL_SPRITES_RANGE = (5, 5)
MAX_DISTRACTORS = 1
MAX_EXTRA_PER_LEGEND = 1
SPRITE_RETRIES = 35
SPRITE_AREA_0 = (5, 16)
SPRITE_AREA_LOW_MIN = 7
CANVAS_SCALE = 1.15
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 12
SPRITE_MAX_W = 18
SPRITE_MAX_ASPECT = 2.5

# -- generation --
GEN_RETRIES = 500
QUEUE_BAND_DEPTH = 8
COMPACT_CLUSTER_MAX_DIM = 22

_STRUCT8 = generate_binary_structure(2, 2)


def _hole_count(mask: np.ndarray) -> int:
    """Count 4-connected holes inside a boolean sprite mask."""
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
        hc = _hole_count(sub)
        entries.append({"color": int(color), "hole": int(hc), "bbox": bbox, "pts": pts})
    holes = [int(e["hole"]) for e in entries]
    assert len(set(holes)) == len(holes), "legend hole counts must be unique"
    entries.sort(key=lambda e: (int(e["hole"]), int(e["bbox"][0]), int(e["bbox"][1])))
    assert entries, "legend mapping must be non-empty: grid must contain colored legend shapes"
    return entries


def _legend_mapping(entries: list[dict]) -> dict[int, int]:
    return {int(e["hole"]): int(e["color"]) for e in entries}


def _legend_union_bbox(entries: list[dict]) -> tuple[int, int, int, int]:
    r0 = min(int(e["bbox"][0]) for e in entries)
    c0 = min(int(e["bbox"][1]) for e in entries)
    r1 = max(int(e["bbox"][2]) for e in entries)
    c1 = max(int(e["bbox"][3]) for e in entries)
    return int(r0), int(c0), int(r1), int(c1)


def _legend_arrangement(entries: list[dict]) -> str:
    if len(entries) == 1:
        r0, c0, r1, c1 = entries[0]["bbox"]
        return "horizontal" if (c1 - c0) >= (r1 - r0) else "vertical"
    row_centers = [0.5 * (int(e["bbox"][0]) + int(e["bbox"][2])) for e in entries]
    col_centers = [0.5 * (int(e["bbox"][1]) + int(e["bbox"][3])) for e in entries]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    return "horizontal" if col_span >= row_span else "vertical"


def _component_record(pts: np.ndarray, pack_id: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[int(r - r0), int(c - c0)] = True
    return {
        "pts": pts,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "hole": int(_hole_count(mask)),
        "pack_id": int(pack_id),
    }


def _packed_sprite_components(g: np.ndarray) -> tuple[list[dict], dict[int, tuple[int, int, int, int]]]:
    """
    Recover grey sprites in two stages:
    1. detect dense packs by 1-step 8-dilation;
    2. split each pack back into 4-connected sprites.
    """
    grey_mask = g == GREY
    comps = components_mask(grey_mask, connectivity=4)
    assert comps, "grid must contain grey sprites"

    dilated = binary_dilation(grey_mask, structure=_STRUCT8).astype(bool)
    pack_lab, n_pack = label_mask(dilated, connectivity=8)
    assert n_pack >= 1, "tight-pack detection failed"

    grouped: dict[int, list[dict]] = defaultdict(list)
    pack_boxes: dict[int, tuple[int, int, int, int]] = {}
    for pid in range(1, int(n_pack) + 1):
        pts = np.argwhere(pack_lab == pid)
        pack_boxes[int(pid)] = bbox_coords(pts)

    for pts in comps:
        r, c = map(int, pts[0])
        pid = int(pack_lab[r, c])
        assert pid >= 1, "sprite must belong to a dilated pack"
        grouped[pid].append(_component_record(pts, pid))

    ordered: list[dict] = []
    for pid in sorted(grouped, key=lambda k: (pack_boxes[k][0], pack_boxes[k][1])):
        group = grouped[pid]
        group.sort(key=lambda rec: (rec["bbox"][0], rec["bbox"][1], rec["bbox"][2], rec["bbox"][3]))
        ordered.extend(group)
    return ordered, pack_boxes


def _queue_side(entries: list[dict], arrangement: str, shape: tuple[int, int]) -> str:
    legend_box = _legend_union_bbox(entries)
    h, w = map(int, shape)
    if arrangement == "horizontal":
        legend_mid = 0.5 * (legend_box[0] + legend_box[2])
        return "below" if legend_mid < 0.5 * (h - 1) else "above"
    legend_mid = 0.5 * (legend_box[1] + legend_box[3])
    return "right" if legend_mid < 0.5 * (w - 1) else "left"


def _fits_with_gap(
    out: np.ndarray, queue_mask: np.ndarray, patch: np.ndarray, r: int, c: int, bg: int
) -> bool:
    """Keep moved sprites non-overlapping while still avoiding non-bg structure."""
    h, w = patch.shape
    hh, ww = out.shape
    if r < 0 or c < 0 or r + h > hh or c + w > ww:
        return False
    if np.any(out[r : r + h, c : c + w][patch != bg] != bg):
        return False
    return not np.any(queue_mask[r : r + h, c : c + w] & (patch != bg))


def _candidate_positions(
    out: np.ndarray,
    patch: np.ndarray,
    legend_box: tuple[int, int, int, int],
    arrangement: str,
    side: str,
):
    hh, ww = out.shape
    h, w = patch.shape
    r0, c0, r1, c1 = map(int, legend_box)
    if arrangement == "horizontal":
        cols = range(0, ww - w + 1)
        if side == "below":
            start = r1 + 3
            stop = hh - h + 1
            rows = range(start, stop)
        else:
            start = r0 - h - 2
            stop = -1
            rows = range(start, stop, -1)
        for r in rows:
            for c in cols:
                yield int(r), int(c)
        return

    rows = range(0, hh - h + 1)
    if side == "right":
        start = c1 + 3
        stop = ww - w + 1
        cols = range(start, stop)
    else:
        start = c0 - w - 2
        stop = -1
        cols = range(start, stop, -1)
    for c in cols:
        for r in rows:
            yield int(r), int(c)


def _solve_variant(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    entries = _legend_entries(g, bg, GREY)
    legend = _legend_mapping(entries)
    arrangement = _legend_arrangement(entries)
    side = _queue_side(entries, arrangement, g.shape)
    legend_box = _legend_union_bbox(entries)

    grey_components, pack_boxes = _packed_sprite_components(g)

    out = g.copy()
    for comp in grey_components:
        for r, c in comp["pts"]:
            out[int(r), int(c)] = bg

    matched = []
    moved_cells = 0
    for comp in grey_components:
        hc = int(comp["hole"])
        if hc not in legend:
            continue
        newc = int(legend[hc])
        patch = np.where(comp["mask"], newc, bg).astype(int)
        matched.append(
            {
                "hole": int(hc),
                "bbox": comp["bbox"],
                "pack_id": int(comp["pack_id"]),
                "patch": patch,
                "source_pts": comp["pts"],
            }
        )
        moved_cells += int(np.sum(comp["mask"]))

    matched.sort(
        key=lambda item: (
            int(item["hole"]),
            int(pack_boxes[int(item["pack_id"])][0]),
            int(pack_boxes[int(item["pack_id"])][1]),
            int(item["bbox"][0]),
            int(item["bbox"][1]),
            int(item["patch"].shape[0]),
            int(item["patch"].shape[1]),
        )
    )

    placements = []
    queue_mask = np.zeros_like(out, dtype=bool)
    for item in matched:
        patch = item["patch"]
        placed = False
        for r, c in _candidate_positions(out, patch, legend_box, arrangement, side):
            if not _fits_with_gap(out, queue_mask, patch, r, c, bg):
                continue
            paste(out, patch, int(r), int(c), transparent=bg)
            queue_mask[r : r + patch.shape[0], c : c + patch.shape[1]] |= patch != bg
            placements.append((int(r), int(c), patch.copy(), int(item["hole"])))
            placed = True
            break
        assert placed, "queue placement failed"

    if return_trace:
        return {
            "out": out,
            "entries": entries,
            "arrangement": arrangement,
            "side": side,
            "legend_box": legend_box,
            "matched_count": len(matched),
            "distractor_count": len(grey_components) - len(matched),
            "moved_cells": int(moved_cells),
            "placements": placements,
            "source_bboxes": [tuple(map(int, item["bbox"])) for item in matched],
            "source_holes": [int(item["hole"]) for item in matched],
            "pack_count": len(pack_boxes),
            "pack_boxes": [pack_boxes[k] for k in sorted(pack_boxes)],
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a tightly packed grey sprite cluster, match each sprite by hole count, then move the matches beside the legend.

    Input structure:
    - Background is the dominant color; a divider still isolates a colored legend panel near an edge or corner.
    - Grey sprites are no longer scattered. They are packed into a dense workspace cluster with only 1-cell seams between neighboring sprite bounding boxes, so the cluster looks almost fused.

    Rule:
    - Read each colored legend exemplar as a hole-count → color mapping.
    - Segment the dense grey pack back into individual 4-connected sprites, count holes for each sprite, and discard grey sprites whose hole count is absent from the legend.
    - Recolor every matching sprite, erase it from the packed source cluster, and repack the successful matches beside the legend in ascending hole-count order.

    Key insight: The task is now three-stage: visually separate near-touching grey sprites, use topology to determine their colors, and then recognize that the matched sprites migrate into an ordered queue next to the legend.
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
    return sum(_legend_span(h) for h, _ in legends) + len(legends) - 1


def _choose_legends(_arrangement: str, max_span: int) -> list[tuple[int, int]]:
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(_arrangement, max_span + 2)
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
    n = GRID_SIZE
    cross_dim = 3 if arrangement == "horizontal" else total
    along_dim = total if arrangement == "horizontal" else 3

    if corner == "tl":
        dr, dc = panel_h, panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = 1, panel_w - along_dim - 1
    elif corner == "tr":
        dr, dc = panel_h, n - 1 - panel_w
        r_min, r_max = 1, panel_h - cross_dim - 1
        c_min, c_max = dc + 2, n - 1 - along_dim
    elif corner == "bl":
        dr, dc = n - 1 - panel_h, panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = 1, panel_w - along_dim - 1
    else:
        dr, dc = n - 1 - panel_h, n - 1 - panel_w
        r_min, r_max = dr + 2, n - 1 - cross_dim
        c_min, c_max = dc + 2, n - 1 - along_dim

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
        blocked[:dr, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[: dr + 1, dc] = 9
    elif corner == "bl":
        grid[dr, : dc + 1] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, dc + 1 :] = 9
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
    kind = random.choice(["horizontal", "vertical"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    return _layout_full_vertical(grid, blocked)


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


def _workspace_bbox(blocked: np.ndarray) -> tuple[int, int, int, int]:
    free = blocked == BG
    labeled, n_comp = label_mask(free, connectivity=4)
    assert n_comp >= 1, "workspace must contain free cells"
    best = max(range(1, n_comp + 1), key=lambda lab: int(np.sum(labeled == lab)))
    pts = np.argwhere(labeled == best)
    return bbox_coords(pts)


# ---------- sprite generation ----------

def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 20 + 8 * hole_count)
    return random.randint(9 + 2 * hole_count, 16 + 3 * hole_count)


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
            p = random.choice(cands)
            forbidden.add(p)
            blob.add(p)
    return forbidden


def _find_start(center: int, csz: int, forbidden: set[tuple[int, int]]) -> tuple[int, int]:
    if (center, center) not in forbidden:
        return center, center
    for d in range(1, csz):
        for dr in range(-d, d + 1):
            for dc in range(-d, d + 1):
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
            np.array([[GREY, GREY, GREY], [GREY, BG, GREY]], dtype=int),
            np.array([[GREY, GREY, BG], [BG, GREY, GREY]], dtype=int),
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
    template = _template_sprite(hole_count)
    if template is not None:
        return template

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
    return None


# ---------- sprite planning & compact placement ----------

def _sprite_plan(legends: list[tuple[int, int]]) -> list[tuple[int, int | None]]:
    base = len(legends)
    min_total = max(TOTAL_SPRITES_RANGE[0], base + 1)
    max_total = min(TOTAL_SPRITES_RANGE[1], base + MAX_EXTRA_PER_LEGEND * len(legends) + MAX_DISTRACTORS)
    if min_total > max_total:
        return []
    total = random.randint(min_total, max_total)

    valid = []
    for n_distractors in range(1, min(MAX_DISTRACTORS, total - base) + 1):
        extra_matches = total - base - n_distractors
        if 0 <= extra_matches <= MAX_EXTRA_PER_LEGEND * len(legends):
            valid.append((n_distractors, extra_matches))
    if not valid:
        return []
    n_distractors, extra_matches = random.choice(valid)

    extras = [0] * len(legends)
    for _ in range(extra_matches):
        choices = [i for i, x in enumerate(extras) if x < MAX_EXTRA_PER_LEGEND]
        extras[random.choice(choices)] += 1

    plan = []
    for i, (hole_count, color) in enumerate(legends):
        for _ in range(1 + extras[i]):
            plan.append((hole_count, color))

    legend_holes = {h for h, _ in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    for _ in range(n_distractors):
        plan.append((random.choice(distractor_holes), None))
    random.shuffle(plan)
    return plan


def _compose_cluster_canvas(sprites: list[np.ndarray], width_limit: int) -> np.ndarray | None:
    positions: list[tuple[int, int]] = []
    x = 0
    y = 0
    row_h = 0
    row_count = 1
    max_w = 0
    for sprite in sprites:
        h, w = sprite.shape
        if x > 0 and x + w > width_limit:
            y += row_h + 1
            x = 0
            row_h = 0
            row_count += 1
        positions.append((int(y), int(x)))
        x += w + 1
        row_h = max(row_h, int(h))
        max_w = max(max_w, x - 1)

    total_h = y + row_h
    total_w = max_w
    if row_count < 2 and total_w > 18:
        return None
    if total_h > COMPACT_CLUSTER_MAX_DIM or total_w > COMPACT_CLUSTER_MAX_DIM:
        return None

    canvas = np.zeros((int(total_h), int(total_w)), dtype=int)
    for (r, c), sprite in zip(positions, sprites):
        paste(canvas, sprite, int(r), int(c), transparent=BG)

    dil = binary_dilation(canvas == GREY, structure=_STRUCT8).astype(bool)
    if label_mask(dil, connectivity=8)[1] != 1:
        return None
    return canvas


def _compact_cluster_patches(sprites: list[np.ndarray], max_h: int, max_w: int) -> list[np.ndarray]:
    min_width = max(int(sprite.shape[1]) for sprite in sprites)
    total_w = sum(int(sprite.shape[1]) for sprite in sprites) + len(sprites) - 1
    canvases = []
    for width in range(int(min_width), int(min(total_w, max_w)) + 1):
        canvas = _compose_cluster_canvas(sprites, int(width))
        if canvas is not None:
            if canvas.shape[0] > max_h or canvas.shape[1] > max_w:
                continue
            canvases.append(canvas)
    canvases.sort(key=lambda arr: (max(arr.shape), arr.shape[0] * arr.shape[1], abs(arr.shape[0] - arr.shape[1])))
    return canvases


def _place_compact_cluster(
    grid: np.ndarray, blocked: np.ndarray, layout: dict, legends: list[tuple[int, int]]
) -> bool:
    plan = _sprite_plan(legends)
    if not plan:
        return False

    items = []
    for hole_count, _color in plan:
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        items.append((sprite.size, sprite))
    items.sort(key=lambda x: (x[1].shape[0], x[1].shape[1], x[0]), reverse=True)
    sprites = [sprite for _, sprite in items]

    wr0, wc0, wr1, wc1 = _workspace_bbox(blocked)
    max_h = int(wr1 - wr0 + 1)
    max_w = int(wc1 - wc0 + 1)

    canvases = _compact_cluster_patches(sprites, max_h, max_w)
    if not canvases:
        return False

    if layout["arrangement"] == "horizontal":
        canvases.sort(key=lambda arr: (arr.shape[0], max(arr.shape), arr.shape[0] * arr.shape[1]))
    else:
        canvases.sort(key=lambda arr: (arr.shape[1], max(arr.shape), arr.shape[0] * arr.shape[1]))

    blocked_mask = blocked != BG
    for canvas in canvases:
        mask = canvas == GREY
        h, w = canvas.shape
        candidates = [(r, c) for r in range(wr0, wr1 - h + 2) for c in range(wc0, wc1 - w + 2)]
        random.shuffle(candidates)
        for r, c in candidates:
            if np.any(blocked_mask[r : r + h, c : c + w] & mask):
                continue
            paste(grid, canvas, int(r), int(c), transparent=BG)
            return True
    return False


# ---------- top-level generation ----------

def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    layout = _add_layout(grid, blocked)
    _reserve_queue_band(blocked, layout)
    legends = [(int(entry["hole"]), int(entry["color"])) for entry in layout["entries"]]

    if not _place_compact_cluster(grid, blocked, layout, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None

    try:
        trace = _solve_variant(grid, return_trace=True)
    except Exception:
        return None

    if trace["pack_count"] != 1:
        return None
    if trace["matched_count"] < len(legends):
        return None
    if trace["matched_count"] < 2:
        return None
    if trace["distractor_count"] < 1:
        return None
    if trace["moved_cells"] <= 0:
        return None
    for r0, c0, r1, c1 in trace["source_bboxes"]:
        src_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == GREY
        if np.any(trace["out"][r0 : r1 + 1, c0 : c1 + 1][src_mask] != BG):
            return None
    if np.array_equal(trace["out"], np.where(grid == GREY, BG, grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 combo input")
