# Variant of e3721c99
# Dimension: D6 — 轨迹：源精灵着色后的额外处理 (trajectory)
# Change: 无轨迹效果——精灵在原位被重着色，不涉及移动或路径... → 着色后精灵与图例之间画连线

from __future__ import annotations

import itertools
import random
import sys

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis (YAML-style, written before implementation as required):
# design:
#   concrete_behavior:
#     input:
#       - Same core scene as the original task: a divider isolates a legend panel,
#         colored legend shapes encode hole-count -> color, and grey sprites in the
#         workspace must be matched by hole count.
#       - No trajectory marks appear in the input. The input remains visually close
#         to the seed task so the new behavior is attributable to output-only D6.
#     output:
#       - Every matched grey sprite is recolored to its legend color.
#       - In addition, a same-color straight connector is drawn from that sprite to
#         the matching legend patch. The connector is axis-aligned and continuous.
#       - Distractor sprites with hole counts absent from the legend are still erased.
#     traced_example:
#       - Original: a 2-hole grey blob below a 2-hole red legend simply turns red.
#       - Variant: that blob still turns red, then a red vertical or horizontal line
#         is added until it visibly reaches the red legend patch.
#       - Difference: topology-based recoloring still decides identity; D6 adds an
#         explicit pairing visualization layer after recoloring.
#   complexity_calibration:
#     - I keep 2-4 legend entries and irregular holey sprites, so the core inference
#       problem is unchanged.
#     - I intentionally restrict matched sprites to one per legend. Allowing 2-3
#       matches per legend would require overlapping or branching routes and would
#       make the output cluttered instead of communicative.
#     - Distractors remain present so the task does not collapse into a trivial
#       color-copy exercise.
#   implicit_assumption_review:
#     original_main_dependencies:
#       - _legend_mapping assumes each legend color has exactly one interior component.
#         This still holds because only the output gains lines; the input does not.
#       - The original recoloring loop assumes writing sprite colors in-place is the
#         entire transformation. That is no longer sufficient.
#       - The original code is safe because it never needs sprite-to-legend geometry.
#         The variant now depends on a reproducible connector endpoint on both sides.
#     new_main_contract:
#       - For each legend, the workspace-facing direction can be inferred from the
#         legends' shared orientation and which half of the grid they occupy.
#       - For each matched sprite, there exists at least one boundary cell aligned on
#         the legend's connector axis; the generator must guarantee this.
#       - Connector corridors are kept free of other sprites, so drawing lines after
#         recoloring does not overwrite unrelated objects.
#   placement_legality:
#     - Legends keep the original divider layouts (full row, full column, or L).
#     - Each legend gets a workspace-facing port on its outer solid edge.
#     - Matched sprites are placed first, constrained to the same row/column lane as
#       their legend port, with a minimum visible gap for the connector.
#     - Reserved connector lanes are then treated as blocked when placing distractors.
#   visual_observability:
#     - The added behavior is directly visible as colored answer-style links.
#     - Straight connectors are easier to infer from 2-4 examples than flood-filled
#       regions or multi-turn paths.
#     - No hidden ordering by color id is used; all routing is determined by visible
#       geometry: legend orientation, panel side, and aligned sprite placement.
#   cleanliness:
#     - Only one matched sprite per legend avoids color tangles and branch ambiguity.
#     - Straight single-cell-width connectors keep the workspace readable.
#     - Distinct legend lanes are preserved by the original one-cell gaps plus
#       corridor reservation during placement.
#   color_inference:
#     - The color mapping remains structurally justified by the legend shapes' hole
#       counts, exactly as in the original task.
#     - The new same-color connectors reinforce, rather than replace, that mapping.


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


def _legend_port(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction == "down":
        return r1, (c0 + c1) // 2
    if direction == "up":
        return r0, (c0 + c1) // 2
    if direction == "right":
        return (r0 + r1) // 2, c1
    if direction == "left":
        return (r0 + r1) // 2, c0
    raise ValueError(f"unknown direction {direction}")


def _extract_legends(g: np.ndarray, bg: int, grey: int) -> dict[int, dict]:
    """
    Recover legend metadata from the input.

    Each legend is the unique interior component of its color. The panel orientation
    is shared across the sample, so the legends' average position determines which
    side faces the workspace and therefore where connector lines must terminate.
    """
    h, w = g.shape
    infos: list[dict] = []
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
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        assert len(candidates) == 1, f"color {color} must have exactly one legend component"
        bbox = candidates[0]
        r0, c0, r1, c1 = bbox
        sub = (g[r0 : r1 + 1, c0 : c1 + 1] == color)
        infos.append(
            {
                "color": int(color),
                "hole_count": _hole_count(sub),
                "bbox": bbox,
            }
        )

    mapping: dict[int, dict] = {}
    assert infos, "legend mapping must be non-empty: grid must contain colored legend shapes"

    heights = {info["bbox"][2] - info["bbox"][0] + 1 for info in infos}
    widths = {info["bbox"][3] - info["bbox"][1] + 1 for info in infos}
    if heights == {3}:
        direction = "down" if np.mean([(d["bbox"][0] + d["bbox"][2]) / 2 for d in infos]) < h / 2 else "up"
        arrangement = "horizontal"
    else:
        assert widths == {3}, "legend patches must share a 3-cell minor axis"
        direction = "right" if np.mean([(d["bbox"][1] + d["bbox"][3]) / 2 for d in infos]) < w / 2 else "left"
        arrangement = "vertical"

    for info in infos:
        info["direction"] = direction
        info["arrangement"] = arrangement
        info["port"] = _legend_port(info["bbox"], direction)
        hc = int(info["hole_count"])
        assert hc not in mapping, f"legend hole count {hc} must map to exactly one color"
        mapping[hc] = info
    return mapping


def _side_cells(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    occ = {tuple(map(int, p)) for p in pts}
    out: list[tuple[int, int]] = []
    for r, c in occ:
        if direction == "down" and (r - 1, c) not in occ:
            out.append((r, c))
        elif direction == "up" and (r + 1, c) not in occ:
            out.append((r, c))
        elif direction == "right" and (r, c - 1) not in occ:
            out.append((r, c))
        elif direction == "left" and (r, c + 1) not in occ:
            out.append((r, c))
    return out


def _select_sprite_port(pts: np.ndarray, direction: str, target: tuple[int, int]) -> tuple[int, int]:
    r_t, c_t = target
    coords = {(int(r), int(c)) for r, c in pts}
    side = []
    for r, c in coords:
        if direction == "down" and (r - 1, c) not in coords:
            side.append((r, c))
        elif direction == "up" and (r + 1, c) not in coords:
            side.append((r, c))
        elif direction == "right" and (r, c - 1) not in coords:
            side.append((r, c))
        elif direction == "left" and (r, c + 1) not in coords:
            side.append((r, c))
    assert side, "matched sprite must expose a connector side"

    if direction in ("down", "up"):
        aligned = [p for p in side if p[1] == c_t]
        if aligned:
            key = (lambda p: p[0]) if direction == "down" else (lambda p: -p[0])
            return min(aligned, key=key)
        return min(side, key=lambda p: (abs(p[1] - c_t), p[0], p[1]))

    aligned = [p for p in side if p[0] == r_t]
    if aligned:
        key = (lambda p: p[1]) if direction == "right" else (lambda p: -p[1])
        return min(aligned, key=key)
    return min(side, key=lambda p: (abs(p[0] - r_t), p[1], p[0]))


def _corridor_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(r, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, c) for r in range(r0, r1 + 1)]

    elbow = (a[0], b[1])
    cells1 = _corridor_cells(a, elbow)
    cells2 = _corridor_cells(elbow, b)
    return cells1 + cells2[1:]


def _draw_connector(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    for r, c in _corridor_cells(a, b):
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey sprites by hole count, then link each successful match back to its legend.

    Input structure:
    - The background is the dominant color, grey sprites (5) occupy the workspace, and a divider isolates a compact legend panel.
    - Each legend patch is a colored interior component whose number of enclosed holes identifies one output color.
    - The panel is arranged as a horizontal strip or a vertical strip, which determines the direction from each legend toward the workspace.

    Rule:
    - Count holes for every legend patch to obtain a hole-count -> color mapping.
    - Count holes for every grey sprite; matching sprites are recolored to the corresponding legend color, while unmatched sprites are erased to background.
    - After recoloring, draw a same-color straight connector from each matched sprite to the workspace-facing side of its legend patch.

    Key insight: The task still classifies sprites by topology, but the final answer also visualizes the pairing by extending a colored route from each matched sprite back to the legend that explains it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = 5

    legend = _extract_legends(g, bg, grey)

    out = g.copy()
    connections: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    grey_comps = components_mask(g == grey, connectivity=4)
    for pts in grey_comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        if hc not in legend:
            for r, c in pts:
                out[r, c] = bg
            continue

        info = legend[hc]
        newc = int(info["color"])
        for r, c in pts:
            out[r, c] = newc
        sprite_port = _select_sprite_port(pts, info["direction"], info["port"])
        connections.append((info["port"], sprite_port, newc))

    for legend_port, sprite_port, color in connections:
        _draw_connector(out, legend_port, sprite_port, color)

    return out


# --------------- generator (holes-first organic blob + aligned connectors) ---------------

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
TOTAL_SPRITES_RANGE = (6, 8)
MAX_DISTRACTORS = 4
SPRITE_RETRIES = 30
SPRITE_AREA_0 = (5, 25)
SPRITE_AREA_LOW_MIN = 8
CANVAS_SCALE = 1.2
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
ALIGN_RETRIES = 80
MIN_CONNECTOR_SPAN = 3

# -- generation --
GEN_RETRIES = 300

_STRUCT8 = generate_binary_structure(2, 2)


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


def _choose_legends(max_span: int) -> list[tuple[int, int]]:
    count = random.randint(*LEGEND_COUNT_RANGE)
    combos = [
        holes
        for holes in itertools.combinations(range(MAX_HOLE_COUNT), count)
        if sum(_legend_span(h) for h in holes) + count - 1 <= max_span
    ]
    if not combos:
        return _choose_legends(max_span + 2)
    holes = list(random.choice(combos))
    random.shuffle(holes)
    colors = random.sample(LEGEND_COLORS, count)
    return list(zip(holes, colors))


def _place_legend_line(
    grid: np.ndarray,
    legends: list[tuple[int, int]],
    arrangement: str,
    direction: str,
    r: int,
    c: int,
) -> list[dict]:
    infos: list[dict] = []
    rr, cc = int(r), int(c)
    for hole_count, color in legends:
        patch = _legend_patch(hole_count, color, arrangement)
        paste(grid, patch, rr, cc)
        bbox = (rr, cc, rr + patch.shape[0] - 1, cc + patch.shape[1] - 1)
        infos.append(
            {
                "hole_count": int(hole_count),
                "color": int(color),
                "bbox": bbox,
                "direction": direction,
                "arrangement": arrangement,
                "port": _legend_port(bbox, direction),
            }
        )
        if arrangement == "horizontal":
            cc += patch.shape[1] + 1
        else:
            rr += patch.shape[0] + 1
    return infos


def _layout_full_horizontal(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends(FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    top = random.choice([True, False])
    c0 = random.randint(1, GRID_SIZE - 1 - total)
    if top:
        div_r = panel_h
        grid[div_r, :] = DIV
        blocked[: div_r + 1, :] = 9
        r0 = random.randint(1, panel_h - 4)
        direction = "down"
    else:
        div_r = GRID_SIZE - 1 - panel_h
        grid[div_r, :] = DIV
        blocked[div_r:, :] = 9
        r0 = random.randint(div_r + 2, GRID_SIZE - 4)
        direction = "up"
    return _place_legend_line(grid, legends, "horizontal", direction, r0, c0)


def _layout_full_vertical(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    legends = _choose_legends(FULL_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    left = random.choice([True, False])
    r0 = random.randint(1, GRID_SIZE - 1 - total)
    if left:
        div_c = panel_w
        grid[:, div_c] = DIV
        blocked[:, : div_c + 1] = 9
        c0 = random.randint(1, panel_w - 4)
        direction = "right"
    else:
        div_c = GRID_SIZE - 1 - panel_w
        grid[:, div_c] = DIV
        blocked[:, div_c:] = 9
        c0 = random.randint(div_c + 2, GRID_SIZE - 4)
        direction = "left"
    return _place_legend_line(grid, legends, "vertical", direction, r0, c0)


def _l_corner_geometry(
    corner: str, panel_h: int, panel_w: int, arrangement: str, total: int
) -> tuple[int, int, int, int, int, int]:
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


def _layout_l(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    arrangement = random.choice(["horizontal", "vertical"])
    legends = _choose_legends(L_LAYOUT_MAX_SPAN)
    total = _line_span(legends)
    along = total + 2 + random.randint(0, min(L_LAYOUT_EXTRA_PADDING, 27 - (total + 2)))
    cross = random.randint(*PANEL_SIZE_RANGE)
    panel_h, panel_w = (cross, along) if arrangement == "horizontal" else (along, cross)
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
        blocked[dr + 1 :, :dc] = 9
        blocked[dr, : dc + 1] = 9
        blocked[dr:, dc] = 9
    else:
        grid[dr, dc:] = DIV
        grid[dr:, dc] = DIV
        blocked[dr + 1 :, dc + 1 :] = 9
        blocked[dr, dc:] = 9
        blocked[dr:, dc] = 9

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    if arrangement == "horizontal":
        direction = "down" if corner in ("tl", "tr") else "up"
    else:
        direction = "right" if corner in ("tl", "bl") else "left"
    return _place_legend_line(grid, legends, arrangement, direction, r0, c0)


def _add_layout(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    kind = random.choice(["horizontal", "vertical", "l"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    if kind == "vertical":
        return _layout_full_vertical(grid, blocked)
    return _layout_l(grid, blocked)


def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 25 + 10 * hole_count)
    return random.randint(10 + 2 * hole_count, 20 + 4 * hole_count)


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
    max_hole = max(2, area // (hole_count + 2)) if hole_count <= 2 else max(1, area // (hole_count * 2 + 1))
    forbidden: set[tuple[int, int]] = set()
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
            pick = random.choice(cands)
            forbidden.add(pick)
            blob.add(pick)
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
    h, w = rmax - rmin + 1, cmax - cmin + 1
    if h > SPRITE_MAX_H or w > SPRITE_MAX_W or max(h, w) > SPRITE_MAX_ASPECT * min(h, w):
        return None
    cropped = mask[rmin : rmax + 1, cmin : cmax + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(hole_count: int) -> np.ndarray | None:
    for _ in range(SPRITE_RETRIES):
        area = _sprite_area(hole_count)
        csz = max(CANVAS_MIN, int(area ** 0.5 * CANVAS_SCALE) + CANVAS_PAD)
        center = csz // 2
        forbidden: set[tuple[int, int]] = set()
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


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _placement_ok(static_blocked: np.ndarray, forbidden_pad: np.ndarray, sprite_mask: np.ndarray, r: int, c: int) -> bool:
    h, w = sprite_mask.shape
    rr0, cc0 = max(0, r - 1), max(0, c - 1)
    rr1, cc1 = min(GRID_SIZE, r + h + 1), min(GRID_SIZE, c + w + 1)
    if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
        return False
    return not np.any(forbidden_pad[r : r + h, c : c + w] & sprite_mask)


def _corridor_clear(occupied: np.ndarray, line_blocked: np.ndarray, corridor: list[tuple[int, int]], sprite_port: tuple[int, int]) -> bool:
    for r, c in corridor:
        if (r, c) == sprite_port:
            continue
        if occupied[r, c] != BG or line_blocked[r, c]:
            return False
    return True


def _aligned_sprite_location(
    static_blocked: np.ndarray,
    occupied: np.ndarray,
    line_blocked: np.ndarray,
    sprite: np.ndarray,
    legend_info: dict,
) -> tuple[int, int, tuple[int, int]] | None:
    sprite_mask = sprite == GREY
    h, w = sprite.shape
    port = legend_info["port"]
    direction = legend_info["direction"]
    side_cells = _side_cells(sprite_mask, direction)
    if not side_cells:
        return None

    random.shuffle(side_cells)
    forbidden_pad = _dilate_mask((occupied != BG) | line_blocked)

    for ar, ac in side_cells:
        if direction in ("down", "up"):
            c = port[1] - ac
            if c < 0 or c + w > GRID_SIZE:
                continue
            if direction == "down":
                r_min = max(0, port[0] + MIN_CONNECTOR_SPAN - ar)
                r_max = GRID_SIZE - h
            else:
                r_min = 0
                r_max = min(GRID_SIZE - h, port[0] - MIN_CONNECTOR_SPAN - ar)
            if r_min > r_max:
                continue
            rows = list(range(r_min, r_max + 1))
            random.shuffle(rows)
            for r in rows[:ALIGN_RETRIES]:
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = np.argwhere(sprite_mask) + np.array([r, c])
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[1] != port[1]:
                    continue
                corridor = _corridor_cells(port, sprite_port)
                if _corridor_clear(occupied, line_blocked, corridor, sprite_port):
                    return int(r), int(c), sprite_port
        else:
            r = port[0] - ar
            if r < 0 or r + h > GRID_SIZE:
                continue
            if direction == "right":
                c_min = max(0, port[1] + MIN_CONNECTOR_SPAN - ac)
                c_max = GRID_SIZE - w
            else:
                c_min = 0
                c_max = min(GRID_SIZE - w, port[1] - MIN_CONNECTOR_SPAN - ac)
            if c_min > c_max:
                continue
            cols = list(range(c_min, c_max + 1))
            random.shuffle(cols)
            for c in cols[:ALIGN_RETRIES]:
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = np.argwhere(sprite_mask) + np.array([r, c])
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[0] != port[0]:
                    continue
                corridor = _corridor_cells(port, sprite_port)
                if _corridor_clear(occupied, line_blocked, corridor, sprite_port):
                    return int(r), int(c), sprite_port
    return None


def _random_sprite_location(
    static_blocked: np.ndarray,
    occupied: np.ndarray,
    line_blocked: np.ndarray,
    sprite: np.ndarray,
) -> tuple[int, int] | None:
    h, w = sprite.shape
    sprite_mask = sprite == GREY
    forbidden_pad = _dilate_mask((occupied != BG) | line_blocked)
    candidates = [(r, c) for r in range(GRID_SIZE - h + 1) for c in range(GRID_SIZE - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
            return int(r), int(c)
    return None


def _sprite_plan(legends: list[dict]) -> list[int]:
    total = random.randint(max(TOTAL_SPRITES_RANGE[0], len(legends) + 1), TOTAL_SPRITES_RANGE[1])
    n_distractors = min(MAX_DISTRACTORS, total - len(legends))
    legend_holes = {info["hole_count"] for info in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    assert distractor_holes, "there must be at least one distractor hole count"
    return [random.choice(distractor_holes) for _ in range(n_distractors)]


def _place_sprites(
    grid: np.ndarray, expected: np.ndarray, blocked: np.ndarray, legends: list[dict]
) -> bool:
    matched_items = []
    for info in legends:
        sprite = _make_sprite(info["hole_count"])
        if sprite is None:
            return False
        matched_items.append((sprite.size, sprite, info))
    matched_items.sort(key=lambda item: item[0], reverse=True)

    occupied = np.zeros_like(grid)
    line_blocked = np.zeros_like(grid, dtype=bool)

    for _, sprite, info in matched_items:
        placed = _aligned_sprite_location(blocked, occupied, line_blocked, sprite, info)
        if placed is None:
            return False
        r, c, sprite_port = placed
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)

        out_patch = np.where(sprite == GREY, int(info["color"]), BG)
        paste(expected, out_patch, r, c, transparent=BG)
        _draw_connector(expected, info["port"], sprite_port, int(info["color"]))
        for rr, cc in _corridor_cells(info["port"], sprite_port):
            line_blocked[rr, cc] = True

    distractors = []
    for hole_count in _sprite_plan(legends):
        sprite = _make_sprite(hole_count)
        if sprite is None:
            return False
        distractors.append((sprite.size, sprite))
    distractors.sort(key=lambda item: item[0], reverse=True)

    for _, sprite in distractors:
        loc = _random_sprite_location(blocked, occupied, line_blocked, sprite)
        if loc is None:
            return False
        r, c = loc
        paste(grid, sprite, r, c, transparent=BG)
        paste(occupied, sprite, r, c, transparent=BG)

    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _add_layout(grid, blocked)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e3721c99 D6 input")
