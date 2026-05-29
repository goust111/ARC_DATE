# Combo variant of e3721c99 (index 26)
# combo_spec: D3/0+D6/2+D9/0
# Name: 分散图例+配对连线+一对一匹配

from __future__ import annotations

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
#       - The divider/panel structure is removed. Instead, there are exactly four
#         colored legend patches, one touching each outer side of the 30x30 grid.
#       - Top/bottom legends are horizontal 3x(2h+1) strips; left/right legends are
#         vertical (2h+1)x3 strips. Their hole counts are all distinct.
#       - There are exactly four grey sprites in the interior, also with distinct
#         hole counts, so the matching relation is strict one-to-one.
#     output:
#       - Each grey sprite is recolored to the color of the legend patch with the
#         same number of holes.
#       - After recoloring, a same-color straight connector is drawn from the
#         legend's inward-facing side to the matched sprite.
#     traced_example:
#       - Original task: several grey sprites could share a 2-hole legend and all
#         simply turn that legend color in place.
#       - This combo: if the top-edge legend has 2 holes and a single interior
#         grey sprite has 2 holes, that sprite turns the legend color and a vertical
#         line is drawn down from the top legend to that sprite. The other three
#         sides perform analogous pairings.
#       - Difference: the solver must now gather legends from all four borders,
#         rely on unique topology for one-to-one pairing, and visualize the match.
#   complexity_calibration:
#     - I keep four irregular holey sprites and four distinct legend hole counts, so
#       topology matching remains central instead of collapsing into color copying.
#     - I remove distractors because D9/0 explicitly asks for strict one-to-one
#       matching; otherwise the output would mix paired and unpaired cases and blur
#       the intended meta change.
#     - Visual variety comes from four different sides, random hole-count assignment,
#       random colors, and random organic sprite geometry, while the routing pattern
#       stays controlled enough to be learnable from 2-4 examples.
#   implicit_assumption_review:
#     original_main_dependencies:
#       - The seed solver assumed legend components do not touch the border. That is
#         false here; border contact is now the main cue for legend detection.
#       - The seed solver allowed many grey components to map to the same hole count.
#         That is false here; grey hole counts are unique and match legends exactly.
#       - The seed solver never needed geometric pairing information after coloring.
#         That is false here because connector endpoints must be recovered.
#     new_main_contract:
#       - Every non-bg, non-grey colored component is a legend patch touching exactly
#         one border and representing one unique hole count.
#       - There are exactly four grey components, each with a distinct hole count,
#         and the set of grey hole counts equals the set of legend hole counts.
#       - For each matched sprite, a boundary cell exists on the side facing its
#         legend so a straight connector can be drawn without ambiguity.
#   placement_legality:
#     - The generator places one legend on each side with corner margins so legends
#       never touch each other.
#     - Each side gets its own sprite-port band in a different quadrant; this keeps
#       connectors non-crossing and leaves a buffer between objects.
#     - Corridors are reserved after each placement, and future sprites are blocked
#       from both occupied cells and dilated corridor cells.
#   visual_observability:
#     - The four-side scatter is obvious in the input because each border hosts a
#       single isolated legend patch.
#     - The output adds same-color inward rays, making the one-to-one pairing rule
#       directly visible rather than hidden in bookkeeping.
#     - No sorting by color id or other non-visual metadata is used; side location
#       and hole count are the only drivers.
#   cleanliness:
#     - Four legends and four sprites are enough to show the fusion without turning
#       the grid into a dense wiring puzzle.
#     - Straight single-cell connectors are easier to parse than bent or branching
#       routes, and side-specific corridor bands keep the scene readable.
#   color_inference:
#     - Colors are still justified structurally: the legend shape's hole count
#       determines which grey sprite receives that color.
#     - The connector color duplicates the sprite color, reinforcing the pairing
#       instead of introducing a second independent color rule.


BG = 0
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
MAX_HOLE_COUNT = 6

CORNER_MARGIN = 4
LEGEND_CENTER_BANDS = {
    "top": (17, 21),
    "bottom": (8, 12),
    "left": (8, 12),
    "right": (17, 21),
}
SPRITE_PORT_BANDS = {
    "top": (8, 12),
    "bottom": (17, 21),
    "left": (8, 12),
    "right": (17, 21),
}

SPRITE_RETRIES = 40
PLACEMENT_RETRIES = 6
GEN_RETRIES = 400
MIN_CONNECTOR_SPAN = 4

SPRITE_AREA_0 = (6, 18)
SPRITE_AREA_LOW_MIN = 10
CANVAS_SCALE = 1.2
CANVAS_MIN = 5
CANVAS_PAD = 2
HOLE_SEED_RADIUS_FACTOR = 0.45
HOLE_SEED_MIN_RADIUS = 2
HOLE_SEED_MIN_SEP = 2
HOLE_SEED_MARGIN = 2
SEAL_ITERS = 5
SPRITE_MAX_H = 12
SPRITE_MAX_W = 14
SPRITE_MAX_ASPECT = 2.4

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


def _direction_for_side(side: str) -> str:
    return {
        "top": "down",
        "bottom": "up",
        "left": "right",
        "right": "left",
    }[side]


def _legend_span(hole_count: int) -> int:
    return 3 if hole_count == 0 else 2 * hole_count + 1


def _legend_patch(hole_count: int, side: str, color: int) -> np.ndarray:
    span = _legend_span(hole_count)
    if side in ("top", "bottom"):
        patch = np.full((3, span), int(color), dtype=int)
        if hole_count:
            patch[1, 1::2] = BG
        return patch
    patch = np.full((span, 3), int(color), dtype=int)
    if hole_count:
        patch[1::2, 1] = BG
    return patch


def _legend_port(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r1, (c0 + c1) // 2
    if side == "bottom":
        return r0, (c0 + c1) // 2
    if side == "left":
        return (r0 + r1) // 2, c1
    if side == "right":
        return (r0 + r1) // 2, c0
    raise ValueError(f"unknown side {side}")


def _touched_sides(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[str]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    out: list[str] = []
    if r0 == 0:
        out.append("top")
    if r1 == h - 1:
        out.append("bottom")
    if c0 == 0:
        out.append("left")
    if c1 == w - 1:
        out.append("right")
    return out


def _extract_legends(g: np.ndarray, bg: int, grey: int) -> dict[int, dict]:
    infos: list[dict] = []
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        candidates = []
        for pts in comps:
            bbox = bbox_coords(pts)
            r0, c0, r1, c1 = bbox
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            sides = _touched_sides(bbox, g.shape)
            if len(sides) != 1:
                continue
            candidates.append((bbox, sides[0]))
        assert len(candidates) == 1, f"color {color} must have exactly one border legend component"
        bbox, side = candidates[0]
        r0, c0, r1, c1 = bbox
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
        hole_count = _hole_count(sub)
        info = {
            "color": int(color),
            "hole_count": int(hole_count),
            "bbox": bbox,
            "side": side,
            "direction": _direction_for_side(side),
            "port": _legend_port(bbox, side),
        }
        infos.append(info)

    assert len(infos) == 4, "combo variant requires exactly four legends"
    assert {info["side"] for info in infos} == set(SIDES), "legends must cover all four sides"

    mapping: dict[int, dict] = {}
    for info in infos:
        hc = int(info["hole_count"])
        assert hc not in mapping, f"legend hole count {hc} must be unique"
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


def _select_sprite_port(
    pts: np.ndarray, direction: str, target: tuple[int, int]
) -> tuple[int, int]:
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
            return min(aligned, key=lambda p: (p[0], p[1]))
        return min(side, key=lambda p: (abs(p[1] - c_t), p[0], p[1]))

    aligned = [p for p in side if p[0] == r_t]
    if aligned:
        return min(aligned, key=lambda p: (p[1], p[0]))
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
    Match four interior grey sprites to four border legends by hole count, then draw the pairing rays.

    Input structure:
    - The background is the dominant color, grey sprites (5) appear in the interior, and colored legend patches are scattered on the top, bottom, left, and right edges.
    - Each legend patch is a single border-touching component whose number of enclosed holes is unique across the four sides.
    - There is exactly one grey sprite for each legend hole count, so the set of grey hole counts matches the set of legend hole counts one-to-one.

    Rule:
    - Detect the four border legends, count their holes, and build a unique hole-count -> legend-color mapping together with each legend's inward-facing port.
    - Count holes for each grey sprite and recolor that sprite to the color of the legend with the same hole count.
    - After recoloring, draw a same-color straight connector from each legend port to the boundary cell of its matched sprite that faces the legend.

    Key insight: The classification cue is still topological hole count, but now the legends must be collected from all four borders and each successful match is made explicit by a non-crossing colored ray from the edge inward.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY

    legend = _extract_legends(g, bg, grey)

    grey_comps = components_mask(g == grey, connectivity=4)
    assert len(grey_comps) == len(legend), "one-to-one variant requires exactly one grey sprite per legend"

    sprite_infos: list[dict] = []
    seen_holes: set[int] = set()
    for pts in grey_comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        assert hc in legend, f"grey sprite hole count {hc} must match some legend"
        assert hc not in seen_holes, "grey sprite hole counts must be unique in the one-to-one variant"
        seen_holes.add(hc)
        sprite_infos.append({"pts": pts, "hole_count": int(hc)})

    assert seen_holes == set(legend), "grey and legend hole-count sets must match exactly"

    out = g.copy()
    connections: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for info in sprite_infos:
        pts = info["pts"]
        legend_info = legend[info["hole_count"]]
        color = int(legend_info["color"])
        for r, c in pts:
            out[r, c] = color
        sprite_port = _select_sprite_port(pts, legend_info["direction"], legend_info["port"])
        connections.append((legend_info["port"], sprite_port, color))

    for legend_port, sprite_port, color in connections:
        _draw_connector(out, legend_port, sprite_port, color)

    return out


def _sprite_area(hole_count: int) -> int:
    if hole_count == 0:
        return random.randint(*SPRITE_AREA_0)
    if hole_count <= 2:
        return random.randint(SPRITE_AREA_LOW_MIN, 22 + 8 * hole_count)
    return random.randint(12 + 2 * hole_count, 18 + 3 * hole_count)


def _place_hole_seeds(
    hole_count: int, center: int, csz: int, area: int
) -> list[tuple[int, int]] | None:
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


def _grow_hole_blobs(
    seeds: list[tuple[int, int]], hole_count: int, area: int, csz: int
) -> set[tuple[int, int]]:
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
                    if any(
                        (nnr, nnc) in forbidden and (nnr, nnc) not in blob
                        for nnr, nnc in neighbors4(nr, nc)
                    ):
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


def _grow_mask(
    csz: int, area: int, start: tuple[int, int], forbidden: set[tuple[int, int]]
) -> np.ndarray:
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


def _crop_and_validate(
    mask: np.ndarray, forbidden: set[tuple[int, int]], hole_count: int
) -> np.ndarray | None:
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


def _placement_ok(
    static_blocked: np.ndarray,
    forbidden_pad: np.ndarray,
    sprite_mask: np.ndarray,
    r: int,
    c: int,
) -> bool:
    h, w = sprite_mask.shape
    rr0, cc0 = max(0, r - 1), max(0, c - 1)
    rr1, cc1 = min(GRID_SIZE, r + h + 1), min(GRID_SIZE, c + w + 1)
    if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
        return False
    return not np.any(forbidden_pad[r : r + h, c : c + w] & sprite_mask)


def _corridor_clear(
    static_blocked: np.ndarray,
    occupied: np.ndarray,
    line_blocked: np.ndarray,
    corridor: list[tuple[int, int]],
    legend_port: tuple[int, int],
    sprite_port: tuple[int, int],
) -> bool:
    for r, c in corridor:
        if (r, c) == legend_port or (r, c) == sprite_port:
            continue
        if static_blocked[r, c] != BG or occupied[r, c] != BG or line_blocked[r, c]:
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
    band = SPRITE_PORT_BANDS[legend_info["side"]]
    side_cells = _side_cells(sprite_mask, direction)
    if not side_cells:
        return None

    forbidden_pad = _dilate_mask((occupied != BG) | line_blocked)
    random.shuffle(side_cells)

    for ar, ac in side_cells:
        if direction in ("down", "up"):
            c = port[1] - ac
            if c < 0 or c + w > GRID_SIZE:
                continue
            candidates = list(range(band[0], band[1] + 1))
            random.shuffle(candidates)
            for sprite_port_r in candidates:
                r = sprite_port_r - ar
                if r < 0 or r + h > GRID_SIZE:
                    continue
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = np.argwhere(sprite_mask) + np.array([r, c])
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[1] != port[1] or not (band[0] <= sprite_port[0] <= band[1]):
                    continue
                if direction == "down" and sprite_port[0] - port[0] < MIN_CONNECTOR_SPAN:
                    continue
                if direction == "up" and port[0] - sprite_port[0] < MIN_CONNECTOR_SPAN:
                    continue
                corridor = _corridor_cells(port, sprite_port)
                if _corridor_clear(static_blocked, occupied, line_blocked, corridor, port, sprite_port):
                    return int(r), int(c), sprite_port
        else:
            r = port[0] - ar
            if r < 0 or r + h > GRID_SIZE:
                continue
            candidates = list(range(band[0], band[1] + 1))
            random.shuffle(candidates)
            for sprite_port_c in candidates:
                c = sprite_port_c - ac
                if c < 0 or c + w > GRID_SIZE:
                    continue
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = np.argwhere(sprite_mask) + np.array([r, c])
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[0] != port[0] or not (band[0] <= sprite_port[1] <= band[1]):
                    continue
                if direction == "right" and sprite_port[1] - port[1] < MIN_CONNECTOR_SPAN:
                    continue
                if direction == "left" and port[1] - sprite_port[1] < MIN_CONNECTOR_SPAN:
                    continue
                corridor = _corridor_cells(port, sprite_port)
                if _corridor_clear(static_blocked, occupied, line_blocked, corridor, port, sprite_port):
                    return int(r), int(c), sprite_port
    return None


def _pick_legend_center(side: str, span: int) -> int:
    half = span // 2
    lo_pref, hi_pref = LEGEND_CENTER_BANDS[side]
    lo = max(CORNER_MARGIN + half, lo_pref)
    hi = min(GRID_SIZE - 1 - CORNER_MARGIN - half, hi_pref)
    assert lo <= hi, f"no legal center band for side {side} span {span}"
    return random.randint(lo, hi)


def _place_legends(grid: np.ndarray) -> list[dict]:
    holes = random.sample(range(MAX_HOLE_COUNT), 4)
    colors = random.sample(LEGEND_COLORS, 4)
    pairs = list(zip(holes, colors))
    random.shuffle(pairs)

    infos: list[dict] = []
    for side, (hole_count, color) in zip(SIDES, pairs):
        span = _legend_span(hole_count)
        patch = _legend_patch(hole_count, side, color)
        center = _pick_legend_center(side, span)
        if side == "top":
            r0, c0 = 0, center - span // 2
        elif side == "bottom":
            r0, c0 = GRID_SIZE - 3, center - span // 2
        elif side == "left":
            r0, c0 = center - span // 2, 0
        else:
            r0, c0 = center - span // 2, GRID_SIZE - 3

        paste(grid, patch, r0, c0)
        bbox = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
        infos.append(
            {
                "side": side,
                "hole_count": int(hole_count),
                "color": int(color),
                "bbox": bbox,
                "direction": _direction_for_side(side),
                "port": _legend_port(bbox, side),
            }
        )

    return infos


def _place_sprites(grid: np.ndarray, expected: np.ndarray, legends: list[dict]) -> bool:
    static_blocked = np.where(grid != BG, 9, 0)
    occupied = np.zeros_like(grid)
    line_blocked = np.zeros_like(grid, dtype=bool)

    legends = sorted(legends, key=lambda info: int(info["hole_count"]), reverse=True)
    for info in legends:
        placed = None
        for _ in range(PLACEMENT_RETRIES):
            sprite = _make_sprite(info["hole_count"])
            if sprite is None:
                continue
            placed = _aligned_sprite_location(static_blocked, occupied, line_blocked, sprite, info)
            if placed is None:
                continue
            r, c, sprite_port = placed
            paste(grid, sprite, r, c, transparent=BG)
            paste(occupied, sprite, r, c, transparent=BG)
            recolored = np.where(sprite == GREY, int(info["color"]), BG)
            paste(expected, recolored, r, c, transparent=BG)
            _draw_connector(expected, info["port"], sprite_port, int(info["color"]))
            for rr, cc in _corridor_cells(info["port"], sprite_port):
                line_blocked[rr, cc] = True
            break
        if placed is None:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    legends = _place_legends(grid)
    expected[:, :] = grid
    if not _place_sprites(grid, expected, legends):
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
    raise RuntimeError("failed to generate e3721c99 combo input")
