# Combo variant of e3721c99 (index 13)
# combo_spec: D6/2+D11/2
# Name: 链式传播+着色连线

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


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The original task colors each grey sprite independently from its own hole count.
#    - In the combo version, direct hole-count matches are only the seeds. After the
#      first recoloring pass, those seeds emit same-color connector lines back to their
#      legends, and uncolored sprites may inherit a color either by touching a colored
#      sprite diagonally/orthogonally or by being crossed by one of those connector
#      lines. Newly colored followers then emit their own connector lines as well.
#    - Example trace for a top legend:
#      1. Seed S is the only component with the legend's hole count, so S becomes red.
#      2. The red connector from the legend to S passes through follower F1, whose own
#         hole count is not listed, so F1 becomes red because it lies on the red line.
#      3. Side follower F2 touches F1 diagonally, so in the next round F2 becomes red.
#      4. Output shows all three red plus their red links to the same legend.
#    - Difference from the base task: non-legend hole counts can survive, but only
#      through a visible chain anchored by a matched seed and its connector.
# b) Complexity calibration:
#    - I keep the original legend semantics and irregular holey sprites so topology is
#      still necessary.
#    - I limit the extra dynamics to two visible support modes: line-crossing and
#      8-neighbor propagation. This is enough to be learnable from a few examples.
#    - Each sample contains at least two chain groups, so the fused behavior repeats
#      instead of appearing as a one-off coincidence.
# c) Implicit assumption review:
#    - The original main() assumed each grey component could be finalized in one pass.
#      That fails here because some non-matching components must remain available for
#      later propagation.
#    - The D6 single-variant assumed only directly matched sprites ever get lines.
#      That is too weak for the combo; propagated followers must also connect back to
#      their inherited legend color.
#    - The propagation step now depends on reproducible connector geometry, so legend
#      direction and sprite ports must be inferable from the input alone.
#    - A new hidden risk is line ambiguity. The generator therefore keeps different
#      color corridors separated so an uncolored component sees at most one candidate
#      color when it is meant to propagate.
# d) Placement legality:
#    - Legends keep the original divider layouts.
#    - Each chain group is placed in the workspace without overlap, with one-cell
#      clearance between unrelated groups.
#    - For chain groups, the seed is aligned with its legend lane, the inline follower
#      sits between legend and seed on that lane, and the side follower touches only
#      the inline follower diagonally.
#    - Final connector corridors are reserved before other groups or distractors are
#      placed, preventing accidental crossings between different colors.
# e) Visual observability:
#    - Humans can see three distinct cues: direct legend-matching seeds, non-matching
#      sprites recolored because a colored line runs through them, and additional
#      sprites recolored because they touch an already recolored follower.
#    - The connector geometry is determined by panel orientation and visible positions,
#      not by hidden color ordering or numeric ids.
# f) Visual cleanliness:
#    - I use at most three legend entries and two chain groups to avoid excessive line
#      clutter.
#    - Connector paths prefer the workspace-facing axis first, so branches emerge from
#      the main lane near the follower instead of scribbling across the legend panel.
#    - Divider color is excluded from legend colors to keep connectors distinguishable.
# g) Color-change inferability:
#    - Colors still originate only from legend hole counts.
#    - Followers do not introduce new mappings; they inherit the single visible color
#      that reaches them through a touched component or a connector line.
# h) Quantitative anti-degeneration target:
#    - Every accepted sample must contain at least two non-legend components colored by
#      line contact and at least two non-legend components colored by adjacency.
# i) New main() assumptions checked against the generator:
#    - Assumes every propagated component has a unique candidate color in its update
#      round: generator isolates color corridors and chain groups.
#    - Assumes propagated components can be linked back to the same legend with a
#      deterministic corridor: generator keeps each group attached to one legend lane.
#    - Assumes legend direction can be inferred globally: generator keeps all legends
#      in one shared strip or corner panel, as in the base task.


BG = 0
DIV = 1
GREY = 5
GRID_SIZE = 30
LEGEND_COLORS = [2, 3, 4, 6, 7, 8, 9]

LEGEND_COUNT_RANGE = (2, 3)
MAX_HOLE_COUNT = 6
PANEL_SIZE_RANGE = (5, 8)
FULL_LAYOUT_MAX_SPAN = 28
L_LAYOUT_MAX_SPAN = 24
L_LAYOUT_EXTRA_PADDING = 3

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
FOLLOWER_MAX_H = 8
FOLLOWER_MAX_W = 8
FOLLOWER_MAX_ASPECT = 2.25

MIN_SEED_SPAN = 7
MIN_INLINE_SPAN = 3
MIN_COMPONENT_GAP = 2
ALIGN_RETRIES = 120
MAX_DISTRACTORS = 2

CHAIN_GROUPS = 2
GEN_RETRIES = 500

_STRUCT8 = generate_binary_structure(2, 2)
_DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
        infos.append(
            {
                "color": int(color),
                "hole_count": _hole_count(sub),
                "bbox": bbox,
            }
        )

    assert infos, "legend mapping must be non-empty: grid must contain colored legend shapes"
    mapping: dict[int, dict] = {}

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


def _component_infos(g: np.ndarray, grey: int) -> list[dict]:
    infos: list[dict] = []
    for pts in components_mask(g == grey, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        coords = []
        coord_set = set()
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            sub[rr - r0, cc - c0] = True
            coords.append((rr, cc))
            coord_set.add((rr, cc))
        infos.append(
            {
                "pts": coords,
                "set": coord_set,
                "hole_count": _hole_count(sub),
            }
        )
    return infos


def _touches8(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in b:
                    return True
    return False


def _adjacency_graph(infos: list[dict]) -> list[set[int]]:
    n = len(infos)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _touches8(infos[i]["set"], infos[j]["set"]):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _side_cells(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    occ = {tuple(map(int, p)) for p in np.argwhere(mask)}
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


def _select_sprite_port(pts: list[tuple[int, int]] | np.ndarray, direction: str, target: tuple[int, int]) -> tuple[int, int]:
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
    assert side, "colored sprite must expose a connector side"

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


def _corridor_cells(a: tuple[int, int], b: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(r, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, c) for r in range(r0, r1 + 1)]

    elbow = (b[0], a[1]) if direction in ("down", "up") else (a[0], b[1])
    cells1 = _corridor_cells(a, elbow, direction)
    cells2 = _corridor_cells(elbow, b, direction)
    return cells1 + cells2[1:]


def _draw_connector(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], direction: str, color: int) -> None:
    for r, c in _corridor_cells(a, b, direction):
        out[r, c] = int(color)


def _corridor_hits_component(corridor: list[tuple[int, int]], component: set[tuple[int, int]]) -> bool:
    for cell in corridor:
        if cell in component:
            return True
    return False


def _solve_state(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_by_hole = _extract_legends(g, bg, GREY)
    legend_by_color = {int(info["color"]): info for info in legend_by_hole.values()}
    infos = _component_infos(g, GREY)
    adj = _adjacency_graph(infos)

    colors: list[int | None] = []
    reasons: list[str | None] = []
    for info in infos:
        hc = int(info["hole_count"])
        if hc in legend_by_hole:
            colors.append(int(legend_by_hole[hc]["color"]))
            reasons.append("seed")
        else:
            colors.append(None)
            reasons.append(None)

    while True:
        snapshot = list(colors)
        line_data: list[tuple[int, int, list[tuple[int, int]]]] = []
        for i, color in enumerate(snapshot):
            if color is None:
                continue
            legend = legend_by_color[int(color)]
            sprite_port = _select_sprite_port(infos[i]["pts"], legend["direction"], legend["port"])
            corridor = _corridor_cells(legend["port"], sprite_port, legend["direction"])
            line_data.append((i, int(color), corridor))

        updates: list[tuple[int, int, str]] = []
        for i, current in enumerate(snapshot):
            if current is not None:
                continue
            candidates: dict[int, set[str]] = {}
            for j in adj[i]:
                if snapshot[j] is not None:
                    candidates.setdefault(int(snapshot[j]), set()).add("adj")
            for _, color, corridor in line_data:
                if _corridor_hits_component(corridor, infos[i]["set"]):
                    candidates.setdefault(int(color), set()).add("line")
            if len(candidates) == 1:
                color, source_set = next(iter(candidates.items()))
                source = "+".join(sorted(source_set))
                updates.append((i, int(color), source))

        if not updates:
            break
        for i, color, source in updates:
            colors[i] = int(color)
            reasons[i] = source

    final_links = []
    for info, color in zip(infos, colors):
        if color is None:
            continue
        legend = legend_by_color[int(color)]
        sprite_port = _select_sprite_port(info["pts"], legend["direction"], legend["port"])
        final_links.append(
            {
                "color": int(color),
                "direction": legend["direction"],
                "legend_port": legend["port"],
                "sprite_port": sprite_port,
                "corridor": _corridor_cells(legend["port"], sprite_port, legend["direction"]),
            }
        )

    return {
        "grid": g,
        "bg": bg,
        "legend_by_hole": legend_by_hole,
        "infos": infos,
        "colors": colors,
        "reasons": reasons,
        "links": final_links,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor direct legend matches first, then let colors spread through nearby sprites and their connector lanes.

    Input structure:
    - The background is the dominant color, grey sprites (5) occupy the workspace, and a divider still isolates a
      compact legend panel near one edge or corner.
    - Each legend patch is an interior colored component whose hole count defines one output color.
    - Some grey sprites directly match legend hole counts, while other grey sprites with non-legend hole counts are
      positioned so they either touch a colored sprite chain or sit on the connector lane between a sprite and a legend.

    Rule:
    - Read the legend exactly as before to obtain a hole-count -> color mapping, and use that mapping to color only
      the directly matching grey sprites in the first round.
    - Then iterate: any still-grey sprite inherits a color if the uniquely visible colored influence reaching it is
      either an 8-neighbor colored sprite or a same-color connector corridor from an already colored sprite to its legend.
    - After propagation stabilizes, erase any remaining grey sprites and draw a same-color connector from every colored
      sprite, including propagated followers, back to the legend that supplies that color.

    Key insight: Hole counts only create the seeds; the full answer is a fused process where connector lines and
    sprite-to-sprite adjacency cooperate to spread a legend color through a visible chain.
    """
    state = _solve_state(grid)
    out = state["grid"].copy()
    bg = int(state["bg"])

    for info, color in zip(state["infos"], state["colors"]):
        newc = bg if color is None else int(color)
        for r, c in info["pts"]:
            out[r, c] = newc

    for link in state["links"]:
        _draw_connector(out, link["legend_port"], link["sprite_port"], link["direction"], link["color"])
    return out


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
    kind = random.choice(["horizontal", "vertical"])
    if kind == "horizontal":
        return _layout_full_horizontal(grid, blocked)
    return _layout_full_vertical(grid, blocked)


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


def _crop_and_validate(
    mask: np.ndarray,
    forbidden: set[tuple[int, int]],
    hole_count: int,
    max_h: int,
    max_w: int,
    max_aspect: float,
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
    if h > max_h or w > max_w or max(h, w) > max_aspect * min(h, w):
        return None
    cropped = mask[rmin : rmax + 1, cmin : cmax + 1].copy()
    if _hole_count(cropped) != hole_count:
        return None
    return np.where(cropped, GREY, BG).astype(int)


def _make_sprite(
    hole_count: int,
    *,
    max_h: int = SPRITE_MAX_H,
    max_w: int = SPRITE_MAX_W,
    max_aspect: float = SPRITE_MAX_ASPECT,
) -> np.ndarray | None:
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
        result = _crop_and_validate(mask, forbidden, hole_count, max_h, max_w, max_aspect)
        if result is not None:
            return result
    return None


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=_STRUCT8).astype(bool)


def _sprite_coords(sprite: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pt)) for pt in np.argwhere(sprite == GREY)]


def _placement_ok(static_blocked: np.ndarray, forbidden_pad: np.ndarray, sprite_mask: np.ndarray, r: int, c: int) -> bool:
    h, w = sprite_mask.shape
    rr0, cc0 = max(0, r - 1), max(0, c - 1)
    rr1, cc1 = min(GRID_SIZE, r + h + 1), min(GRID_SIZE, c + w + 1)
    if np.any(static_blocked[rr0:rr1, cc0:cc1] != BG):
        return False
    return not np.any(forbidden_pad[r : r + h, c : c + w] & sprite_mask)


def _corridor_clear_against_global(global_occ: np.ndarray, global_line_blocked: np.ndarray, corridor: list[tuple[int, int]]) -> bool:
    for r, c in corridor:
        if global_occ[r, c] != BG or global_line_blocked[r, c]:
            return False
    return True


def _aligned_seed_location(
    static_blocked: np.ndarray,
    global_occ: np.ndarray,
    global_line_blocked: np.ndarray,
    sprite: np.ndarray,
    legend_info: dict,
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]] | None:
    sprite_mask = sprite == GREY
    h, w = sprite.shape
    port = legend_info["port"]
    direction = legend_info["direction"]
    side_cells = _side_cells(sprite_mask, direction)
    if not side_cells:
        return None

    random.shuffle(side_cells)
    forbidden_pad = _dilate_mask((global_occ != BG) | global_line_blocked)
    for ar, ac in side_cells:
        if direction in ("down", "up"):
            c = port[1] - ac
            if c < 0 or c + w > GRID_SIZE:
                continue
            if direction == "down":
                r_min = max(0, port[0] + MIN_SEED_SPAN - ar)
                r_max = GRID_SIZE - h
            else:
                r_min = 0
                r_max = min(GRID_SIZE - h, port[0] - MIN_SEED_SPAN - ar)
            if r_min > r_max:
                continue
            rows = list(range(r_min, r_max + 1))
            random.shuffle(rows)
            for r in rows[:ALIGN_RETRIES]:
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = [(r + rr, c + cc) for rr, cc in _sprite_coords(sprite)]
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[1] != port[1]:
                    continue
                corridor = _corridor_cells(port, sprite_port, direction)
                if _corridor_clear_against_global(global_occ, global_line_blocked, corridor[:-1]):
                    return int(r), int(c), pts, sprite_port, corridor
        else:
            r = port[0] - ar
            if r < 0 or r + h > GRID_SIZE:
                continue
            if direction == "right":
                c_min = max(0, port[1] + MIN_SEED_SPAN - ac)
                c_max = GRID_SIZE - w
            else:
                c_min = 0
                c_max = min(GRID_SIZE - w, port[1] - MIN_SEED_SPAN - ac)
            if c_min > c_max:
                continue
            cols = list(range(c_min, c_max + 1))
            random.shuffle(cols)
            for c in cols[:ALIGN_RETRIES]:
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = [(r + rr, c + cc) for rr, cc in _sprite_coords(sprite)]
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[0] != port[0]:
                    continue
                corridor = _corridor_cells(port, sprite_port, direction)
                if _corridor_clear_against_global(global_occ, global_line_blocked, corridor[:-1]):
                    return int(r), int(c), pts, sprite_port, corridor
    return None


def _inline_follower_location(
    static_blocked: np.ndarray,
    occupied: np.ndarray,
    global_occ: np.ndarray,
    global_line_blocked: np.ndarray,
    sprite: np.ndarray,
    legend_info: dict,
    seed_port: tuple[int, int],
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]] | None:
    sprite_mask = sprite == GREY
    h, w = sprite.shape
    port = legend_info["port"]
    direction = legend_info["direction"]
    side_cells = _side_cells(sprite_mask, direction)
    if not side_cells:
        return None

    random.shuffle(side_cells)
    forbidden_pad = _dilate_mask((occupied != BG) | global_line_blocked)
    for ar, ac in side_cells:
        if direction in ("down", "up"):
            c = port[1] - ac
            if c < 0 or c + w > GRID_SIZE:
                continue
            if direction == "down":
                r_min = max(0, port[0] + MIN_INLINE_SPAN - ar)
                r_max = min(GRID_SIZE - h, seed_port[0] - MIN_COMPONENT_GAP - ar)
            else:
                r_min = max(0, seed_port[0] + MIN_COMPONENT_GAP - ar)
                r_max = min(GRID_SIZE - h, port[0] - MIN_INLINE_SPAN - ar)
            if r_min > r_max:
                continue
            rows = list(range(r_min, r_max + 1))
            random.shuffle(rows)
            for r in rows[:ALIGN_RETRIES]:
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = [(r + rr, c + cc) for rr, cc in _sprite_coords(sprite)]
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[1] != port[1]:
                    continue
                corridor = _corridor_cells(port, sprite_port, direction)
                if _corridor_clear_against_global(global_occ, global_line_blocked, corridor[:-1]):
                    return int(r), int(c), pts, sprite_port, corridor
        else:
            r = port[0] - ar
            if r < 0 or r + h > GRID_SIZE:
                continue
            if direction == "right":
                c_min = max(0, port[1] + MIN_INLINE_SPAN - ac)
                c_max = min(GRID_SIZE - w, seed_port[1] - MIN_COMPONENT_GAP - ac)
            else:
                c_min = max(0, seed_port[1] + MIN_COMPONENT_GAP - ac)
                c_max = min(GRID_SIZE - w, port[1] - MIN_INLINE_SPAN - ac)
            if c_min > c_max:
                continue
            cols = list(range(c_min, c_max + 1))
            random.shuffle(cols)
            for c in cols[:ALIGN_RETRIES]:
                if not _placement_ok(static_blocked, forbidden_pad, sprite_mask, r, c):
                    continue
                pts = [(r + rr, c + cc) for rr, cc in _sprite_coords(sprite)]
                sprite_port = _select_sprite_port(pts, direction, port)
                if sprite_port[0] != port[0]:
                    continue
                corridor = _corridor_cells(port, sprite_port, direction)
                if _corridor_clear_against_global(global_occ, global_line_blocked, corridor[:-1]):
                    return int(r), int(c), pts, sprite_port, corridor
    return None


def _diagonal_attach_location(
    static_blocked: np.ndarray,
    global_occ: np.ndarray,
    global_line_blocked: np.ndarray,
    sprite: np.ndarray,
    parent_coords: set[tuple[int, int]],
    other_coords: set[tuple[int, int]],
    avoid_corridor: set[tuple[int, int]],
    legend_info: dict,
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]] | None:
    sprite_pts = _sprite_coords(sprite)
    offsets = set()
    for pr, pc in parent_coords:
        for sr, sc in sprite_pts:
            for dr, dc in _DIAGS:
                offsets.add((pr + dr - sr, pc + dc - sc))
    offsets = list(offsets)
    random.shuffle(offsets)

    all_occ = parent_coords | other_coords
    for off_r, off_c in offsets:
        placed = [(off_r + sr, off_c + sc) for sr, sc in sprite_pts]
        placed_set = set(placed)
        if any(r < 0 or c < 0 or r >= GRID_SIZE or c >= GRID_SIZE for r, c in placed_set):
            continue
        if placed_set & all_occ:
            continue
        if placed_set & avoid_corridor:
            continue
        invalid = False
        touches_parent = False
        for r, c in placed_set:
            for rr in range(max(0, r - 1), min(GRID_SIZE, r + 2)):
                for cc in range(max(0, c - 1), min(GRID_SIZE, c + 2)):
                    if static_blocked[rr, cc] != BG or global_line_blocked[rr, cc]:
                        invalid = True
                        break
                if invalid:
                    break
            if invalid:
                break
            for rr, cc in neighbors4(r, c):
                if (rr, cc) in all_occ:
                    invalid = True
                    break
            if invalid:
                break
            for dr, dc in _DIAGS:
                nb = (r + dr, c + dc)
                if nb in parent_coords:
                    touches_parent = True
                elif nb in other_coords:
                    invalid = True
                    break
            if invalid:
                break
        if invalid or not touches_parent:
            continue
        sprite_port = _select_sprite_port(placed, legend_info["direction"], legend_info["port"])
        corridor = _corridor_cells(legend_info["port"], sprite_port, legend_info["direction"])
        if not _corridor_clear_against_global(global_occ, global_line_blocked, corridor):
            continue
        return int(off_r), int(off_c), placed, sprite_port, corridor
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


def _group_specs(legends: list[dict]) -> list[dict]:
    legend_holes = {info["hole_count"] for info in legends}
    nonlegend_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    small_nonlegend = [h for h in nonlegend_holes if h <= 3] or nonlegend_holes
    chain_count = min(CHAIN_GROUPS, len(legends))
    chain_indices = set(random.sample(range(len(legends)), chain_count))
    specs = []
    for i, info in enumerate(legends):
        if i in chain_indices:
            specs.append(
                {
                    "info": info,
                    "chain": True,
                    "inline_hole": int(random.choice(small_nonlegend)),
                    "side_hole": int(random.choice(small_nonlegend)),
                }
            )
        else:
            specs.append({"info": info, "chain": False})
    specs.sort(key=lambda spec: 0 if spec["chain"] else 1)
    return specs


def _register_corridors(line_blocked: np.ndarray, corridors: list[list[tuple[int, int]]]) -> None:
    for corridor in corridors:
        for r, c in corridor:
            line_blocked[r, c] = True


def _place_groups(grid: np.ndarray, blocked: np.ndarray, legends: list[dict]) -> bool:
    occupied = np.zeros_like(grid)
    line_blocked = np.zeros_like(grid, dtype=bool)

    for spec in _group_specs(legends):
        info = spec["info"]
        global_occ_before = occupied.copy()
        global_line_before = line_blocked.copy()

        seed_sprite = _make_sprite(int(info["hole_count"]))
        if seed_sprite is None:
            return False
        seed_place = _aligned_seed_location(blocked, global_occ_before, global_line_before, seed_sprite, info)
        if seed_place is None:
            return False
        seed_r, seed_c, seed_pts, seed_port, seed_corridor = seed_place
        paste(grid, seed_sprite, seed_r, seed_c, transparent=BG)
        paste(occupied, seed_sprite, seed_r, seed_c, transparent=BG)
        group_components = [seed_pts]
        group_corridors = [seed_corridor]

        if spec["chain"]:
            inline_sprite = _make_sprite(
                int(spec["inline_hole"]),
                max_h=FOLLOWER_MAX_H,
                max_w=FOLLOWER_MAX_W,
                max_aspect=FOLLOWER_MAX_ASPECT,
            )
            if inline_sprite is None:
                return False
            inline_place = _inline_follower_location(
                blocked,
                occupied,
                global_occ_before,
                global_line_before,
                inline_sprite,
                info,
                seed_port,
            )
            if inline_place is None:
                return False
            inline_r, inline_c, inline_pts, _, inline_corridor = inline_place
            paste(grid, inline_sprite, inline_r, inline_c, transparent=BG)
            paste(occupied, inline_sprite, inline_r, inline_c, transparent=BG)
            group_components.append(inline_pts)
            group_corridors.append(inline_corridor)

            side_sprite = _make_sprite(
                int(spec["side_hole"]),
                max_h=FOLLOWER_MAX_H,
                max_w=FOLLOWER_MAX_W,
                max_aspect=FOLLOWER_MAX_ASPECT,
            )
            if side_sprite is None:
                return False
            parent_coords = set(inline_pts)
            other_coords = set(seed_pts)
            side_place = _diagonal_attach_location(
                blocked,
                global_occ_before,
                global_line_before,
                side_sprite,
                parent_coords,
                other_coords,
                set(seed_corridor),
                info,
            )
            if side_place is None:
                return False
            side_r, side_c, side_pts, _, side_corridor = side_place
            paste(grid, side_sprite, side_r, side_c, transparent=BG)
            paste(occupied, side_sprite, side_r, side_c, transparent=BG)
            group_components.append(side_pts)
            group_corridors.append(side_corridor)

        for comp_pts, corridor in zip(group_components, group_corridors):
            sprite_port = _select_sprite_port(comp_pts, info["direction"], info["port"])
            expected_corridor = _corridor_cells(info["port"], sprite_port, info["direction"])
            if expected_corridor != corridor:
                return False
            if not _corridor_clear_against_global(global_occ_before, global_line_before, corridor):
                return False

        _register_corridors(line_blocked, group_corridors)

    legend_holes = {info["hole_count"] for info in legends}
    distractor_holes = [h for h in range(MAX_HOLE_COUNT) if h not in legend_holes]
    n_distractors = random.randint(0, MAX_DISTRACTORS)
    for _ in range(n_distractors):
        hole_count = int(random.choice(distractor_holes))
        sprite = _make_sprite(
            hole_count,
            max_h=FOLLOWER_MAX_H,
            max_w=FOLLOWER_MAX_W,
            max_aspect=FOLLOWER_MAX_ASPECT,
        )
        if sprite is None:
            return False
        loc = _random_sprite_location(blocked, occupied, line_blocked, sprite)
        if loc is None:
            return False
        paste(grid, sprite, loc[0], loc[1], transparent=BG)
        paste(occupied, sprite, loc[0], loc[1], transparent=BG)

    return True


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    state = _solve_state(grid)
    legend_holes = set(state["legend_by_hole"])
    line_promoted = 0
    adjacency_promoted = 0
    propagated = 0
    for info, color, reason in zip(state["infos"], state["colors"], state["reasons"]):
        if color is None or reason is None or reason == "seed":
            continue
        if int(info["hole_count"]) in legend_holes:
            continue
        propagated += 1
        if "line" in reason:
            line_promoted += 1
        if "adj" in reason:
            adjacency_promoted += 1
    return {
        "propagated": propagated,
        "line_promoted": line_promoted,
        "adjacency_promoted": adjacency_promoted,
    }


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    blocked = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    legends = _add_layout(grid, blocked)
    if not _place_groups(grid, blocked, legends):
        return None
    if int(np.sum(grid == BG)) <= int(np.sum(grid == GREY)):
        return None
    metrics = _behavior_metrics(grid)
    if metrics["line_promoted"] < 2 or metrics["adjacency_promoted"] < 2:
        return None
    _ = main(grid)
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
