# Combo variant of 4a21e3da (index 40)
# combo_spec: D10/2+D13/0
# Name: 颜色编码目标位置+连通归属

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the fusion:
# - Original 4a21e3da keeps whole geometric regions that touch a control edge, then
#   sends each kept region to its spatially corresponding corner.
# - This fusion changes two different inference steps:
#   1. D13/0: retention is no longer "which rectangle/quadrant touches the control edge".
#      We instead split the foreground into off-line 4-connected components and keep only
#      the components that touch a split-line foreground pixel. To make this non-degenerate,
#      the generator deliberately breaks the original single-component assumption: each
#      quadrant gets its own small component, and exactly one edge-touching quadrant is
#      isolated from the split line and disappears, while the opposite quadrant is attached
#      to the split line and survives. This guarantees visible disagreement with the
#      original region rule.
#   2. D10/2: kept components no longer go to their source corner. Each component carries
#      one colored key pixel, and the four grid corners show four different legend colors.
#      A kept component slides to the corner whose legend color matches its key color.
#      The output omits the legend itself so moved pieces do not overlap the corner markers.
# - Small concrete trace (top+left controls):
#   original: TL/TR/BL kept, BR removed, and each kept part goes to TL/TR/BL respectively.
#   fused: if BR component touches the split line it is kept; if TR stays one cell away from
#   both split lines it disappears. The remaining TL/BL/BR pieces then ignore source location
#   and move to the corners whose colors match their internal key pixels.
# - Complexity calibration:
#   exactly four corner legend colors and exactly four central components are used. This is
#   enough to make the color lookup visible, but still small enough that 2-3 examples can
#   reveal the rule. Each sample is forced to have at least one "kept but nonlocal target"
#   component and one "has a color key but still vanishes" component, so neither axis can
#   collapse into the other.
# - Hidden assumptions broken from the original solver:
#   * foreground is no longer monochrome
#   * foreground is no longer required to be one connected sprite
#   * control color is no longer "the only non-bg border color"; corners now hold legend colors
#   * retention no longer depends on rectangular region identity
#   The new solver therefore detects control pixels by border position, treats corners as a
#   separate legend layer, finds the base color by frequency, and uses component adjacency to
#   split-line pixels rather than quadrant membership.
# - Placement legality / cleanliness:
#   corner legend = four 1x1 markers at the exact corners; control pixels stay on non-corner
#   border cells; the four components live in a centered bbox with margin >= 4 and never
#   overlap each other or the split lines. Each kept component has a unique target corner, so
#   corner placements cannot collide. The central scene stays readable because legend, control,
#   split-line pixels, and movable components all occupy distinct roles.


CORNER_NAMES = ("TL", "TR", "BL", "BR")
QUADRANT_EDGES = {
    "TL": {"top", "left"},
    "TR": {"top", "right"},
    "BL": {"bottom", "left"},
    "BR": {"bottom", "right"},
}
PIECE_SHAPES = [
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
]


def _corner_coords(H: int, W: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (0, 0),
        "TR": (0, W - 1),
        "BL": (H - 1, 0),
        "BR": (H - 1, W - 1),
    }


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _split_info(ctrl_pts: list[tuple[int, int]], H: int, W: int) -> tuple[set[str], int | None, int | None]:
    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            c_split = c
        elif r == H - 1:
            edges.add("bottom")
            c_split = c
        elif c == 0:
            edges.add("left")
            r_split = r
        elif c == W - 1:
            edges.add("right")
            r_split = r
    return edges, c_split, r_split


def _scene_layers(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    corners = _corner_coords(H, W)
    legend = {name: int(g[r, c]) for name, (r, c) in corners.items()}
    assert all(color != bg for color in legend.values())
    assert len(set(legend.values())) == 4

    ctrl_pts: list[tuple[int, int]] = []
    for c in range(1, W - 1):
        if int(g[0, c]) != bg:
            ctrl_pts.append((0, c))
        if int(g[H - 1, c]) != bg:
            ctrl_pts.append((H - 1, c))
    for r in range(1, H - 1):
        if int(g[r, 0]) != bg:
            ctrl_pts.append((r, 0))
        if int(g[r, W - 1]) != bg:
            ctrl_pts.append((r, W - 1))
    assert len(ctrl_pts) in (1, 2)
    ctrl_color = int(g[ctrl_pts[0]])
    assert all(int(g[r, c]) == ctrl_color for r, c in ctrl_pts)

    sprite_mask = g != bg
    for r, c in corners.values():
        sprite_mask[r, c] = False
    for r, c in ctrl_pts:
        sprite_mask[r, c] = False

    edges, c_split, r_split = _split_info(ctrl_pts, H, W)
    split_mask = np.zeros((H, W), dtype=bool)
    if c_split is not None:
        split_mask[:, c_split] = True
    if r_split is not None:
        split_mask[r_split, :] = True
    split_mask &= sprite_mask

    sprite_colors = [int(v) for v in g[sprite_mask]]
    base_color = int(Counter(sprite_colors).most_common(1)[0][0])

    return {
        "grid": g,
        "H": H,
        "W": W,
        "bg": bg,
        "legend": legend,
        "ctrl_pts": ctrl_pts,
        "ctrl_color": ctrl_color,
        "edges": edges,
        "c_split": c_split,
        "r_split": r_split,
        "sprite_mask": sprite_mask,
        "split_mask": split_mask,
        "base_color": base_color,
    }


def _touches_split(comp: list[tuple[int, int]], split_pts: set[tuple[int, int]]) -> bool:
    for r, c in comp:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in split_pts:
                return True
    return False


def _quadrant_of(comp: list[tuple[int, int]], r_split: int, c_split: int) -> str:
    r = comp[0][0]
    c = comp[0][1]
    return ("T" if r < r_split else "B") + ("L" if c < c_split else "R")


def _component_tag_color(g: np.ndarray, comp: list[tuple[int, int]], base_color: int) -> int:
    others = {int(g[r, c]) for r, c in comp if int(g[r, c]) != base_color}
    assert len(others) == 1
    return int(next(iter(others)))


def _corner_shift(comp: list[tuple[int, int]], target: str, H: int, W: int) -> tuple[int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    if target == "TL":
        return -rmin, -cmin
    if target == "TR":
        return -rmin, (W - 1) - cmax
    if target == "BL":
        return (H - 1) - rmax, -cmin
    return (H - 1) - rmax, (W - 1) - cmax


def _draw_rays(out: np.ndarray, scene: dict) -> None:
    g = scene["grid"]
    bg = scene["bg"]
    sprite_mask = scene["sprite_mask"]
    ctrl_color = scene["ctrl_color"]
    H, W = scene["H"], scene["W"]

    for r, c in scene["ctrl_pts"]:
        if r == 0:
            rows = [rr for rr in range(H) if sprite_mask[rr, c]]
            if rows:
                for rr in range(0, max(rows) + 1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl_color
        elif r == H - 1:
            rows = [rr for rr in range(H) if sprite_mask[rr, c]]
            if rows:
                for rr in range(H - 1, min(rows) - 1, -1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl_color
        elif c == 0:
            cols = [cc for cc in range(W) if sprite_mask[r, cc]]
            if cols:
                for cc in range(0, max(cols) + 1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl_color
        elif c == W - 1:
            cols = [cc for cc in range(W) if sprite_mask[r, cc]]
            if cols:
                for cc in range(W - 1, min(cols) - 1, -1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl_color

    for r, c in scene["ctrl_pts"]:
        out[r, c] = ctrl_color


def main(grid: np.ndarray) -> np.ndarray:
    scene = _scene_layers(grid)
    g = scene["grid"]
    H, W = scene["H"], scene["W"]
    bg = scene["bg"]

    out = np.full((H, W), bg, dtype=int)
    _draw_rays(out, scene)

    split_pts = {(int(r), int(c)) for r, c in np.argwhere(scene["split_mask"])}
    for r, c in split_pts:
        out[r, c] = int(g[r, c])

    color_to_corner = {color: name for name, color in scene["legend"].items()}
    off_split_mask = scene["sprite_mask"] & ~scene["split_mask"]
    comps = _connected_components(off_split_mask)

    for comp in comps:
        if not _touches_split(comp, split_pts):
            continue
        tag_color = _component_tag_color(g, comp, scene["base_color"])
        target = color_to_corner[tag_color]
        dr, dc = _corner_shift(comp, target, H, W)
        for r, c in comp:
            out[r + dr, c + dc] = int(g[r, c])

    return out


def _piece_dims(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _choose_shape(max_h: int, max_w: int) -> frozenset[tuple[int, int]] | None:
    valid = [shape for shape in PIECE_SHAPES if _piece_dims(shape)[0] <= max_h and _piece_dims(shape)[1] <= max_w]
    if not valid:
        return None
    return random.choice(valid)


def _place_component(
    quadrant: str,
    rect: tuple[int, int, int, int],
    kept: bool,
    attach_mode: str | None,
) -> tuple[set[tuple[int, int]], tuple[int, int] | None]:
    r0, r1, c0, c1 = rect
    if kept:
        shape = _choose_shape(r1 - r0 + 1, c1 - c0 + 1)
        if shape is None or attach_mode is None:
            raise ValueError("kept component cannot be placed")
        h, w = _piece_dims(shape)
        if attach_mode == "vertical":
            rr0 = random.randint(r0, r1 - h + 1)
            cc0 = c1 - w + 1 if quadrant in ("TL", "BL") else c0
        else:
            rr0 = r1 - h + 1 if quadrant in ("TL", "TR") else r0
            cc0 = random.randint(c0, c1 - w + 1)
        pts = {(rr0 + r, cc0 + c) for r, c in shape}
        if attach_mode == "vertical":
            if quadrant in ("TL", "BL"):
                edge_col = max(c for _, c in pts)
            else:
                edge_col = min(c for _, c in pts)
            border_rows = sorted(r for r, c in pts if c == edge_col)
            split_pt = (border_rows[len(border_rows) // 2], c1 + 1 if quadrant in ("TL", "BL") else c0 - 1)
        else:
            if quadrant in ("TL", "TR"):
                edge_row = max(r for r, _ in pts)
            else:
                edge_row = min(r for r, _ in pts)
            border_cols = sorted(c for r, c in pts if r == edge_row)
            split_pt = (r1 + 1 if quadrant in ("TL", "TR") else r0 - 1, border_cols[len(border_cols) // 2])
        return pts, split_pt

    inner_r0 = r0 + (1 if quadrant in ("BL", "BR") else 0)
    inner_r1 = r1 - (1 if quadrant in ("TL", "TR") else 0)
    inner_c0 = c0 + (1 if quadrant in ("TR", "BR") else 0)
    inner_c1 = c1 - (1 if quadrant in ("TL", "BL") else 0)
    shape = _choose_shape(inner_r1 - inner_r0 + 1, inner_c1 - inner_c0 + 1)
    if shape is None:
        raise ValueError("vanishing component cannot be placed")
    h, w = _piece_dims(shape)
    rr0 = random.randint(inner_r0, inner_r1 - h + 1)
    cc0 = random.randint(inner_c0, inner_c1 - w + 1)
    pts = {(rr0 + r, cc0 + c) for r, c in shape}
    return pts, None


def _paint_component(
    grid: np.ndarray,
    pts: set[tuple[int, int]],
    base_color: int,
    tag_color: int,
    r_split: int,
    c_split: int,
) -> None:
    tag_pt = max(sorted(pts), key=lambda rc: abs(rc[0] - r_split) + abs(rc[1] - c_split))
    for r, c in pts:
        grid[r, c] = base_color
    tr, tc = tag_pt
    grid[tr, tc] = tag_color


def _opposite_quadrant(edges: set[str]) -> str:
    for quadrant, q_edges in QUADRANT_EDGES.items():
        if not (q_edges & edges):
            return quadrant
    raise ValueError("no opposite quadrant")


def _quadrant_rects(r_split: int, c_split: int, top_h: int, bottom_h: int, left_w: int, right_w: int) -> dict[str, tuple[int, int, int, int]]:
    return {
        "TL": (r_split - top_h, r_split - 1, c_split - left_w, c_split - 1),
        "TR": (r_split - top_h, r_split - 1, c_split + 1, c_split + right_w),
        "BL": (r_split + 1, r_split + bottom_h, c_split - left_w, c_split - 1),
        "BR": (r_split + 1, r_split + bottom_h, c_split + 1, c_split + right_w),
    }


def _analyze_generated(grid: np.ndarray) -> dict:
    scene = _scene_layers(grid)
    assert scene["r_split"] is not None and scene["c_split"] is not None
    split_pts = {(int(r), int(c)) for r, c in np.argwhere(scene["split_mask"])}
    comps = _connected_components(scene["sprite_mask"] & ~scene["split_mask"])
    color_to_corner = {color: name for name, color in scene["legend"].items()}
    info = []
    for comp in comps:
        quadrant = _quadrant_of(comp, scene["r_split"], scene["c_split"])
        tag_color = _component_tag_color(scene["grid"], comp, scene["base_color"])
        info.append(
            {
                "quadrant": quadrant,
                "kept": _touches_split(comp, split_pts),
                "target": color_to_corner[tag_color],
            }
        )
    return {"scene": scene, "components": info}


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, ctrl_color, base_color = 1, 2, 7

    vertical_edge = random.choice(["top", "bottom"])
    horizontal_edge = random.choice(["left", "right"])
    edges = {vertical_edge, horizontal_edge}
    r_split = random.randint(int(H * 0.4), int(H * 0.6))
    c_split = random.randint(int(W * 0.4), int(W * 0.6))

    top_h, bottom_h = random.choice([(3, 3), (3, 4), (4, 3)])
    left_w, right_w = random.choice([(3, 4), (4, 3), (4, 4), (3, 5), (5, 3)])
    if r_split - top_h < 4 or r_split + bottom_h > H - 5:
        return None
    if c_split - left_w < 4 or c_split + right_w > W - 5:
        return None

    ctrl_pts = [
        (0 if vertical_edge == "top" else H - 1, c_split),
        (r_split, 0 if horizontal_edge == "left" else W - 1),
    ]
    rects = _quadrant_rects(r_split, c_split, top_h, bottom_h, left_w, right_w)
    opposite = _opposite_quadrant(edges)
    edge_quadrants = [q for q in CORNER_NAMES if q != opposite]
    vanish_quadrant = random.choice(edge_quadrants)
    kept_quadrants = [q for q in CORNER_NAMES if q != vanish_quadrant]

    attach_modes: dict[str, str] = {}
    for _ in range(20):
        attach_modes = {q: random.choice(["vertical", "horizontal"]) for q in kept_quadrants}
        if "vertical" in attach_modes.values() and "horizontal" in attach_modes.values():
            break
    else:
        return None

    legend_colors = random.sample([3, 4, 5, 6], 4)
    corner_legend = dict(zip(CORNER_NAMES, legend_colors))
    targets = random.sample(CORNER_NAMES, 4)
    quadrant_to_target = dict(zip(CORNER_NAMES, targets))
    if all(quadrant_to_target[q] == q for q in kept_quadrants):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for name, (r, c) in _corner_coords(H, W).items():
        inp[r, c] = corner_legend[name]
    for r, c in ctrl_pts:
        inp[r, c] = ctrl_color

    split_pixels: set[tuple[int, int]] = set()
    vertical_rows: set[int] = set()
    horizontal_cols: set[int] = set()

    for quadrant in CORNER_NAMES:
        kept = quadrant in kept_quadrants
        try:
            pts, split_pt = _place_component(quadrant, rects[quadrant], kept, attach_modes.get(quadrant))
        except ValueError:
            return None
        tag_color = corner_legend[quadrant_to_target[quadrant]]
        _paint_component(inp, pts, base_color, tag_color, r_split, c_split)
        if kept:
            assert split_pt is not None
            split_pixels.add(split_pt)
            if split_pt[1] == c_split:
                vertical_rows.add(split_pt[0])
            if split_pt[0] == r_split:
                horizontal_cols.add(split_pt[1])

    vertical_candidates = [r for r in range(r_split - top_h, r_split + bottom_h + 1) if r != r_split]
    horizontal_candidates = [c for c in range(c_split - left_w, c_split + right_w + 1) if c != c_split]
    while len(vertical_rows) < 2:
        vertical_rows.add(random.choice(vertical_candidates))
    while len(horizontal_cols) < 2:
        horizontal_cols.add(random.choice(horizontal_candidates))
    while len(vertical_rows) > 4 or len(horizontal_cols) > 4:
        return None

    for r in vertical_rows:
        split_pixels.add((r, c_split))
    for c in horizontal_cols:
        split_pixels.add((r_split, c))
    if not (2 <= len(vertical_rows) <= 4 and 2 <= len(horizontal_cols) <= 4):
        return None

    for r, c in split_pixels:
        if inp[r, c] != bg:
            return None
        inp[r, c] = base_color

    analysis = _analyze_generated(inp)
    kept_info = [item for item in analysis["components"] if item["kept"]]
    vanished_info = [item for item in analysis["components"] if not item["kept"]]
    if len(kept_info) != 3 or len(vanished_info) != 1:
        return None
    if not any(item["quadrant"] == opposite and item["kept"] for item in analysis["components"]):
        return None
    if not any(item["quadrant"] == vanish_quadrant and not item["kept"] for item in analysis["components"]):
        return None
    if not any(item["quadrant"] != item["target"] for item in kept_info):
        return None

    try:
        _ = main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
