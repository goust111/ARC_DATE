# Combo variant of d35bdbdc (index 26)
# combo_spec: D7/0+D12/2
# Name: 精灵嵌套在大框架内+分支路径

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, paste


# Design analysis for the combo variant
# =====================================
# concretization and behavior trace:
#   - Keep the original color-transfer rule: a destination sprite uses its center
#     color as a lookup key into another sprite's border color, then copies that
#     sprite's center color.
#   - Replace the old 2-endpoint grey trail by a true 4-neighbor tree with
#     exactly 3 leaves. In the base layout those leaves sit at three sprite ports:
#       top-left sprite -> right-side leaf
#       bottom-left sprite -> right-side leaf
#       top-right sprite -> downward leaf
#   - Nest every sprite inside one large frame structure: an outer rectangular
#     frame plus one connected inner frame segment. The grey tree runs inside that
#     framed region and visually branches near the inner frame segment.
#   - Concrete example in the base orientation:
#       frame = outer border + inner L-like frame line on the right
#       grey tree leaves = (2,4), (7,4), (4,7)
#       destinations = centers (2,2), (7,2), (2,7)
#       source = center (7,7)
#     Input shows three destination centers all equal to the source border color.
#     Output keeps the frame and the whole grey tree, deletes the source sprite,
#     and changes all three destination centers to the source center color.
#
# complexity calibration:
#   - Fixing 3 destinations makes the branching rule unmistakable: with only
#     2 leaves the variant would collapse back toward the original task.
#   - Fixing 4 total sprites avoids overcrowding inside a 10x10 framed layout
#     while still preserving a non-destination sprite that gets deleted.
#   - The repeated 3-to-1 lookup gives strong support for the color rule: humans
#     can see the same source color propagated into multiple destinations.
#
# implicit-assumption audit of the original solver:
#   - Original code assumed grey structure had exactly 2 relevant endpoints.
#     A branched tree breaks _find_trail_endpoints and every downstream 2-item
#     matching step, so destination discovery must be rewritten around tree leaves.
#   - Original sprite detection relied on connected components of non-bg/non-grey
#     pixels. In the framed layout sprites may sit tightly against the frame, so
#     component-based detection can merge a sprite with the frame structure.
#     This variant therefore detects sprites from their local 3x3 box/plus pattern
#     around the unique center cell instead of from global components.
#   - Original cleanup deleted every non-destination non-grey component. The large
#     frame is a deliberate persistent object now, so the solver must detect and
#     preserve it explicitly.
#
# placement legality:
#   - Every sprite is fully in-bounds and placed in a dedicated 3x3 slot.
#   - The frame, grey tree, and sprite slots are disjoint by construction.
#   - Symmetry transforms are applied to whole finished grids, so legality is
#     preserved automatically across all layout variants.
#
# visual observability and cleanliness:
#   - The outer frame is large and persistent, so the nesting change is obvious.
#   - The grey tree is a clean T-shape with exactly three leaves, making the
#     branching change easy to spot.
#   - The source sprite sits in the remaining framed compartment, so its removal
#     in the output is visible without clutter.
#
# inferability of the change:
#   - Border colors stay distinct, and the frame color is kept distinct from all
#     sprite colors.
#   - All three destination centers share the same lookup key, so the repeated
#     transfer makes the rule easier to infer from a few examples.
#
# post-change solver assumptions that the generator guarantees:
#   - There is exactly one large monochrome frame component, and it is the largest
#     non-bg/non-grey monochrome component in the grid.
#   - Every sprite is a full 3x3 box or plus around a unique center cell.
#   - The grey pixels form one 4-neighbor tree with exactly three leaves.
#   - Each grey leaf is a valid port for exactly one destination sprite.


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNT = 4
DEST_COUNT = 3
GEN_RETRIES = 20
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BASE_DEST_CENTERS = [(2, 2), (7, 2), (2, 7)]
BASE_SOURCE_CENTER = (7, 7)
BASE_TREE_CELLS = {
    (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4),
    (4, 5), (4, 6), (4, 7),
}


def _bg_boundary(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _largest_frame_component(g: np.ndarray, bg: int, grey: int) -> dict:
    best = None
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, grey)):
        for pts in components_mask(g == color, connectivity=4):
            area = len(pts)
            if best is None or area > best["area"]:
                rows = [int(r) for r, _c in pts]
                cols = [int(c) for _r, c in pts]
                best = {
                    "color": int(color),
                    "pts": {(int(r), int(c)) for r, c in pts},
                    "area": area,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
    assert best is not None and best["area"] >= 20, "expected a large frame component"
    return best


def _detect_local_sprites(g: np.ndarray, bg: int, grey: int, frame_color: int) -> list[dict]:
    sprites: list[dict] = []
    seen_centers: set[tuple[int, int]] = set()
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            center = int(g[r, c])
            if center in (bg, grey, frame_color):
                continue
            four = [int(g[r - 1, c]), int(g[r + 1, c]), int(g[r, c - 1]), int(g[r, c + 1])]
            if len(set(four)) != 1:
                continue
            border = four[0]
            if border in (bg, grey, frame_color, center):
                continue
            diags = [int(g[r - 1, c - 1]), int(g[r - 1, c + 1]), int(g[r + 1, c - 1]), int(g[r + 1, c + 1])]
            if all(d == border for d in diags):
                shape = "box"
                pts = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}
            elif all(d != border for d in diags):
                shape = "plus"
                pts = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            else:
                continue
            if any(int(g[rr, cc]) in (bg, grey, frame_color) for rr, cc in pts):
                continue
            if (r, c) in seen_centers:
                continue
            seen_centers.add((r, c))
            sprites.append(
                {
                    "pts": pts,
                    "border": int(border),
                    "center": int(center),
                    "center_pos": (r, c),
                    "shape": shape,
                }
            )
    assert len(sprites) == SPRITE_COUNT, f"expected {SPRITE_COUNT} framed sprites"
    return sprites


def _sprite_port_cells(sprite: dict, g: np.ndarray) -> list[tuple[int, int]]:
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= er < g.shape[0] and 0 <= ec < g.shape[1]):
            continue
        if int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def _grey_tree_stats(cells: set[tuple[int, int]]) -> dict:
    adj = {cell: [nb for nb in neighbors4(*cell) if nb in cells] for cell in cells}
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    edges = sum(len(nbs) for nbs in adj.values()) // 2
    leaves = sorted(cell for cell, nbs in adj.items() if len(nbs) == 1)
    branch_nodes = sorted(cell for cell, nbs in adj.items() if len(nbs) >= 3)
    return {
        "connected": len(seen) == len(cells),
        "is_tree": len(seen) == len(cells) and edges == len(cells) - 1,
        "leaves": leaves,
        "branch_nodes": branch_nodes,
    }


def _find_tree_destinations(
    g: np.ndarray, grey: int, sprites: list[dict], frame_bbox: tuple[int, int, int, int],
) -> tuple[list[dict], set[tuple[int, int]]]:
    comps = components_mask(g == grey, connectivity=4)
    assert len(comps) == 1, "expected one grey tree component"
    grey_cells = {(int(r), int(c)) for r, c in comps[0]}
    stats = _grey_tree_stats(grey_cells)
    assert stats["is_tree"], "grey structure must be a tree"
    leaves = stats["leaves"]
    assert len(leaves) == DEST_COUNT, "expected three grey leaves"
    assert stats["branch_nodes"], "expected a branching grey node"

    r0, c0, r1, c1 = frame_bbox
    for sp in sprites:
        cr, cc = sp["center_pos"]
        assert r0 < cr < r1 and c0 < cc < c1, "sprites must be nested inside the frame"

    sprite_ports = {id(sp): set(_sprite_port_cells(sp, g)) for sp in sprites}
    dests: list[dict] = []
    used: set[int] = set()
    for leaf in leaves:
        cands = [sp for sp in sprites if leaf in sprite_ports[id(sp)]]
        assert len(cands) == 1, "each grey leaf should match exactly one destination sprite"
        sp = cands[0]
        sid = id(sp)
        assert sid not in used, "destination sprites should not receive multiple leaves"
        used.add(sid)
        dests.append(sp)
    return dests, grey_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors into framed destination sprites reached by a branched grey tree.

    Input structure:
    - A large monochrome frame encloses all bordered sprites, and a grey (5) path tree sits inside that framed layout.
    - Each sprite is a full 3x3 box or plus with one border color and one unique center pixel.
    - The grey pixels form a 4-neighbor branching tree whose three leaf cells are sprite ports.

    Rule:
    - Preserve the frame and the full grey tree.
    - Every sprite touched by a grey leaf port is a destination; keep exactly those destination sprites.
    - For each destination, use its original center color as a lookup key, find the source sprite whose border color matches that key, and copy the source center color into the destination center.
    - Delete the non-destination sprites inside the frame.

    Key insight:
    - The task now combines hierarchy and path tracing: first isolate the large frame from the small sprites, then follow all branches of the grey tree to collect every leaf-connected destination before applying the same border-to-center color transfer rule.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG
    grey = GREY

    frame = _largest_frame_component(g, bg, grey)
    sprites = _detect_local_sprites(g, bg, grey, int(frame["color"]))
    dests, grey_cells = _find_tree_destinations(g, grey, sprites, frame["bbox"])
    dest_ids = {id(sp) for sp in dests}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = np.full(g.shape, bg, dtype=int)
    for r, c in frame["pts"]:
        out[r, c] = int(frame["color"])
    for r, c in grey_cells:
        out[r, c] = grey
    for sp in dests:
        for r, c in sp["pts"]:
            out[r, c] = int(g[r, c])

    for d in dests:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        assert len(srcs) == 1, "each destination key should match exactly one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in DIRS4:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    paste(grid, patch, center[0] - 1, center[1] - 1, transparent=BG, clip=False)


def _base_frame_cells(frame_color: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(GRID_W):
        cells.add((0, c))
        cells.add((GRID_H - 1, c))
    for r in range(GRID_H):
        cells.add((r, 0))
        cells.add((r, GRID_W - 1))
    for r in range(0, 9):
        cells.add((r, 5))
    for c in range(5, GRID_W):
        cells.add((5, c))
    return cells


def _draw_base_structure(frame_color: int) -> np.ndarray:
    g = np.zeros((GRID_H, GRID_W), dtype=int)
    for r, c in _base_frame_cells(frame_color):
        g[r, c] = frame_color
    for r, c in BASE_TREE_CELLS:
        g[r, c] = GREY
    return g


def _transform_grid(g: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return g.copy()
    if mode == "flip_h":
        return np.fliplr(g)
    if mode == "flip_v":
        return np.flipud(g)
    if mode == "rot180":
        return np.flipud(np.fliplr(g))
    if mode == "transpose":
        return np.transpose(g)
    if mode == "transpose_h":
        return np.fliplr(np.transpose(g))
    if mode == "transpose_v":
        return np.flipud(np.transpose(g))
    if mode == "anti_transpose":
        return np.flipud(np.fliplr(np.transpose(g)))
    raise ValueError(mode)


def _build_base_pair() -> tuple[np.ndarray, np.ndarray]:
    shape_type = random.choice(["box", "plus"])
    frame_color = random.choice([c for c in range(1, 10) if c != GREY])

    palette = [c for c in range(1, 10) if c not in (GREY, frame_color)]
    border_colors = random.sample(palette, SPRITE_COUNT)
    dest_borders = border_colors[:DEST_COUNT]
    source_border = border_colors[DEST_COUNT]

    source_center_choices = [c for c in palette if c not in border_colors]
    source_center = random.choice(source_center_choices)

    output_grid = _draw_base_structure(frame_color)
    input_grid = output_grid.copy()

    dest_center_color = source_border
    output_dest_center = source_center

    for center, border_color in zip(BASE_DEST_CENTERS, dest_borders, strict=True):
        out_patch = _make_sprite_patch(shape_type, border_color, output_dest_center)
        in_patch = _make_sprite_patch(shape_type, border_color, dest_center_color)
        _place_sprite(output_grid, out_patch, center)
        _place_sprite(input_grid, in_patch, center)

    source_patch = _make_sprite_patch(shape_type, source_border, source_center)
    _place_sprite(input_grid, source_patch, BASE_SOURCE_CENTER)

    return input_grid, output_grid


def generate_input() -> np.ndarray:
    modes = [
        "identity",
        "flip_h",
        "flip_v",
        "rot180",
        "transpose",
        "transpose_h",
        "transpose_v",
        "anti_transpose",
    ]
    for _ in range(GEN_RETRIES):
        base_input, base_output = _build_base_pair()
        mode = random.choice(modes)
        transformed_input = _transform_grid(base_input, mode)
        transformed_output = _transform_grid(base_output, mode)
        out = main(transformed_input)
        if np.array_equal(out, transformed_output):
            return transformed_input
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
