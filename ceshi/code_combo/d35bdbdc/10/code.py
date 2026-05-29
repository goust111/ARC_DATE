# Combo variant of d35bdbdc (index 10)
# combo_spec: D2/0+D7/0
# Name: 边框任意位置连接+分支路径

import random
from collections import deque
from itertools import permutations, product

import numpy as np

from arc_utils import components_mask, neighbors4, paste


# Design analysis for the combo variant
# =====================================
# concretization and behavior trace:
#   - Keep the original object vocabulary: several 3x3 bordered sprites, each with
#     one unique center pixel and a distinct border color.
#   - D7/0 changes the grey structure from a single 8-connected stroke with two
#     ends into one true 4-neighbor tree. The solver must therefore inspect every
#     leaf of that tree, not just "start/end".
#   - D2/0 changes what counts as a connection: a grey leaf now only needs to touch
#     any border pixel of a sprite by 4-neighbor adjacency. It no longer has to sit
#     on the old center-aligned 2-step port cell.
#   - Concrete example:
#       old: a destination box centered at (5,5) could only connect through
#            (3,5), (7,5), (5,3), or (5,7), because the endpoint had to align with
#            the center and see a border pixel in between.
#       new: a grey leaf may instead sit at (3,4) touching the sprite's upper-left
#            border corner at (4,4). That leaf is visibly attached to the border,
#            but it is not one of the old center-aligned ports.
#       combined with branching: three such leaves can hang off one tree, each
#            touching a different sprite border, so the solver must both trace the
#            tree and test border contact over the full sprite outline.
#
# complexity calibration:
#   - Destination count is fixed at 3. That is enough to force a visible branch and
#     repeated application of the color-transfer rule without overcrowding 10x10.
#   - The generator fixes the sprite shape to filled boxes. This is deliberate:
#     plus-shapes have only the four center-aligned arm pixels, which would make
#     "touch any border pixel" visually collapse back toward the old port geometry.
#     Box sprites provide corner-border contacts, making D2 observable.
#   - Border colors stay unique, while source count remains 1-2. This preserves the
#     original lookup logic and creates repeated evidence for the same mapping.
#
# implicit-assumption audit of the original main():
#   - _find_trail_endpoints assumed exactly two usable grey endpoints and a
#     one-stroke traversal. That breaks immediately on a tree with 3 leaves.
#   - _ports_for_sprite assumed center-aligned 2-step ports only. That misses the
#     new corner/side border contacts entirely.
#   - The fixed 2-element destination list and endpoint bipartite matching both
#     depended on there being exactly two destinations.
#   - The delete-non-dests step and the border->source color lookup do not depend
#     on linearity or on the old port geometry, so they can remain with only small
#     structural changes.
#
# placement legality:
#   - Destination sprites are placed first with minimum center spacing so their
#     border-adjacent touch cells do not ambiguously overlap.
#   - After placement, each destination enumerates every background cell that touches
#     one of its border pixels. Tree leaves are chosen only from touch cells that
#     belong to exactly one destination, and at least one chosen leaf must be
#     outside the old center-aligned port set.
#   - The tree is then routed through background cells while blocking all other
#     border-touch cells, so only the intended leaves visibly connect to sprites.
#   - Source and distractor sprites are added last and are rejected if any grey cell
#     would touch their borders, preventing accidental extra destinations.
#
# visual observability and cleanliness:
#   - A 4-neighbor tree makes the branch point and leaf count easy to see.
#   - Box sprites with at least one corner-touching leaf make the D2 change visible:
#     in the examples, at least one connection is clearly attached to a border
#     location that the original solver would not have accepted.
#   - Blocking non-leaf touch cells keeps the grey network from "hugging" sprite
#     edges, so the connection points remain crisp and readable.
#
# change inferability:
#   - The transferred color still comes from "sprite whose border color equals the
#     destination's input center color", so the new reasoning burden is purely about
#     which sprites count as destinations.
#   - Repeated leaves and repeated source reuse provide multiple aligned examples of
#     the same lookup rule, which helps humans infer the transformation from a small
#     number of training pairs.
#
# post-change solver assumptions that the generator guarantees:
#   - There is exactly one grey 4-connected component.
#   - That component is a tree with at least 3 leaves and at least one branch node.
#   - Every grey leaf is 4-neighbor adjacent to the border of exactly one sprite.
#   - At least one grey leaf is not an old center-aligned 2-step port for its
#     matched sprite, so the dataset does not degenerate back to the original task.


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
DEST_COUNT = 3
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [3, 1]  # weights for [0, 1] interior destinations
TREE_CELLS_MIN, TREE_CELLS_MAX = 10, 24
MIN_CENTER_DIST = 4
GEN_RETRIES = 600
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """Find 4-connected two-color components with one unique center cell."""
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color in (bg, grey) or center_color == grey:
            continue
        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": [(int(r), int(c)) for r, c in pts],
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _sprite_border_cells(sprite: dict) -> set[tuple[int, int]]:
    return {tuple(map(int, pt)) for pt in sprite["pts"] if tuple(map(int, pt)) != sprite["center_pos"]}


def _all_border_touch_cells(sprite: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    """All in-bounds cells adjacent to some border pixel, regardless of current color."""
    h, w = shape
    sprite_cells = {tuple(map(int, pt)) for pt in sprite["pts"]}
    touch: set[tuple[int, int]] = set()
    for br, bc in _sprite_border_cells(sprite):
        for nr, nc in neighbors4(br, bc):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in sprite_cells:
                touch.add((nr, nc))
    return touch


def _old_style_port_cells(sprite: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    """Center-aligned 2-step ports from the original task, used only for diagnostics."""
    h, w = shape
    cr, cc = sprite["center_pos"]
    border_cells = _sprite_border_cells(sprite)
    ports: set[tuple[int, int]] = set()
    for dr, dc in DIRS4:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= er < h and 0 <= ec < w):
            continue
        if (ir, ic) in border_cells:
            ports.add((er, ec))
    return ports


def _border_touch_cells(sprite: dict, g: np.ndarray) -> set[tuple[int, int]]:
    """Currently empty cells that could be used as border-touch connections."""
    return {cell for cell in _all_border_touch_cells(sprite, g.shape) if int(g[cell]) == BG}


def _grey_tree_stats(cells: set[tuple[int, int]]) -> dict:
    """Connectivity and degree summary for the grey structure under 4-neighborhood."""
    if not cells:
        return {
            "connected": False,
            "is_tree": False,
            "cell_count": 0,
            "edges": 0,
            "leaves": [],
            "branch_nodes": [],
        }
    adj = {cell: [nb for nb in neighbors4(*cell) if nb in cells] for cell in cells}
    q = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    edges = sum(len(nbs) for nbs in adj.values()) // 2
    leaves = sorted(cell for cell, nbs in adj.items() if len(nbs) == 1)
    branch_nodes = sorted(cell for cell, nbs in adj.items() if len(nbs) >= 3)
    connected = len(seen) == len(cells)
    return {
        "connected": connected,
        "is_tree": connected and edges == len(cells) - 1,
        "cell_count": len(cells),
        "edges": edges,
        "leaves": leaves,
        "branch_nodes": branch_nodes,
    }


def _find_tree_destinations(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[list[dict], list[tuple[int, int]]]:
    """Match every grey leaf to the unique sprite whose border it touches."""
    comps = components_mask(g == grey, connectivity=4)
    assert len(comps) == 1, "expected exactly one grey component"
    cells = {tuple(map(int, pt)) for pt in comps[0]}
    stats = _grey_tree_stats(cells)
    assert stats["is_tree"], "grey structure must be a 4-neighbor tree"
    leaf_cells = stats["leaves"]
    assert len(leaf_cells) >= 3, "expected a branched tree with at least three leaves"
    assert stats["branch_nodes"], "expected a real branch node in the grey tree"

    sprite_touch_sets = {id(sp): _all_border_touch_cells(sp, g.shape) for sp in sprites}
    dests: list[dict] = []
    seen_sprites: set[int] = set()
    for leaf in leaf_cells:
        cands = [sp for sp in sprites if leaf in sprite_touch_sets[id(sp)]]
        assert len(cands) == 1, "each grey leaf should touch exactly one sprite border"
        sp = cands[0]
        sid = id(sp)
        assert sid not in seen_sprites, "each destination sprite should be matched by only one grey leaf"
        seen_sprites.add(sid)
        dests.append(sp)
    return dests, leaf_cells


def _keep_only_connected_objects(out: np.ndarray, grey: int, dests: list[dict], bg: int) -> None:
    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dests:
        for r, c in d["pts"]:
            if out[r, c] in (bg, grey) or keep[r, c]:
                continue
            keep[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer source center colors into every sprite reached by a branched border-touching grey tree.

    Input structure:
    - The grid contains one grey (5) branching network and several small bordered sprites, each with one border color and one different center pixel.
    - The grey pixels form a single 4-neighbor tree, and each tree leaf touches some destination sprite at any border pixel rather than only through the old center-aligned port geometry.

    Rule:
    - Every sprite whose border is touched by a grey leaf is a destination and remains in the output.
    - For each destination, use its original center color as a lookup key: find the sprite whose border color matches that key, then copy that sprite's center color into the destination center.
    - Remove every sprite not matched by a grey leaf; keep the entire grey tree unchanged.

    Key insight:
    - The task now combines two shifts at once: you must trace all leaves of a branched grey tree, and each leaf can connect through any border pixel, so destination detection requires whole-border contact checks before applying the original border-to-center color lookup.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite"

    dests, _leaf_cells = _find_tree_destinations(g, grey, sprites)
    dest_ids = {id(sp) for sp in dests}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    _keep_only_connected_objects(out, grey, dests, bg)

    for d in dests:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination key should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


def _make_sprite_patch(border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 filled box sprite."""
    patch = np.full((3, 3), border_color, dtype=int)
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and int(patch[dr, dc]) != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and 4-touching with existing non-background, non-grey objects."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if int(patch[dr, dc]) == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if int(grid[rr, cc]) != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in neighbors4(rr, cc):
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, GREY):
                return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _center_distance_ok(center: tuple[int, int], centers: list[tuple[int, int]]) -> bool:
    return all(abs(center[0] - r) + abs(center[1] - c) >= MIN_CENTER_DIST for r, c in centers)


def _visible_patch_cells(patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and int(patch[dr, dc]) != BG:
                pts.append((rr, cc))
    return pts


def _placed_sprite_record(
    patch: np.ndarray, r0: int, c0: int, border_color: int, center_color: int,
) -> dict:
    pts = _visible_patch_cells(patch, r0, c0)
    center_pos = (r0 + 1, c0 + 1)
    return {
        "top_left": (r0, c0),
        "pts": pts,
        "border": border_color,
        "center": center_color,
        "center_pos": center_pos,
    }


def _touch_and_old_cells(sprite: dict, grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    touch = _border_touch_cells(sprite, grid)
    old_ports = _old_style_port_cells(sprite, grid.shape) & touch
    return touch, old_ports


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    remaining = list(range(DEST_COUNT, num_sprites))
    if not remaining:
        return None
    possible_sources = [n for n in SOURCE_COUNTS if 1 <= n <= len(remaining)]
    if not possible_sources:
        return None
    num_sources = random.choice(possible_sources)
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    source_center_pool = [c for c in range(1, 10) if c != GREY and c not in border_colors]
    random.shuffle(source_center_pool)
    if len(source_center_pool) < num_sources:
        return None
    source_center_colors = source_center_pool[:num_sources]

    if num_sources == 1:
        dest_to_source = [0, 0, 0]
    else:
        dest_to_source = [0, 1, random.choice([0, 1])]
        random.shuffle(dest_to_source)

    dest_center_colors = [border_colors[source_indices[idx]] for idx in dest_to_source]
    output_dest_centers = [source_center_colors[idx] for idx in dest_to_source]

    distractor_pool = [c for c in range(1, 10) if c != GREY and c not in border_colors]
    if not distractor_pool:
        return None
    distractor_center_colors = [random.choice(distractor_pool) for _ in distractor_indices]

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
    }


def _place_destinations(output_grid: np.ndarray, border_colors: list[int], output_dest_centers: list[int]) -> list[dict] | None:
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(*p)]
    int_pos = [p for p in all_pos if not _is_edge(*p)]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dests: list[dict] = []
    centers: list[tuple[int, int]] = []
    for i in range(DEST_COUNT):
        patch = _make_sprite_patch(border_colors[i], output_dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            center = (r0 + 1, c0 + 1)
            if not _center_distance_ok(center, centers):
                continue
            _place_sprite(output_grid, patch, r0, c0)
            record = _placed_sprite_record(patch, r0, c0, border_colors[i], output_dest_centers[i])
            touch, _old = _touch_and_old_cells(record, output_grid)
            if touch:
                dests.append(record)
                centers.append(center)
                placed = True
                break
            _erase_sprite(output_grid, patch, r0, c0)
        if not placed:
            return None

    touch_map: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dests):
        touch, old_ports = _touch_and_old_cells(d, output_grid)
        d["touch_cells"] = sorted(touch)
        d["old_touch_cells"] = sorted(old_ports)
        for cell in touch:
            touch_map.setdefault(cell, []).append(i)

    for i, d in enumerate(dests):
        unique_touch = [cell for cell in d["touch_cells"] if touch_map[cell] == [i]]
        unique_new_touch = [cell for cell in unique_touch if cell not in d["old_touch_cells"]]
        d["unique_touch_cells"] = unique_touch
        d["unique_new_touch_cells"] = unique_new_touch

    if any(not d["unique_touch_cells"] for d in dests):
        return None
    if not any(d["unique_new_touch_cells"] for d in dests):
        return None
    return dests


def _random_bfs_path(
    blocked: set[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path: list[tuple[int, int]] = []
            at = cur
            while at is not None:
                path.append(at)
                at = prev[at]
            path.reverse()
            return path
        nbs = list(neighbors4(*cur))
        random.shuffle(nbs)
        for nb in nbs:
            if nb in prev:
                continue
            if not (0 <= nb[0] < GRID_H and 0 <= nb[1] < GRID_W):
                continue
            if nb != goal and nb in blocked:
                continue
            prev[nb] = cur
            q.append(nb)
    return None


def _connect_leaves_via_hub(
    output_grid: np.ndarray,
    leaves: tuple[tuple[int, int], ...],
    hub: tuple[int, int],
    blocked_touch: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    sprite_cells = {tuple(map(int, pt)) for pt in zip(*np.where(output_grid != BG), strict=False)}
    blocked_base = sprite_cells | blocked_touch
    if hub in blocked_base or hub in leaves:
        return None
    hub_neighbors = [
        (hub[0] + dr, hub[1] + dc)
        for dr, dc in DIRS4
        if 0 <= hub[0] + dr < GRID_H and 0 <= hub[1] + dc < GRID_W
    ]
    if sum(1 for nb in hub_neighbors if nb not in blocked_base) < DEST_COUNT:
        return None

    perm_list = list(permutations(range(len(leaves))))
    sample_count = min(6, len(perm_list))
    for order in random.sample(perm_list, k=sample_count):
        occupied = {hub}
        ok = True
        for idx in order:
            start = leaves[idx]
            other_leaves = {leaves[j] for j in range(len(leaves)) if j != idx}
            blocked = blocked_base | (occupied - {hub}) | other_leaves
            path = _random_bfs_path(blocked, start, hub)
            if path is None:
                ok = False
                break
            occupied.update(path)
        if not ok:
            continue
        stats = _grey_tree_stats(occupied)
        if not stats["is_tree"]:
            continue
        if stats["leaves"] != sorted(leaves):
            continue
        if hub not in stats["branch_nodes"]:
            continue
        if not (TREE_CELLS_MIN <= stats["cell_count"] <= TREE_CELLS_MAX):
            continue
        return occupied
    return None


def _choose_tree(
    output_grid: np.ndarray, dests: list[dict],
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[tuple[int, int], ...]] | None:
    leaf_products = list(product(*(d["unique_touch_cells"] for d in dests)))
    leaf_products = [cells for cells in leaf_products if len(set(cells)) == DEST_COUNT]
    random.shuffle(leaf_products)

    all_touch_cells = {
        cell
        for d in dests
        for cell in d["touch_cells"]
    }
    hub_candidates = [
        (r, c)
        for r in range(1, GRID_H - 1)
        for c in range(1, GRID_W - 1)
        if int(output_grid[r, c]) == BG and (r, c) not in all_touch_cells
    ]
    random.shuffle(hub_candidates)

    for leaves in leaf_products:
        if not any(leaves[i] in dests[i]["unique_new_touch_cells"] for i in range(DEST_COUNT)):
            continue
        blocked_touch = all_touch_cells - set(leaves)
        for hub in hub_candidates:
            if min(abs(leaf[0] - hub[0]) + abs(leaf[1] - hub[1]) for leaf in leaves) < 2:
                continue
            tree_cells = _connect_leaves_via_hub(output_grid, leaves, hub, blocked_touch)
            if tree_cells is not None:
                return tree_cells, hub, leaves
    return None


def _conflicts_with_grey_contacts(
    grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, border_color: int, center_color: int,
    grey_cells: set[tuple[int, int]],
) -> bool:
    _place_sprite(grid, patch, r0, c0)
    record = _placed_sprite_record(patch, r0, c0, border_color, center_color)
    touch = _all_border_touch_cells(record, grid.shape)
    _erase_sprite(grid, patch, r0, c0)
    return bool(touch & grey_cells)


def _fusion_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sprites = _detect_bordered_sprites(g, BG, GREY)
    grey_comps = components_mask(g == GREY, connectivity=4)
    assert len(grey_comps) == 1
    grey_cells = {tuple(map(int, pt)) for pt in grey_comps[0]}
    stats = _grey_tree_stats(grey_cells)
    touch_sets = {id(sp): _all_border_touch_cells(sp, g.shape) for sp in sprites}
    old_sets = {id(sp): _old_style_port_cells(sp, g.shape) for sp in sprites}
    non_old_leaves = 0
    for leaf in stats["leaves"]:
        matched = [sp for sp in sprites if leaf in touch_sets[id(sp)]]
        if len(matched) != 1:
            continue
        if leaf not in old_sets[id(matched[0])]:
            non_old_leaves += 1
    return {
        "leaf_count": len(stats["leaves"]),
        "branch_count": len(stats["branch_nodes"]),
        "non_old_leaf_count": non_old_leaves,
    }


def _gen_attempt() -> np.ndarray | None:
    palette = [c for c in range(1, 10) if c != GREY]
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(palette, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dests = _place_destinations(output_grid, border_colors, colors["output_dest_centers"])
    if dests is None:
        return None

    tree_info = _choose_tree(output_grid, dests)
    if tree_info is None:
        return None
    tree_cells, _hub, chosen_leaves = tree_info

    for r, c in tree_cells:
        if int(output_grid[r, c]) != BG:
            return None
        output_grid[r, c] = GREY

    stats = _grey_tree_stats(tree_cells)
    if stats["leaves"] != sorted(chosen_leaves):
        return None

    input_grid = output_grid.copy()
    for i, d in enumerate(dests):
        cr, cc = d["center_pos"]
        input_grid[cr, cc] = colors["dest_center_colors"][i]

    grey_cells = set(tree_cells)
    sprites_to_place = (
        [(border_colors[si], colors["source_center_colors"][j]) for j, si in enumerate(colors["source_indices"])]
        + [(border_colors[di], colors["distractor_center_colors"][j]) for j, di in enumerate(colors["distractor_indices"])]
    )
    all_pos = _sprite_positions()
    for border_color, center_color in sprites_to_place:
        patch = _make_sprite_patch(border_color, center_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            if _conflicts_with_grey_contacts(input_grid, patch, r0, c0, border_color, center_color, grey_cells):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(input_grid)
        metrics = _fusion_metrics(input_grid)
    except Exception:
        return None
    if metrics["leaf_count"] < 3 or metrics["branch_count"] < 1 or metrics["non_old_leaf_count"] < 1:
        return None
    if np.array_equal(out, output_grid):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
