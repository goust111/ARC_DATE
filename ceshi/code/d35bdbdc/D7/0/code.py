# Variant of d35bdbdc
# Dimension: D7 — 灰色路径结构 (meta)
# Change: 灰色像素构成连通路径，端点连接到目标精灵的中心位置... → 分支路径（树形结构）

import random
from collections import deque
from itertools import product, permutations

import numpy as np

from arc_utils import components_mask, neighbors4, paste


# Design analysis for the D7 variant
# ==================================
# concretization:
#   - Keep the original bordered 3x3 sprites and the same color-transfer rule.
#   - Replace the old "one grey trail with exactly two endpoints" by a single grey
#     4-neighbor tree with exactly three leaf cells. Each leaf is a valid port for
#     a different destination sprite.
#   - The port geometry stays visible and unchanged: a grey leaf sits two Manhattan
#     steps from a sprite center and the intermediate cell is that sprite's border.
#   - Input therefore shows three destination sprites tied together by one branched
#     grey network; output keeps that whole network and only those connected
#     destinations, while other sprites disappear.
#   - Concrete execution change:
#       old: grey leaves = 2 -> directly pick 2 destinations.
#       new: grey leaves = 3 -> collect every grey leaf, match each leaf to the
#            unique sprite for which it is a valid port, then update all matched
#            destinations.
#
# complexity calibration:
#   - Three leaf ports are enough to make branching visually obvious; two would
#     collapse back to the old linear behavior.
#   - Destination count is fixed at 3, while source count stays 1-2. This keeps the
#     color-lookup rule recognizable without exploding the number of cases.
#   - The generator still uses 4-5 sprites total, so examples remain compact and
#     readable on a 10x10 grid.
#
# implicit-assumption audit of the original main():
#   - _find_trail_endpoints assumed exactly two grey endpoints or a one-stroke DFS.
#     This fails for branches because a tree has more than two leaves.
#   - trail_eps / eps_pair / fixed-length dest list all assumed "2 destinations".
#   - The old bipartite tiebreak handled ambiguous endpoint assignment. In this
#     variant the generator instead guarantees that every leaf port belongs to
#     exactly one sprite, so the solver can use direct matching.
#   - The deletion and border->source lookup logic do not depend on linearity and
#     can be retained almost unchanged.
#
# placement legality:
#   - Destination sprites are placed first.
#   - After that the generator chooses one hub background cell and connects the
#     three chosen port cells to that hub with pairwise non-overlapping 4-neighbor
#     branches. This guarantees a connected acyclic grey skeleton with a real branch.
#   - Non-destination sprites are placed only if they do not overlap existing cells
#     and do not form any valid port relation with any grey cell, so the solver sees
#     exactly the intended three destinations.
#
# visual observability and cleanliness:
#   - Using a 4-neighbor tree makes the branch point and the three leaves stable and
#     easy to spot. This is more legible than relying on 8-neighbor diagonal
#     "pseudo-branches", which can look like thick noise.
#   - The center-color transfer rule is unchanged, so the new thing a human has to
#     notice is specifically "follow all branches to all connected sprites".
#
# color inferability:
#   - Border colors remain unique.
#   - Every destination center is still a lookup key into some source border color.
#   - With 1-2 sources and 3 destinations, repeated lookups appear naturally, which
#     gives the human multiple supporting examples of the same mapping behavior.
#
# post-change solver assumptions that the generator must guarantee:
#   - There is exactly one grey 4-connected component.
#   - Its leaves are exactly the destination ports.
#   - Each grey leaf matches exactly one sprite, and each destination sprite receives
#     exactly one leaf.
#   - At least one grey cell has 4-degree >= 3 so the structure is a true branch.


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
DEST_COUNT = 3
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [3, 1]  # weights for [0, 1] interior destinations
TREE_CELLS_MIN, TREE_CELLS_MAX = 10, 24
MIN_CENTER_DIST = 4
GEN_RETRIES = 500
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
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _sprite_port_cells(sprite: dict, g: np.ndarray) -> list[tuple[int, int]]:
    """All in-bounds cells that satisfy the sprite's 2-step port geometry."""
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < g.shape[0] and 0 <= ic < g.shape[1]):
            continue
        if not (0 <= er < g.shape[0] and 0 <= ec < g.shape[1]):
            continue
        if int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def _grey_tree_stats(cells: set[tuple[int, int]]) -> dict:
    """Connectivity / degree summary under 4-neighborhood."""
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
    """Match every grey leaf of the 4-neighbor tree to exactly one sprite port."""
    comps = components_mask(g == grey, connectivity=4)
    assert len(comps) == 1, "expected one grey tree component"
    cells = set((int(r), int(c)) for r, c in comps[0])
    stats = _grey_tree_stats(cells)
    assert stats["is_tree"], "grey structure must be an acyclic 4-neighbor tree"
    leaf_cells = stats["leaves"]
    assert len(leaf_cells) >= 3, "expected at least three grey leaves for branching"
    assert stats["branch_nodes"], "expected at least one branching grey cell"

    sprite_port_sets = {id(sp): set(_sprite_port_cells(sp, g)) for sp in sprites}
    dests: list[dict] = []
    seen_sprites: set[int] = set()
    for leaf in leaf_cells:
        cands = [sp for sp in sprites if leaf in sprite_port_sets[id(sp)]]
        assert len(cands) == 1, "each grey leaf should connect to exactly one sprite"
        sp = cands[0]
        sid = id(sp)
        assert sid not in seen_sprites, "each destination sprite should have exactly one leaf"
        seen_sprites.add(sid)
        dests.append(sp)
    return dests, leaf_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors into every sprite reached by a branched grey tree, then remove the rest.

    Input structure:
    - The grid contains a grey (5) branching network plus several small bordered sprites; each sprite has one border color and one distinct center pixel.
    - The grey pixels form one 4-neighbor tree rather than a single stroke, and each grey leaf sits at a valid 2-step port of a different destination sprite.

    Rule:
    - Every sprite touched by a grey leaf port is a destination and remains in the output.
    - For each destination, use its original center color as a lookup key: find the source sprite whose border color matches that key, and copy the source center color into the destination center.
    - Delete every sprite that is not reached by a grey leaf; keep the full grey tree unchanged.

    Key insight:
    - The hard part is no longer identifying two trail endpoints; it is tracing the whole grey tree and collecting all leaf-connected sprites before applying the same border-to-center color lookup as in the original task.
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

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dests:
        for r, c in d["pts"]:
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc]:
                    continue
                if out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

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


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return the 3x3 sprite patch for the chosen shape."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in DIRS4:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and 4-touching with existing non-background, non-grey objects."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
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
    """All top-left positions whose center cell stays on the grid."""
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


def _valid_port_cells_for_center(
    grid: np.ndarray, cr: int, cc: int, border_color: int,
) -> list[tuple[int, int]]:
    """All in-bounds 2-step port cells whose intermediate cell is border-colored."""
    ports: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W):
            continue
        if not (0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == border_color:
            ports.append((er, ec))
    return ports


def _open_port_cells_for_center(
    grid: np.ndarray, cr: int, cc: int, border_color: int,
) -> list[tuple[int, int]]:
    return [p for p in _valid_port_cells_for_center(grid, cr, cc, border_color) if int(grid[p]) == BG]


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    dest_indices = [0, 1, 2]
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
        "dest_indices": dest_indices,
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
    }


def _place_destinations(
    output_grid: np.ndarray, shape_type: str, border_colors: list[int], output_dest_centers: list[int],
) -> list[dict] | None:
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(*p)]
    int_pos = [p for p in all_pos if not _is_edge(*p)]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dests: list[dict] = []
    centers: list[tuple[int, int]] = []
    for i in range(DEST_COUNT):
        patch = _make_sprite_patch(shape_type, border_colors[i], output_dest_centers[i])
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
            ports = _open_port_cells_for_center(output_grid, center[0], center[1], border_colors[i])
            if ports:
                dests.append(
                    {
                        "top_left": (r0, c0),
                        "center_pos": center,
                        "border": border_colors[i],
                        "ports": sorted(ports),
                    }
                )
                centers.append(center)
                placed = True
                break
            _erase_sprite(output_grid, patch, r0, c0)
        if not placed:
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


def _connect_ports_via_hub(
    output_grid: np.ndarray, ports: tuple[tuple[int, int], ...], hub: tuple[int, int],
) -> set[tuple[int, int]] | None:
    sprite_cells = set((int(r), int(c)) for r, c in zip(*np.where(output_grid != BG), strict=False))
    if hub in sprite_cells or hub in ports:
        return None
    hub_neighbors = [(hub[0] + dr, hub[1] + dc) for dr, dc in DIRS4]
    open_hub_neighbors = [nb for nb in hub_neighbors if 0 <= nb[0] < GRID_H and 0 <= nb[1] < GRID_W and nb not in sprite_cells]
    if len(open_hub_neighbors) < DEST_COUNT:
        return None

    perm_list = list(permutations(range(len(ports))))
    for order in random.sample(perm_list, k=min(6, len(perm_list))):
        occupied = {hub}
        ok = True
        for idx in order:
            start = ports[idx]
            other_ports = {ports[j] for j in range(len(ports)) if j != idx}
            blocked = sprite_cells | (occupied - {hub}) | other_ports
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
        if stats["leaves"] != sorted(ports):
            continue
        if hub not in stats["branch_nodes"]:
            continue
        if not (TREE_CELLS_MIN <= stats["cell_count"] <= TREE_CELLS_MAX):
            continue
        return occupied
    return None


def _ports_match_destinations_uniquely(
    output_grid: np.ndarray, dests: list[dict], ports: tuple[tuple[int, int], ...],
) -> bool:
    for i, leaf in enumerate(ports):
        hits = []
        for j, d in enumerate(dests):
            cr, cc = d["center_pos"]
            if leaf in _valid_port_cells_for_center(output_grid, cr, cc, d["border"]):
                hits.append(j)
        if hits != [i]:
            return False
    return True


def _choose_tree(
    output_grid: np.ndarray, dests: list[dict],
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[tuple[int, int], ...]] | None:
    port_products = list(product(*(d["ports"] for d in dests)))
    port_products = [ports for ports in port_products if len(set(ports)) == DEST_COUNT]
    random.shuffle(port_products)

    hub_candidates = [
        (r, c)
        for r in range(1, GRID_H - 1)
        for c in range(1, GRID_W - 1)
        if int(output_grid[r, c]) == BG
    ]
    random.shuffle(hub_candidates)

    for hub in hub_candidates:
        for ports in port_products:
            if not _ports_match_destinations_uniquely(output_grid, dests, ports):
                continue
            if min(abs(p[0] - hub[0]) + abs(p[1] - hub[1]) for p in ports) < 2:
                continue
            tree_cells = _connect_ports_via_hub(output_grid, ports, hub)
            if tree_cells is not None:
                return tree_cells, hub, ports
    return None


def _conflicts_with_grey_ports(
    grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, border_color: int, grey_cells: set[tuple[int, int]],
) -> bool:
    _place_sprite(grid, patch, r0, c0)
    center = (r0 + 1, c0 + 1)
    ports = set(_valid_port_cells_for_center(grid, center[0], center[1], border_color))
    _erase_sprite(grid, patch, r0, c0)
    return bool(ports & grey_cells)


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    palette = [c for c in range(1, 10) if c != GREY]
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(palette, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dests = _place_destinations(output_grid, shape_type, border_colors, colors["output_dest_centers"])
    if dests is None:
        return None

    tree_info = _choose_tree(output_grid, dests)
    if tree_info is None:
        return None
    tree_cells, _hub, chosen_ports = tree_info

    for r, c in tree_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    stats = _grey_tree_stats(tree_cells)
    if stats["leaves"] != sorted(chosen_ports):
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
        patch = _make_sprite_patch(shape_type, border_color, center_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            if _conflicts_with_grey_ports(input_grid, patch, r0, c0, border_color, grey_cells):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(input_grid)
    except Exception:
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
