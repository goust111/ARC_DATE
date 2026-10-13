from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ANCHOR = 5
PATH = 7
BODY_COLORS = [1, 2, 3, 4, 6, 8, 9]
GEN_TRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    deltas = list(_neighbors8(0, 0)) if connectivity == 8 else list(_neighbors4(0, 0))
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _sprite_rel_cells_from_crop(crop: np.ndarray) -> dict[tuple[int, int], int]:
    anchor_pos = np.argwhere(crop == ANCHOR)
    assert anchor_pos.shape[0] == 1
    ar, ac = map(int, anchor_pos[0])
    rel_cells: dict[tuple[int, int], int] = {}
    for r in range(crop.shape[0]):
        for c in range(crop.shape[1]):
            color = int(crop[r, c])
            if color != BG:
                rel_cells[(r - ar, c - ac)] = color
    return rel_cells


def _render_rel_cells(rel_cells: dict[tuple[int, int], int]) -> np.ndarray:
    assert rel_cells
    rows = [r for r, _ in rel_cells]
    cols = [c for _, c in rel_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in rel_cells.items():
        out[r - r0, c - c0] = int(color)
    return out


def _overlay_rel_sprites(sprites: list[dict]) -> np.ndarray:
    rel_cells: dict[tuple[int, int], int] = {}
    for sprite in sprites:
        for pos, color in sprite["rel_cells"].items():
            rel_cells[pos] = int(color)
    return _render_rel_cells(rel_cells)


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != PATH)
    comps = _component_coords(mask, connectivity=8)
    sprites: list[dict] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        assert ANCHOR in colors and len(colors) == 2
        anchor_cells = [(r, c) for r, c in comp if int(grid[r, c]) == ANCHOR]
        assert len(anchor_cells) == 1
        body_colors = [c for c in colors if c != ANCHOR]
        assert len(body_colors) == 1 and body_colors[0] not in (BG, PATH)
        bbox = _bbox_from_coords(comp)
        crop = _crop_bbox(grid, bbox)
        rel_cells = _sprite_rel_cells_from_crop(crop)
        h, w = crop.shape
        assert 3 <= h <= 5 and 3 <= w <= 5
        sprites.append(
            {
                "coords": comp,
                "bbox": bbox,
                "crop": crop,
                "rel_cells": rel_cells,
                "body_color": int(body_colors[0]),
            }
        )
    assert 4 <= len(sprites) <= 6
    return sprites


def _bbox_adjacent(node: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = node
    r0, c0, r1, c1 = bbox
    for nr, nc in _neighbors4(r, c):
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            return True
    return False


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    path_nodes = {(int(r), int(c)) for r, c in np.argwhere(g == PATH)}
    border_sources = [
        (r, c)
        for r, c in path_nodes
        if r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1)
    ]
    assert len(border_sources) == 1
    source = border_sources[0]

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in path_nodes:
        cur = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nxt = (r + dr, c + dc)
            if nxt in path_nodes:
                cur.append(nxt)
        neighbors[(r, c)] = cur

    dist = {source: 0}
    q = deque([source])
    while q:
        node = q.popleft()
        for nxt in neighbors[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    assert len(dist) == len(path_nodes)

    endpoints = [
        node
        for node, nbrs in neighbors.items()
        if node != source and len(nbrs) == 1
    ]
    assert len(endpoints) >= 2

    sprites = _extract_sprites(g)
    reached: dict[int, tuple[int, tuple[int, int]]] = {}
    for endpoint in endpoints:
        matches = [i for i, sprite in enumerate(sprites) if _bbox_adjacent(endpoint, sprite["bbox"])]
        assert len(matches) <= 1
        if not matches:
            continue
        idx = matches[0]
        key = (int(dist[endpoint]), endpoint)
        if idx not in reached or key < reached[idx]:
            reached[idx] = key

    assert len(reached) >= 2
    ordered = [
        {
            "index": idx,
            "dist": info[0],
            "endpoint": info[1],
            **sprites[idx],
        }
        for idx, info in sorted(reached.items(), key=lambda item: (item[1][0], item[1][1]))
    ]
    out = _overlay_rel_sprites(ordered)
    if return_meta:
        return out, {"sprites": sprites, "selected": ordered, "source": source, "endpoints": endpoints}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the sprites reached by the sparse orange path, align them by gray anchors, and crop the overlay.

    Input structure:
    - The grid has black background, 4-6 separated multicolor sprites, and one sparse orange path starting from a border source.
    - Each sprite contains exactly one gray anchor pixel plus one unique body color, and each sprite fits inside a 3x3 to 5x5 bounding box.
    - Orange path pixels are connected only by horizontal or vertical jumps of two cells, and some path endpoints sit 4-neighbor adjacent to sprite bounding boxes.

    Rule:
    - Find the unique orange source pixel on the border and follow the whole connected orange path using two-cell horizontal or vertical steps.
    - Keep only the sprites whose bounding boxes are adjacent to path endpoints, ordered by the first time their endpoint is reached from the source.
    - Crop each selected sprite to its own bounding box, align all crops so their gray anchors coincide, overlay later-reached sprites on top of earlier ones, then crop to the minimal non-black bounding box.

    Key insight:
    - The orange path is not output itself; it decides which sprites participate in the anchor-based overlay and in what stacking order.
    """
    return _solve(grid, return_meta=False)


def _random_sprite(body_color: int) -> dict | None:
    for _ in range(200):
        target_area = random.randint(4, 8)
        body: set[tuple[int, int]] = set()
        frontier = [(dr, dc) for dr, dc in _neighbors4(0, 0) if -2 <= dr <= 2 and -2 <= dc <= 2]
        while frontier and len(body) < target_area:
            cell = random.choice(frontier)
            body.add(cell)
            frontier = []
            seed = body | {(0, 0)}
            for r, c in seed:
                for nr, nc in _neighbors4(r, c):
                    if -2 <= nr <= 2 and -2 <= nc <= 2 and (nr, nc) not in body and (nr, nc) != (0, 0):
                        frontier.append((nr, nc))
        if len(body) < 4:
            continue
        cells = set(body) | {(0, 0)}
        bbox = _bbox_from_coords(list(cells))
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        if not (3 <= h <= 5 and 3 <= w <= 5):
            continue
        rel_cells = {(0, 0): ANCHOR}
        for pos in body:
            rel_cells[pos] = int(body_color)
        crop = _render_rel_cells(rel_cells)
        return {"body_color": int(body_color), "rel_cells": rel_cells, "bbox_rel": bbox, "crop": crop}
    return None


def _build_tree(h: int, w: int, selected_count: int) -> dict | None:
    lattice_h = (h + 1) // 2
    lattice_w = (w + 1) // 2
    if lattice_h < 8 or lattice_w < 8:
        return None

    for _ in range(200):
        src_r = random.randint(3, lattice_h - 4)
        c1 = random.randint(2, 3)
        source = (src_r, 0)
        nodes = {source}
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

        def add_line(start: tuple[int, int], dr: int, dc: int, length: int):
            cur = start
            for _ in range(length):
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nxt[0] < lattice_h and 0 <= nxt[1] < lattice_w):
                    return None
                if nxt in nodes:
                    return None
                edges.append((cur, nxt))
                nodes.add(nxt)
                cur = nxt
            return cur

        j1 = add_line(source, 0, 1, c1)
        if j1 is None:
            continue

        leaves: list[tuple[int, int]] = []
        if selected_count == 2:
            up_max = min(4, src_r)
            down_max = min(4, lattice_h - 1 - src_r)
            if up_max < 2 or down_max < 2:
                continue
            up_len = random.randint(2, up_max)
            down_len = random.randint(2, down_max)
            if c1 + up_len == c1 + down_len:
                continue
            up_leaf = add_line(j1, -1, 0, up_len)
            down_leaf = add_line(j1, 1, 0, down_len)
            if up_leaf is None or down_leaf is None:
                continue
            leaves = [up_leaf, down_leaf]
        elif selected_count == 3:
            up_max = min(4, src_r)
            down_max = min(4, lattice_h - 1 - src_r)
            right_max = min(4, lattice_w - 1 - c1)
            if min(up_max, down_max, right_max) < 2:
                continue
            lens = [random.randint(2, up_max), random.randint(2, down_max), random.randint(2, right_max)]
            dists = {c1 + lens[0], c1 + lens[1], c1 + lens[2]}
            if len(dists) != 3:
                continue
            up_leaf = add_line(j1, -1, 0, lens[0])
            down_leaf = add_line(j1, 1, 0, lens[1])
            right_leaf = add_line(j1, 0, 1, lens[2])
            if up_leaf is None or down_leaf is None or right_leaf is None:
                continue
            leaves = [up_leaf, down_leaf, right_leaf]
        else:
            c2 = random.randint(c1 + 2, min(lattice_w - 3, c1 + 4))
            j2 = add_line(j1, 0, 1, c2 - c1)
            if j2 is None:
                continue
            up1_max = min(4, src_r)
            down1_max = min(4, lattice_h - 1 - src_r)
            up2_max = min(4, src_r)
            down2_max = min(4, lattice_h - 1 - src_r)
            if min(up1_max, down1_max, up2_max, down2_max) < 2:
                continue
            lens = [
                random.randint(2, up1_max),
                random.randint(2, down1_max),
                random.randint(2, up2_max),
                random.randint(2, down2_max),
            ]
            dists = {c1 + lens[0], c1 + lens[1], c2 + lens[2], c2 + lens[3]}
            if len(dists) != 4:
                continue
            up1 = add_line(j1, -1, 0, lens[0])
            down1 = add_line(j1, 1, 0, lens[1])
            up2 = add_line(j2, -1, 0, lens[2])
            down2 = add_line(j2, 1, 0, lens[3])
            if up1 is None or down1 is None or up2 is None or down2 is None:
                continue
            leaves = [up1, down1, up2, down2]

        degree = {node: 0 for node in nodes}
        graph: dict[tuple[int, int], list[tuple[int, int]]] = {node: [] for node in nodes}
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
            graph[a].append(b)
            graph[b].append(a)
        actual_nodes = {(2 * r, 2 * c) for r, c in nodes}
        actual_segments = set(actual_nodes)
        for (r0, c0), (r1, c1_) in edges:
            actual_segments.add((r0 + r1, c0 + c1_))
        dist = {source: 0}
        q = deque([source])
        while q:
            node = q.popleft()
            for nxt in graph[node]:
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    q.append(nxt)
        ordered_leaves = sorted(
            [(leaf, dist[leaf]) for leaf in leaves],
            key=lambda item: (item[1], item[0]),
        )
        if len({d for _, d in ordered_leaves}) != selected_count:
            continue
        return {
            "source": (2 * source[0], 2 * source[1]),
            "path_nodes": actual_nodes,
            "path_segments": actual_segments,
            "ordered_leaves": [((2 * leaf[0], 2 * leaf[1]), d) for leaf, d in ordered_leaves],
        }
    return None


def _shift_bbox(bbox_rel: tuple[int, int, int, int], anchor: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox_rel
    ar, ac = anchor
    return ar + r0, ac + c0, ar + r1, ac + c1


def _bbox_conflicts(new_bbox: tuple[int, int, int, int], placed_bboxes: list[tuple[int, int, int, int]]) -> bool:
    nr0, nc0, nr1, nc1 = new_bbox
    for r0, c0, r1, c1 in placed_bboxes:
        if not (nr1 < r0 - 1 or nr0 > r1 + 1 or nc1 < c0 - 1 or nc0 > c1 + 1):
            return True
    return False


def _bbox_contains_any(bbox: tuple[int, int, int, int], cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)


def _adjacent_path_nodes(bbox: tuple[int, int, int, int], path_nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [node for node in path_nodes if _bbox_adjacent(node, bbox)]


def _place_target_sprite(
    sprite: dict,
    leaf: tuple[int, int],
    h: int,
    w: int,
    path_nodes: set[tuple[int, int]],
    path_segments: set[tuple[int, int]],
    placed_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    minr, minc, maxr, maxc = sprite["bbox_rel"]
    row_lo = -minr
    row_hi = h - 1 - maxr
    col_lo = -minc
    col_hi = w - 1 - maxc
    lr, lc = leaf
    sides = ["E", "W", "S", "N"]
    random.shuffle(sides)

    for side in sides:
        if side == "E":
            anchor_c = lc + 1 - minc
            if not (col_lo <= anchor_c <= col_hi):
                continue
            lo = max(row_lo, lr - maxr)
            hi = min(row_hi, lr - minr)
            axis = "row"
        elif side == "W":
            anchor_c = lc - 1 - maxc
            if not (col_lo <= anchor_c <= col_hi):
                continue
            lo = max(row_lo, lr - maxr)
            hi = min(row_hi, lr - minr)
            axis = "row"
        elif side == "S":
            anchor_r = lr + 1 - minr
            if not (row_lo <= anchor_r <= row_hi):
                continue
            lo = max(col_lo, lc - maxc)
            hi = min(col_hi, lc - minc)
            axis = "col"
        else:
            anchor_r = lr - 1 - maxr
            if not (row_lo <= anchor_r <= row_hi):
                continue
            lo = max(col_lo, lc - maxc)
            hi = min(col_hi, lc - minc)
            axis = "col"

        if lo > hi:
            continue
        choices = list(range(lo, hi + 1))
        random.shuffle(choices)
        for free in choices:
            if axis == "row":
                anchor = (free, anchor_c)
            else:
                anchor = (anchor_r, free)
            bbox = _shift_bbox(sprite["bbox_rel"], anchor)
            if _bbox_conflicts(bbox, placed_bboxes):
                continue
            if _bbox_contains_any(bbox, path_segments):
                continue
            adjacent = _adjacent_path_nodes(bbox, path_nodes)
            if adjacent != [leaf]:
                continue
            return anchor
    return None


def _place_free_sprite(
    sprite: dict,
    h: int,
    w: int,
    path_nodes: set[tuple[int, int]],
    path_segments: set[tuple[int, int]],
    placed_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    minr, minc, maxr, maxc = sprite["bbox_rel"]
    row_lo = -minr
    row_hi = h - 1 - maxr
    col_lo = -minc
    col_hi = w - 1 - maxc
    for _ in range(300):
        anchor = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
        bbox = _shift_bbox(sprite["bbox_rel"], anchor)
        if _bbox_conflicts(bbox, placed_bboxes):
            continue
        if _bbox_contains_any(bbox, path_segments):
            continue
        if _adjacent_path_nodes(bbox, path_nodes):
            continue
        return anchor
    return None


def _paint_sprite(grid: np.ndarray, sprite: dict, anchor: tuple[int, int]) -> None:
    ar, ac = anchor
    for (dr, dc), color in sprite["rel_cells"].items():
        grid[ar + dr, ac + dc] = int(color)


def _sprite_set_semantics(sprites: list[dict], selected_count: int) -> bool:
    selected = sprites[:selected_count]
    unselected = sprites[selected_count:]
    selected_out = _overlay_rel_sprites(selected)
    if any(np.array_equal(_overlay_rel_sprites(selected[:i] + selected[i + 1 :]), selected_out) for i in range(selected_count)):
        return False
    if np.array_equal(_overlay_rel_sprites(list(reversed(selected))), selected_out):
        return False
    if any(np.array_equal(sprite["crop"], selected_out) for sprite in selected):
        return False
    if not any(not np.array_equal(_overlay_rel_sprites(selected + [sprite]), selected_out) for sprite in unselected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.choice([19, 21, 23, 25])
    w = random.choice([19, 21, 23, 25])
    total_sprites = random.randint(4, 6)
    selected_count = random.randint(2, min(4, total_sprites - 1))
    colors = random.sample(BODY_COLORS, total_sprites)

    sprites: list[dict] = []
    for color in colors:
        sprite = _random_sprite(color)
        if sprite is None:
            return None
        sprites.append(sprite)
    if not _sprite_set_semantics(sprites, selected_count):
        return None

    tree = _build_tree(h, w, selected_count)
    if tree is None:
        return None

    path_nodes = tree["path_nodes"]
    path_segments = tree["path_segments"]
    placed_bboxes: list[tuple[int, int, int, int]] = []
    anchors: list[tuple[int, int]] = []

    for sprite, (leaf, _dist) in zip(sprites[:selected_count], tree["ordered_leaves"]):
        anchor = _place_target_sprite(sprite, leaf, h, w, path_nodes, path_segments, placed_bboxes)
        if anchor is None:
            return None
        anchors.append(anchor)
        placed_bboxes.append(_shift_bbox(sprite["bbox_rel"], anchor))

    for sprite in sprites[selected_count:]:
        anchor = _place_free_sprite(sprite, h, w, path_nodes, path_segments, placed_bboxes)
        if anchor is None:
            return None
        anchors.append(anchor)
        placed_bboxes.append(_shift_bbox(sprite["bbox_rel"], anchor))

    grid = np.zeros((h, w), dtype=int)
    for r, c in path_nodes:
        grid[r, c] = PATH
    for sprite, anchor in zip(sprites, anchors):
        _paint_sprite(grid, sprite, anchor)

    rot = random.randrange(4)
    if rot:
        grid = np.rot90(grid, rot).copy()

    out, meta = _solve(grid, return_meta=True)
    if len(meta["sprites"]) != total_sprites:
        return None
    if len(meta["selected"]) != selected_count:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 2")
