from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
BLOCK_COLORS = [1, 2, 3, 4, 6, 8, 9]
GROUP_RETRIES = 80
GEN_RETRIES = 200


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _extract_blocks(
    grid: np.ndarray, bg: int
) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    block_id_grid = np.full((h, w), -1, dtype=int)
    blocks: list[dict] = []
    for color in sorted(set(map(int, grid.reshape(-1))) - {int(bg), GREY, ORANGE}):
        for coords in _connected_components(grid == int(color)):
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = _bbox(coords)
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            area = int(coords.shape[0])
            if area != bh * bw:
                continue
            if not bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))):
                continue
            bid = len(blocks)
            block_id_grid[coords[:, 0], coords[:, 1]] = bid
            blocks.append(
                {
                    "id": bid,
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": area,
                    "short": min(bh, bw),
                }
            )
    return blocks, block_id_grid


def _group_blocks(grid: np.ndarray, bg: int) -> tuple[list[dict], list[tuple[np.ndarray, set[int]]]]:
    blocks, block_id_grid = _extract_blocks(grid, bg)
    if not blocks:
        return [], []

    dsu = DSU(len(blocks))
    line_infos: list[tuple[np.ndarray, set[int]]] = []
    for coords in _connected_components(grid == GREY):
        touched: set[int] = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if _in_bounds(grid.shape, nr, nc):
                    bid = int(block_id_grid[nr, nc])
                    if bid >= 0:
                        touched.add(bid)
        if len(touched) >= 2:
            touched_list = sorted(touched)
            for other in touched_list[1:]:
                dsu.union(touched_list[0], other)
            line_infos.append((coords, touched))

    grouped: dict[int, dict] = {}
    for block in blocks:
        root = dsu.find(int(block["id"]))
        if root not in grouped:
            grouped[root] = {"blocks": [], "footprint": set()}
        grouped[root]["blocks"].append(block)
        grouped[root]["footprint"].update((int(r), int(c)) for r, c in block["coords"])

    for coords, touched in line_infos:
        root = dsu.find(next(iter(touched)))
        if root not in grouped:
            continue
        grouped[root]["footprint"].update((int(r), int(c)) for r, c in coords)

    groups = []
    for key in sorted(grouped):
        groups.append(grouped[key])
    return groups, line_infos


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for coords in _connected_components(grid == ORANGE):
        if int(coords.shape[0]) == 1:
            markers.append((int(coords[0, 0]), int(coords[0, 1])))
    markers.sort()
    return markers


def _distance_to_cells(point: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    pr, pc = point
    return min(abs(pr - r) + abs(pc - c) for r, c in cells)


def _assign_markers(groups: list[dict], markers: list[tuple[int, int]]) -> list[int]:
    assert groups and len(groups) == len(markers)
    n = len(groups)
    dist = [
        [_distance_to_cells(markers[j], groups[i]["footprint"]) for j in range(n)]
        for i in range(n)
    ]
    memo: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {}

    def dfs(i: int, used_mask: int) -> tuple[int, tuple[int, ...]]:
        if i == n:
            return 0, ()
        key = (i, used_mask)
        if key in memo:
            return memo[key]
        best_cost = None
        best_assign = None
        for j in range(n):
            if used_mask & (1 << j):
                continue
            tail_cost, tail_assign = dfs(i + 1, used_mask | (1 << j))
            total = dist[i][j] + tail_cost
            assign = (j,) + tail_assign
            if best_cost is None or total < best_cost or (total == best_cost and assign < best_assign):
                best_cost = total
                best_assign = assign
        assert best_cost is not None and best_assign is not None
        memo[key] = (best_cost, best_assign)
        return memo[key]

    return list(dfs(0, 0)[1])


def _draw_nested(
    out: np.ndarray,
    center: tuple[int, int],
    colors_inner_to_outer: list[int],
    inner_size: int,
) -> None:
    total_side = inner_size + 2 * (len(colors_inner_to_outer) - 1)
    half = total_side // 2
    top = center[0] - half
    left = center[1] - half
    for depth, color in enumerate(reversed(colors_inner_to_outer)):
        side = total_side - 2 * depth
        r0 = top + depth
        c0 = left + depth
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        if depth == len(colors_inner_to_outer) - 1:
            rr0 = max(0, r0)
            cc0 = max(0, c0)
            rr1 = min(out.shape[0] - 1, r1)
            cc1 = min(out.shape[1] - 1, c1)
            if rr0 <= rr1 and cc0 <= cc1:
                out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)
            continue
        for c in range(c0, c1 + 1):
            if _in_bounds(out.shape, r0, c):
                out[r0, c] = int(color)
            if _in_bounds(out.shape, r1, c):
                out[r1, c] = int(color)
        for r in range(r0 + 1, r1):
            if _in_bounds(out.shape, r, c0):
                out[r, c0] = int(color)
            if _in_bounds(out.shape, r, c1):
                out[r, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group line-linked colored blocks and redraw each group as concentric nested squares at its marker.

    Input structure:
    - The dominant background is black.
    - Several solid monochrome rectangular blocks of various sizes are scattered in the grid.
    - One-pixel-wide grey orthogonal line components touch block edges and connect the blocks into disconnected groups.
    - Each group has one isolated orange marker pixel nearby that does not touch any block or grey line.

    Rule:
    - Identify the colored rectangles, then use grey-line reachability to group rectangles that are connected through the same line network.
    - Within each group, sort the rectangles by area from small to large.
    - Use the smallest rectangle's color for the innermost filled square, with side length equal to that rectangle's shorter side.
    - Wrap that center with one-pixel-thick square frames using the remaining group colors in increasing-area order, and center the whole nested structure on the group's orange marker.
    - Remove every original rectangle, grey line, and orange marker; output only the generated nested structures.

    Key insight:
    - The grey geometry is only a grouping signal: first recover the connectivity graph of rectangles, then convert each group's area order into a centered concentric-frame drawing at its marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    groups, _line_infos = _group_blocks(g, bg)
    markers = _extract_markers(g)
    assert groups and len(groups) == len(markers)

    assignment = _assign_markers(groups, markers)
    out = np.full_like(g, int(bg))

    for group_idx, marker_idx in enumerate(assignment):
        blocks = sorted(
            groups[group_idx]["blocks"],
            key=lambda block: (
                int(block["area"]),
                int(block["short"]),
                int(block["bbox"][0]),
                int(block["bbox"][1]),
                int(block["color"]),
            ),
        )
        colors = [int(block["color"]) for block in blocks]
        inner_size = int(blocks[0]["short"])
        _draw_nested(out, markers[marker_idx], colors, inner_size)

    return out


def _adjacent_block_ids(block_grid: np.ndarray, r: int, c: int) -> set[int]:
    ids: set[int] = set()
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if _in_bounds(block_grid.shape, nr, nc) and int(block_grid[nr, nc]) >= 0:
            ids.add(int(block_grid[nr, nc]))
    return ids


def _mark_line_forbidden(line_mask: np.ndarray) -> np.ndarray:
    forbidden = line_mask.copy()
    h, w = line_mask.shape
    for r, c in np.argwhere(line_mask):
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                forbidden[nr, nc] = True
    return forbidden


def _border_adjacent_cells(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0:
            cells.add((r, c0 - 1))
        if c1 + 1 < shape[1]:
            cells.add((r, c1 + 1))
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0:
            cells.add((r0 - 1, c))
        if r1 + 1 < shape[0]:
            cells.add((r1 + 1, c))
    return sorted(cells)


def _route_line(
    canvas: np.ndarray,
    block_grid: np.ndarray,
    line_mask: np.ndarray,
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    id_a: int,
    id_b: int,
) -> list[tuple[int, int]] | None:
    shape = canvas.shape
    forbidden = _mark_line_forbidden(line_mask)

    def allowed(cell: tuple[int, int]) -> bool:
        r, c = cell
        if not _in_bounds(shape, r, c):
            return False
        if int(canvas[r, c]) != 0:
            return False
        if forbidden[r, c]:
            return False
        touched = _adjacent_block_ids(block_grid, r, c)
        return touched.issubset({id_a, id_b})

    starts = [cell for cell in _border_adjacent_cells(bbox_a, shape) if allowed(cell) and id_a in _adjacent_block_ids(block_grid, cell[0], cell[1])]
    ends = {cell for cell in _border_adjacent_cells(bbox_b, shape) if allowed(cell) and id_b in _adjacent_block_ids(block_grid, cell[0], cell[1])}
    if not starts or not ends:
        return None

    q = deque(starts)
    prev: dict[tuple[int, int], tuple[int, int] | None] = {cell: None for cell in starts}
    while q:
        cell = q.popleft()
        if cell in ends:
            path = []
            cur = cell
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        r, c = cell
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt not in prev and allowed(nxt):
                prev[nxt] = cell
                q.append(nxt)
    return None


def _can_place_rect(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occ.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= h0 or c1 >= w0:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0 - 1, r1 + pad)
    cc1 = min(w0 - 1, c1 + pad)
    return not bool(np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _all_dim_candidates(dim_cap: int) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for h in range(1, dim_cap + 1):
        for w in range(h, dim_cap + 1):
            area = h * w
            out.setdefault(area, []).append((h, w))
    return out


def _sample_group_specs(block_count: int, dim_cap: int) -> list[dict] | None:
    area_to_dims = _all_dim_candidates(dim_cap)
    all_areas = sorted(area_to_dims)
    smallest_choices = []
    for area, dims_list in area_to_dims.items():
        if sum(1 for other in all_areas if other > area) < block_count - 1:
            continue
        for h, w in dims_list:
            if h % 2 == 1:
                smallest_choices.append((area, h, w))
    if not smallest_choices:
        return None

    random.shuffle(smallest_choices)
    colors = random.sample(BLOCK_COLORS, block_count)
    for area0, h0, w0 in smallest_choices:
        larger = [area for area in all_areas if area > area0]
        if len(larger) < block_count - 1:
            continue
        random.shuffle(larger)
        chosen_areas = sorted(larger[: block_count - 1])
        specs = [{"h": h0, "w": w0, "area": area0, "short": h0}]
        ok = True
        for area in chosen_areas:
            dims = random.choice(area_to_dims[area])
            specs.append({"h": dims[0], "w": dims[1], "area": area, "short": dims[0]})
        specs.sort(key=lambda spec: spec["area"])
        for idx, spec in enumerate(specs):
            h, w = int(spec["h"]), int(spec["w"])
            if h != w and random.random() < 0.5:
                h, w = w, h
            spec["h"] = h
            spec["w"] = w
            spec["color"] = int(colors[idx])
            spec["short"] = min(h, w)
        return specs
    return None


def _choose_connection_order(blocks: list[dict]) -> list[int]:
    remaining = set(range(len(blocks)))
    current = random.randrange(len(blocks))
    order = [current]
    remaining.remove(current)
    while remaining:
        cr0, cc0, cr1, cc1 = blocks[current]["bbox"]
        center_r = (cr0 + cr1) / 2.0
        center_c = (cc0 + cc1) / 2.0
        current = min(
            remaining,
            key=lambda idx: abs(((blocks[idx]["bbox"][0] + blocks[idx]["bbox"][2]) / 2.0) - center_r)
            + abs(((blocks[idx]["bbox"][1] + blocks[idx]["bbox"][3]) / 2.0) - center_c),
        )
        order.append(current)
        remaining.remove(current)
    return order


def _generate_group_patch(block_count: int, dim_cap: int) -> dict | None:
    for _ in range(GROUP_RETRIES):
        specs = _sample_group_specs(block_count, dim_cap)
        if specs is None:
            continue
        work_h = random.randint(max(13, dim_cap + 7), 18)
        work_w = random.randint(max(13, dim_cap + 7), 18)
        canvas = np.zeros((work_h, work_w), dtype=int)
        occ = np.zeros((work_h, work_w), dtype=bool)
        block_grid = np.full((work_h, work_w), -1, dtype=int)
        blocks: list[dict] = []
        shuffled_specs = specs[:]
        random.shuffle(shuffled_specs)

        placed = True
        for block_id, spec in enumerate(shuffled_specs):
            candidates = []
            for r0 in range(1, work_h - int(spec["h"])):
                for c0 in range(1, work_w - int(spec["w"])):
                    if _can_place_rect(occ, r0, c0, int(spec["h"]), int(spec["w"]), pad=1):
                        candidates.append((r0, c0))
            if not candidates:
                placed = False
                break
            r0, c0 = random.choice(candidates)
            r1 = r0 + int(spec["h"]) - 1
            c1 = c0 + int(spec["w"]) - 1
            canvas[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
            occ[r0 : r1 + 1, c0 : c1 + 1] = True
            block_grid[r0 : r1 + 1, c0 : c1 + 1] = block_id
            blocks.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": int(spec["color"]),
                    "area": int(spec["area"]),
                    "short": int(spec["short"]),
                }
            )
        if not placed:
            continue

        line_mask = np.zeros((work_h, work_w), dtype=bool)
        order = _choose_connection_order(blocks)
        ok = True
        for a_idx, b_idx in zip(order, order[1:]):
            path = _route_line(
                canvas,
                block_grid,
                line_mask,
                blocks[a_idx]["bbox"],
                blocks[b_idx]["bbox"],
                a_idx,
                b_idx,
            )
            if path is None:
                ok = False
                break
            for r, c in path:
                canvas[r, c] = GREY
                line_mask[r, c] = True
        if not ok:
            continue

        used_cells = {(int(r), int(c)) for r, c in np.argwhere(canvas != 0)}
        marker_candidates = []
        for r in range(work_h):
            for c in range(work_w):
                if canvas[r, c] != 0:
                    continue
                isolated = True
                for dr, dc in DIR8:
                    nr, nc = r + dr, c + dc
                    if _in_bounds(canvas.shape, nr, nc) and int(canvas[nr, nc]) != 0:
                        isolated = False
                        break
                if not isolated:
                    continue
                dist = min(abs(r - rr) + abs(c - cc) for rr, cc in used_cells)
                if 2 <= dist <= 5:
                    marker_candidates.append((r, c))
        if not marker_candidates:
            continue
        marker = random.choice(marker_candidates)
        canvas[marker[0], marker[1]] = ORANGE

        used = np.argwhere(canvas != 0)
        r0, c0, r1, c1 = _bbox(used)
        r0 = max(0, r0 - 1)
        c0 = max(0, c0 - 1)
        r1 = min(work_h - 1, r1 + 1)
        c1 = min(work_w - 1, c1 + 1)
        patch = canvas[r0 : r1 + 1, c0 : c1 + 1].copy()
        marker_local = (marker[0] - r0, marker[1] - c0)
        sorted_specs = sorted(specs, key=lambda spec: int(spec["area"]))
        colors_inner_to_outer = [int(spec["color"]) for spec in sorted_specs]
        inner_size = int(sorted_specs[0]["short"])
        return {
            "patch": patch,
            "marker": marker_local,
            "colors": colors_inner_to_outer,
            "inner_size": inner_size,
            "side": inner_size + 2 * (len(colors_inner_to_outer) - 1),
        }
    return None


def _box_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _cells_have_gap(occupied: np.ndarray, cells: np.ndarray, gap: int = 1) -> bool:
    for r, c in cells:
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                nr, nc = int(r + dr), int(c + dc)
                if _in_bounds(occupied.shape, nr, nc) and occupied[nr, nc]:
                    return False
    return True


def _render_expected(shape: tuple[int, int], placed_groups: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for group in placed_groups:
        _draw_nested(out, group["global_marker"], group["colors"], group["inner_size"])
    return out


def _generate_attempt() -> np.ndarray | None:
    group_count = random.choice([1, 2, 3])
    dim_cap = 7 if group_count == 1 else 6 if group_count == 2 else 5
    block_choices = [2, 3, 4] if group_count < 3 else [2, 3]
    group_patches = []
    for _ in range(group_count):
        patch = _generate_group_patch(random.choice(block_choices), dim_cap)
        if patch is None:
            return None
        group_patches.append(patch)

    min_dim = max(
        18,
        max(int(group["patch"].shape[0]) for group in group_patches) + 2,
        max(int(group["patch"].shape[1]) for group in group_patches) + 2,
        max(int(group["side"]) for group in group_patches) + 2,
    )
    if min_dim > 30:
        return None
    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    output_boxes: list[tuple[int, int, int, int]] = []
    placed_groups: list[dict] = []

    order = sorted(range(group_count), key=lambda idx: (group_patches[idx]["side"], group_patches[idx]["patch"].size), reverse=True)
    for idx in order:
        group = group_patches[idx]
        patch = group["patch"]
        ph, pw = patch.shape
        mr, mc = group["marker"]
        side = int(group["side"])
        half = side // 2
        cells = np.argwhere(patch != 0)
        candidates = []
        for r0 in range(0, h - ph + 1):
            for c0 in range(0, w - pw + 1):
                gr = r0 + int(mr)
                gc = c0 + int(mc)
                out_box = (gr - half, gc - half, gr + half, gc + half)
                if out_box[0] < 0 or out_box[1] < 0 or out_box[2] >= h or out_box[3] >= w:
                    continue
                if any(_box_overlaps(out_box, other, gap=1) for other in output_boxes):
                    continue
                global_cells = cells + np.array([r0, c0], dtype=int)
                if not _cells_have_gap(occupied, global_cells, gap=1):
                    continue
                intrudes = False
                for other_box in output_boxes:
                    for rr, cc in global_cells:
                        if other_box[0] - 1 <= int(rr) <= other_box[2] + 1 and other_box[1] - 1 <= int(cc) <= other_box[3] + 1:
                            intrudes = True
                            break
                    if intrudes:
                        break
                if intrudes:
                    continue
                candidates.append((r0, c0, out_box))
        if not candidates:
            return None
        r0, c0, out_box = random.choice(candidates)
        global_cells = cells + np.array([r0, c0], dtype=int)
        for rr, cc in global_cells:
            grid[int(rr), int(cc)] = int(patch[int(rr - r0), int(cc - c0)])
            occupied[int(rr), int(cc)] = True
        output_boxes.append(out_box)
        placed_groups.append(
            {
                "global_marker": (r0 + int(mr), c0 + int(mc)),
                "colors": list(group["colors"]),
                "inner_size": int(group["inner_size"]),
            }
        )

    expected = _render_expected(grid.shape, placed_groups)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
