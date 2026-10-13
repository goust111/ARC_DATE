import random
from collections import Counter

import numpy as np
from scipy.ndimage import label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    struct = STRUCT4 if int(connectivity) == 4 else np.ones((3, 3), dtype=int)
    lab, n = label(mask.astype(bool), structure=struct)
    return lab, int(n)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    lab, n = label_mask(mask, connectivity=connectivity)
    return [np.argwhere(lab == k) for k in range(1, n + 1)]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        (coords[:, 0] == 0).any()
        or (coords[:, 0] == h - 1).any()
        or (coords[:, 1] == 0).any()
        or (coords[:, 1] == w - 1).any()
    )


def _infer_marker_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        comps = components_mask(grid == int(color), connectivity=4)
        if int(np.count_nonzero(grid == int(color))) > 1 and comps and all(len(comp) == 1 for comp in comps):
            candidates.append(int(color))
    assert len(candidates) == 1, candidates
    return int(candidates[0])


def _detect_roles(grid: np.ndarray):
    bg = get_bg(grid)
    marker = _infer_marker_color(grid, bg)

    objects = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg), int(marker)}):
        for coords in components_mask(grid == int(color), connectivity=4):
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )

    assert objects
    wall = max(objects, key=lambda obj: obj["area"])
    wall_coords = {tuple(map(int, cell)) for cell in wall["coords"]}
    bar_objects = [obj for obj in objects if obj is not wall]
    assert bar_objects

    bar_mask = np.zeros_like(grid, dtype=bool)
    height_to_color: dict[int, int] = {}
    for obj in bar_objects:
        r0, _c0, r1, _c1 = obj["bbox"]
        height = int(r1 - r0 + 1)
        assert height not in height_to_color
        height_to_color[height] = int(obj["color"])
        for r, c in obj["coords"]:
            bar_mask[int(r), int(c)] = True

    wall_mask = np.zeros_like(grid, dtype=bool)
    for r, c in wall_coords:
        wall_mask[r, c] = True

    return bg, marker, int(wall["color"]), wall_mask, bar_mask, height_to_color


def _door_mask(grid: np.ndarray, marker: int, wall_mask: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g, dtype=bool)
    for r0, c0 in np.argwhere(g == int(marker)):
        r = int(r0)
        c = int(c0)
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        horizontal_wall = bool(wall_mask[r, c - 1] and wall_mask[r, c + 1] and not wall_mask[r - 1, c] and not wall_mask[r + 1, c])
        vertical_wall = bool(wall_mask[r - 1, c] and wall_mask[r + 1, c] and not wall_mask[r, c - 1] and not wall_mask[r, c + 1])
        if horizontal_wall ^ vertical_wall:
            out[r, c] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the grey door-plugs in contour walls, merge the true suites, and recolor each suite from the bar whose height matches its chip count.

    Input structure:
    - One contour-color wall complex encloses multiple background chambers on a larger background.
    - Grey single pixels have two roles: some are free-standing chips inside chambers, while others sit in one-cell gaps on interior wall segments as door-plugs.
    - Separate single-color vertical bars stand on one side of the grid with distinct colors and distinct heights.

    Rule:
    - Identify a grey pixel as a door when it is embedded in a wall gap: the two neighbors along the wall are wall-colored and the two perpendicular neighbors are not walls.
    - Treat doors as open passage cells, then split all non-wall, non-bar cells into 4-connected components; the enclosed components are the suites.
    - For each suite, count only the grey pixels that are not doors, find the bar whose height equals that count, and fill the whole suite background (including door cells and chip cells) with that bar color.

    Key insight:
    - The ambiguous grey pixels are disambiguated by topology, not color: a wall-embedded grey cell changes connectivity and must be removed from the count, while a free room pixel remains a counted chip.
    """
    g = np.asarray(grid, dtype=int)
    bg, marker, _wall_color, wall_mask, bar_mask, height_to_color = _detect_roles(g)
    doors = _door_mask(g, marker, wall_mask)

    passable = ~wall_mask & ~bar_mask
    lab, n = label_mask(passable, connectivity=4)
    out = g.copy()

    for comp_id in range(1, n + 1):
        coords = np.argwhere(lab == comp_id)
        if coords.size == 0 or _touches_border(coords, g.shape):
            continue
        mask = lab == comp_id
        count = int(np.count_nonzero(mask & (g == int(marker)) & ~doors))
        assert count in height_to_color, (count, sorted(height_to_color))
        out[mask] = int(height_to_color[count])

    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _assign_distinct_counts(suite_infos: list[dict]) -> list[int] | None:
    order = sorted(range(len(suite_infos)), key=lambda i: (suite_infos[i]["hi"], suite_infos[i]["lo"]))
    chosen = [0] * len(suite_infos)
    used: set[int] = set()

    def rec(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        info = suite_infos[idx]
        options = list(range(int(info["lo"]), int(info["hi"]) + 1))
        random.shuffle(options)
        options.sort(key=lambda v: (abs(v - info["target"]), random.random()))
        for value in options:
            if value in used:
                continue
            used.add(value)
            chosen[idx] = int(value)
            if rec(k + 1):
                return True
            used.remove(value)
        return False

    return chosen if rec(0) else None


def _suite_roots(chamber_count: int, open_edges: list[tuple[int, int, str]]) -> dict[int, list[int]]:
    uf = _UnionFind(chamber_count)
    for a, b, _orient in open_edges:
        uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for chamber in range(chamber_count):
        groups.setdefault(uf.find(chamber), []).append(chamber)
    return groups


def _choose_open_edges(nr: int, nc: int, suite_count: int) -> list[tuple[int, int, str]] | None:
    def cid(r: int, c: int) -> int:
        return r * nc + c

    edges: list[tuple[int, int, str]] = []
    for r in range(nr):
        for c in range(nc - 1):
            edges.append((cid(r, c), cid(r, c + 1), "vertical"))
    for r in range(nr - 1):
        for c in range(nc):
            edges.append((cid(r, c), cid(r + 1, c), "horizontal"))

    vertical_edges = [edge for edge in edges if edge[2] == "vertical"]
    horizontal_edges = [edge for edge in edges if edge[2] == "horizontal"]
    open_edge_count = nr * nc - suite_count
    if open_edge_count < 2:
        return None

    uf = _UnionFind(nr * nc)
    selected: list[tuple[int, int, str]] = []
    required = [random.choice(vertical_edges), random.choice(horizontal_edges)]
    random.shuffle(required)
    for edge in required:
        a, b, _ = edge
        if uf.union(a, b):
            selected.append(edge)

    remaining = [edge for edge in edges if edge not in selected]
    random.shuffle(remaining)
    for edge in remaining:
        if len(selected) == open_edge_count:
            break
        a, b, _ = edge
        if uf.union(a, b):
            selected.append(edge)

    return selected if len(selected) == open_edge_count else None


def _chamber_starts(lengths: list[int]) -> list[int]:
    starts = []
    cur = 1
    for size in lengths:
        starts.append(cur)
        cur += size + 1
    return starts


def _build_patch(
    row_heights: list[int],
    col_widths: list[int],
    wall: int,
    bg: int,
    marker: int,
    open_edges: list[tuple[int, int, str]],
):
    nr = len(row_heights)
    nc = len(col_widths)
    patch_h = sum(row_heights) + nr + 1
    patch_w = sum(col_widths) + nc + 1
    patch = np.full((patch_h, patch_w), int(wall), dtype=int)

    row_starts = _chamber_starts(row_heights)
    col_starts = _chamber_starts(col_widths)
    chamber_boxes: dict[int, tuple[int, int, int, int]] = {}

    for r in range(nr):
        for c in range(nc):
            cid = r * nc + c
            r0 = row_starts[r]
            c0 = col_starts[c]
            h = row_heights[r]
            w = col_widths[c]
            patch[r0 : r0 + h, c0 : c0 + w] = int(bg)
            chamber_boxes[cid] = (r0, c0, r0 + h - 1, c0 + w - 1)

    door_cells: list[tuple[int, int]] = []
    for a, b, orient in open_edges:
        ra, ca = divmod(a, nc)
        rb, cb = divmod(b, nc)
        if orient == "vertical":
            row = ra
            left = min(ca, cb)
            wall_c = col_starts[left] + col_widths[left]
            candidates = list(range(row_starts[row] + 1, row_starts[row] + row_heights[row] - 1))
            if not candidates:
                return None
            door_r = random.choice(candidates)
            patch[door_r, wall_c] = int(marker)
            door_cells.append((door_r, wall_c))
        else:
            col = ca
            top = min(ra, rb)
            wall_r = row_starts[top] + row_heights[top]
            candidates = list(range(col_starts[col] + 1, col_starts[col] + col_widths[col] - 1))
            if not candidates:
                return None
            door_c = random.choice(candidates)
            patch[wall_r, door_c] = int(marker)
            door_cells.append((wall_r, door_c))

    return patch, chamber_boxes, door_cells


def _safe_cells_for_chamber(
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    bg: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    candidates = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(patch[r, c]) != int(bg):
                continue
            if all(in_bounds(patch.shape, rr, cc) and int(patch[rr, cc]) == int(bg) for rr, cc in neighbors4(r, c)):
                candidates.append((r, c))

    parity0 = [cell for cell in candidates if (cell[0] + cell[1]) % 2 == 0]
    parity1 = [cell for cell in candidates if (cell[0] + cell[1]) % 2 == 1]
    if len(parity0) == len(parity1):
        return random.choice([parity0, parity1])
    return parity0 if len(parity0) > len(parity1) else parity1


def _sample_suite_cells(
    chamber_cells: dict[int, list[tuple[int, int]]],
    chambers: list[int],
    count: int,
) -> list[tuple[int, int]] | None:
    pools = {ch: chamber_cells[ch][:] for ch in chambers}
    for cells in pools.values():
        random.shuffle(cells)

    picks: list[tuple[int, int]] = []
    if count > 1:
        populated = [ch for ch in chambers if pools[ch]]
        if len(populated) >= 2:
            chosen_chambers = random.sample(populated, 2)
            for ch in chosen_chambers:
                picks.append(pools[ch].pop())

    remaining = count - len(picks)
    all_cells = [(ch, cell) for ch in chambers for cell in pools[ch]]
    if len(all_cells) < remaining:
        return None
    random.shuffle(all_cells)
    picks.extend(cell for _ch, cell in all_cells[:remaining])
    return picks


def _non_monotone(order: list[int]) -> bool:
    return order != sorted(order) and order != sorted(order, reverse=True)


def _gen_attempt() -> np.ndarray | None:
    marker = 5
    nr, nc = random.choice([(2, 2), (2, 3), (3, 2)])
    chamber_count = nr * nc
    suite_count = random.randint(2, min(3, chamber_count - 2))

    row_heights = [random.randint(4, 6 if nr == 2 else 5) for _ in range(nr)]
    col_widths = [random.randint(4, 6 if nc == 2 else 5) for _ in range(nc)]

    open_edges = _choose_open_edges(nr, nc, suite_count)
    if open_edges is None:
        return None

    palette = list(range(10))
    palette.remove(marker)
    bg = random.choice(palette)
    palette.remove(bg)
    wall = random.choice(palette)
    palette.remove(wall)

    built = _build_patch(row_heights, col_widths, wall, bg, marker, open_edges)
    if built is None:
        return None
    patch, chamber_boxes, _door_cells = built

    suites = _suite_roots(chamber_count, open_edges)
    chamber_cells: dict[int, list[tuple[int, int]]] = {}
    suite_infos: list[dict] = []
    multi_suite_indices = []

    for chamber, box in chamber_boxes.items():
        chamber_cells[chamber] = _safe_cells_for_chamber(patch, box, bg)
        if not chamber_cells[chamber]:
            return None

    for root, chambers in suites.items():
        capacities = [len(chamber_cells[ch]) for ch in chambers]
        total_cap = int(sum(capacities))
        single_cap = int(max(capacities))
        if total_cap < 1:
            return None
        info = {
            "root": root,
            "chambers": chambers,
            "lo": 1,
            "hi": total_cap,
            "target": (1 + total_cap) / 2,
        }
        if len(chambers) > 1 and total_cap > single_cap:
            multi_suite_indices.append(len(suite_infos))
            info["lo"] = single_cap + 1
            info["target"] = info["lo"]
        suite_infos.append(info)

    if not multi_suite_indices:
        return None

    preferred = random.choice(multi_suite_indices)
    for i, info in enumerate(suite_infos):
        if i != preferred and len(info["chambers"]) > 1:
            capacities = [len(chamber_cells[ch]) for ch in info["chambers"]]
            info["lo"] = 1
            info["target"] = min(info["hi"], max(1, max(capacities)))

    counts = _assign_distinct_counts(suite_infos)
    if counts is None:
        return None

    used_heights = list(counts)
    extra_options = [h for h in range(1, max(used_heights) + 4) if h not in used_heights]
    if not extra_options or not palette:
        return None
    extra_height = random.choice(extra_options)
    all_heights = used_heights + [extra_height]
    random.shuffle(all_heights)
    if not _non_monotone(all_heights):
        return None

    if len(all_heights) > len(palette):
        return None
    bar_colors = random.sample(palette, len(all_heights))

    for info, count in zip(suite_infos, counts):
        picks = _sample_suite_cells(chamber_cells, info["chambers"], count)
        if picks is None or len(picks) != count:
            return None
        for r, c in picks:
            patch[r, c] = int(marker)

    patch_h, patch_w = patch.shape
    bar_region_w = 2 * len(all_heights) - 1
    gap = random.randint(1, 2)
    grid_h = max(patch_h, max(all_heights))
    grid_w = patch_w + gap + bar_region_w
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    patch_r0 = random.randint(0, grid_h - patch_h)
    side = random.choice(["left", "right"])
    if side == "left":
        bar_c0 = 0
        patch_c0 = bar_region_w + gap
    else:
        patch_c0 = 0
        bar_c0 = patch_w + gap

    grid[patch_r0 : patch_r0 + patch_h, patch_c0 : patch_c0 + patch_w] = patch

    for idx, (height, color) in enumerate(zip(all_heights, bar_colors)):
        c = bar_c0 + 2 * idx
        r0 = grid_h - height
        grid[r0:grid_h, c] = int(color)

    if np.count_nonzero(grid == int(marker)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
