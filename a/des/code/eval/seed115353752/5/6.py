from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
CONTAINER_COLOR = 7  # orange
DIR_TO_ROT = {
    (0, 1): 0,   # right
    (1, 0): 1,   # down
    (0, -1): 2,  # left
    (-1, 0): 3,  # up
}
NON_SPECIAL_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_pattern(block: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(block != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(block[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    quarter_turns %= 4
    if quarter_turns == 0:
        return np.array(arr, copy=True)
    return np.rot90(arr, k=(-quarter_turns) % 4)


def _find_containers(grid: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    h, w = grid.shape
    mask = grid == CONTAINER_COLOR
    comps = _components(mask)
    containers: list[dict] = []
    border_owner = np.full((h, w), -1, dtype=int)
    interior_mask = np.zeros((h, w), dtype=bool)

    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue
        if len(pts) != 2 * height + 2 * width - 4:
            continue
        rect = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(rect[0, :] == CONTAINER_COLOR):
            continue
        if not np.all(rect[-1, :] == CONTAINER_COLOR):
            continue
        if not np.all(rect[:, 0] == CONTAINER_COLOR):
            continue
        if not np.all(rect[:, -1] == CONTAINER_COLOR):
            continue
        if np.any(rect[1:-1, 1:-1] == CONTAINER_COLOR):
            continue

        container_id = len(containers)
        for r, c in pts:
            border_owner[r, c] = container_id
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True
        containers.append(
            {
                "id": container_id,
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
            }
        )

    assert containers, "expected at least one orange container"
    return containers, border_owner, interior_mask


def _source_info(grid: np.ndarray, containers: list[dict], bg: int) -> tuple[int, np.ndarray]:
    source_id = None
    source_pattern = None
    for container in containers:
        r0, c0, r1, c1 = container["inner_bbox"]
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block != bg):
            assert source_id is None, "expected exactly one source container"
            source_id = int(container["id"])
            source_pattern = _crop_pattern(block, bg)
    assert source_id is not None and source_pattern is not None
    return source_id, source_pattern


def _parse_arrow_direction(local: np.ndarray) -> tuple[int, int] | None:
    height, width = local.shape
    if height == 2 and width >= 3:
        row_counts = [int(local[r].sum()) for r in range(height)]
        if sorted(row_counts) != [2, width]:
            return None
        short_row = 0 if row_counts[0] == 2 else 1
        short_cols = np.where(local[short_row])[0].tolist()
        if short_cols == [0, 1]:
            return (0, -1)
        if short_cols == [width - 2, width - 1]:
            return (0, 1)
        return None
    if width == 2 and height >= 3:
        col_counts = [int(local[:, c].sum()) for c in range(width)]
        if sorted(col_counts) != [2, height]:
            return None
        short_col = 0 if col_counts[0] == 2 else 1
        short_rows = np.where(local[:, short_col])[0].tolist()
        if short_rows == [0, 1]:
            return (-1, 0)
        if short_rows == [height - 2, height - 1]:
            return (1, 0)
        return None
    return None


def _find_arrows(
    grid: np.ndarray,
    border_owner: np.ndarray,
    interior_mask: np.ndarray,
    containers: list[dict],
    bg: int,
) -> list[tuple[int, int, tuple[int, int]]]:
    outside_mask = (grid != bg) & (grid != CONTAINER_COLOR) & (~interior_mask)
    arrows: list[tuple[int, int, tuple[int, int]]] = []
    for pts in _components(outside_mask):
        values = {int(grid[r, c]) for r, c in pts}
        if len(values) != 1:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local[r - r0, c - c0] = True
        direction = _parse_arrow_direction(local)
        if direction is None:
            continue

        adjacent = set()
        for r, c in pts:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    owner = int(border_owner[nr, nc])
                    if owner >= 0:
                        adjacent.add(owner)
        assert len(adjacent) == 2, "each arrow must connect exactly two containers"

        dr, dc = direction
        ordered = sorted(
            adjacent,
            key=lambda cid: containers[cid]["center"][0] * dr + containers[cid]["center"][1] * dc,
        )
        src, dst = ordered[0], ordered[1]
        arrows.append((int(src), int(dst), direction))

    assert arrows, "expected at least one arrow"
    return arrows


def _path_contributions(
    source_id: int, edges: list[tuple[int, int, tuple[int, int]]], n_nodes: int
) -> dict[int, list[tuple[int, int]]]:
    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for src, dst, direction in edges:
        graph[int(src)].append((int(dst), DIR_TO_ROT[direction]))

    contributions: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def dfs(node: int, rot: int, dist: int, seen: set[int]) -> None:
        for dst, edge_rot in graph.get(node, []):
            if dst in seen:
                continue
            next_rot = (rot + edge_rot) % 4
            next_dist = dist + 1
            contributions[dst].append((next_dist, next_rot))
            dfs(dst, next_rot, next_dist, seen | {dst})

    dfs(int(source_id), 0, 0, {int(source_id)})
    for node in range(n_nodes):
        contributions.setdefault(node, [])
    return contributions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the colored arrow transfer chain between orange containers.

    Each transfer copies the current source pattern into the destination container,
    rotating by the arrow direction and aligning the copied pattern to the
    destination interior's top-left corner. If multiple paths reach a container,
    shorter chains write first and therefore win on overlaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    containers, border_owner, interior_mask = _find_containers(g)
    source_id, source_pattern = _source_info(g, containers, bg)
    arrows = _find_arrows(g, border_owner, interior_mask, containers, bg)
    contributions = _path_contributions(source_id, arrows, len(containers))

    out = np.array(g, copy=True)
    for container in containers:
        cid = int(container["id"])
        if cid == source_id:
            continue
        r0, c0, r1, c1 = container["inner_bbox"]
        inner_h = r1 - r0 + 1
        inner_w = c1 - c0 + 1

        for dist, rot in sorted(contributions[cid], key=lambda item: (item[0], item[1])):
            del dist
            rotated = _rotate_cw(source_pattern, rot)
            ph, pw = rotated.shape
            assert ph <= inner_h and pw <= inner_w, "rotated pattern must fit container interior"
            for rr in range(ph):
                for cc in range(pw):
                    value = int(rotated[rr, cc])
                    if value == bg:
                        continue
                    if int(out[r0 + rr, c0 + cc]) == bg:
                        out[r0 + rr, c0 + cc] = value

    return out


TEMPLATES = [
    {
        "name": "chain4",
        "slots": [(0, 0), (0, 1), (1, 1), (1, 2)],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
    {
        "name": "star5",
        "slots": [(1, 1), (0, 1), (1, 2), (2, 1), (1, 0)],
        "edges": [(0, 1), (0, 2), (0, 3), (0, 4)],
    },
    {
        "name": "merge5",
        "slots": [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
    },
    {
        "name": "merge6",
        "slots": [(0, 1), (1, 1), (0, 2), (1, 2), (2, 1), (2, 2)],
        "edges": [(0, 1), (1, 4), (0, 2), (2, 3), (3, 5), (5, 4)],
    },
]

TRANSFORMS = [
    lambda r, c: (r, c),
    lambda r, c: (c, 2 - r),
    lambda r, c: (2 - r, 2 - c),
    lambda r, c: (2 - c, r),
    lambda r, c: (r, 2 - c),
    lambda r, c: (2 - r, c),
    lambda r, c: (c, r),
    lambda r, c: (2 - c, 2 - r),
]

MAX_GENERATION_ATTEMPTS = 400


def _connected_mask(height: int, width: int, pixels: int) -> np.ndarray | None:
    total = height * width
    if pixels > total:
        return None
    mask = np.zeros((height, width), dtype=bool)
    start_r = random.randrange(height)
    start_c = random.randrange(width)
    mask[start_r, start_c] = True
    while int(mask.sum()) < pixels:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return mask


def _make_pattern(inner_h: int, inner_w: int) -> np.ndarray | None:
    max_dim = min(inner_h, inner_w)
    if max_dim < 2:
        return None
    pat_h = random.randint(2, max_dim)
    pat_w = random.randint(2, max_dim)
    pixels = random.randint(2, min(6, pat_h * pat_w))
    mask = _connected_mask(pat_h, pat_w, pixels)
    if mask is None:
        return None

    crop = _crop_pattern(mask.astype(int), 0).astype(bool)
    colors_available = NON_SPECIAL_COLORS.copy()
    color_count = random.randint(1, min(2, len(colors_available)))
    colors = random.sample(colors_available, color_count)
    pattern = np.full(crop.shape, BG, dtype=int)
    coords = np.argwhere(crop)
    for rr, cc in coords:
        pattern[int(rr), int(cc)] = random.choice(colors)
    if color_count > 1:
        used = {int(pattern[int(rr), int(cc)]) for rr, cc in coords}
        if len(used) < 2:
            rr, cc = coords[0]
            pattern[int(rr), int(cc)] = colors[0]
            rr, cc = coords[-1]
            pattern[int(rr), int(cc)] = colors[1]
    return pattern


def _apply_transform(template: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    transform = random.choice(TRANSFORMS)
    slots = [transform(r, c) for r, c in template["slots"]]
    edges = list(template["edges"])
    return slots, edges


def _slot_positions(outer_h: int, outer_w: int) -> tuple[list[int], list[int], int, int]:
    top_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    row_gaps = [random.randint(3, 5), random.randint(3, 5)]
    col_gaps = [random.randint(3, 5), random.randint(3, 5)]

    rows = [
        top_margin,
        top_margin + outer_h + row_gaps[0],
        top_margin + 2 * outer_h + row_gaps[0] + row_gaps[1],
    ]
    cols = [
        left_margin,
        left_margin + outer_w + col_gaps[0],
        left_margin + 2 * outer_w + col_gaps[0] + col_gaps[1],
    ]
    height = rows[-1] + outer_h + bottom_margin
    width = cols[-1] + outer_w + right_margin
    return rows, cols, height, width


def _draw_container(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int) -> None:
    grid[top, left : left + outer_w] = CONTAINER_COLOR
    grid[top + outer_h - 1, left : left + outer_w] = CONTAINER_COLOR
    grid[top : top + outer_h, left] = CONTAINER_COLOR
    grid[top : top + outer_h, left + outer_w - 1] = CONTAINER_COLOR


def _arrow_mask(direction: tuple[int, int], gap: int) -> np.ndarray:
    if direction == (0, 1):
        mask = np.zeros((2, gap), dtype=bool)
        mask[1, :] = True
        mask[0, -2:] = True
        return mask
    if direction == (0, -1):
        mask = np.zeros((2, gap), dtype=bool)
        mask[1, :] = True
        mask[0, :2] = True
        return mask
    if direction == (1, 0):
        mask = np.zeros((gap, 2), dtype=bool)
        mask[:, 0] = True
        mask[-2:, 1] = True
        return mask
    mask = np.zeros((gap, 2), dtype=bool)
    mask[:, 1] = True
    mask[:2, 0] = True
    return mask


def _place_arrow(
    grid: np.ndarray,
    src_box: tuple[int, int],
    dst_box: tuple[int, int],
    outer_h: int,
    outer_w: int,
    color: int,
) -> tuple[int, int]:
    sr, sc = src_box
    dr, dc = dst_box
    step = (int(np.sign(dr - sr)), int(np.sign(dc - sc)))
    direction = step
    assert direction in DIR_TO_ROT

    if direction == (0, 1):
        gap = dc - (sc + outer_w)
        ar = random.randint(sr + 1, sr + outer_h - 3)
        ac = sc + outer_w
    elif direction == (0, -1):
        gap = sc - (dc + outer_w)
        ar = random.randint(sr + 1, sr + outer_h - 3)
        ac = dc + outer_w
    elif direction == (1, 0):
        gap = dr - (sr + outer_h)
        ar = sr + outer_h
        ac = random.randint(sc + 1, sc + outer_w - 3)
    else:
        gap = sr - (dr + outer_h)
        ar = dr + outer_h
        ac = random.randint(sc + 1, sc + outer_w - 3)

    assert 3 <= gap <= 5
    mask = _arrow_mask(direction, gap)
    for rr, cc in np.argwhere(mask):
        grid[ar + int(rr), ac + int(cc)] = color
    return direction


def _paths_and_rotations(template_edges: list[tuple[int, int]], slots: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    edges: list[tuple[int, int, tuple[int, int]]] = []
    for src, dst in template_edges:
        sr, sc = slots[src]
        dr, dc = slots[dst]
        direction = (int(np.sign(dr - sr)), int(np.sign(dc - sc)))
        edges.append((src, dst, direction))
    return _path_contributions(0, edges, len(slots))


def _pattern_has_rotation_signal(pattern: np.ndarray, contributions: dict[int, list[tuple[int, int]]]) -> bool:
    rots = {rot for items in contributions.values() for _dist, rot in items}
    rots.discard(0)
    if not rots:
        return True
    return any(not np.array_equal(pattern, _rotate_cw(pattern, rot)) for rot in rots)


def _attempt_generate() -> np.ndarray | None:
    template = random.choice(TEMPLATES)
    slots, template_edges = _apply_transform(template)

    inner_h = random.randint(4, 6)
    inner_w = random.randint(4, 6)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    rows, cols, height, width = _slot_positions(outer_h, outer_w)

    grid = np.full((height, width), BG, dtype=int)
    boxes: list[tuple[int, int]] = []
    for slot_r, slot_c in slots:
        top = rows[slot_r]
        left = cols[slot_c]
        boxes.append((top, left))
        _draw_container(grid, top, left, outer_h, outer_w)

    needed_colors = len(template_edges) + 2
    if needed_colors > len(NON_SPECIAL_COLORS):
        return None
    sampled = random.sample(NON_SPECIAL_COLORS, needed_colors)
    arrow_colors = sampled[: len(template_edges)]
    pattern_palette = sampled[len(template_edges) :]

    pattern = _make_pattern(inner_h, inner_w)
    if pattern is None:
        return None
    pattern_values = sorted(set(int(v) for v in np.unique(pattern)) - {BG})
    remap = {}
    for value in pattern_values:
        remap[value] = random.choice(pattern_palette)
    for value, replacement in remap.items():
        pattern[pattern == value] = replacement

    contributions = _paths_and_rotations(template_edges, slots)
    if not _pattern_has_rotation_signal(pattern, contributions):
        return None

    src_top, src_left = boxes[0]
    ph, pw = pattern.shape
    off_r = random.randint(0, inner_h - ph)
    off_c = random.randint(0, inner_w - pw)
    grid[src_top + 1 + off_r : src_top + 1 + off_r + ph, src_left + 1 + off_c : src_left + 1 + off_c + pw] = np.where(
        pattern != BG,
        pattern,
        grid[src_top + 1 + off_r : src_top + 1 + off_r + ph, src_left + 1 + off_c : src_left + 1 + off_c + pw],
    )

    for color, (src, dst) in zip(arrow_colors, template_edges, strict=False):
        _place_arrow(grid, boxes[src], boxes[dst], outer_h, outer_w, color)

    if _most_common_color(grid) != BG:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    containers, _border_owner, _interior_mask = _find_containers(grid)
    filled_targets = 0
    for container in containers:
        cid = int(container["id"])
        if cid == 0:
            continue
        r0, c0, r1, c1 = container["inner_bbox"]
        if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BG):
            filled_targets += 1
    if filled_targets < len(containers) - 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
