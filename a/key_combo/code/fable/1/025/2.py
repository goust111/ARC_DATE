import random
from collections import Counter, defaultdict, deque

import numpy as np


WALL_COLOR = 2
MAX_GENERATION_ATTEMPTS = 400

MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 24
MIN_REGION_HEIGHT = 4
MIN_REGION_WIDTH = 4
MIN_SPLITS = 3
MAX_SPLITS = 6

BACKGROUND_CHOICES = [0, 1, 3, 4, 5, 6, 7, 8, 9]
SPRITE_PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]

BASE_SHAPES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    current = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            current += 1
            labels[r, c] = current
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        q.append((nr, nc))
    return labels, current


def _apply_transform(grid: np.ndarray, code: int) -> np.ndarray:
    arr = np.asarray(grid)
    rot = int(code) % 4
    flip = int(code) // 4
    out = np.rot90(arr, rot)
    if flip:
        out = np.fliplr(out)
    return out.copy()


def _canon_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _wall_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((nr, nc))
    return out


def _is_vertex(mask: np.ndarray, r: int, c: int) -> bool:
    nbrs = _wall_neighbors(mask, r, c)
    if len(nbrs) != 2:
        return True
    vertical = sum(1 for nr, nc in nbrs if nr != r)
    horizontal = sum(1 for nr, nc in nbrs if nc != c)
    return vertical == 1 and horizontal == 1


def _extract_wall_segments(mask: np.ndarray) -> list[tuple[str, list[tuple[int, int]]]]:
    h, w = mask.shape
    vertices = [(r, c) for r in range(h) for c in range(w) if mask[r, c] and _is_vertex(mask, r, c)]
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[tuple[str, list[tuple[int, int]]]] = []

    for start in vertices:
        sr, sc = start
        for nxt in _wall_neighbors(mask, sr, sc):
            edge = _canon_edge(start, nxt)
            if edge in visited_edges:
                continue
            orientation = "v" if nxt[0] != sr else "h"
            path = [start]
            prev = start
            cur = nxt
            while True:
                visited_edges.add(_canon_edge(prev, cur))
                path.append(cur)
                if cur != start and _is_vertex(mask, cur[0], cur[1]):
                    break

                if orientation == "v":
                    candidates = []
                    for rr in (cur[0] - 1, cur[0] + 1):
                        if 0 <= rr < h and mask[rr, cur[1]] and (rr, cur[1]) != prev:
                            candidates.append((rr, cur[1]))
                else:
                    candidates = []
                    for cc in (cur[1] - 1, cur[1] + 1):
                        if 0 <= cc < w and mask[cur[0], cc] and (cur[0], cc) != prev:
                            candidates.append((cur[0], cc))

                if len(candidates) != 1:
                    break
                prev, cur = cur, candidates[0]
            segments.append((orientation, path))

    return segments


def _analyze_regions(grid: np.ndarray) -> tuple[np.ndarray, dict[int, set[int]], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert np.any(g == WALL_COLOR), "expected visible red walls"

    region_labels, n_regions = label_mask(g != WALL_COLOR)
    assert n_regions >= 2, "walls must split the board into multiple non-wall regions"

    start_region = int(region_labels[h - 1, 0])
    assert start_region > 0, "the lower-left cell must lie in a non-wall region"

    graph: dict[int, set[int]] = {rid: set() for rid in range(1, n_regions + 1)}
    for orientation, path in _extract_wall_segments(g == WALL_COLOR):
        side_a: set[int] = set()
        side_b: set[int] = set()
        for r, c in path:
            if orientation == "h":
                if r - 1 >= 0 and region_labels[r - 1, c] > 0:
                    side_a.add(int(region_labels[r - 1, c]))
                if r + 1 < h and region_labels[r + 1, c] > 0:
                    side_b.add(int(region_labels[r + 1, c]))
            else:
                if c - 1 >= 0 and region_labels[r, c - 1] > 0:
                    side_a.add(int(region_labels[r, c - 1]))
                if c + 1 < w and region_labels[r, c + 1] > 0:
                    side_b.add(int(region_labels[r, c + 1]))

        assert len(side_a) == 1 and len(side_b) == 1, "each wall segment must separate exactly two regions"
        a = next(iter(side_a))
        b = next(iter(side_b))
        assert a != b
        graph[a].add(b)
        graph[b].add(a)

    parity = {start_region: 0}
    q = deque([start_region])
    while q:
        region = q.popleft()
        for neighbor in graph[region]:
            expected = 1 - parity[region]
            if neighbor in parity:
                assert parity[neighbor] == expected, "region graph must be bipartite"
            else:
                parity[neighbor] = expected
                q.append(neighbor)

    assert len(parity) == n_regions, "every region must be reachable through wall adjacencies"
    return region_labels, graph, parity


def _max_vertical_slide(
    region_labels: np.ndarray, region_id: int, sprite_coords: np.ndarray, direction: int
) -> int:
    h, _ = region_labels.shape
    rows = sprite_coords[:, 0]
    cols = sprite_coords[:, 1]
    step = 0
    while True:
        next_rows = rows + direction * (step + 1)
        if np.any(next_rows < 0) or np.any(next_rows >= h):
            return step
        if not np.all(region_labels[next_rows, cols] == int(region_id)):
            return step
        step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each region's monochrome sprite vertically according to the wall-domain parity rooted at the lower-left region.

    Input structure:
    - The grid contains a dominant background color, one-cell-thick red wall lines, and some monochrome sprites.
    - Treat every non-wall 4-connected area as one background region even when a sprite occupies part of it.
    - Each occupied region contains exactly one monochrome sprite, while some regions may be empty.

    Rule:
    - Decompose the wall network into straight wall segments and connect the two non-wall regions separated by each segment.
    - The region containing the lower-left cell is class A; crossing one wall segment flips between class A and class B.
    - In class A regions, slide the whole sprite straight downward inside its own region; in class B regions, slide it straight upward.
    - A sprite keeps its color and columns, and stops at the first wall or grid boundary that blocks any further move.

    Key insight:
    - The movement direction is not encoded by the sprite itself, but by the hidden bipartite parity of wall-separated regions measured from the lower-left anchor region.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    region_labels, _graph, parity = _analyze_regions(g)

    out = g.copy()
    for region_id in range(1, int(region_labels.max()) + 1):
        region_mask = region_labels == region_id
        colors = sorted(set(map(int, g[region_mask])) - {bg})
        assert len(colors) <= 1, "each region may contain at most one sprite color"
        if not colors:
            continue

        color = colors[0]
        sprite_mask = (g == color) & region_mask
        sprite_labels, n_sprite = label_mask(sprite_mask)
        assert n_sprite == 1, "each region must contain one connected monochrome sprite"
        sprite_coords = np.argwhere(sprite_labels == 1)

        direction = 1 if parity[region_id] == 0 else -1
        shift = _max_vertical_slide(region_labels, region_id, sprite_coords, direction)
        new_coords = sprite_coords.copy()
        new_coords[:, 0] += direction * shift

        out[sprite_coords[:, 0], sprite_coords[:, 1]] = bg
        out[new_coords[:, 0], new_coords[:, 1]] = color

    return out


def _shape_variants(max_h: int, max_w: int) -> list[np.ndarray]:
    variants = []
    seen = set()
    for base in BASE_SHAPES:
        for code in range(8):
            variant = _apply_transform(base.astype(int), code).astype(bool)
            key = (variant.shape, variant.tobytes())
            if key in seen:
                continue
            if variant.shape[0] <= max_h and variant.shape[1] <= max_w:
                seen.add(key)
                variants.append(variant)
    return variants


def _draw_loop(grid: np.ndarray, top: int, left: int, bottom: int, right: int) -> None:
    grid[top, left : right + 1] = WALL_COLOR
    grid[bottom, left : right + 1] = WALL_COLOR
    grid[top : bottom + 1, left] = WALL_COLOR
    grid[top : bottom + 1, right] = WALL_COLOR


def _disjoint(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _make_loop_layout(height: int, width: int, bg: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), bg, dtype=int)

    depths = [1, 2]
    if min(height, width) >= 20 and random.random() < 0.6:
        depths.append(random.choice([1, 2, 3]))
    random.shuffle(depths)

    groups: list[dict] = []
    for depth in depths:
        min_size = 4 * depth + 1
        max_h = min(height - 2, min_size + 6)
        max_w = min(width - 2, min_size + 6)
        if max_h < min_size or max_w < min_size:
            return None

        placed = False
        for _ in range(80):
            gh = random.randint(min_size, max_h)
            gw = random.randint(min_size, max_w)
            top = random.randint(1, height - gh - 1)
            left = random.randint(1, width - gw - 1)
            rect = (top, left, top + gh - 1, left + gw - 1)
            if all(_disjoint(rect, other["outer_rect"], pad=1) for other in groups):
                groups.append({"depth": depth, "outer_rect": rect})
                placed = True
                break
        if not placed:
            return None

    for group in groups:
        top, left, bottom, right = group["outer_rect"]
        for level in range(group["depth"]):
            offset = 2 * level
            _draw_loop(grid, top + offset, left + offset, bottom - offset, right - offset)

    return grid, groups


def _region_boxes(region_labels: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    return {
        region_id: bbox_coords(np.argwhere(region_labels == region_id))
        for region_id in range(1, int(region_labels.max()) + 1)
    }


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = int(color)


def _place_sprite(
    grid: np.ndarray,
    region_labels: np.ndarray,
    region_id: int,
    mask: np.ndarray,
    color: int,
    direction: int,
) -> bool:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(region_labels == region_id))
    sh, sw = mask.shape
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    if sh >= region_h or sw > region_w:
        return False

    if direction > 0:
        row_choices = list(range(r0, r1 - sh + 1))
    else:
        row_choices = list(range(r0 + 1, r1 - sh + 2))
    col_choices = list(range(c0, c1 - sw + 2))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    start_region = int(region_labels[-1, 0])
    for top in row_choices:
        for left in col_choices:
            rows, cols = np.nonzero(mask)
            abs_rows = top + rows
            abs_cols = left + cols
            if not np.all(region_labels[abs_rows, abs_cols] == region_id):
                continue
            if region_id == start_region and np.any((abs_rows == grid.shape[0] - 1) & (abs_cols == 0)):
                continue

            coords = np.column_stack([abs_rows, abs_cols])
            if _max_vertical_slide(region_labels, region_id, coords, direction) < 1:
                continue

            _paint_mask(grid, top, left, mask, color)
            return True
    return False


def _validate_example(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    out = main(g)
    bg = get_bg(g)
    region_labels, _graph, parity = _analyze_regions(g)

    assert np.array_equal(out[g == WALL_COLOR], g[g == WALL_COLOR])

    sprite_stats = []
    area_parities: defaultdict[int, set[int]] = defaultdict(set)
    color_parities: defaultdict[int, set[int]] = defaultdict(set)

    for region_id in range(1, int(region_labels.max()) + 1):
        region_mask = region_labels == region_id
        in_colors = sorted(set(map(int, g[region_mask])) - {bg})
        out_colors = sorted(set(map(int, out[region_mask])) - {bg})
        assert in_colors == out_colors
        if not in_colors:
            continue

        color = in_colors[0]
        in_coords = np.argwhere((g == color) & region_mask)
        out_coords = np.argwhere((out == color) & region_mask)
        assert len(in_coords) == len(out_coords) > 0

        direction = 1 if parity[region_id] == 0 else -1
        shift = _max_vertical_slide(region_labels, region_id, in_coords, direction)
        assert shift >= 1

        expected = in_coords.copy()
        expected[:, 0] += direction * shift
        assert np.array_equal(out_coords, expected)

        area = int(len(in_coords))
        sprite_stats.append((region_id, parity[region_id], color, area, float(in_coords[:, 0].mean())))
        area_parities[area].add(parity[region_id])
        color_parities[color].add(parity[region_id])

    assert len(sprite_stats) >= 3
    assert len(sprite_stats) < int(region_labels.max())
    assert {par for _rid, par, _color, _area, _row in sprite_stats} == {0, 1}
    assert any(len(pars) == 2 for pars in color_parities.values())
    assert any(len(pars) == 2 for pars in area_parities.values())

    assert np.any(out != g)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    bg = random.choice(BACKGROUND_CHOICES)
    layout = _make_loop_layout(height, width, bg)
    if layout is None:
        return None
    wall_grid, groups = layout
    region_labels, _graph, parity = _analyze_regions(wall_grid)
    n_regions = int(region_labels.max())
    if n_regions < 4:
        return None

    boxes = _region_boxes(region_labels)
    outside_region = int(region_labels[-1, 0])

    center_regions_by_parity: dict[int, list[int]] = {0: [], 1: []}
    center_regions: list[int] = []
    for group in groups:
        top, left, bottom, right = group["outer_rect"]
        margin = 2 * group["depth"] - 1
        probe = (top + margin, left + margin)
        region_id = int(region_labels[probe])
        assert region_id != outside_region
        assert parity[region_id] == group["depth"] % 2
        center_regions.append(region_id)
        center_regions_by_parity[parity[region_id]].append(region_id)

    if not center_regions_by_parity[0] or not center_regions_by_parity[1]:
        return None

    shared_down = random.choice(center_regions_by_parity[0])
    shared_up = random.choice(center_regions_by_parity[1])
    selected = [shared_down, shared_up]
    extras = [rid for rid in center_regions if rid not in selected]
    if outside_region not in selected:
        extras.append(outside_region)
    random.shuffle(extras)
    for region_id in extras:
        if len(selected) >= min(5, n_regions - 1):
            break
        selected.append(region_id)
    if len(selected) < 3:
        return None

    down_h = boxes[shared_down][2] - boxes[shared_down][0] + 1
    down_w = boxes[shared_down][3] - boxes[shared_down][1] + 1
    up_h = boxes[shared_up][2] - boxes[shared_up][0] + 1
    up_w = boxes[shared_up][3] - boxes[shared_up][1] + 1
    shared_candidates = _shape_variants(min(down_h, up_h) - 1, min(down_w, up_w))
    if not shared_candidates:
        return None

    shared_mask = random.choice(shared_candidates)
    shared_color_choices = [color for color in SPRITE_PALETTE if color != bg]
    shared_color = random.choice(shared_color_choices)

    grid = wall_grid.copy()
    planned = {
        shared_down: (shared_mask, shared_color),
        shared_up: (shared_mask, shared_color),
    }

    for region_id in selected:
        r0, c0, r1, c1 = boxes[region_id]
        region_h = r1 - r0 + 1
        region_w = c1 - c0 + 1
        if region_id in planned:
            mask, color = planned[region_id]
        else:
            candidates = _shape_variants(region_h - 1, region_w)
            if not candidates:
                return None
            mask = random.choice(candidates)
            color_choices = [value for value in SPRITE_PALETTE if value != bg]
            color = random.choice(color_choices)

        direction = 1 if parity[region_id] == 0 else -1
        if not _place_sprite(grid, region_labels, region_id, mask, color, direction):
            return None

    if _validate_example(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-parity gravity example")
