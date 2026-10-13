from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
PALETTE = tuple(range(1, 10))

# Each family is a 4-rectangle path: box0-box1-box2-box3.
FAMILIES = (
    ((1, 3, 9, 11), (8, 6, 18, 16), (2, 12, 10, 20), (8, 17, 20, 23)),
    ((2, 4, 8, 10), (7, 8, 13, 16), (1, 15, 11, 25), (8, 18, 18, 24)),
    ((1, 3, 9, 9), (5, 6, 15, 16), (3, 13, 13, 21), (12, 17, 20, 25)),
)
TRANSFORMS = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "diag", "anti")
GEN_ATTEMPTS = 400


def _border_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _border_set(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return set(_border_coords(box))


def _paint_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _visible_boxes(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != 0)
    if len(colors) != 4:
        raise AssertionError("expected exactly four visible rectangle colors")
    return {color: _bbox_of_mask(grid == color) for color in colors}


def _build_occlusion_edges(
    grid: np.ndarray,
    boxes: dict[int, tuple[int, int, int, int]],
) -> dict[int, set[int]]:
    colors = sorted(boxes)
    border_sets = {color: _border_set(boxes[color]) for color in colors}
    edges = {color: set() for color in colors}

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            inter = border_sets[a] & border_sets[b]
            if not inter:
                continue
            visible = {int(grid[r, c]) for r, c in inter if int(grid[r, c]) in (a, b)}
            if len(visible) != 1:
                raise AssertionError("crossing visibility must identify a unique occluder")
            top = next(iter(visible))
            if top == a:
                edges[b].add(a)
            else:
                edges[a].add(b)
    return edges


def _unique_topological_order(edges: dict[int, set[int]]) -> list[int]:
    indeg = {node: 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            indeg[dst] += 1

    remaining = set(edges)
    order: list[int] = []
    while remaining:
        zeros = sorted(node for node in remaining if indeg[node] == 0)
        if len(zeros) != 1:
            raise AssertionError("occlusion graph must have a unique topological order")
        node = zeros[0]
        remaining.remove(node)
        order.append(node)
        for dst in edges[node]:
            indeg[dst] -= 1
    return order


def _render_borders(
    shape: tuple[int, int],
    boxes: dict[int, tuple[int, int, int, int]],
    ordered_colors: list[int],
) -> np.ndarray:
    scene = np.zeros(shape, dtype=int)
    for color in ordered_colors:
        _paint_border(scene, boxes[color], color)
    return scene


def _draw_rays(
    scene: np.ndarray,
    boxes: dict[int, tuple[int, int, int, int]],
    ordered_colors: list[int],
) -> tuple[np.ndarray, Counter]:
    h, w = map(int, scene.shape)
    ray_mask = np.zeros((h, w), dtype=bool)
    ray_counts: Counter = Counter()

    for color in ordered_colors:
        r0, c0, r1, c1 = boxes[color]
        if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
            raise AssertionError("rectangle centers must land on integer cells")
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        for dr, dc in DIAG4:
            r = cr + dr
            c = cc + dc
            while 0 <= r < h and 0 <= c < w and not ray_mask[r, c]:
                ray_mask[r, c] = True
                scene[r, c] = int(color)
                ray_counts[int(color)] += 1
                r += dr
                c += dc
    return ray_mask, ray_counts


def _fill_black_regions(
    scene: np.ndarray,
    ray_mask: np.ndarray,
    *,
    strict: bool,
) -> tuple[np.ndarray, list[Counter]]:
    h, w = map(int, scene.shape)
    out = scene.copy()
    seen = np.zeros((h, w), dtype=bool)
    region_touches: list[Counter] = []

    for r in range(h):
        for c in range(w):
            if scene[r, c] != 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and scene[nr, nc] == 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            touch: Counter = Counter()
            for rr, cc in comp:
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and ray_mask[nr, nc]:
                        touch[int(scene[nr, nc])] += 1

            if not touch:
                raise AssertionError("every black region must touch at least one ray")

            ranked = sorted(touch.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
            if strict and len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                raise AssertionError("fill-color majority must be unique")
            fill_color = int(ranked[0][0])
            for rr, cc in comp:
                out[rr, cc] = fill_color
            region_touches.append(touch)

    return out, region_touches


def _simulate(
    shape: tuple[int, int],
    boxes: dict[int, tuple[int, int, int, int]],
    ordered_colors: list[int],
    *,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray, Counter, list[Counter]]:
    work = _render_borders(shape, boxes, ordered_colors)
    ray_mask, ray_counts = _draw_rays(work, boxes, ordered_colors)
    filled, region_touches = _fill_black_regions(work, ray_mask, strict=strict)
    out = filled.copy()
    for color in ordered_colors:
        _paint_border(out, boxes[color], color)
    return out, ray_mask, ray_counts, region_touches


def _ray_signature(
    shape: tuple[int, int],
    boxes: dict[int, tuple[int, int, int, int]],
    ordered_colors: list[int],
) -> np.ndarray:
    scene = _render_borders(shape, boxes, ordered_colors)
    ray_mask, _ = _draw_rays(scene, boxes, ordered_colors)
    return ray_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapping rectangle borders, infer their occlusion order, then color the ray-partitioned background by ray ownership.

    Input structure:
    - A black background contains exactly four single-pixel-thick colored rectangular borders.
    - Border overlaps are rendered by visibility only: the top border color overwrites lower colors at crossing pixels.
    - Each rectangle's full geometry is recoverable from the bounding box of its visible pixels.

    Rule:
    - Reconstruct each rectangle from the bbox of one visible color.
    - For every pair of reconstructed borders, inspect their crossing pixels; if the visible color there is B, then B occludes A and adds edge A->B.
    - Topologically sort the resulting occlusion chain from bottom to top, draw four diagonal rays from each rectangle center in that order, stopping each ray before an earlier ray pixel or the grid boundary.
    - Fill every remaining black 4-connected region with the majority color among the ray pixels touching that region, then redraw the full rectangle borders in bottom-to-top order.

    Key insight:
    - The decisive ordering signal is not spatial placement but crossing visibility: only the occlusion graph reveals which rays fire first and therefore which later rays get blocked, so the final background partition depends on inferred layer order.
    """
    g = np.asarray(grid, dtype=int)
    boxes = _visible_boxes(g)
    edges = _build_occlusion_edges(g, boxes)
    order = _unique_topological_order(edges)
    out, _ray_mask, _ray_counts, _region_touches = _simulate(g.shape, boxes, order, strict=True)
    return out


def _transform_point(r: int, c: int, n: int, mode: str) -> tuple[int, int]:
    if mode == "id":
        return int(r), int(c)
    if mode == "rot90":
        return int(c), int(n - 1 - r)
    if mode == "rot180":
        return int(n - 1 - r), int(n - 1 - c)
    if mode == "rot270":
        return int(n - 1 - c), int(r)
    if mode == "flip_h":
        return int(n - 1 - r), int(c)
    if mode == "flip_v":
        return int(r), int(n - 1 - c)
    if mode == "diag":
        return int(c), int(r)
    if mode == "anti":
        return int(n - 1 - c), int(n - 1 - r)
    raise ValueError(f"unknown transform: {mode}")


def _transform_box(box: tuple[int, int, int, int], n: int, mode: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts = [_transform_point(r, c, n, mode) for r, c in corners]
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _translated_boxes(
    boxes: tuple[tuple[int, int, int, int], ...],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple((r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in boxes)


def _render_input(
    n: int,
    boxes: tuple[tuple[int, int, int, int], ...],
    colors: list[int],
    ordered_indices: list[int],
) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for idx in ordered_indices:
        _paint_border(grid, boxes[idx], colors[idx])
    return grid


def _validate_instance(
    inp: np.ndarray,
    boxes: tuple[tuple[int, int, int, int], ...],
    colors: list[int],
    ordered_indices: list[int],
) -> bool:
    ordered_colors = [int(colors[idx]) for idx in ordered_indices]
    boxes_by_color = {int(colors[idx]): tuple(map(int, boxes[idx])) for idx in range(4)}

    recovered_boxes = _visible_boxes(inp)
    if recovered_boxes != boxes_by_color:
        return False

    edges = _build_occlusion_edges(inp, recovered_boxes)
    if _unique_topological_order(edges) != ordered_colors:
        return False

    expected, ray_mask, ray_counts, region_touches = _simulate(inp.shape, boxes_by_color, ordered_colors, strict=True)
    if not np.array_equal(main(inp), expected):
        return False

    if len(ray_counts) != 4 or any(int(ray_counts[color]) <= 0 for color in ordered_colors):
        return False

    if not any(len(touch) == 1 for touch in region_touches):
        return False
    if not any(len(touch) > 1 for touch in region_touches):
        return False

    true_sig = ray_mask.tobytes()
    alts = [
        [ordered_colors[1], ordered_colors[0], ordered_colors[2], ordered_colors[3]],
        [ordered_colors[0], ordered_colors[2], ordered_colors[1], ordered_colors[3]],
        list(reversed(ordered_colors)),
    ]
    diff_count = 0
    for alt in alts:
        if alt == ordered_colors:
            continue
        alt_sig = _ray_signature(inp.shape, boxes_by_color, alt).tobytes()
        if alt_sig != true_sig:
            diff_count += 1
    if diff_count < 2:
        return False

    if np.array_equal(inp, expected):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n = random.randint(27, 29)
        family = random.choice(FAMILIES)

        min_r = min(box[0] for box in family)
        max_r = max(box[2] for box in family)
        min_c = min(box[1] for box in family)
        max_c = max(box[3] for box in family)
        dr = random.randint(1 - min_r, (n - 2) - max_r)
        dc = random.randint(1 - min_c, (n - 2) - max_c)

        boxes = _translated_boxes(family, dr, dc)
        mode = random.choice(TRANSFORMS)
        boxes = tuple(_transform_box(box, n, mode) for box in boxes)

        colors = random.sample(PALETTE, 4)
        ordered_indices = [0, 1, 2, 3] if random.random() < 0.5 else [3, 2, 1, 0]
        inp = _render_input(n, boxes, colors, ordered_indices)

        try:
            if _validate_instance(inp, boxes, colors, ordered_indices):
                return inp
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid occlusion-ray ARC instance")
