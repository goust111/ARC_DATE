from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

DEPTH_TO_RING_COLOR = {1: RED, 2: GREEN, 3: YELLOW}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


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


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_masks(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    nbrs = _neighbors4 if connectivity == 4 else _neighbors8
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = np.zeros_like(mask, dtype=bool)
            comp[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in nbrs(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        comp[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _flood_from_boundary(passable: np.ndarray) -> np.ndarray:
    height, width = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(height):
        for c in (0, width - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _compute_ring_interior_domain(ring_mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(ring_mask)
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(ring_mask.shape[0] - 1, r1 + 1)
    c1 = min(ring_mask.shape[1] - 1, c1 + 1)
    local_ring = ring_mask[r0 : r1 + 1, c0 : c1 + 1]
    passable = ~local_ring
    outside = _flood_from_boundary(passable)
    interior_local = passable & ~outside
    comps = _component_masks(interior_local, connectivity=4)
    assert len(comps) == 1 and interior_local.any(), "each ring must enclose one interior domain"
    interior = np.zeros_like(ring_mask, dtype=bool)
    interior[r0 : r1 + 1, c0 : c1 + 1] = interior_local
    return interior


def _boundary_of_filled(filled: np.ndarray) -> np.ndarray:
    height, width = filled.shape
    out = np.zeros_like(filled, dtype=bool)
    for r, c in np.argwhere(filled):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not _in_bounds(filled.shape, nr, nc) or not filled[nr, nc]:
                out[rr, cc] = True
                break
    return out


def _analyze_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert colors == [BLUE], "expected only blue rings"

    ring_components = _component_masks(g == BLUE, connectivity=4)
    assert 3 <= len(ring_components) <= 5, "expected 3-5 rings"

    rings: list[dict] = []
    for idx, ring_mask in enumerate(ring_components):
        interior = _compute_ring_interior_domain(ring_mask)
        rings.append(
            {
                "id": idx,
                "mask": ring_mask,
                "coords": np.argwhere(ring_mask),
                "interior": interior,
            }
        )

    for ring in rings:
        containers = []
        for other in rings:
            if other["id"] == ring["id"]:
                continue
            coords = ring["coords"]
            inside = other["interior"][coords[:, 0], coords[:, 1]]
            if bool(np.any(inside)):
                assert bool(np.all(inside)), "containment must be all-or-nothing"
                containers.append(other["id"])
        ring["depth"] = 1 + len(containers)
        assert ring["depth"] in DEPTH_TO_RING_COLOR, "nesting depth must be at most 3"

    depths = {int(r["depth"]) for r in rings}
    assert 1 in depths and 2 in depths, "scene must contain both odd and even depths"
    return rings


def _render_output_from_rings(input_grid: np.ndarray, rings: list[dict]) -> np.ndarray:
    out = np.zeros_like(input_grid)
    for ring in rings:
        out[ring["mask"]] = DEPTH_TO_RING_COLOR[int(ring["depth"])]

    for ring in sorted(rings, key=lambda item: int(item["depth"]), reverse=True):
        depth = int(ring["depth"])
        paint_inside = depth % 2 == 0
        outline_color = ORANGE if paint_inside else GRAY
        for r, c in ring["coords"]:
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors8(rr, cc):
                if not _in_bounds(input_grid.shape, nr, nc):
                    continue
                if int(input_grid[nr, nc]) != BLACK or int(out[nr, nc]) != BLACK:
                    continue
                in_hole = bool(ring["interior"][nr, nc])
                if in_hole == paint_inside:
                    out[nr, nc] = outline_color
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor each blue ring by its nesting depth and draw only the depth-selected side of its outline.

    Input structure:
    - The grid is a black background containing 3-5 blue 4-connected closed rings.
    - Every ring is a one-pixel-thick contour enclosing one geometric interior region.
    - Some rings lie inside the interior of other rings, creating 2-3 nesting levels.

    Rule:
    - Infer each ring's depth from containment: depth 1 rings are not inside any other ring, depth 2 rings are inside one ring, and depth 3 rings are inside two rings.
    - Recolor the ring body by depth: 1 -> red, 2 -> green, 3 -> yellow.
    - Process rings from deepest to shallowest; odd depths paint gray on black cells that are 8-neighbors on the outside of the ring, while even depths paint orange on black cells that are 8-neighbors on the inside of the ring.
    - If a candidate outline cell was already painted by a deeper ring, keep the earlier color.

    Key insight:
    - All input rings look identical in color, so the only way to decide both the recolor and which side gets outlined is to recover the nesting relation from enclosed space rather than from size or distance.
    """
    g = np.asarray(input_grid, dtype=int)
    rings = _analyze_rings(g)
    return _render_output_from_rings(g, rings)


def _shift_mask(local_mask: np.ndarray, shape: tuple[int, int], r0: int, c0: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = local_mask.shape
    out[r0 : r0 + h, c0 : c0 + w] = local_mask
    return out


def _dilate8(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            src_r0 = max(0, -dr)
            src_r1 = min(mask.shape[0], mask.shape[0] - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(mask.shape[1], mask.shape[1] - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(mask.shape[0], mask.shape[0] + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(mask.shape[1], mask.shape[1] + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _erode8(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            shifted = np.zeros_like(mask, dtype=bool)
            src_r0 = max(0, -dr)
            src_r1 = min(mask.shape[0], mask.shape[0] - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(mask.shape[1], mask.shape[1] - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(mask.shape[0], mask.shape[0] + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(mask.shape[1], mask.shape[1] + dc)
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
            out &= shifted
    return out


def _mask_component_count(mask: np.ndarray, connectivity: int = 4) -> int:
    return len(_component_masks(mask, connectivity=connectivity))


def _node_specs() -> list[dict]:
    if random.random() < 0.65:
        root0 = {"children": [{"children": [{"children": []}]}]}
        root1 = {"children": []}
        roots = [root0, root1]
        if random.random() < 0.55:
            root1["children"].append({"children": []})
        return roots

    roots = [{"children": []}, {"children": []}]
    total = random.randint(3, 5)
    remaining = total - 2
    roots[random.randrange(2)]["children"].append({"children": []})
    remaining -= 1
    while remaining > 0:
        choice = random.choice(roots)
        if len(choice["children"]) < 2:
            choice["children"].append({"children": []})
            remaining -= 1
        else:
            break
    return roots


def _flatten_nodes(nodes: list[dict], depth: int = 1) -> list[dict]:
    flat = []
    for node in nodes:
        node["depth"] = depth
        flat.append(node)
        flat.extend(_flatten_nodes(node["children"], depth + 1))
    return flat


def _subtree_levels(node: dict) -> int:
    if not node["children"]:
        return 1
    return 1 + max(_subtree_levels(child) for child in node["children"])


def _min_size(node: dict) -> tuple[int, int]:
    if not node["children"]:
        return 5, 5
    if len(node["children"]) == 1:
        ch, cw = _min_size(node["children"][0])
        return ch + 4, cw + 4
    a, b = node["children"]
    ah, aw = _min_size(a)
    bh, bw = _min_size(b)
    if node["split"] == "h":
        return max(ah, bh) + 4, aw + bw + 5
    return ah + bh + 5, max(aw, bw) + 4


def _assign_splits(node: dict) -> None:
    if len(node["children"]) == 2:
        node["split"] = random.choice(["h", "v"])
    for child in node["children"]:
        _assign_splits(child)


def _rect_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = rect
    filled = np.zeros(shape, dtype=bool)
    filled[r0 : r1 + 1, c0 : c1 + 1] = True
    ring = np.zeros(shape, dtype=bool)
    ring[r0, c0 : c1 + 1] = True
    ring[r1, c0 : c1 + 1] = True
    ring[r0 : r1 + 1, c0] = True
    ring[r0 : r1 + 1, c1] = True
    return filled, ring


def _place_in_container(container: tuple[int, int, int, int], node: dict) -> tuple[int, int, int, int] | None:
    cr0, cc0, cr1, cc1 = container
    min_h, min_w = _min_size(node)
    max_h = cr1 - cr0 + 1
    max_w = cc1 - cc0 + 1
    if min_h > max_h or min_w > max_w:
        return None

    extra_h = min(4, max_h - min_h)
    extra_w = min(4, max_w - min_w)
    h = min_h + random.randint(0, extra_h)
    w = min_w + random.randint(0, extra_w)
    r0 = random.randint(cr0, cr1 - h + 1)
    c0 = random.randint(cc0, cc1 - w + 1)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _child_containers(rect: tuple[int, int, int, int], node: dict) -> list[tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = rect
    inner = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    if inner[0] > inner[2] or inner[1] > inner[3]:
        return None
    if not node["children"]:
        return []
    if len(node["children"]) == 1:
        return [inner]

    a, b = node["children"]
    ah, aw = _min_size(a)
    bh, bw = _min_size(b)
    ir0, ic0, ir1, ic1 = inner
    ih = ir1 - ir0 + 1
    iw = ic1 - ic0 + 1

    if node["split"] == "h":
        need_w = aw + bw + 1
        if need_w > iw or max(ah, bh) > ih:
            return None
        cut = random.randint(ic0 + aw, ic1 - bw)
        left = (ir0, ic0, ir1, cut - 1)
        right = (ir0, cut + 1, ir1, ic1)
        return [left, right]

    need_h = ah + bh + 1
    if need_h > ih or max(aw, bw) > iw:
        return None
    cut = random.randint(ir0 + ah, ir1 - bh)
    top = (ir0, ic0, cut - 1, ic1)
    bottom = (cut + 1, ic0, ir1, ic1)
    return [top, bottom]


def _place_subtree(node: dict, container: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    for _ in range(60):
        rect = _place_in_container(container, node)
        if rect is None:
            return False
        child_boxes = _child_containers(rect, node)
        if child_boxes is None:
            continue
        filled, ring = _rect_mask(shape, rect)
        node["ring_mask"] = ring
        node["ring_coords"] = np.argwhere(ring)
        node["domain_mask"] = filled & ~ring
        ok = True
        for child, child_container in zip(node["children"], child_boxes):
            if not _place_subtree(child, child_container, shape):
                ok = False
                break
        if ok:
            return True
    return False


def _build_scene() -> tuple[np.ndarray, list[dict], np.ndarray] | None:
    roots = _node_specs()
    for root in roots:
        _assign_splits(root)
    flat = _flatten_nodes(roots)
    max_depth = max(int(node["depth"]) for node in flat)
    total = len(flat)
    grid_h = random.randint(20 if max_depth == 3 else 18, 28)
    grid_w = random.randint(20 if total >= 4 else 18, 28)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    outer = (1, 1, grid_h - 2, grid_w - 2)
    if len(roots) != 2:
        return None

    left_root, right_root = roots
    lh, lw = _min_size(left_root)
    rh, rw = _min_size(right_root)
    split = random.choice(["h", "v"])
    if split == "h":
        if lw + rw + 2 > outer[3] - outer[1] + 1 or max(lh, rh) > outer[2] - outer[0] + 1:
            split = "v"
    if split == "v":
        if lh + rh + 2 > outer[2] - outer[0] + 1 or max(lw, rw) > outer[3] - outer[1] + 1:
            return None

    if split == "h":
        cut_min = outer[1] + lw
        cut_max = outer[3] - rw - 1
        if cut_min > cut_max:
            return None
        cut = random.randint(cut_min, cut_max)
        containers = [(outer[0], outer[1], outer[2], cut - 1), (outer[0], cut + 1, outer[2], outer[3])]
    else:
        cut_min = outer[0] + lh
        cut_max = outer[2] - rh - 1
        if cut_min > cut_max:
            return None
        cut = random.randint(cut_min, cut_max)
        containers = [(outer[0], outer[1], cut - 1, outer[3]), (cut + 1, outer[1], outer[2], outer[3])]

    for root, container in zip(roots, containers):
        if not _place_subtree(root, container, grid.shape):
            return None

    nodes = _flatten_nodes(roots)
    for node in nodes:
        grid[node["ring_mask"]] = BLUE

    expected = np.zeros_like(grid)
    for node in nodes:
        expected[node["ring_mask"]] = DEPTH_TO_RING_COLOR[int(node["depth"])]
    nodes_by_depth = sorted(nodes, key=lambda item: int(item["depth"]), reverse=True)
    for node in nodes_by_depth:
        paint_inside = int(node["depth"]) % 2 == 0
        outline_color = ORANGE if paint_inside else GRAY
        for r, c in node["ring_coords"]:
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors8(rr, cc):
                if not _in_bounds(grid.shape, nr, nc) or grid[nr, nc] != BLACK or expected[nr, nc] != BLACK:
                    continue
                if bool(node["domain_mask"][nr, nc]) == paint_inside:
                    expected[nr, nc] = outline_color
    return grid, nodes, expected


def _validate_scene(grid: np.ndarray, nodes: list[dict], expected: np.ndarray) -> bool:
    try:
        derived = _analyze_rings(grid)
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    depth_by_mask = {}
    for ring in derived:
        depth_by_mask[ring["mask"].tobytes()] = int(ring["depth"])
    planned_depths = []
    actual_depths = []
    for node in nodes:
        key = node["ring_mask"].tobytes()
        if key not in depth_by_mask:
            return False
        planned_depths.append(int(node["depth"]))
        actual_depths.append(depth_by_mask[key])
    if sorted(planned_depths) != sorted(actual_depths):
        return False

    odd_count = sum(depth % 2 == 1 for depth in planned_depths)
    even_count = sum(depth % 2 == 0 for depth in planned_depths)
    if odd_count == 0 or even_count == 0:
        return False

    if len(set(planned_depths)) < 2:
        return False
    if int((grid == BLUE).sum()) <= 20:
        return False
    if np.unique(expected).size < 4:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_scene()
        if built is None:
            continue
        grid, nodes, expected = built
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        if _validate_scene(grid, nodes, expected):
            return grid
    raise RuntimeError("failed to generate a valid nested-ring ARC input")
