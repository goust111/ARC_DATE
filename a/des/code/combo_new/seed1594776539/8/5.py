from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GRID_SIZE = 30
RING_AREA_MAX_HEIGHT = 28  # reserve one blank row and one palette row
MAX_DEPTH = 3
MAX_GENERATION_ATTEMPTS = 400
MAX_SUBTREE_DIM = 24
SIBLING_GAP = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    expected_area = 2 * height + 2 * width - 4
    if len(coords) != expected_area:
        return False

    pts = {(int(r), int(c)) for r, c in coords}
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0 + 1, r1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    return True


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested grey rectangle rings by their containment depth using the bottom palette bar.

    Input structure:
    - The background is black.
    - Several grey one-pixel-thick closed rectangular rings appear in the grid.
    - Rings may be disjoint or nested; nested rings sit strictly inside the interiors of larger rings.
    - The bottom row contains a left-to-right color bar whose non-black cells list the output colors for increasing depths.

    Rule:
    - Read the bottom-row color bar from left to right, ignoring black cells.
    - Find every grey 4-connected component that is exactly the border of an axis-aligned rectangle.
    - For each ring, count how many other rings strictly contain it; that count is its nesting depth.
    - Replace the ring's grey pixels with the palette color for that depth, and clear the entire bottom row to black.

    Key insight:
    - All rings share the same input color, so the crucial structure is the strict containment hierarchy, while the bottom color bar is the visible depth-to-color lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _w = g.shape
    assert h >= 2

    palette = [int(g[h - 1, c]) for c in range(g.shape[1]) if int(g[h - 1, c]) != BLACK]
    assert palette, "expected a non-empty palette bar on the bottom row"

    out = g.copy()
    out[h - 1, :] = BLACK

    rings = []
    for coords in _components(out == GREY):
        if not _is_rectangular_ring(coords):
            continue
        rings.append({"coords": coords, "bbox": _bbox(coords)})

    assert rings, "expected at least one grey rectangular ring"

    for ring in rings:
        depth = sum(_contains(other["bbox"], ring["bbox"]) for other in rings if other is not ring)
        assert depth < len(palette), "palette must cover every ring depth"
        color = int(palette[depth])
        out[ring["coords"][:, 0], ring["coords"][:, 1]] = color

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _layout_positions(total: int, sizes: list[int], gap: int) -> list[int]:
    required = sum(sizes) + gap * max(0, len(sizes) - 1)
    assert required <= total
    extras = _random_composition(total - required, len(sizes) + 1)
    pos = extras[0]
    starts: list[int] = []
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if i + 1 < len(sizes):
            pos += gap + extras[i + 1]
    return starts


def _refresh_min_size(node: dict) -> tuple[int, int]:
    children: list[dict] = node["children"]
    if not children:
        node["min_h"] = 3
        node["min_w"] = 3
        return 3, 3

    child_sizes = [_refresh_min_size(child) for child in children]
    child_hs = [size[0] for size in child_sizes]
    child_ws = [size[1] for size in child_sizes]
    if node["axis"] == "h":
        min_h = max(child_hs) + 4
        min_w = sum(child_ws) + SIBLING_GAP * (len(children) - 1) + 4
    else:
        min_h = sum(child_hs) + SIBLING_GAP * (len(children) - 1) + 4
        min_w = max(child_ws) + 4
    node["min_h"] = int(min_h)
    node["min_w"] = int(min_w)
    return int(min_h), int(min_w)


def _build_tree(depth: int, target_depth: int) -> dict:
    if depth == target_depth:
        return {
            "depth": depth,
            "children": [],
            "axis": random.choice(("h", "v")),
            "min_h": 3,
            "min_w": 3,
        }

    primary = _build_tree(depth + 1, target_depth)
    best = {"depth": depth, "children": [primary], "axis": random.choice(("h", "v"))}
    _refresh_min_size(best)

    if random.random() < 0.55:
        for _ in range(8):
            extra_target = random.randint(depth + 1, target_depth)
            extra = _build_tree(depth + 1, extra_target)
            candidate = {
                "depth": depth,
                "children": [primary, extra],
                "axis": random.choice(("h", "v")),
            }
            cand_h, cand_w = _refresh_min_size(candidate)
            if cand_h <= MAX_SUBTREE_DIM and cand_w <= MAX_SUBTREE_DIM:
                best = candidate
                break

    _refresh_min_size(best)
    return best


def _place_tree(node: dict, r0: int, c0: int, height: int, width: int, placed: list[tuple[int, tuple[int, int, int, int]]]) -> None:
    bbox = (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))
    placed.append((int(node["depth"]), bbox))

    children: list[dict] = node["children"]
    if not children:
        return

    inner_r0 = r0 + 2
    inner_c0 = c0 + 2
    inner_h = height - 4
    inner_w = width - 4
    child_hs = [int(child["min_h"]) for child in children]
    child_ws = [int(child["min_w"]) for child in children]

    if node["axis"] == "h":
        xs = _layout_positions(inner_w, child_ws, SIBLING_GAP)
        for child, child_h, child_w, x in zip(children, child_hs, child_ws, xs, strict=False):
            y = random.randint(0, inner_h - child_h)
            _place_tree(child, inner_r0 + y, inner_c0 + x, child_h, child_w, placed)
    else:
        ys = _layout_positions(inner_h, child_hs, SIBLING_GAP)
        for child, child_h, child_w, y in zip(children, child_hs, child_ws, ys, strict=False):
            x = random.randint(0, inner_w - child_w)
            _place_tree(child, inner_r0 + y, inner_c0 + x, child_h, child_w, placed)


def _build_forest() -> tuple[list[dict], str]:
    max_depth = random.randint(0, MAX_DEPTH)
    if max_depth >= 3:
        root_count = random.randint(1, 2)
    else:
        root_count = random.randint(1, 3)

    targets = [max_depth]
    for _ in range(root_count - 1):
        targets.append(random.randint(0, max_depth))
    random.shuffle(targets)

    roots = [_build_tree(0, target) for target in targets]
    return roots, random.choice(("h", "v"))


def _render_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    roots, forest_axis = _build_forest()
    root_hs = [int(root["min_h"]) for root in roots]
    root_ws = [int(root["min_w"]) for root in roots]

    if forest_axis == "h":
        min_ring_h = max(root_hs)
        min_ring_w = sum(root_ws) + SIBLING_GAP * (len(roots) - 1)
    else:
        min_ring_h = sum(root_hs) + SIBLING_GAP * (len(roots) - 1)
        min_ring_w = max(root_ws)

    if min_ring_h > RING_AREA_MAX_HEIGHT or min_ring_w > MAX_GRID_SIZE:
        return None

    max_ring_h = min(RING_AREA_MAX_HEIGHT, min_ring_h + 8)
    max_ring_w = min(MAX_GRID_SIZE, min_ring_w + 8)
    ring_h = random.randint(min_ring_h, max_ring_h)
    ring_w = random.randint(min_ring_w, max_ring_w)
    full_h = ring_h + 2
    if full_h > MAX_GRID_SIZE:
        return None

    if forest_axis == "h":
        root_cs = _layout_positions(ring_w, root_ws, SIBLING_GAP)
        root_rs = [random.randint(0, ring_h - root_h) for root_h in root_hs]
    else:
        root_rs = _layout_positions(ring_h, root_hs, SIBLING_GAP)
        root_cs = [random.randint(0, ring_w - root_w) for root_w in root_ws]

    placed: list[tuple[int, tuple[int, int, int, int]]] = []
    for root, rr, cc, rh, rw in zip(roots, root_rs, root_cs, root_hs, root_ws, strict=False):
        _place_tree(root, rr, cc, rh, rw, placed)

    max_depth_used = max(depth for depth, _bbox in placed)
    palette = random.sample(PALETTE_POOL, max_depth_used + 1)

    grid = np.zeros((full_h, ring_w), dtype=int)
    out = np.zeros_like(grid)
    for depth, bbox in placed:
        _paint_ring(grid, bbox, GREY)
        _paint_ring(out, bbox, int(palette[depth]))

    palette_start = random.randint(0, ring_w - len(palette))
    for i, color in enumerate(palette):
        grid[full_h - 1, palette_start + i] = int(color)

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rendered = _render_attempt()
        if rendered is None:
            continue
        grid, expected = rendered
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid nested-ring recoloring input")
