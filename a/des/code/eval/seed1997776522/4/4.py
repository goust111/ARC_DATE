from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np


BG = 0
BOUNDARY = 3
DEPTH_FILL = {1: 1, 2: 4, 3: 6, 4: 7}
MARKER_COLORS = (2, 5, 8, 9)

MIN_DEPTH = 2
MAX_DEPTH = 4
MIN_SHAPE_DIM = 5
MAX_INNER_SHAPE_DIM = 8
WRAP_BAND_RANGE = (2, 4)  # inclusive low, exclusive high
WRAP_SLACK_RANGE = (0, 3)
GRID_MARGIN_RANGE = (1, 5)
GRID_MAX_DIM = 34
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                out[rr, cc] = True
                break
    return out


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = comp_id
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
            components.append(np.asarray(pts, dtype=int))
            comp_id += 1
    return labels, components


def _scale_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = mask.shape
    assert in_h > 0 and in_w > 0 and out_h > 0 and out_w > 0
    row_edges = np.rint(np.linspace(0, out_h, in_h + 1)).astype(int)
    col_edges = np.rint(np.linspace(0, out_w, in_w + 1)).astype(int)
    scaled = np.zeros((out_h, out_w), dtype=bool)
    for r in range(in_h):
        r0, r1 = int(row_edges[r]), int(row_edges[r + 1])
        if r0 == r1:
            continue
        for c in range(in_w):
            if not mask[r, c]:
                continue
            c0, c1 = int(col_edges[c]), int(col_edges[c + 1])
            if c0 == c1:
                continue
            scaled[r0:r1, c0:c1] = True
    return scaled


def _detect_marker_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, grid.ravel())) - {BG, BOUNDARY})
    assert len(colors) == 1, "expected exactly one marker color"
    marker_color = int(colors[0])
    assert int(np.count_nonzero(grid == marker_color)) == 1, "marker must be unique"
    return marker_color


def _region_adjacency(labels: np.ndarray, boundary_mask: np.ndarray) -> list[set[int]]:
    n = int(labels.max()) + 1
    adj = [set() for _ in range(n)]
    h, w = labels.shape
    for r, c in np.argwhere(boundary_mask):
        rr = int(r)
        cc = int(c)
        neigh_ids = {
            int(labels[nr, nc])
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] >= 0
        }
        for a, b in combinations(sorted(neigh_ids), 2):
            adj[a].add(b)
            adj[b].add(a)
    return adj


def _region_depths(labels: np.ndarray, components: list[np.ndarray], boundary_mask: np.ndarray) -> dict[int, int]:
    outside_ids = {
        int(labels[0, c]) for c in range(labels.shape[1]) if labels[0, c] >= 0
    } | {
        int(labels[-1, c]) for c in range(labels.shape[1]) if labels[-1, c] >= 0
    } | {
        int(labels[r, 0]) for r in range(labels.shape[0]) if labels[r, 0] >= 0
    } | {
        int(labels[r, -1]) for r in range(labels.shape[0]) if labels[r, -1] >= 0
    }
    assert len(outside_ids) == 1, "expected one outside background region"
    outside_id = next(iter(outside_ids))

    adj = _region_adjacency(labels, boundary_mask)
    depths = {outside_id: 0}
    q = deque([outside_id])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in depths:
                depths[nxt] = depths[cur] + 1
                q.append(nxt)
    assert len(depths) == len(components), "all non-boundary regions must be connected through nesting"
    return depths


def _ring_cells_between(
    labels: np.ndarray,
    depths: dict[int, int],
    boundary_mask: np.ndarray,
    low_depth: int,
    high_depth: int,
) -> np.ndarray:
    h, w = labels.shape
    pts: list[tuple[int, int]] = []
    for r, c in np.argwhere(boundary_mask):
        rr = int(r)
        cc = int(c)
        neigh_depths = {
            int(depths[int(labels[nr, nc])])
            for nr, nc in _neighbors4(rr, cc)
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] >= 0
        }
        if low_depth in neigh_depths and high_depth in neigh_depths:
            pts.append((rr, cc))
    assert pts, f"no ring found between depths {low_depth} and {high_depth}"
    return np.asarray(pts, dtype=int)


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker_color = _detect_marker_color(g)
    passable = g != BOUNDARY
    labels, components = _label_components(passable)
    assert components, "expected at least the outside region"
    depths = _region_depths(labels, components, g == BOUNDARY)

    marker_pos = tuple(map(int, np.argwhere(g == marker_color)[0]))
    marker_region = int(labels[marker_pos])
    deepest_depth = int(max(depths.values()))
    assert deepest_depth >= 1
    assert depths[marker_region] == deepest_depth, "marker must sit in the deepest enclosed region"

    outer_ring = _ring_cells_between(labels, depths, g == BOUNDARY, 0, 1)
    inner_ring = _ring_cells_between(labels, depths, g == BOUNDARY, deepest_depth - 1, deepest_depth)
    return {
        "labels": labels,
        "components": components,
        "depths": depths,
        "marker_color": marker_color,
        "marker_region": marker_region,
        "deepest_depth": deepest_depth,
        "outer_ring": outer_ring,
        "inner_ring": inner_ring,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color nested enclosed regions by depth, then project the innermost ring onto the outermost ring area.

    Input structure:
    - black background
    - green ring boundaries with 2-4 nesting levels
    - one unique non-green, non-black marker pixel inside the deepest region

    Rule:
    - Treat green as walls and compute the depth of every non-green region.
    - Fill depth-1/2/3/4 regions with blue/yellow/magenta/orange, leaving the marker unchanged.
    - Extract the innermost ring's binary shape, scale it to the outermost ring's bbox, and paint it
      in the marker color over that outer bbox.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    labels = info["labels"]
    depths = info["depths"]
    marker_color = int(info["marker_color"])
    outer_ring = np.asarray(info["outer_ring"], dtype=int)
    inner_ring = np.asarray(info["inner_ring"], dtype=int)

    out = g.copy()
    for comp_id, depth in depths.items():
        depth = int(depth)
        if depth == 0:
            continue
        fill_color = DEPTH_FILL[min(depth, max(DEPTH_FILL))]
        region_mask = labels == int(comp_id)
        out[region_mask & (g == BG)] = fill_color

    ir0, ic0, ir1, ic1 = _bbox(inner_ring)
    or0, oc0, or1, oc1 = _bbox(outer_ring)
    inner_mask = np.zeros((ir1 - ir0 + 1, ic1 - ic0 + 1), dtype=bool)
    inner_mask[inner_ring[:, 0] - ir0, inner_ring[:, 1] - ic0] = True
    scaled = _scale_mask(inner_mask, or1 - or0 + 1, oc1 - oc0 + 1)
    outer_view = out[or0 : or1 + 1, oc0 : oc1 + 1]
    outer_view[scaled] = marker_color
    return out


def _shape_rect(h: int, w: int, band: int, rng: np.random.Generator) -> np.ndarray:
    _ = band, rng
    return np.ones((h, w), dtype=bool)


def _shape_corner_cut(h: int, w: int, band: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    cut_h = int(rng.integers(1, band + 1))
    cut_w = int(rng.integers(1, band + 1))
    corner = int(rng.integers(0, 4))
    if corner == 0:
        mask[:cut_h, :cut_w] = False
    elif corner == 1:
        mask[:cut_h, w - cut_w :] = False
    elif corner == 2:
        mask[h - cut_h :, :cut_w] = False
    else:
        mask[h - cut_h :, w - cut_w :] = False
    return mask


def _shape_side_notch(h: int, w: int, band: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    side = int(rng.integers(0, 4))
    if side in (0, 1):
        depth = int(rng.integers(1, band + 1))
        max_span = max(2, w - 2)
        min_span = min(max_span, max(2, w // 3))
        span = int(rng.integers(min_span, max_span + 1))
        c0 = int(rng.integers(1, w - span))
        if side == 0:
            mask[:depth, c0 : c0 + span] = False
        else:
            mask[h - depth :, c0 : c0 + span] = False
    else:
        depth = int(rng.integers(1, band + 1))
        max_span = max(2, h - 2)
        min_span = min(max_span, max(2, h // 3))
        span = int(rng.integers(min_span, max_span + 1))
        r0 = int(rng.integers(1, h - span))
        if side == 2:
            mask[r0 : r0 + span, :depth] = False
        else:
            mask[r0 : r0 + span, w - depth :] = False
    return mask


def _shape_double_corner(h: int, w: int, band: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    cut1_h = int(rng.integers(1, band + 1))
    cut1_w = int(rng.integers(1, band + 1))
    cut2_h = int(rng.integers(1, band + 1))
    cut2_w = int(rng.integers(1, band + 1))
    pair = int(rng.integers(0, 2))
    if pair == 0:
        mask[:cut1_h, :cut1_w] = False
        mask[h - cut2_h :, w - cut2_w :] = False
    else:
        mask[:cut1_h, w - cut1_w :] = False
        mask[h - cut2_h :, :cut2_w] = False
    return mask


def _shape_stair_cut(h: int, w: int, band: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    corner = int(rng.integers(0, 4))
    height = int(rng.integers(1, band + 1))
    width = int(rng.integers(1, band + 1))
    if corner == 0:
        for r in range(height):
            span = width - min(width - 1, (r * width) // max(1, height))
            mask[r, :span] = False
    elif corner == 1:
        for r in range(height):
            span = width - min(width - 1, (r * width) // max(1, height))
            mask[r, w - span :] = False
    elif corner == 2:
        for r in range(height):
            span = width - min(width - 1, (r * width) // max(1, height))
            mask[h - 1 - r, :span] = False
    else:
        for r in range(height):
            span = width - min(width - 1, (r * width) // max(1, height))
            mask[h - 1 - r, w - span :] = False
    return mask


SHAPE_FAMILIES: tuple[tuple[str, object], ...] = (
    ("rect", _shape_rect),
    ("corner_cut", _shape_corner_cut),
    ("side_notch", _shape_side_notch),
    ("double_corner", _shape_double_corner),
    ("stair_cut", _shape_stair_cut),
)


def _make_shape_mask(
    h: int,
    w: int,
    band: int,
    rng: np.random.Generator,
    family_name: str | None = None,
) -> tuple[np.ndarray, str]:
    choices = list(SHAPE_FAMILIES)
    if family_name is None:
        family_name, func = choices[int(rng.integers(0, len(choices)))]
    else:
        func = dict(choices)[family_name]

    for _ in range(40):
        mask = np.asarray(func(h, w, band, rng), dtype=bool)
        k = int(rng.integers(0, 4 if h == w else 2)) * (1 if h == w else 2)
        if k:
            mask = np.rot90(mask, k=k)
        if mask.shape != (h, w):
            mask = np.asarray(mask, dtype=bool)
            assert mask.shape == (h, w)
        interior = mask & ~_perimeter(mask)
        if not np.any(interior):
            continue
        if _count_components(mask) != 1 or _count_components(interior) != 1:
            continue
        if not np.any(mask[0]) or not np.any(mask[-1]) or not np.any(mask[:, 0]) or not np.any(mask[:, -1]):
            continue
        return mask, str(family_name)
    raise AssertionError("failed to build a valid shape mask")


def _render_ring(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[_perimeter(mask)] = BOUNDARY
    return patch


def _pick_marker_cell(mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    interior = np.argwhere(mask & ~_perimeter(mask))
    assert len(interior) > 0
    h, w = mask.shape
    corner = int(rng.integers(0, 4))
    if corner == 0:
        key = lambda p: (int(p[0]) + int(p[1]), int(p[0]), int(p[1]))
    elif corner == 1:
        key = lambda p: (int(p[0]) + (w - 1 - int(p[1])), int(p[0]), -int(p[1]))
    elif corner == 2:
        key = lambda p: ((h - 1 - int(p[0])) + int(p[1]), -int(p[0]), int(p[1]))
    else:
        key = lambda p: ((h - 1 - int(p[0])) + (w - 1 - int(p[1])), -int(p[0]), -int(p[1]))
    best = min(interior.tolist(), key=key)
    return int(best[0]), int(best[1])


def _make_deepest_patch(rng: np.random.Generator, marker_color: int, family_name: str) -> np.ndarray:
    for _ in range(60):
        h = int(rng.integers(MIN_SHAPE_DIM, MAX_INNER_SHAPE_DIM + 1))
        w = int(rng.integers(MIN_SHAPE_DIM, MAX_INNER_SHAPE_DIM + 1))
        mask, _family = _make_shape_mask(h, w, band=2, rng=rng, family_name=family_name)
        patch = _render_ring(mask)
        mr, mc = _pick_marker_cell(mask, rng)
        patch[mr, mc] = marker_color
        return patch
    raise AssertionError("failed to build deepest patch")


def _wrap_patch(
    child_patch: np.ndarray,
    rng: np.random.Generator,
    family_name: str,
) -> np.ndarray:
    child_h, child_w = child_patch.shape
    for _ in range(80):
        band = int(rng.integers(*WRAP_BAND_RANGE))
        slack_h = int(rng.integers(*WRAP_SLACK_RANGE))
        slack_w = int(rng.integers(*WRAP_SLACK_RANGE))
        h = child_h + 2 * band + slack_h
        w = child_w + 2 * band + slack_w
        if h > GRID_MAX_DIM or w > GRID_MAX_DIM:
            continue
        mask, _family = _make_shape_mask(h, w, band=band, rng=rng, family_name=family_name)
        safe_h = h - 2 * band
        safe_w = w - 2 * band
        if safe_h < child_h or safe_w < child_w:
            continue
        r0 = band + int(rng.integers(0, safe_h - child_h + 1))
        c0 = band + int(rng.integers(0, safe_w - child_w + 1))
        if not np.all(mask[r0 : r0 + child_h, c0 : c0 + child_w]):
            continue

        patch = _render_ring(mask)
        child_non_bg = child_patch != BG
        child_view = patch[r0 : r0 + child_h, c0 : c0 + child_w]
        child_view[child_non_bg] = child_patch[child_non_bg]
        return patch
    raise AssertionError("failed to wrap child patch")


def _family_sequence(depth: int, rng: np.random.Generator) -> list[str]:
    names = [name for name, _func in SHAPE_FAMILIES]
    if depth <= len(names):
        choice = rng.choice(names, size=depth, replace=False)
        return [str(x) for x in choice.tolist()]
    out = []
    prev = None
    for _ in range(depth):
        options = [name for name in names if name != prev]
        cur = str(options[int(rng.integers(0, len(options)))])
        out.append(cur)
        prev = cur
    return out


def _build_nested_patch(depth: int, marker_color: int, rng: np.random.Generator) -> np.ndarray:
    families = _family_sequence(depth, rng)
    patch = _make_deepest_patch(rng, marker_color, families[-1])
    for family_name in reversed(families[:-1]):
        patch = _wrap_patch(patch, rng, family_name)
    return patch


def _embed_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ph, pw = patch.shape
    for _ in range(60):
        mt = int(rng.integers(*GRID_MARGIN_RANGE))
        mb = int(rng.integers(*GRID_MARGIN_RANGE))
        ml = int(rng.integers(*GRID_MARGIN_RANGE))
        mr = int(rng.integers(*GRID_MARGIN_RANGE))
        h = ph + mt + mb
        w = pw + ml + mr
        if h > GRID_MAX_DIM or w > GRID_MAX_DIM:
            continue
        grid = np.full((h, w), BG, dtype=int)
        non_bg = patch != BG
        patch_view = grid[mt : mt + ph, ml : ml + pw]
        patch_view[non_bg] = patch[non_bg]
        return grid
    raise AssertionError("failed to embed patch on canvas")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        depth = int(rng.integers(MIN_DEPTH, MAX_DEPTH + 1))
        marker_color = int(MARKER_COLORS[int(rng.integers(0, len(MARKER_COLORS)))])
        patch = _build_nested_patch(depth, marker_color, rng)
        grid = _embed_patch(patch, rng)

        try:
            info = _analyze_input(grid)
            if int(info["deepest_depth"]) != depth:
                continue

            out = main(grid)
            change_count = int(np.count_nonzero(out != grid))
            if change_count < 12:
                continue
            if int(np.count_nonzero(out == marker_color)) < 6:
                continue
            fill_cells = sum(int(np.count_nonzero(out == DEPTH_FILL[d])) for d in range(1, depth + 1))
            if fill_cells < 10:
                continue
            return grid
        except AssertionError:
            continue
    raise RuntimeError("generate_input failed to produce a valid sample")
