from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(coords)
    return labels, components


def _extract_palette(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row = grid[0]
    cols = [int(c) for c in np.where((row != BLACK) & (row != GREY))[0]]
    assert len(cols) == 3, "expected exactly three palette pixels on row 0"
    assert cols[1] == cols[0] + 1 and cols[2] == cols[1] + 1, "palette must be contiguous"
    palette = [int(row[c]) for c in cols]
    return palette, cols


def _boundary_mask(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros_like(filled, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(filled.shape, nr, nc) or not filled[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _is_simple_cycle(mask: np.ndarray) -> bool:
    labels, comps = _label_components(mask)
    if len(comps) != 1:
        return False
    pts = comps[0]
    for r, c in pts:
        degree = 0
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape, nr, nc) and labels[nr, nc] == 1:
                degree += 1
        if degree != 2:
            return False
    return True


def _analyze_structure(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette, palette_cols = _extract_palette(g)
    black_labels, black_components = _label_components(g == BLACK)
    grey_labels, grey_components = _label_components(g == GREY)
    assert grey_components, "expected at least one grey boundary"

    for label_id, coords in enumerate(grey_components, start=1):
        local = np.zeros(g.shape, dtype=bool)
        for r, c in coords:
            local[r, c] = True
        assert _is_simple_cycle(local), "each grey component must be a simple 4-connected loop"

    adjacency: dict[int, set[int]] = {idx: set() for idx in range(1, len(black_components) + 1)}
    for coords in grey_components:
        touching: set[int] = set()
        for r, c in coords:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if _in_bounds(g.shape, nr, nc) and black_labels[nr, nc] > 0:
                    touching.add(int(black_labels[nr, nc]))
        assert len(touching) == 2, "each boundary loop must separate exactly two black regions"
        a, b = sorted(touching)
        adjacency[a].add(b)
        adjacency[b].add(a)

    exterior = {
        int(black_labels[r, c])
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if black_labels[r, c] > 0 and (r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1))
    }
    assert len(exterior) == 1, "expected a single exterior black region"
    root = next(iter(exterior))

    depths: dict[int, int] = {root: 0}
    q: deque[int] = deque([root])
    while q:
        cur = q.popleft()
        for nxt in sorted(adjacency[cur]):
            if nxt not in depths:
                depths[nxt] = depths[cur] + 1
                q.append(nxt)
    assert len(depths) == len(black_components), "all black regions must be reachable from exterior"

    positive_depths = sorted(d for d in depths.values() if d > 0)
    assert positive_depths == list(range(1, len(grey_components) + 1)), "depths must form consecutive nesting levels"

    return {
        "palette": palette,
        "palette_cols": palette_cols,
        "black_labels": black_labels,
        "black_components": black_components,
        "grey_labels": grey_labels,
        "grey_components": grey_components,
        "adjacency": adjacency,
        "depths": depths,
        "root": root,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed black region by the cyclic palette index determined by its nested-boundary depth.

    Input structure:
    - The grid has a black background with a 3-cell palette on row 0 and one or more grey 4-connected closed loops elsewhere.
    - Grey pixels form 2 or 3 nested, non-touching boundary rings with at least one black cell of separation.
    - Black cells are partitioned into connected regions by those boundary rings.

    Rule:
    - Keep the row-0 palette cells and every grey boundary pixel unchanged.
    - Treat each grey loop as a separator between two black 4-connected regions and compute every black region's depth by counting how many loops must be crossed to reach the exterior.
    - Leave depth-0 black cells unchanged, and fill each depth-d region with the palette color at index (d-1) mod 3, read left-to-right from the palette.

    Key insight:
    - The fill color is controlled only by topological nesting depth, so the solver must first recover the boundary-separation graph before it can choose the palette color for each interior.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_structure(g)
    palette = info["palette"]
    black_labels = info["black_labels"]
    depths = info["depths"]

    out = g.copy()
    for comp_id, depth in depths.items():
        if depth == 0:
            continue
        out[black_labels == comp_id] = palette[(depth - 1) % len(palette)]
    return out


def _apply_notch(filled: np.ndarray, side: str, start: int, end: int) -> None:
    if side == "top":
        filled[0, start : end + 1] = False
    elif side == "bottom":
        filled[-1, start : end + 1] = False
    elif side == "left":
        filled[start : end + 1, 0] = False
    else:
        filled[start : end + 1, -1] = False


def _make_loop_mask(height: int, width: int) -> np.ndarray | None:
    filled = np.ones((height, width), dtype=bool)
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    notch_count = random.randint(0, min(3, len(sides)))
    chosen = sides[:notch_count]

    for side in chosen:
        if side in ("top", "bottom"):
            if width < 5:
                continue
            start = random.randint(1, width - 3)
            end = random.randint(start, width - 2)
        else:
            if height < 5:
                continue
            start = random.randint(1, height - 3)
            end = random.randint(start, height - 2)
        _apply_notch(filled, side, start, end)

    boundary = _boundary_mask(filled)
    area = int(np.count_nonzero(boundary))
    if area < 12 or area > 56:
        return None
    if not _is_simple_cycle(boundary):
        return None
    return boundary


def _pick_bbox(
    available: tuple[int, int, int, int],
    remaining_layers: int,
) -> tuple[int, int, int, int] | None:
    ar0, ac0, ar1, ac1 = available
    avail_h = ar1 - ar0 + 1
    avail_w = ac1 - ac0 + 1
    min_side = 4 + 4 * (remaining_layers - 1)
    if avail_h < min_side or avail_w < min_side:
        return None

    max_h = min(avail_h, min_side + 4)
    max_w = min(avail_w, min_side + 4)
    height = random.randint(min_side, max_h)
    width = random.randint(min_side, max_w)
    r0 = random.randint(ar0, ar1 - height + 1)
    c0 = random.randint(ac0, ac1 - width + 1)
    return r0, c0, r0 + height - 1, c0 + width - 1


def _render_candidate() -> np.ndarray | None:
    layer_count = random.choices([2, 3], weights=[1, 5], k=1)[0]
    height = random.randint(18 if layer_count == 3 else 16, 24)
    width = random.randint(18 if layer_count == 3 else 16, 24)
    grid = np.full((height, width), BLACK, dtype=int)

    palette_start = random.randint(1, width - 4)
    grid[0, palette_start : palette_start + 3] = np.array(PALETTE_COLORS, dtype=int)

    available = (2, 1, height - 2, width - 2)
    irregular = False
    for depth_idx in range(layer_count):
        bbox = _pick_bbox(available, layer_count - depth_idx)
        if bbox is None:
            return None
        r0, c0, r1, c1 = bbox
        local = _make_loop_mask(r1 - r0 + 1, c1 - c0 + 1)
        if local is None:
            return None
        irregular = irregular or int(np.count_nonzero(local[0])) != local.shape[1]
        patch = np.where(local, GREY, BLACK)
        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((patch == GREY) & (view != BLACK)):
            return None
        view[patch == GREY] = GREY
        rectangle_perimeter = 2 * (local.shape[0] + local.shape[1]) - 4
        irregular = irregular or int(np.count_nonzero(local)) != rectangle_perimeter
        available = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)

    if not irregular and random.random() < 0.75:
        return None
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    info = _analyze_structure(grid)
    grey_components = info["grey_components"]
    depths = info["depths"]
    adjacency = info["adjacency"]

    if len(grey_components) not in (2, 3):
        return False
    degree_multiset = sorted(len(neigh) for neigh in adjacency.values())
    expected = [1, 1] + [2] * (len(adjacency) - 2)
    if degree_multiset != expected:
        return False
    if max(depths.values()) != len(grey_components):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[grid == GREY], grid[grid == GREY]):
        return False

    palette = info["palette"]
    black_labels = info["black_labels"]
    for comp_id, depth in depths.items():
        mask = black_labels == comp_id
        colors = set(map(int, out[mask]))
        if depth == 0:
            if colors != {BLACK}:
                return False
        else:
            expected_color = palette[(depth - 1) % len(palette)]
            if colors != {expected_color}:
                return False

    positive_colors = {
        palette[(depth - 1) % len(palette)]
        for depth in depths.values()
        if depth > 0
    }
    if len(positive_colors) < min(len(grey_components), len(PALETTE_COLORS)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(2000):
        grid = _render_candidate()
        if grid is None:
            continue
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid nested-boundary ARC input")
