from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8
MAROON = 9

PALETTE = np.array([YELLOW, GREY, RED, TEAL, MAROON], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(g == int(color)):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        cells = []
        while q:
            cr, cc2 = q.popleft()
            cells.append((cr, cc2))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not seen[nr, nc] and g[nr, nc] == int(color):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        coords = np.array(cells, dtype=int)
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "area": int(len(cells)),
            }
        )
    return comps


def _interior_mask(boundary_mask: np.ndarray) -> np.ndarray:
    h, w = boundary_mask.shape
    passable = ~boundary_mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return passable & ~outside


def _degree_two_cycle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    for r, c in coords:
        deg = 0
        for dr, dc in DIR4:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                deg += 1
        if deg != 2:
            return False
    return len(_components_of_color(mask.astype(int), 1)) == 1


def _center_pixel(interior_mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(interior_mask)
    centroid = coords.mean(axis=0)
    d2 = ((coords - centroid) ** 2).sum(axis=1)
    order = np.lexsort((coords[:, 1], coords[:, 0], d2))
    chosen = coords[int(order[0])]
    return int(chosen[0]), int(chosen[1])


def _marker_side_and_count(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, int]:
    r0, c0, r1, c1 = bbox
    counts = {
        "top": int(np.sum(grid[r0 - 1, c0 : c1 + 1] == RED)) if r0 > 0 else 0,
        "bottom": int(np.sum(grid[r1 + 1, c0 : c1 + 1] == RED)) if r1 + 1 < grid.shape[0] else 0,
        "left": int(np.sum(grid[r0 : r1 + 1, c0 - 1] == RED)) if c0 > 0 else 0,
        "right": int(np.sum(grid[r0 : r1 + 1, c1 + 1] == RED)) if c1 + 1 < grid.shape[1] else 0,
    }
    sides = [side for side, count in counts.items() if count > 0]
    assert len(sides) == 1, f"expected exactly one marker side, got {counts}"
    side = sides[0]
    k = counts[side]
    assert 1 <= k <= 4
    return side, k


def _reflect(mask: np.ndarray, side: str, k: int, axis_mode: str) -> np.ndarray:
    use_horizontal = (k % 2 == 1) if axis_mode == "parity" else side in {"top", "bottom"}
    return np.flipud(mask) if use_horizontal else np.fliplr(mask)


def _replica_origin(bbox: tuple[int, int, int, int], patch_shape: tuple[int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h, w = patch_shape
    if side == "top":
        return int(r1 + 2), int(c0)
    if side == "bottom":
        return int(r0 - h - 1), int(c0)
    if side == "left":
        return int(r0), int(c1 + 2)
    return int(r0), int(c0 - w - 1)


def _paint_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            gr = top + r
            gc = left + c
            if 0 <= gr < dst.shape[0] and 0 <= gc < dst.shape[1] and dst[gr, gc] == BLACK:
                dst[gr, gc] = int(color)


def _paint_replica_fill(
    dst: np.ndarray,
    interior_mask: np.ndarray,
    top: int,
    left: int,
    k: int,
    palette_mode: str,
) -> None:
    coords = np.argwhere(interior_mask)
    if coords.size == 0:
        return
    center_r, center_c = _center_pixel(interior_mask)
    offset = int(k) if palette_mode == "k" else 1
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr == center_r and cc == center_c:
            continue
        gr = top + rr
        gc = left + cc
        if 0 <= gr < dst.shape[0] and 0 <= gc < dst.shape[1] and dst[gr, gc] == BLACK:
            d = abs(rr - center_r) + abs(cc - center_c)
            dst[gr, gc] = int(PALETTE[(d - 1 + offset) % len(PALETTE)])


def _transform(grid: np.ndarray, axis_mode: str = "parity", palette_mode: str = "k") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _components_of_color(g, GREEN)
    assert 2 <= len(components) <= 3

    for comp in components:
        bbox = comp["bbox"]
        patch = _crop(g, bbox)
        boundary_mask = patch == GREEN
        assert _degree_two_cycle(boundary_mask), "green object must be a one-cell-thick closed cycle"
        interior_mask = _interior_mask(boundary_mask)
        assert int(interior_mask.sum()) >= 1, "outline must enclose an interior"

        side, k = _marker_side_and_count(g, bbox)
        reflected_boundary = _reflect(boundary_mask, side, k, axis_mode)
        reflected_interior = _reflect(interior_mask, side, k, axis_mode)
        top, left = _replica_origin(bbox, patch.shape, side)

        _paint_mask(out, reflected_boundary, top, left, BLUE)
        _paint_replica_fill(out, reflected_interior, top, left, k, palette_mode)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the outside red markers for each green outline, then place an opposite-side mirror copy whose axis and interior ring colors both depend on that count.

    Input structure:
    - The grid is a black background containing 2-3 green one-cell-thick closed outline objects.
    - Each green outline has exactly one adjacent red marker line on one outer bbox side, with 1-4 red pixels running parallel to that side.
    - Outline interiors are black, objects are well separated, and no other colors appear in the input.

    Rule:
    - For each green outline, count its adjacent red markers to get k.
    - Place a blue mirrored copy on the side opposite the markers with a one-cell gap; the opposite side is determined by the marker side, while the mirror axis is determined by parity: odd k flips vertically across a horizontal axis, even k flips horizontally across a vertical axis.
    - Fill the copy's interior from the nearest interior pixel to the interior centroid using Manhattan-distance rings; the centroid pixel stays black, and every other interior pixel gets palette [4,5,2,8,9][(d-1+k) mod 5].
    - Keep the original green outlines and red markers unchanged, and only paint replica pixels onto black background cells, clipping anything that would fall outside the grid.

    Key insight:
    - The same count k is consumed twice: parity chooses the mirror axis, while the absolute value shifts the cyclic interior palette, so neither branch can be inferred away.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREEN}
    return _transform(g, axis_mode="parity", palette_mode="k")


def _path_mask(vertices: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in vertices]
    cols = [c for _, c in vertices]
    r0 = min(rows)
    c0 = min(cols)
    shifted = [(r - r0, c - c0) for r, c in vertices]
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for (r0, c0), (r1, c1) in zip(shifted, shifted[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                mask[r0, c] = True
        else:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                mask[r, c0] = True
    coords = np.argwhere(mask)
    rr0, cc0, rr1, cc1 = _bbox_from_coords(coords)
    return mask[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _shape_variants() -> list[np.ndarray]:
    base_paths = [
        [(0, 0), (0, 3), (2, 3), (2, 2), (3, 2), (3, 0), (0, 0)],
        [(0, 1), (0, 3), (3, 3), (3, 0), (2, 0), (2, 1), (0, 1)],
        [(0, 0), (0, 2), (1, 2), (1, 3), (3, 3), (3, 0), (0, 0)],
    ]
    out: list[np.ndarray] = []
    seen = set()
    for path in base_paths:
        base = _path_mask(path)
        for rot in range(4):
            turned = np.rot90(base, rot)
            for candidate in (turned, np.fliplr(turned)):
                key = (candidate.shape, candidate.astype(np.uint8).tobytes())
                if key in seen:
                    continue
                seen.add(key)
                interior = _interior_mask(candidate)
                if not _degree_two_cycle(candidate):
                    continue
                if not (6 <= int(candidate.sum()) <= 12):
                    continue
                if int(interior.sum()) < 2:
                    continue
                if np.array_equal(candidate, np.flipud(candidate)):
                    continue
                if np.array_equal(candidate, np.fliplr(candidate)):
                    continue
                out.append(candidate.copy())
    assert len(out) >= 4
    return out


GEN_SHAPES = _shape_variants()
GEN_ROWS = (18, 25)
GEN_COLS = (20, 27)
GEN_ATTEMPTS = 600
GEN_BBOX_GAP = 5
GEN_PAIR_GAP = 2


def _bbox_min_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if a[2] < b[0]:
        dr = b[0] - a[2] - 1
    elif b[2] < a[0]:
        dr = a[0] - b[2] - 1
    else:
        dr = 0
    if a[3] < b[1]:
        dc = b[1] - a[3] - 1
    elif b[3] < a[1]:
        dc = a[1] - b[3] - 1
    else:
        dc = 0
    if dr == 0:
        return int(dc)
    if dc == 0:
        return int(dr)
    return int(dr + dc)


def _union_bbox(*bboxes: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0 = min(b[0] for b in bboxes)
    c0 = min(b[1] for b in bboxes)
    r1 = max(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_positions(
    bbox: tuple[int, int, int, int],
    side: str,
    start: int,
    k: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    if side == "top":
        row = r0 - 1
        cells = [(row, c0 + start + i) for i in range(k)]
    elif side == "bottom":
        row = r1 + 1
        cells = [(row, c0 + start + i) for i in range(k)]
    elif side == "left":
        col = c0 - 1
        cells = [(r0 + start + i, col) for i in range(k)]
    else:
        col = c1 + 1
        cells = [(r0 + start + i, col) for i in range(k)]
    coords = np.array(cells, dtype=int)
    return cells, _bbox_from_coords(coords)


def _replica_colored_cells(
    boundary_mask: np.ndarray,
    side: str,
    k: int,
    bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    interior_mask = _interior_mask(boundary_mask)
    refl_boundary = _reflect(boundary_mask, side, k, "parity")
    refl_interior = _reflect(interior_mask, side, k, "parity")
    top, left = _replica_origin(bbox, boundary_mask.shape, side)
    colored = []
    for r, c in np.argwhere(refl_boundary):
        colored.append((top + int(r), left + int(c)))
    center = _center_pixel(refl_interior)
    for r, c in np.argwhere(refl_interior):
        rr = int(r)
        cc = int(c)
        if (rr, cc) != center:
            colored.append((top + rr, left + cc))
    coords = np.array(colored, dtype=int)
    return colored, _bbox_from_coords(coords)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _choose_specs() -> list[dict]:
    n = random.choice([2, 3])
    odd_k = random.choice([1, 3])
    even_k = random.choice([2, 4])
    odd_side = random.choice(["left", "right"])
    even_side = random.choice(["top", "bottom"])

    specs = [
        {"k": odd_k, "side": odd_side},
        {"k": even_k, "side": even_side},
    ]
    if n == 3:
        remaining_counts = [k for k in [1, 2, 3, 4] if k not in {odd_k, even_k}]
        remaining_sides = [side for side in SIDES if side not in {odd_side, even_side}]
        specs.append({"k": random.choice(remaining_counts), "side": random.choice(remaining_sides)})

    shape_choices = random.sample(GEN_SHAPES, len(specs))
    for spec, shape in zip(specs, shape_choices):
        spec["shape"] = shape.copy()
        spec["shape_key"] = _shape_key(shape)
        span = int(shape.shape[1] if spec["side"] in {"top", "bottom"} else shape.shape[0])
        assert spec["k"] <= span
    random.shuffle(specs)
    return specs


def _candidate_placements(
    grid_shape: tuple[int, int],
    spec: dict,
    occupied: np.ndarray,
    object_bboxes: list[tuple[int, int, int, int]],
    pair_bboxes: list[tuple[int, int, int, int]],
) -> list[dict]:
    H, W = grid_shape
    mask = spec["shape"]
    h, w = mask.shape
    side = spec["side"]
    k = spec["k"]
    span = int(w if side in {"top", "bottom"} else h)
    out = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_bbox_min_gap(bbox, other) < GEN_BBOX_GAP for other in object_bboxes):
                continue
            for start in range(span - k + 1):
                markers, marker_bbox = _marker_positions(bbox, side, start, k)
                replica_cells, replica_bbox = _replica_colored_cells(mask, side, k, bbox)
                original_cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]
                all_cells = original_cells + markers + replica_cells
                if any(not (0 <= r < H and 0 <= c < W) for r, c in all_cells):
                    continue
                if any(occupied[r, c] for r, c in all_cells):
                    continue
                pair_bbox = _union_bbox(bbox, marker_bbox, replica_bbox)
                if any(_bbox_min_gap(pair_bbox, other) < GEN_PAIR_GAP for other in pair_bboxes):
                    continue
                out.append(
                    {
                        "bbox": bbox,
                        "marker_cells": markers,
                        "pair_bbox": pair_bbox,
                    }
                )
    return out


def _semantic_checks(inp: np.ndarray) -> bool:
    comps = _components_of_color(inp, GREEN)
    if len(comps) not in {2, 3}:
        return False

    saw_odd_lr = False
    saw_even_tb = False
    has_odd = False
    has_even = False
    seen_keys = set()
    bboxes = []
    for comp in comps:
        bbox = comp["bbox"]
        bboxes.append(bbox)
        patch = _crop(inp, bbox) == GREEN
        if _shape_key(patch) in seen_keys:
            return False
        seen_keys.add(_shape_key(patch))
        if not _degree_two_cycle(patch):
            return False
        interior = _interior_mask(patch)
        if int(interior.sum()) < 2:
            return False
        if np.array_equal(patch, np.flipud(patch)) or np.array_equal(patch, np.fliplr(patch)):
            return False
        side, k = _marker_side_and_count(inp, bbox)
        if k % 2 == 1:
            has_odd = True
            if side in {"left", "right"}:
                saw_odd_lr = True
        else:
            has_even = True
            if side in {"top", "bottom"}:
                saw_even_tb = True

    for i, bbox_a in enumerate(bboxes):
        for bbox_b in bboxes[i + 1 :]:
            if _bbox_min_gap(bbox_a, bbox_b) < GEN_BBOX_GAP:
                return False

    if not (has_odd and has_even and saw_odd_lr and saw_even_tb):
        return False

    correct = _transform(inp, axis_mode="parity", palette_mode="k")
    if np.array_equal(correct, inp):
        return False
    if np.array_equal(correct, _transform(inp, axis_mode="side", palette_mode="k")):
        return False
    if np.array_equal(correct, _transform(inp, axis_mode="parity", palette_mode="fixed1")):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        H = random.randint(*GEN_ROWS)
        W = random.randint(*GEN_COLS)
        grid = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        object_bboxes: list[tuple[int, int, int, int]] = []
        pair_bboxes: list[tuple[int, int, int, int]] = []
        ok = True

        for spec in _choose_specs():
            candidates = _candidate_placements((H, W), spec, occupied, object_bboxes, pair_bboxes)
            if not candidates:
                ok = False
                break
            chosen = random.choice(candidates)
            bbox = chosen["bbox"]
            r0, c0, _r1, _c1 = bbox
            mask = spec["shape"]
            for r, c in np.argwhere(mask):
                grid[r0 + int(r), c0 + int(c)] = GREEN
                occupied[r0 + int(r), c0 + int(c)] = True
            for r, c in chosen["marker_cells"]:
                grid[r, c] = RED
                occupied[r, c] = True
            object_bboxes.append(bbox)
            pair_bboxes.append(chosen["pair_bbox"])

        if not ok:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if not _semantic_checks(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
