import numpy as np
from collections import deque


BG = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACE_ATTEMPTS = 160
PLACEMENT_GAP = 1
EXTRA_COMPONENTS_RANGE = (0, 2)
MODES = ("separate", "dominance", "tie", "mixed")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "color": color,
                    "size": len(cells),
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _rect_outside_distance(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return int(max(dr, dc))


def _claims_pixel(comp: dict, r: int, c: int) -> bool:
    d = _rect_outside_distance(r, c, comp["bbox"])
    return 1 <= d <= int(comp["size"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed cluster into rectangular ring bands whose radius equals the seed's pixel count.

    Input structure:
    - A black background contains several small 4-connected monochrome seed clusters.
    - Each cluster has a unique color, contains 2-6 pixels, and is separated from the others.
    - Every cluster has a tight axis-aligned bounding box that may contain background gaps.

    Rule:
    - For each cluster, count its pixels as N and compute its tight bounding box.
    - Keep the original seed pixels unchanged.
    - Paint every background pixel whose rectangular outside-distance from that box is between 1 and N, using the seed's color; this forms N outward rectangular ring layers around the box.
    - If multiple clusters claim the same background pixel, the cluster with larger N wins. If the largest N is tied, that pixel stays black.
    - Growth beyond the grid is clipped by the grid boundary.

    Key insight:
    The seed size controls the number of rectangular growth layers, and overlap resolution depends only on comparing those sizes, not on the colors or exact seed shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    comps = _find_components(g, bg)

    out = g.copy()
    for r, c in np.argwhere(g == bg):
        rr = int(r)
        cc = int(c)
        best_size = -1
        best_color = bg
        tied = False
        for comp in comps:
            if not _claims_pixel(comp, rr, cc):
                continue
            size = int(comp["size"])
            if size > best_size:
                best_size = size
                best_color = int(comp["color"])
                tied = False
            elif size == best_size:
                tied = True
        if best_size >= 1 and not tied:
            out[rr, cc] = best_color
        else:
            out[rr, cc] = bg
    return out


def _random_shape(rng: np.random.Generator, size: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != size:
            continue
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if max(h, w) > 4:
            continue
        if size > 2 and h * w == size:
            continue
        if size >= 4 and (h == 1 or w == 1):
            continue
        return np.array(norm, dtype=int)
    raise RuntimeError(f"failed to sample connected shape of size {size}")


def _make_spec(rng: np.random.Generator, size: int | None = None) -> dict:
    if size is None:
        size = int(rng.integers(2, 7))
    cells = _random_shape(rng, int(size))
    h = int(cells[:, 0].max()) + 1
    w = int(cells[:, 1].max()) + 1
    return {"size": int(size), "cells": cells, "shape": (h, w)}


def _bbox_for(spec: dict, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = spec["shape"]
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _materialize(spec: dict, top: int, left: int) -> dict:
    cells = [(int(top + r), int(left + c)) for r, c in spec["cells"]]
    return {
        "size": int(spec["size"]),
        "shape": spec["shape"],
        "cells": cells,
        "bbox": _bbox_for(spec, top, left),
    }


def _boxes_clear(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _safe_range(limit: int, span: int, radius: int) -> tuple[int, int] | None:
    lo = radius
    hi = limit - span - radius
    if lo > hi:
        return None
    return int(lo), int(hi)


def _can_place(spec: dict, top: int, left: int, h: int, w: int, placed: list[dict], unclipped: bool) -> bool:
    box = _bbox_for(spec, top, left)
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    if unclipped:
        radius = int(spec["size"])
        if r0 - radius < 0 or c0 - radius < 0 or r1 + radius >= h or c1 + radius >= w:
            return False
    for other in placed:
        if not _boxes_clear(box, other["bbox"], gap=PLACEMENT_GAP):
            return False
    return True


def _random_safe_position(rng: np.random.Generator, h: int, w: int, spec: dict) -> tuple[int, int] | None:
    bounds_r = _safe_range(h, spec["shape"][0], int(spec["size"]))
    bounds_c = _safe_range(w, spec["shape"][1], int(spec["size"]))
    if bounds_r is None or bounds_c is None:
        return None
    top = int(rng.integers(bounds_r[0], bounds_r[1] + 1))
    left = int(rng.integers(bounds_c[0], bounds_c[1] + 1))
    return top, left


def _random_edge_position(rng: np.random.Generator, h: int, w: int, spec: dict) -> tuple[int, int] | None:
    sh, sw = spec["shape"]
    radius = int(spec["size"])
    if sh > h or sw > w:
        return None
    side = str(rng.choice(["top", "bottom", "left", "right"]))
    safe_r = _safe_range(h, sh, radius)
    safe_c = _safe_range(w, sw, radius)
    if side == "top":
        top = int(rng.integers(0, min(radius, h - sh) + 1))
        if safe_c is not None:
            left = int(rng.integers(safe_c[0], safe_c[1] + 1))
        else:
            left = int(rng.integers(0, w - sw + 1))
    elif side == "bottom":
        low = max(0, h - sh - radius)
        top = int(rng.integers(low, h - sh + 1))
        if safe_c is not None:
            left = int(rng.integers(safe_c[0], safe_c[1] + 1))
        else:
            left = int(rng.integers(0, w - sw + 1))
    elif side == "left":
        left = int(rng.integers(0, min(radius, w - sw) + 1))
        if safe_r is not None:
            top = int(rng.integers(safe_r[0], safe_r[1] + 1))
        else:
            top = int(rng.integers(0, h - sh + 1))
    else:
        low = max(0, w - sw - radius)
        left = int(rng.integers(low, w - sw + 1))
        if safe_r is not None:
            top = int(rng.integers(safe_r[0], safe_r[1] + 1))
        else:
            top = int(rng.integers(0, h - sh + 1))
    return top, left


def _claim_stats(h: int, w: int, placed: list[dict]) -> dict:
    seed_mask = np.zeros((h, w), dtype=bool)
    for comp in placed:
        for r, c in comp["cells"]:
            seed_mask[r, c] = True

    dominance = 0
    tie = 0
    claimed = 0
    for r in range(h):
        for c in range(w):
            if seed_mask[r, c]:
                continue
            best_size = -1
            best_count = 0
            hit = False
            for comp in placed:
                d = _rect_outside_distance(r, c, comp["bbox"])
                if 1 <= d <= int(comp["size"]):
                    hit = True
                    size = int(comp["size"])
                    if size > best_size:
                        best_size = size
                        best_count = 1
                    elif size == best_size:
                        best_count += 1
            if hit:
                claimed += 1
                if best_count >= 2:
                    tie += 1
                elif best_count == 1 and any(
                    1 <= _rect_outside_distance(r, c, comp["bbox"]) <= int(comp["size"]) for comp in placed
                ):
                    contenders = sum(
                        1 <= _rect_outside_distance(r, c, comp["bbox"]) <= int(comp["size"]) for comp in placed
                    )
                    if contenders >= 2:
                        dominance += 1

    edge_clipped = any(
        comp["bbox"][0] - int(comp["size"]) < 0
        or comp["bbox"][1] - int(comp["size"]) < 0
        or comp["bbox"][2] + int(comp["size"]) >= h
        or comp["bbox"][3] + int(comp["size"]) >= w
        for comp in placed
    )
    return {"dominance": dominance, "tie": tie, "claimed": claimed, "edge_clipped": edge_clipped}


def _validate_mode(mode: str, clip_required: bool, stats: dict) -> bool:
    if stats["claimed"] == 0:
        return False
    if clip_required != bool(stats["edge_clipped"]):
        return False
    if mode == "separate":
        return stats["dominance"] == 0 and stats["tie"] == 0
    if mode == "dominance":
        return stats["dominance"] > 0 and stats["tie"] == 0
    if mode == "tie":
        return stats["tie"] > 0 and stats["dominance"] == 0
    return stats["dominance"] > 0 and stats["tie"] > 0


def _place_random_spec(
    rng: np.random.Generator, h: int, w: int, spec: dict, placed: list[dict], clipped: bool
) -> dict | None:
    for _ in range(PLACE_ATTEMPTS):
        pos = _random_edge_position(rng, h, w, spec) if clipped else _random_safe_position(rng, h, w, spec)
        if pos is None:
            return None
        top, left = pos
        if _can_place(spec, top, left, h, w, placed, unclipped=not clipped):
            return _materialize(spec, top, left)
    return None


def _place_overlap_pair(rng: np.random.Generator, h: int, w: int, spec_a: dict, spec_b: dict, want_tie: bool) -> list[dict] | None:
    for _ in range(PLACE_ATTEMPTS):
        pos_a = _random_safe_position(rng, h, w, spec_a)
        if pos_a is None:
            return None
        a = _materialize(spec_a, pos_a[0], pos_a[1])
        for _ in range(PLACE_ATTEMPTS):
            horizontal = bool(rng.integers(2))
            gap_hi = max(1, min(4, int(spec_a["size"]) + int(spec_b["size"]) - 1))
            gap = int(rng.integers(1, gap_hi + 1))
            jitter = int(rng.integers(-min(int(spec_a["size"]), int(spec_b["size"])), min(int(spec_a["size"]), int(spec_b["size"])) + 1))
            sign = -1 if bool(rng.integers(2)) else 1
            if horizontal:
                top_b = a["bbox"][0] + jitter
                if sign > 0:
                    left_b = a["bbox"][3] + gap + 1
                else:
                    left_b = a["bbox"][1] - gap - spec_b["shape"][1]
            else:
                left_b = a["bbox"][1] + jitter
                if sign > 0:
                    top_b = a["bbox"][2] + gap + 1
                else:
                    top_b = a["bbox"][0] - gap - spec_b["shape"][0]
            if not _can_place(spec_b, top_b, left_b, h, w, [a], unclipped=True):
                continue
            b = _materialize(spec_b, top_b, left_b)
            stats = _claim_stats(h, w, [a, b])
            if want_tie and stats["tie"] > 0 and stats["dominance"] == 0:
                return [a, b]
            if not want_tie and stats["dominance"] > 0 and stats["tie"] == 0:
                return [a, b]
    return None


def _place_mixed_triplet(rng: np.random.Generator, h: int, w: int) -> list[dict] | None:
    tie_size = int(rng.integers(3, 6))
    big_size = int(rng.integers(tie_size + 1, 7))
    spec_a = _make_spec(rng, tie_size)
    spec_b = _make_spec(rng, tie_size)
    spec_c = _make_spec(rng, big_size)
    for _ in range(PLACE_ATTEMPTS):
        pair = _place_overlap_pair(rng, h, w, spec_a, spec_b, want_tie=True)
        if pair is None:
            return None
        anchor = pair[int(rng.integers(2))]
        for _ in range(PLACE_ATTEMPTS):
            horizontal = bool(rng.integers(2))
            gap_hi = max(1, min(4, big_size + anchor["size"] - 1))
            gap = int(rng.integers(1, gap_hi + 1))
            jitter = int(rng.integers(-anchor["size"], anchor["size"] + 1))
            sign = -1 if bool(rng.integers(2)) else 1
            if horizontal:
                top_c = anchor["bbox"][0] + jitter
                if sign > 0:
                    left_c = anchor["bbox"][3] + gap + 1
                else:
                    left_c = anchor["bbox"][1] - gap - spec_c["shape"][1]
            else:
                left_c = anchor["bbox"][1] + jitter
                if sign > 0:
                    top_c = anchor["bbox"][2] + gap + 1
                else:
                    top_c = anchor["bbox"][0] - gap - spec_c["shape"][0]
            if not _can_place(spec_c, top_c, left_c, h, w, pair, unclipped=True):
                continue
            placed = pair + [_materialize(spec_c, top_c, left_c)]
            stats = _claim_stats(h, w, placed)
            if stats["tie"] > 0 and stats["dominance"] > 0 and not stats["edge_clipped"]:
                return placed
    return None


def _render_grid(h: int, w: int, placed: list[dict], colors: np.ndarray) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for comp, color in zip(placed, colors):
        for r, c in comp["cells"]:
            grid[r, c] = int(color)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    mode = str(rng.choice(MODES))
    clip_required = bool(rng.integers(2))

    if mode == "separate":
        count = int(rng.integers(2, 5))
        base_specs = [_make_spec(rng) for _ in range(count)]
        placed: list[dict] = []
        for spec in base_specs:
            ok = False
            for _ in range(PLACE_ATTEMPTS):
                comp = _place_random_spec(rng, h, w, spec, placed, clipped=False)
                if comp is None:
                    break
                temp = placed + [comp]
                stats = _claim_stats(h, w, temp)
                if stats["dominance"] == 0 and stats["tie"] == 0 and not stats["edge_clipped"]:
                    placed = temp
                    ok = True
                    break
            if not ok:
                return None
    elif mode == "dominance":
        sizes = sorted(rng.choice(np.arange(2, 7), size=2, replace=False))
        pair = _place_overlap_pair(rng, h, w, _make_spec(rng, int(sizes[1])), _make_spec(rng, int(sizes[0])), want_tie=False)
        if pair is None:
            return None
        placed = pair
        extras = int(rng.integers(EXTRA_COMPONENTS_RANGE[0], EXTRA_COMPONENTS_RANGE[1] + 1))
        for _ in range(extras):
            spec = _make_spec(rng)
            comp = _place_random_spec(rng, h, w, spec, placed, clipped=False)
            if comp is None:
                continue
            temp = placed + [comp]
            stats = _claim_stats(h, w, temp)
            if stats["dominance"] > 0 and stats["tie"] == 0 and not stats["edge_clipped"]:
                placed = temp
    elif mode == "tie":
        size = int(rng.integers(2, 7))
        pair = _place_overlap_pair(rng, h, w, _make_spec(rng, size), _make_spec(rng, size), want_tie=True)
        if pair is None:
            return None
        placed = pair
        extras = int(rng.integers(EXTRA_COMPONENTS_RANGE[0], EXTRA_COMPONENTS_RANGE[1] + 1))
        for _ in range(extras):
            spec = _make_spec(rng)
            comp = _place_random_spec(rng, h, w, spec, placed, clipped=False)
            if comp is None:
                continue
            temp = placed + [comp]
            stats = _claim_stats(h, w, temp)
            if stats["tie"] > 0 and stats["dominance"] == 0 and not stats["edge_clipped"]:
                placed = temp
    else:
        placed = _place_mixed_triplet(rng, h, w)
        if placed is None:
            return None

    if clip_required:
        clipped_spec = _make_spec(rng)
        added = False
        for _ in range(PLACE_ATTEMPTS):
            comp = _place_random_spec(rng, h, w, clipped_spec, placed, clipped=True)
            if comp is None:
                break
            temp = placed + [comp]
            if _validate_mode(mode, True, _claim_stats(h, w, temp)):
                placed = temp
                added = True
                break
        if not added:
            return None
    else:
        stats = _claim_stats(h, w, placed)
        if stats["edge_clipped"]:
            return None

    stats = _claim_stats(h, w, placed)
    if not _validate_mode(mode, clip_required, stats):
        return None

    colors = rng.choice(PALETTE, size=len(placed), replace=False)
    grid = _render_grid(h, w, placed, colors)
    out = main(grid)
    if not np.any(out != grid):
        return None
    comps = _find_components(grid, BG)
    if len(comps) != len(placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
