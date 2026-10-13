import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
USABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500
PLACEMENT_GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, allowed_mask: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    if allowed_mask is None:
        allowed_mask = np.ones((h, w), dtype=bool)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not allowed_mask[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not allowed_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": np.array(pts, dtype=int)})
    return comps


def _normalize_shape(coords: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_ring_shape(norm: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    if h < 3 or w < 3:
        return False
    border = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r == 0 or r == h - 1 or c == 0 or c == w - 1
    }
    return set(norm) == border


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for comp in _monochrome_components(grid, grid != bg):
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if coords.shape[0] != 2 * (h + w) - 4:
            continue
        norm = {(int(r - r0), int(c - c0)) for r, c in coords}
        border = {
            (rr, cc)
            for rr in range(h)
            for cc in range(w)
            if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1
        }
        if norm != border:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        if np.any(interior == bg) or np.any(interior == color):
            continue
        candidates.append((h * w, color, (r0, c0, r1, c1)))
    assert candidates, "no valid frame found"
    candidates.sort(reverse=True)
    best_area = candidates[0][0]
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "frame must be unique"
    _, color, bbox = best[0]
    return int(color), bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each region inside the framed mosaic using the outside stamp with the same shape.

    Input structure:
    - The background is black.
    - One grey 1-cell-thick rectangular border encloses the main pattern area.
    - The border interior is fully tiled by multiple differently colored 4-connected regions.
    - Outside the border are several scattered monochrome 4-connected stamp shapes, each in a distinct color.

    Rule:
    - Find the grey framed rectangle and keep only that rectangle in the output.
    - Normalize every interior region shape by translating its top-left occupied cell to the origin.
    - Normalize every outside stamp shape the same way.
    - Match each interior region to the unique outside stamp with the same normalized shape.
    - Recolor every cell of that interior region with the matched stamp's color, while keeping the grey border unchanged.

    Key insight:
    - The colors inside the frame are decoys; the hidden lookup key is exact shape equality under translation only.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color, (r0, c0, r1, c1) = _find_frame(g, bg)

    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    inner = crop[1:-1, 1:-1]
    assert inner.size > 0, "frame interior must be non-empty"
    assert np.all(inner != bg), "frame interior must be fully filled"

    outside_mask = np.ones(g.shape, dtype=bool)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    outside_mask &= g != bg
    outside_mask &= g != frame_color

    stamp_colors: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in _monochrome_components(g, outside_mask):
        shape, _, _ = _normalize_shape(comp["coords"])
        assert shape not in stamp_colors, "outside stamp shapes must be unique"
        stamp_colors[shape] = int(comp["color"])

    region_components = _monochrome_components(inner, np.ones(inner.shape, dtype=bool))
    assert len(region_components) == len(stamp_colors), "region/stamp counts must match"

    seen_regions: set[tuple[tuple[int, int], ...]] = set()
    for comp in region_components:
        shape, _, _ = _normalize_shape(comp["coords"])
        assert shape in stamp_colors, "missing outside stamp for an interior region"
        assert shape not in seen_regions, "interior region shapes must be unique"
        seen_regions.add(shape)
        color = stamp_colors[shape]
        for rr, cc in comp["coords"]:
            crop[1 + int(rr), 1 + int(cc)] = color

    return crop


def _pick_seeds(h: int, w: int, n_regions: int, rng: random.Random) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = [rng.choice(cells)]
    while len(seeds) < n_regions:
        sample = rng.sample(cells, min(len(cells), 20))
        best = None
        best_dist = -1
        for cand in sample:
            if cand in seeds:
                continue
            dist = min(abs(cand[0] - sr) + abs(cand[1] - sc) for sr, sc in seeds)
            if dist > best_dist:
                best_dist = dist
                best = cand
        if best is None:
            for cand in cells:
                if cand not in seeds:
                    best = cand
                    break
        seeds.append(best)
    return seeds


def _random_targets(total: int, n_regions: int, min_area: int, rng: random.Random) -> list[int]:
    remaining = total - n_regions * min_area
    weights = [rng.random() + 0.05 for _ in range(n_regions)]
    weight_sum = sum(weights)
    targets = [min_area + int(remaining * w / weight_sum) for w in weights]
    while sum(targets) < total:
        targets[rng.randrange(n_regions)] += 1
    while sum(targets) > total:
        idx = rng.randrange(n_regions)
        if targets[idx] > min_area:
            targets[idx] -= 1
    rng.shuffle(targets)
    return targets


def _grow_partition(h: int, w: int, n_regions: int, rng: random.Random) -> np.ndarray | None:
    min_area = 4
    if h * w < n_regions * min_area:
        return None
    labels = np.full((h, w), -1, dtype=int)
    seeds = _pick_seeds(h, w, n_regions, rng)
    targets = _random_targets(h * w, n_regions, min_area, rng)
    sizes = [0] * n_regions
    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
        sizes[idx] = 1

    while np.any(labels < 0):
        frontiers: list[list[tuple[int, int]]] = []
        active: list[int] = []
        for idx in range(n_regions):
            coords = np.argwhere(labels == idx)
            boundary: set[tuple[int, int]] = set()
            for r, c in coords:
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] < 0:
                        boundary.add((nr, nc))
            frontiers.append(list(boundary))
            if boundary:
                active.append(idx)
        if not active:
            return None

        weights = []
        for idx in active:
            deficit = max(0, targets[idx] - sizes[idx])
            weights.append((deficit + 1.0) * (len(frontiers[idx]) + 0.5))
        region = rng.choices(active, weights=weights, k=1)[0]
        candidates = frontiers[region]
        scored = []
        for r, c in candidates:
            same_neighbors = sum(
                1
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == region
            )
            scored.append((same_neighbors + rng.random(), r, c))
        scored.sort(reverse=True)
        _, r, c = scored[0]
        labels[r, c] = region
        sizes[region] += 1

    return labels


def _partition_shapes(labels: np.ndarray) -> list[dict] | None:
    n_regions = int(labels.max()) + 1
    shapes: list[dict] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    rectangular = 0
    for idx in range(n_regions):
        coords = np.argwhere(labels == idx)
        if coords.shape[0] < 4:
            return None
        norm, h, w = _normalize_shape(coords)
        if norm in seen:
            return None
        if _is_ring_shape(norm, h, w):
            return None
        seen.add(norm)
        rectangular += int(len(norm) == h * w)
        shapes.append(
            {
                "id": idx,
                "coords": coords,
                "norm": norm,
                "h": h,
                "w": w,
                "area": int(coords.shape[0]),
            }
        )
    if rectangular == n_regions:
        return None
    return shapes


def _build_partition(rng: random.Random) -> tuple[np.ndarray, list[dict]]:
    for _ in range(200):
        n_regions = rng.choice([3, 4])
        h = rng.randint(6, 10)
        w = rng.randint(6, 10)
        labels = _grow_partition(h, w, n_regions, rng)
        if labels is None:
            continue
        shapes = _partition_shapes(labels)
        if shapes is None:
            continue
        return labels, shapes
    raise RuntimeError("failed to build a valid partition")


def _place_rectangles(
    frame_hw: tuple[int, int], stamp_hws: list[tuple[int, int]], rng: random.Random
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    fh, fw = frame_hw
    max_h = max([fh] + [h for h, _ in stamp_hws])
    max_w = max([fw] + [w for _, w in stamp_hws])
    padded_area = (fh + 2) * (fw + 2) + sum((h + 2) * (w + 2) for h, w in stamp_hws)
    min_side = max(max_h, max_w, 14, int(np.ceil(np.sqrt(padded_area))))

    for _ in range(80):
        canvas_h = rng.randint(min_side, 30)
        canvas_w = rng.randint(min_side, 30)
        if canvas_h * canvas_w < padded_area:
            continue

        blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
        frame_pos = None
        stamp_positions: list[tuple[int, int] | None] = [None] * len(stamp_hws)

        objects = [("frame", -1, fh, fw)] + [
            ("stamp", idx, h, w) for idx, (h, w) in enumerate(stamp_hws)
        ]
        objects.sort(key=lambda item: item[2] * item[3], reverse=True)

        ok = True
        for kind, idx, obj_h, obj_w in objects:
            candidates = []
            for top in range(canvas_h - obj_h + 1):
                for left in range(canvas_w - obj_w + 1):
                    if blocked[top : top + obj_h, left : left + obj_w].any():
                        continue
                    candidates.append((top, left))
            if not candidates:
                ok = False
                break
            top, left = rng.choice(candidates)
            r0 = max(0, top - PLACEMENT_GAP)
            c0 = max(0, left - PLACEMENT_GAP)
            r1 = min(canvas_h, top + obj_h + PLACEMENT_GAP)
            c1 = min(canvas_w, left + obj_w + PLACEMENT_GAP)
            blocked[r0:r1, c0:c1] = True
            if kind == "frame":
                frame_pos = (top, left)
            else:
                stamp_positions[idx] = (top, left)

        if ok and frame_pos is not None and all(pos is not None for pos in stamp_positions):
            return frame_pos, [pos for pos in stamp_positions if pos is not None], (canvas_h, canvas_w)

    return None


def _expected_output(labels: np.ndarray, stamp_colors: list[int]) -> np.ndarray:
    h, w = labels.shape
    out = np.full((h + 2, w + 2), FRAME_COLOR, dtype=int)
    mapping = np.array(stamp_colors, dtype=int)
    out[1:-1, 1:-1] = mapping[labels]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    labels, shapes = _build_partition(rng)
    n_regions = len(shapes)

    chosen = rng.sample(USABLE_COLORS, 2 * n_regions)
    region_colors = chosen[:n_regions]
    stamp_colors = chosen[n_regions:]

    placement = _place_rectangles(
        (labels.shape[0] + 2, labels.shape[1] + 2),
        [(shape["h"], shape["w"]) for shape in shapes],
        rng,
    )
    if placement is None:
        return None
    frame_pos, stamp_positions, (canvas_h, canvas_w) = placement

    grid = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)
    fr, fc = frame_pos
    fh, fw = labels.shape[0] + 2, labels.shape[1] + 2

    grid[fr, fc : fc + fw] = FRAME_COLOR
    grid[fr + fh - 1, fc : fc + fw] = FRAME_COLOR
    grid[fr : fr + fh, fc] = FRAME_COLOR
    grid[fr : fr + fh, fc + fw - 1] = FRAME_COLOR
    for idx in range(n_regions):
        grid[fr + 1 : fr + 1 + labels.shape[0], fc + 1 : fc + 1 + labels.shape[1]][labels == idx] = region_colors[idx]

    for shape, color, (top, left) in zip(shapes, stamp_colors, stamp_positions):
        for dr, dc in shape["norm"]:
            grid[top + dr, left + dc] = color

    expected = _expected_output(labels, stamp_colors)
    if not np.array_equal(main(grid), expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if Counter(grid.flatten()).most_common(1)[0][0] != BG_COLOR:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
