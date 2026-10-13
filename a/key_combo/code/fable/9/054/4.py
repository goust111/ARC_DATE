from __future__ import annotations

import numpy as np


MIN_PERIOD = 2
MAX_PERIOD = 5
PERIOD_CHOICES = (3, 4)
MAX_GENERATION_ATTEMPTS = 800


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _most_common(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return int(uniq[np.argmax(counts)])


def _minimal_period(seq: list[int]) -> int:
    n = len(seq)
    for p in range(1, n):
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return n


def _render_ring_field(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    sequence: list[int],
    bg: int,
) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    dist = np.maximum(np.abs(rr - center[0]), np.abs(cc - center[1]))
    out = np.full((h, w), int(bg), dtype=int)
    palette = np.asarray(sequence, dtype=int)
    inside = dist <= int(radius)
    out[inside] = palette[dist[inside] % len(sequence)]
    return out


def _find_occluder_rect(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, tuple[int, int, int, int], int]] = []
    for color in map(int, np.unique(g)):
        mask = g == color
        count = int(mask.sum())
        if count < 9:
            continue
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if count != area:
            continue
        if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
            candidates.append((area, box, color))
    assert len(candidates) == 1, "expected a unique solid occluder rectangle"
    _area, rect, color = candidates[0]
    return rect, int(color)


def _enumerate_models(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    rect, occ_color = _find_occluder_rect(g)
    r0, c0, r1, c1 = rect

    occ_mask = np.zeros_like(g, dtype=bool)
    occ_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    bg = _most_common(g[~occ_mask])

    visible_color_mask = (~occ_mask) & (g != bg)
    visible_bg_mask = (~occ_mask) & (g == bg)
    color_rows, color_cols = np.nonzero(visible_color_mask)
    bg_rows, bg_cols = np.nonzero(visible_bg_mask)
    color_vals = g[visible_color_mask].astype(int)
    assert color_vals.size > 0, "expected visible ring fragments outside the occluder"
    assert bg_rows.size > 0, "expected visible background outside the ring field"

    models: list[dict] = []
    for center_r in range(r0, r1 + 1):
        for center_c in range(c0, c1 + 1):
            d_color = np.maximum(np.abs(color_rows - center_r), np.abs(color_cols - center_c))
            radius = int(d_color.max())
            if radius < 2:
                continue

            d_bg = np.maximum(np.abs(bg_rows - center_r), np.abs(bg_cols - center_c))
            if int(d_bg.min()) != radius + 1:
                continue

            per_distance: dict[int, int] = {}
            consistent = True
            for dist, color in zip(d_color.tolist(), color_vals.tolist()):
                prev = per_distance.get(int(dist))
                if prev is None:
                    per_distance[int(dist)] = int(color)
                elif prev != int(color):
                    consistent = False
                    break
            if not consistent:
                continue

            for period in range(MIN_PERIOD, MAX_PERIOD + 1):
                residue_colors = [-1] * period
                ok = True
                for dist, color in per_distance.items():
                    idx = dist % period
                    if residue_colors[idx] == -1:
                        residue_colors[idx] = color
                    elif residue_colors[idx] != color:
                        ok = False
                        break
                if not ok or any(color == -1 for color in residue_colors):
                    continue
                if _minimal_period(residue_colors) != period:
                    continue

                models.append(
                    {
                        "center": (int(center_r), int(center_c)),
                        "radius": int(radius),
                        "period": int(period),
                        "sequence": [int(x) for x in residue_colors],
                    }
                )

    return rect, int(occ_color), int(bg), models


def _recover_patch_from_model(
    rect: tuple[int, int, int, int],
    bg: int,
    model: dict,
) -> np.ndarray:
    r0, c0, r1, c1 = rect
    center_r, center_c = model["center"]
    radius = int(model["radius"])
    sequence = [int(x) for x in model["sequence"]]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            dist = max(abs(r - center_r), abs(c - center_c))
            if dist <= radius:
                out[r - r0, c - c0] = sequence[dist % len(sequence)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the occluded crop of a finite cyclic Chebyshev ring field.

    Input structure:
    - The grid contains a finite concentric square-ring field around one hidden center cell.
    - Ring color depends only on Chebyshev distance from that center and cycles with an unknown period.
    - A single solid rectangle of one color occludes the center and several inner rings; cells outside the cutoff radius are background.

    Rule:
    - Identify the unique solid occluder rectangle and treat its contents as missing.
    - Search for the unique center, color-cycle period, phase, and cutoff radius consistent with all visible ring fragments and the first visible background beyond the outer ring.
    - Reconstruct only the occluded rectangle: cells whose Chebyshev distance to the recovered center is within the cutoff take the cycle color for that distance; farther cells stay background.

    Key insight:
    - The answer is not a symmetry copy from outside the mask: the center, the ring-color phase, and the outer cutoff all have to be inferred from surviving fragments of the square rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect, _occ_color, bg, models = _enumerate_models(g)
    assert len(models) == 1, "expected a unique valid ring-field model"
    return _recover_patch_from_model(rect, bg, models[0])


def _visible_residues(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    rect: tuple[int, int, int, int],
    period: int,
) -> set[int]:
    h, w = shape
    r0, c0, r1, c1 = rect
    residues: set[int] = set()
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            dist = max(abs(r - center[0]), abs(c - center[1]))
            if dist <= radius:
                residues.add(int(dist % period))
    return residues


def _visible_corner_count(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    rect: tuple[int, int, int, int],
) -> int:
    h, w = shape
    r0, c0, r1, c1 = rect
    total = 0
    for dist in range(1, radius + 1):
        for rr in (center[0] - dist, center[0] + dist):
            for cc in (center[1] - dist, center[1] + dist):
                if 0 <= rr < h and 0 <= cc < w and not (r0 <= rr <= r1 and c0 <= cc <= c1):
                    total += 1
    return total


def _visible_quadrants(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    rect: tuple[int, int, int, int],
) -> int:
    h, w = shape
    r0, c0, r1, c1 = rect
    quadrants: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            dist = max(abs(r - center[0]), abs(c - center[1]))
            if dist > radius:
                continue
            sr = 0 if r < center[0] else 1
            sc = 0 if c < center[1] else 1
            quadrants.add((sr, sc))
    return len(quadrants)


def _sample_extent(low: int, high: int) -> int:
    return int(np.random.randint(low, high + 1))


def _generate_attempt() -> np.ndarray | None:
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    bg = 0

    feasible_radii = [
        r
        for r in range(5, min((h - 3) // 2, (w - 3) // 2) + 1)
        if (2 * r + 1) ** 2 <= int(0.45 * h * w)
    ]
    if not feasible_radii:
        return None
    radius = int(np.random.choice(feasible_radii))
    center = (
        int(np.random.randint(radius + 1, h - radius - 1)),
        int(np.random.randint(radius + 1, w - radius - 1)),
    )

    period = int(np.random.choice(PERIOD_CHOICES))
    palette = np.random.permutation(np.arange(1, 10)).tolist()
    sequence = [int(x) for x in palette[:period]]
    occ_color = int(palette[period])

    inner_cover_max = min(4, radius - 1)
    if inner_cover_max < 2:
        return None
    inner_cover = int(np.random.randint(2, inner_cover_max + 1))
    allow_hidden_bg = bool(np.random.randint(0, 2))

    side_limits = [center[0], h - 1 - center[0], center[1], w - 1 - center[1]]
    max_inside = radius
    max_hidden = radius + 1

    extents: list[int] = []
    for limit in side_limits:
        hi = min(limit, max_hidden if allow_hidden_bg else max_inside)
        lo = inner_cover
        if hi < lo:
            return None
        extents.append(_sample_extent(lo, hi))

    if np.random.rand() < 0.75 and extents[0] == extents[1] and extents[2] == extents[3]:
        idx = int(np.random.randint(0, 4))
        if extents[idx] < min(side_limits[idx], max_hidden):
            extents[idx] += 1

    up, down, left, right = extents
    rect = (
        center[0] - up,
        center[1] - left,
        center[0] + down,
        center[1] + right,
    )

    full = _render_ring_field((h, w), center, radius, sequence, bg)
    inp = full.copy()
    r0, c0, r1, c1 = rect
    inp[r0 : r1 + 1, c0 : c1 + 1] = occ_color

    occ_mask = np.zeros_like(inp, dtype=bool)
    occ_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    if _most_common(inp[~occ_mask]) != bg:
        return None

    if _visible_residues(inp.shape, center, radius, rect, period) != set(range(period)):
        return None
    if _visible_corner_count(inp.shape, center, radius, rect) < max(4, period + 1):
        return None
    if _visible_quadrants(inp.shape, center, radius, rect) < 3:
        return None

    hidden = _crop(full, rect)
    if len(np.unique(hidden)) < 2:
        return None

    _rect, _occ_color, found_bg, models = _enumerate_models(inp)
    if _rect != rect or _occ_color != occ_color or found_bg != bg:
        return None
    if len(models) != 1:
        return None
    model = models[0]
    if model["center"] != center or model["radius"] != radius or model["period"] != period:
        return None
    if model["sequence"] != sequence:
        return None

    if not np.array_equal(main(inp), hidden):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded ring-field sample")
