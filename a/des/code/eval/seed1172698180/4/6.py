from __future__ import annotations

import numpy as np


BLACK = 0
AZURE = 8
MAROON = 9
PATTERN_COLORS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
GENERATION_RETRIES = 400


def _reflect(shape: tuple[int, int], axis: str, r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    if axis == "vertical":
        return int(r), int(w - 1 - c)
    assert axis == "horizontal"
    return int(h - 1 - r), int(c)


def _axis_metrics(g: np.ndarray, axis: str) -> tuple[int, int, int]:
    h, w = map(int, g.shape)
    conflicts = 0
    supported_maroon = 0
    matched_pairs = 0

    for r in range(h):
        for c in range(w):
            rr, cc = _reflect((h, w), axis, r, c)
            if (rr, cc) < (r, c):
                continue

            a = int(g[r, c])
            b = int(g[rr, cc])
            if a == MAROON or b == MAROON:
                if a == MAROON and b != MAROON:
                    supported_maroon += 1
                if b == MAROON and a != MAROON and (rr, cc) != (r, c):
                    supported_maroon += 1
                continue

            if a == b:
                matched_pairs += 1
            else:
                conflicts += 1

    return int(conflicts), int(supported_maroon), int(matched_pairs)


def _choose_axis(g: np.ndarray) -> str:
    scored: list[tuple[tuple[int, int, int], str]] = []
    for axis in ("vertical", "horizontal"):
        conflicts, supported, matched = _axis_metrics(g, axis)
        scored.append(((conflicts, -supported, -matched), axis))

    scored.sort()
    best_key, best_axis = scored[0]
    assert sum(1 for key, _axis in scored if key == best_key) == 1, "symmetry axis must be unique"
    return best_axis


def _paint_border(g: np.ndarray, thickness: int) -> np.ndarray:
    out = np.array(g, dtype=int).copy()
    t = int(thickness)
    out[:t, :] = AZURE
    out[-t:, :] = AZURE
    out[:, :t] = AZURE
    out[:, -t:] = AZURE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair maroon symmetry breaks, then grow the azure border by the maroon area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 7 and w >= 7, "grid too small"
    assert np.all(g[0, :] == AZURE) and np.all(g[-1, :] == AZURE), "missing top/bottom azure frame"
    assert np.all(g[:, 0] == AZURE) and np.all(g[:, -1] == AZURE), "missing left/right azure frame"

    maroon_mask = g == MAROON
    maroon_count = int(np.count_nonzero(maroon_mask))
    assert maroon_count > 0, "expected at least one maroon defect pixel"

    axis = _choose_axis(g)
    repaired = g.copy()
    for r, c in np.argwhere(maroon_mask):
        rr, cc = _reflect((h, w), axis, int(r), int(c))
        value = int(g[rr, cc])
        assert value != MAROON, "defect pixels must have visible mirrored counterparts"
        repaired[r, c] = value

    max_thickness = (min(h, w) + 1) // 2
    thickness = max(1, min(maroon_count, max_thickness))
    return _paint_border(repaired, thickness)


def _rand_with_parity(rng: np.random.Generator, lo: int, hi: int, parity: int) -> int:
    values = [v for v in range(lo, hi + 1) if v % 2 == parity]
    assert values, "no values with requested parity"
    return int(rng.choice(values))


def _paint_blocks(panel: np.ndarray, palette: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = map(int, panel.shape)
    base = int(rng.choice(palette))
    panel[:, :] = base

    for color in palette:
        rh = int(rng.integers(1, max(2, h // 2 + 1)))
        rw = int(rng.integers(1, max(2, w // 2 + 1)))
        r0 = int(rng.integers(0, h - rh + 1))
        c0 = int(rng.integers(0, w - rw + 1))
        panel[r0 : r0 + rh, c0 : c0 + rw] = int(color)

    extra = int(rng.integers(5, 11))
    for _ in range(extra):
        rh = int(rng.integers(1, h + 1))
        rw = int(rng.integers(1, w + 1))
        r0 = int(rng.integers(0, h - rh + 1))
        c0 = int(rng.integers(0, w - rw + 1))
        panel[r0 : r0 + rh, c0 : c0 + rw] = int(rng.choice(palette))

    return panel


def _build_pattern(ph: int, pw: int, axis: str, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        palette_size = int(rng.integers(3, 6))
        palette = np.array(rng.choice(PATTERN_COLORS, size=palette_size, replace=False), dtype=int)

        if axis == "vertical":
            half = pw // 2
            seed = _paint_blocks(np.empty((ph, half), dtype=int), palette, rng)
            if pw % 2 == 1:
                center = _paint_blocks(np.empty((ph, 1), dtype=int), palette, rng)
                pattern = np.concatenate([seed, center, seed[:, ::-1]], axis=1)
            else:
                pattern = np.concatenate([seed, seed[:, ::-1]], axis=1)
            if np.array_equal(pattern, pattern[::-1, :]):
                continue
        else:
            half = ph // 2
            seed = _paint_blocks(np.empty((half, pw), dtype=int), palette, rng)
            if ph % 2 == 1:
                center = _paint_blocks(np.empty((1, pw), dtype=int), palette, rng)
                pattern = np.concatenate([seed, center, seed[::-1, :]], axis=0)
            else:
                pattern = np.concatenate([seed, seed[::-1, :]], axis=0)
            if np.array_equal(pattern, pattern[:, ::-1]):
                continue

        if len(np.unique(pattern)) < 3:
            continue
        return pattern

    return None


def _side_dims(ph: int, pw: int, axis: str) -> tuple[int, int]:
    if axis == "vertical":
        return int(ph), int(pw // 2)
    return int(ph // 2), int(pw)


def _area_decompositions(side_h: int, side_w: int) -> dict[int, list[list[int]]]:
    areas = sorted({h * w for h in range(1, side_h + 1) for w in range(1, side_w + 1)})
    decomps: dict[int, list[list[int]]] = {}

    for a in areas:
        decomps.setdefault(int(a), []).append([int(a)])
    for a in areas:
        for b in areas:
            decomps.setdefault(int(a + b), []).append([int(a), int(b)])
    for a in areas:
        for b in areas:
            for c in areas:
                decomps.setdefault(int(a + b + c), []).append([int(a), int(b), int(c)])

    return decomps


def _place_rectangles(
    side_h: int,
    side_w: int,
    areas: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]] | None:
    dims_by_area = {
        area: [(h, w) for h in range(1, side_h + 1) for w in range(1, side_w + 1) if h * w == area]
        for area in set(areas)
    }

    for _ in range(120):
        occupied = np.zeros((side_h, side_w), dtype=bool)
        rects: list[tuple[int, int, int, int]] = []
        ok = True

        for area in sorted(areas, reverse=True):
            dims = list(dims_by_area[area])
            rng.shuffle(dims)
            placed = False

            for rh, rw in dims:
                positions = [
                    (r0, c0)
                    for r0 in range(side_h - rh + 1)
                    for c0 in range(side_w - rw + 1)
                    if not occupied[r0 : r0 + rh, c0 : c0 + rw].any()
                ]
                if not positions:
                    continue
                r0, c0 = positions[int(rng.integers(len(positions)))]
                occupied[r0 : r0 + rh, c0 : c0 + rw] = True
                rects.append((int(r0), int(c0), int(rh), int(rw)))
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok:
            return rects

    return None


def _apply_maroon_rects(
    clean: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    axis: str,
    pattern_origin: tuple[int, int],
    pattern_shape: tuple[int, int],
    use_far_side: bool,
) -> np.ndarray:
    out = clean.copy()
    pr0, pc0 = map(int, pattern_origin)
    ph, pw = map(int, pattern_shape)

    if axis == "vertical":
        side_col0 = int(pw - pw // 2) if use_far_side else 0
        for r0, c0, rh, rw in rects:
            rr0 = pr0 + r0
            cc0 = pc0 + side_col0 + c0
            out[rr0 : rr0 + rh, cc0 : cc0 + rw] = MAROON
    else:
        side_row0 = int(ph - ph // 2) if use_far_side else 0
        for r0, c0, rh, rw in rects:
            rr0 = pr0 + side_row0 + r0
            cc0 = pc0 + c0
            out[rr0 : rr0 + rh, cc0 : cc0 + rw] = MAROON

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    axis = "vertical" if bool(rng.integers(0, 2)) else "horizontal"
    capped = bool(rng.integers(0, 2))
    axis_parity = int(rng.integers(0, 2))

    if axis == "vertical":
        h = int(rng.integers(13, 22))
        w = _rand_with_parity(rng, 13, 21, axis_parity)
    else:
        h = _rand_with_parity(rng, 13, 21, axis_parity)
        w = int(rng.integers(13, 22))

    row_margin_choices = [m for m in range(1, 4) if h - 2 - 2 * m >= 7]
    col_margin_choices = [m for m in range(1, 4) if w - 2 - 2 * m >= 7]
    if not row_margin_choices or not col_margin_choices:
        return None

    mr = int(rng.choice(row_margin_choices))
    mc = int(rng.choice(col_margin_choices))
    ph = int(h - 2 - 2 * mr)
    pw = int(w - 2 - 2 * mc)

    side_h, side_w = _side_dims(ph, pw, axis)
    if side_h < 3 or side_w < 3:
        return None

    pattern = _build_pattern(ph, pw, axis, rng)
    if pattern is None:
        return None

    clean = np.full((h, w), BLACK, dtype=int)
    clean = _paint_border(clean, 1)
    pr0 = 1 + mr
    pc0 = 1 + mc
    clean[pr0 : pr0 + ph, pc0 : pc0 + pw] = pattern

    if axis == "vertical" and np.array_equal(clean, clean[::-1, :]):
        return None
    if axis == "horizontal" and np.array_equal(clean, clean[:, ::-1]):
        return None

    max_thickness = (min(h, w) + 1) // 2
    decomps = _area_decompositions(side_h, side_w)
    sums = sorted(decomps)
    if capped:
        candidates = [s for s in sums if max_thickness < s <= max_thickness + 8]
    else:
        candidates = [s for s in sums if 1 <= s <= max_thickness]
    if not candidates:
        return None

    target_area = int(rng.choice(candidates))
    area_options = [areas for areas in decomps[target_area] if len(areas) <= 3]
    if not area_options:
        return None

    areas = list(area_options[int(rng.integers(len(area_options)))])
    rects = _place_rectangles(side_h, side_w, areas, rng)
    if rects is None:
        return None

    use_far_side = bool(rng.integers(0, 2))
    inp = _apply_maroon_rects(clean, rects, axis, (pr0, pc0), (ph, pw), use_far_side)

    detected = _choose_axis(inp)
    if detected != axis:
        return None

    out = _paint_border(clean, min(target_area, max_thickness))
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
