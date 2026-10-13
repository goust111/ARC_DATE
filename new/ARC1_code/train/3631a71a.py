from __future__ import annotations

import numpy as np


Coord = tuple[int, int]


def _orbit(start: Coord, size: int, margin: int) -> set[Coord]:
    seen = {start}
    queue = [start]
    for r, c in queue:
        candidates = [(c, r)]
        if r >= margin:
            candidates.append((size - 1 + margin - r, c))
        if c >= margin:
            candidates.append((r, size - 1 + margin - c))

        for rr, cc in candidates:
            if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return seen


def _orbits(size: int, margin: int) -> list[set[Coord]]:
    seen: set[Coord] = set()
    result: list[set[Coord]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orb = _orbit((r, c), size, margin)
            seen.update(orb)
            result.append(orb)
    return result


def _infer_margin_and_mask(g: np.ndarray) -> tuple[int, int, list[set[Coord]]]:
    size = int(g.shape[0])
    colors = [int(c) for c in np.unique(g)]
    matches: list[tuple[int, int, list[set[Coord]]]] = []

    for margin in range(size // 2 + 1):
        orbits = _orbits(size, margin)
        for mask_color in colors:
            valid = True
            for orb in orbits:
                values = [int(g[r, c]) for r, c in orb]
                visible = {value for value in values if value != mask_color}
                if len(visible) > 1 or (mask_color in values and len(visible) != 1):
                    valid = False
                    break
            if valid:
                matches.append((margin, mask_color, orbits))

    assert len(matches) == 1, "expected a unique symmetry margin and mask color"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace masked pixels with the unique visible color from the same symmetry orbit.

    Input structure:
    - The grid is square.
    - One color marks unknown cells and is the only color whose removal makes all
      symmetry orbits internally consistent.
    - The underlying image is invariant under main-diagonal transpose and under
      row/column reflection of the trailing square after a leading prefix.

    Rule:
    - Infer the leading prefix length, the mask color, and the corresponding
      symmetry orbits.
    - For every orbit containing masked pixels, replace them with the orbit's
      unique non-mask color.
    - Preserve all already visible pixels.

    Key insight:
    - The mask is transparent damage over a dihedral-symmetric pattern, so even
      cells whose transpose is also masked are determined by another orbit mate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width, "expected a square grid"

    _margin, mask_color, orbits = _infer_margin_and_mask(g)
    out = g.copy()
    for orb in orbits:
        values = [int(g[r, c]) for r, c in orb]
        if mask_color not in values:
            continue
        visible = {value for value in values if value != mask_color}
        assert len(visible) == 1, "masked orbit must have exactly one visible fill color"
        fill = visible.pop()
        for r, c in orb:
            if int(g[r, c]) == mask_color:
                out[r, c] = fill

    assert not np.any(out == mask_color), "all mask pixels must be filled"
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GEN_GRID_SIZE = 30
GEN_MARGIN = 2
GEN_BG_COLOR, GEN_MASK_COLOR = 0, 9
GEN_FILL_COLORS = range(1, 9)

# Output orbit allocation.
GEN_BG_ORBITS_MIN, GEN_BG_ORBITS_MAX_EXCL = 55, 89
GEN_NON_BG_ORBIT_MIN, GEN_NON_BG_ORBIT_MAX = 1, 20
GEN_BG_PIXELS_MIN, GEN_BG_PIXELS_MAX = 288, 612

# Mask blob layout and constraints.
GEN_BLOB_COUNT_MIN, GEN_BLOB_COUNT_MAX_EXCL = 2, 6
GEN_MASK_PIXELS_MIN, GEN_MASK_PIXELS_MAX_EXCL = 49, 124
GEN_BLOB_PIXELS_MIN, GEN_BLOB_PIXELS_MAX = 4, 60
GEN_BLOB_MAX_HEIGHT, GEN_BLOB_MAX_WIDTH = 14, 18
GEN_BLOB_MAX_ASPECT = 5
GEN_BLOB_EXTRA_AREA_FLOOR = 3
GEN_FORBIDDEN_CORNER = 2
GEN_MIN_MASKED_COLORS, GEN_ORBIT_MASK_CAP_DIVISOR = 2, 2

# Generation retries.
GEN_RETRY_LIMIT = 300


def _gen_split_bounded(total: int, parts: int, low: int, high: int) -> list[int] | None:
    if total < parts * low or total > parts * high:
        return None

    values: list[int] = []
    remaining = int(total)
    for i in range(parts):
        slots = parts - i - 1
        lo = max(low, remaining - high * slots)
        hi = min(high, remaining - low * slots)
        if lo > hi:
            return None
        value = int(np.random.randint(lo, hi + 1))
        values.append(value)
        remaining -= value
    return values


def _gen_orbit_lookup(orbits: list[set[tuple[int, int]]], size: int) -> np.ndarray:
    lookup = np.full((size, size), -1, dtype=int)
    for i, orb in enumerate(orbits):
        for r, c in orb:
            lookup[r, c] = i
    return lookup


def _gen_build_output() -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    orbits = _orbits(GEN_GRID_SIZE, GEN_MARGIN)
    bg_orbit_count = int(np.random.randint(GEN_BG_ORBITS_MIN, GEN_BG_ORBITS_MAX_EXCL))
    non_bg_total = len(orbits) - bg_orbit_count
    non_bg_counts = _gen_split_bounded(
        non_bg_total,
        len(GEN_FILL_COLORS),
        GEN_NON_BG_ORBIT_MIN,
        GEN_NON_BG_ORBIT_MAX,
    )
    if non_bg_counts is None:
        return None

    orbit_colors = [GEN_BG_COLOR] * bg_orbit_count
    for color, count in zip(GEN_FILL_COLORS, non_bg_counts):
        orbit_colors.extend([color] * count)
    np.random.shuffle(orbit_colors)

    out = np.zeros((GEN_GRID_SIZE, GEN_GRID_SIZE), dtype=int)
    for orb, color in zip(orbits, orbit_colors):
        for r, c in orb:
            out[r, c] = color

    bg_pixels = int(np.count_nonzero(out == GEN_BG_COLOR))
    if not (GEN_BG_PIXELS_MIN <= bg_pixels <= GEN_BG_PIXELS_MAX):
        return None

    lookup = _gen_orbit_lookup(orbits, GEN_GRID_SIZE)
    caps = [len(orb) // GEN_ORBIT_MASK_CAP_DIVISOR for orb in orbits]
    return out, lookup, caps


def _gen_blob_shape(pixel_count: int) -> tuple[list[tuple[int, int]], int, int]:
    options: list[tuple[int, int]] = []
    for h in range(1, min(GEN_BLOB_MAX_HEIGHT, pixel_count) + 1):
        for w in range(1, min(GEN_BLOB_MAX_WIDTH, pixel_count) + 1):
            area = h * w
            if area < pixel_count:
                continue
            if max(h, w) / max(1, min(h, w)) > GEN_BLOB_MAX_ASPECT:
                continue
            if area - pixel_count > max(GEN_BLOB_EXTRA_AREA_FLOOR, min(h, w) + 1):
                continue
            options.append((h, w))

    h, w = options[int(np.random.randint(0, len(options)))]
    local = np.zeros((h, w), dtype=bool)
    full_rows, rem = divmod(pixel_count, w)
    if full_rows:
        local[:full_rows, :] = True
    if rem:
        start = int(np.random.randint(0, w - rem + 1))
        local[full_rows, start : start + rem] = True

    if bool(np.random.randint(0, 2)):
        local = np.flipud(local)
    if bool(np.random.randint(0, 2)):
        local = np.fliplr(local)
    if bool(np.random.randint(0, 2)):
        local = local.T

    coords = [(int(r), int(c)) for r, c in np.argwhere(local)]
    return coords, int(local.shape[0]), int(local.shape[1])


def _gen_can_place_blob(
    mask: np.ndarray,
    coords: list[tuple[int, int]],
    r0: int,
    c0: int,
    orbit_lookup: np.ndarray,
    caps: list[int],
    orbit_mask_counts: list[int],
) -> bool:
    size = int(mask.shape[0])
    local_orbit_counts: dict[int, int] = {}

    for dr, dc in coords:
        r = r0 + dr
        c = c0 + dc
        if r < GEN_FORBIDDEN_CORNER and c < GEN_FORBIDDEN_CORNER:
            return False
        if mask[r, c]:
            return False
        if r > 0 and mask[r - 1, c]:
            return False
        if r + 1 < size and mask[r + 1, c]:
            return False
        if c > 0 and mask[r, c - 1]:
            return False
        if c + 1 < size and mask[r, c + 1]:
            return False

        orbit_id = int(orbit_lookup[r, c])
        local_orbit_counts[orbit_id] = local_orbit_counts.get(orbit_id, 0) + 1
        if orbit_mask_counts[orbit_id] + local_orbit_counts[orbit_id] > caps[orbit_id]:
            return False

    return True


def _gen_place_blob(
    mask: np.ndarray,
    coords: list[tuple[int, int]],
    height: int,
    width: int,
    orbit_lookup: np.ndarray,
    caps: list[int],
    orbit_mask_counts: list[int],
) -> bool:
    size = int(mask.shape[0])
    positions: list[tuple[int, int]] = []
    for r0 in range(size - height + 1):
        for c0 in range(size - width + 1):
            if _gen_can_place_blob(mask, coords, r0, c0, orbit_lookup, caps, orbit_mask_counts):
                positions.append((r0, c0))

    if not positions:
        return False

    r0, c0 = positions[int(np.random.randint(0, len(positions)))]
    for dr, dc in coords:
        r = r0 + dr
        c = c0 + dc
        mask[r, c] = True
        orbit_id = int(orbit_lookup[r, c])
        orbit_mask_counts[orbit_id] += 1
    return True


def _gen_build_mask(orbit_lookup: np.ndarray, caps: list[int]) -> np.ndarray | None:
    """Place separated blobs while enforcing the per-orbit mask cap."""
    blob_count = int(np.random.randint(GEN_BLOB_COUNT_MIN, GEN_BLOB_COUNT_MAX_EXCL))
    total_mask = int(np.random.randint(GEN_MASK_PIXELS_MIN, GEN_MASK_PIXELS_MAX_EXCL))
    while total_mask > GEN_BLOB_PIXELS_MAX * blob_count:
        blob_count = int(np.random.randint(GEN_BLOB_COUNT_MIN, GEN_BLOB_COUNT_MAX_EXCL))

    blob_sizes = _gen_split_bounded(
        total_mask,
        blob_count,
        GEN_BLOB_PIXELS_MIN,
        GEN_BLOB_PIXELS_MAX,
    )
    if blob_sizes is None:
        return None
    np.random.shuffle(blob_sizes)

    mask = np.zeros((GEN_GRID_SIZE, GEN_GRID_SIZE), dtype=bool)
    orbit_mask_counts = [0] * len(caps)
    for blob_size in blob_sizes:
        coords, height, width = _gen_blob_shape(blob_size)
        if not _gen_place_blob(mask, coords, height, width, orbit_lookup, caps, orbit_mask_counts):
            return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    built = _gen_build_output()
    if built is None:
        return None
    out, orbit_lookup, caps = built

    mask = _gen_build_mask(orbit_lookup, caps)
    if mask is None:
        return None

    if len(set(map(int, out[mask]))) < GEN_MIN_MASKED_COLORS:
        return None

    inp = out.copy()
    inp[mask] = GEN_MASK_COLOR
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid 3631a71a input")
