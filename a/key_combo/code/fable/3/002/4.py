from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(10))

_GRID_SIZES = (9, 11, 13, 15, 17, 19, 21, 23, 25)
_GEN_RETRY_LIMIT = 500
_SPRITE_RETRY_LIMIT = 120
_GEN_MODE_COUNT = 4
_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _rot90_coord(n: int, r: int, c: int) -> tuple[int, int]:
    return int(c), int(n - 1 - r)


def _rotation_orbit(n: int, r: int, c: int) -> list[tuple[int, int]]:
    orbit: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    for _ in range(4):
        if (rr, cc) not in orbit:
            orbit.append((rr, cc))
        rr, cc = _rot90_coord(n, rr, cc)
    return orbit


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_anomalies(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1, "expected an odd square grid"

    seen = np.zeros((h, w), dtype=bool)
    anomalies: list[tuple[int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            orbit = _rotation_orbit(h, r, c)
            for rr, cc in orbit:
                seen[rr, cc] = True

            values = [int(g[rr, cc]) for rr, cc in orbit]
            counts = Counter(values)
            if len(counts) == 1:
                continue

            assert len(orbit) == 4, "only four-cell rotation orbits can contain an anomaly"
            assert len(counts) == 2 and sorted(counts.values()) == [1, 3], (
                "each damaged orbit must have exactly three matching base pixels"
            )
            odd_value = next(value for value, count in counts.items() if count == 1)
            odd_index = values.index(odd_value)
            ar, ac = orbit[odd_index]
            anomalies.append((ar, ac, odd_value))

    assert anomalies, "expected at least one stamped anomaly"
    return anomalies


def _render_anomaly_crop(anomalies: list[tuple[int, int, int]]) -> np.ndarray:
    coords = [(r, c) for r, c, _value in anomalies]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c, value in anomalies:
        out[r - r0, c - c0] = int(value)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the hidden sprite made by pixels that break fourfold rotational symmetry.

    Input structure:
    - The input is an odd square color grid whose undamaged base pattern is
      invariant under 90 degree rotation about the center.
    - A small sprite has been stamped over some base cells, and its colors are
      also normal colors elsewhere in the base pattern.
    - No stamped cell shares its rotation orbit with another stamped cell, so
      every damaged four-cell orbit still has three unchanged matching values.

    Rule:
    - Partition the grid into 90/180/270 degree rotation orbits around the
      center of the square.
    - Orbits whose four visible values match are undamaged base pattern.
    - In an orbit with three equal values and one different value, keep the
      different current pixel as one sprite pixel.
    - Return the smallest bounding-box crop of all sprite pixels, preserving
      their colors and filling all other crop cells with black.

    Key insight:
    - The rotational symmetry is an anomaly detector, not a repair target: the
      minority value in each broken orbit is itself part of the hidden sprite.
    """
    anomalies = _detect_anomalies(grid)
    return _render_anomaly_crop(anomalies)


def _all_rotation_orbits(n: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = _rotation_orbit(n, r, c)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _orbit_key(n: int, r: int, c: int) -> tuple[int, int]:
    return min(_rotation_orbit(n, r, c))


def _is_rotationally_symmetric(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    if h != w:
        return False
    for r in range(h):
        for c in range(w):
            rr, cc = _rot90_coord(h, r, c)
            if int(g[r, c]) != int(g[rr, cc]):
                return False
    return True


def _sample_palette(rng: np.random.Generator) -> list[int]:
    nonblack_count = int(rng.integers(3, 6))
    nonblack = rng.choice(np.arange(1, 10), size=nonblack_count, replace=False).astype(int).tolist()
    palette = nonblack[:]
    if rng.random() < 0.45:
        palette.append(BLACK)
    rng.shuffle(palette)
    return [int(c) for c in palette]


def _build_symmetric_base(n: int, palette: list[int], rng: np.random.Generator) -> np.ndarray:
    base = np.zeros((n, n), dtype=int)
    orbits = _all_rotation_orbits(n)
    rng.shuffle(orbits)

    forced = palette[:]
    rng.shuffle(forced)
    for orbit in orbits:
        color = int(forced.pop()) if forced else int(rng.choice(palette))
        for r, c in orbit:
            base[r, c] = color
    return base


def _connected_sprite_mask(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        h = int(rng.integers(2, 7))
        w = int(rng.integers(2, 7))
        max_area = min(h * w - 1, 11)
        if max_area < 3:
            continue
        target_area = int(rng.integers(3, max_area + 1))

        cells = {(int(rng.integers(0, h)), int(rng.integers(0, w)))}
        moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
        grow_attempts = 0
        while len(cells) < target_area and grow_attempts < 300:
            grow_attempts += 1
            r, c = list(cells)[int(rng.integers(0, len(cells)))]
            dr, dc = moves[int(rng.integers(0, len(moves)))]
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
        if len(cells) != target_area:
            continue

        r0, c0, r1, c1 = _bbox_from_coords(list(cells))
        crop_h = r1 - r0 + 1
        crop_w = c1 - c0 + 1
        if crop_h < 2 or crop_w < 2:
            continue
        if len(cells) >= crop_h * crop_w:
            continue

        mask = np.zeros((crop_h, crop_w), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        return mask
    return None


def _sprite_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        coords.append((int(top + r), int(left + c)))
    return coords


def _valid_sprite_placement(n: int, mask: np.ndarray, top: int, left: int) -> bool:
    coords = _sprite_coords(mask, top, left)
    keys = [_orbit_key(n, r, c) for r, c in coords]
    return len(keys) == len(set(keys))


def _choose_placement(
    n: int,
    mask: np.ndarray,
    rng: np.random.Generator,
    mode: int,
) -> list[tuple[int, int]] | None:
    h, w = map(int, mask.shape)
    valid: list[tuple[int, int]] = []
    for top in range(n - h + 1):
        for left in range(n - w + 1):
            if _valid_sprite_placement(n, mask, top, left):
                valid.append((top, left))
    if not valid:
        return None

    center = (n - 1) / 2.0
    if mode == 1:
        subset = [(r, c) for r, c in valid if r == 0 or c == 0 or r + h == n or c + w == n]
        if subset:
            valid = subset
    elif mode == 2:
        ranked = sorted(
            valid,
            key=lambda rc: abs(rc[0] + (h - 1) / 2.0 - center) + abs(rc[1] + (w - 1) / 2.0 - center),
        )
        valid = ranked[: max(1, len(ranked) // 3)]
    elif mode == 3:
        subset = [
            (r, c)
            for r, c in valid
            if r + (h - 1) / 2.0 >= center and c + (w - 1) / 2.0 >= center
        ]
        if subset:
            valid = subset

    top, left = valid[int(rng.integers(0, len(valid)))]
    return _sprite_coords(mask, top, left)


def _assign_sprite_values(
    base: np.ndarray,
    coords: list[tuple[int, int]],
    palette: list[int],
    rng: np.random.Generator,
) -> list[int] | None:
    nonblack = [int(c) for c in palette if int(c) != BLACK]
    if len(nonblack) < 2:
        return None

    sprite_palette_size = min(len(nonblack), int(rng.integers(2, min(4, len(nonblack)) + 1)))
    sprite_palette = rng.choice(np.array(nonblack, dtype=int), size=sprite_palette_size, replace=False).astype(int).tolist()

    values: list[int] = []
    for r, c in coords:
        options = [color for color in sprite_palette if color != int(base[r, c])]
        if not options:
            options = [color for color in nonblack if color != int(base[r, c])]
        if not options:
            return None
        values.append(int(options[int(rng.integers(0, len(options)))]))

    if len(set(values)) < 2:
        first = values[0]
        for idx, (r, c) in enumerate(coords):
            options = [color for color in nonblack if color != first and color != int(base[r, c])]
            if options:
                values[idx] = int(options[int(rng.integers(0, len(options)))])
                break
    if len(set(values)) < 2:
        return None
    return values


def _expected_from_stamp(coords: list[tuple[int, int]], values: list[int]) -> np.ndarray:
    anomalies = [(r, c, int(value)) for (r, c), value in zip(coords, values)]
    return _render_anomaly_crop(anomalies)


def _stamp_grid(base: np.ndarray, coords: list[tuple[int, int]], values: list[int]) -> np.ndarray:
    grid = base.copy()
    for (r, c), value in zip(coords, values):
        grid[r, c] = int(value)
    return grid


def _semantic_valid(
    base: np.ndarray,
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    values: list[int],
    expected: np.ndarray,
) -> bool:
    if not _is_rotationally_symmetric(base):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    coord_set = set(coords)
    if len(coord_set) != len(coords):
        return False
    n = int(base.shape[0])
    anomaly_mask = np.zeros(base.shape, dtype=bool)
    for r, c in coords:
        anomaly_mask[r, c] = True
        if int(base[r, c]) == int(grid[r, c]):
            return False
        orbit = _rotation_orbit(n, r, c)
        if sum((rr, cc) in coord_set for rr, cc in orbit) != 1:
            return False
        orbit_values = [int(grid[rr, cc]) for rr, cc in orbit]
        if sorted(Counter(orbit_values).values()) != [1, 3]:
            return False

    if len(set(values)) < 2:
        return False
    for color in set(values):
        if not np.any((grid == int(color)) & ~anomaly_mask):
            return False

    coords_only = [(r, c) for r, c in coords]
    r0, c0, r1, c1 = _bbox_from_coords(coords_only)
    if len(coords) >= (r1 - r0 + 1) * (c1 - c0 + 1):
        return False
    if not np.any(expected == BLACK):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt(mode: int) -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    n = int(rng.choice(np.array(_GRID_SIZES, dtype=int)))
    palette = _sample_palette(rng)
    base = _build_symmetric_base(n, palette, rng)

    for _ in range(_SPRITE_RETRY_LIMIT):
        mask = _connected_sprite_mask(rng)
        if mask is None:
            continue
        coords = _choose_placement(n, mask, rng, int(mode))
        if coords is None:
            continue

        values = _assign_sprite_values(base, coords, palette, rng)
        if values is None:
            continue
        grid = _stamp_grid(base, coords, values)
        expected = _expected_from_stamp(coords, values)
        if _semantic_valid(base, grid, coords, values, expected):
            return grid, expected
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_CALL_COUNT % _GEN_MODE_COUNT
    for _ in range(_GEN_RETRY_LIMIT):
        result = _gen_attempt(mode)
        if result is None:
            continue
        grid, expected = result
        output_signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if output_signature in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(output_signature)
        _GEN_CALL_COUNT += 1
        return grid
    raise RuntimeError("failed to generate a valid rotational-anomaly sprite input")
