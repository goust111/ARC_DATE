from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, neighbors4


COLOR_POOL = tuple(range(1, 10))
MIN_COLORS = 3
MAX_COLORS = 4
MAX_ATTEMPTS = 400


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    return find_components(
        grid,
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )


def _mirror_col(axis_twice: int, col: int) -> int:
    return int(axis_twice - col)


def _classify_color(grid: np.ndarray, color: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]], int]:
    components = _color_components(grid, color)
    large = [comp for comp in components if int(comp["area"]) >= 5]
    small = [comp for comp in components if int(comp["area"]) <= 2]
    assert len(large) == 1, f"color {color} must have exactly one cluster"
    assert len(large) + len(small) == len(components), f"color {color} has unsupported component size"

    cluster = {tuple(map(int, coord)) for coord in large[0]["coords"]}
    markers = sorted(tuple(map(int, coord)) for comp in small for coord in comp["coords"])
    assert len(markers) == 2, f"color {color} must have exactly two marker pixels"

    axis_twice = int(markers[0][1] + markers[1][1])
    return cluster, markers, axis_twice


def _denoise_cluster(shape: tuple[int, int], cluster: set[tuple[int, int]], axis_twice: int) -> set[tuple[int, int]]:
    height, width = map(int, shape)
    kept: set[tuple[int, int]] = set()
    for r, c in cluster:
        mc = _mirror_col(axis_twice, c)
        if 0 <= mc < width and 0 <= r < height and (r, mc) in cluster:
            kept.add((r, c))
    return kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove one-sided noise from each color layer using the symmetry axis encoded by that color's marker pair.

    Input structure:
    - The background is black.
    - Each non-background color has one 4-connected cluster of 8-15 pixels plus two isolated same-color marker pixels.
    - The marker pixels are far from the cluster, and their column midpoint gives that color layer's vertical symmetry axis.

    Rule:
    - For each non-background color, split its 4-connected components into one large cluster component and two marker pixels.
    - Compute the vertical axis from the two marker columns: axis = (c1 + c2) / 2.
    - Keep only cluster pixels whose horizontal mirror across that axis is another cluster pixel of the same color, then erase all marker pixels.

    Key insight:
    - The apparently stray same-color singletons are not noise; they explicitly encode the symmetry axis needed to decide which cluster pixels are real and which are one-sided defects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        cluster, _markers, axis_twice = _classify_color(g, color)
        kept = _denoise_cluster(g.shape, cluster, axis_twice)
        for r, c in kept:
            out[r, c] = int(color)
    return out


def _orbit_cells(row: int, offset: int, axis_twice: int) -> tuple[tuple[int, int], ...]:
    if axis_twice % 2 == 0:
        axis = axis_twice // 2
        if offset == 0:
            return ((int(row), int(axis)),)
        return ((int(row), int(axis - offset)), (int(row), int(axis + offset)))

    left = axis_twice // 2 - offset
    right = axis_twice // 2 + 1 + offset
    return ((int(row), int(left)), (int(row), int(right)))


def _touches(orbit: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    return any((nr, nc) in occupied for r, c in orbit for nr, nc in neighbors4(r, c))


def _distinct_rows(cells: set[tuple[int, int]]) -> int:
    return len({r for r, _c in cells})


def _core_width(cells: set[tuple[int, int]]) -> int:
    cols = [c for _r, c in cells]
    return max(cols) - min(cols) + 1


def _grow_symmetric_core(
    rng: np.random.Generator,
    width: int,
    row0: int,
    band_h: int,
    axis_twice: int,
    core_area: int,
    max_offset: int,
) -> set[tuple[int, int]] | None:
    rows = list(range(int(row0), int(row0 + band_h)))
    start_row = int(rng.choice(rows))
    occupied = set(_orbit_cells(start_row, 0, axis_twice))
    if len(occupied) > core_area:
        return None

    while len(occupied) < core_area:
        remaining = core_area - len(occupied)
        candidates: list[tuple[tuple[tuple[int, int], ...], int]] = []
        for row in rows:
            for offset in range(max_offset + 1):
                orbit = _orbit_cells(row, offset, axis_twice)
                if any(not (0 <= c < width) for _r, c in orbit):
                    continue
                if any(cell in occupied for cell in orbit):
                    continue
                if len(orbit) > remaining:
                    continue
                if not _touches(orbit, occupied):
                    continue
                span = max(abs(axis_twice - 2 * c) for _r, c in orbit)
                candidates.append((orbit, span))

        if not candidates:
            return None

        spans = np.array([span for _orbit, span in candidates], dtype=float)
        probs = spans + 1.0
        probs /= probs.sum()
        orbit = candidates[int(rng.choice(len(candidates), p=probs))][0]
        occupied.update(orbit)

    if _distinct_rows(occupied) < 3:
        return None
    if _core_width(occupied) < 3:
        return None
    return occupied


def _side_extent(cells: set[tuple[int, int]], axis_twice: int, side: int) -> int:
    if side < 0:
        values = [axis_twice - 2 * c for _r, c in cells if 2 * c < axis_twice]
    else:
        values = [2 * c - axis_twice for _r, c in cells if 2 * c > axis_twice]
    return max(values, default=0)


def _noise_candidates(
    cluster: set[tuple[int, int]],
    row0: int,
    band_h: int,
    width: int,
    axis_twice: int,
    side: int,
) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for r, c in cluster:
        for nr, nc in neighbors4(r, c):
            if not (row0 <= nr < row0 + band_h and 0 <= nc < width):
                continue
            if (nr, nc) in cluster or (nr, nc) in seen:
                continue
            if side < 0 and not (2 * nc < axis_twice):
                continue
            if side > 0 and not (2 * nc > axis_twice):
                continue
            mc = _mirror_col(axis_twice, nc)
            if not (0 <= mc < width):
                continue
            if (nr, mc) in cluster:
                continue
            seen.add((nr, nc))
    return sorted(seen)


def _add_noise(
    rng: np.random.Generator,
    core: set[tuple[int, int]],
    row0: int,
    band_h: int,
    width: int,
    axis_twice: int,
    noise_count: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    side = -1 if bool(rng.integers(0, 2)) else 1
    cluster = set(core)
    noise: set[tuple[int, int]] = set()
    core_extent = _side_extent(core, axis_twice, side)

    for step in range(noise_count):
        candidates = _noise_candidates(cluster, row0, band_h, width, axis_twice, side)
        if not candidates:
            return None

        if step == 0:
            outward = [
                cell
                for cell in candidates
                if (
                    axis_twice - 2 * cell[1] > core_extent if side < 0 else 2 * cell[1] - axis_twice > core_extent
                )
            ]
            if outward:
                candidates = outward

        choice = candidates[int(rng.integers(0, len(candidates)))]
        cluster.add(choice)
        noise.add(choice)

    return cluster, noise


def _marker_cols(axis_twice: int, width: int, cluster_cols: list[int]) -> tuple[int, int] | None:
    leftmost = min(cluster_cols)
    rightmost = max(cluster_cols)
    if axis_twice % 2 == 0:
        axis = axis_twice // 2
        min_offset = max(axis - leftmost + 3, rightmost - axis + 3)
        max_offset = min(axis, width - 1 - axis)
        if min_offset > max_offset:
            return None
        offset = min_offset
        return axis - offset, axis + offset

    left_axis = axis_twice // 2
    right_axis = left_axis + 1
    min_offset = max(left_axis - leftmost + 3, rightmost - right_axis + 3)
    max_offset = min(left_axis, width - 1 - right_axis)
    if min_offset > max_offset:
        return None
    offset = min_offset
    return left_axis - offset, right_axis + offset


def _chebyshev_far(markers: list[tuple[int, int]], cluster: set[tuple[int, int]]) -> bool:
    for mr, mc in markers:
        for r, c in cluster:
            if max(abs(mr - r), abs(mc - c)) < 3:
                return False
    return True


def _build_object(
    rng: np.random.Generator,
    width: int,
    row0: int,
    band_h: int,
    used_axes: set[int],
    color: int,
) -> dict | None:
    rows = list(range(int(row0), int(row0 + band_h)))
    candidate_axes = [a for a in range(3, 2 * width - 3) if a not in used_axes]
    rng.shuffle(candidate_axes)

    for axis_twice in candidate_axes:
        max_offset = min(axis_twice // 2, (2 * width - 2 - axis_twice) // 2, 3)
        if max_offset < 1:
            continue

        noise_count = int(rng.integers(1, 4))
        feasible_core = [
            area
            for area in range(6, 13)
            if 8 <= area + noise_count <= 15 and (axis_twice % 2 == 0 or area % 2 == 0)
        ]
        if not feasible_core:
            continue

        core_area = int(rng.choice(feasible_core))
        core = _grow_symmetric_core(rng, width, row0, band_h, axis_twice, core_area, max_offset)
        if core is None:
            continue

        cluster_noise = _add_noise(rng, core, row0, band_h, width, axis_twice, noise_count)
        if cluster_noise is None:
            continue
        cluster, noise = cluster_noise

        cols = [c for _r, c in cluster]
        if min(cols) + max(cols) == axis_twice:
            continue

        marker_cols = _marker_cols(axis_twice, width, cols)
        if marker_cols is None:
            continue

        marker_row = int(rng.choice(rows))
        markers = [(marker_row, int(marker_cols[0])), (marker_row, int(marker_cols[1]))]
        if not _chebyshev_far(markers, cluster):
            continue

        kept = _denoise_cluster((row0 + band_h, width), cluster, axis_twice)
        if len(noise) != noise_count or len(cluster) != core_area + noise_count:
            continue
        if kept != cluster - noise:
            continue
        if len(kept) < 6:
            continue

        return {
            "color": int(color),
            "axis_twice": int(axis_twice),
            "cluster": cluster,
            "noise": noise,
            "markers": markers,
            "kept": kept,
        }

    return None


def _paint_objects(shape: tuple[int, int], objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["cluster"]:
            inp[r, c] = color
        for r, c in obj["markers"]:
            inp[r, c] = color
        for r, c in obj["kept"]:
            out[r, c] = color
    return inp, out


def _no_adjacent_cross_color(grid: np.ndarray) -> bool:
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == 0:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        other = int(grid[rr, cc])
                        if other != 0 and other != color:
                            return False
    return True


def _validate_instance(inp: np.ndarray, expected: np.ndarray, objects: list[dict]) -> None:
    assert inp.shape == expected.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert 3 <= len(objects) <= 4
    assert _no_adjacent_cross_color(inp)
    assert np.array_equal(main(inp), expected)

    axis_values = [int(obj["axis_twice"]) for obj in objects]
    assert len(axis_values) == len(set(axis_values)), "axes must be unique"

    for obj in objects:
        color = int(obj["color"])
        cluster, markers, axis_twice = _classify_color(inp, color)
        assert cluster == obj["cluster"]
        assert sorted(markers) == sorted(obj["markers"])
        assert axis_twice == int(obj["axis_twice"])
        assert 8 <= len(cluster) <= 15
        assert _chebyshev_far(markers, cluster)

        bbox = bbox_coords(np.asarray(sorted(cluster)))
        assert int(bbox[1] + bbox[3]) != axis_twice, "bbox center must not reveal the true axis"

        noise = {cell for cell in cluster if cell not in _denoise_cluster(inp.shape, cluster, axis_twice)}
        assert noise == obj["noise"]
        assert 1 <= len(noise) <= 3

        side_flags = {int(np.sign(2 * c - axis_twice)) for _r, c in noise}
        assert len(side_flags) == 1 and 0 not in side_flags

        for r, c in noise:
            mc = _mirror_col(axis_twice, c)
            assert 0 <= mc < inp.shape[1]
            assert (r, mc) not in cluster

        kept = _denoise_cluster(inp.shape, cluster, axis_twice)
        assert kept == obj["kept"]
        for r, c in kept:
            mc = _mirror_col(axis_twice, c)
            assert (r, mc) in cluster
            assert expected[r, c] == color
        for r, c in markers:
            assert expected[r, c] == 0

    assert not np.array_equal(inp, expected)


def _build_instance(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    n_colors = int(rng.integers(MIN_COLORS, MAX_COLORS + 1))
    band_heights = [int(rng.integers(4, 6)) for _ in range(n_colors)]
    width = int(rng.integers(16, 23))

    cursor = 1
    objects: list[dict] = []
    used_axes: set[int] = set()
    colors = [int(v) for v in rng.choice(COLOR_POOL, size=n_colors, replace=False)]

    for band_h, color in zip(band_heights, colors):
        obj = _build_object(rng, width, cursor, band_h, used_axes, color)
        if obj is None:
            return None
        objects.append(obj)
        used_axes.add(int(obj["axis_twice"]))
        cursor += band_h + 1

    height = cursor
    inp, expected = _paint_objects((height, width), objects)
    _validate_instance(inp, expected, objects)
    return inp, expected, objects


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        built = _build_instance(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
