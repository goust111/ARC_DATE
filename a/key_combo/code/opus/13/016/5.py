import numpy as np


BACKGROUND = 0
ANOMALY = 2
RING_COLORS = (1, 3, 4, 6)
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _d4_orbit(r: int, c: int, size: int) -> tuple[tuple[int, int], ...]:
    n = size - 1
    pts = {
        (r, c),
        (c, n - r),
        (n - r, n - c),
        (n - c, r),
        (r, n - c),
        (n - r, c),
        (c, r),
        (n - c, n - r),
    }
    return tuple(sorted(pts))


def _ring_coords(size: int, depth: int) -> list[tuple[int, int]]:
    top = left = depth
    bottom = right = size - 1 - depth
    coords: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        coords.append((top, c))
    for r in range(top + 1, bottom):
        coords.append((r, right))
    for c in range(right, left - 1, -1):
        coords.append((bottom, c))
    for r in range(bottom - 1, top, -1):
        coords.append((r, left))
    return coords


def _edge_noncorner_coords(size: int, depth: int) -> list[tuple[int, int]]:
    top = left = depth
    bottom = right = size - 1 - depth
    coords: list[tuple[int, int]] = []
    for c in range(left + 1, right):
        coords.append((top, c))
        coords.append((bottom, c))
    for r in range(top + 1, bottom):
        coords.append((r, left))
        coords.append((r, right))
    return sorted(set(coords))


def _ring_edge_orbits(size: int, depth: int) -> list[tuple[tuple[int, int], ...]]:
    candidates = set(_edge_noncorner_coords(size, depth))
    orbits: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[int, int]] = set()
    for pos in sorted(candidates):
        if pos in seen:
            continue
        orbit = tuple(p for p in _d4_orbit(pos[0], pos[1], size) if p in candidates)
        seen.update(orbit)
        orbits.append(tuple(sorted(orbit)))
    return orbits


def _layer_depth(r: int, c: int, size: int) -> int:
    return min(r, c, size - 1 - r, size - 1 - c)


def _detect_layers(grid: np.ndarray) -> list[dict[str, object]]:
    size = int(grid.shape[0])
    layers: list[dict[str, object]] = []
    for depth in range(size // 2):
        if size - 2 * depth < 3:
            break
        coords = _ring_coords(size, depth)
        values = [int(grid[r, c]) for r, c in coords]
        if all(v == BACKGROUND for v in values):
            break
        base_colors = sorted({v for v in values if v not in (BACKGROUND, ANOMALY)})
        assert len(base_colors) == 1, "each ring must have one original non-red color"
        layers.append(
            {
                "depth": depth,
                "color": base_colors[0],
                "coords": coords,
            }
        )
    return layers


def _find_anomalies(grid: np.ndarray) -> set[tuple[int, int]]:
    size = int(grid.shape[0])
    anomalies: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = _d4_orbit(r, c, size)
            seen.update(orbit)
            colors = {int(grid[rr, cc]) for rr, cc in orbit}
            if ANOMALY not in colors:
                continue
            if not any(color not in (BACKGROUND, ANOMALY) for color in colors):
                continue
            for rr, cc in orbit:
                if int(grid[rr, cc]) == ANOMALY:
                    anomalies.add((rr, cc))
    return anomalies


def _render_output(counts: list[int], colors: list[int]) -> np.ndarray:
    width = max(counts)
    assert width > 0, "at least one ring must contain an anomaly"
    out = np.zeros((len(counts), width), dtype=int)
    for row, (count, color) in enumerate(zip(counts, colors, strict=True)):
        if count == 0:
            continue
        start = (width - count) // 2
        out[row, start : start + count] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect D4-symmetry anomalies on concentric square rings and compress each ring into a centered count bar.

    Input structure:
    - The input is an odd square grid on a black background.
    - It contains 3-4 concentric one-cell-thick square rings, each with its own non-red color.
    - Some non-corner edge cells on those rings may be replaced by red anomaly pixels.

    Rule:
    - Identify the concentric ring layers from outermost to innermost and recover each layer's original color from its non-red pixels.
    - For every cell, consider its D4 symmetry orbit under horizontal, vertical, diagonal, and rotational square symmetries.
    - If an orbit mixes red with a non-red ring color, count the red cells in that orbit as anomalies, then total anomalies per ring.
    - Output one row per ring, with width equal to the largest anomaly count, and draw each count as a horizontally centered segment in that ring's color.

    Key insight:
    - The concentric rings define the grouping, while D4 symmetry defines which red pixels are true anomalies; the output is a histogram-like compression of those per-ring anomaly counts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = map(int, g.shape)
    assert rows == cols and rows % 2 == 1, "expected an odd square grid"

    layers = _detect_layers(g)
    assert layers, "expected at least one concentric ring"
    anomalies = _find_anomalies(g)

    counts: list[int] = []
    colors: list[int] = []
    for layer in layers:
        depth = int(layer["depth"])
        color = int(layer["color"])
        count = sum(1 for r, c in anomalies if _layer_depth(r, c, rows) == depth)
        counts.append(count)
        colors.append(color)

    return _render_output(counts, colors)


# --------------- generator ---------------


def _choose_layout(call_id: int) -> tuple[int, int]:
    k = 3 if call_id % 2 == 0 else 4
    size_choices = {
        3: (9, 11, 13, 15, 17),
        4: (11, 13, 15, 17),
    }[k]
    if call_id % 6 == 0:
        size = size_choices[0]
    elif call_id % 6 == 3:
        size = size_choices[-1]
    else:
        size = int(_RNG.choice(size_choices))
    return size, k


def _sample_counts(size: int, k: int, call_id: int) -> list[int]:
    max_counts = []
    for depth in range(k):
        side = size - 2 * depth
        max_counts.append(3 if side == 3 else 4)

    for _ in range(200):
        counts = [int(_RNG.integers(0, limit + 1)) for limit in max_counts]
        if max(counts) == 0:
            continue
        mode = call_id % 4
        if mode == 0 and not any(c == 0 for c in counts):
            continue
        if mode == 1 and (max(counts) < 2 or len(set(counts)) < 2):
            continue
        if mode == 2:
            target = 4 if any(limit == 4 for limit in max_counts) else 3
            if target not in counts:
                continue
        if mode == 3 and not (any(c == 0 for c in counts) and any(c >= 2 for c in counts)):
            continue
        return counts

    counts = [0] * k
    counts[int(_RNG.integers(0, k))] = 1
    return counts


def _sample_anomaly_positions(size: int, depth: int, count: int) -> set[tuple[int, int]] | None:
    if count == 0:
        return set()
    coords = _edge_noncorner_coords(size, depth)
    orbits = _ring_edge_orbits(size, depth)
    if count > len(coords):
        return None
    for _ in range(300):
        picks = _RNG.choice(len(coords), size=count, replace=False)
        chosen = {coords[int(idx)] for idx in picks}
        if any(set(orbit).issubset(chosen) for orbit in orbits):
            continue
        return chosen
    return None


def _build_input(size: int, colors: list[int], counts: list[int]) -> np.ndarray | None:
    grid = np.zeros((size, size), dtype=int)
    for depth, color in enumerate(colors):
        for r, c in _ring_coords(size, depth):
            grid[r, c] = int(color)

    for depth, count in enumerate(counts):
        chosen = _sample_anomaly_positions(size, depth, count)
        if chosen is None:
            return None
        for r, c in chosen:
            grid[r, c] = ANOMALY

    return grid


def _gen_attempt(call_id: int) -> np.ndarray | None:
    size, k = _choose_layout(call_id)
    colors = list(_RNG.choice(RING_COLORS, size=k, replace=False))
    counts = _sample_counts(size, k, call_id)

    grid = _build_input(size, colors, counts)
    if grid is None:
        return None

    expected = _render_output(counts, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(MAX_GENERATION_ATTEMPTS):
        call_id = _GEN_COUNTER
        _GEN_COUNTER += 1
        grid = _gen_attempt(call_id)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 5")
