from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
FAMILY_NAMES = ("hv", "diag")
MAX_GENERATION_ATTEMPTS = 400
MIN_GRID_SIZE = 9
MAX_GRID_SIZE = 21
PALETTE_MIN = 4
PALETTE_MAX = 6
MIN_DECOY_OUTSIDE_COUNT = 4


TEMPLATE_LIBRARY = [
    np.array(
        [
            [1, 1, 0, 2],
            [0, 1, 2, 2],
            [3, 0, 2, 0],
            [3, 3, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 0, 2],
            [3, 0, 1, 2, 2],
            [3, 3, 0, 2, 0],
            [0, 3, 4, 4, 4],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 2, 2],
            [1, 3, 0, 2],
            [1, 3, 4, 0],
            [0, 3, 4, 4],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [0, 2, 1, 3],
            [2, 2, 0, 3],
            [0, 2, 3, 3],
        ],
        dtype=int,
    ),
]


def _hv_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    orbit = {
        (int(r), int(c)),
        (n - 1 - int(r), int(c)),
        (int(r), n - 1 - int(c)),
        (n - 1 - int(r), n - 1 - int(c)),
    }
    return tuple(sorted(orbit))


def _diag_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    orbit = {
        (int(r), int(c)),
        (int(c), int(r)),
        (n - 1 - int(c), n - 1 - int(r)),
        (n - 1 - int(r), n - 1 - int(c)),
    }
    return tuple(sorted(orbit))


def _orbit_fn(family: str):
    if family == "hv":
        return _hv_orbit
    if family == "diag":
        return _diag_orbit
    raise AssertionError(f"unknown family {family!r}")


def _iter_orbits(n: int, family: str):
    orbit_fn = _orbit_fn(family)
    seen: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = orbit_fn(n, r, c)
            seen.update(orbit)
            yield orbit


def _analyze_family(grid: np.ndarray, family: str) -> tuple[int, list[tuple[int, int, int]]]:
    n = int(grid.shape[0])
    total_cost = 0
    anomalies: list[tuple[int, int, int]] = []
    for orbit in _iter_orbits(n, family):
        values = [int(grid[r, c]) for r, c in orbit]
        counts = Counter(values)
        total_cost += len(orbit) - counts.most_common(1)[0][1]
        if len(orbit) != 4:
            continue
        freq = sorted(counts.values())
        if freq != [1, 3]:
            continue
        majority = next(color for color, count in counts.items() if count == 3)
        minority = next(color for color, count in counts.items() if count == 1)
        for (r, c), value in zip(orbit, values):
            if value == minority:
                anomalies.append((int(r), int(c), int(majority)))
                break
    return total_cost, anomalies


def _choose_family(grid: np.ndarray) -> tuple[str, list[tuple[int, int, int]]]:
    hv_cost, hv_anomalies = _analyze_family(grid, "hv")
    diag_cost, diag_anomalies = _analyze_family(grid, "diag")
    if hv_cost < diag_cost:
        return "hv", hv_anomalies
    if diag_cost < hv_cost:
        return "diag", diag_anomalies
    if len(hv_anomalies) >= len(diag_anomalies):
        return "hv", hv_anomalies
    return "diag", diag_anomalies


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the nearly valid mirror-symmetry family, then crop and repaint the broken minority cells with their orbit majorities.

    Input structure:
    - The input is a dense square color pattern.
    - One symmetry family is almost valid: either horizontal+vertical mirrors or the two diagonal mirrors.
    - A small set of off-axis cells breaks that family, with each affected four-cell orbit containing exactly three matching cells and one mismatching cell.

    Rule:
    - Compare the two symmetry families and choose the one needing fewer cell changes to become perfectly symmetric.
    - Within that chosen family, collect every four-cell orbit whose colors have a 3-to-1 split.
    - Treat the minority cell as the anomaly, and use the majority color as the cell's corrected value.
    - Output the minimal bounding box covering all anomaly cells, painting only those anomaly positions with their corrected colors and painting everything else black.

    Key insight:
    - Symmetry is only a filter for locating errors: the answer keeps the error positions but recolors them to the orbit majority, so copying the current anomaly colors yields the decoy instead of the hidden picture.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square grid"

    _family, anomalies = _choose_family(g)
    assert anomalies, "expected at least one broken four-cell orbit"

    rows = [r for r, _, _ in anomalies]
    cols = [c for _, c, _ in anomalies]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c, color in anomalies:
        out[r - r0, c - c0] = int(color)
    return out


def _transform_template(template: np.ndarray) -> np.ndarray:
    out = np.array(template, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    return np.rot90(out, k).copy()


def _template_coords(template: np.ndarray) -> list[tuple[int, int, int]]:
    coords: list[tuple[int, int, int]] = []
    h, w = template.shape
    for r in range(h):
        for c in range(w):
            value = int(template[r, c])
            if value != 0:
                coords.append((r, c, value))
    return coords


def _map_template_colors(template: np.ndarray, palette: list[int]) -> tuple[np.ndarray, dict[int, int]]:
    labels = sorted({int(v) for v in template.reshape(-1) if int(v) != 0})
    assert labels
    assert len(labels) <= len(palette)
    chosen = random.sample(palette, len(labels))
    color_map = {label: int(color) for label, color in zip(labels, chosen)}
    actual = np.zeros_like(template)
    for label, color in color_map.items():
        actual[template == label] = color
    return actual, color_map


def _decoy_mapping(colors: list[int]) -> dict[int, int]:
    assert len(colors) >= 2
    shift = random.randint(1, len(colors) - 1)
    return {color: colors[(idx + shift) % len(colors)] for idx, color in enumerate(colors)}


def _family_transform(n: int, family: str, variant: int):
    if family == "hv":
        transforms = (
            lambda r, c: (r, c),
            lambda r, c: (r, n - 1 - c),
            lambda r, c: (n - 1 - r, c),
            lambda r, c: (n - 1 - r, n - 1 - c),
        )
    else:
        transforms = (
            lambda r, c: (r, c),
            lambda r, c: (c, r),
            lambda r, c: (n - 1 - c, n - 1 - r),
            lambda r, c: (n - 1 - r, n - 1 - c),
        )
    return transforms[int(variant) % 4]


def _diag_candidates(n: int, template: np.ndarray) -> list[tuple[int, int]]:
    coords = _template_coords(template)
    h, w = template.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(n - h + 1):
        for c0 in range(n - w + 1):
            ok = True
            for dr, dc, _ in coords:
                r = r0 + dr
                c = c0 + dc
                if not (r < c and r + c < n - 1):
                    ok = False
                    break
            if ok:
                candidates.append((r0, c0))
    return candidates


def _pick_layout(n: int, family: str, template: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    coords = _template_coords(template)
    h, w = template.shape
    if family == "hv":
        max_r0 = n // 2 - h
        max_c0 = n // 2 - w
        if max_r0 < 0 or max_c0 < 0:
            raise ValueError("template does not fit the hv fundamental region")
        r0 = random.randint(0, max_r0)
        c0 = random.randint(0, max_c0)
    else:
        candidates = _diag_candidates(n, template)
        if not candidates:
            raise ValueError("template does not fit the diagonal fundamental region")
        r0, c0 = random.choice(candidates)

    transform = _family_transform(n, family, random.randrange(4))
    positions: list[tuple[int, int]] = []
    placed = np.zeros_like(template)
    for dr, dc, value in coords:
        ar, ac = transform(r0 + dr, c0 + dc)
        positions.append((int(ar), int(ac)))
        placed[dr, dc] = value
    assert len(set(positions)) == len(positions)
    return positions, placed


def _build_base_grid(
    n: int,
    family: str,
    palette: list[int],
    hidden_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    seen: set[tuple[int, int]] = set()
    orbit_fn = _orbit_fn(family)
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = orbit_fn(n, r, c)
            seen.update(orbit)
            forced = [hidden_colors[pos] for pos in orbit if pos in hidden_colors]
            assert len(forced) <= 1, "anomaly positions must occupy distinct orbits"
            color = forced[0] if forced else random.choice(palette)
            for rr, cc in orbit:
                grid[rr, cc] = int(color)
    return grid


def _render_expected(positions: list[tuple[int, int]], hidden_colors: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for r, c in positions:
        out[r - r0, c - c0] = int(hidden_colors[(r, c)])
    return out


def _decoy_colors_ok(grid: np.ndarray, positions: list[tuple[int, int]], decoy_colors: dict[tuple[int, int], int]) -> bool:
    anomaly_set = set(positions)
    counts: Counter[int] = Counter()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if (r, c) in anomaly_set:
                continue
            counts[int(grid[r, c])] += 1
    needed = {int(decoy_colors[pos]) for pos in positions}
    return all(counts[color] >= MIN_DECOY_OUTSIDE_COUNT for color in needed)


def _family_scores_ok(grid: np.ndarray, family: str, anomaly_count: int) -> bool:
    hv_cost, _ = _analyze_family(grid, "hv")
    diag_cost, _ = _analyze_family(grid, "diag")
    good = hv_cost if family == "hv" else diag_cost
    bad = diag_cost if family == "hv" else hv_cost
    return bad >= good + max(8, anomaly_count)


def _gen_attempt() -> np.ndarray | None:
    family = random.choice(FAMILY_NAMES)
    template = _transform_template(random.choice(TEMPLATE_LIBRARY))
    min_n = max(MIN_GRID_SIZE, 2 * max(template.shape) + 1)
    n = random.randint(min_n, MAX_GRID_SIZE)
    palette_size = random.randint(PALETTE_MIN, PALETTE_MAX)
    palette = random.sample(list(range(1, 10)), palette_size)

    hidden_template, color_map = _map_template_colors(template, palette)
    try:
        positions, _placed = _pick_layout(n, family, hidden_template)
    except ValueError:
        return None

    hidden_coords = _template_coords(hidden_template)
    assert len(hidden_coords) == len(positions)
    hidden_colors = {
        positions[idx]: int(color)
        for idx, (_, _, color) in enumerate(hidden_coords)
    }

    cycle = _decoy_mapping(list(color_map.values()))
    decoy_colors = {pos: int(cycle[color]) for pos, color in hidden_colors.items()}

    grid = _build_base_grid(n, family, palette, hidden_colors)
    for pos, decoy in decoy_colors.items():
        grid[pos] = int(decoy)

    if not _decoy_colors_ok(grid, positions, decoy_colors):
        return None
    if not _family_scores_ok(grid, family, len(positions)):
        return None

    expected = _render_expected(positions, hidden_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
