from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache

import numpy as np


OUTPUT_BG = 0
INPUT_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300


@lru_cache(maxsize=None)
def _all_orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            orbit = _d4_orbit(n, r, c)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return tuple(orbits)


def _d4_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    pts = {
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    }
    return tuple(sorted(pts))


def _defect_counts(grid: np.ndarray) -> Counter[int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "input must be a square grid"

    counts: Counter[int] = Counter()
    for orbit in _all_orbits(int(g.shape[0])):
        values = [int(g[r, c]) for r, c in orbit]
        freq = Counter(values)
        if len(freq) == 1:
            continue

        majority_color, majority_count = freq.most_common(1)[0]
        assert len(orbit) >= 4, "defective orbits must have size at least four"
        assert majority_count == len(orbit) - 1, "each non-uniform orbit must have exactly one defect"

        for color in values:
            if color != majority_color:
                counts[color] += 1

    assert counts, "expected at least one defect color"
    return counts


def _render_histogram(defect_counts: Counter[int]) -> np.ndarray:
    ordered = sorted(defect_counts.items(), key=lambda item: item[1])
    width = max(count for _color, count in ordered)
    out = np.full((len(ordered), width), OUTPUT_BG, dtype=int)
    for r, (color, count) in enumerate(ordered):
        out[r, :count] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize symmetry-breaking defect colors as histogram bars ordered by increasing defect count.

    Input structure:
    - The input is a fully colored square grid with no explicit background.
    - The intended base pattern is invariant under horizontal, vertical, and both diagonal reflections.
    - A few single cells have been recolored, with at most one defect in any D4 symmetry orbit and every defective orbit containing at least four cells.

    Rule:
    - Partition the grid into D4 reflection orbits.
    - In each orbit, treat the majority color as the true symmetric color and the lone minority cell as the defect.
    - Count how many defects use each defect color.
    - Output one row per defect color, sorted by increasing count, as a left-aligned solid bar of that color on black with width equal to the largest count.

    Key insight:
    - Defect colors are not recognizable by palette alone because they also appear elsewhere legitimately; the only reliable cue is that one cell disagrees with the majority of its symmetry orbit.
    """
    return _render_histogram(_defect_counts(grid))


@lru_cache(maxsize=None)
def _invariants(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rr, cc = np.indices((n, n))
    edge = np.minimum.reduce([rr, cc, n - 1 - rr, n - 1 - cc])
    diag = np.minimum(np.abs(rr - cc), np.abs(rr + cc - (n - 1)))
    dr2 = np.abs(2 * rr - (n - 1))
    dc2 = np.abs(2 * cc - (n - 1))
    radius = np.maximum(dr2, dc2)
    diamond = dr2 + dc2
    return edge, diag, radius, diamond


def _sample_cuts(feature: np.ndarray, max_bins: int = 4) -> tuple[int, ...]:
    values = sorted({int(v) for v in feature.flat})
    n_bins = random.randint(2, min(max_bins, len(values)))
    cut_positions = sorted(random.sample(range(1, len(values)), n_bins - 1))
    return tuple(values[pos] for pos in cut_positions)


def _color_table(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    cycle = random.sample(palette, len(palette))
    p = len(cycle)
    row_offsets = [random.randrange(p) for _ in range(rows)]
    col_step = random.randrange(1, p)
    phase = random.randrange(p)

    table = np.empty((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            table[r, c] = cycle[(phase + row_offsets[r] + col_step * c) % p]
    return table


def _build_base_pattern(n: int) -> np.ndarray:
    palette_size = random.randint(4, 6)
    palette = random.sample(list(INPUT_COLORS), palette_size)
    edge, diag, radius, diamond = _invariants(n)
    feature_pairs = [(edge, diag), (edge, radius), (diag, diamond)]
    feature_a, feature_b = random.choice(feature_pairs)

    cuts_a = _sample_cuts(feature_a)
    cuts_b = _sample_cuts(feature_b)
    bins_a = np.digitize(feature_a, cuts_a)
    bins_b = np.digitize(feature_b, cuts_b)
    table = _color_table(len(cuts_a) + 1, len(cuts_b) + 1, palette)
    return table[bins_a, bins_b]


def _orbit_infos(grid: np.ndarray) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for orbit in _all_orbits(int(grid.shape[0])):
        color = int(grid[orbit[0]])
        infos.append(
            {
                "coords": orbit,
                "rep": orbit[0],
                "size": len(orbit),
                "color": color,
            }
        )
    return infos


def _eligible_defect_colors(base: np.ndarray, orbit_infos: list[dict[str, object]]) -> list[int]:
    orbit_counts = Counter(int(info["color"]) for info in orbit_infos)
    cell_counts = Counter(int(v) for v in base.flat)
    return [
        color
        for color, orbit_count in orbit_counts.items()
        if orbit_count >= 4 and cell_counts[color] >= 8
    ]


def _sample_defect_spec(colors: list[int], host_orbits: list[dict[str, object]]) -> list[tuple[int, int]] | None:
    max_colors = min(4, len(colors))
    if max_colors < 2:
        return None

    for _ in range(60):
        n_colors = random.randint(2, max_colors)
        chosen_colors = random.sample(colors, n_colors)
        max_count = min(5, len(host_orbits))
        if max_count < n_colors:
            continue

        counts = sorted(random.sample(range(1, max_count + 1), n_colors))
        if sum(counts) > len(host_orbits):
            continue

        capacities = {
            color: sum(1 for info in host_orbits if int(info["color"]) != color)
            for color in chosen_colors
        }
        if all(count <= capacities[color] for color, count in zip(chosen_colors, counts)):
            return list(zip(chosen_colors, counts))
    return None


def _pick_host_orbits(
    host_orbits: list[dict[str, object]],
    defect_spec: list[tuple[int, int]],
) -> list[tuple[dict[str, object], int]] | None:
    assignments: list[tuple[dict[str, object], int]] = []
    used_orbits: set[tuple[tuple[int, int], ...]] = set()
    used_reps: list[tuple[int, int]] = []

    for defect_color, count in sorted(defect_spec, key=lambda item: item[1], reverse=True):
        options = [info for info in host_orbits if info["coords"] not in used_orbits and int(info["color"]) != defect_color]
        random.shuffle(options)

        chosen: list[dict[str, object]] = []
        for info in options:
            rep = tuple(info["rep"])
            if all(max(abs(rep[0] - rr), abs(rep[1] - cc)) >= 2 for rr, cc in used_reps):
                chosen.append(info)
                if len(chosen) == count:
                    break

        if len(chosen) < count:
            chosen = options[:count]
        if len(chosen) < count:
            return None

        for info in chosen:
            used_orbits.add(info["coords"])
            used_reps.append(tuple(info["rep"]))
            assignments.append((info, defect_color))

    return assignments


def _inject_defects(base: np.ndarray, assignments: list[tuple[dict[str, object], int]]) -> tuple[np.ndarray, Counter[int]]:
    grid = base.copy()
    expected: Counter[int] = Counter()
    for info, defect_color in assignments:
        coords = list(info["coords"])
        defect_r, defect_c = random.choice(coords)
        grid[defect_r, defect_c] = defect_color
        expected[defect_color] += 1
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(10, 20)
    base = _build_base_pattern(n)
    if len(set(map(int, base.flat))) < 4:
        return None

    orbit_infos = _orbit_infos(base)
    host_orbits = [info for info in orbit_infos if int(info["size"]) >= 4]
    if len(host_orbits) < 6:
        return None

    defect_colors = _eligible_defect_colors(base, orbit_infos)
    defect_spec = _sample_defect_spec(defect_colors, host_orbits)
    if defect_spec is None:
        return None

    assignments = _pick_host_orbits(host_orbits, defect_spec)
    if assignments is None:
        return None

    inp, expected = _inject_defects(base, assignments)
    if not np.array_equal(main(inp), _render_histogram(expected)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-defect histogram input")
