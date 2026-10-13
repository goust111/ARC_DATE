from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.is_dir():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _orbit_coords(height: int, width: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r, c),
        (r, width - 1 - c),
        (height - 1 - r, c),
        (height - 1 - r, width - 1 - c),
    )


def _orbit_signature(values: list[int]) -> tuple[int, int] | None:
    counts = Counter(values)
    if len(counts) == 1:
        return None
    if sorted(counts.values()) != [1, 3]:
        raise AssertionError(f"invalid orbit multiplicities: {counts}")
    minority = next(color for color, count in counts.items() if count == 1)
    majority = next(color for color, count in counts.items() if count == 3)
    return int(minority), int(majority)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fold all asymmetry-breaking minority pixels from a double-mirror wallpaper into one quadrant.

    Input structure:
    - The grid has even height and width and is otherwise filled by a multicolor wallpaper mirrored across both center axes.
    - Some four-cell mirror orbits contain one defect pixel whose color differs from the other three cells in that orbit.
    - Distinct defects belong to distinct four-cell orbits, so their folded representatives never collide.

    Rule:
    - Inspect every four-cell orbit induced by the horizontal and vertical mirror axes.
    - If an orbit contains three cells of one color and one cell of another color, record the minority color as the defect signal.
    - Return a black grid of half the input height and half the input width, placing each defect color at that orbit's upper-left representative coordinate.

    Key insight:
    - The symmetric wallpaper is only a carrier; the answer appears when every four-way mirror orbit is collapsed onto its upper-left representative and only the minority color survives.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height % 2 == 0 and width % 2 == 0, "grid must have even height and width"

    half_h, half_w = height // 2, width // 2
    out = np.full((half_h, half_w), Color.BLACK, dtype=int)

    for r in range(half_h):
        for c in range(half_w):
            values = [int(g[rr, cc]) for rr, cc in _orbit_coords(height, width, r, c)]
            signature = _orbit_signature(values)
            if signature is None:
                continue
            minority, _majority = signature
            out[r, c] = minority

    return out


def _expand_macro_pattern(macro: np.ndarray, height: int, width: int) -> np.ndarray:
    scale_r = (height + macro.shape[0] - 1) // macro.shape[0]
    scale_c = (width + macro.shape[1] - 1) // macro.shape[1]
    return np.kron(macro, np.ones((scale_r, scale_c), dtype=int))[:height, :width]


def _build_wallpaper_quadrant(half_h: int, half_w: int, rng: np.random.Generator) -> np.ndarray:
    palette_size = int(rng.integers(3, min(6, len(Color.NOT_BLACK)) + 1))
    palette = rng.choice(np.asarray(Color.NOT_BLACK, dtype=int), size=palette_size, replace=False)

    macro_h = int(rng.integers(2, min(half_h, 6) + 1))
    macro_w = int(rng.integers(2, min(half_w, 6) + 1))
    weights = rng.dirichlet(np.full(palette_size, 0.8))
    macro = rng.choice(palette, size=(macro_h, macro_w), p=weights)
    quadrant = _expand_macro_pattern(macro, half_h, half_w)

    for _ in range(int(rng.integers(2, 6))):
        color = int(rng.choice(palette))
        mode = int(rng.integers(4))
        if mode == 0:
            r0, r1 = sorted(map(int, rng.integers(0, half_h, size=2)))
            c0, c1 = sorted(map(int, rng.integers(0, half_w, size=2)))
            quadrant[r0 : r1 + 1, c0 : c1 + 1] = color
        elif mode == 1:
            r = int(rng.integers(0, half_h))
            thickness = int(rng.integers(1, min(3, half_h) + 1))
            quadrant[r : min(half_h, r + thickness), :] = color
        elif mode == 2:
            c = int(rng.integers(0, half_w))
            thickness = int(rng.integers(1, min(3, half_w) + 1))
            quadrant[:, c : min(half_w, c + thickness)] = color
        else:
            center_r = int(rng.integers(0, half_h))
            center_c = int(rng.integers(0, half_w))
            radius = int(rng.integers(0, 2))
            for rr in range(max(0, center_r - radius), min(half_h, center_r + radius + 1)):
                for cc in range(max(0, center_c - radius), min(half_w, center_c + radius + 1)):
                    quadrant[rr, cc] = color

    if np.unique(quadrant).size < 2:
        quadrant[0, 0] = int(palette[0])
        quadrant[-1, -1] = int(palette[-1])
    return quadrant


def _reflect_quadrant(quadrant: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [quadrant, quadrant[:, ::-1]],
            [quadrant[::-1, :], quadrant[::-1, ::-1]],
        ]
    )


def _random_partition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(map(int, rng.choice(np.arange(1, total), size=parts - 1, replace=False)))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _neighbor_cells(cell: tuple[int, int], half_h: int, half_w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < half_h and 0 <= cc < half_w:
                out.append((rr, cc))
    return out


def _seed_candidates(half_h: int, half_w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    all_cells = [(r, c) for r in range(half_h) for c in range(half_w)]
    special = [
        cell
        for cell in all_cells
        if cell[0] in (0, half_h - 1) or cell[1] in (0, half_w - 1)
    ]
    return all_cells, special


def _grow_component(
    size: int,
    used: set[tuple[int, int]],
    half_h: int,
    half_w: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    all_cells, special_cells = _seed_candidates(half_h, half_w)
    seed_pool = [cell for cell in special_cells if cell not in used]
    if not seed_pool:
        seed_pool = [cell for cell in all_cells if cell not in used]
    if not seed_pool:
        return None

    comp = {seed_pool[int(rng.integers(len(seed_pool)))]}
    while len(comp) < size:
        frontier = {
            neighbor
            for cell in comp
            for neighbor in _neighbor_cells(cell, half_h, half_w)
            if neighbor not in used and neighbor not in comp
        }
        if frontier:
            frontier_list = sorted(frontier)
            comp.add(frontier_list[int(rng.integers(len(frontier_list)))])
            continue

        fallback = [cell for cell in all_cells if cell not in used and cell not in comp]
        if not fallback:
            return None
        comp.add(fallback[int(rng.integers(len(fallback)))])
    return comp


def _sample_signal_coords(half_h: int, half_w: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    area = half_h * half_w
    max_signals = min(12, max(4, area // 2))

    for _ in range(200):
        count = int(rng.integers(4, max_signals + 1))
        max_components = min(3, count)
        component_count = int(rng.integers(1, max_components + 1))
        component_sizes = _random_partition(count, component_count, rng)

        used: set[tuple[int, int]] = set()
        coords: set[tuple[int, int]] = set()
        for size in component_sizes:
            comp = _grow_component(size, used, half_h, half_w, rng)
            if comp is None:
                break
            coords |= comp
            used |= comp
        if len(coords) != count:
            continue

        rows = {r for r, _c in coords}
        cols = {c for _r, c in coords}
        touches_border = any(r in (0, half_h - 1) or c in (0, half_w - 1) for r, c in coords)
        has_adjacent_pair = any(
            max(abs(r0 - r1), abs(c0 - c1)) == 1
            for i, (r0, c0) in enumerate(sorted(coords))
            for r1, c1 in sorted(coords)[i + 1 :]
        )
        if len(rows) < 2 or len(cols) < 2 or not touches_border or not has_adjacent_pair:
            continue
        return sorted(coords)

    return None


def _assign_defect_quadrants(count: int, rng: np.random.Generator) -> list[int]:
    quadrants = [0, 1, 2, 3]
    while len(quadrants) < count:
        quadrants.append(int(rng.integers(0, 4)))
    rng.shuffle(quadrants)
    return quadrants


def _actual_defect_coord(
    half_h: int,
    half_w: int,
    rep_r: int,
    rep_c: int,
    quadrant_index: int,
) -> tuple[int, int]:
    variants = (
        (rep_r, rep_c),
        (rep_r, 2 * half_w - 1 - rep_c),
        (2 * half_h - 1 - rep_r, rep_c),
        (2 * half_h - 1 - rep_r, 2 * half_w - 1 - rep_c),
    )
    return variants[int(quadrant_index)]


def _build_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    half_h = int(rng.integers(4, 16))
    half_w = int(rng.integers(4, 16))

    quadrant = _build_wallpaper_quadrant(half_h, half_w, rng)
    signal_coords = _sample_signal_coords(half_h, half_w, rng)
    if signal_coords is None:
        return None

    full = _reflect_quadrant(quadrant)
    expected = np.full((half_h, half_w), Color.BLACK, dtype=int)
    defect_quadrants = _assign_defect_quadrants(len(signal_coords), rng)
    assignments: list[dict[str, int]] = []

    for (rep_r, rep_c), defect_quadrant in zip(signal_coords, defect_quadrants):
        base_color = int(quadrant[rep_r, rep_c])
        candidates = [color for color in Color.NOT_BLACK if int(color) != base_color]
        if not candidates:
            return None
        defect_color = int(candidates[int(rng.integers(len(candidates)))])
        expected[rep_r, rep_c] = defect_color
        rr, cc = _actual_defect_coord(half_h, half_w, rep_r, rep_c, defect_quadrant)
        full[rr, cc] = defect_color
        assignments.append(
            {
                "rep_r": rep_r,
                "rep_c": rep_c,
                "quadrant": defect_quadrant,
                "defect_color": defect_color,
                "base_color": base_color,
            }
        )

    meta = {
        "quadrants": defect_quadrants,
        "assignments": assignments,
        "signal_count": len(signal_coords),
    }
    return full, expected, meta


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict[str, object]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if inp.shape[0] % 2 or inp.shape[1] % 2:
        return False
    if expected.shape != (inp.shape[0] // 2, inp.shape[1] // 2):
        return False

    half_h, half_w = expected.shape
    observed = np.full_like(expected, Color.BLACK)
    for r in range(half_h):
        for c in range(half_w):
            values = [int(inp[rr, cc]) for rr, cc in _orbit_coords(inp.shape[0], inp.shape[1], r, c)]
            counts = Counter(values)
            if len(counts) == 1:
                continue
            if sorted(counts.values()) != [1, 3]:
                return False
            observed[r, c] = next(int(color) for color, count in counts.items() if count == 1)

    if not np.array_equal(observed, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    quadrants = list(meta["quadrants"])
    signal_count = int(meta["signal_count"])
    if signal_count < 4 or len(quadrants) != signal_count:
        return False
    if set(quadrants) != {0, 1, 2, 3}:
        return False

    quadrant_counts = Counter(quadrants)
    if any(count <= 0 or count >= signal_count for count in quadrant_counts.values()):
        return False

    visible_by_quadrant = [0, 0, 0, 0]
    for assignment in meta["assignments"]:
        visible_by_quadrant[int(assignment["quadrant"])] += 1
        rep_r = int(assignment["rep_r"])
        rep_c = int(assignment["rep_c"])
        defect_color = int(assignment["defect_color"])
        base_color = int(assignment["base_color"])
        if defect_color == base_color or int(expected[rep_r, rep_c]) != defect_color:
            return False

    return all(0 < count < signal_count for count in visible_by_quadrant)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    built = _build_case(rng)
    if built is None:
        return None
    inp, expected, meta = built
    return inp if _validate_case(inp, expected, meta) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-defect folding puzzle")
