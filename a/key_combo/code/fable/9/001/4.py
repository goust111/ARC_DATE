from __future__ import annotations

from collections import Counter
import random

import numpy as np


ALL_COLORS = tuple(range(10))
GEN_RETRIES = 400
GRID_SIDE_OPTIONS = (6, 8, 10)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rot90_orbit(side: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = side - 1
    return (
        (r, c),
        (c, last - r),
        (last - r, last - c),
        (last - c, r),
    )


_ORBIT_CACHE: dict[int, list[tuple[tuple[int, int], ...]]] = {}


def _orbit_list(side: int) -> list[tuple[tuple[int, int], ...]]:
    cached = _ORBIT_CACHE.get(side)
    if cached is not None:
        return cached
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(side):
        for c in range(side):
            orbit = _rot90_orbit(side, r, c)
            key = tuple(sorted(set(orbit)))
            if len(key) != 4 or key in seen:
                continue
            seen.add(key)
            orbits.append(orbit)
    _ORBIT_CACHE[side] = orbits
    return orbits


def _cycle_ok(succ: dict[int, int], pred: dict[int, int]) -> bool:
    colors = set(succ) | set(pred)
    if len(colors) < 4 or set(succ) != colors or set(pred) != colors:
        return False
    start = next(iter(colors))
    seen: set[int] = set()
    cur = start
    while cur not in seen:
        seen.add(cur)
        cur = succ[cur]
    return cur == start and seen == colors


def _candidate_info(
    grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    side: int,
) -> dict | None:
    patch = grid[r0 : r0 + side, c0 : c0 + side]
    inside = int(np.count_nonzero(patch != bg))
    if inside < 11:
        return None

    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    partial_data: list[tuple[tuple[tuple[int, int], ...], list[int]]] = []
    full_count = 0

    for orbit in _orbit_list(side):
        vals = [int(patch[r, c]) for r, c in orbit]
        present = [i for i, v in enumerate(vals) if v != bg]
        if not present:
            continue
        if len(present) in (1, 2):
            return None
        if len(present) == 4:
            if len(set(vals)) != 4:
                return None
            for i in range(4):
                a = vals[i]
                b = vals[(i + 1) % 4]
                if a in succ and succ[a] != b:
                    return None
                if b in pred and pred[b] != a:
                    return None
                succ[a] = b
                pred[b] = a
            full_count += 1
        else:
            if len({vals[i] for i in present}) != 3:
                return None
            partial_data.append((orbit, vals))

    if full_count < 2 or not partial_data or not _cycle_ok(succ, pred):
        return None

    holes: list[tuple[int, int, int]] = []
    for orbit, vals in partial_data:
        missing = next(i for i, v in enumerate(vals) if v == bg)
        for i in range(4):
            a = vals[i]
            b = vals[(i + 1) % 4]
            if a != bg and b != bg and succ.get(a) != b:
                return None
        options: list[int] = []
        prev_color = vals[(missing - 1) % 4]
        next_color = vals[(missing + 1) % 4]
        if prev_color == bg or next_color == bg:
            return None
        options.append(succ.get(prev_color, -1))
        options.append(pred.get(next_color, -1))
        if options[0] == -1 or options[1] == -1 or options[0] != options[1]:
            return None
        need = options[0]
        if need in {vals[i] for i in range(4) if i != missing}:
            return None
        rr, cc = orbit[missing]
        holes.append((r0 + rr, c0 + cc, need))

    return {
        "r0": r0,
        "c0": c0,
        "side": side,
        "inside": inside,
        "full_count": full_count,
        "partial_count": len(holes),
        "holes": holes,
    }


def _find_pattern_square(grid: np.ndarray, bg: int) -> dict:
    h, w = grid.shape
    fg = (grid != bg).astype(int)
    pref = np.pad(fg, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)

    def rect_sum(r0: int, c0: int, side: int) -> int:
        r1 = r0 + side
        c1 = c0 + side
        return int(pref[r1, c1] - pref[r0, c1] - pref[r1, c0] + pref[r0, c0])

    best: dict | None = None
    best_key = None
    for side in range(4, min(h, w) + 1, 2):
        for r0 in range(h - side + 1):
            for c0 in range(w - side + 1):
                if rect_sum(r0, c0, side) < 11:
                    continue
                info = _candidate_info(grid, bg, r0, c0, side)
                if info is None:
                    continue
                key = (
                    info["inside"],
                    info["full_count"],
                    info["partial_count"],
                    info["side"],
                    -info["r0"],
                    -info["c0"],
                )
                if best_key is None or key > best_key:
                    best = info
                    best_key = key
    if best is None:
        raise AssertionError("expected one rotation-color square")
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the coupled rotation-color symmetry by moving matching stray pixels into the missing orbit slots.

    Input structure:
    - A dense multicolor pattern lies inside one hidden even-sided square region on a dominant background.
    - Each used orbit in that square has four rotational positions; a full orbit advances one fixed global color wheel by one step under each 90-degree turn.
    - Some square orbits are missing exactly one member, while outside the square there are isolated stray pixels and bystander pixels.

    Rule:
    - Infer the square and the global successor relation on colors from complete four-cell orbits.
    - For every three-of-four orbit, deduce the missing position and its required color from the same orbit plus the inferred color wheel.
    - Move exactly the outside stray pixels whose colors match those required colors into the holes, erase their old locations to background, and leave all other cells unchanged.

    Key insight:
    - The symmetry is not pure geometry: a 90-degree rotation must be paired with one step around a global color cycle, so both the missing locations and the missing colors come from the coupled rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = _get_bg(g)
    info = _find_pattern_square(g, bg)
    r0, c0, side = info["r0"], info["c0"], info["side"]

    needed = Counter(color for _r, _c, color in info["holes"])
    outside_by_color: dict[int, list[tuple[int, int]]] = {color: [] for color in needed}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r0 <= r < r0 + side and c0 <= c < c0 + side:
                continue
            color = int(g[r, c])
            if color in outside_by_color:
                outside_by_color[color].append((r, c))

    out = g.copy()
    for color, count in needed.items():
        coords = outside_by_color.get(color, [])
        assert len(coords) >= count, "missing-color strays must exist outside the square"
        for r, c in sorted(coords)[:count]:
            out[r, c] = bg

    for r, c, color in info["holes"]:
        out[r, c] = color
    return out


def _corner_orbit(side: int) -> tuple[tuple[int, int], ...]:
    for orbit in _orbit_list(side):
        if (0, 0) in orbit:
            return orbit
    raise AssertionError("expected a corner orbit")


def _orbit_distance(side: int, orbit: tuple[tuple[int, int], ...]) -> float:
    center = (side - 1) / 2.0
    return max(abs(r - center) + abs(c - center) for r, c in orbit)


def _pick_used_orbits(side: int) -> tuple[list[tuple[tuple[int, int], ...]], set[int]]:
    orbits = _orbit_list(side)
    corner = _corner_orbit(side)
    others = [orbit for orbit in orbits if orbit != corner]
    others.sort(key=lambda orbit: (_orbit_distance(side, orbit), random.random()))

    max_used = min(len(orbits), 8)
    min_used = min(5, max_used)
    used_count = random.randint(min_used, max_used)
    used = [corner] + random.sample(others, used_count - 1)

    partial_max = min(4, used_count - 2)
    partial_count = random.randint(1, partial_max)
    partial_candidates = list(range(1, used_count))
    partial_indices = set(random.sample(partial_candidates, partial_count))
    return used, partial_indices


def _isolated_positions(
    h: int,
    w: int,
    square_box: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = square_box
    cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1)
    ]
    random.shuffle(cells)

    picked: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked
    return None


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    side_choices = [side for side in GRID_SIDE_OPTIONS if side <= min(h, w) - 4]
    if not side_choices:
        return None
    side = random.choice(side_choices)

    bg = random.choice(ALL_COLORS)
    cycle = random.sample([c for c in ALL_COLORS if c != bg], 4)

    r0 = random.randint(2, h - side - 2)
    c0 = random.randint(2, w - side - 2)
    square_box = (r0, c0, r0 + side - 1, c0 + side - 1)

    used_orbits, partial_indices = _pick_used_orbits(side)
    out = np.full((h, w), bg, dtype=int)
    inp = np.full((h, w), bg, dtype=int)
    missing_colors: list[int] = []

    for orbit_index, orbit in enumerate(used_orbits):
        start = random.randrange(4)
        colors = [cycle[(start + i) % 4] for i in range(4)]
        missing_slot = random.randrange(4) if orbit_index in partial_indices else None
        for i, (rr, cc) in enumerate(orbit):
            color = colors[i]
            out[r0 + rr, c0 + cc] = color
            if i != missing_slot:
                inp[r0 + rr, c0 + cc] = color
            else:
                missing_colors.append(color)

    bystander_count = random.randint(1, 3)
    total_outside = len(missing_colors) + bystander_count
    stray_positions = _isolated_positions(h, w, square_box, total_outside)
    if stray_positions is None:
        return None

    for pos, color in zip(stray_positions[: len(missing_colors)], missing_colors):
        inp[pos] = color

    forbidden = set(missing_colors)
    bystander_palette = [c for c in ALL_COLORS if c != bg and c not in forbidden]
    if not bystander_palette:
        return None
    for pos in stray_positions[len(missing_colors) :]:
        color = random.choice(bystander_palette)
        inp[pos] = color
        out[pos] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    built = _build_example()
    if built is None:
        return None
    inp, expected = built
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
