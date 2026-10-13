from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int = 8):
    if int(connectivity) == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            comp: set[Coord] = set()
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.add((int(rr), int(cc)))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _reflect(axis: int, cell: Coord) -> Coord:
    r, c = cell
    return int(r), int(2 * int(axis) - int(c))


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _best_axis(grid: np.ndarray, bg: int, pattern: set[Coord]) -> tuple[int, set[Coord]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, set[Coord]]] = []
    for axis in range(1, w - 1):
        if any(c == axis for _r, c in pattern):
            continue
        holes: set[Coord] = set()
        paired = 0
        ok = True
        for cell in pattern:
            mate = _reflect(axis, cell)
            if not _in_bounds((h, w), mate):
                ok = False
                break
            if mate in pattern:
                paired += 1
            elif int(grid[mate]) == int(bg):
                holes.add(mate)
            else:
                ok = False
                break
        if not ok or not holes:
            continue

        completed = pattern | holes
        if any(_reflect(axis, cell) not in completed for cell in completed):
            continue
        if paired <= 2 * len(holes):
            continue
        candidates.append((int(paired), -int(len(holes)), int(axis), holes))

    assert candidates, "expected a unique explainable vertical mirror axis"
    best_score = max((paired, neg_holes) for paired, neg_holes, _axis, _holes in candidates)
    best = [item for item in candidates if (item[0], item[1]) == best_score]
    assert len(best) == 1, "mirror axis must be uniquely determined by the color layer"
    _paired, _neg_holes, axis, holes = best[0]
    return int(axis), set(holes)


def _analyze_scene(grid: np.ndarray) -> tuple[int, dict[int, dict]]:
    bg = _get_bg(grid)
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != int(bg))
    assert len(colors) >= 2, "expected multiple foreground colors"

    info: dict[int, dict] = {}
    for color in colors:
        comps = _components(grid == int(color), connectivity=8)
        assert comps, "each foreground color must appear"
        pattern_parts = [comp for comp in comps if len(comp) > 1]
        stray_parts = [comp for comp in comps if len(comp) == 1]
        assert pattern_parts, "each color needs non-singleton pattern pieces"
        assert stray_parts, "each color needs isolated stray pixels"

        pattern = set().union(*pattern_parts)
        strays = {next(iter(comp)) for comp in stray_parts}
        axis, holes = _best_axis(grid, bg, pattern)

        all_same_color = pattern | strays
        assert all(c != axis for _r, c in all_same_color), "a color must not draw its own axis"
        assert all(int(grid[r, c]) == int(bg) for r, c in holes), "missing cells must be background"
        info[int(color)] = {"axis": int(axis), "pattern": pattern, "strays": strays, "holes": holes}

    axes = [item["axis"] for item in info.values()]
    assert len(axes) == len(set(axes)), "color axes must be distinct"
    return int(bg), info


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, info = _analyze_scene(g)

    hole_owner: dict[Coord, int] = {}
    for color, item in info.items():
        for hole in item["holes"]:
            assert hole not in hole_owner, "holes of different colors must not overlap"
            hole_owner[hole] = int(color)

    out = g.copy()
    used_holes: dict[Coord, int] = {}
    for color, item in info.items():
        axis = int(item["axis"])
        for stray in item["strays"]:
            dest = _reflect(axis, stray)
            assert _in_bounds(g.shape, dest), "a reflected stray must stay inside the grid"
            assert dest in hole_owner, "every stray must land on a visible missing cell"
            target_color = int(hole_owner[dest])
            assert target_color != int(color), "strays must patch a different color layer"
            assert dest not in used_holes, "each missing cell receives exactly one stray"
            used_holes[dest] = int(color)
            out[stray] = int(bg)
            out[dest] = target_color

    assert set(used_holes) == set(hole_owner), "every missing cell must be supplied by one stray"
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each color's isolated stray pixels into another color's missing symmetric cells, recoloring the landing cells.

    Input structure:
    - The background is the dominant color.
    - Each foreground color has a hidden vertical integer-column axis; that color never draws on its own axis.
    - The non-singleton pieces of one color form a nearly left-right symmetric layer with one or more background holes.
    - The singleton pieces of that same color are stray pixels, not part of its layer.

    Rule:
    - For every color, find the unique vertical axis that best completes its non-singleton layer by mirror reflection.
    - The background mirror mates of the layer pixels are that color's holes.
    - Reflect each singleton stray across its own color's axis; it must land on exactly one other color's hole.
    - Clear the stray's original cell and fill the landing hole with the target layer's color.

    Key insight:
    - The odd pixels are not removed locally; each color's noise is a mirrored, recolored patch for a different color layer.
    """
    out, _info = _solve_details(grid)
    return out


# ---------------- generator ----------------

_BG = 0
_PALETTE = tuple(range(1, 10))
_GEN_RETRIES = 600


def _cheb_near(cell: Coord, cells: set[Coord], radius: int = 1) -> bool:
    r, c = cell
    for rr, cc in cells:
        if max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) <= int(radius):
            return True
    return False


def _coord_components(coords: set[Coord], connectivity: int = 8) -> list[set[Coord]]:
    if not coords:
        return []
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    r1 = max(r for r, _c in coords)
    c1 = max(c for _r, c in coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    comps = []
    for comp in _components(mask, connectivity=connectivity):
        comps.append({(r + r0, c + c0) for r, c in comp})
    return comps


def _choose_bands(n: int) -> tuple[int, list[tuple[int, int]]]:
    for _ in range(100):
        heights = [random.randint(5, 7) for _ in range(n)]
        min_h = sum(heights) + n + 1
        if min_h <= 30:
            break
    else:
        heights = [5] * n
        min_h = sum(heights) + n + 1
    h = random.randint(max(20, min_h), 30)
    gaps = [1] * (n + 1)
    extra = h - sum(heights) - sum(gaps)
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    bands: list[tuple[int, int]] = []
    r = gaps[0]
    for i, height in enumerate(heights):
        bands.append((int(r), int(height)))
        r += height + gaps[i + 1]
    return int(h), bands


def _choose_axes(w: int, n: int) -> list[int] | None:
    for _ in range(200):
        center = random.randint(7, w - 8)
        lo = max(4, center - 5)
        hi = min(w - 5, center + 5)
        pool = list(range(lo, hi + 1))
        random.shuffle(pool)
        axes: list[int] = []
        for axis in pool:
            if all(abs(axis - other) >= 2 for other in axes):
                axes.append(int(axis))
                if len(axes) == n:
                    random.shuffle(axes)
                    return axes
    return None


def _make_full_pattern(axis: int, top: int, height: int) -> tuple[set[Coord], list[Coord]]:
    max_offset = random.choice([2, 2, 3])
    arm_lengths = []
    for _ in range(height):
        arm_lengths.append(random.choices([1, 2, 3], weights=[2, 4, 2])[0])
    arm_lengths = [min(length, max_offset) for length in arm_lengths]
    if sum(1 for length in arm_lengths if length >= 2) < 3:
        rows = list(range(height))
        random.shuffle(rows)
        for row in rows[:3]:
            arm_lengths[row] = 2

    coords: set[Coord] = set()
    removable: list[Coord] = []
    for rr, arm in enumerate(arm_lengths):
        r = top + rr
        for d in range(1, arm + 1):
            coords.add((r, axis - d))
            coords.add((r, axis + d))
        if arm >= 2:
            removable.append((r, axis - arm))
            removable.append((r, axis + arm))
    random.shuffle(removable)
    return coords, removable


def _pattern_after_holes_ok(full: set[Coord], holes: set[Coord]) -> bool:
    pattern = full - holes
    comps = _coord_components(pattern, connectivity=8)
    return bool(comps) and all(len(comp) > 1 for comp in comps)


def _choose_holes(
    shape: tuple[int, int],
    target_axis: int,
    source_axis: int,
    full: set[Coord],
    removable: list[Coord],
    all_full: set[Coord],
    reserved_strays: set[Coord],
    same_source_strays: set[Coord],
    count: int,
) -> tuple[set[Coord], set[Coord]] | None:
    candidates = []
    for hole in removable:
        stray = _reflect(source_axis, hole)
        if not _in_bounds(shape, stray):
            continue
        if stray == hole or stray[1] == source_axis:
            continue
        if stray in all_full or stray in reserved_strays:
            continue
        if _cheb_near(stray, all_full, radius=1):
            continue
        if _cheb_near(stray, reserved_strays, radius=1):
            continue
        if _cheb_near(stray, same_source_strays, radius=1):
            continue
        pair = frozenset({hole, _reflect(target_axis, hole)})
        candidates.append((hole, stray, pair))

    random.shuffle(candidates)
    combos = list(itertools.combinations(candidates, int(count)))
    random.shuffle(combos)
    for combo in combos[:400]:
        pairs = [item[2] for item in combo]
        holes = {item[0] for item in combo}
        strays = {item[1] for item in combo}
        if len(set(pairs)) != len(pairs) or len(holes) != count or len(strays) != count:
            continue
        if not _pattern_after_holes_ok(full, holes):
            continue
        return holes, strays
    return None


def _render_input_and_expected(
    shape: tuple[int, int],
    colors: list[int],
    full_by_color: dict[int, set[Coord]],
    holes_by_color: dict[int, set[Coord]],
    strays_by_color: dict[int, set[Coord]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full(shape, _BG, dtype=int)
    expected = np.full(shape, _BG, dtype=int)

    for color in colors:
        for cell in full_by_color[color]:
            expected[cell] = int(color)
        for cell in full_by_color[color] - holes_by_color[color]:
            if int(inp[cell]) != _BG:
                return None
            inp[cell] = int(color)

    for color in colors:
        for stray in strays_by_color[color]:
            if int(inp[stray]) != _BG:
                return None
            inp[stray] = int(color)
    return inp, expected


def _semantic_oracle(
    inp: np.ndarray,
    expected: np.ndarray,
    true_axes: dict[int, int],
    holes_by_color: dict[int, set[Coord]],
    strays_by_color: dict[int, set[Coord]],
) -> bool:
    try:
        actual, info = _solve_details(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    for color, axis in true_axes.items():
        if int(info[color]["axis"]) != int(axis):
            return False
        if set(info[color]["holes"]) != set(holes_by_color[color]):
            return False
        if set(info[color]["strays"]) != set(strays_by_color[color]):
            return False
    total_holes = sum(len(v) for v in holes_by_color.values())
    if total_holes < len(holes_by_color):
        return False
    if int(np.count_nonzero(inp != expected)) != 2 * total_holes:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 3, 4])
    h, bands = _choose_bands(n)
    w = random.randint(20, 30)
    axes = _choose_axes(w, n)
    if axes is None:
        return None

    colors = random.sample(_PALETTE, n)
    order = list(range(n))
    random.shuffle(order)
    colors = [colors[i] for i in order]
    bands = [bands[i] for i in order]

    full_by_color: dict[int, set[Coord]] = {}
    removable_by_color: dict[int, list[Coord]] = {}
    true_axes: dict[int, int] = {}
    for color, axis, (top, height) in zip(colors, axes, bands):
        full, removable = _make_full_pattern(axis, top, height)
        full_by_color[int(color)] = full
        removable_by_color[int(color)] = removable
        true_axes[int(color)] = int(axis)

    all_full: set[Coord] = set().union(*full_by_color.values())
    if sum(len(v) for v in full_by_color.values()) != len(all_full):
        return None

    hole_counts = {color: random.choice([1, 1, 2]) for color in colors}
    if all(count == 1 for count in hole_counts.values()) and random.random() < 0.65:
        hole_counts[random.choice(colors)] = 2

    holes_by_color = {color: set() for color in colors}
    strays_by_color = {color: set() for color in colors}
    reserved_strays: set[Coord] = set()

    for i, target_color in enumerate(colors):
        source_color = colors[(i - 1) % n]
        chosen = _choose_holes(
            (h, w),
            true_axes[target_color],
            true_axes[source_color],
            full_by_color[target_color],
            removable_by_color[target_color],
            all_full,
            reserved_strays,
            strays_by_color[source_color],
            hole_counts[target_color],
        )
        if chosen is None:
            return None
        holes, strays = chosen
        holes_by_color[target_color] = holes
        strays_by_color[source_color].update(strays)
        reserved_strays.update(strays)

    if any(not holes_by_color[color] or not strays_by_color[color] for color in colors):
        return None

    rendered = _render_input_and_expected((h, w), colors, full_by_color, holes_by_color, strays_by_color)
    if rendered is None:
        return None
    inp, expected = rendered
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_oracle(inp, expected, true_axes, holes_by_color, strays_by_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid mirrored-fragment input")
