from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
MASK = 5
SYMBOL_COLOR = 5
FG_COLORS = (BLUE, RED, GREEN)
MASK_COLORS = (5, 6, 7, 8, 9)
GRID_SIZES = (7, 9, 11)
MAX_ATTEMPTS = 400


_SYMBOL_BY_COLOR = {
    BLUE: np.array(
        [
            [0, 5, 0],
            [5, 5, 5],
            [0, 5, 0],
        ],
        dtype=int,
    ),
    RED: np.array(
        [
            [5, 5, 5],
            [0, 5, 0],
            [0, 5, 0],
        ],
        dtype=int,
    ),
    GREEN: np.array(
        [
            [0, 0, 5],
            [0, 0, 5],
            [5, 5, 5],
        ],
        dtype=int,
    ),
}

_BAGS: dict[str, list[int]] = {}
_ORBIT_CACHE: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {}


def _symmetry_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = {
        (r, c),
        (r, n - 1 - c),
        (n - 1 - r, c),
        (n - 1 - r, n - 1 - c),
        (c, r),
        (c, n - 1 - r),
        (n - 1 - c, r),
        (n - 1 - c, n - 1 - r),
    }
    return tuple(sorted(cells))


def _all_orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    cached = _ORBIT_CACHE.get(n)
    if cached is not None:
        return cached

    seen = set()
    orbits = []
    for r in range(n):
        for c in range(n):
            orbit = _symmetry_orbit(n, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            orbits.append(orbit)

    out = tuple(orbits)
    _ORBIT_CACHE[n] = out
    return out


def _mask_candidates(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    n = g.shape[0]
    candidates = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        ok = True
        touches_mask = False
        for orbit in _all_orbits(n):
            values = [int(g[r, c]) for r, c in orbit]
            if color in values:
                touches_mask = True
            visible = {v for v in values if v != color}
            if len(visible) != 1:
                ok = False
                break
        if ok and touches_mask:
            candidates.append(color)
    return candidates


def _detect_mask_color(grid: np.ndarray) -> int:
    candidates = _mask_candidates(grid)
    assert len(candidates) == 1, "expected a unique mask color"
    return int(candidates[0])


def _restore_mask(grid: np.ndarray, mask_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for orbit in _all_orbits(g.shape[0]):
        values = [int(g[r, c]) for r, c in orbit]
        visible = {v for v in values if v != mask_color}
        assert len(visible) == 1, "each symmetry orbit must expose one visible color"
        fill = int(next(iter(visible)))
        if mask_color in values:
            assert fill != BG, "mask pixels should only hide foreground colors"
        for r, c in orbit:
            if int(g[r, c]) == mask_color:
                out[r, c] = fill
            else:
                assert int(g[r, c]) == fill, "visible orbit members must already agree"
    return out


def _count_foreground(grid: np.ndarray, *, exclude: set[int] | None = None) -> Counter[int]:
    blocked = {BG}
    if exclude is not None:
        blocked |= set(int(v) for v in exclude)
    return Counter(
        int(v)
        for v in np.asarray(grid, dtype=int).flat
        if int(v) in FG_COLORS and int(v) not in blocked
    )


def _unique_majority(counts: Counter[int]) -> int | None:
    if not counts:
        return None
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _pick_from_bag(name: str, values: tuple[int, ...], rng: np.random.Generator) -> int:
    bag = _BAGS.setdefault(name, [])
    if not bag:
        bag.extend(int(v) for v in values)
        rng.shuffle(bag)
    return int(bag.pop())


def _symbol_for(color: int, symbol_color: int) -> np.ndarray:
    template = _SYMBOL_BY_COLOR[int(color)]
    return np.where(template == SYMBOL_COLOR, int(symbol_color), BG).astype(int)


def _build_full_grid(
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]], dict[int, int], int] | None:
    n = _pick_from_bag("size", GRID_SIZES, rng)
    n_colors = _pick_from_bag("num_colors", (2, 3), rng)
    post_major = _pick_from_bag("post_major", FG_COLORS, rng)
    pre_major_choices = tuple(color for color in FG_COLORS if color != post_major)
    pre_major = int(rng.choice(pre_major_choices))

    active_colors = [post_major, pre_major]
    if n_colors == 3:
        active_colors.append(next(color for color in FG_COLORS if color not in active_colors))
    rng.shuffle(active_colors)

    all_orbits = list(_all_orbits(n))
    rng.shuffle(all_orbits)
    usable_orbits = [orbit for orbit in all_orbits if len(orbit) > 1]
    if len(usable_orbits) < n_colors + 3:
        return None

    min_fg_orbits = max(n_colors + 3, 5)
    max_fg_orbits = min(len(all_orbits) - 1, n_colors + 7)
    if min_fg_orbits > max_fg_orbits:
        return None
    fg_orbit_count = int(rng.integers(min_fg_orbits, max_fg_orbits + 1))
    selected = usable_orbits[:fg_orbit_count]
    selected.sort(key=len, reverse=True)
    if len({len(orbit) for orbit in selected}) < 2:
        return None

    weights = {color: 0.18 for color in active_colors}
    weights[post_major] = 0.46 if n_colors == 2 else 0.42
    weights[pre_major] = 0.36 if n_colors == 2 else 0.34
    remaining_color = [color for color in active_colors if color not in (post_major, pre_major)]
    if remaining_color:
        weights[remaining_color[0]] = 0.24

    orbit_color: dict[tuple[tuple[int, int], ...], int] = {}
    orbit_infos: list[tuple[tuple[int, int], ...]] = []

    forced_colors = [post_major, pre_major]
    if remaining_color:
        forced_colors.append(remaining_color[0])
    for orbit, color in zip(selected[: len(forced_colors)], forced_colors):
        orbit_infos.append(orbit)
        orbit_color[orbit] = int(color)

    probabilities = np.array([weights[c] for c in active_colors], dtype=float)
    probabilities /= probabilities.sum()
    for orbit in selected[len(forced_colors) :]:
        color = int(rng.choice(active_colors, p=probabilities))
        orbit_infos.append(orbit)
        orbit_color[orbit] = color

    full = np.zeros((n, n), dtype=int)
    full_counts = Counter()
    per_color_orbits = Counter()
    for orbit in orbit_infos:
        color = orbit_color[orbit]
        per_color_orbits[color] += 1
        full_counts[color] += len(orbit)
        for r, c in orbit:
            full[r, c] = color

    if set(full_counts) != set(active_colors):
        return None
    if _unique_majority(full_counts) != post_major:
        return None
    if full_counts[pre_major] < max(count for color, count in full_counts.items() if color != post_major):
        return None
    if full_counts[pre_major] <= per_color_orbits[post_major]:
        return None

    filled_ratio = sum(full_counts.values()) / float(n * n)
    if not 0.22 <= filled_ratio <= 0.68:
        return None

    return full, orbit_infos, orbit_color, post_major


def _mask_input(
    rng: np.random.Generator,
    full: np.ndarray,
    orbit_infos: list[tuple[tuple[int, int], ...]],
    orbit_color: dict[tuple[tuple[int, int], ...], int],
    post_major: int,
    mask_color: int,
) -> np.ndarray | None:
    full_counts = _count_foreground(full)
    ordered = [color for color, _count in full_counts.most_common()]
    if len(ordered) < 2:
        return None
    preferred_visible = ordered[1]

    inp = full.copy()
    masked_cells = 0
    masked_underlying: set[int] = set()

    for orbit in orbit_infos:
        color = orbit_color[orbit]
        size = len(orbit)
        if size == 1:
            continue
        max_hide = size - 1
        if color == post_major:
            lo = max(1, size // 2)
            hi = max_hide
        elif color == preferred_visible:
            lo = 0
            hi = min(max_hide, max(1, size // 3))
        else:
            lo = 0
            hi = min(max_hide, max(1, (2 * size) // 3))
        if hi < lo:
            continue
        hide = int(rng.integers(lo, hi + 1))
        if hide == 0:
            continue
        chosen = rng.choice(size, size=hide, replace=False)
        masked_underlying.add(color)
        masked_cells += hide
        for idx in np.atleast_1d(chosen):
            r, c = orbit[int(idx)]
            inp[r, c] = mask_color

    if masked_cells == 0 or len(masked_underlying) < 2:
        return None
    if not (4 <= masked_cells <= full.size // 3):
        return None

    visible_counts = _count_foreground(inp, exclude={mask_color})
    visible_major = _unique_majority(visible_counts)
    if visible_major is None or visible_major == post_major:
        return None

    candidates = _mask_candidates(inp)
    if candidates != [mask_color]:
        return None

    restored = _restore_mask(inp, mask_color)
    if not np.array_equal(restored, full):
        return None

    solved_counts = _count_foreground(restored)
    if _unique_majority(solved_counts) != post_major:
        return None

    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the D4-symmetric masked pattern, then emit the mask-colored 3x3 symbol for the completed majority color.

    Input structure:
    - The input is a square black-background grid with 2-3 foreground colors from blue, red, and green.
    - Ignoring one mask color, every orbit under the square's eight symmetries is monochrome.
    - Some foreground cells in those orbits are replaced by the mask, but every affected orbit still shows at least one true cell.

    Rule:
    - Identify the unique mask color whose removal makes every D4 symmetry orbit consistent.
    - Replace each mask cell by the unique visible foreground color from its full symmetry orbit.
    - Count the completed foreground colors; the strict majority color chooses the output symbol:
      red -> top T, blue -> plus, green -> lower-right L.
    - Return a 3x3 black grid whose symbol pixels use the detected mask color.

    Key insight:
    - The apparent color majority in the input is unreliable because the mask hides different colors unevenly; only symmetry completion reveals the true lookup key.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, w = g.shape
    assert n == w, "expected a square grid"

    mask_color = _detect_mask_color(g)
    restored = _restore_mask(g, mask_color)
    counts = _count_foreground(restored, exclude={mask_color})
    assert 2 <= len(counts) <= 3, "expected 2-3 foreground colors after restoration"
    majority = _unique_majority(counts)
    assert majority in _SYMBOL_BY_COLOR, "expected a unique encoded majority color"
    return _symbol_for(int(majority), mask_color)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    built = _build_full_grid(rng)
    if built is None:
        return None
    full, orbit_infos, orbit_color, post_major = built
    mask_color = _pick_from_bag("mask_color", MASK_COLORS, rng)
    inp = _mask_input(rng, full, orbit_infos, orbit_color, post_major, mask_color)
    if inp is None:
        return None

    visible_major = _unique_majority(_count_foreground(inp, exclude={mask_color}))
    assert visible_major is not None and visible_major != post_major
    assert np.array_equal(main(inp), _symbol_for(post_major, mask_color))
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid masked-majority symmetry sample")
