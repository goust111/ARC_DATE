from __future__ import annotations

from collections import Counter
import random

import numpy as np


Segment = tuple[int, int, int]  # (k=r+c, minimum b=r-c, maximum b=r-c)
PairSpec = tuple[int, int, int, int]  # (lower k, upper k, minimum b, maximum b)


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected components using the common ARC (row, column) convention."""
    h, w = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    components: list[list[tuple[int, int]]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = (r + dr, c + dc)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
        components.append(component)
    return components


def _segments(grid: np.ndarray, color: int) -> list[Segment]:
    """Recognize one-cell-wide, contiguous anti-diagonal segments of a color."""
    segments: list[Segment] = []
    for component in _components8(grid == int(color)):
        ks = {r + c for r, c in component}
        assert len(ks) == 1, "each foreground component must be one anti-diagonal segment"
        k = int(next(iter(ks)))
        bs = sorted(r - c for r, c in component)
        assert len(bs) >= 2, "segments must contain at least two cells"
        assert bs == list(range(bs[0], bs[-1] + 1, 2)), "segments must be contiguous"
        segments.append((k, int(bs[0]), int(bs[-1])))
    assert len(segments) >= 2 and len(segments) % 2 == 0, "segments must occur in pairs"
    return segments


def _region_between(shape: tuple[int, int], segments: list[Segment]) -> np.ndarray:
    """Pair consecutive visible boundaries on each b-axis and mark their interiors."""
    h, w = map(int, shape)
    region = np.zeros((h, w), dtype=bool)
    for b in range(-(w - 1), h):
        active = sorted((seg for seg in segments if seg[1] <= b <= seg[2]), key=lambda s: s[0])
        assert len(active) % 2 == 0, "every crossed strip must have two visible boundaries"
        for index in range(0, len(active), 2):
            k0 = active[index][0]
            k1 = active[index + 1][0]
            assert k0 < k1, "paired segments must be distinct"
            r0 = max(0, b)
            r1 = min(h - 1, w - 1 + b)
            for r in range(r0, r1 + 1):
                c = r - b
                if k0 < r + c < k1:
                    region[r, c] = True
    return region


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each color's paired diagonal strips, leaving their overlap as background.

    Input structure:
    - The most frequent color is the background.
    - Each of exactly two foreground colors forms paired, parallel short segments;
      every segment lies on one anti-diagonal (r+c is constant).

    Rule:
    - For each color, pair its visible segment boundaries and fill the background
      cells between each pair along the main-diagonal direction.
    - Paint a background cell only when it belongs to exactly one color's fill
      region. Cells belonging to both regions remain background, and every
      original segment cell remains unchanged.

    Key insight:
    - The two parallelogram fills are computed independently, but their masks are
      combined by exclusive-or rather than by overlaying one color on the other.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    bg = _background(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 2, "expected exactly two segment colors"

    regions = [_region_between(g.shape, _segments(g, color)) for color in colors]
    out = g.copy()
    original_background = g == bg
    only_first = original_background & regions[0] & ~regions[1]
    only_second = original_background & regions[1] & ~regions[0]
    out[only_first] = colors[0]
    out[only_second] = colors[1]
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _spec_from_anchor(
    shape: tuple[int, int], r0: int, c0: int, length: int, distance: int
) -> PairSpec | None:
    """Make two anti-diagonal segments, the second translated by (distance, distance)."""
    h, w = shape
    if length < 2 or distance < 2:
        return None
    if r0 < 1 or r0 + distance + length - 1 > h - 2:
        return None
    if c0 - length + 1 < 1 or c0 + distance > w - 2:
        return None
    k0 = r0 + c0
    b0 = r0 - c0
    return int(k0), int(k0 + 2 * distance), int(b0), int(b0 + 2 * (length - 1))


def _random_spec(shape: tuple[int, int], anchor: PairSpec | None = None) -> PairSpec | None:
    h, w = shape
    length = random.randint(3, min(7, h - 7, w - 4))
    distance = random.randint(3, min(7, h - length - 2, w - length - 1))
    if anchor is None:
        r_hi = h - distance - length - 1
        c_lo = length
        c_hi = w - distance - 2
        if r_hi < 1 or c_hi < c_lo:
            return None
        r0 = random.randint(1, r_hi)
        c0 = random.randint(c_lo, c_hi)
    else:
        ar = (anchor[0] + anchor[2]) // 2
        ac = (anchor[0] - anchor[2]) // 2
        r0 = ar + random.randint(-4, 4)
        c0 = ac + random.randint(-4, 4)
    return _spec_from_anchor(shape, r0, c0, length, distance)


def _spec_region(shape: tuple[int, int], spec: PairSpec) -> np.ndarray:
    rr, cc = np.indices(shape)
    kk = rr + cc
    bb = rr - cc
    k0, k1, b0, b1 = spec
    return (k0 < kk) & (kk < k1) & (b0 <= bb) & (bb <= b1)


def _spec_lines(shape: tuple[int, int], spec: PairSpec) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    k0, k1, b0, b1 = spec
    for k in (k0, k1):
        for b in range(b0, b1 + 1, 2):
            r = (k + b) // 2
            c = (k - b) // 2
            assert 0 <= r < shape[0] and 0 <= c < shape[1]
            mask[r, c] = True
    return mask


def _union(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for mask in masks:
        result |= mask
    return result


def _touches8(first: np.ndarray, second: np.ndarray) -> bool:
    rows, cols = np.where(first)
    h, w = first.shape
    for r, c in zip(map(int, rows), map(int, cols)):
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        if bool(second[r0:r1, c0:c1].any()):
            return True
    return False


def _render_input(
    shape: tuple[int, int], bg: int, colors: tuple[int, int], specs: list[list[PairSpec]]
) -> np.ndarray | None:
    grid = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for color, family in zip(colors, specs):
        family_lines: list[np.ndarray] = []
        for spec in family:
            lines = _spec_lines(shape, spec)
            if bool((occupied & lines).any()) or any(_touches8(lines, old) for old in family_lines):
                return None
            grid[lines] = color
            occupied |= lines
            family_lines.append(lines)
    return grid


def _family_regions(shape: tuple[int, int], specs: list[list[PairSpec]]) -> list[np.ndarray]:
    return [_union([_spec_region(shape, spec) for spec in family], shape) for family in specs]


def _render_expected(
    inp: np.ndarray,
    bg: int,
    colors: tuple[int, int],
    regions: list[np.ndarray],
) -> np.ndarray:
    expected = inp.copy()
    background = inp == bg
    expected[background & regions[0] & ~regions[1]] = colors[0]
    expected[background & regions[1] & ~regions[0]] = colors[1]
    return expected


def _choose_primary_specs(shape: tuple[int, int], mode: str) -> list[list[PairSpec]] | None:
    first = _random_spec(shape)
    if first is None:
        return None
    first_region = _spec_region(shape, first)
    first_lines = _spec_lines(shape, first)

    for _ in range(700):
        second = _random_spec(shape, anchor=first if mode == "overlap" else None)
        if second is None:
            continue
        second_region = _spec_region(shape, second)
        second_lines = _spec_lines(shape, second)
        if bool((first_lines & second_lines).any()):
            continue

        line_cells = first_lines | second_lines
        common = first_region & second_region & ~line_cells
        first_only = first_region & ~second_region & ~line_cells
        second_only = second_region & ~first_region & ~line_cells
        common_count = int(common.sum())
        if int(first_only.sum()) < 8 or int(second_only.sum()) < 8:
            continue
        if mode == "overlap":
            upper = min(18, max(3, int(0.35 * min(first_region.sum(), second_region.sum()))))
            if not (3 <= common_count <= upper):
                continue
        elif common_count != 0:
            continue
        return [[first], [second]]
    return None


def _add_extra_pair(shape: tuple[int, int], specs: list[list[PairSpec]]) -> bool:
    """Add a second, visibly separate strip to one randomly chosen color family."""
    family_index = random.randrange(2)
    all_regions = _family_regions(shape, specs)
    all_lines = _union(
        [_spec_lines(shape, spec) for family in specs for spec in family], shape
    )
    same_lines = _union([_spec_lines(shape, spec) for spec in specs[family_index]], shape)
    for _ in range(800):
        candidate = _random_spec(shape)
        if candidate is None:
            continue
        region = _spec_region(shape, candidate)
        lines = _spec_lines(shape, candidate)
        if bool((region & (all_regions[0] | all_regions[1])).any()):
            continue
        if bool((lines & all_lines).any()) or _touches8(lines, same_lines):
            continue
        if int((region & ~all_lines & ~lines).sum()) < 8:
            continue
        specs[family_index].append(candidate)
        return True
    return False


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    colors: tuple[int, int],
    specs: list[list[PairSpec]],
    mode: str,
) -> bool:
    """Check the stated geometry and the dependencies, not just solver closure."""
    if _background(inp) != bg or set(map(int, np.unique(inp))) != {bg, *colors}:
        return False
    regions = _family_regions(inp.shape, specs)
    original_background = inp == bg
    overlap = original_background & regions[0] & regions[1]
    exclusive = [
        original_background & regions[0] & ~regions[1],
        original_background & regions[1] & ~regions[0],
    ]
    if mode == "overlap":
        if int(overlap.sum()) < 3 or not np.all(expected[overlap] == bg):
            return False
    elif bool(overlap.any()):
        return False
    for index, color in enumerate(colors):
        if int(exclusive[index].sum()) < 8 or not np.all(expected[exclusive[index]] == color):
            return False
        line_mask = inp == color
        if not np.array_equal(expected[line_mask], inp[line_mask]):
            return False
        if len(_components8(line_mask)) != 2 * len(specs[index]):
            return False

    # Every pair must affect the answer: remove it from the geometric oracle and
    # require a different output. This rules out decorative or redundant pairs.
    for family_index, family in enumerate(specs):
        for omitted in range(len(family)):
            reduced_specs = [list(specs[0]), list(specs[1])]
            reduced_specs[family_index].pop(omitted)
            reduced_regions = _family_regions(inp.shape, reduced_specs)
            if np.array_equal(_render_expected(inp, bg, colors, reduced_regions), expected):
                return False
    try:
        return bool(np.array_equal(main(inp), expected))
    except (AssertionError, IndexError):
        return False


def _gen_attempt(mode: str, want_extra: bool) -> np.ndarray | None:
    minimum = 23 if want_extra else 18
    shape = (random.randint(minimum, 30), random.randint(minimum, 30))
    bg, first_color, second_color = random.sample(range(10), 3)
    colors = (first_color, second_color)
    specs = _choose_primary_specs(shape, mode)
    if specs is None:
        return None
    if want_extra and not _add_extra_pair(shape, specs):
        return None
    inp = _render_input(shape, bg, colors, specs)
    if inp is None:
        return None
    regions = _family_regions(shape, specs)
    expected = _render_expected(inp, bg, colors, regions)
    if np.array_equal(inp, expected):
        return None
    return inp if _validate_semantics(inp, expected, bg, colors, specs, mode) else None


# Any three consecutive generated training examples include both the overlap
# counterexample and the deliberately non-overlapping lure.
_BRANCH_CYCLE = ["overlap", "overlap", "disjoint"]
random.shuffle(_BRANCH_CYCLE)
_generation_count = 0


def generate_input() -> np.ndarray:
    global _generation_count
    mode = _BRANCH_CYCLE[_generation_count % len(_BRANCH_CYCLE)]
    _generation_count += 1
    want_extra = random.random() < 0.35
    for _ in range(300):
        result = _gen_attempt(mode, want_extra)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid two-color diagonal-fill puzzle")
