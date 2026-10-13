from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
TEAL = 8
OBJECT_COLORS = (1, 2, 3, 4, 6, 9)
CORNERS = ("TL", "TR", "BL", "BR")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True components as (n, 2) arrays of (row, col)."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _colored_objects(g: np.ndarray) -> dict[int, list[np.ndarray]]:
    by_color: dict[int, list[np.ndarray]] = defaultdict(list)
    for color in set(map(int, g.ravel())) - {BLACK, GREY}:
        by_color[color] = _components(g == color)
    return by_color


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _move_stage(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Perform only the multiplicity-selected translations."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert set(map(int, g.ravel())) <= set(range(10))
    assert BLACK in g and GREY in g

    by_color = _colored_objects(g)
    repeated = [color for color, objects in by_color.items() if len(objects) > 1]
    assert len(repeated) == 1, "exactly one object color must occur more than once"
    mover_color = repeated[0]
    movers = by_color[mover_color]

    out = g.copy()
    shifted_objects: list[np.ndarray] = []
    for coords in movers:
        r0, c0, r1, c1 = _bbox(coords)
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        assert len(coords) == rh * rw
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == mover_color)
        assert (r0 == 0) != (r1 == h - 1)
        assert (c0 == 0) != (c1 == w - 1)
        out[coords[:, 0], coords[:, 1]] = BLACK

        dr = rh if r0 == 0 else -rh
        dc = rw if c0 == 0 else -rw
        shifted = coords + np.array((dr, dc), dtype=int)
        assert np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < h))
        assert np.all((0 <= shifted[:, 1]) & (shifted[:, 1] < w))
        shifted_objects.append(shifted)

    for shifted in shifted_objects:
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == BLACK), "movement targets must be empty"
        out[shifted[:, 0], shifted[:, 1]] = mover_color
    return out, shifted_objects


def _paint_extremes(moved: np.ndarray) -> np.ndarray:
    regions = _components(moved == BLACK)
    assert len(regions) >= 2
    areas = np.asarray([len(region) for region in regions], dtype=int)
    small = np.flatnonzero(areas == areas.min())
    large = np.flatnonzero(areas == areas.max())
    assert len(small) == 1 and len(large) == 1
    assert int(small[0]) != int(large[0])
    out = moved.copy()
    sreg, lreg = regions[int(small[0])], regions[int(large[0])]
    out[sreg[:, 0], sreg[:, 1]] = ORANGE
    out[lreg[:, 0], lreg[:, 1]] = TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move the repeated-color corner rectangles inward, then color the extreme black regions.

    Input structure:
    - A black grid contains one-cell-wide grey partition walls and solid monochrome
      rectangles, each touching one row boundary and one column boundary.
    - Exactly one rectangle color occurs in multiple separate objects; every other
      rectangle color occurs once.

    Rule:
    - Clear every rectangle of the repeated color and translate it inward by its
      own height away from the touched row edge and by its own width away from the
      touched column edge. Keep grey walls and singleton-color rectangles fixed.
    - In that moved configuration, find all 4-connected black regions, recolor the
      unique smallest one orange and the unique largest one teal, and keep all
      other cells unchanged.

    Key insight:
    - Multiplicity selects the movers, whose size and touched edges determine their
      translations; only their post-motion positions complete the partitions on
      which the unique area extremes must be measured.
    """
    moved, _ = _move_stage(grid)
    return _paint_extremes(moved)


def _corner_box(corner: str, h: int, w: int, rh: int, rw: int, moved: bool) -> tuple[int, int, int, int]:
    if moved:
        r0 = rh if corner[0] == "T" else h - 2 * rh
        c0 = rw if corner[1] == "L" else w - 2 * rw
    else:
        r0 = 0 if corner[0] == "T" else h - rh
        c0 = 0 if corner[1] == "L" else w - rw
    return r0, c0, r0 + rh - 1, c0 + rw - 1


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _region_signature(g: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(len(x) for x in _components(g == BLACK)))


def _independent_expected(inp: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Render the stated two stages from generator metadata, independently of main."""
    moved = inp.copy()
    mover_color = int(specs[0]["color"])
    for spec in specs:
        if spec["moving"]:
            r0, c0, r1, c1 = spec["source"]
            moved[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for spec in specs:
        if spec["moving"]:
            r0, c0, r1, c1 = spec["target"]
            assert np.all(moved[r0 : r1 + 1, c0 : c1 + 1] == BLACK)
            moved[r0 : r1 + 1, c0 : c1 + 1] = mover_color
    return moved, _paint_extremes(moved)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    moving_count = random.choice((2, 2, 3))
    moving_corners = set(random.sample(CORNERS, moving_count))
    singleton_corners = [corner for corner in CORNERS if corner not in moving_corners]
    mover_color = random.choice(OBJECT_COLORS)
    singleton_colors = random.sample([c for c in OBJECT_COLORS if c != mover_color], len(singleton_corners))

    dims: dict[str, tuple[int, int]] = {}
    source: dict[str, tuple[int, int, int, int]] = {}
    target: dict[str, tuple[int, int, int, int]] = {}
    for corner in CORNERS:
        rh, rw = random.randint(1, 3), random.randint(1, 3)
        dims[corner] = (rh, rw)
        source[corner] = _corner_box(corner, h, w, rh, rw, False)
        target[corner] = _corner_box(corner, h, w, rh, rw, True)

    moving_targets = [target[c] for c in moving_corners]
    if any(_overlap(a, b) for i, a in enumerate(moving_targets) for b in moving_targets[i + 1 :]):
        return None
    if any(_overlap(target[m], source[s]) for m in moving_corners for s in singleton_corners):
        return None

    # Build the post-motion partition skeleton. Two permanent crossing walls make
    # the input itself a tempting but wrong partition; each mover adds another
    # full barrier whose gap exists only before motion.
    candidate_rows = [r for r in range(5, h - 5) if all(not (b[0] <= r <= b[2]) for b in moving_targets)]
    candidate_cols = [c for c in range(5, w - 5) if all(not (b[1] <= c <= b[3]) for b in moving_targets)]
    if not candidate_rows or not candidate_cols:
        return None
    base_r, base_c = random.choice(candidate_rows), random.choice(candidate_cols)
    moved = np.full((h, w), BLACK, dtype=int)
    moved[base_r, :] = GREY
    moved[:, base_c] = GREY

    barrier_keys: set[tuple[str, int]] = set()
    for box in moving_targets:
        r0, c0, r1, c1 = box
        choices = [("r", random.randint(r0, r1)), ("c", random.randint(c0, c1))]
        random.shuffle(choices)
        key = next((x for x in choices if x not in barrier_keys), None)
        if key is None:
            return None
        barrier_keys.add(key)
        if key[0] == "r":
            moved[key[1], :] = GREY
        else:
            moved[:, key[1]] = GREY

    # Every target must be entirely empty before its colored rectangle arrives.
    for r0, c0, r1, c1 in moving_targets:
        moved[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for r0, c0, r1, c1 in moving_targets:
        moved[r0 : r1 + 1, c0 : c1 + 1] = mover_color

    specs: list[dict] = []
    for corner in moving_corners:
        specs.append({"moving": True, "color": mover_color, "source": source[corner], "target": target[corner]})
    for corner, color in zip(singleton_corners, singleton_colors):
        r0, c0, r1, c1 = source[corner]
        moved[r0 : r1 + 1, c0 : c1 + 1] = color
        specs.append({"moving": False, "color": color, "source": source[corner], "target": source[corner]})

    post_regions = _components(moved == BLACK)
    areas = np.array([len(x) for x in post_regions], dtype=int)
    if len(areas) < 5 or np.count_nonzero(areas == areas.min()) != 1 or np.count_nonzero(areas == areas.max()) != 1:
        return None
    if int(areas.max()) < 2 * int(areas.min()):
        return None

    inp = moved.copy()
    for spec in specs:
        if spec["moving"]:
            r0, c0, r1, c1 = spec["target"]
            inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for spec in specs:
        if spec["moving"]:
            r0, c0, r1, c1 = spec["source"]
            if np.any(inp[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
                return None
            inp[r0 : r1 + 1, c0 : c1 + 1] = mover_color

    # Semantic checks: input has a genuine lure partition, motion changes it,
    # every moved rectangle is causally necessary, and grey walls are necessary.
    input_regions = _components(inp == BLACK)
    input_areas = np.array([len(x) for x in input_regions], dtype=int)
    if len(input_regions) < 2 or np.count_nonzero(input_areas == input_areas.min()) != 1 or np.count_nonzero(input_areas == input_areas.max()) != 1:
        return None
    if _region_signature(inp) == _region_signature(moved):
        return None
    expected_moved, expected = _independent_expected(inp, specs)
    if not np.array_equal(expected_moved, moved):
        return None
    try:
        wrong_direct = _paint_extremes(inp)
    except AssertionError:
        return None
    if np.array_equal(wrong_direct, expected):
        return None
    full_signature = _region_signature(moved)
    for spec in specs:
        if not spec["moving"]:
            continue
        altered = moved.copy()
        r0, c0, r1, c1 = spec["target"]
        altered[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        if _region_signature(altered) == full_signature:
            return None
    no_walls = moved.copy()
    no_walls[no_walls == GREY] = BLACK
    if _region_signature(no_walls) == full_signature:
        return None
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except (AssertionError, ValueError):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate instance satisfying the full causal rule."""
    for _ in range(1200):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid move-then-partition puzzle")
