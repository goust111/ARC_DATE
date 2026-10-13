from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


RED = 2
ARM_OFFSETS = (
    (0, -3), (0, -2), (0, 2), (0, 3),
    (-3, 0), (-2, 0), (2, 0), (3, 0),
)
GAP_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ARM_PAIRS = (
    ((0, -3), (0, -2)),
    ((0, 2), (0, 3)),
    ((-3, 0), (-2, 0)),
    ((2, 0), (3, 0)),
)


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    """Small self-contained counterpart of arc_utils.components_mask."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    component.add((nr, nc))
                    queue.append((nr, nc))
        result.append(component)
    return result


def _two_pixel_bars(grid: np.ndarray, bg: int) -> list[set[tuple[int, int]]]:
    foreground = sorted(set(map(int, grid.ravel())) - {bg})
    assert len(foreground) == 1, "input must have one foreground color"
    assert foreground[0] != RED, "red is reserved for output connectors"
    bars = _components4(grid == foreground[0])
    assert bars, "expected foreground bars"
    for bar in bars:
        assert len(bar) == 2, "every foreground component must be a two-pixel bar"
        (r0, c0), (r1, c1) = tuple(bar)
        assert abs(r0 - r1) + abs(c0 - c1) == 1, "bar pixels must be adjacent"
    return bars


def _find_centers(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    bars = _two_pixel_bars(grid, bg)
    bar_lookup = {frozenset(bar) for bar in bars}
    centers = []
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            if int(grid[r, c]) != bg:
                continue
            if any(int(grid[r + dr, c + dc]) != bg for dr, dc in GAP_OFFSETS):
                continue
            required = [
                frozenset((r + dr, c + dc) for dr, dc in pair)
                for pair in ARM_PAIRS
            ]
            if all(bar in bar_lookup for bar in required):
                centers.append((r, c))
    return centers


def _own_bar_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in ARM_OFFSETS}


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    lo, hi = sorted((r0, r1))
    return [(r, c0) for r in range(lo, hi + 1)]


def _adjacent_pairs(centers: list[tuple[int, int]]):
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in centers:
        rows[r].append(c)
        cols[c].append(r)
    for r, cs in rows.items():
        ordered = sorted(cs)
        for c0, c1 in zip(ordered, ordered[1:]):
            yield (r, c0), (r, c1)
    for c, rs in cols.items():
        ordered = sorted(rs)
        for r0, r1 in zip(ordered, ordered[1:]):
            yield (r0, c), (r1, c)


def main(grid: np.ndarray) -> np.ndarray:
    """Connect adjacent aligned hidden anchors with red lines through clear background.

    Input structure:
    - The dominant color is the background, and one other color forms isolated
      horizontal or vertical two-pixel bars. Four radial bars at offsets two and
      three around an empty cell imply a missing plus center; incomplete groups
      are distractors.

    Rule:
    - Detect only empty cells having all four complete two-pixel radial bars.
      In each row and column, sort these centers and consider consecutive centers.
      Connect a pair exactly when its inclusive center-to-center segment contains
      only background apart from the two endpoint motifs' own bars. Recolor every
      background cell on such a segment, including both centers, red; preserve all
      bars, distractors, and blocked segments.

    Key insight:
    - The visible bars define invisible anchors first; those anchors then form a
      row/column visibility graph whose consecutive clear edges are the red lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    assert bg != RED, "red is reserved for output connectors"
    centers = _find_centers(g, bg)

    out = g.copy()
    for a, b in _adjacent_pairs(centers):
        path = _segment(a, b)
        exempt = _own_bar_cells(a) | _own_bar_cells(b)
        if all(int(g[r, c]) == bg or (r, c) in exempt for r, c in path):
            for r, c in path:
                if int(g[r, c]) == bg:
                    out[r, c] = RED
    return out


def _isolated_from_foreground(grid: np.ndarray, cells: set[tuple[int, int]], bg: int) -> bool:
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != bg for r, c in cells):
        return False
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) != bg:
                    return False
    return True


def _place_bar(grid: np.ndarray, cells: set[tuple[int, int]], color: int, bg: int) -> bool:
    if not _isolated_from_foreground(grid, cells, bg):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _place_anchor(grid: np.ndarray, center: tuple[int, int], color: int, bg: int) -> bool:
    placed: list[set[tuple[int, int]]] = []
    r, c = center
    for pair in ARM_PAIRS:
        bar = {(r + dr, c + dc) for dr, dc in pair}
        if not _place_bar(grid, bar, color, bg):
            for old in placed:
                for rr, cc in old:
                    grid[rr, cc] = bg
            return False
        placed.append(bar)
    return True


def _oracle(grid: np.ndarray, centers: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, int, int]:
    """Independent expected construction from the generator's declared anchors."""
    out = grid.copy()
    clear_count = blocked_count = 0
    for a, b in _adjacent_pairs(centers):
        own = _own_bar_cells(a) | _own_bar_cells(b)
        path = _segment(a, b)
        clear = not any(int(grid[r, c]) != bg and (r, c) not in own for r, c in path)
        if clear:
            clear_count += 1
            for r, c in path:
                if int(grid[r, c]) == bg:
                    out[r, c] = RED
        else:
            blocked_count += 1
    return out, clear_count, blocked_count


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()
    bg = rng.choice([0, 8, 9])
    fg = rng.choice([x for x in range(1, 10) if x not in {bg, RED}])
    topology = rng.choice(("rectangle", "tee"))

    if topology == "rectangle":
        base = [(0, 0), (0, 10), (10, 0), (10, 10)]
    else:
        base = [(0, 0), (0, 10), (0, 20), (10, 10)]
    if rng.random() < 0.5:
        base = [(c, r) for r, c in base]

    span_r = max(r for r, _ in base)
    span_c = max(c for _, c in base)
    h = rng.randint(span_r + 8, 30)
    w = rng.randint(span_c + 8, 30)
    shift_r = rng.randint(3, h - span_r - 4)
    shift_c = rng.randint(3, w - span_c - 4)
    centers = [(r + shift_r, c + shift_c) for r, c in base]
    grid = np.full((h, w), bg, dtype=int)
    if not all(_place_anchor(grid, center, fg, bg) for center in centers):
        return None

    pairs = list(_adjacent_pairs(centers))
    # Keep at least one clear horizontal and one clear vertical edge after the
    # obstruction is installed, so every generated example covers both axes.
    blockable = [
        edge for edge in pairs
        if {a[0] == b[0] for a, b in pairs if (a, b) != edge} == {True, False}
    ]
    blocked_pair = rng.choice(blockable)
    (r0, c0), (r1, c1) = blocked_pair
    if r0 == r1:
        mid = (r0, (c0 + c1) // 2)
        blocker = {mid, (mid[0] + rng.choice((-1, 1)), mid[1])}
    else:
        mid = ((r0 + r1) // 2, c0)
        blocker = {mid, (mid[0], mid[1] + rng.choice((-1, 1)))}
    if not _place_bar(grid, blocker, fg, bg):
        return None

    # Add one to three incomplete two/three-arm motifs. Re-scan after each
    # proposal so a distractor can never accidentally become a fifth anchor.
    incomplete_count = rng.randint(1, 3)
    placed_incomplete = 0
    for _ in range(500):
        if placed_incomplete >= incomplete_count:
            break
        cr = rng.randint(3, h - 4)
        cc = rng.randint(3, w - 4)
        chosen = rng.sample(list(ARM_PAIRS), rng.choice((2, 3)))
        trial = grid.copy()
        ok = True
        for pair in chosen:
            cells = {(cr + dr, cc + dc) for dr, dc in pair}
            if not _place_bar(trial, cells, fg, bg):
                ok = False
                break
        if ok:
            try:
                found = set(_find_centers(trial, bg))
            except AssertionError:
                ok = False
            if ok and found == set(centers):
                grid = trial
                placed_incomplete += 1
    if placed_incomplete != incomplete_count:
        return None

    # Semantic checks: exact anchors, both path orientations, a clear branch,
    # a blocked branch, and observable dependence on both recognition and sight.
    if set(_find_centers(grid, bg)) != set(centers):
        return None
    expected, clear_count, blocked_count = _oracle(grid, centers, bg)
    if clear_count < 1 or blocked_count < 1 or not np.array_equal(main(grid), expected):
        return None
    orientations = {a[0] == b[0] for a, b in pairs}
    if orientations != {True, False}:
        return None
    clear_orientations = {
        a[0] == b[0]
        for a, b in pairs
        if _oracle(grid, [a, b], bg)[1] == 1
    }
    if clear_orientations != {True, False}:
        return None

    no_blocker = grid.copy()
    for r, c in blocker:
        no_blocker[r, c] = bg
    if np.array_equal(main(no_blocker), expected):
        return None

    # Remove one complete arm from an anchor incident to a clear edge. Its loss
    # must change the answer, proving all four-arm recognition is necessary.
    clear_edge = next(
        (a, b) for a, b in pairs
        if _oracle(grid, [a, b], bg)[1] == 1
    )
    victim = clear_edge[0]
    vr, vc = victim
    missing_arm = rng.choice(ARM_PAIRS)
    without_arm = grid.copy()
    for dr, dc in missing_arm:
        without_arm[vr + dr, vc + dc] = bg
    if victim in _find_centers(without_arm, bg) or np.array_equal(main(without_arm), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-anchor scene")
