from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _solid_square(mask: np.ndarray) -> tuple[int, int, int] | None:
    """Return (top, left, side) when mask is exactly one filled square."""
    cells = np.argwhere(mask)
    if cells.size == 0:
        return None
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    n = int(r1 - r0 + 1)
    if n != int(c1 - c0 + 1) or int(cells.shape[0]) != n * n:
        return None
    if not bool(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].all()):
        return None
    return int(r0), int(c0), n


def _feasible_windows(channel: np.ndarray, n: int) -> np.ndarray:
    """Top-left positions whose complete n-square lies in the channel."""
    h, w = channel.shape
    feasible = np.zeros((h - n + 1, w - n + 1), dtype=bool)
    # Small ARC grids make this direct form clearer than an integral image.
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            feasible[r, c] = bool(channel[r : r + n, c : c + n].all())
    return feasible


def _reachable(feasible: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    seen = np.zeros_like(feasible, dtype=bool)
    sr, sc = start
    assert feasible[sr, sc], "mover must begin at a legal placement"
    seen[sr, sc] = True
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if (0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1]
                    and feasible[rr, cc] and not seen[rr, cc]):
                seen[rr, cc] = True
                queue.append((rr, cc))
    return seen


def _exact_pocket_caps(
    channel: np.ndarray, feasible: np.ndarray, n: int
) -> list[tuple[int, int]]:
    """Find exact-fit terminal chambers by their three exposed outer sides."""
    h, w = channel.shape
    result: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(feasible)):
        side_clear = [
            r == 0 or not bool(channel[r - 1, c : c + n].any()),
            r + n == h or not bool(channel[r + n, c : c + n].any()),
            c == 0 or not bool(channel[r : r + n, c - 1].any()),
            c + n == w or not bool(channel[r : r + n, c + n].any()),
        ]
        if sum(side_clear) == 3:
            result.append((int(r), int(c)))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Slide the solid square into the unique reachable, exactly fitting channel pocket.

    Input structure:
    - The most common color is background; one other color forms a connected,
      branching channel with terminal square pockets of several side lengths.
    - The remaining color forms one solid N×N mover lying on the channel.

    Rule:
    - Restore the mover's cells to channel color and regard each top-left position
      whose full N×N window is channel as a legal placement.
    - Search those placements by 4-neighbor slides.  Among terminal N×N pockets
      (three exposed sides and one channel entrance), fill the unique reachable one
      with the mover color.  Wrong-sized and bottleneck-isolated pockets stay fixed.

    Key insight:
    - Shape match alone retains the isolated decoy, while reachability alone retains
      wrong-sized chambers; exact fit and width-constrained path search are both needed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    foreground = sorted(set(map(int, g.ravel())) - {bg})
    assert len(foreground) == 2, "expected channel and mover colors"

    square_specs = []
    for color in foreground:
        spec = _solid_square(g == color)
        if spec is not None:
            square_specs.append((int(color), spec))
    assert len(square_specs) == 1, "the mover must be the unique solid-square color"

    mover, (sr, sc, n) = square_specs[0]
    river = next(color for color in foreground if color != mover)
    channel = (g == river) | (g == mover)
    feasible = _feasible_windows(channel, n)
    reached = _reachable(feasible, (sr, sc))
    pockets = _exact_pocket_caps(channel, feasible, n)
    destinations = [p for p in pockets if reached[p] and p != (sr, sc)]
    assert len(destinations) == 1, "expected one reachable exact-fit pocket"

    tr, tc = destinations[0]
    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = mover
    return out


def _rect(mask: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    mask[r : r + h, c : c + w] = True


def _orient(a: np.ndarray, code: int) -> np.ndarray:
    """One of all eight dihedral orientations, preserving a rectangular canvas."""
    code = int(code)
    if code >= 4:
        a = np.fliplr(a)
        code -= 4
    return np.rot90(a, code).copy()


def _pad_randomly(
    a: np.ndarray, value: int
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:
    room_r = 30 - a.shape[0]
    room_c = 30 - a.shape[1]
    total_r = random.randint(0, min(3, room_r))
    total_c = random.randint(0, min(3, room_c))
    top = random.randint(0, total_r)
    left = random.randint(0, total_c)
    padding = ((top, total_r - top), (left, total_c - left))
    return np.pad(a, padding, constant_values=int(value)), padding


def _build_canonical(n: int, add_small: bool) -> tuple[np.ndarray, dict]:
    """Build the geometry as a boolean channel plus named semantic landmarks."""
    h, w = 5 * n + 7, 5 * n + 8
    channel = np.zeros((h, w), dtype=bool)
    upper = 2
    base = 2 * n + 4
    bend_c = 3 * n + 2

    # A reachable winding N-wide route: lower leg, rising leg, upper leg.
    _rect(channel, base, n + 3, n, 3 * n - 1)
    _rect(channel, upper, bend_c, base + n - upper, n)
    target = (upper, 4 * n + 5)
    _rect(channel, upper, bend_c, n, target[1] + n - bend_c)

    # A reachable but over-sized terminal chamber at the lower leg's left end.
    large_side = random.choice((n + 1, n + 2))
    large_top = base - (large_side - n + 1) // 2
    large_left = n + 3 - large_side
    _rect(channel, large_top, large_left, large_side, large_side)

    # The nearer same-sized decoy is joined by a long (N-1)-wide bottleneck.
    neck_c = 2 * n + 3
    pocket_top = base + n + 3
    decoy = (pocket_top, neck_c - 1)
    _rect(channel, base + n, neck_c, pocket_top - (base + n) + 1, n - 1)
    _rect(channel, decoy[0], decoy[1], n, n)

    # Sometimes add a fourth endpoint: a visibly too-small side pocket.
    small = None
    if add_small:
        side = n - 1
        branch_r = upper + n + 1
        small_c = bend_c - n - 4
        small = (branch_r, small_c, side)
        _rect(channel, branch_r, small_c, side, bend_c - small_c)

    source = (base, n + 3)
    facts = {
        "source": source,
        "target": target,
        "decoy": decoy,
        "large": (large_top, large_left, large_side),
        "small": small,
        "neck": (base + n, neck_c, pocket_top - (base + n) + 1, n - 1),
    }
    return channel, facts


def _validate_semantics(channel: np.ndarray, facts: dict, n: int) -> None:
    """Check the intended rule itself, including shortcut-rejecting counterexamples."""
    source = facts["source"]
    target = facts["target"]
    decoy = facts["decoy"]
    feasible = _feasible_windows(channel, n)
    reached = _reachable(feasible, source)
    caps = _exact_pocket_caps(channel, feasible, n)

    assert set(caps) == {target, decoy}
    assert reached[target] and not reached[decoy]
    assert sum(bool(reached[p]) for p in caps) == 1
    assert facts["large"][2] != n
    if facts["small"] is not None:
        assert facts["small"][2] != n
    # The tempting exact match is geometrically nearer, excluding nearest-target logic.
    distance = lambda p: abs(p[0] - source[0]) + abs(p[1] - source[1])
    assert distance(decoy) < distance(target)

    # Widening only the stated bottleneck to N must make the decoy reachable.
    widened = channel.copy()
    rr, cc, hh, ww = facts["neck"]
    _rect(widened, rr, cc - 1, hh, ww + 1)
    widened_f = _feasible_windows(widened, n)
    assert _reachable(widened_f, source)[decoy]


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate instances with all semantic distractors."""
    for _ in range(200):
        n = random.choice((2, 3, 4))
        add_small = random.random() < 0.55
        channel, facts = _build_canonical(n, add_small)
        try:
            _validate_semantics(channel, facts, n)
        except AssertionError:
            continue

        bg, river, mover = random.sample(range(10), 3)
        inp = np.full(channel.shape, bg, dtype=int)
        inp[channel] = river
        sr, sc = facts["source"]
        tr, tc = facts["target"]
        inp[sr : sr + n, sc : sc + n] = mover

        expected = inp.copy()
        expected[sr : sr + n, sc : sc + n] = river
        expected[tr : tr + n, tc : tc + n] = mover

        orientation = random.randrange(8)
        inp = _orient(inp, orientation)
        expected = _orient(expected, orientation)
        inp, padding = _pad_randomly(inp, bg)
        expected = np.pad(expected, padding, constant_values=bg)
        if max(inp.shape) > 30 or _background(inp) != bg:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue

        if not np.array_equal(actual, expected):
            continue

        # The two moved squares are disjoint, so exactly 2*N*N cells must change.
        changed = actual != inp
        if int(changed.sum()) != 2 * n * n:
            continue
        if int(np.count_nonzero(inp == mover)) != n * n:
            continue
        if int(np.count_nonzero(actual == mover)) != n * n:
            continue
        return inp
    raise RuntimeError("failed to generate a valid exact-pocket instance")
