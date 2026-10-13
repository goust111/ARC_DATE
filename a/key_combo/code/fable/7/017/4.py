"""ARC task: filter colored X sprites by a folded 3x3 fingerprint."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


# Coordinates are always (row, column).  This is the hidden folding bijection:
# each pair is (coordinate in a 5x5 X, coordinate in the 3x3 legend).
FOLD_MAP = (
    ((0, 0), (0, 0)),
    ((1, 1), (0, 1)),
    ((0, 4), (0, 2)),
    ((3, 1), (1, 0)),
    ((2, 2), (1, 1)),
    ((1, 3), (1, 2)),
    ((4, 0), (2, 0)),
    ((3, 3), (2, 1)),
    ((4, 4), (2, 2)),
)
X_COORDS = frozenset(src for src, _ in FOLD_MAP)


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected True components using the common (r, c) convention."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result: list[list[tuple[int, int]]] = []
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        component = [seed]
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    p = (r + dr, c + dc)
                    if 0 <= p[0] < h and 0 <= p[1] < w and p in unseen:
                        unseen.remove(p)
                        queue.append(p)
                        component.append(p)
        result.append(component)
    return result


def _component_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*component)
    return min(rows), min(cols), max(rows), max(cols)


def _fold_x(patch: np.ndarray) -> np.ndarray:
    folded = np.empty((3, 3), dtype=patch.dtype)
    for (sr, sc), (dr, dc) in FOLD_MAP:
        folded[dr, dc] = patch[sr, sc]
    return folded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold every colored 5x5 X into a fingerprint and erase those not matching the corner legend.

    Input structure:
    A dominant-color background contains one solid colored 3x3 legend in a
    corner and several separated 5x5 X sprites.  An X has colored pixels only
    on its two diagonals; its nine diagonal pixels may have arbitrary colors.

    Rule:
    Keep the 3x3 legend.  Fold each X by mapping its northwest arm to the top
    edge, northeast arm to the right edge, southwest arm to the left edge,
    southeast arm to the bottom edge, and its center to the legend center.
    Preserve the original X exactly iff the resulting 3x3 array equals the
    legend cell by cell; otherwise erase all nine of its pixels to background.

    Key insight:
    The diagonal-to-border fold is a positional bijection.  Equal color counts
    are insufficient because a decoy has the legend's multiset in wrong cells.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a two-dimensional grid"
    bg = _background(g)
    components = _components8(g != bg)

    legends: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    x_objects: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for component in components:
        r0, c0, r1, c1 = _component_bbox(component)
        local = {(r - r0, c - c0) for r, c in component}
        if (r1 - r0 + 1, c1 - c0 + 1) == (3, 3) and len(component) == 9:
            legends.append(((r0, c0, r1, c1), g[r0 : r1 + 1, c0 : c1 + 1]))
        elif (r1 - r0 + 1, c1 - c0 + 1) == (5, 5) and local == X_COORDS:
            x_objects.append(((r0, c0, r1, c1), g[r0 : r1 + 1, c0 : c1 + 1]))
        else:
            raise AssertionError("foreground component is neither the legend nor a 5x5 X")

    assert len(legends) == 1, "expected one uniquely shaped 3x3 legend"
    assert x_objects, "expected at least one X sprite"
    (lr0, lc0, lr1, lc1), legend = legends[0]
    h, w = g.shape
    assert (lr0, lc0) in {(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)}, \
        "the legend must occupy a grid corner"

    out = g.copy()
    for (r0, c0, r1, c1), patch in x_objects:
        if not np.array_equal(_fold_x(patch), legend):
            for r, c in X_COORDS:
                out[r0 + r, c0 + c] = bg
    return out


def _x_from_fingerprint(fingerprint: np.ndarray, bg: int) -> np.ndarray:
    patch = np.full((5, 5), bg, dtype=int)
    for (sr, sc), (dr, dc) in FOLD_MAP:
        patch[sr, sc] = int(fingerprint[dr, dc])
    return patch


def _rects_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    """True when two inclusive rectangles have at least ``gap`` blank rows/columns."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _semantic_validation(
    grid: np.ndarray,
    expected: np.ndarray,
    legend: np.ndarray,
    records: list[tuple[str, int, int, np.ndarray]],
    bg: int,
) -> None:
    """Check puzzle semantics independently of the solver's final equality test."""
    kinds = Counter(kind for kind, *_ in records)
    assert kinds["match"] >= 1 and kinds["multiset_decoy"] >= 1 and kinds["other_decoy"] >= 1
    legend_count = Counter(map(int, legend.flat))
    changed = False
    for kind, r0, c0, fingerprint in records:
        patch_mask = np.zeros((5, 5), dtype=bool)
        for r, c in X_COORDS:
            patch_mask[r, c] = True
        original = grid[r0 : r0 + 5, c0 : c0 + 5]
        assert np.count_nonzero(original != bg) == 9
        assert np.all(original[~patch_mask] == bg)
        if kind == "match":
            assert np.array_equal(fingerprint, legend)
            assert np.array_equal(expected[r0 : r0 + 5, c0 : c0 + 5], original)
        else:
            assert not np.array_equal(fingerprint, legend)
            assert np.all(expected[r0 : r0 + 5, c0 : c0 + 5] == bg)
            changed = True
        if kind == "multiset_decoy":
            assert Counter(map(int, fingerprint.flat)) == legend_count
        if kind == "other_decoy":
            assert Counter(map(int, fingerprint.flat)) != legend_count
    assert changed and not np.array_equal(grid, expected)


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate folding-filter instance."""
    for _ in range(300):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        bg = 0
        grid = np.full((h, w), bg, dtype=int)

        # Four to six visible colors make positional arrangement salient while
        # retaining repetitions, so a multiset comparison remains tempting.
        palette = random.sample(range(1, 10), random.randint(4, 6))
        cells = palette.copy()
        cells.extend(random.choices(palette, k=9 - len(cells)))
        random.shuffle(cells)
        legend = np.array(cells, dtype=int).reshape(3, 3)

        corner = random.randrange(4)
        lr = 0 if corner < 2 else h - 3
        lc = 0 if corner % 2 == 0 else w - 3
        grid[lr : lr + 3, lc : lc + 3] = legend
        occupied = [(lr, lc, lr + 2, lc + 2)]

        total = random.randint(3, 6)
        match_count = random.randint(1, min(2, total - 2))
        kinds = ["match"] * match_count + ["multiset_decoy", "other_decoy"]
        kinds += random.choices(["match", "multiset_decoy", "other_decoy"], k=total - len(kinds))
        random.shuffle(kinds)

        records: list[tuple[str, int, int, np.ndarray]] = []
        failed = False
        for kind in kinds:
            if kind == "match":
                fingerprint = legend.copy()
            elif kind == "multiset_decoy":
                flat = legend.ravel().copy()
                for _swap_try in range(50):
                    i, j = random.sample(range(9), 2)
                    if flat[i] != flat[j]:
                        flat[i], flat[j] = flat[j], flat[i]
                        break
                fingerprint = flat.reshape(3, 3)
                if np.array_equal(fingerprint, legend):
                    failed = True
                    break
            else:
                fingerprint = legend.copy()
                unused = list(set(range(1, 10)) - set(map(int, legend.flat)))
                if not unused:
                    failed = True
                    break
                rr, cc = random.randrange(3), random.randrange(3)
                fingerprint[rr, cc] = random.choice(unused)

            location = None
            for _place_try in range(300):
                r0 = random.randint(1, h - 6)
                c0 = random.randint(1, w - 6)
                rect = (r0, c0, r0 + 4, c0 + 4)
                if all(_rects_clear(rect, other) for other in occupied):
                    location = (r0, c0, rect)
                    break
            if location is None:
                failed = True
                break
            r0, c0, rect = location
            grid[r0 : r0 + 5, c0 : c0 + 5] = _x_from_fingerprint(fingerprint, bg)
            occupied.append(rect)
            records.append((kind, r0, c0, fingerprint.copy()))

        if failed:
            continue
        expected = grid.copy()
        for kind, r0, c0, _ in records:
            if kind != "match":
                for r, c in X_COORDS:
                    expected[r0 + r, c0 + c] = bg

        try:
            _semantic_validation(grid, expected, legend, records, bg)
            if np.array_equal(main(grid), expected):
                return grid
        except AssertionError:
            pass
    raise RuntimeError("failed to generate a valid folding-fingerprint input")
