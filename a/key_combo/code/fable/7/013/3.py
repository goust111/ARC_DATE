"""ARC task: repair three differently posed, independently corrupted sprites."""

from __future__ import annotations

import random
from collections import Counter, deque
from itertools import product

import numpy as np


def _transform(a: np.ndarray, code: int) -> np.ndarray:
    """One of the eight square-grid rigid transforms (four rotations + reflections)."""
    code = int(code)
    if code < 4:
        return np.rot90(a, code).copy()
    return np.fliplr(np.rot90(a, code - 4)).copy()


def _inverse_transform(a: np.ndarray, code: int) -> np.ndarray:
    """Undo ``_transform(..., code)`` without relying on sprite contents."""
    code = int(code)
    if code < 4:
        return np.rot90(a, (-code) % 4).copy()
    # Every reflection in D4 is an involution in the convention above.
    return _transform(a, code)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (row, column) coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive (r0, c0, r1, c1) bounding box in the shared ARC convention."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _majority(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Cellwise majority; all-different cells choose a only for alignment scoring."""
    return np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))


def _alignment_candidates(crops: list[np.ndarray]):
    """Enumerate relative poses, fixing crop zero as the common reference pose."""
    valid1 = [code for code in range(8) if _transform(crops[1], code).shape == crops[0].shape]
    valid2 = [code for code in range(8) if _transform(crops[2], code).shape == crops[0].shape]
    for t1, t2 in product(valid1, valid2):
        aligned = [crops[0], _transform(crops[1], t1), _transform(crops[2], t2)]
        truth = _majority(*aligned)
        loss = sum(int(np.count_nonzero(x != truth)) for x in aligned)
        yield loss, (0, t1, t2), aligned, truth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair three corrupted rigidly transformed copies by pose-normalized majority vote.

    Input structure:
    - A dominant background separates exactly three 4-connected multicolor sprites.
    - Their square bounding-box crops have equal size and are three distinct rigid poses
      of one asymmetric pattern, with a few wrong cells in every copy.
    - In the common sprite coordinate system, at most one copy is wrong at each cell.

    Rule:
    - Crop the three sprites in their own grid positions and exhaustively align their
      rotations/reflections, choosing the unique alignment with least majority disagreement.
    - Take the cellwise three-way majority in that normalized pose, undo each copy's
      alignment, and replace its full original bounding-box crop with the repaired sprite.

    Key insight:
    - Direct coordinates compare different poses; only after rigid alignment do two clean
      observations agree at every corrupted slot and reveal the true color by majority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    comps = _components(g != bg)
    assert len(comps) == 3, "expected exactly three separated sprites"
    boxes = sorted((_bbox(coords) for coords in comps), key=lambda b: (b[0], b[1]))
    shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
    assert len(shapes) == 1, "sprite bounding boxes must be equally sized"
    shape = next(iter(shapes))
    crops = [g[r0 : r1 + 1, c0 : c1 + 1].copy() for r0, c0, r1, c1 in boxes]

    ranked = sorted(_alignment_candidates(crops), key=lambda item: item[0])
    assert ranked[0][0] < ranked[1][0], "relative sprite poses must be uniquely determined"
    _, transforms, _, truth = ranked[0]

    out = g.copy()
    for box, code in zip(boxes, transforms):
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = _inverse_transform(truth, code)
    return out


def _has_no_self_symmetry(sprite: np.ndarray) -> bool:
    return all(not np.array_equal(sprite, _transform(sprite, k)) for k in range(1, 8))


def _make_truth(n: int, bg: int, colors: list[int]) -> np.ndarray:
    """Build a connected, visibly multicolor sprite with a stable full bounding box."""
    sprite = np.empty((n, n), dtype=int)
    sprite[:, :] = np.random.choice(colors, size=(n, n))
    # Interior transparent cells make the object a sprite rather than a solid color panel.
    interior = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
    random.shuffle(interior)
    for r, c in interior[: random.randint(1, max(1, len(interior) // 4))]:
        sprite[r, c] = bg
    # Force every selected visible color to occur, without weakening the solid border.
    for c, color in enumerate(colors):
        sprite[0, c] = color
    return sprite


def _place_boxes(h: int, w: int, n: int) -> list[tuple[int, int]] | None:
    """Place three n-by-n boxes with a two-cell visual gap."""
    candidates = [(r, c) for r in range(1, h - n) for c in range(1, w - n)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(r + n + 1 < rr or rr + n + 1 < r or c + n + 1 < cc or cc + n + 1 < c
               for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == 3:
                return chosen
    return None


def _unique_best_and_expected(inp: np.ndarray, expected: np.ndarray) -> bool:
    """Semantic oracle: require unique poses, exact repair, and non-degenerate changes."""
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(inp, expected):
        return False

    bg = int(Counter(map(int, inp.ravel())).most_common(1)[0][0])
    boxes = sorted((_bbox(x) for x in _components(inp != bg)), key=lambda b: (b[0], b[1]))
    if len(boxes) != 3:
        return False
    # Each copy must genuinely need repair, excluding a pristine-copy shortcut.
    return all(np.any(inp[r0 : r1 + 1, c0 : c1 + 1] != expected[r0 : r1 + 1, c0 : c1 + 1])
               for r0, c0, r1, c1 in boxes)


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance satisfying all puzzle promises."""
    for _ in range(500):
        n = random.randint(4, 7)
        h = random.randint(max(16, n * 2 + 5), 30)
        w = random.randint(max(16, n * 2 + 5), 30)
        bg = random.randrange(10)
        colors = random.sample([x for x in range(10) if x != bg], random.randint(3, min(5, n)))
        truth = _make_truth(n, bg, colors)
        if not _has_no_self_symmetry(truth):
            continue

        poses = random.sample(range(8), 3)
        locations = _place_boxes(h, w, n)
        if locations is None:
            continue

        # Every copy has 1--3 corruptions; normalized slots are pairwise disjoint.
        slots = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
        total = min(len(slots), sum(random.randint(1, 3) for _ in range(3)))
        if total < 3:
            continue
        chosen_slots = random.sample(slots, total)
        counts = [1, 1, 1]
        for _extra in range(total - 3):
            eligible = [i for i in range(3) if counts[i] < 3]
            counts[random.choice(eligible)] += 1

        inp = np.full((h, w), bg, dtype=int)
        expected = inp.copy()
        cursor = 0
        for i, ((r0, c0), pose) in enumerate(zip(locations, poses)):
            damaged = truth.copy()
            for r, c in chosen_slots[cursor : cursor + counts[i]]:
                damaged[r, c] = random.choice([x for x in range(10) if x != int(truth[r, c])])
            cursor += counts[i]
            observed = _transform(damaged, pose)
            clean = _transform(truth, pose)
            inp[r0 : r0 + n, c0 : c0 + n] = observed
            expected[r0 : r0 + n, c0 : c0 + n] = clean

        # Distinct poses make normalization indispensable; this further rejects any
        # rare corrupted pattern whose best relative alignment is tied or incorrect.
        if _unique_best_and_expected(inp, expected):
            return inp
    raise RuntimeError("failed to generate a unique non-degenerate three-pose instance")
