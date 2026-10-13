from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


RED = 2
GENERATION_ATTEMPTS = 800


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude:
        colors -= {int(c) for c in exclude}
    return sorted(colors)


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    ranked = counts.most_common()
    assert not ranked or len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    return int(ranked[0][0])


def _complete_pair_candidates(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    candidates = []
    for pair in itertools.combinations(_colors_present(grid, exclude={RED}), 2):
        mask = np.isin(grid, pair)
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        row_count = int(rows.sum())
        col_count = int(cols.sum())
        if not (2 <= row_count < h and 2 <= col_count < w):
            continue
        if int(mask.sum()) != row_count * col_count:
            continue

        patch = grid[np.ix_(rows, cols)]
        if set(map(int, patch.ravel())) != set(map(int, pair)):
            continue
        candidates.append(
            {
                "pair": tuple(map(int, pair)),
                "rows": np.flatnonzero(rows),
                "cols": np.flatnonzero(cols),
                "patch": patch.copy(),
            }
        )
    return candidates


def _extract_mother(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    candidates = _complete_pair_candidates(grid)
    assert len(candidates) == 1, "expected exactly one row/column-product color pair"

    patch = candidates[0]["patch"]
    counts = Counter(map(int, patch.ravel()))
    eye_colors = [color for color, count in counts.items() if int(count) == 1]
    assert len(eye_colors) == 1, "expected exactly one singleton eye color"

    eye_coords = np.argwhere(patch == int(eye_colors[0]))
    assert eye_coords.shape[0] == 1
    eye_r, eye_c = map(int, eye_coords[0])
    return patch.copy(), (eye_r, eye_c)


def _stamp(
    out: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> None:
    h, w = patch.shape
    H, W = out.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(H, int(top) + h)
    c1 = min(W, int(left) + w)
    if r0 >= r1 or c0 >= c1:
        return
    pr0 = r0 - int(top)
    pc0 = c0 - int(left)
    out[r0:r1, c0:c1] = patch[pr0 : pr0 + (r1 - r0), pc0 : pc0 + (c1 - c0)]


def _render_output(
    shape: tuple[int, int],
    bg: int,
    mother: np.ndarray,
    eye: tuple[int, int],
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    eye_r, eye_c = map(int, eye)
    for ar, ac in sorted((int(r), int(c)) for r, c in anchors):
        _stamp(out, mother, int(ar) - eye_r, int(ac) - eye_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract a diluted complete two-color mother image and stamp it at every red anchor.

    Input structure:
    - A dominant background/filler canvas contains exactly one non-red color pair whose
      pixels fill every intersection of the rows and columns touched by that pair.
    - Other non-red color pairs act as decoys, including near-complete row/column
      products with one or two missing intersections.
    - In the compacted two-color mother image, one color appears in exactly one cell.
    - Red singleton pixels are anchors only; red is excluded from the color-pair search.

    Rule:
    - Find the unique complete row/column-product color pair and compact it by keeping
      only the rows and columns touched by that pair.
    - Use the singleton-color cell inside the compacted mother image as the alignment
      cell.
    - Return an all-background canvas of the original size, stamping the mother image at
      every red anchor so that the alignment cell lands exactly on the anchor.

    Key insight:
    - The stamp never appears compactly in the input; it must be recovered from the
      dispersed row/column product before the red pixels can be interpreted as anchors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    mother, eye = _extract_mother(g)
    anchor_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert anchor_coords, "expected at least one red anchor"
    return _render_output(g.shape, bg, mother, eye, anchor_coords)


# --------------- generator ---------------


def _choose_spread_indices(n: int, k: int) -> list[int] | None:
    pool = list(range(1, int(n) - 1))
    if len(pool) < k:
        return None
    for _ in range(100):
        idx = sorted(random.sample(pool, int(k)))
        if any(b - a > 1 for a, b in zip(idx, idx[1:])):
            return idx
    return sorted(random.sample(pool, int(k)))


def _product_cells(rows: list[int], cols: list[int]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r in rows for c in cols}


def _make_mother(height: int, width: int, body_color: int, eye_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    mother = np.full((int(height), int(width)), int(body_color), dtype=int)
    eye = (random.randrange(int(height)), random.randrange(int(width)))
    mother[eye] = int(eye_color)
    return mother, eye


def _paint_product(
    grid: np.ndarray,
    rows: list[int],
    cols: list[int],
    patch: np.ndarray,
    *,
    skip: set[tuple[int, int]] | None = None,
) -> None:
    skip = set() if skip is None else set(skip)
    for rr, r in enumerate(rows):
        for cc, c in enumerate(cols):
            if (int(rr), int(cc)) in skip:
                continue
            grid[int(r), int(c)] = int(patch[int(rr), int(cc)])


def _random_decoy_patch(height: int, width: int, color_a: int, color_b: int) -> np.ndarray:
    patch = np.full((int(height), int(width)), int(color_a), dtype=int)
    for r in range(int(height)):
        for c in range(int(width)):
            if random.random() < 0.38:
                patch[r, c] = int(color_b)
    if not np.any(patch == int(color_b)):
        patch[random.randrange(int(height)), random.randrange(int(width))] = int(color_b)
    if not np.any(patch == int(color_a)):
        patch[random.randrange(int(height)), random.randrange(int(width))] = int(color_a)
    return patch


def _choose_missing_cells(patch: np.ndarray, count: int) -> set[tuple[int, int]] | None:
    h, w = patch.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    for combo in itertools.combinations(cells, int(count)):
        remaining = patch.copy()
        for r, c in combo:
            remaining[r, c] = -1
        if all(np.any(remaining[r, :] != -1) for r in range(h)) and all(
            np.any(remaining[:, c] != -1) for c in range(w)
        ):
            colors = {int(v) for v in remaining.ravel() if int(v) != -1}
            if len(colors) == 2:
                return set(combo)
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _stamp_box(anchor: tuple[int, int], eye: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ar, ac = map(int, anchor)
    eye_r, eye_c = map(int, eye)
    h, w = map(int, shape)
    top = ar - eye_r
    left = ac - eye_c
    return top, left, top + h - 1, left + w - 1


def _box_in_bounds(box: tuple[int, int, int, int], H: int, W: int) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < int(H) and 0 <= c0 <= c1 < int(W)


def _isolated_from_red(anchor: tuple[int, int], anchors: list[tuple[int, int]]) -> bool:
    r, c = map(int, anchor)
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in anchors)


def _place_anchors(
    H: int,
    W: int,
    mother_shape: tuple[int, int],
    eye: tuple[int, int],
    protected: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    candidates = []
    for r in range(H):
        for c in range(W):
            if (r, c) in protected:
                continue
            box = _stamp_box((r, c), eye, mother_shape)
            if _box_in_bounds(box, H, W):
                candidates.append((r, c))
    random.shuffle(candidates)

    anchors: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for anchor in candidates:
        if not _isolated_from_red(anchor, anchors):
            continue
        box = _stamp_box(anchor, eye, mother_shape)
        if any(_boxes_overlap(box, old) for old in boxes):
            continue
        anchors.append(anchor)
        boxes.append(box)
        if len(anchors) == int(count):
            return anchors
    return None


def _near_complete_deficit(grid: np.ndarray, pair: tuple[int, int]) -> int | None:
    mask = np.isin(grid, pair)
    if not bool(mask.any()):
        return None
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    deficit = int(rows.sum()) * int(cols.sum()) - int(mask.sum())
    return int(deficit)


def _has_axis_gap(indices: np.ndarray) -> bool:
    values = list(map(int, indices))
    return any(b - a > 1 for a, b in zip(values, values[1:]))


def _semantic_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    target_pair: tuple[int, int],
    mother: np.ndarray,
    eye: tuple[int, int],
    decoy_pair: tuple[int, int],
    anchors: list[tuple[int, int]],
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        candidates = _complete_pair_candidates(grid)
        solved = main(grid)
    except Exception:
        return False

    if len(candidates) != 1:
        return False
    candidate = candidates[0]
    if set(candidate["pair"]) != set(map(int, target_pair)):
        return False
    if not np.array_equal(candidate["patch"], mother):
        return False
    if not np.array_equal(solved, expected):
        return False
    if _near_complete_deficit(grid, decoy_pair) not in (1, 2):
        return False
    if len(anchors) < 2:
        return False
    if not (_has_axis_gap(candidate["rows"]) and _has_axis_gap(candidate["cols"])):
        return False

    counts = Counter(map(int, mother.ravel()))
    if sorted(counts.values()) != [1, int(mother.size) - 1]:
        return False
    eye_r, eye_c = map(int, eye)
    return int(mother[eye_r, eye_c]) in counts


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(18, 30)
    W = random.randint(18, 30)
    colors = [c for c in range(10) if c != RED]
    bg, fill_a, fill_b, body, eye_color, decoy_a, decoy_b = random.sample(colors, 7)

    ph = random.randint(3, 6)
    pw = random.randint(3, 6)
    target_rows = _choose_spread_indices(H, ph)
    target_cols = _choose_spread_indices(W, pw)
    if target_rows is None or target_cols is None:
        return None

    mother, eye = _make_mother(ph, pw, body, eye_color)
    target_cells = _product_cells(target_rows, target_cols)

    for _ in range(100):
        dh = random.randint(3, 6)
        dw = random.randint(3, 6)
        decoy_rows = _choose_spread_indices(H, dh)
        decoy_cols = _choose_spread_indices(W, dw)
        if decoy_rows is None or decoy_cols is None:
            continue
        decoy_product = _product_cells(decoy_rows, decoy_cols)
        if decoy_product.isdisjoint(target_cells):
            break
    else:
        return None

    decoy_patch = _random_decoy_patch(dh, dw, decoy_a, decoy_b)
    missing_count = random.choice([1, 2])
    missing = _choose_missing_cells(decoy_patch, missing_count)
    if missing is None:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    for r in range(H):
        for c in range(W):
            x = random.random()
            if x < 0.08:
                grid[r, c] = int(fill_a)
            elif x < 0.18:
                grid[r, c] = int(fill_b)

    _paint_product(grid, target_rows, target_cols, mother)
    _paint_product(grid, decoy_rows, decoy_cols, decoy_patch, skip=missing)

    protected = target_cells | decoy_product
    anchor_count = random.randint(2, 4)
    anchors = _place_anchors(H, W, mother.shape, eye, protected, anchor_count)
    if anchors is None:
        return None
    for r, c in anchors:
        grid[int(r), int(c)] = RED

    expected = _render_output(grid.shape, bg, mother, eye, anchors)
    if not _semantic_valid(
        grid,
        expected,
        (body, eye_color),
        mother,
        eye,
        (decoy_a, decoy_b),
        anchors,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dispersed-stamp input")
