import random
from collections import deque

import numpy as np


BG = 0
BODY = 1
LEAF = 7

MAX_GENERATION_ATTEMPTS = 400
HEIGHT_RANGE = (6, 10)
WIDTH_RANGE = (8, 13)


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-tile the seed shape, then erase full rays named by seam-facing leaf pairs.

    Input structure:
    - A black background contains one irregular 4-connected blue shape.
    - Some single orange leaf pixels lie exactly on the outer border cells.
    - Every leaf touches the blue shape from the inward side by exactly one step.

    Rule:
    - Build a 2x2 mirror tiling: original at upper-left, left-right mirror at upper-right,
      up-down mirror at lower-left, and the double mirror at lower-right.
    - In the tiled image, detect orange leaf pairs that face each other across the central
      vertical seam or the central horizontal seam.
    - For every seam-facing horizontal pair, erase the entire output row to black; then
      for every seam-facing vertical pair, erase the entire output column to black.

    Key insight:
    - The cutting pairs do not exist in the input itself; they emerge only after mirror
      tiling brings edge leaves to the internal seams, where their shared row or column
      names the full-width or full-height erase ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 2, "expected a non-empty grid"

    top = np.concatenate((g, np.fliplr(g)), axis=1)
    tiled = np.concatenate((top, np.flipud(top)), axis=0)

    h, w = g.shape
    out = tiled.copy()

    cut_rows = [r for r in range(2 * h) if tiled[r, w - 1] == LEAF and tiled[r, w] == LEAF]
    cut_cols = [c for c in range(2 * w) if tiled[h - 1, c] == LEAF and tiled[h, c] == LEAF]

    for r in cut_rows:
        out[r, :] = BG
    for c in cut_cols:
        out[:, c] = BG

    return out


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    return cells > 0 and len(_components(mask)) == 1


def _has_no_holes(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return not np.any((~mask) & (~exterior))


def _mirror_tile(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, np.fliplr(grid)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _seam_cuts_from_tiled(tiled: np.ndarray, h: int, w: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(2 * h) if tiled[r, w - 1] == LEAF and tiled[r, w] == LEAF]
    cols = [c for c in range(2 * w) if tiled[h - 1, c] == LEAF and tiled[h, c] == LEAF]
    return rows, cols


def _leaf_plan(h: int, w: int) -> dict[str, list[int]] | None:
    rows = list(range(1, h - 1))
    cols = list(range(1, w - 1))
    if len(rows) < 2 or len(cols) < 2:
        return None

    branch = random.choice(["both", "right_only", "bottom_only", "both_with_distractors"])
    if branch == "both":
        nr = random.randint(1, min(2, len(rows)))
        nb = random.randint(1, min(2, len(cols)))
        nl = 0
        nt = 0
    elif branch == "right_only":
        nr = random.randint(1, min(2, len(rows)))
        nb = 0
        max_left = min(2, len(rows) - nr)
        max_top = min(2, len(cols))
        nl = random.randint(0, max_left)
        nt = random.randint(0, max_top)
    elif branch == "bottom_only":
        nr = 0
        nb = random.randint(1, min(2, len(cols)))
        max_left = min(2, len(rows))
        max_top = min(2, len(cols) - nb)
        nl = random.randint(0, max_left)
        nt = random.randint(0, max_top)
    else:
        nr = random.randint(1, min(2, len(rows)))
        nb = random.randint(1, min(2, len(cols)))
        max_left = min(1, len(rows) - nr)
        max_top = min(1, len(cols) - nb)
        nl = random.randint(0, max_left)
        nt = random.randint(0, max_top)

    total = nr + nb + nl + nt
    if not (2 <= total <= 4):
        return None

    right_rows = sorted(random.sample(rows, nr))
    bottom_cols = sorted(random.sample(cols, nb))

    left_candidates = [r for r in rows if r not in right_rows]
    top_candidates = [c for c in cols if c not in bottom_cols]
    if nl > len(left_candidates) or nt > len(top_candidates):
        return None

    left_rows = sorted(random.sample(left_candidates, nl))
    top_cols = sorted(random.sample(top_candidates, nt))
    return {
        "right": right_rows,
        "bottom": bottom_cols,
        "left": left_rows,
        "top": top_cols,
    }


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    if random.random() < 0.5:
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
    return path


def _grow_body(h: int, w: int, plan: dict[str, list[int]]) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)

    support = set()
    for r in plan["right"]:
        support.add((r, w - 2))
    for r in plan["left"]:
        support.add((r, 1))
    for c in plan["bottom"]:
        support.add((h - 2, c))
    for c in plan["top"]:
        support.add((1, c))

    interior_rows = list(range(2, h - 2)) if h >= 6 else [h // 2]
    interior_cols = list(range(2, w - 2)) if w >= 6 else [w // 2]
    center = (random.choice(interior_rows), random.choice(interior_cols))
    support.add(center)

    support_list = list(support)
    random.shuffle(support_list)
    anchor = center
    for cell in support_list:
        for r, c in _path_between(anchor, cell):
            mask[r, c] = True
        anchor = cell
    for cell in support_list:
        for r, c in _path_between(center, cell):
            mask[r, c] = True

    interior_area = max(1, (h - 2) * (w - 2))
    target = random.randint(max(len(support) + 6, interior_area // 3), max(len(support) + 8, int(interior_area * 0.62)))
    frontier = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 1 <= nr <= h - 2 and 1 <= nc <= w - 2 and not mask[nr, nc]:
                frontier.add((nr, nc))

    while int(mask.sum()) < target and frontier:
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        for rr, cc in _neighbors4(nr, nc):
            if 1 <= rr <= h - 2 and 1 <= cc <= w - 2 and not mask[rr, cc]:
                frontier.add((rr, cc))

    if not _is_connected(mask):
        return None
    if not _has_no_holes(mask):
        return None

    for r in range(h):
        if mask[r, 0] or mask[r, w - 1]:
            return None
    for c in range(w):
        if mask[0, c] or mask[h - 1, c]:
            return None

    return mask


def _render_input(body_mask: np.ndarray, plan: dict[str, list[int]]) -> np.ndarray:
    h, w = body_mask.shape
    grid = np.zeros((h, w), dtype=int)
    grid[body_mask] = BODY

    for r in plan["right"]:
        grid[r, w - 1] = LEAF
    for r in plan["left"]:
        grid[r, 0] = LEAF
    for c in plan["bottom"]:
        grid[h - 1, c] = LEAF
    for c in plan["top"]:
        grid[0, c] = LEAF
    return grid


def _detect_input_leaves(grid: np.ndarray) -> dict[str, list[int]]:
    h, w = grid.shape
    out = {"right": [], "left": [], "bottom": [], "top": []}
    for r in range(h):
        if grid[r, 0] == LEAF:
            out["left"].append(r)
        if grid[r, w - 1] == LEAF:
            out["right"].append(r)
    for c in range(w):
        if grid[0, c] == LEAF:
            out["top"].append(c)
        if grid[h - 1, c] == LEAF:
            out["bottom"].append(c)
    return out


def _validate_input(grid: np.ndarray) -> tuple[dict[str, list[int]], np.ndarray, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 6 <= h <= 10 and 8 <= w <= 13
    assert max(h, w) <= 30
    assert set(np.unique(g)).issubset({BG, BODY, LEAF})
    assert np.any(g == BODY)

    body = g == BODY
    assert _is_connected(body), "blue body must be a single 4-connected component"
    assert _has_no_holes(body), "blue body must be hole-free"
    assert not np.any(body[0, :]) and not np.any(body[h - 1, :])
    assert not np.any(body[:, 0]) and not np.any(body[:, w - 1])

    leaves = _detect_input_leaves(g)
    total_leaves = sum(len(v) for v in leaves.values())
    assert 2 <= total_leaves <= 4
    assert len(leaves["right"]) <= 2 and len(leaves["left"]) <= 2
    assert len(leaves["bottom"]) <= 2 and len(leaves["top"]) <= 2

    for r in leaves["left"]:
        assert g[r, 1] == BODY
        assert sum(g[nr, nc] == BODY for nr, nc in _neighbors4(r, 0) if _in_bounds(g.shape, nr, nc)) == 1
    for r in leaves["right"]:
        assert g[r, w - 2] == BODY
        assert sum(g[nr, nc] == BODY for nr, nc in _neighbors4(r, w - 1) if _in_bounds(g.shape, nr, nc)) == 1
    for c in leaves["top"]:
        assert g[1, c] == BODY
        assert sum(g[nr, nc] == BODY for nr, nc in _neighbors4(0, c) if _in_bounds(g.shape, nr, nc)) == 1
    for c in leaves["bottom"]:
        assert g[h - 2, c] == BODY
        assert sum(g[nr, nc] == BODY for nr, nc in _neighbors4(h - 1, c) if _in_bounds(g.shape, nr, nc)) == 1

    assert set(leaves["left"]).isdisjoint(leaves["right"])
    assert set(leaves["top"]).isdisjoint(leaves["bottom"])

    tiled = _mirror_tile(g)
    rows, cols = _seam_cuts_from_tiled(tiled, h, w)

    expected_rows = sorted(leaves["right"] + [2 * h - 1 - r for r in leaves["right"]])
    expected_cols = sorted(leaves["bottom"] + [2 * w - 1 - c for c in leaves["bottom"]])
    assert rows == expected_rows, (rows, expected_rows)
    assert cols == expected_cols, (cols, expected_cols)
    assert rows or cols, "at least one seam pair must exist"

    row_origins = set(leaves["right"])
    col_origins = set(leaves["bottom"])
    assert set(leaves["left"]).isdisjoint(row_origins)
    assert set(leaves["top"]).isdisjoint(col_origins)

    return leaves, tiled, rows, cols


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    h, w = inp.shape
    leaves, tiled, rows, cols = _validate_input(inp)

    assert out.shape == (2 * h, 2 * w)
    plain_tiling = _mirror_tile(inp)
    assert np.array_equal(tiled, plain_tiling)
    assert np.array_equal(out, main(inp))
    assert not np.array_equal(out, plain_tiling), "output must differ from plain tiling"

    expected = plain_tiling.copy()
    for r in rows:
        expected[r, :] = BG
    for c in cols:
        expected[:, c] = BG
    assert np.array_equal(out, expected)

    for r in rows:
        assert np.all(out[r, :] == BG)
    for c in cols:
        assert np.all(out[:, c] == BG)

    surviving_left = [(r, 0) for r in leaves["left"]] + [(2 * h - 1 - r, 0) for r in leaves["left"]]
    surviving_right = [(r, 2 * w - 1) for r in leaves["left"]] + [(2 * h - 1 - r, 2 * w - 1) for r in leaves["left"]]
    surviving_top = [(0, c) for c in leaves["top"]] + [(0, 2 * w - 1 - c) for c in leaves["top"]]
    surviving_bottom = [(2 * h - 1, c) for c in leaves["top"]] + [(2 * h - 1, 2 * w - 1 - c) for c in leaves["top"]]
    for r, c in surviving_left + surviving_right + surviving_top + surviving_bottom:
        assert out[r, c] == LEAF


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    if 2 * h > 30 or 2 * w > 30:
        return None

    plan = _leaf_plan(h, w)
    if plan is None:
        return None

    body = _grow_body(h, w, plan)
    if body is None:
        return None

    grid = _render_input(body, plan)
    try:
        out = main(grid)
        _validate_output(grid, out)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
