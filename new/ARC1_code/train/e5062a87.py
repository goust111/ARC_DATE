from __future__ import annotations

import numpy as np

from arc_utils import bbox, is_contiguous
import itertools
import random


BG = 0
RED = 2
GRAY = 5


def _offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(int(r), int(c)) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the red sprite into every matching empty slot in the gray-and-black pattern.

    Input structure
    - The grid uses black empty cells, gray support cells, and one red marker sprite.
    - The red sprite bbox defines the stamp mask.
    - Gray cells inside the sprite bbox define the support signature; if an enclosed interior gray support exists, it is the anchor signature.

    Rule
    - Scan candidate bbox positions from top-left to bottom-right.
    - A candidate matches when every red-mask position is still black and every support-signature position is gray.
    - If the sprite bbox has no gray support cells, only border-touching black slots of the same shape match.
    - Paint each matched red-mask position red, preserving all other cells.

    Key insight
    The red cells are a stamp; gray support inside the stamp window, or the grid border for unsupported solid stamps, identifies the blank copies to complete.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(np.unique(g)).issubset({BG, RED, GRAY}), "expected only black, red, and gray"

    red_box = bbox(g == RED)
    assert red_box is not None, "expected a red sprite"
    r0, c0, r1, c1 = map(int, red_box)
    sprite = g[r0 : r1 + 1, c0 : c1 + 1]
    bh, bw = sprite.shape
    assert bh > 0 and bw > 0

    stamp = _offsets(sprite == RED)
    assert stamp, "expected at least one red stamp cell"

    gray_offsets = _offsets(sprite == GRAY)
    interior_gray = [
        (r, c)
        for r, c in gray_offsets
        if 0 < r < bh - 1 and 0 < c < bw - 1
    ]
    support = interior_gray or gray_offsets
    if not support:
        assert np.all(sprite == RED), "unsupported stamp should fill its bbox"

    h, w = g.shape
    out = g.copy()
    painted = 0
    for ar in range(h - bh + 1):
        for ac in range(w - bw + 1):
            cells = [(ar + dr, ac + dc) for dr, dc in stamp]
            if not all(out[r, c] == BG for r, c in cells):
                continue

            if support:
                if not all(g[ar + dr, ac + dc] == GRAY for dr, dc in support):
                    continue
            else:
                touches_border = ar == 0 or ac == 0 or ar + bh == h or ac + bw == w
                if not touches_border:
                    continue

            for r, c in cells:
                out[r, c] = RED
                painted += 1

    assert painted > 0, "expected at least one matching empty slot"
    return out

# --------------- generator (o5x) ---------------

# Palette and fixed canvas.
_GEN_BG, _GEN_RED, _GEN_GRAY = 0, 2, 5
_GEN_H, _GEN_W = 10, 10

# Sprite shape and support selection.
_GEN_SUPPORTED_PROB = 0.7
_GEN_SUPPORTED_DIMS = ((2, 3), (2, 4), (3, 2), (3, 3))
_GEN_UNSUPPORTED_DIMS = ((1, 4), (2, 2), (2, 3), (3, 2))
_GEN_MIN_RED, _GEN_MAX_RED = 4, 6
_GEN_MAX_SUPPORT = 3
_GEN_INTERIOR_SUPPORT_PROB = 0.45
_GEN_CONNECTIVITY = 4

# Placement count and gray density targets.
_GEN_MIN_PLACES, _GEN_MAX_PLACES = 2, 5
_GEN_MIN_GRAY, _GEN_MAX_GRAY = 43, 61

# Rejection-sampling limits.
_GEN_BLOCK_EXTRA_TRIES, _GEN_ATTEMPTS = 120, 300

_GEN_MASK_CACHE: dict[tuple[int, int, int], list[np.ndarray]] = {}


def _gen_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _gen_mask_options(h: int, w: int, n: int) -> list[np.ndarray]:
    key = (h, w, n)
    if key in _GEN_MASK_CACHE:
        return _GEN_MASK_CACHE[key]

    opts = []
    for cells in itertools.combinations(range(h * w), n):
        mask = np.zeros((h, w), dtype=bool)
        rr, cc = np.unravel_index(cells, (h, w))
        mask[rr, cc] = True
        rs, cs = np.where(mask)
        if rs.min() != 0 or rs.max() != h - 1 or cs.min() != 0 or cs.max() != w - 1:
            continue
        if is_contiguous(mask.astype(int), background=0, connectivity=_GEN_CONNECTIVITY):
            opts.append(mask)
    _GEN_MASK_CACHE[key] = opts
    return opts


def _gen_touches_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r + h == _GEN_H or c + w == _GEN_W


def _gen_windows_overlap(a: tuple[int, int], b: tuple[int, int], h: int, w: int) -> bool:
    ar, ac = a
    br, bc = b
    return not (ar + h <= br or br + h <= ar or ac + w <= bc or bc + w <= ac)


def _gen_pick_origins(h: int, w: int, n: int, border_only: bool) -> list[tuple[int, int]] | None:
    candidates = []
    for r in range(_GEN_H - h + 1):
        for c in range(_GEN_W - w + 1):
            if border_only and not _gen_touches_border(r, c, h, w):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)

    picked: list[tuple[int, int]] = []
    for pos in candidates:
        if all(not _gen_windows_overlap(pos, old, h, w) for old in picked):
            picked.append(pos)
            if len(picked) == n:
                return picked
    return None


def _gen_matching_origins(
    grid: np.ndarray,
    h: int,
    w: int,
    stamp: list[tuple[int, int]],
    support: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    matches = []
    for r in range(_GEN_H - h + 1):
        for c in range(_GEN_W - w + 1):
            if any(grid[r + dr, c + dc] != _GEN_BG for dr, dc in stamp):
                continue
            if support:
                if any(grid[r + dr, c + dc] != _GEN_GRAY for dr, dc in support):
                    continue
            elif not _gen_touches_border(r, c, h, w):
                continue
            matches.append((r, c))
    return matches


def _gen_block_extras(
    inp: np.ndarray,
    out: np.ndarray,
    h: int,
    w: int,
    stamp: list[tuple[int, int]],
    support: list[tuple[int, int]],
    targets: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> bool:
    for _ in range(_GEN_BLOCK_EXTRA_TRIES):
        matches = _gen_matching_origins(inp, h, w, stamp, support)
        extras = [pos for pos in matches if pos not in targets]
        if not extras:
            return True

        r, c = random.choice(extras)
        stamp_cells = [(r + dr, c + dc) for dr, dc in stamp]
        random.shuffle(stamp_cells)
        for rr, cc in stamp_cells:
            if (rr, cc) not in protected and inp[rr, cc] == _GEN_BG:
                inp[rr, cc] = _GEN_GRAY
                out[rr, cc] = _GEN_GRAY
                break
        else:
            support_cells = [(r + dr, c + dc) for dr, dc in support]
            random.shuffle(support_cells)
            for rr, cc in support_cells:
                if (rr, cc) not in protected and inp[rr, cc] == _GEN_GRAY:
                    inp[rr, cc] = _GEN_BG
                    out[rr, cc] = _GEN_BG
                    break
            else:
                return False
    return False


def _gen_support_mask(h: int, w: int, red_mask: np.ndarray) -> np.ndarray:
    open_cells = [(r, c) for r in range(h) for c in range(w) if not red_mask[r, c]]
    interior = [(r, c) for r, c in open_cells if 0 < r < h - 1 and 0 < c < w - 1]
    support_mask = np.zeros((h, w), dtype=bool)
    if interior and random.random() < _GEN_INTERIOR_SUPPORT_PROB:
        support_mask[random.choice(interior)] = True
    else:
        support_n = random.randint(1, min(_GEN_MAX_SUPPORT, len(open_cells)))
        for r, c in random.sample(open_cells, support_n):
            support_mask[r, c] = True
    return support_mask


def _gen_make_sprite() -> tuple[int, int, np.ndarray, np.ndarray] | None:
    if random.random() < _GEN_SUPPORTED_PROB:
        h, w = random.choice(_GEN_SUPPORTED_DIMS)
        max_red = min(_GEN_MAX_RED, h * w - 1)
        red_n = random.randint(_GEN_MIN_RED, max_red)
        masks = _gen_mask_options(h, w, red_n)
        if not masks:
            return None
        red_mask = random.choice(masks).copy()
        return h, w, red_mask, _gen_support_mask(h, w, red_mask)

    h, w = random.choice(_GEN_UNSUPPORTED_DIMS)
    return h, w, np.ones((h, w), dtype=bool), np.zeros((h, w), dtype=bool)


def _gen_collect_cells(
    origins: list[tuple[int, int]],
    h: int,
    w: int,
    stamp: list[tuple[int, int]],
    support: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    protected: set[tuple[int, int]] = set()
    stamp_cells: set[tuple[int, int]] = set()
    support_cells: set[tuple[int, int]] = set()
    for r0, c0 in origins:
        for dr in range(h):
            for dc in range(w):
                protected.add((r0 + dr, c0 + dc))
        for dr, dc in stamp:
            stamp_cells.add((r0 + dr, c0 + dc))
        for dr, dc in support:
            support_cells.add((r0 + dr, c0 + dc))
    return protected, stamp_cells, support_cells


def _gen_render_output(
    protected: set[tuple[int, int]],
    stamp_cells: set[tuple[int, int]],
    support_cells: set[tuple[int, int]],
    gray_target: int,
) -> np.ndarray:
    out = np.zeros((_GEN_H, _GEN_W), dtype=int)
    for r, c in support_cells:
        out[r, c] = _GEN_GRAY
    for r, c in stamp_cells:
        out[r, c] = _GEN_RED

    filler = [
        (r, c)
        for r in range(_GEN_H)
        for c in range(_GEN_W)
        if (r, c) not in protected
    ]
    random.shuffle(filler)
    for r, c in filler[: max(0, gray_target - len(support_cells))]:
        out[r, c] = _GEN_GRAY
    return out


def _gen_attempt() -> np.ndarray | None:
    made = _gen_make_sprite()
    if made is None:
        return None
    h, w, red_mask, support_mask = made
    stamp = _gen_offsets(red_mask)
    support = _gen_offsets(support_mask)

    n_places = random.randint(_GEN_MIN_PLACES, _GEN_MAX_PLACES)
    origins = _gen_pick_origins(h, w, n_places, border_only=not support)
    if origins is None:
        return None
    original = random.choice(origins)
    targets = set(origins) - {original}

    protected, stamp_cells, support_cells = _gen_collect_cells(origins, h, w, stamp, support)
    max_gray = _GEN_H * _GEN_W - len(protected) + len(support_cells)
    if max_gray < _GEN_MIN_GRAY:
        return None
    gray_target = random.randint(_GEN_MIN_GRAY, min(_GEN_MAX_GRAY, max_gray))

    out = _gen_render_output(protected, stamp_cells, support_cells, gray_target)
    inp = out.copy()
    for r0, c0 in targets:
        for dr, dc in stamp:
            inp[r0 + dr, c0 + dc] = _GEN_BG

    if not _gen_block_extras(inp, out, h, w, stamp, support, targets, protected):
        return None
    gray_count = int(np.count_nonzero(out == _GEN_GRAY))
    if not (_GEN_MIN_GRAY <= gray_count <= _GEN_MAX_GRAY):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e5062a87 input")
