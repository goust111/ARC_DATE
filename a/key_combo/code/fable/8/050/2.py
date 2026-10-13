from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
YELLOW = 4
HEAD = 5
TAIL = 9
FG_PALETTE = [1, 2, 3, 6, 7, 8]
ANCHOR_VECTORS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

PANEL_HEIGHT_RANGE = (7, 12)
PANEL_WIDTH_RANGE = (7, 12)
SHAPE_BBOX_HEIGHTS = (2, 3, 4, 5)
SHAPE_BBOX_WIDTHS = (2, 3, 4, 5)
SHAPE_AREA_RANGE = (4, 8)

MAX_MASK_ATTEMPTS = 60
MAX_SHAPE_ATTEMPTS = 120
MAX_PANEL_ATTEMPTS = 300
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALLS = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nxt in _neighbors4(*cell):
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def _rotate_rel(coords: np.ndarray, k: int) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    kk = int(k) % 4
    if kk == 0:
        return pts.copy()
    if kk == 1:
        return np.stack([-pts[:, 1], pts[:, 0]], axis=1)
    if kk == 2:
        return -pts
    return np.stack([pts[:, 1], -pts[:, 0]], axis=1)


def _overlay(panel_shape: tuple[int, int], upper_coords: np.ndarray, upper_color: int,
             lower_coords: np.ndarray, lower_color: int, bg: int = BACKGROUND) -> np.ndarray:
    out = np.full(panel_shape, bg, dtype=int)
    for r, c in upper_coords:
        out[int(r), int(c)] = int(upper_color)
    for r, c in lower_coords:
        rr = int(r)
        cc = int(c)
        if out[rr, cc] == bg:
            out[rr, cc] = int(lower_color)
        elif out[rr, cc] == int(upper_color):
            out[rr, cc] = YELLOW
    return out


def _find_separator_row(grid: np.ndarray) -> int:
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    color_counts = {int(c): int(n) for c, n in zip(colors, counts)}
    rows = []
    for r in range(h):
        row_color = int(grid[r, 0])
        if np.all(grid[r] == row_color) and color_counts[row_color] == w:
            rows.append(r)
    assert len(rows) == 1, "expected exactly one full-width separator row"
    return int(rows[0])


def _parse_panel(panel: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    head_coords = np.argwhere(panel == HEAD)
    tail_coords = np.argwhere(panel == TAIL)
    assert head_coords.shape == (1, 2), "expected one gray anchor head"
    assert tail_coords.shape == (1, 2), "expected one maroon anchor tail"
    head = head_coords[0].astype(int)
    tail = tail_coords[0].astype(int)
    assert int(np.abs(head - tail).sum()) == 1, "anchor segment must be length two"

    fg_colors = sorted(set(map(int, np.unique(panel))) - {bg, HEAD, TAIL})
    assert len(fg_colors) == 1, "each panel must contain exactly one foreground color"
    fg_color = int(fg_colors[0])
    fg_coords = np.argwhere(panel == fg_color).astype(int)
    assert fg_coords.shape[0] >= 1 and _is_connected(fg_coords), "foreground must be one 4-connected shape"
    return head, tail, fg_coords, fg_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Register the lower panel to the upper anchor segment, then transparently overlay the two shapes.

    Input structure:
    - A single full-width separator row splits the grid into equal-size upper and lower panels.
    - Each panel contains one monochrome foreground shape plus a directed two-cell anchor segment:
      gray is the head and maroon is the tail.
    - The lower panel's anchor-and-shape rigid body may appear in any 90-degree orientation.

    Rule:
    - Keep the upper panel fixed.
    - Rotate the lower panel by a multiple of 90 degrees and translate it so the lower anchor head
      lands on the upper head and the lower tail lands on the upper tail.
    - Remove both anchors and merge the two foreground shapes on the upper panel's canvas:
      upper-only cells keep the upper color, lower-only cells keep the lower color, and overlaps become yellow.

    Key insight:
    - The gray-to-maroon anchor is a directed baseline, so head-vs-tail order makes the rotation unique;
      only after restoring the lower rigid body's pose can the transparent overlay be computed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)

    sep = _find_separator_row(g)
    assert 2 * sep + 1 == g.shape[0], "separator must split the grid into equal-height panels"
    upper = g[:sep].copy()
    lower = g[sep + 1 :].copy()
    assert upper.shape == lower.shape

    upper_head, upper_tail, upper_fg, upper_color = _parse_panel(upper, bg)
    lower_head, lower_tail, lower_fg, lower_color = _parse_panel(lower, bg)
    assert upper_color != lower_color, "panel foreground colors must differ"

    upper_vec = (upper_tail - upper_head).astype(int)
    lower_vec = (lower_tail - lower_head).astype(int)

    candidate_rotations = [k for k in range(4) if np.array_equal(_rotate_rel(lower_vec[None, :], k)[0], upper_vec)]
    assert len(candidate_rotations) == 1, "directed anchors must determine a unique rotation"
    k = candidate_rotations[0]

    lower_rel = lower_fg - lower_head
    transformed = _rotate_rel(lower_rel, k) + upper_head
    h, w = upper.shape
    assert np.all((0 <= transformed[:, 0]) & (transformed[:, 0] < h))
    assert np.all((0 <= transformed[:, 1]) & (transformed[:, 1] < w))

    return _overlay(upper.shape, upper_fg, upper_color, transformed, lower_color, bg=bg)


def _build_connected_mask(height: int, width: int, area: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(MAX_MASK_ATTEMPTS):
        cells = {(rng.randrange(height), rng.randrange(width))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        if _is_connected(coords):
            return coords
    return None


def _shape_is_asymmetric(coords: np.ndarray) -> bool:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = coords - np.array([r0, c0], dtype=int)
    norm = {tuple(map(int, rc)) for rc in rel}
    for k in (1, 2, 3):
        rot = _rotate_rel(rel, k)
        rr0 = int(rot[:, 0].min())
        cc0 = int(rot[:, 1].min())
        rot_norm = {tuple(map(int, rc)) for rc in rot - np.array([rr0, cc0], dtype=int)}
        if rot_norm == norm:
            return False
    return True


def _bbox_min_anchor_distance(r0: int, c0: int, bh: int, bw: int,
                              anchor_cells: set[tuple[int, int]]) -> int:
    best = 10 ** 9
    for rr in range(r0, r0 + bh):
        for cc in range(c0, c0 + bw):
            for ar, ac in anchor_cells:
                best = min(best, abs(rr - ar) + abs(cc - ac))
    return int(best)


def _gen_shape(panel_shape: tuple[int, int], head: tuple[int, int], tail: tuple[int, int],
               rng: random.Random) -> np.ndarray | None:
    H, W = panel_shape
    anchor_cells = {tuple(map(int, head)), tuple(map(int, tail))}
    boxes = []
    for bh in SHAPE_BBOX_HEIGHTS:
        for bw in SHAPE_BBOX_WIDTHS:
            if bh > H or bw > W:
                continue
            for r0 in range(H - bh + 1):
                for c0 in range(W - bw + 1):
                    inside_anchor = any(r0 <= ar < r0 + bh and c0 <= ac < c0 + bw for ar, ac in anchor_cells)
                    if inside_anchor:
                        continue
                    dist = _bbox_min_anchor_distance(r0, c0, bh, bw, anchor_cells)
                    if 1 <= dist <= 3:
                        boxes.append((r0, c0, bh, bw))
    rng.shuffle(boxes)

    for r0, c0, bh, bw in boxes[:MAX_SHAPE_ATTEMPTS]:
        lo = max(SHAPE_AREA_RANGE[0], (bh * bw + 1) // 2)
        hi = min(SHAPE_AREA_RANGE[1], bh * bw - 1 if bh * bw > 4 else bh * bw)
        if lo > hi:
            continue
        area = rng.randint(lo, hi)
        local = _build_connected_mask(bh, bw, area, rng)
        if local is None:
            continue
        coords = local + np.array([r0, c0], dtype=int)
        if any(tuple(map(int, rc)) in anchor_cells for rc in coords):
            continue
        dists = [min(abs(int(r) - ar) + abs(int(c) - ac) for ar, ac in anchor_cells) for r, c in coords]
        if min(dists) > 2:
            continue
        if not _shape_is_asymmetric(coords):
            continue
        return coords.astype(int)
    return None


def _feasible_heads(panel_shape: tuple[int, int], tail_rel: np.ndarray, shape_rel: np.ndarray,
                    forbidden: tuple[int, int] | None = None) -> list[np.ndarray]:
    H, W = panel_shape
    heads: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            head = np.array([r, c], dtype=int)
            tail = head + tail_rel
            coords = shape_rel + head
            if not (0 <= tail[0] < H and 0 <= tail[1] < W):
                continue
            if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= H) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= W):
                continue
            if np.any((coords[:, 0] == head[0]) & (coords[:, 1] == head[1])):
                continue
            if np.any((coords[:, 0] == tail[0]) & (coords[:, 1] == tail[1])):
                continue
            if forbidden is not None and abs(r - forbidden[0]) + abs(c - forbidden[1]) < 2:
                continue
            heads.append(head)
    return heads


def _coords_equal(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.array(sorted(map(tuple, np.asarray(a, dtype=int))), dtype=int)
    bb = np.array(sorted(map(tuple, np.asarray(b, dtype=int))), dtype=int)
    return aa.shape == bb.shape and np.array_equal(aa, bb)


def _validate_attempt(inp: np.ndarray, expected: np.ndarray, upper_head: np.ndarray, upper_tail: np.ndarray,
                      upper_fg: np.ndarray, lower_input_head: np.ndarray, lower_input_fg: np.ndarray,
                      aligned_lower_fg: np.ndarray, k: int, want_overlap: bool) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    overlap = {tuple(map(int, rc)) for rc in upper_fg} & {tuple(map(int, rc)) for rc in aligned_lower_fg}
    if want_overlap and len(overlap) == 0:
        return False
    if not want_overlap and overlap:
        return False
    if len({tuple(map(int, rc)) for rc in upper_fg} - overlap) == 0:
        return False
    if len({tuple(map(int, rc)) for rc in aligned_lower_fg} - overlap) == 0:
        return False

    if k != 0:
        shift_only = lower_input_fg + (upper_head - lower_input_head)
        if _coords_equal(shift_only, aligned_lower_fg):
            return False
        if np.all((0 <= shift_only[:, 0]) & (shift_only[:, 0] < expected.shape[0]) &
                  (0 <= shift_only[:, 1]) & (shift_only[:, 1] < expected.shape[1])):
            if _coords_equal(shift_only, aligned_lower_fg):
                return False

    lower_input_rel = lower_input_fg - lower_input_head
    swapped = _rotate_rel(lower_input_rel, (k + 2) % 4) + upper_tail
    if np.all((0 <= swapped[:, 0]) & (swapped[:, 0] < expected.shape[0]) &
              (0 <= swapped[:, 1]) & (swapped[:, 1] < expected.shape[1])):
        if _coords_equal(swapped, aligned_lower_fg):
            return False

    return True


def _gen_attempt(rng: random.Random, call_index: int) -> np.ndarray | None:
    H = rng.randint(*PANEL_HEIGHT_RANGE)
    W = rng.randint(*PANEL_WIDTH_RANGE)
    colors = rng.sample(FG_PALETTE, 3)
    upper_color, lower_color, separator = colors

    upper_vec = np.array(ANCHOR_VECTORS[rng.randrange(4)], dtype=int)
    head_candidates = []
    for r in range(2, H - 2):
        for c in range(2, W - 2):
            head = np.array([r, c], dtype=int)
            tail = head + upper_vec
            if 1 <= tail[0] < H - 1 and 1 <= tail[1] < W - 1:
                head_candidates.append(head)
    if not head_candidates:
        return None

    upper_head = rng.choice(head_candidates).astype(int)
    upper_tail = upper_head + upper_vec

    upper_fg = _gen_shape((H, W), tuple(upper_head), tuple(upper_tail), rng)
    if upper_fg is None:
        return None

    rotation_cycle = [1, 0, 2, 3]
    k = rotation_cycle[call_index % len(rotation_cycle)]
    want_overlap = (call_index % 3) != 1

    aligned_lower = None
    for _ in range(MAX_PANEL_ATTEMPTS):
        candidate = _gen_shape((H, W), tuple(upper_head), tuple(upper_tail), rng)
        if candidate is None:
            continue
        overlap = {tuple(map(int, rc)) for rc in upper_fg} & {tuple(map(int, rc)) for rc in candidate}
        if want_overlap and not overlap:
            continue
        if (not want_overlap) and overlap:
            continue
        if len({tuple(map(int, rc)) for rc in upper_fg} - overlap) < 2:
            continue
        if len({tuple(map(int, rc)) for rc in candidate} - overlap) < 2:
            continue
        aligned_lower = candidate.astype(int)
        break
    if aligned_lower is None:
        return None

    aligned_lower_rel = aligned_lower - upper_head
    inv_k = (-k) % 4
    lower_input_rel = _rotate_rel(aligned_lower_rel, inv_k)
    lower_tail_rel = _rotate_rel((upper_tail - upper_head)[None, :], inv_k)[0]

    feasible = _feasible_heads((H, W), lower_tail_rel, lower_input_rel, forbidden=tuple(map(int, upper_head)))
    if not feasible:
        return None
    lower_head = rng.choice(feasible).astype(int)
    lower_tail = lower_head + lower_tail_rel
    lower_fg = lower_input_rel + lower_head

    if k != 0 and _coords_equal(lower_fg + (upper_head - lower_head), aligned_lower):
        return None

    upper_panel = np.full((H, W), BACKGROUND, dtype=int)
    upper_panel[upper_fg[:, 0], upper_fg[:, 1]] = upper_color
    upper_panel[tuple(upper_head)] = HEAD
    upper_panel[tuple(upper_tail)] = TAIL

    lower_panel = np.full((H, W), BACKGROUND, dtype=int)
    lower_panel[lower_fg[:, 0], lower_fg[:, 1]] = lower_color
    lower_panel[tuple(lower_head)] = HEAD
    lower_panel[tuple(lower_tail)] = TAIL

    expected = _overlay((H, W), upper_fg, upper_color, aligned_lower, lower_color)
    inp = np.vstack([upper_panel, np.full((1, W), separator, dtype=int), lower_panel])

    if not _validate_attempt(
        inp,
        expected,
        upper_head,
        upper_tail,
        upper_fg,
        lower_head,
        lower_fg,
        aligned_lower,
        k,
        want_overlap,
    ):
        return None

    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    call_index = _GEN_CALLS
    _GEN_CALLS += 1
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng, call_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-registration overlay input")
