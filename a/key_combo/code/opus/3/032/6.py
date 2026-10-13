from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, neighbors4


BLACK = Color.BLACK
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
PINK = Color.PINK
TEAL = Color.TEAL
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _full_black_lines(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLACK)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLACK)]


def _compress_grid(grid: np.ndarray) -> tuple[np.ndarray, int]:
    row_blanks = _full_black_lines(grid, axis=0)
    col_blanks = _full_black_lines(grid, axis=1)
    row_internal = [r for r in row_blanks if 0 < r < grid.shape[0] - 1]
    col_internal = [c for c in col_blanks if 0 < c < grid.shape[1] - 1]

    if bool(row_internal) != bool(col_internal):
        separator_axis = 0 if row_internal else 1
    elif bool(row_blanks) != bool(col_blanks):
        separator_axis = 0 if row_blanks else 1
    else:
        raise AssertionError("expected separator lines on exactly one axis")

    if separator_axis == 0:
        keep_rows = [r for r in range(grid.shape[0]) if r not in set(row_blanks)]
        compressed = grid[keep_rows, :]
    else:
        keep_cols = [c for c in range(grid.shape[1]) if c not in set(col_blanks)]
        compressed = grid[:, keep_cols]

    box = bbox(compressed != BLACK)
    assert box is not None, "compressed scene must contain foreground"
    return crop_bbox(compressed, box), separator_axis


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    nxts = sorted((nr, nc) for nr, nc in neighbors4(*start) if (nr, nc) in pts)
    assert len(nxts) == 2

    path = [start]
    prev = start
    cur = nxts[0]
    while cur != start:
        path.append(cur)
        next_cells = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if (nr, nc) in pts and (nr, nc) != prev
        ]
        assert len(next_cells) == 1, (cur, next_cells)
        prev, cur = cur, next_cells[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _classify_cycle(path: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    turn_signs: dict[tuple[int, int], int] = {}
    counts: Counter[int] = Counter()
    labels = {cell: "straight" for cell in path}
    n = len(path)

    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            continue
        sign = 1 if cross > 0 else -1
        turn_signs[(r, c)] = sign
        counts[sign] += 1

    assert turn_signs
    assert abs(counts[1] - counts[-1]) == 4, counts
    convex_sign = 1 if counts[1] > counts[-1] else -1
    for cell, sign in turn_signs.items():
        labels[cell] = "convex" if sign == convex_sign else "concave"
    return labels


def _block_distance(i: int, start: int, end: int) -> int:
    if start <= i < end:
        return 0
    if i < start:
        return (start - i + 1) // 2
    return (i - end + 2) // 2


def _marker_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == PINK)
    assert coords.shape[0] == 4, "expected one 2x2 pink marker"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (2, 2)
    assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == PINK)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Compress away one-axis black separator lines, then recolor the cyan loop by bend class with a higher-priority hourglass mask from the pink marker.

    Input structure:
    - The grid has black background, one cyan one-pixel-wide orthogonal cycle, and one solid 2x2 pink marker.
    - Some complete black separator lines span the whole grid on exactly one axis; removing them yields the true geometry.
    - The cyan loop is 4-connected, closed, and every loop pixel has degree two.

    Rule:
    - Detect the separator-line axis, delete those full black rows or columns, and crop to the remaining non-black content.
    - Trace the cyan cycle on the compressed grid and classify each loop pixel as straight, convex bend, or concave bend by the majority turn sign.
    - On the same compressed grid, measure each loop pixel's block-distance from the 2x2 pink marker; the separator axis is the hourglass narrow axis and the other axis is the opening axis.
    - If a loop pixel has nonzero distance on both axes and narrow-distance <= opening-distance, recolor it yellow regardless of bend type.
    - Otherwise convex bends become red, concave bends become green, straight pixels stay cyan, and the pink marker remains unchanged.

    Key insight:
    - The output needs two independent analyses on the compressed coordinates: global loop-turn structure decides convex versus concave, but the marker-centered hourglass mask has higher priority and can override either bend color on the same pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    compressed, separator_axis = _compress_grid(g)
    marker_box = _marker_bbox(compressed)

    fg_colors = sorted(set(map(int, compressed.reshape(-1))) - {BLACK, PINK})
    assert fg_colors == [TEAL], fg_colors

    path = _trace_cycle(np.argwhere(compressed == TEAL))
    labels = _classify_cycle(path)
    mr0, mc0, mr1, mc1 = marker_box
    mrow_end = mr1 + 1
    mcol_end = mc1 + 1

    out = compressed.copy()
    for r, c in path:
        row_dist = _block_distance(r, mr0, mrow_end)
        col_dist = _block_distance(c, mc0, mcol_end)
        narrow_dist = row_dist if separator_axis == 0 else col_dist
        opening_dist = col_dist if separator_axis == 0 else row_dist
        if opening_dist > 0 and narrow_dist > 0 and narrow_dist <= opening_dist:
            out[r, c] = YELLOW
        elif labels[(r, c)] == "convex":
            out[r, c] = RED
        elif labels[(r, c)] == "concave":
            out[r, c] = GREEN
        else:
            out[r, c] = TEAL

    return out


LOOP_H_RANGE = (8, 12)
LOOP_W_RANGE = (10, 16)
NOTCH_COUNT_CHOICES = [2, 2, 3, 3, 4]
NOTCH_WIDTH_RANGE = (2, 4)
EDGE_MARGIN = 2
SAME_SIDE_GAP = 2
MAX_NOTCH_DEPTH = 3
GEN_RETRIES = 500
LOOP_ATTEMPTS = 120
SEPARATOR_GAP_RANGE = (1, 4)
SEPARATOR_MARGIN_RANGE = (0, 3)


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    _append_segment(path, top[0], 0, top[0], 0)
    for c in range(width - 1):
        _append_segment(path, top[c], c, top[c], c + 1)
        _append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _append_segment(path, bottom[c], c, bottom[c], c - 1)
        _append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _sample_intervals(width: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    candidates: list[tuple[int, int]] = []
    for span in range(NOTCH_WIDTH_RANGE[0], min(NOTCH_WIDTH_RANGE[1], width - 2 * EDGE_MARGIN) + 1):
        hi = width - EDGE_MARGIN - span
        for start in range(EDGE_MARGIN, hi + 1):
            candidates.append((start, start + span - 1))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for start, end in candidates:
        if any(not (end + SAME_SIDE_GAP < s or start > e + SAME_SIDE_GAP) for s, e in chosen):
            continue
        chosen.append((start, end))
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _apply_notch(top: list[int], bottom: list[int], side: str, start: int, end: int) -> bool:
    cols = range(start, end + 1)
    if side == "top":
        max_depth = min(MAX_NOTCH_DEPTH, min(bottom[c] - 2 for c in cols))
        if max_depth < 1:
            return False
        depth = random.randint(1, max_depth)
        for c in cols:
            top[c] = depth
        return True

    max_raise = min(MAX_NOTCH_DEPTH, min(bottom[c] - top[c] - 2 for c in cols))
    if max_raise < 1:
        return False
    raise_amt = random.randint(1, max_raise)
    for c in cols:
        bottom[c] -= raise_amt
    return True


def _build_loop_patch(top: list[int], bottom: list[int]) -> np.ndarray | None:
    if any(t + 2 > b for t, b in zip(top, bottom)):
        return None
    path = _path_from_profiles(top, bottom)
    if len(set(path)) != len(path):
        return None

    coords = np.asarray(path, dtype=int)
    patch = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=int)
    patch[coords[:, 0], coords[:, 1]] = TEAL

    try:
        labels = _classify_cycle(_trace_cycle(np.argwhere(patch == TEAL)))
    except AssertionError:
        return None

    concave = sum(label == "concave" for label in labels.values())
    if concave < 4:
        return None
    return patch


def _random_loop_patch() -> np.ndarray:
    for _ in range(LOOP_ATTEMPTS):
        height = random.randint(*LOOP_H_RANGE)
        width = random.randint(*LOOP_W_RANGE)
        notch_count = random.choice(NOTCH_COUNT_CHOICES)
        top_count = random.randint(0, notch_count)
        bottom_count = notch_count - top_count
        top_intervals = _sample_intervals(width, top_count)
        bottom_intervals = _sample_intervals(width, bottom_count)
        if top_intervals is None or bottom_intervals is None:
            continue

        top = [0] * width
        bottom = [height - 1] * width
        plan = [("top", s, e) for s, e in top_intervals] + [("bottom", s, e) for s, e in bottom_intervals]
        random.shuffle(plan)

        ok = True
        for side, start, end in plan:
            if not _apply_notch(top, bottom, side, start, end):
                ok = False
                break
        if not ok:
            continue

        patch = _build_loop_patch(top, bottom)
        if patch is None:
            continue
        patch = np.rot90(patch, random.randrange(4)).copy()
        if random.random() < 0.5:
            patch = np.fliplr(patch).copy()
        if random.random() < 0.5:
            patch = np.flipud(patch).copy()
        return patch
    raise RuntimeError("failed to generate loop patch")


def _interior_mask(loop_patch: np.ndarray) -> np.ndarray:
    loop_mask = loop_patch == TEAL
    padded = np.pad(loop_mask, 1, constant_values=False)
    exterior = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    exterior[0, 0] = True
    h, w = padded.shape

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if exterior[nr, nc] or padded[nr, nc]:
                continue
            exterior[nr, nc] = True
            q.append((nr, nc))

    return (~padded & ~exterior)[1:-1, 1:-1]


def _inside_marker_candidates(loop_patch: np.ndarray) -> list[tuple[int, int]]:
    interior = _interior_mask(loop_patch)
    loop_mask = loop_patch == TEAL
    h, w = loop_patch.shape
    out: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if not np.all(interior[r : r + 2, c : c + 2]):
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 3)
            c0 = max(0, c - 1)
            c1 = min(w, c + 3)
            if np.any(loop_mask[r0:r1, c0:c1]):
                continue
            out.append((r, c))
    return out


def _build_inside_scene(loop_patch: np.ndarray) -> np.ndarray | None:
    candidates = _inside_marker_candidates(loop_patch)
    if not candidates:
        return None
    r, c = random.choice(candidates)
    comp = loop_patch.copy()
    comp[r : r + 2, c : c + 2] = PINK
    return comp


def _outside_positions(limit: int) -> list[int]:
    if limit <= 2:
        return [0]
    lo = 1 if limit >= 5 else 0
    hi = limit - 3 if limit >= 5 else limit - 2
    return list(range(lo, hi + 1))


def _build_outside_scene(loop_patch: np.ndarray, separator_axis: int) -> tuple[np.ndarray, dict]:
    h, w = loop_patch.shape
    if separator_axis == 0:
        side = random.choice(["top", "bottom"])
        comp = np.zeros((h + 2, w), dtype=int)
        cols = _outside_positions(w)
        c = random.choice(cols)
        if side == "top":
            comp[2:, :] = loop_patch
            comp[0:2, c : c + 2] = PINK
            return comp, {"between_after": 1}
        comp[:h, :] = loop_patch
        comp[h : h + 2, c : c + 2] = PINK
        return comp, {"between_after": h - 1}

    side = random.choice(["left", "right"])
    comp = np.zeros((h, w + 2), dtype=int)
    rows = _outside_positions(h)
    r = random.choice(rows)
    if side == "left":
        comp[:, 2:] = loop_patch
        comp[r : r + 2, 0:2] = PINK
        return comp, {"between_after": 1}
    comp[:, :w] = loop_patch
    comp[r : r + 2, w : w + 2] = PINK
    return comp, {"between_after": w - 1}


def _expand_with_separators(comp: np.ndarray, separator_axis: int, meta: dict, inside_branch: bool) -> np.ndarray:
    before = random.randint(*SEPARATOR_MARGIN_RANGE)
    after = random.randint(*SEPARATOR_MARGIN_RANGE)
    between = 0 if inside_branch else random.randint(*SEPARATOR_GAP_RANGE)
    if before == 0 and after == 0 and between == 0:
        before = 1

    if separator_axis == 0:
        rows: list[np.ndarray] = [np.zeros(comp.shape[1], dtype=int) for _ in range(before)]
        split = int(meta.get("between_after", -1))
        for r in range(comp.shape[0]):
            rows.append(comp[r, :].copy())
            if r == split:
                rows.extend(np.zeros(comp.shape[1], dtype=int) for _ in range(between))
        rows.extend(np.zeros(comp.shape[1], dtype=int) for _ in range(after))
        raw = np.stack(rows, axis=0)
        assert not any(np.all(raw[:, c] == BLACK) for c in range(raw.shape[1]))
        return raw

    cols: list[np.ndarray] = [np.zeros(comp.shape[0], dtype=int) for _ in range(before)]
    split = int(meta.get("between_after", -1))
    for c in range(comp.shape[1]):
        cols.append(comp[:, c].copy())
        if c == split:
            cols.extend(np.zeros(comp.shape[0], dtype=int) for _ in range(between))
    cols.extend(np.zeros(comp.shape[0], dtype=int) for _ in range(after))
    raw = np.stack(cols, axis=1)
    assert not any(np.all(raw[r, :] == BLACK) for r in range(raw.shape[0]))
    return raw


def _marker_clear(raw: np.ndarray) -> bool:
    marker = np.argwhere(raw == PINK)
    loop = raw == TEAL
    for r, c in marker:
        for dr, dc in DIRS8:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < raw.shape[0] and 0 <= nc < raw.shape[1] and loop[nr, nc]:
                return False
    return True


def _output_has_all_loop_classes(out: np.ndarray) -> bool:
    colors = set(map(int, out.reshape(-1)))
    return {PINK, TEAL, RED, GREEN, YELLOW}.issubset(colors)


def _gen_attempt() -> np.ndarray | None:
    separator_axis = random.randint(0, 1)
    want_inside = random.random() < 0.45

    loop_patch = _random_loop_patch()
    comp = _build_inside_scene(loop_patch) if want_inside else None
    inside_branch = comp is not None
    if comp is None:
        comp, meta = _build_outside_scene(loop_patch, separator_axis)
    else:
        meta = {}

    raw = _expand_with_separators(comp, separator_axis, meta, inside_branch)
    if max(raw.shape) > 30:
        return None
    if not _marker_clear(raw):
        return None

    try:
        out = main(raw)
    except Exception:
        return None
    if max(out.shape) > 30:
        return None
    if not _output_has_all_loop_classes(out):
        return None
    if out.shape == raw.shape:
        return None
    if separator_axis == 0 and out.shape[0] >= raw.shape[0]:
        return None
    if separator_axis == 1 and out.shape[1] >= raw.shape[1]:
        return None
    return raw


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
