from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _trace_cycle(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in _neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        nxt = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(nxt) == 1
        prev, cur = cur, nxt[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


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


def _turn_sign_counts(path: list[tuple[int, int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross:
            counts[1 if cross > 0 else -1] += 1
    return counts


def _classify_loops(mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, GREEN, 0).astype(int)
    for comp in _components(mask):
        path = _trace_cycle(comp)
        counts = _turn_sign_counts(path)
        assert counts
        convex_sign = 1 if counts[1] > counts[-1] else -1
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
            out[r, c] = YELLOW if sign == convex_sign else RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two separated path panels, then recolor each emergent closed loop by bend type.

    Input structure:
    - The grid has black background and exactly one full grey separator row.
    - Above the separator, blue one-pixel-wide open orthogonal paths occupy an upper panel.
    - Below the separator, red one-pixel-wide open orthogonal paths occupy a same-sized lower panel.

    Rule:
    - Remove the grey separator and align the two panels cell-by-cell.
    - Any cell occupied in either panel becomes green, forming one or more closed orthogonal loops.
    - Trace each loop: straight cells stay green, bend cells whose turn direction is the loop's majority sign become yellow, and bend cells of the minority sign become red.

    Key insight:
    - The meaningful closed cycles do not exist in either panel alone; they only appear after the transparent overlay, so loop tracing must happen after merging.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(sep_rows) == 1
    sep = sep_rows[0]
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape and top.size > 0

    merged = (top != 0) | (bottom != 0)
    return _classify_loops(merged)


def _make_loop_patch(height: int, width: int, require_concave: bool) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(120):
        top = [0] * width
        bottom = [height - 1] * width

        style_choices = ["none", "top", "bottom", "both"]
        if require_concave:
            style_choices = ["top", "bottom", "both"]
        style = random.choice(style_choices)

        def add_notch(profile_top: list[int], profile_bottom: list[int], side: str, forbidden: tuple[int, int] | None):
            lo = 1
            hi = width - 3
            if hi < lo:
                return False
            starts = list(range(lo, hi + 1))
            random.shuffle(starts)
            for start in starts:
                end_max = width - 2
                end_min = start + 1
                if end_min > end_max:
                    continue
                ends = list(range(end_min, end_max + 1))
                random.shuffle(ends)
                for end in ends:
                    if forbidden is not None and not (end < forbidden[0] - 1 or start > forbidden[1] + 1):
                        continue
                    cols = range(start, end + 1)
                    if side == "top":
                        limit = min(profile_bottom[c] - 2 for c in cols)
                        if limit < 1:
                            continue
                        depth = random.randint(1, limit)
                        for c in cols:
                            profile_top[c] = depth
                    else:
                        limit = min((height - 1) - profile_top[c] - 2 for c in cols)
                        if limit < 1:
                            continue
                        depth = random.randint(1, limit)
                        for c in cols:
                            profile_bottom[c] = height - 1 - depth
                    return True, (start, end)
            return False, forbidden

        interval = None
        if style in ("top", "both"):
            ok, interval = add_notch(top, bottom, "top", interval)
            if not ok:
                continue
        if style in ("bottom", "both"):
            ok, interval = add_notch(top, bottom, "bottom", interval if style == "both" else None)
            if not ok:
                continue

        if any(top[c] + 2 > bottom[c] for c in range(width)):
            continue

        path = _path_from_profiles(top, bottom)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in path:
            mask[r, c] = True

        comps = _components(mask)
        if len(comps) != 1:
            continue
        try:
            traced = _trace_cycle(comps[0])
        except AssertionError:
            continue
        if len(traced) < 16:
            continue
        counts = _turn_sign_counts(traced)
        has_concave = counts[1] > 0 and counts[-1] > 0
        if require_concave and not has_concave:
            continue
        return mask, traced
    return None


def _paint_blocked(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    coords = np.argwhere(patch)
    for pr, pc in coords:
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        for dr, dc in DIRS8 + [(0, 0)]:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                blocked[nr, nc] = True


def _place_patch(panel_shape: tuple[int, int], blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = panel_shape
    ph, pw = patch.shape
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(candidates)
    coords = np.argwhere(patch)
    for r0, c0 in candidates:
        if all(not blocked[r0 + int(pr), c0 + int(pc)] for pr, pc in coords):
            return r0, c0
    return None


def _balanced_partition(total: int, parts: int) -> list[int]:
    base = total // parts
    rem = total % parts
    lengths = [base] * parts
    picks = list(range(parts))
    random.shuffle(picks)
    for idx in picks[:rem]:
        lengths[idx] += 1
    random.shuffle(lengths)
    return lengths


def _assign_open_segments(path: list[tuple[int, int]], top: np.ndarray, bottom: np.ndarray) -> None:
    n = len(path)
    candidates = [parts for parts in (8, 6, 4) if parts <= n and 4 <= n // parts <= 12]
    if not candidates:
        candidates = [parts for parts in (6, 4) if parts <= n and n // parts >= 3]
    parts = random.choice(candidates)
    lengths = _balanced_partition(n, parts)
    start = random.randrange(n)
    first_top = bool(random.getrandbits(1))

    idx = start
    for seg_idx, length in enumerate(lengths):
        use_top = first_top if seg_idx % 2 == 0 else not first_top
        for _ in range(length):
            r, c = path[idx % n]
            if use_top:
                top[r, c] = BLUE
            else:
                bottom[r, c] = RED
            idx += 1


def _open_paths_only(grid: np.ndarray, color: int) -> bool:
    mask = grid == color
    for comp in _components(mask):
        pts = set(comp)
        degrees = [sum((nr, nc) in pts for nr, nc in _neighbors4(r, c)) for r, c in comp]
        if len(comp) < 2:
            return False
        if any(d not in (1, 2) for d in degrees):
            return False
        if degrees.count(1) != 2:
            return False
    return np.any(mask)


def _build_candidate() -> np.ndarray | None:
    panel_h = random.randint(7, 14)
    panel_w = random.randint(14, 28)
    top = np.zeros((panel_h, panel_w), dtype=int)
    bottom = np.zeros((panel_h, panel_w), dtype=int)
    blocked = np.zeros((panel_h, panel_w), dtype=bool)

    max_loops = 1 if panel_w < 18 else 2 if panel_w < 24 or panel_h < 10 else 3
    num_loops = random.randint(1, max_loops)
    require_concave = random.random() < 0.8
    concave_used = False

    for loop_idx in range(num_loops):
        force_concave = require_concave and not concave_used and (loop_idx == num_loops - 1 or random.random() < 0.7)
        placed = False
        for _ in range(150):
            ph = random.randint(5, min(11, panel_h - 1))
            pw = random.randint(5, min(13, panel_w - 1))
            if ph > panel_h or pw > panel_w:
                continue
            made = _make_loop_patch(ph, pw, force_concave)
            if made is None:
                continue
            patch, local_path = made
            pos = _place_patch((panel_h, panel_w), blocked, patch)
            if pos is None:
                continue
            r0, c0 = pos
            global_path = [(r0 + r, c0 + c) for r, c in local_path]
            _assign_open_segments(global_path, top, bottom)
            _paint_blocked(blocked, patch, r0, c0)
            counts = _turn_sign_counts(local_path)
            concave_used = concave_used or (counts[1] > 0 and counts[-1] > 0)
            placed = True
            break
        if not placed:
            return None

    if require_concave and not concave_used:
        return None
    if not _open_paths_only(top, BLUE) or not _open_paths_only(bottom, RED):
        return None

    inp = np.vstack([
        top,
        np.full((1, panel_w), GREY, dtype=int),
        bottom,
    ])
    try:
        out = main(inp)
    except Exception:
        return None

    overlay = (top != 0) | (bottom != 0)
    if not np.array_equal(out, _classify_loops(overlay)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid overlay-loop ARC input")
