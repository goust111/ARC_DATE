import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg, neighbors4


BG = Color.BLACK
LOOP = Color.BLUE
CONTROL = Color.GREY
CONCAVE = Color.RED
CONVEX_ODD = Color.GREEN
CONVEX_EVEN = Color.YELLOW

GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    nbrs = sorted((nr, nc) for nr, nc in neighbors4(*start) if (nr, nc) in pts)
    assert len(nbrs) == 2

    path = [start]
    prev = start
    cur = nbrs[0]
    while cur != start:
        path.append(cur)
        nxt = [(nr, nc) for nr, nc in neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(nxt) == 1, (cur, nxt)
        prev, cur = cur, nxt[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _classify_path(path: list[tuple[int, int]]):
    turns: dict[tuple[int, int], int] = {}
    counts: Counter[int] = Counter()
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
        turns[(r, c)] = sign
        counts[sign] += 1

    assert turns
    assert abs(counts[1] - counts[-1]) == 4, counts
    convex_sign = 1 if counts[1] > counts[-1] else -1
    straight = set(path) - set(turns)
    convex = {pt for pt, sign in turns.items() if sign == convex_sign}
    concave = set(turns) - convex
    return straight, convex, concave


def _enclosed_mask(loop_mask: np.ndarray) -> np.ndarray:
    h, w = loop_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not loop_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not loop_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~loop_mask) & (~outside)


def _find_control(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[int(r) - r0, int(c) - c0] = True
    enclosed = _enclosed_mask(local)
    markers = np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == CONTROL)
    hits = []
    for mr, mc in markers:
        if enclosed[int(mr), int(mc)]:
            hits.append((r0 + int(mr), c0 + int(mc)))
    assert len(hits) == 1
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Classify loop pixels by straightness, convexity, and convex-corner distance parity to an interior control pixel.

    Input structure:
    - The grid has black background, one or two blue one-pixel-wide closed orthogonal loops, and one grey control pixel inside each loop.
    - Every blue loop pixel has exactly two 4-neighbor blue loop neighbors, and each loop has both convex and concave bends.

    Rule:
    - Trace each blue loop cyclically and label each loop pixel as straight or bend according to whether the path turns there.
    - Among the bends of one loop, the majority turn direction marks convex bends and the minority direction marks concave bends.
    - Keep straight pixels blue and keep each grey control pixel unchanged.
    - Recolor concave bends red.
    - For convex bends, compute the Chebyshev distance to that loop's interior grey control pixel: even distances become yellow and odd distances become green.

    Key insight:
    - The output requires three nested structural decisions: trace the cycle, separate bends from straights, infer convex vs concave from turn-sign majority, then split only the convex bends by Chebyshev-distance parity to the unique enclosed control pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert get_bg(g) == BG

    out = g.copy()
    loop_components = components_mask(g == LOOP, connectivity=4)
    assert 1 <= len(loop_components) <= 2
    assert np.count_nonzero(g == CONTROL) == len(loop_components)

    used_controls = set()
    for coords in loop_components:
        path = _trace_cycle(coords)
        _, convex, concave = _classify_path(path)
        control = _find_control(g, coords)
        used_controls.add(control)
        cr, cc = control

        for r, c in concave:
            out[r, c] = CONCAVE
        for r, c in convex:
            dist = max(abs(r - cr), abs(c - cc))
            out[r, c] = CONVEX_EVEN if dist % 2 == 0 else CONVEX_ODD

    assert len(used_controls) == np.count_nonzero(g == CONTROL)
    return out


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[0] == cells[-1]:
        cells = cells[:-1]
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


def _notch_options(top: list[int], bottom: list[int], side: str, max_depth: int):
    height = max(bottom) + 1
    width = len(top)
    profile = top if side == "top" else bottom
    base = 0 if side == "top" else height - 1
    options = []
    for start in range(1, width - 3):
        for end in range(start + 1, width - 1):
            lo = max(0, start - 1)
            hi = min(width, end + 2)
            if any(v != base for v in profile[lo:hi]):
                continue
            if side == "top":
                limit = min(bottom[c] - 2 for c in range(lo, hi))
            else:
                limit = min((height - 1) - top[c] - 2 for c in range(lo, hi))
            for depth in range(1, min(max_depth, limit) + 1):
                options.append((side, start, end, depth))
    return options


def _apply_notch(top: list[int], bottom: list[int], option) -> None:
    side, start, end, depth = option
    if side == "top":
        for c in range(start, end + 1):
            top[c] = depth
    else:
        height = max(bottom) + 1
        for c in range(start, end + 1):
            bottom[c] = height - 1 - depth


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, random.randrange(4))


def _build_loop_patch() -> np.ndarray | None:
    height = random.randint(7, 11)
    width = random.randint(8, 13)
    top = [0] * width
    bottom = [height - 1] * width
    target_notches = random.choices([1, 2, 3], weights=[3, 4, 2], k=1)[0]
    applied = 0
    for _ in range(target_notches * 4):
        side = random.choice(["top", "bottom"])
        options = _notch_options(top, bottom, side, max_depth=max(1, height // 2 - 1))
        if not options:
            side = "bottom" if side == "top" else "top"
            options = _notch_options(top, bottom, side, max_depth=max(1, height // 2 - 1))
        if not options:
            continue
        _apply_notch(top, bottom, random.choice(options))
        applied += 1
        if applied >= target_notches:
            break
    if applied == 0:
        return None

    path = _path_from_profiles(top, bottom)
    patch = np.zeros((height, width), dtype=int)
    for r, c in path:
        patch[r, c] = LOOP

    coords = np.argwhere(patch == LOOP)
    try:
        traced = _trace_cycle(coords)
        straight, convex, concave = _classify_path(traced)
    except AssertionError:
        return None
    if not straight or len(concave) < 1 or len(convex) < 5:
        return None

    enclosed = _enclosed_mask(patch == LOOP)
    interior = [tuple(map(int, p)) for p in np.argwhere(enclosed)]
    if len(interior) < 3:
        return None

    good_controls = []
    for r, c in interior:
        parities = {max(abs(r - rr), abs(c - cc)) % 2 for rr, cc in convex}
        if parities == {0, 1}:
            good_controls.append((r, c))
    if not good_controls:
        return None

    control = random.choice(good_controls)
    patch[control] = CONTROL
    patch = _transform_patch(patch)

    coords = np.argwhere(patch == LOOP)
    try:
        traced = _trace_cycle(coords)
        _, convex, concave = _classify_path(traced)
        control = tuple(map(int, np.argwhere(patch == CONTROL)[0]))
    except (AssertionError, IndexError):
        return None
    parities = {max(abs(control[0] - rr), abs(control[1] - cc)) % 2 for rr, cc in convex}
    if len(concave) < 1 or parities != {0, 1}:
        return None
    return patch


def _place_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = canvas[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _compose_canvas(patches: list[np.ndarray]) -> np.ndarray | None:
    if len(patches) == 1:
        patch = patches[0]
        ph, pw = patch.shape
        h = random.randint(max(15, ph + 4), min(24, ph + 10, 30))
        w = random.randint(max(15, pw + 4), min(24, pw + 10, 30))
        canvas = np.zeros((h, w), dtype=int)
        top = random.randint(1, h - ph - 1)
        left = random.randint(1, w - pw - 1)
        _place_patch(canvas, patch, top, left)
        return canvas

    p1, p2 = patches
    if random.random() < 0.5:
        p1, p2 = p2, p1
    h1, w1 = p1.shape
    h2, w2 = p2.shape
    gap = random.randint(3, 6)

    if random.random() < 0.5:
        h = max(16, max(h1, h2) + random.randint(4, 8))
        w = w1 + w2 + gap + random.randint(4, 8)
        if h > 30 or w > 30:
            return None
        canvas = np.zeros((h, w), dtype=int)
        top1 = random.randint(1, h - h1 - 1)
        top2 = random.randint(1, h - h2 - 1)
        left1 = 1
        left2 = w - w2 - 1
        if left2 - (left1 + w1) < gap:
            return None
        _place_patch(canvas, p1, top1, left1)
        _place_patch(canvas, p2, top2, left2)
        return canvas

    h = h1 + h2 + gap + random.randint(4, 8)
    w = max(16, max(w1, w2) + random.randint(4, 8))
    if h > 30 or w > 30:
        return None
    canvas = np.zeros((h, w), dtype=int)
    left1 = random.randint(1, w - w1 - 1)
    left2 = random.randint(1, w - w2 - 1)
    top1 = 1
    top2 = h - h2 - 1
    if top2 - (top1 + h1) < gap:
        return None
    _place_patch(canvas, p1, top1, left1)
    _place_patch(canvas, p2, top2, left2)
    return canvas


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if get_bg(grid) != BG:
        return False

    comps = components_mask(grid == LOOP, connectivity=4)
    if not (1 <= len(comps) <= 2):
        return False
    if np.count_nonzero(grid == CONTROL) != len(comps):
        return False

    used_controls = set()
    for coords in comps:
        try:
            path = _trace_cycle(coords)
            straight, convex, concave = _classify_path(path)
            control = _find_control(grid, coords)
        except AssertionError:
            return False
        if not straight or len(concave) < 1 or len(convex) - len(concave) != 4:
            return False
        parities = {max(abs(control[0] - r), abs(control[1] - c)) % 2 for r, c in convex}
        if parities != {0, 1}:
            return False
        used_controls.add(control)

    if len(used_controls) != len(comps):
        return False

    out = main(grid)
    colors = set(map(int, out.flatten()))
    return {LOOP, CONTROL, CONCAVE, CONVEX_ODD, CONVEX_EVEN}.issubset(colors)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        count = random.choices([1, 2], weights=[2, 3], k=1)[0]
        patches = []
        ok = True
        for _ in range(count):
            patch = None
            for _ in range(80):
                patch = _build_loop_patch()
                if patch is not None:
                    break
            if patch is None:
                ok = False
                break
            patches.append(patch)
        if not ok:
            continue
        canvas = _compose_canvas(patches)
        if canvas is None:
            continue
        if _validate_input(canvas):
            return canvas
    raise RuntimeError("failed to generate a valid input")
