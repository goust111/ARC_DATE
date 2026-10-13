import random
from collections import deque

import numpy as np


GREY = 5
BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "expected non-empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return mask | (~mask & ~outside)


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return out
    out[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return out


def _extract_shells(mask: np.ndarray) -> list[np.ndarray]:
    filled = _fill_holes(mask)
    current = filled.copy()
    shells: list[np.ndarray] = []
    covered = np.zeros_like(mask, dtype=bool)

    while np.any(current):
        nxt = _erode4(current)
        shell = current & ~nxt
        present = shell & mask
        if not np.any(present):
            break
        shells.append(present)
        covered |= present
        current = nxt

    assert np.array_equal(covered, mask), "inner grey pixels must form nested shells"
    return shells


def _trace_path(path_mask: np.ndarray, frame_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    pts = [tuple(map(int, p)) for p in np.argwhere(path_mask)]
    assert pts, "expected a colored path outside the frame"

    endpoints = []
    for r, c in pts:
        degree = sum(
            1
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc]
        )
        assert degree <= 2, "path must not branch"
        if degree == 1:
            endpoints.append((r, c))

    assert len(endpoints) == 2, "path must have exactly two endpoints"
    starts = [
        (r, c)
        for r, c in endpoints
        if any(
            0 <= nr < h and 0 <= nc < w and frame_mask[nr, nc]
            for nr, nc in _neighbors4(r, c)
        )
    ]
    assert len(starts) == 1, "exactly one endpoint must touch the frame"

    start = starts[0]
    order = [start]
    visited = {start}
    prev = None
    cur = start
    while True:
        nxts = [
            (nr, nc)
            for nr, nc in _neighbors4(*cur)
            if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc] and (nr, nc) != prev
        ]
        if not nxts:
            break
        assert len(nxts) == 1, "path must be a single chain"
        prev, cur = cur, nxts[0]
        assert cur not in visited, "path must not self-intersect"
        visited.add(cur)
        order.append(cur)

    assert len(visited) == len(pts), "path must be connected"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the outer path's color-segment order to recolor the nested grey shells inside the central frame.

    Input structure:
    - A black background contains one large grey rectangular frame.
    - Inside the frame sits a grey nested closed shape: several concentric shells of the same form, shrinking inward one pixel per layer.
    - Outside the frame there is a single one-pixel-wide colored path with no branches; one endpoint touches the frame and the other endpoint is free.

    Rule:
    - Find the endpoint adjacent to the frame and traverse the path from there to the far end.
    - Compress the traversed path into consecutive color segments and read their colors in order.
    - Recolor the innermost framed shape's shells from outermost to innermost using that segment order, erase the path to background, and keep the grey frame unchanged.

    Key insight:
    - The path geometry is irrelevant except for locating the start; the usable signal is the sequence of visible color runs, which matches the shell order revealed by peeling the nested grey shape layer by layer.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    grey_mask = g == GREY
    r0, c0, r1, c1 = _bbox_from_mask(grey_mask)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must enclose a non-trivial interior"
    assert np.all(g[r0, c0 : c1 + 1] == GREY)
    assert np.all(g[r1, c0 : c1 + 1] == GREY)
    assert np.all(g[r0 : r1 + 1, c0] == GREY)
    assert np.all(g[r0 : r1 + 1, c1] == GREY)

    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[r0, c0 : c1 + 1] = True
    frame_mask[r1, c0 : c1 + 1] = True
    frame_mask[r0 : r1 + 1, c0] = True
    frame_mask[r0 : r1 + 1, c1] = True

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_grey = inner == GREY
    assert np.any(inner_grey), "expected nested grey shells inside the frame"
    shells = _extract_shells(inner_grey)

    path_mask = (g != bg) & (g != GREY)
    path = _trace_path(path_mask, frame_mask)
    segment_colors = []
    prev_color = None
    for r, c in path:
        color = int(g[r, c])
        if color != prev_color:
            segment_colors.append(color)
            prev_color = color

    assert len(segment_colors) == len(shells), "segment count must equal shell count"

    out = g.copy()
    out[path_mask] = bg
    inner_out = out[r0 + 1 : r1, c0 + 1 : c1]
    inner_out[inner_grey] = bg
    for shell, color in zip(shells, segment_colors):
        inner_out[shell] = int(color)

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _make_rectangle_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _make_diamond_mask(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    center = radius
    rr, cc = np.indices((size, size))
    return (np.abs(rr - center) + np.abs(cc - center) <= radius)


def _shells_from_base(base_mask: np.ndarray, layers: int) -> tuple[list[np.ndarray], np.ndarray]:
    shells = []
    current = base_mask.copy()
    for _ in range(layers):
        nxt = _erode4(current)
        shell = current & ~nxt
        assert np.any(shell), "each requested layer must be non-empty"
        shells.append(shell)
        current = nxt
    assert np.any(current), "core must remain to create a visible enclosed center"
    return shells, current


def _choose_shape() -> tuple[np.ndarray, list[np.ndarray]]:
    layers = random.randint(2, 5)
    kind = random.choice(("rectangle", "diamond"))

    if kind == "rectangle":
        core_h = random.randint(1, 4)
        core_w = random.randint(1, 6)
        base_h = core_h + 2 * layers
        base_w = core_w + 2 * layers
        base = _make_rectangle_mask(base_h, base_w)
    else:
        core_radius = random.randint(0, 2)
        base_radius = layers + core_radius
        base = _make_diamond_mask(base_radius)

    shells, _ = _shells_from_base(base, layers)
    return base, shells


def _choose_canvas(frame_h: int, frame_w: int, side: str) -> tuple[int, int, int, int] | None:
    path_margin_min = 4
    soft_cap = 7

    if side in ("top", "bottom"):
        max_path = 30 - frame_h - 1
        if max_path < path_margin_min:
            return None
        path_margin = random.randint(path_margin_min, min(soft_cap, max_path))
        other_max = min(soft_cap, 30 - frame_h - path_margin)
        if other_max < 1:
            return None
        other_margin = random.randint(1, other_max)
        left_max = min(soft_cap, 30 - frame_w - 1)
        if left_max < 1:
            return None
        left = random.randint(1, left_max)
        right_max = min(soft_cap, 30 - frame_w - left)
        if right_max < 1:
            return None
        right = random.randint(1, right_max)
        if side == "top":
            top, bottom = path_margin, other_margin
        else:
            top, bottom = other_margin, path_margin
    else:
        max_path = 30 - frame_w - 1
        if max_path < path_margin_min:
            return None
        path_margin = random.randint(path_margin_min, min(soft_cap, max_path))
        other_max = min(soft_cap, 30 - frame_w - path_margin)
        if other_max < 1:
            return None
        other_margin = random.randint(1, other_max)
        top_max = min(soft_cap, 30 - frame_h - 1)
        if top_max < 1:
            return None
        top = random.randint(1, top_max)
        bottom_max = min(soft_cap, 30 - frame_h - top)
        if bottom_max < 1:
            return None
        bottom = random.randint(1, bottom_max)
        if side == "left":
            left, right = path_margin, other_margin
        else:
            left, right = other_margin, path_margin

    return top, left, bottom, right


def _path_start(frame_bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return r0 - 1, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1 + 1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0 - 1
    return random.randint(r0 + 1, r1 - 1), c1 + 1


def _build_polyline_path(
    height: int,
    width: int,
    frame_bbox: tuple[int, int, int, int],
    side: str,
    min_length: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = frame_bbox
    start_r, start_c = _path_start(frame_bbox, side)

    if side == "top":
        primary = start_r
        step = -1
        primary_limit = start_r
        secondary = start_c
        secondary_size = width
        to_rc = lambda p, s: (p, s)
    elif side == "bottom":
        primary = start_r
        step = 1
        primary_limit = height - 1 - start_r
        secondary = start_c
        secondary_size = width
        to_rc = lambda p, s: (p, s)
    elif side == "left":
        primary = start_c
        step = -1
        primary_limit = start_c
        secondary = start_r
        secondary_size = height
        to_rc = lambda p, s: (s, p)
    else:
        primary = start_c
        step = 1
        primary_limit = width - 1 - start_c
        secondary = start_r
        secondary_size = height
        to_rc = lambda p, s: (s, p)

    if primary_limit < 2:
        return None

    levels = random.randint(2, min(4, primary_limit))
    total_away = random.randint(levels, primary_limit)
    away_lengths = _positive_composition(total_away, levels)
    path = [to_rc(primary, secondary)]

    for i, away in enumerate(away_lengths):
        for _ in range(away):
            primary += step
            path.append(to_rc(primary, secondary))

        need_lateral = i < len(away_lengths) - 1 or random.random() < 0.7
        if not need_lateral:
            continue
        choices = []
        if secondary > 0:
            choices.append(-1)
        if secondary < secondary_size - 1:
            choices.append(1)
        if not choices:
            break
        direction = random.choice(choices)
        room = secondary if direction < 0 else (secondary_size - 1 - secondary)
        lateral = random.randint(1, min(6, room))
        for _ in range(lateral):
            secondary += direction
            path.append(to_rc(primary, secondary))

    if len(path) < min_length:
        return None
    if len(path) != len(set(path)):
        return None

    frame_mask = np.zeros((height, width), dtype=bool)
    frame_mask[r0, c0 : c1 + 1] = True
    frame_mask[r1, c0 : c1 + 1] = True
    frame_mask[r0 : r1 + 1, c0] = True
    frame_mask[r0 : r1 + 1, c1] = True
    for idx, (r, c) in enumerate(path):
        if idx > 0 and any(
            0 <= nr < height and 0 <= nc < width and frame_mask[nr, nc]
            for nr, nc in _neighbors4(r, c)
        ):
            return None

    return path


def _segment_colors(path_len: int, layers: int) -> tuple[list[int], list[int]]:
    lengths = _positive_composition(path_len, layers)
    colors = []
    prev = None
    for _ in range(layers):
        choices = [color for color in PALETTE if color != prev]
        color = random.choice(choices)
        colors.append(color)
        prev = color
    return lengths, colors


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _gen_attempt() -> np.ndarray | None:
    base_mask, shells = _choose_shape()
    layers = len(shells)
    shape_h, shape_w = base_mask.shape
    moat = random.randint(1, 3)
    frame_h = shape_h + 2 * moat + 2
    frame_w = shape_w + 2 * moat + 2
    side = random.choice(("top", "bottom", "left", "right"))
    margins = _choose_canvas(frame_h, frame_w, side)
    if margins is None:
        return None
    top, left, bottom, right = margins
    height = frame_h + top + bottom
    width = frame_w + left + right
    if height > 30 or width > 30:
        return None

    r0, c0 = top, left
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    frame_bbox = (r0, c0, r1, c1)
    shape_r0 = r0 + 1 + moat
    shape_c0 = c0 + 1 + moat

    path = _build_polyline_path(height, width, frame_bbox, side, min_length=layers + 3)
    if path is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    _paint_frame(inp, r0, c0, r1, c1)

    shell_union = np.zeros((shape_h, shape_w), dtype=bool)
    for shell in shells:
        shell_union |= shell
    shape_patch = inp[shape_r0 : shape_r0 + shape_h, shape_c0 : shape_c0 + shape_w]
    shape_patch[shell_union] = GREY

    seg_lengths, seg_colors = _segment_colors(len(path), layers)
    idx = 0
    for seg_len, color in zip(seg_lengths, seg_colors):
        for _ in range(seg_len):
            r, c = path[idx]
            inp[r, c] = int(color)
            idx += 1

    expected = inp.copy()
    for r, c in path:
        expected[r, c] = BG
    expected_patch = expected[shape_r0 : shape_r0 + shape_h, shape_c0 : shape_c0 + shape_w]
    expected_patch[shell_union] = BG
    for shell, color in zip(shells, seg_colors):
        expected_patch[shell] = int(color)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
