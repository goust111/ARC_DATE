from __future__ import annotations

import random

import numpy as np


MOTIF_COLOR = 9
DEPTH_CHOICES = (2, 3, 4)
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GRID_SIZE = 42
GEN_RETRY_LIMIT = 400

BASE_PATTERNS = [
    np.array([[1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]
SMALL_PATTERN_IDS = (0, 1, 2, 3, 4, 5, 6, 7)


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in CARDINAL_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border_points = {(int(r), int(c)) for r, c in coords}
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(border_points) != expected:
        return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in border_points or (r1, c) not in border_points:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in border_points or (r, c1) not in border_points:
            return None
    return int(r0), int(c0), int(r1), int(c1)


def _expand_bbox(bbox: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 - amount), int(c0 - amount), int(r1 + amount), int(c1 + amount)


def _contract_bbox(bbox: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    nr0, nc0, nr1, nc1 = r0 + amount, c0 + amount, r1 - amount, c1 - amount
    if nr0 > nr1 or nc0 > nc1:
        return None
    return int(nr0), int(nc0), int(nr1), int(nc1)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paint_layer(out: np.ndarray, coords: np.ndarray, color: int, dr: int, dc: int, bg: int) -> None:
    h, w = out.shape
    for r, c in coords:
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == bg:
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify nested rectangular frame chains, remove the frames, and repaint the innermost motif
    as outward 2-pixel cardinal copies colored by the frame colors from inner to outer.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    frames: list[dict[str, object]] = []
    frame_mask = np.zeros((h, w), dtype=bool)
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        for coords in _components(g == int(color)):
            box = _frame_bbox(coords)
            if box is None:
                continue
            frames.append({"bbox": box, "color": int(color)})
            frame_mask[coords[:, 0], coords[:, 1]] = True

    assert frames, "expected at least one rectangular frame"
    frame_by_bbox = {tuple(frame["bbox"]): frame for frame in frames}
    assert len(frame_by_bbox) == len(frames), "frame bounding boxes must be unique"

    innermost = [frame for frame in frames if _contract_bbox(tuple(frame["bbox"]), 2) not in frame_by_bbox]
    assert innermost, "expected at least one innermost frame"

    out = np.full_like(g, bg)
    used_boxes: set[tuple[int, int, int, int]] = set()

    for frame in innermost:
        box = tuple(frame["bbox"])
        chain_colors: list[int] = []
        while box in frame_by_bbox:
            cur = frame_by_bbox[box]
            chain_colors.append(int(cur["color"]))
            used_boxes.add(box)
            box = _expand_bbox(box, 2)

        r0, c0, r1, c1 = tuple(frame["bbox"])
        interior = (~frame_mask) & (g != bg)
        motif_coords = np.argwhere(interior[r0 + 1 : r1, c0 + 1 : c1])
        assert motif_coords.size, "innermost frame must contain a motif"
        motif_coords = motif_coords + np.array([r0 + 1, c0 + 1], dtype=int)

        for r, c in motif_coords:
            out[int(r), int(c)] = int(g[int(r), int(c)])

        for layer, color in enumerate(chain_colors, start=1):
            step = 2 * layer
            _paint_layer(out, motif_coords, color, -step, 0, bg)
            _paint_layer(out, motif_coords, color, step, 0, bg)
            _paint_layer(out, motif_coords, color, 0, -step, bg)
            _paint_layer(out, motif_coords, color, 0, step, bg)

    assert used_boxes == set(frame_by_bbox), "every frame should belong to a nested chain"
    return out


def _random_pattern(use_large: bool, tight: bool = False) -> np.ndarray:
    if tight:
        candidates = [BASE_PATTERNS[i] for i in (0, 1, 2, 5)]
    elif use_large:
        candidates = BASE_PATTERNS
    else:
        candidates = [BASE_PATTERNS[i] for i in SMALL_PATTERN_IDS]
    pattern = random.choice(candidates).copy()
    pattern = np.rot90(pattern, random.randrange(4))
    if random.random() < 0.5:
        pattern = np.fliplr(pattern)
    if random.random() < 0.5:
        pattern = np.flipud(pattern)
    return pattern.astype(bool, copy=False)


def _sample_group(bg: int, n_groups: int, depth: int) -> dict[str, object]:
    pattern = _random_pattern(use_large=(n_groups == 1), tight=(n_groups == 2 and depth == 4))
    extra_limit = 2 if n_groups == 1 else 0
    inner_h = int(pattern.shape[0] + 4 + random.randint(0, extra_limit))
    inner_w = int(pattern.shape[1] + 4 + random.randint(0, extra_limit))
    outer_h = inner_h + 4 * (depth - 1)
    outer_w = inner_w + 4 * (depth - 1)
    palette = [c for c in range(10) if c not in {bg, MOTIF_COLOR}]
    colors_outer_to_inner = random.sample(palette, depth)
    return {
        "depth": depth,
        "pattern": pattern,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "colors_outer_to_inner": colors_outer_to_inner,
    }


def _layout_groups(groups: list[dict[str, object]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    sizes = [(int(group["outer_h"]), int(group["outer_w"])) for group in groups]
    if len(groups) == 1:
        gh, gw = sizes[0]
        top = random.randint(1, 3)
        bottom = random.randint(1, 4)
        left = random.randint(1, 3)
        right = random.randint(1, 4)
        h = gh + top + bottom + random.randint(0, 4)
        w = gw + left + right + random.randint(0, 4)
        if max(h, w) > MAX_GRID_SIZE:
            return None
        r = random.randint(top, h - bottom - gh)
        c = random.randint(left, w - right - gw)
        return h, w, [(r, c)]

    horizontal = random.random() < 0.5
    gap = random.randint(2, 3)
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)

    order = [0, 1]
    random.shuffle(order)
    (h0, w0), (h1, w1) = sizes[order[0]], sizes[order[1]]

    if horizontal:
        h = max(h0, h1) + top + bottom + random.randint(0, 3)
        w = w0 + w1 + gap + left + right + random.randint(0, 3)
        if max(h, w) > MAX_GRID_SIZE:
            return None
        extra_x = w - left - right - (w0 + gap + w1)
        x0 = left + random.randint(0, extra_x)
        x1 = x0 + w0 + gap
        y0 = random.randint(top, h - bottom - h0)
        y1 = random.randint(top, h - bottom - h1)
        ordered_origins = [(y0, x0), (y1, x1)]
    else:
        h = h0 + h1 + gap + top + bottom + random.randint(0, 3)
        w = max(w0, w1) + left + right + random.randint(0, 3)
        if max(h, w) > MAX_GRID_SIZE:
            return None
        extra_y = h - top - bottom - (h0 + gap + h1)
        y0 = top + random.randint(0, extra_y)
        y1 = y0 + h0 + gap
        x0 = random.randint(left, w - right - w0)
        x1 = random.randint(left, w - right - w1)
        ordered_origins = [(y0, x0), (y1, x1)]

    origins = [(-1, -1)] * len(groups)
    for idx, origin in zip(order, ordered_origins, strict=False):
        origins[idx] = origin
    return h, w, origins


def _render_group(
    inp: np.ndarray, out: np.ndarray, top: int, left: int, spec: dict[str, object], bg: int
) -> None:
    depth = int(spec["depth"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    pattern = np.asarray(spec["pattern"], dtype=bool)
    colors_outer_to_inner = list(spec["colors_outer_to_inner"])

    for level, color in enumerate(colors_outer_to_inner):
        inset = 2 * level
        bbox = (top + inset, left + inset, top + outer_h - 1 - inset, left + outer_w - 1 - inset)
        _draw_frame(inp, bbox, int(color))

    inner_top = top + 2 * (depth - 1)
    inner_left = left + 2 * (depth - 1)
    inner_bottom = inner_top + inner_h - 1
    inner_right = inner_left + inner_w - 1

    min_r = inner_top + 2
    max_r = inner_bottom - int(pattern.shape[0]) - 1
    min_c = inner_left + 2
    max_c = inner_right - int(pattern.shape[1]) - 1
    pr = random.randint(min_r, max_r)
    pc = random.randint(min_c, max_c)

    motif_coords = np.argwhere(pattern) + np.array([pr, pc], dtype=int)
    inp[motif_coords[:, 0], motif_coords[:, 1]] = MOTIF_COLOR
    out[motif_coords[:, 0], motif_coords[:, 1]] = MOTIF_COLOR

    for layer, color in enumerate(reversed(colors_outer_to_inner), start=1):
        step = 2 * layer
        _paint_layer(out, motif_coords, int(color), -step, 0, bg)
        _paint_layer(out, motif_coords, int(color), step, 0, bg)
        _paint_layer(out, motif_coords, int(color), 0, -step, bg)
        _paint_layer(out, motif_coords, int(color), 0, step, bg)


def _try_generate_case() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c != MOTIF_COLOR])
    n_groups = random.choice((1, 2))
    depth = random.choices(DEPTH_CHOICES, weights=(1, 1, 2), k=1)[0]
    groups = [_sample_group(bg, n_groups, depth) for _ in range(n_groups)]
    layout = _layout_groups(groups)
    if layout is None:
        return None
    h, w, origins = layout

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    for spec, (top, left) in zip(groups, origins, strict=False):
        _render_group(inp, out, top, left, spec, bg)

    if _get_bg(inp) != bg:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _try_generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
