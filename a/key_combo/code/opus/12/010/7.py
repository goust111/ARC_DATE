from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
FRAME_SIZE_RANGE = (5, 8)
GRID_SIZE_RANGE = (13, 16)
FRAME_COUNT_CHOICES = (3, 4)
GEN_ATTEMPTS = 800


def _border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = (int(v) for v in bbox)
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _border_set(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return set(_border_coords(bbox))


def _interior_slices(bbox: tuple[int, int, int, int]) -> tuple[slice, slice]:
    r0, c0, r1, c1 = (int(v) for v in bbox)
    return slice(r0 + 1, r1), slice(c0 + 1, c1)


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return (
        or0 <= ir0
        and oc0 <= ic0
        and or1 >= ir1
        and oc1 >= ic1
        and (or0, oc0, or1, oc1) != (ir0, ic0, ir1, ic1)
    )


def _pair_intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    return _border_set(a) & _border_set(b)


def _isolated_intersection_points(points: set[tuple[int, int]]) -> bool:
    pts = list(points)
    for i, (r0, c0) in enumerate(pts):
        for r1, c1 in pts[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _side_visible_counts(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = (int(v) for v in bbox)
    top = int(np.count_nonzero(grid[r0, c0 : c1 + 1] == color))
    bottom = int(np.count_nonzero(grid[r1, c0 : c1 + 1] == color))
    left = int(np.count_nonzero(grid[r0 : r1 + 1, c0] == color))
    right = int(np.count_nonzero(grid[r0 : r1 + 1, c1] == color))
    return top, bottom, left, right


def _recover_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in (int(c) for c in np.unique(g) if int(c) != BG):
        coords = np.argwhere(g == color)
        if coords.shape[0] < 8:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < FRAME_SIZE_RANGE[0] or w < FRAME_SIZE_RANGE[0]:
            continue
        border = _border_set(bbox)
        if any((int(r), int(c)) not in border for r, c in coords):
            continue
        side_counts = _side_visible_counts(g, bbox, color)
        if min(side_counts) < 1:
            continue
        frames.append(
            {
                "color": color,
                "bbox": bbox,
                "border": border,
                "visible_count": int(coords.shape[0]),
            }
        )
    assert 3 <= len(frames) <= 4, "expected 3-4 recoverable frame colors"
    return frames


def _infer_bottom_to_top_order(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    by_color = {int(frame["color"]): frame for frame in frames}
    above: dict[int, set[int]] = {color: set() for color in by_color}
    below: dict[int, set[int]] = {color: set() for color in by_color}

    colors = list(by_color)
    for i, ca in enumerate(colors):
        for cb in colors[i + 1 :]:
            inter = by_color[ca]["border"] & by_color[cb]["border"]
            if not inter:
                continue
            visible = {int(g[r, c]) for r, c in inter}
            assert len(visible) == 1, "crossing pixels must show a single visible color"
            top = visible.pop()
            assert top in (ca, cb), "crossing must reveal one of the two frame colors"
            lower = cb if top == ca else ca
            above[lower].add(top)
            below[top].add(lower)

    indegree = {color: len(preds) for color, preds in below.items()}
    available = [color for color, deg in indegree.items() if deg == 0]
    order: list[int] = []
    while available:
        assert len(available) == 1, "layer order must be uniquely determined"
        color = available.pop()
        order.append(color)
        for higher in list(above[color]):
            below[higher].remove(color)
            indegree[higher] -= 1
            if indegree[higher] == 0:
                available.append(higher)

    assert len(order) == len(colors), "pairwise evidence must determine all layers"
    return [by_color[color] for color in order]


def _render_output_from_order(shape: tuple[int, int], ordered_frames: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    border_grid = np.zeros(shape, dtype=int)
    for frame in ordered_frames:
        color = int(frame["color"])
        rr, cc = _interior_slices(frame["bbox"])
        out[rr, cc] = color
        for r, c in frame["border"]:
            border_grid[r, c] = color
    mask = border_grid != BG
    out[mask] = border_grid[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the front-to-back order of overlapping rectangular frames and fill them layer by layer.

    Input structure:
    - A black grid contains 3-4 one-pixel-thick rectangular frames, each in a distinct non-black color.
    - Frame borders partially overlap only at isolated crossing pixels, where only the top frame's color is visible.
    - Additional isolated foreground noise pixels may appear away from the true frame borders.

    Rule:
    - Ignore the isolated distractor pixels and recover each rectangular frame from the visible border pixels of its color.
    - For every pair of frames that cross, read the crossing color as evidence of which frame is on top, then combine all pairwise clues into one unique bottom-to-top order.
    - Fill each frame interior with its own border color from bottom layer to top layer so higher layers overwrite lower fills, while keeping every visible border pixel unchanged.

    Key insight:
    - The overlap pixels are the order clues: each crossing reveals which rectangle is in front, and that reconstructed layer order is exactly what determines the final fill priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _recover_frames(g)
    ordered = _infer_bottom_to_top_order(g, frames)
    return _render_output_from_order(g.shape, ordered)


def _random_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    bh = random.randint(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1])
    bw = random.randint(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1])
    r0 = random.randint(0, h - bh)
    c0 = random.randint(0, w - bw)
    return (r0, c0, r0 + bh - 1, c0 + bw - 1)


def _sample_ordered_rectangles(
    h: int, w: int, n_frames: int
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    for idx in range(n_frames):
        placed = False
        for _ in range(300):
            bbox = _random_bbox(h, w)
            if any(bbox == prev for prev in rects):
                continue
            if any(_strictly_contains(bbox, prev) or _strictly_contains(prev, bbox) for prev in rects):
                continue
            bad_pair = False
            for prev in rects:
                inter = _pair_intersection(bbox, prev)
                if inter and (len(inter) > 4 or not _isolated_intersection_points(inter)):
                    bad_pair = True
                    break
            if bad_pair:
                continue
            if idx > 0 and not _pair_intersection(rects[-1], bbox):
                continue
            rects.append(bbox)
            placed = True
            break
        if not placed:
            return None

    border_hits: defaultdict[tuple[int, int], int] = defaultdict(int)
    for bbox in rects:
        for cell in _border_coords(bbox):
            border_hits[cell] += 1
    if max(border_hits.values(), default=0) > 2:
        return None
    return rects


def _render_input(
    shape: tuple[int, int],
    ordered_rects: list[tuple[int, int, int, int]],
    ordered_colors: list[int],
    noise: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for bbox, color in zip(ordered_rects, ordered_colors):
        for r, c in _border_coords(bbox):
            grid[r, c] = int(color)
    for r, c, color in noise:
        grid[int(r), int(c)] = int(color)
    return grid


def _valid_noise_cells(
    h: int, w: int, rects: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]]:
    forbidden = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in rects:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    cells = [(r, c) for r in range(h) for c in range(w) if not forbidden[r, c]]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
    return chosen


def _semantic_ok(
    inp: np.ndarray,
    ordered_rects: list[tuple[int, int, int, int]],
    ordered_colors: list[int],
) -> bool:
    expected_frames = {
        (int(color), tuple(int(v) for v in bbox))
        for bbox, color in zip(ordered_rects, ordered_colors)
    }
    try:
        recovered = _recover_frames(inp)
        recovered_frames = {
            (int(frame["color"]), tuple(int(v) for v in frame["bbox"])) for frame in recovered
        }
        if recovered_frames != expected_frames:
            return False

        recovered_order = _infer_bottom_to_top_order(inp, recovered)
        if [int(frame["color"]) for frame in recovered_order] != [int(c) for c in ordered_colors]:
            return False
    except AssertionError:
        return False

    for bbox, color in zip(ordered_rects, ordered_colors):
        if min(_side_visible_counts(inp, bbox, color)) < 2:
            return False

    expected_out = _render_output_from_order(
        inp.shape,
        [
            {"color": int(color), "bbox": bbox, "border": _border_set(bbox)}
            for bbox, color in zip(ordered_rects, ordered_colors)
        ],
    )
    try:
        actual_out = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual_out, expected_out):
        return False
    if np.array_equal(actual_out, inp):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    n_frames = random.choice(FRAME_COUNT_CHOICES)
    colors = random.sample(PALETTE, n_frames)

    ordered_rects = _sample_ordered_rectangles(h, w, n_frames)
    if ordered_rects is None:
        return None

    available_noise_cells = _valid_noise_cells(h, w, ordered_rects)
    if len(available_noise_cells) < 2:
        return None
    noise_count = random.randint(2, min(6, len(available_noise_cells)))
    noise_colors = [color for color in PALETTE if color not in colors]
    noise: list[tuple[int, int, int]] = []
    for r, c in available_noise_cells[:noise_count]:
        noise.append((r, c, random.choice(noise_colors)))

    inp = _render_input((h, w), ordered_rects, colors, noise)
    if not _semantic_ok(inp, ordered_rects, colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-frame task")
