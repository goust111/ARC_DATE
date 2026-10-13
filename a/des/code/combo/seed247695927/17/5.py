from __future__ import annotations

import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.asarray(coords, dtype=int)
    return (
        int(np.min(pts[:, 0])),
        int(np.min(pts[:, 1])),
        int(np.max(pts[:, 0])),
        int(np.max(pts[:, 1])),
    )


def _legend_pairs(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left == bg or right == bg:
            break
        pairs.append((left, right))
    assert pairs, "expected a non-empty legend in columns 0 and 1"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 2

    bg = get_bg(g)
    pairs = _legend_pairs(g, bg)
    sources = [source for source, _fill in pairs]
    assert len(set(sources)) == len(sources), "legend source colors must be unique"

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[: len(pairs), :2] = True
    dot_mask = (g != bg) & ~legend_mask
    assert dot_mask.any(), "expected non-legend source dots"
    assert set(map(int, np.unique(g[dot_mask]))).issubset(set(sources))

    dots_by_color: dict[int, np.ndarray] = {}
    all_dots = np.zeros_like(g, dtype=bool)
    for source, _fill in pairs:
        coords = np.argwhere(dot_mask & (g == source))
        assert len(coords) > 0, "each source color must appear outside the legend"
        dots_by_color[source] = coords
        all_dots[coords[:, 0], coords[:, 1]] = True

    out = np.full_like(g, bg)
    for source, fill in reversed(pairs):
        r0, c0, r1, c1 = bbox_coords(dots_by_color[source])
        r0 = max(0, r0 - 1)
        c0 = max(0, c0 - 1)
        r1 = min(h - 1, r1 + 1)
        c1 = min(w - 1, c1 + 1)
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[~all_dots[r0 : r1 + 1, c0 : c1 + 1]] = fill

    out[all_dots] = g[all_dots]
    return out


COLORS = tuple(range(10))
LEGEND_COLS = 2
PAD = 1
RETRIES = 500


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _sample_pairs(rng: random.Random, n_pairs: int, bg: int) -> list[tuple[int, int]]:
    palette = [c for c in COLORS if c != bg]
    chosen = rng.sample(palette, 2 * n_pairs)
    sources = chosen[:n_pairs]
    fills = chosen[n_pairs:]
    if rng.random() < 0.35:
        fills = fills[1:] + fills[:1]
    return list(zip(sources, fills))


def _box_dims(dot_rows: int, dot_cols: int) -> tuple[int, int]:
    return (2 * dot_rows + 1, 2 * dot_cols + 1)


def _sample_free_box(
    rng: random.Random,
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    placed: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(120):
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(LEGEND_COLS, w - box_w)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(_boxes_separated(box, other, gap=1) for other in placed):
            return box
    return None


def _sample_overlap_box(
    rng: random.Random,
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    target: tuple[int, int, int, int],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        min_r = max(0, target[0] - box_h + 2)
        max_r = min(h - box_h, target[2] - 1)
        min_c = max(LEGEND_COLS, target[1] - box_w + 2)
        max_c = min(w - box_w, target[3] - 1)
        if min_r > max_r or min_c > max_c:
            continue
        r0 = rng.randint(min_r, max_r)
        c0 = rng.randint(min_c, max_c)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if not _boxes_intersect(box, target):
            continue
        inter = _intersection(box, target)
        assert inter is not None
        if inter[2] - inter[0] < 1 or inter[3] - inter[1] < 1:
            continue
        if any(not _boxes_separated(box, other, gap=0) and other != target and _intersection(box, other) is None for other in existing):
            continue
        return box
    return None


def _dot_coords_from_box(
    rng: random.Random,
    box: tuple[int, int, int, int],
    dot_rows: int,
    dot_cols: int,
) -> np.ndarray:
    r0, c0, _r1, _c1 = box
    coords = []
    for i in range(dot_rows):
        for j in range(dot_cols):
            coords.append((r0 + 1 + 2 * i, c0 + 1 + 2 * j))
    coords = np.asarray(coords, dtype=int)

    if dot_rows >= 3 and dot_cols >= 3:
        boundary = (
            (coords[:, 0] == coords[:, 0].min())
            | (coords[:, 0] == coords[:, 0].max())
            | (coords[:, 1] == coords[:, 1].min())
            | (coords[:, 1] == coords[:, 1].max())
        )
        interior = np.where(~boundary)[0].tolist()
        rng.shuffle(interior)
        remove_n = rng.randint(0, len(interior))
        if remove_n:
            keep = np.ones(len(coords), dtype=bool)
            keep[interior[:remove_n]] = False
            coords = coords[keep]
    return coords


def _build_scene(
    rng: random.Random,
    h: int,
    w: int,
    shapes: list[tuple[int, int]],
    mode: str,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for idx, (dot_rows, dot_cols) in enumerate(shapes):
        box_h, box_w = _box_dims(dot_rows, dot_cols)
        if idx == 0 or mode == "separate":
            box = _sample_free_box(rng, h, w, box_h, box_w, boxes)
        else:
            if idx == 1 or rng.random() < 0.5:
                target = rng.choice(boxes)
                box = _sample_overlap_box(rng, h, w, box_h, box_w, target, boxes)
            else:
                box = _sample_free_box(rng, h, w, box_h, box_w, boxes)
        if box is None:
            return None
        if mode == "separate" and any(_boxes_intersect(box, other) for other in boxes):
            return None
        boxes.append(box)

    if mode == "overlap" and not any(
        _boxes_intersect(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
    ):
        return None

    dot_sets: list[np.ndarray] = []
    occupied = set()
    for box, (dot_rows, dot_cols) in zip(boxes, shapes):
        coords = _dot_coords_from_box(rng, box, dot_rows, dot_cols)
        tuples = [tuple(map(int, p)) for p in coords]
        if len(set(tuples)) != len(tuples):
            return None
        if any(p in occupied for p in tuples):
            return None
        occupied.update(tuples)
        dot_sets.append(coords)
    return boxes, dot_sets


def _render_input(
    bg: int,
    h: int,
    w: int,
    pairs: list[tuple[int, int]],
    dots_by_color: list[np.ndarray],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    grid[: len(pairs), :2] = np.asarray(pairs, dtype=int)
    for (source, _fill), coords in zip(pairs, dots_by_color):
        grid[coords[:, 0], coords[:, 1]] = source
    return grid


def _visible_overlap(inp: np.ndarray, out: np.ndarray, pairs: list[tuple[int, int]]) -> bool:
    bg = get_bg(inp)
    legend_rows = len(pairs)
    legend_mask = np.zeros_like(inp, dtype=bool)
    legend_mask[:legend_rows, :2] = True
    dots = (inp != bg) & ~legend_mask
    for hi in range(len(pairs)):
        hi_fill = pairs[hi][1]
        hi_coords = np.argwhere(dots & (inp == pairs[hi][0]))
        hi_box = bbox_coords(hi_coords)
        hi_box = (max(0, hi_box[0] - 1), max(0, hi_box[1] - 1), min(inp.shape[0] - 1, hi_box[2] + 1), min(inp.shape[1] - 1, hi_box[3] + 1))
        for lo in range(hi + 1, len(pairs)):
            lo_fill = pairs[lo][1]
            lo_coords = np.argwhere(dots & (inp == pairs[lo][0]))
            lo_box = bbox_coords(lo_coords)
            lo_box = (max(0, lo_box[0] - 1), max(0, lo_box[1] - 1), min(inp.shape[0] - 1, lo_box[2] + 1), min(inp.shape[1] - 1, lo_box[3] + 1))
            inter = _intersection(hi_box, lo_box)
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            if np.any(out[r0 : r1 + 1, c0 : c1 + 1] == hi_fill) and np.any(out == lo_fill):
                return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.choice((2, 3))
    h = rng.randint(12, 22)
    w = rng.randint(14, 24)
    bg = rng.choice(COLORS)
    pairs = _sample_pairs(rng, n_pairs, bg)
    mode = rng.choice(("overlap", "separate"))

    shapes = [(rng.randint(2, 5), rng.randint(2, 5)) for _ in range(n_pairs)]
    if h < max(_box_dims(r, c)[0] for r, c in shapes) or w - LEGEND_COLS < max(_box_dims(r, c)[1] for r, c in shapes):
        return None

    built = _build_scene(rng, h, w, shapes, mode)
    if built is None:
        return None
    boxes, dots = built

    inp = _render_input(bg, h, w, pairs, dots)
    try:
        out = main(inp)
    except AssertionError:
        return None

    if mode == "separate" and any(_boxes_intersect(boxes[i], boxes[j]) for i in range(n_pairs) for j in range(i + 1, n_pairs)):
        return None
    if mode == "overlap":
        if not any(_boxes_intersect(boxes[i], boxes[j]) for i in range(n_pairs) for j in range(i + 1, n_pairs)):
            return None
        if not _visible_overlap(inp, out, pairs):
            return None

    if len(np.unique(inp)) < min(4, n_pairs + 2):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
