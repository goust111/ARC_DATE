from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
FG_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_legend_boxes(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(g[r0 : r1 + 1, c0 : c1 + 1] == GREY, border):
            boxes.append((r0, c0, r1, c1))
    boxes.sort()
    assert 2 <= len(boxes) <= 4
    return boxes


def _marker_info(g: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, list[tuple[int, int]]]:
    h, w = g.shape
    r0, c0, _r1, _c1 = box
    candidates: list[tuple[tuple[int, int, int], int, list[tuple[int, int]]]] = []
    probe_cols = []
    if c0 - 1 >= 0:
        probe_cols.append((0, c0 - 1))
    if c0 + 1 < w:
        probe_cols.append((1, c0 + 1))

    for priority, col in probe_cols:
        for sr in range(max(0, r0 - 1), min(h, r0 + 5)):
            color = int(g[sr, col])
            if color in (BLACK, GREY):
                continue
            if sr > 0 and int(g[sr - 1, col]) == color:
                continue
            length = 0
            while sr + length < h and int(g[sr + length, col]) == color and length < 5:
                length += 1
            if not (1 <= length <= 4):
                continue
            if sr + length < h and int(g[sr + length, col]) == color:
                continue
            cells = [(int(sr + k), int(col)) for k in range(length)]
            score = (priority, abs(sr - (r0 + 1)), sr)
            candidates.append((score, length, cells))

    assert candidates
    candidates.sort(key=lambda item: item[0])
    _score, length, cells = candidates[0]
    return int(length), cells


def _infer_tile(sample: np.ndarray) -> np.ndarray:
    candidates: list[np.ndarray] = []
    h, w = sample.shape
    for size in (2, 3):
        if h % size != 0 or w % size != 0:
            continue
        tile = sample[:size, :size]
        tiled = np.tile(tile, (h // size, w // size))
        if np.array_equal(tiled, sample):
            candidates.append(tile.copy())
    assert len(candidates) == 1
    return candidates[0]


def _legend_patterns(g: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)
    for box in _find_legend_boxes(g):
        r0, c0, r1, c1 = box
        count, marker_cells = _marker_info(g, box)
        sample = g[r0 + 1 : r1, c0 + 1 : c1]
        tile = _infer_tile(sample)
        assert count not in patterns
        patterns[count] = tile
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        for rr, cc in marker_cells:
            legend_mask[rr, cc] = True
    assert patterns
    return patterns, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black region with the legend pattern keyed by its number of boundary colors.

    Input structure:
    - Background is black.
    - A top or left legend contains 2-4 grey-framed samples.
    - Each sample has a short vertical marker whose length encodes 1-4.
    - Main-area polygons are enclosed by colored orthogonal line segments on black background.

    Rule:
    - Count the number of distinct non-black, non-grey colors on each region boundary.
    - Read the sample whose marker length matches that count.
    - Extract its repeated 2x2 or 3x3 tile and tile-fill the region from the region's top-left cell.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    patterns, legend_mask = _legend_patterns(g)

    region_mask = (g == BLACK) & (~legend_mask)
    h, w = g.shape
    for comp in _components(region_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        boundary_colors: set[int] = set()
        comp_set = {tuple(map(int, cell)) for cell in comp}
        for r0, c0 in comp_set:
            for nr, nc in _neighbors4(r0, c0):
                if (nr, nc) in comp_set:
                    continue
                color = int(g[nr, nc])
                if color not in (BLACK, GREY):
                    boundary_colors.add(color)

        count = len(boundary_colors)
        assert count in patterns
        tile = patterns[count]
        th, tw = tile.shape
        rr0, cc0, _rr1, _cc1 = _bbox(comp)
        for r0, c0 in comp_set:
            out[r0, c0] = int(tile[(r0 - rr0) % th, (c0 - cc0) % tw])

    return out


def _random_tile(rng: np.random.Generator, size: int) -> np.ndarray:
    for _ in range(200):
        n_colors = int(rng.integers(2, 5))
        chosen = rng.choice(FG_COLORS, size=n_colors, replace=False)
        tile = rng.choice(chosen, size=(size, size), replace=True).astype(int)
        if len(np.unique(tile)) < 2:
            continue
        tiled = np.tile(tile, (2, 2))
        if size == 3:
            tile2 = tiled[:2, :2]
            if np.array_equal(np.tile(tile2, (3, 3)), tiled):
                continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate tile")


def _sample_box(tile: np.ndarray) -> np.ndarray:
    interior = np.tile(tile, (2, 2))
    h, w = interior.shape
    box = np.full((h + 2, w + 2), BLACK, dtype=int)
    box[0, :] = GREY
    box[-1, :] = GREY
    box[:, 0] = GREY
    box[:, -1] = GREY
    box[1:-1, 1:-1] = interior
    return box


def _side_colors_for_count(rng: np.random.Generator, count: int) -> tuple[int, int, int, int]:
    colors = [int(x) for x in rng.choice(FG_COLORS, size=count, replace=False)]
    if count == 1:
        top = right = bottom = left = colors[0]
    elif count == 2:
        a, b = colors
        options = [
            (a, b, a, b),
            (a, a, b, b),
            (a, b, b, a),
        ]
        top, right, bottom, left = options[int(rng.integers(len(options)))]
    elif count == 3:
        repeated = colors + [colors[int(rng.integers(3))]]
        rng.shuffle(repeated)
        top, right, bottom, left = map(int, repeated)
    else:
        rng.shuffle(colors)
        top, right, bottom, left = map(int, colors)
    return int(top), int(right), int(bottom), int(left)


def _region_patches(
    rng: np.random.Generator, tile: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    inner_h = int(rng.integers(4, 9))
    inner_w = int(rng.integers(4, 11))
    top, right, bottom, left = _side_colors_for_count(rng, count)
    inp = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    inp[0, :] = top
    inp[-1, :] = bottom
    inp[:, 0] = left
    inp[:, -1] = right
    inp[0, 0] = top
    inp[0, -1] = top
    inp[-1, 0] = bottom
    inp[-1, -1] = bottom

    out = inp.copy()
    th, tw = tile.shape
    for r in range(inner_h):
        for c in range(inner_w):
            out[1 + r, 1 + c] = int(tile[r % th, c % tw])
    return inp, out


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _render_legend(
    inp: np.ndarray,
    out: np.ndarray,
    samples: list[dict[str, object]],
    orientation: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if orientation == "top":
        top_margin = int(rng.integers(1, 3))
        left_margin = int(rng.integers(1, 3))
        gaps = [int(rng.integers(1, 4)) for _ in range(max(0, len(samples) - 1))]
        current = left_margin
        legend_bottom = 0
        for idx, sample in enumerate(samples):
            box = np.asarray(sample["box"], dtype=int)
            count = int(sample["count"])
            marker_color = int(sample["marker_color"])
            box_r = top_margin
            box_c = current + 1
            _paste(inp, box, box_r, box_c)
            _paste(out, box, box_r, box_c)
            inp[box_r + 1 : box_r + 1 + count, current] = marker_color
            out[box_r + 1 : box_r + 1 + count, current] = marker_color
            legend_bottom = max(legend_bottom, box_r + box.shape[0] - 1)
            current = box_c + box.shape[1]
            if idx < len(gaps):
                current += gaps[idx]
        return legend_bottom + 2, 1

    top_margin = int(rng.integers(1, 3))
    left_margin = int(rng.integers(1, 3))
    gaps = [int(rng.integers(1, 4)) for _ in range(max(0, len(samples) - 1))]
    current = top_margin
    legend_right = 0
    for idx, sample in enumerate(samples):
        box = np.asarray(sample["box"], dtype=int)
        count = int(sample["count"])
        marker_color = int(sample["marker_color"])
        box_r = current
        box_c = left_margin + 1
        _paste(inp, box, box_r, box_c)
        _paste(out, box, box_r, box_c)
        inp[box_r + 1 : box_r + 1 + count, left_margin] = marker_color
        out[box_r + 1 : box_r + 1 + count, left_margin] = marker_color
        legend_right = max(legend_right, box_c + box.shape[1] - 1)
        current = box_r + box.shape[0]
        if idx < len(gaps):
            current += gaps[idx]
    return 1, legend_right + 2


def _layout_sizes(
    samples: list[dict[str, object]], region_count: int, orientation: str, rng: np.random.Generator
) -> tuple[int, int]:
    box_heights = [int(np.asarray(sample["box"]).shape[0]) for sample in samples]
    box_widths = [int(np.asarray(sample["box"]).shape[1]) for sample in samples]
    if orientation == "top":
        total_span = sum(1 + width for width in box_widths) + 3 * max(0, len(samples) - 1) + 4
        min_h = max(24, max(box_heights) + 14)
        min_w = max(34, total_span + 2)
        extra_h = int(rng.integers(4, 10))
        extra_w = int(rng.integers(4, 10))
        return min_h + extra_h, min_w + extra_w

    total_span = sum(box_heights) + 3 * max(0, len(samples) - 1) + 4
    min_h = max(28, total_span + 2)
    min_w = max(30, max(box_widths) + 16)
    extra_h = int(rng.integers(4, 10))
    extra_w = int(rng.integers(4, 10))
    return min_h + extra_h, min_w + extra_w


def _place_regions(
    inp: np.ndarray,
    out: np.ndarray,
    patches: list[tuple[np.ndarray, np.ndarray]],
    free_r0: int,
    free_c0: int,
    rng: np.random.Generator,
) -> bool:
    h, w = inp.shape
    free_r1 = h - 2
    free_c1 = w - 2
    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1],
        reverse=True,
    )
    placed: list[tuple[int, int, int, int]] = []
    for idx in order:
        in_patch, out_patch = patches[idx]
        ph, pw = in_patch.shape
        max_r0 = free_r1 - ph + 1
        max_c0 = free_c1 - pw + 1
        if free_r0 > max_r0 or free_c0 > max_c0:
            return False
        for _ in range(250):
            r0 = int(rng.integers(free_r0, max_r0 + 1))
            c0 = int(rng.integers(free_c0, max_c0 + 1))
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_overlap(bbox, other, gap=2) for other in placed):
                continue
            _paste(inp, in_patch, r0, c0)
            _paste(out, out_patch, r0, c0)
            placed.append(bbox)
            break
        else:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    orientation = str(rng.choice(np.array(["top", "left"])))
    legend_count = int(rng.integers(2, 5))
    legend_counts = [int(x) for x in rng.choice(np.arange(1, 5), size=legend_count, replace=False)]

    signatures: set[tuple[int, ...]] = set()
    samples: list[dict[str, object]] = []
    marker_colors = [int(x) for x in rng.choice(FG_COLORS, size=legend_count, replace=False)]
    for idx, count in enumerate(legend_counts):
        for _ in range(100):
            tile_size = int(rng.choice(np.array([2, 3])))
            tile = _random_tile(rng, tile_size)
            signature = (tile_size, *tile.flatten().tolist())
            if signature in signatures:
                continue
            signatures.add(signature)
            samples.append(
                {
                    "count": count,
                    "tile": tile,
                    "box": _sample_box(tile),
                    "marker_color": marker_colors[idx],
                }
            )
            break
        else:
            return None

    rng.shuffle(samples)
    region_counts = legend_counts.copy()
    if rng.random() < 0.5:
        region_counts.append(int(rng.choice(np.array(legend_counts))))
    rng.shuffle(region_counts)

    grid_h, grid_w = _layout_sizes(samples, len(region_counts), orientation, rng)
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    out = inp.copy()

    free_r0, free_c0 = _render_legend(inp, out, samples, orientation, rng)

    tile_by_count = {int(sample["count"]): np.asarray(sample["tile"], dtype=int) for sample in samples}
    patches = [_region_patches(rng, tile_by_count[count], count) for count in region_counts]
    if not _place_regions(inp, out, patches, free_r0, free_c0, rng):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
