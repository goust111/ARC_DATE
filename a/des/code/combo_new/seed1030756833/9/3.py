from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 5
YELLOW = 4
MAGENTA = 6
AZURE = 8
MARKER_COLORS = (YELLOW, MAGENTA, AZURE)
MARKER_SET = set(MARKER_COLORS)

HFLIP = "hflip"
ROTCW = "rotcw"
VFLIP = "vflip"
LEGEND_OPS = {YELLOW: HFLIP, MAGENTA: ROTCW, AZURE: VFLIP}

LEGEND_H = 11
LEGEND_W = 10
LEGEND_BAND_TOPS = (0, 4, 8)
LEGEND_BAND_H = 3

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = int(r) + dr
                    cc = int(c) + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        q.append((rr, cc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _transform(arr: np.ndarray, op: str) -> np.ndarray:
    if op == HFLIP:
        return np.fliplr(arr).copy()
    if op == VFLIP:
        return np.flipud(arr).copy()
    if op == ROTCW:
        return np.rot90(arr, -1).copy()
    raise ValueError(f"unknown operation: {op}")


def _operation_from_pair(source: np.ndarray, target: np.ndarray) -> str | None:
    matches = []
    for op in (HFLIP, ROTCW, VFLIP):
        transformed = _transform(source, op)
        if transformed.shape == target.shape and np.array_equal(transformed, target):
            matches.append(op)
    if len(matches) != 1:
        return None
    return matches[0]


def _parse_legend_at(g: np.ndarray, top: int, left: int) -> tuple[tuple[int, int, int, int], dict[int, str]] | None:
    patch = g[top : top + LEGEND_H, left : left + LEGEND_W]
    if patch.shape != (LEGEND_H, LEGEND_W):
        return None

    mapping: dict[int, str] = {}
    for band_top in LEGEND_BAND_TOPS:
        band = patch[band_top : band_top + LEGEND_BAND_H, :]
        marker_cells = [(int(r), int(c)) for r, c in np.argwhere(np.isin(band, MARKER_COLORS))]
        if len(marker_cells) != 1:
            return None
        mr, mc = marker_cells[0]
        marker = int(band[mr, mc])
        if marker in mapping:
            return None

        example_mask = band != BG
        example_mask[mr, mc] = False
        comps = _components(example_mask)
        if len(comps) != 2:
            return None

        examples = []
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            crop = band[r0 : r1 + 1, c0 : c1 + 1]
            if not bool(np.all(crop != BG)):
                return None
            examples.append((c0, crop.copy()))
        examples.sort(key=lambda item: item[0])

        source = examples[0][1]
        target = examples[1][1]
        op = _operation_from_pair(source, target)
        if op is None:
            return None
        mapping[marker] = op

    if set(mapping) != MARKER_SET:
        return None
    return (top, left, top + LEGEND_H - 1, left + LEGEND_W - 1), mapping


def _find_legend(g: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, str]]:
    h, w = g.shape
    candidates = [
        (0, 0),
        (0, w - LEGEND_W),
        (h - LEGEND_H, 0),
        (h - LEGEND_H, w - LEGEND_W),
    ]
    found = []
    for top, left in candidates:
        if top < 0 or left < 0:
            continue
        parsed = _parse_legend_at(g, top, left)
        if parsed is not None:
            found.append(parsed)
    assert len(found) == 1, "expected one corner legend"
    return found[0]


def _clear_box(arr: np.ndarray, box: tuple[int, int, int, int], color: int = BG) -> None:
    r0, c0, r1, c1 = box
    arr[r0 : r1 + 1, c0 : c1 + 1] = color


def _centered_top_left(r0: int, c0: int, h: int, w: int, new_h: int, new_w: int) -> tuple[int, int]:
    top2 = 2 * int(r0) + int(h) - 1 - (int(new_h) - 1)
    left2 = 2 * int(c0) + int(w) - 1 - (int(new_w) - 1)
    assert top2 % 2 == 0 and left2 % 2 == 0, "transformed rectangle must land on grid cells"
    return top2 // 2, left2 // 2


def _find_regions(g: np.ndarray, legend_box: tuple[int, int, int, int]) -> list[dict]:
    clean = g.copy()
    _clear_box(clean, legend_box)

    regions: list[dict] = []
    covered = np.zeros(clean.shape, dtype=bool)
    for color in sorted(int(v) for v in np.unique(clean)):
        if color == BG or color in MARKER_SET:
            continue
        for coords in _components(clean == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue

            crop = clean[r0 : r1 + 1, c0 : c1 + 1]
            marker_mask = np.isin(crop, MARKER_COLORS)
            if int(marker_mask.sum()) != 1:
                continue
            body_mask = crop == color
            if int(body_mask.sum()) != h * w - 1:
                continue
            if not bool(np.all(body_mask | marker_mask)):
                continue

            marker_pos = np.argwhere(marker_mask)[0]
            mr = int(marker_pos[0])
            mc = int(marker_pos[1])
            assert 0 < mr < h - 1 and 0 < mc < w - 1, "marker must be internal to its rectangle"
            regions.append(
                {
                    "color": color,
                    "marker": int(crop[mr, mc]),
                    "bbox": (r0, c0, r1, c1),
                    "patch": crop.copy(),
                }
            )
            covered[r0 : r1 + 1, c0 : c1 + 1] = True

    non_bg = clean != BG
    assert regions, "expected at least one marked rectangle"
    assert bool(np.array_equal(non_bg, covered)), "all non-legend foreground must belong to marked rectangles"
    return sorted(regions, key=lambda item: item["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each marked solid rectangle according to the geometry shown in the corner legend.

    Input structure:
    - The background is grey.
    - A corner legend has three marker-labelled entries. Each entry shows a small source rectangle and its transformed target rectangle.
    - Away from the legend are several non-overlapping solid rectangles of distinct colors, each with exactly one internal marker pixel colored yellow, magenta, or azure.

    Rule:
    - Decode the corner legend by comparing each source example with its target example: yellow maps to horizontal reflection, magenta maps to clockwise 90-degree rotation, and azure maps to vertical reflection.
    - For every colored rectangle, use its internal marker color as the lookup key and transform the whole rectangle patch, including the marker pixel.
    - Clear the legend to grey. Clear each original rectangle, then repaint its transformed patch with the original rectangle center as the anchor.

    Key insight:
    - The marker color inside a rectangle is not the operation itself; it is a key into the visual legend, and the resulting transform is applied in the rectangle's own coordinate frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    legend_box, mapping = _find_legend(g)
    regions = _find_regions(g, legend_box)

    out = g.copy()
    _clear_box(out, legend_box)
    for region in regions:
        _clear_box(out, region["bbox"])

    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        patch = region["patch"]
        op = mapping[int(region["marker"])]
        transformed = _transform(patch, op)
        nr, nc = _centered_top_left(r0, c0, patch.shape[0], patch.shape[1], transformed.shape[0], transformed.shape[1])
        tr, tc = transformed.shape
        assert 0 <= nr and 0 <= nc and nr + tr <= h and nc + tc <= w, "transformed rectangle must stay in bounds"
        assert bool(np.all(out[nr : nr + tr, nc : nc + tc] == BG)), "transformed rectangles must not overlap"
        out[nr : nr + tr, nc : nc + tc] = transformed

    return out


# --------------- generator ---------------

BODY_COLORS = tuple(c for c in range(10) if c not in {BG, *MARKER_COLORS})
GRID_SIZE_RANGE = (22, 30)
OBJECT_COUNT_RANGE = (3, 5)
RECT_SIZE_RANGE = (3, 7)
PLACEMENT_PAD = 1
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 300
DEMO_DOT_POSITIONS = ((0, 0), (0, 2), (1, 0), (1, 2))


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _legend_origin(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - LEGEND_W
    if corner == "bl":
        return h - LEGEND_H, 0
    if corner == "br":
        return h - LEGEND_H, w - LEGEND_W
    raise ValueError(corner)


def _random_demo_source(fill: int, dot: int) -> np.ndarray:
    src = np.full((2, 3), fill, dtype=int)
    r, c = random.choice(DEMO_DOT_POSITIONS)
    src[r, c] = dot
    return src


def _make_legend() -> np.ndarray:
    patch = np.full((LEGEND_H, LEGEND_W), BG, dtype=int)
    fill, dot = random.sample(BODY_COLORS, 2)
    row_markers = list(MARKER_COLORS)
    random.shuffle(row_markers)

    for band_top, marker in zip(LEGEND_BAND_TOPS, row_markers):
        source = _random_demo_source(fill, dot)
        target = _transform(source, LEGEND_OPS[int(marker)])
        patch[band_top + 1, 0] = int(marker)
        patch[band_top : band_top + source.shape[0], 2 : 2 + source.shape[1]] = source
        patch[band_top : band_top + target.shape[0], 7 : 7 + target.shape[1]] = target
    return patch


def _random_rect_size(op: str) -> tuple[int, int]:
    lo, hi = RECT_SIZE_RANGE
    if op == ROTCW:
        rectangular = [(h, w) for h in range(lo, hi + 1) for w in range(lo, hi + 1) if h % 2 == w % 2 and h != w]
        square = [(h, h) for h in range(lo, hi + 1)]
        if random.random() < 0.8 and rectangular:
            return random.choice(rectangular)
        return random.choice(square)
    return random.randint(lo, hi), random.randint(lo, hi)


def _new_bbox_for_rect(r0: int, c0: int, h: int, w: int, op: str) -> tuple[int, int, int, int]:
    dummy = np.zeros((h, w), dtype=int)
    transformed = _transform(dummy, op)
    nr, nc = _centered_top_left(r0, c0, h, w, transformed.shape[0], transformed.shape[1])
    return nr, nc, nr + transformed.shape[0] - 1, nc + transformed.shape[1] - 1


def _make_object_patch(h: int, w: int, color: int, marker: int) -> np.ndarray:
    patch = np.full((h, w), int(color), dtype=int)
    mr = random.randint(1, h - 2)
    mc = random.randint(1, w - 2)
    patch[mr, mc] = int(marker)
    return patch


def _sample_specs() -> list[dict]:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(BODY_COLORS, count)
    markers = list(MARKER_COLORS)
    while len(markers) < count:
        markers.append(random.choice(MARKER_COLORS))
    random.shuffle(markers)

    specs = []
    for color, marker in zip(colors, markers):
        op = LEGEND_OPS[int(marker)]
        h, w = _random_rect_size(op)
        specs.append(
            {
                "color": int(color),
                "marker": int(marker),
                "op": op,
                "height": int(h),
                "width": int(w),
            }
        )
    random.shuffle(specs)
    return specs


def _place_specs(h: int, w: int, legend_box: tuple[int, int, int, int], specs: list[dict]) -> list[dict] | None:
    occupied = [_expand_box(legend_box, PLACEMENT_PAD, h, w)]
    placed: list[dict] = []

    for spec in specs:
        sh = int(spec["height"])
        sw = int(spec["width"])
        op = str(spec["op"])
        candidates = []
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, h - sh)
            c0 = random.randint(0, w - sw)
            old_box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            try:
                new_box = _new_bbox_for_rect(r0, c0, sh, sw, op)
            except AssertionError:
                continue
            if new_box[0] < 0 or new_box[1] < 0 or new_box[2] >= h or new_box[3] >= w:
                continue

            footprint = _expand_box(_union_box(old_box, new_box), PLACEMENT_PAD, h, w)
            if any(_boxes_overlap(footprint, box) for box in occupied):
                continue
            candidates.append((old_box, new_box, footprint))

        if not candidates:
            return None
        old_box, new_box, footprint = random.choice(candidates)
        spec = dict(spec)
        spec["bbox"] = old_box
        spec["new_bbox"] = new_box
        spec["patch"] = _make_object_patch(sh, sw, int(spec["color"]), int(spec["marker"]))
        placed.append(spec)
        occupied.append(footprint)

    return placed


def _render_input(h: int, w: int, legend: np.ndarray, legend_box: tuple[int, int, int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    lr0, lc0, _lr1, _lc1 = legend_box
    grid[lr0 : lr0 + LEGEND_H, lc0 : lc0 + LEGEND_W] = legend
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["patch"]
    return grid


def _expected_output(h: int, w: int, specs: list[dict]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for spec in sorted(specs, key=lambda item: item["bbox"]):
        transformed = _transform(spec["patch"], str(spec["op"]))
        r0, c0, r1, c1 = spec["new_bbox"]
        assert transformed.shape == (r1 - r0 + 1, c1 - c0 + 1)
        assert bool(np.all(out[r0 : r1 + 1, c0 : c1 + 1] == BG))
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    corner = random.choice(("tl", "tr", "bl", "br"))
    lr0, lc0 = _legend_origin(h, w, corner)
    legend_box = (lr0, lc0, lr0 + LEGEND_H - 1, lc0 + LEGEND_W - 1)

    specs = _sample_specs()
    placed = _place_specs(h, w, legend_box, specs)
    if placed is None:
        return None

    legend = _make_legend()
    inp = _render_input(h, w, legend, legend_box, placed)
    expected = _expected_output(h, w, placed)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1030756833/9/3 input")
