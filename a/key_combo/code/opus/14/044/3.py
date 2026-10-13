from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4)
RING_COUNT = 4
RING_GAP = 2
RING_STEP = RING_GAP + 1
PATTERNS = (
    (0, 1, 1, 3),
    (0, 0, 2, 3),
    (0, 1, 2, 2),
    (0, 0, 1, 3),
    (0, 1, 3, 3),
)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_pixels(box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return np.array(sorted(set(cells)), dtype=int)


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    box = _bbox(coords)
    expected = _ring_pixels(box)
    if coords.shape != expected.shape:
        return False
    return set(map(tuple, coords)) == set(map(tuple, expected))


def _extract_rings(grid: np.ndarray) -> list[dict]:
    rings = []
    for coords in _components(grid == GREY):
        if not _is_rectangular_ring(coords):
            continue
        box = _bbox(coords)
        rings.append(
            {
                "coords": coords,
                "box": box,
                "area": (box[2] - box[0] + 1) * (box[3] - box[1] + 1),
            }
        )
    assert len(rings) == RING_COUNT, "expected exactly four grey rectangular rings"
    rings.sort(key=lambda ring: (-ring["area"], ring["box"]))
    return rings


def _extract_markers(grid: np.ndarray) -> list[dict]:
    markers: list[dict] = []
    marker_mask = (grid != BG) & (grid != GREY)
    for coords in _components(marker_mask):
        assert coords.shape[0] == 1, "markers must be single pixels"
        r, c = map(int, coords[0])
        markers.append({"r": r, "c": c, "color": int(grid[r, c])})
    assert len(markers) == len(MARKER_COLORS), "expected four marker pixels"
    assert sorted(marker["color"] for marker in markers) == list(MARKER_COLORS)
    return markers


def _min_sqdist_to_coords(r: int, c: int, coords: np.ndarray) -> int:
    dr = coords[:, 0] - int(r)
    dc = coords[:, 1] - int(c)
    return int(np.min(dr * dr + dc * dc))


def _marker_signature(marker: dict, rings: list[dict]) -> tuple[int, int]:
    dists = [_min_sqdist_to_coords(marker["r"], marker["c"], ring["coords"]) for ring in rings]
    best = min(dists)
    assert dists.count(best) == 1, "nearest ring must be unique"
    return int(dists.index(best)), int(best)


def _paint_ring(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_output(shape: tuple[int, int], ring_boxes: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for box, color in zip(ring_boxes, colors, strict=True):
        _paint_ring(out, box, int(color))
    return out


def _direct_nearest_shortcut(shape: tuple[int, int], ring_boxes: list[tuple[int, int, int, int]], markers: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    by_layer: dict[int, list[dict]] = {}
    for marker in markers:
        by_layer.setdefault(int(marker["layer"]), []).append(marker)
    for layer, box in enumerate(ring_boxes):
        chosen = None
        if layer in by_layer:
            chosen = min(by_layer[layer], key=lambda item: (item["dist2"], item["r"], item["c"]))
            color = int(chosen["color"])
        else:
            color = GREY
        _paint_ring(out, box, color)
    return out


def _sorted_recolor(shape: tuple[int, int], ring_boxes: list[tuple[int, int, int, int]], markers: list[dict], key: tuple[str, ...]) -> np.ndarray:
    order = sorted(markers, key=lambda marker: tuple(marker[name] for name in key))
    colors = [int(marker["color"]) for marker in order]
    return _render_output(shape, ring_boxes, colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor concentric rectangular rings using the outer-to-inner order induced by marker-to-ring proximity.

    Input structure:
    - The grid has a black background, four nested one-cell-thick grey rectangular rings, and four detached single-pixel markers.
    - The grey rings are concentric and separated by black gaps; the four marker colors are blue, red, green, and yellow.
    - Every marker is off the rings, and its nearest ring layer is uniquely determined by minimum Euclidean distance to that layer's pixels.

    Rule:
    - For each marker, compute its minimum Euclidean distance to each grey ring layer and identify the unique nearest layer.
    - Sort the markers first by that nearest layer from outer to inner; if two markers share the same nearest layer, sort those ties by the smaller minimum distance to that layer.
    - Recolor the four grey rings from outer to inner using the sorted marker-color sequence, then erase all marker pixels back to black.

    Key insight:
    - The markers do not directly paint the rings they sit nearest to; instead, their proximity ranking encodes the global outside-to-inside recoloring order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    rings = _extract_rings(g)
    markers = _extract_markers(g)

    enriched = []
    for marker in markers:
        layer, dist2 = _marker_signature(marker, rings)
        enriched.append({**marker, "layer": layer, "dist2": dist2})

    ordered = sorted(enriched, key=lambda item: (item["layer"], item["dist2"]))
    colors = [int(item["color"]) for item in ordered]
    out = _render_output(g.shape, [ring["box"] for ring in rings], colors)
    return out


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[dict]] | None:
    gap = RING_GAP
    inner_h = int(rng.integers(5, 9))
    inner_w = int(rng.integers(5, 9))
    outer_h = inner_h + 6 * (gap + 1)
    outer_w = inner_w + 6 * (gap + 1)

    total_extra_h = int(rng.integers(4, max(5, 31 - outer_h)))
    total_extra_w = int(rng.integers(4, max(5, 31 - outer_w)))
    if outer_h + total_extra_h > 30 or outer_w + total_extra_w > 30:
        return None

    top = total_extra_h // 2
    left = total_extra_w // 2
    h = outer_h + total_extra_h
    w = outer_w + total_extra_w

    boxes: list[tuple[int, int, int, int]] = []
    for i in range(RING_COUNT):
        off = i * (gap + 1)
        boxes.append((top + off, left + off, top + outer_h - 1 - off, left + outer_w - 1 - off))

    grid = np.full((h, w), BG, dtype=int)
    for box in boxes:
        _paint_ring(grid, box, GREY)

    ring_coords = [{"box": box, "coords": _ring_pixels(box)} for box in boxes]
    candidates_by_layer: dict[int, list[dict]] = {layer: [] for layer in range(RING_COUNT)}
    ring_pixels = {tuple(cell) for ring in ring_coords for cell in ring["coords"]}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in ring_pixels:
                continue
            dists = [_min_sqdist_to_coords(r, c, ring["coords"]) for ring in ring_coords]
            best = min(dists)
            if dists.count(best) != 1:
                continue
            layer = int(dists.index(best))
            candidates_by_layer[layer].append({"r": r, "c": c, "layer": layer, "dist2": int(best)})

    pattern = PATTERNS[int(rng.integers(len(PATTERNS)))]
    pools = []
    for layer in pattern:
        pool = candidates_by_layer[layer][:]
        rng.shuffle(pool)
        pools.append(pool)

    chosen: list[dict] = []

    def far_enough(a: dict, b: dict) -> bool:
        return max(abs(a["r"] - b["r"]), abs(a["c"] - b["c"])) >= 2

    def backtrack(i: int) -> bool:
        if i == len(pattern):
            return True
        for cand in pools[i]:
            if any(cand["r"] == prev["r"] and cand["c"] == prev["c"] for prev in chosen):
                continue
            if any(not far_enough(cand, prev) for prev in chosen):
                continue
            if any(prev["layer"] == cand["layer"] and prev["dist2"] == cand["dist2"] for prev in chosen):
                continue
            chosen.append(cand)
            if backtrack(i + 1):
                return True
            chosen.pop()
        return False

    if not backtrack(0):
        return None

    rows = [item["r"] for item in chosen]
    cols = [item["c"] for item in chosen]
    if max(rows) - min(rows) < 4 or max(cols) - min(cols) < 4:
        return None

    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    markers: list[dict] = []
    for cand, color in zip(chosen, colors, strict=True):
        marker = {**cand, "color": int(color)}
        markers.append(marker)
        grid[marker["r"], marker["c"]] = marker["color"]

    ordered = sorted(markers, key=lambda item: (item["layer"], item["dist2"]))
    keys = [(item["layer"], item["dist2"]) for item in ordered]
    if len(set(keys)) != len(keys):
        return None
    if len({item["layer"] for item in markers}) == RING_COUNT:
        return None

    expected = _render_output(grid.shape, boxes, [item["color"] for item in ordered])
    wrong_direct = _direct_nearest_shortcut(grid.shape, boxes, markers)
    wrong_distance = _sorted_recolor(grid.shape, boxes, markers, ("dist2", "r", "c"))
    wrong_coords = _sorted_recolor(grid.shape, boxes, markers, ("r", "c"))
    if np.array_equal(expected, wrong_direct):
        return None
    if np.array_equal(expected, wrong_distance):
        return None
    if np.array_equal(expected, wrong_coords):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, boxes, markers


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        built = _build_case(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid concentric-ring task")
