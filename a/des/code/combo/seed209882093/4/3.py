from __future__ import annotations

import itertools

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _line_info(coords: np.ndarray) -> dict[str, object] | None:
    coords = np.asarray(coords, dtype=int)
    rows = coords[:, 0]
    cols = coords[:, 1]
    area = int(coords.shape[0])

    if np.all(rows == rows[0]):
        c0 = int(cols.min())
        c1 = int(cols.max())
        if c1 - c0 + 1 != area:
            return None
        return {"orientation": "h", "length": area, "coords": coords}

    if np.all(cols == cols[0]):
        r0 = int(rows.min())
        r1 = int(rows.max())
        if r1 - r0 + 1 != area:
            return None
        return {"orientation": "v", "length": area, "coords": coords}

    return None


def _extract_objects(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    markers: list[dict[str, object]] = []
    borders: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == int(color))
        if len(comps) == 1 and int(comps[0].shape[0]) == 1:
            r, c = map(int, comps[0][0])
            markers.append({"color": int(color), "pos": (r, c)})
            continue

        if len(comps) != 4:
            raise AssertionError(f"color {color} should have exactly four line fragments")

        infos = [_line_info(comp) for comp in comps]
        if any(info is None for info in infos):
            raise AssertionError(f"color {color} is not made of straight line segments")

        horizontals = [info for info in infos if info["orientation"] == "h"]
        verticals = [info for info in infos if info["orientation"] == "v"]
        if len(horizontals) != 2 or len(verticals) != 2:
            raise AssertionError(f"color {color} should have two horizontal and two vertical segments")

        hlen = int(horizontals[0]["length"])
        vlen = int(verticals[0]["length"])
        if any(int(info["length"]) != hlen for info in horizontals):
            raise AssertionError(f"horizontal lengths for color {color} do not match")
        if any(int(info["length"]) != vlen for info in verticals):
            raise AssertionError(f"vertical lengths for color {color} do not match")
        if hlen != vlen + 2:
            raise AssertionError(f"color {color} does not satisfy frame side-length constraint")
        if hlen < 5 or hlen % 2 == 0:
            raise AssertionError(f"color {color} must correspond to an odd-sized frame")

        all_coords = np.concatenate([np.asarray(info["coords"], dtype=int) for info in infos], axis=0)
        centroid = all_coords.mean(axis=0)
        borders.append(
            {
                "border_color": int(color),
                "size": int(hlen),
                "centroid": (float(centroid[0]), float(centroid[1])),
            }
        )

    if len(markers) != len(borders):
        raise AssertionError("marker count must match frame count")

    borders.sort(key=lambda item: (item["centroid"][0], item["centroid"][1]))
    markers.sort(key=lambda item: (item["pos"][0], item["pos"][1]))
    return borders, markers, bg


def _assign_markers(
    borders: list[dict[str, object]], markers: list[dict[str, object]]
) -> list[tuple[dict[str, object], dict[str, object]]]:
    n = len(borders)
    if n == 0:
        return []

    best_cost: float | None = None
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(n)):
        cost = 0.0
        for i, j in enumerate(perm):
            cr, cc = borders[i]["centroid"]
            mr, mc = markers[j]["pos"]
            dr = float(cr) - float(mr)
            dc = float(cc) - float(mc)
            cost += dr * dr + dc * dc
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm

    assert best_perm is not None
    return [(borders[i], markers[j]) for i, j in enumerate(best_perm)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color line fragments into square frames centered on nearby
    marker pixels, then fill each frame interior with the marker color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    borders, markers, bg = _extract_objects(g)

    out = np.full((h, w), bg, dtype=int)
    for border, marker in _assign_markers(borders, markers):
        size = int(border["size"])
        half = size // 2
        mr, mc = marker["pos"]
        r0 = int(mr - half)
        c0 = int(mc - half)
        r1 = int(mr + half)
        c1 = int(mc + half)
        if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
            raise AssertionError("reconstructed frame would leave the grid")

        border_color = int(border["border_color"])
        fill_color = int(marker["color"])
        out[r0, c0 : c1 + 1] = border_color
        out[r1, c0 : c1 + 1] = border_color
        out[r0 : r1 + 1, c0] = border_color
        out[r0 : r1 + 1, c1] = border_color
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _frame_cells(center: tuple[int, int], size: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mr, mc = map(int, center)
    half = size // 2
    r0, c0 = mr - half, mc - half
    r1, c1 = mr + half, mc + half

    border = set()
    for c in range(c0, c1 + 1):
        border.add((int(r0), int(c)))
        border.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        border.add((int(r), int(c0)))
        border.add((int(r), int(c1)))

    interior = [(int(r), int(c)) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return sorted(border), interior


def _scattered_segment_cells(
    center: tuple[int, int], size: int, gap_h: int, gap_v: int, shift_h: int, shift_v: int
) -> list[list[tuple[int, int]]]:
    mr, mc = map(int, center)
    half = size // 2
    vlen = size - 2
    vhalf = vlen // 2

    top_row = mr - (half + gap_h)
    bottom_row = mr + (half + gap_h)
    left_col = mc - (half + gap_v)
    right_col = mc + (half + gap_v)

    top_mid = mc + shift_h
    bottom_mid = mc - shift_h
    left_mid = mr + shift_v
    right_mid = mr - shift_v

    top = [(int(top_row), int(c)) for c in range(top_mid - half, top_mid + half + 1)]
    bottom = [(int(bottom_row), int(c)) for c in range(bottom_mid - half, bottom_mid + half + 1)]
    left = [(int(r), int(left_col)) for r in range(left_mid - vhalf, left_mid + vhalf + 1)]
    right = [(int(r), int(right_col)) for r in range(right_mid - vhalf, right_mid + vhalf + 1)]
    return [top, bottom, left, right]


def _tile_boxes(
    rows: int, cols: int, tile_h: int, tile_w: int, gap_r: int, gap_c: int, margin: tuple[int, int, int, int]
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    top, bottom, left, right = map(int, margin)
    boxes: list[tuple[int, int, int, int]] = []
    for rr in range(rows):
        for cc in range(cols):
            r0 = top + rr * (tile_h + gap_r)
            c0 = left + cc * (tile_w + gap_c)
            boxes.append((int(r0), int(c0), int(r0 + tile_h - 1), int(c0 + tile_w - 1)))
    height = top + rows * tile_h + (rows - 1) * gap_r + bottom
    width = left + cols * tile_w + (cols - 1) * gap_c + right
    return boxes, (int(height), int(width))


def _sample_object(
    rng: np.random.Generator, tile: tuple[int, int, int, int], side_pool: tuple[int, ...]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = map(int, tile)
    tile_h = r1 - r0 + 1
    tile_w = c1 - c0 + 1

    candidates: list[tuple[int, int]] = []
    for size in side_pool:
        half = size // 2
        max_gap = min((tile_h - 1) // 2 - half, (tile_w - 1) // 2 - half, 4)
        if max_gap >= 2:
            candidates.append((int(size), int(max_gap)))
    if not candidates:
        return None

    size, max_gap = candidates[int(rng.integers(0, len(candidates)))]
    gap_h = int(rng.integers(2, max_gap + 1))
    gap_v = int(rng.integers(2, max_gap + 1))
    shift_h = int(rng.integers(-(gap_v - 1), gap_v))
    shift_v = int(rng.integers(-(gap_h - 1), gap_h))
    if gap_h == gap_v and shift_h == 0 and shift_v == 0:
        shift_h = 1 if gap_v > 1 else 0

    radius = size // 2 + max(gap_h, gap_v)
    row_lo = r0 + radius
    row_hi = r1 - radius
    col_lo = c0 + radius
    col_hi = c1 - radius
    if row_lo > row_hi or col_lo > col_hi:
        return None

    center = (
        int(rng.integers(row_lo, row_hi + 1)),
        int(rng.integers(col_lo, col_hi + 1)),
    )
    segments = _scattered_segment_cells(center, size, gap_h, gap_v, shift_h, shift_v)
    input_cells = sorted({cell for segment in segments for cell in segment} | {center})
    frame_border, frame_fill = _frame_cells(center, size)
    output_cells = frame_border + frame_fill

    for cells in segments:
        info = _line_info(np.asarray(cells, dtype=int))
        if info is None:
            return None

    for cells in (input_cells, output_cells):
        for rr, cc in cells:
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                return None

    return {
        "center": center,
        "size": int(size),
        "segments": segments,
        "frame_border": frame_border,
        "frame_fill": frame_fill,
    }


def _make_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = int(rng.integers(2, 5))
    if n_objects == 2:
        rows, cols = ((1, 2) if int(rng.integers(0, 2)) == 0 else (2, 1))
        tile_h = int(rng.integers(14, 18))
        tile_w = int(rng.integers(14, 18))
        side_pool = (5, 7, 9)
    elif n_objects == 3:
        rows, cols = (2, 2)
        tile_h = int(rng.integers(13, 17))
        tile_w = int(rng.integers(13, 17))
        side_pool = (5, 7, 9)
    else:
        rows, cols = (2, 2)
        tile_h = int(rng.integers(13, 16))
        tile_w = int(rng.integers(13, 16))
        side_pool = (5, 7)

    gap_r = int(rng.integers(1, 4))
    gap_c = int(rng.integers(1, 4))
    margin = (
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
        int(rng.integers(1, 4)),
    )
    tiles, shape = _tile_boxes(rows, cols, tile_h, tile_w, gap_r, gap_c, margin)
    tile_indices = [int(i) for i in rng.choice(len(tiles), size=n_objects, replace=False)]
    chosen_tiles = [tiles[i] for i in tile_indices]

    if 2 * n_objects > 9:
        return None
    palette = [int(v) for v in rng.choice(np.arange(1, 10), size=2 * n_objects, replace=False)]
    border_colors = palette[:n_objects]
    fill_colors = palette[n_objects:]

    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    for tile, border_color, fill_color in zip(chosen_tiles, border_colors, fill_colors):
        obj = _sample_object(rng, tile, side_pool)
        if obj is None:
            return None

        center = obj["center"]
        for segment in obj["segments"]:
            _paint(inp, segment, border_color)
        _paint(inp, [center], fill_color)

        _paint(out, obj["frame_fill"], fill_color)
        _paint(out, obj["frame_border"], border_color)

    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        example = _make_example(rng)
        if example is None:
            continue
        inp, out = example
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
