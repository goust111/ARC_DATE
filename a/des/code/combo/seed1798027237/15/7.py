from __future__ import annotations

from collections import deque

import numpy as np


BG = 8
FRAME = 5
MARKER = 9
OBJECT_COLORS = (0, 1, 2, 3, 4, 6, 7)

GEN_ATTEMPTS = 400
GEN_OBJECT_ATTEMPTS = 200
GEN_MAX_HEIGHT = 36
GEN_MAX_WIDTH = 36


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    return len(_mask_components(mask)) == 1


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    open_bg = ~mask
    padded = np.pad(open_bg, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    ph, pw = padded.shape
    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = open_bg & ~seen[1 : h + 1, 1 : w + 1]
    return len(_mask_components(holes))


def _scale_mask(mask: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    src_h, src_w = mask.shape
    out = np.zeros((dst_h, dst_w), dtype=bool)
    for sr in range(src_h):
        r0 = (sr * dst_h) // src_h
        r1 = ((sr + 1) * dst_h + src_h - 1) // src_h
        for sc in range(src_w):
            if not mask[sr, sc]:
                continue
            c0 = (sc * dst_w) // src_w
            c1 = ((sc + 1) * dst_w + src_w - 1) // src_w
            out[r0:r1, c0:c1] = True
    return out


def _find_marker_segments(grid: np.ndarray) -> dict[tuple[int, int], int]:
    segments: dict[tuple[int, int], int] = {}
    for comp in _mask_components(grid == MARKER):
        r0, c0, r1, c1 = _bbox(comp)
        assert r0 == r1, "marker must be a horizontal line"
        assert len(comp) == c1 - c0 + 1, "marker must be contiguous"
        key = (r1, c1)
        assert key not in segments, "markers must be unique"
        segments[key] = c1 - c0 + 1
    return segments


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    markers = _find_marker_segments(grid)
    frames: list[dict[str, object]] = []
    for comp in _mask_components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(comp) != int(border.sum()) or not np.array_equal(sub == FRAME, border):
            continue
        key = (r0 - 1, c0 - 1)
        assert key in markers, "each frame must have a top-left marker"
        marker_len = markers[key]
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "marker_len": marker_len,
            }
        )

    lengths = [int(frame["marker_len"]) for frame in frames]
    assert len(lengths) == len(set(lengths)), "marker lengths must be unique"
    assert frames, "expected at least one frame"
    return frames


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, FRAME, MARKER):
            continue
        for comp in _mask_components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
            objects.append(
                {
                    "color": color,
                    "coords": comp,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "holes": _count_holes(mask),
                }
            )
    assert objects, "expected at least one foreground object"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Match each object to the frame whose maroon marker length equals its hole count."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    frame_by_holes = {int(frame["marker_len"]): frame for frame in frames}
    objects = _find_objects(g)

    out = g.copy()
    used_lengths: set[int] = set()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
        hole_count = int(obj["holes"])
        if hole_count == 0:
            continue
        assert hole_count in frame_by_holes, "missing frame for positive hole count"
        assert hole_count not in used_lengths, "positive hole counts must map uniquely"
        used_lengths.add(hole_count)
        ir0, ic0, ir1, ic1 = frame_by_holes[hole_count]["interior"]
        scaled = _scale_mask(obj["mask"], ir1 - ir0 + 1, ic1 - ic0 + 1)
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
        out[ir0 : ir1 + 1, ic0 : ic1 + 1][scaled] = int(obj["color"])
    return out


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    return out.copy()


def _add_tabs(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pad = 2
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask
    tab_count = int(rng.integers(0, 3))
    for _ in range(tab_count):
        coords = np.argwhere(canvas)
        r0, c0, r1, c1 = _bbox(coords)
        side = int(rng.integers(0, 4))
        if side == 0 and r0 > 0:
            length = int(rng.integers(1, max(2, (c1 - c0 + 2) // 2)))
            start = int(rng.integers(c0, c1 - length + 2))
            height = 1
            canvas[r0 - height : r0, start : start + length] = True
        elif side == 1 and r1 + 1 < canvas.shape[0]:
            length = int(rng.integers(1, max(2, (c1 - c0 + 2) // 2)))
            start = int(rng.integers(c0, c1 - length + 2))
            height = 1
            canvas[r1 + 1 : r1 + 1 + height, start : start + length] = True
        elif side == 2 and c0 > 0:
            length = int(rng.integers(1, max(2, (r1 - r0 + 2) // 2)))
            start = int(rng.integers(r0, r1 - length + 2))
            width = 1
            canvas[start : start + length, c0 - width : c0] = True
        elif side == 3 and c1 + 1 < canvas.shape[1]:
            length = int(rng.integers(1, max(2, (r1 - r0 + 2) // 2)))
            start = int(rng.integers(r0, r1 - length + 2))
            width = 1
            canvas[start : start + length, c1 + 1 : c1 + 1 + width] = True
    return _crop_mask(canvas)


def _gen_zero_hole_object(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(GEN_OBJECT_ATTEMPTS):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        target = int(rng.integers(max(4, (h * w) // 2), h * w + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True
        mask = _random_transform(_crop_mask(mask), rng)
        if min(mask.shape) < 2 or max(mask.shape) > 6:
            continue
        if _is_connected(mask) and _count_holes(mask) == 0:
            return mask
    return None


def _gen_holed_object(hole_count: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(GEN_OBJECT_ATTEMPTS):
        horizontal = bool(rng.integers(0, 2))
        hole_heights = [int(rng.integers(1, 3)) for _ in range(hole_count)]
        hole_widths = [int(rng.integers(1, 3)) for _ in range(hole_count)]

        if horizontal:
            band_h = max(hole_heights)
            outer_h = band_h + 2
            outer_w = sum(hole_widths) + hole_count + 1
            mask = np.ones((outer_h, outer_w), dtype=bool)
            x = 1
            for hh, ww in zip(hole_heights, hole_widths, strict=False):
                y = 1 + int(rng.integers(0, band_h - hh + 1))
                mask[y : y + hh, x : x + ww] = False
                x += ww + 1
        else:
            band_w = max(hole_widths)
            outer_h = sum(hole_heights) + hole_count + 1
            outer_w = band_w + 2
            mask = np.ones((outer_h, outer_w), dtype=bool)
            y = 1
            for hh, ww in zip(hole_heights, hole_widths, strict=False):
                x = 1 + int(rng.integers(0, band_w - ww + 1))
                mask[y : y + hh, x : x + ww] = False
                y += hh + 1

        mask = _add_tabs(mask, rng)
        mask = _random_transform(mask, rng)
        if min(mask.shape) < 3 or max(mask.shape) > 7:
            continue
        if _is_connected(mask) and _count_holes(mask) == hole_count:
            return mask
    return None


def _pack_items(
    sizes: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], int, int]:
    assert sizes
    min_gap = 2
    best_rows: tuple[list[int], list[int]] | None = None
    best_score: tuple[int, int, int] | None = None
    n = len(sizes)

    for bits in range(1 << n):
        rows = ([], [])
        for i in range(n):
            rows[(bits >> i) & 1].append(i)
        if not rows[0]:
            continue

        widths = []
        heights = []
        for row in rows:
            if row:
                widths.append(sum(sizes[i][1] for i in row) + min_gap * (len(row) - 1))
                heights.append(max(sizes[i][0] for i in row))
            else:
                widths.append(0)
                heights.append(0)
        total_h = heights[0] + (2 if rows[1] else 0) + heights[1]
        score = (max(widths), total_h, abs(len(rows[0]) - len(rows[1])))
        if best_score is None or score < best_score:
            best_score = score
            best_rows = (rows[0][:], rows[1][:])

    assert best_rows is not None

    placements: list[tuple[int, int] | None] = [None] * n
    zone_w = 0
    y = 0
    for row_idx, row in enumerate(best_rows):
        if not row:
            continue
        row_h = max(sizes[i][0] for i in row)
        x = 0
        for pos, idx in enumerate(row):
            h, w = sizes[idx]
            y_offset = int(rng.integers(0, row_h - h + 1))
            placements[idx] = (y + y_offset, x)
            x += w
            if pos < len(row) - 1:
                x += int(rng.integers(2, 5))
        zone_w = max(zone_w, x)
        y += row_h
        if row_idx == 0 and best_rows[1]:
            y += int(rng.integers(2, 4))
    zone_h = y
    return [pos for pos in placements if pos is not None], zone_h, zone_w


def _draw_object(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    rows = top + coords[:, 0]
    cols = left + coords[:, 1]
    grid[rows, cols] = color
    return np.column_stack([rows, cols]).astype(int)


def _draw_frame(
    grid: np.ndarray,
    marker_len: int,
    interior_shape: tuple[int, int],
    top: int,
    left: int,
) -> tuple[int, int, int, int]:
    total_h = interior_shape[0] + 3
    total_w = interior_shape[1] + 2 + marker_len
    assert total_h >= 4 and total_w >= marker_len + 3

    marker_row = top
    marker_c0 = left
    frame_r0 = top + 1
    frame_c0 = left + marker_len
    frame_r1 = frame_r0 + interior_shape[0] + 1
    frame_c1 = frame_c0 + interior_shape[1] + 1

    grid[marker_row, marker_c0 : marker_c0 + marker_len] = MARKER
    grid[frame_r0 : frame_r1 + 1, frame_c0] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c1] = FRAME
    grid[frame_r0, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r1, frame_c0 : frame_c1 + 1] = FRAME

    assert frame_r1 - frame_r0 + 1 == total_h - 1
    assert frame_c1 - frame_c0 + 1 == total_w - marker_len
    return frame_r0 + 1, frame_c0 + 1, frame_r1 - 1, frame_c1 - 1


def _build_expected(
    inp: np.ndarray,
    objects: list[dict[str, object]],
    frame_interiors: dict[int, tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
        holes = int(obj["holes"])
        if holes == 0:
            continue
        ir0, ic0, ir1, ic1 = frame_interiors[holes]
        scaled = _scale_mask(obj["mask"], ir1 - ir0 + 1, ic1 - ic0 + 1)
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
        out[ir0 : ir1 + 1, ic0 : ic1 + 1][scaled] = int(obj["color"])
    return out


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    positive_count = int(rng.integers(1, 4))
    zero_count = int(rng.integers(1, 3))
    positive_holes = list(rng.choice(np.array([1, 2, 3]), size=positive_count, replace=False))
    rng.shuffle(positive_holes)

    colors = list(rng.choice(np.array(OBJECT_COLORS), size=positive_count + zero_count, replace=False))
    rng.shuffle(colors)

    objects_data: list[dict[str, object]] = []
    for hole_count in positive_holes:
        mask = _gen_holed_object(int(hole_count), rng)
        if mask is None:
            return None
        objects_data.append({"holes": int(hole_count), "mask": mask})
    for _ in range(zero_count):
        mask = _gen_zero_hole_object(rng)
        if mask is None:
            return None
        objects_data.append({"holes": 0, "mask": mask})
    rng.shuffle(objects_data)

    for obj, color in zip(objects_data, colors, strict=False):
        obj["color"] = int(color)

    frame_lengths = set(int(h) for h in positive_holes)
    if len(frame_lengths) < 4 and bool(rng.integers(0, 2)):
        extras = [v for v in (1, 2, 3, 4) if v not in frame_lengths]
        if extras:
            frame_lengths.add(int(rng.choice(np.array(extras))))

    frames_data: list[dict[str, object]] = []
    for marker_len in sorted(frame_lengths):
        matching = [obj for obj in objects_data if int(obj["holes"]) == marker_len]
        if matching:
            src_mask = matching[0]["mask"]
            scale = 3 if max(src_mask.shape) <= 4 and bool(rng.integers(0, 2)) else 2
            interior_shape = (src_mask.shape[0] * scale, src_mask.shape[1] * scale)
        else:
            interior_shape = (int(rng.integers(4, 8)), int(rng.integers(4, 8)))
        frames_data.append(
            {
                "marker_len": int(marker_len),
                "interior_shape": interior_shape,
                "size": (interior_shape[0] + 3, interior_shape[1] + 2 + int(marker_len)),
            }
        )
    rng.shuffle(frames_data)

    object_sizes = [tuple(map(int, obj["mask"].shape)) for obj in objects_data]
    frame_sizes = [tuple(map(int, frame["size"])) for frame in frames_data]

    object_positions, obj_zone_h, obj_zone_w = _pack_items(object_sizes, rng)
    frame_positions, frame_zone_h, frame_zone_w = _pack_items(frame_sizes, rng)

    half_h = max(obj_zone_h, frame_zone_h) + int(rng.integers(1, 4))
    sep_gap = int(rng.integers(2, 5))
    usable_w = max(obj_zone_w, frame_zone_w) + int(rng.integers(2, 6))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    grid_h = 2 * half_h + sep_gap
    grid_w = usable_w + left_margin + right_margin

    if grid_h > GEN_MAX_HEIGHT or grid_w > GEN_MAX_WIDTH:
        return None

    top_y0 = int(rng.integers(0, half_h - obj_zone_h + 1))
    bottom_y0 = half_h + sep_gap + int(rng.integers(0, half_h - frame_zone_h + 1))
    top_x0 = left_margin + int(rng.integers(0, usable_w - obj_zone_w + 1))
    bottom_x0 = left_margin + int(rng.integers(0, usable_w - frame_zone_w + 1))

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    rendered_objects: list[dict[str, object]] = []
    for obj, (dr, dc) in zip(objects_data, object_positions, strict=False):
        top = top_y0 + dr
        left = top_x0 + dc
        coords = _draw_object(inp, obj["mask"], int(obj["color"]), top, left)
        rendered_objects.append(
            {
                "color": int(obj["color"]),
                "mask": obj["mask"],
                "holes": int(obj["holes"]),
                "coords": coords,
            }
        )

    frame_interiors: dict[int, tuple[int, int, int, int]] = {}
    for frame, (dr, dc) in zip(frames_data, frame_positions, strict=False):
        top = bottom_y0 + dr
        left = bottom_x0 + dc
        interior = _draw_frame(inp, int(frame["marker_len"]), frame["interior_shape"], top, left)
        frame_interiors[int(frame["marker_len"])] = interior

    if len(frame_interiors) != len(frames_data):
        return None

    expected = _build_expected(inp, rendered_objects, frame_interiors)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
