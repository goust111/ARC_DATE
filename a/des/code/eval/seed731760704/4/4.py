from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0

# Fixed visible colors used by the generator only.
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

# Generator ranges.
TOP_H_RANGE = (11, 16)
GRID_W_RANGE = (22, 30)
OUTPUT_H_RANGE = (9, 15)
OUTPUT_W_RANGE = (9, 15)
FRAME_MARGIN_RANGE = (1, 3)
BOTTOM_EXTRA_H_RANGE = (2, 5)
SHAPE_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _extract_source_shape(top: np.ndarray, coords: np.ndarray) -> dict:
    vals = [int(top[r, c]) for r, c in coords]
    counts = Counter(vals)
    assert len(counts) == 2, "each source object must have one base color and one marker color"

    marker_colors = [color for color, count in counts.items() if count == 1]
    assert len(marker_colors) == 1, "marker color must be unique within each source object"
    marker_color = int(marker_colors[0])
    base_colors = [color for color, count in counts.items() if color != marker_color and count > 1]
    assert len(base_colors) == 1, "source object must have exactly one base color"
    base_color = int(base_colors[0])

    marker = None
    for r, c in coords:
        if int(top[r, c]) == marker_color:
            marker = (int(r), int(c))
            break
    assert marker is not None
    mr, mc = marker

    r0, c0, r1, c1 = _bbox(coords)
    sides = []
    if mr == r0:
        sides.append("up")
    if mr == r1:
        sides.append("down")
    if mc == c0:
        sides.append("left")
    if mc == c1:
        sides.append("right")
    assert len(sides) == 1, "marker must sit on exactly one outer side"
    direction = sides[0]

    stamp_offsets: list[tuple[int, int]] = []
    for r, c in coords:
        dr = int(r - mr)
        dc = int(c - mc)
        if direction in ("up", "down"):
            stamp_offsets.append((-dr, dc))
        else:
            stamp_offsets.append((dr, -dc))

    return {
        "color": base_color,
        "area": int(coords.shape[0]),
        "direction": direction,
        "stamp_offsets": stamp_offsets,
    }


def _find_frame(bottom: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(bottom):
        color = int(color)
        if color == BLACK:
            continue
        for coords in _components(bottom == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue
            perimeter = 2 * (h + w) - 4
            if int(coords.shape[0]) != perimeter:
                continue
            sub = bottom[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if np.any(sub[~border] == color):
                continue
            candidates.append((h * w, r0, c0, r1, c1))
    assert candidates, "expected one target frame in the lower region"
    candidates.sort(reverse=True)
    _area, r0, c0, r1, c1 = candidates[0]
    return int(r0), int(c0), int(r1), int(c1)


def _stamp_cells(offsets: list[tuple[int, int]], anchor: tuple[int, int], shape: tuple[int, int]) -> tuple[list[tuple[int, int]], bool]:
    h, w = shape
    ar, ac = anchor
    cells: list[tuple[int, int]] = []
    clipped = False
    for dr, dc in offsets:
        rr = int(ar + dr)
        cc = int(ac + dc)
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
        else:
            clipped = True
    return cells, clipped


def _render_output(shapes: list[dict], anchors: dict[int, tuple[int, int]], out_shape: tuple[int, int]) -> np.ndarray:
    out = np.full(out_shape, BLACK, dtype=int)
    for shape in sorted(shapes, key=lambda item: int(item["area"])):
        cells, _ = _stamp_cells(shape["stamp_offsets"], anchors[int(shape["color"])], out_shape)
        for r, c in cells:
            out[r, c] = int(shape["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the upper objects as stamps. The singleton marker inside each stamp
    encodes which side should face away from the target anchor. Match stamps to
    lower anchors by color, place them so marker and anchor coincide, and stamp
    them from smaller area to larger area so later stamps overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(1, g.shape[0] - 1) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != BLACK]
    assert len(sep_rows) == 1, "expected one full non-black separator row"
    sep = int(sep_rows[0])

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.size and bottom.size

    source_comps = _components(top != BLACK)
    assert source_comps, "expected source objects above the separator"
    shapes = [_extract_source_shape(top, coords) for coords in source_comps]

    fr0, fc0, fr1, fc1 = _find_frame(bottom)
    interior = bottom[fr0 + 1 : fr1, fc0 + 1 : fc1]
    out_shape = tuple(map(int, interior.shape))
    assert out_shape[0] > 0 and out_shape[1] > 0

    anchors: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(interior != BLACK):
        color = int(interior[r, c])
        assert color not in anchors, "each anchor color must appear exactly once"
        anchors[color] = (int(r), int(c))

    shape_colors = {int(shape["color"]) for shape in shapes}
    assert shape_colors == set(anchors), "source colors and anchor colors must match exactly"
    return _render_output(shapes, anchors, out_shape)


def _normalize(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = {(r - min_r, c - min_c) for r, c in coords}
    return norm, (int(min_r), int(min_c))


def _build_vertical_base() -> tuple[set[tuple[int, int]], tuple[int, int]]:
    height = random.randint(3, 7)
    top_half = random.choice((1, 2))
    left = -top_half
    right = top_half
    spans = [(left, right)]
    for _ in range(1, height):
        prev_l, prev_r = spans[-1]
        candidates = []
        for new_l in range(prev_l - 1, prev_l + 2):
            for new_r in range(prev_r - 1, prev_r + 2):
                if new_l > new_r:
                    continue
                width = new_r - new_l + 1
                if not 2 <= width <= 8:
                    continue
                if max(new_l, prev_l) > min(new_r, prev_r):
                    continue
                candidates.append((new_l, new_r))
        assert candidates
        spans.append(random.choice(candidates))

    coords = {(r, c) for r, (l, rr) in enumerate(spans) for c in range(l, rr + 1)}
    norm, (min_r, min_c) = _normalize(coords)
    marker = (0 - min_r, 0 - min_c)
    return norm, marker


def _build_horizontal_base() -> tuple[set[tuple[int, int]], tuple[int, int]]:
    width = random.randint(3, 7)
    left_half = random.choice((1, 2))
    top = -left_half
    bottom = left_half
    spans = [(top, bottom)]
    for _ in range(1, width):
        prev_t, prev_b = spans[-1]
        candidates = []
        for new_t in range(prev_t - 1, prev_t + 2):
            for new_b in range(prev_b - 1, prev_b + 2):
                if new_t > new_b:
                    continue
                height = new_b - new_t + 1
                if not 2 <= height <= 8:
                    continue
                if max(new_t, prev_t) > min(new_b, prev_b):
                    continue
                candidates.append((new_t, new_b))
        assert candidates
        spans.append(random.choice(candidates))

    coords = {(r, c) for c, (t, b) in enumerate(spans) for r in range(t, b + 1)}
    norm, (min_r, min_c) = _normalize(coords)
    marker = (0 - min_r, 0 - min_c)
    return norm, marker


def _flip_vertical(coords: set[tuple[int, int]], marker: tuple[int, int]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    max_r = max(r for r, _ in coords)
    flipped = {(max_r - r, c) for r, c in coords}
    return flipped, (int(max_r - marker[0]), int(marker[1]))


def _flip_horizontal(coords: set[tuple[int, int]], marker: tuple[int, int]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    max_c = max(c for _, c in coords)
    flipped = {(r, max_c - c) for r, c in coords}
    return flipped, (int(marker[0]), int(max_c - marker[1]))


def _make_shape(direction: str, color: int) -> dict:
    if direction == "up":
        coords, marker = _build_vertical_base()
    elif direction == "down":
        coords, marker = _build_vertical_base()
        coords, marker = _flip_vertical(coords, marker)
    elif direction == "left":
        coords, marker = _build_horizontal_base()
    elif direction == "right":
        coords, marker = _build_horizontal_base()
        coords, marker = _flip_horizontal(coords, marker)
    else:
        raise ValueError(direction)

    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in coords:
        patch[r, c] = int(color)
    patch[marker] = GREY

    stamp_offsets: list[tuple[int, int]] = []
    mr, mc = marker
    for r, c in coords:
        dr = int(r - mr)
        dc = int(c - mc)
        if direction in ("up", "down"):
            stamp_offsets.append((-dr, dc))
        else:
            stamp_offsets.append((dr, -dc))

    return {
        "color": int(color),
        "direction": direction,
        "patch": patch,
        "mask": patch != BLACK,
        "area": int(len(coords)),
        "stamp_offsets": stamp_offsets,
    }


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _place_sources(shapes: list[dict], top_h: int, width: int) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((top_h, width), dtype=bool)
    placed: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(shapes)), key=lambda i: int(shapes[i]["patch"].shape[0] * shapes[i]["patch"].shape[1]), reverse=True)

    for idx in order:
        patch = shapes[idx]["patch"]
        mask = shapes[idx]["mask"]
        h, w = map(int, patch.shape)
        if h > top_h or w > width:
            return None
        dilated = _dilate(occupied)
        candidates = []
        for r in range(top_h - h + 1):
            for c in range(width - w + 1):
                if np.any(dilated[r : r + h, c : c + w] & mask):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        placed[idx] = random.choice(candidates)
        r0, c0 = placed[idx]
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
    return placed


def _safe_anchor_cells(shape: dict, out_shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = out_shape
    safe: list[tuple[int, int]] = []
    clipped: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            _cells, did_clip = _stamp_cells(shape["stamp_offsets"], (r, c), out_shape)
            if did_clip:
                clipped.append((r, c))
            else:
                safe.append((r, c))
    return safe, clipped


def _choose_anchor_positions(shapes: list[dict], out_shape: tuple[int, int]) -> dict[int, tuple[int, int]] | None:
    n = len(shapes)
    order = list(range(n))
    random.shuffle(order)
    need_clip = random.random() < 0.5
    need_overlap = random.random() < 0.5
    clip_idx = order[0] if need_clip else None
    overlap_idx = order[1] if need_overlap and n >= 2 else None

    anchors: dict[int, tuple[int, int]] = {}
    used_anchors: set[tuple[int, int]] = set()
    occupied_stamp: set[tuple[int, int]] = set()

    for pos, idx in enumerate(order):
        shape = shapes[idx]
        safe, clipped = _safe_anchor_cells(shape, out_shape)

        if idx == clip_idx:
            anchor_pool = [cell for cell in clipped if cell not in used_anchors]
        else:
            anchor_pool = [cell for cell in safe if cell not in used_anchors]
            if not anchor_pool:
                anchor_pool = [cell for cell in clipped if cell not in used_anchors]

        if idx == overlap_idx:
            overlap_pool = []
            for cell in anchor_pool:
                stamp, _ = _stamp_cells(shape["stamp_offsets"], cell, out_shape)
                if any(pt in occupied_stamp for pt in stamp):
                    overlap_pool.append(cell)
            anchor_pool = overlap_pool

        if not anchor_pool:
            return None

        if pos == 0 and idx != clip_idx:
            random.shuffle(anchor_pool)
            anchor_pool.sort(key=lambda cell: abs(cell[0] - out_shape[0] // 2) + abs(cell[1] - out_shape[1] // 2))
            anchor = anchor_pool[0]
        else:
            anchor = random.choice(anchor_pool)

        anchors[int(shape["color"])] = anchor
        used_anchors.add(anchor)
        stamp, _ = _stamp_cells(shape["stamp_offsets"], anchor, out_shape)
        occupied_stamp.update(stamp)

    if need_clip or need_overlap:
        stats = _anchor_stats(shapes, anchors, out_shape)
        if need_clip and not stats["has_clip"]:
            return None
        if need_overlap and not stats["has_overlap"]:
            return None
    return anchors


def _anchor_stats(shapes: list[dict], anchors: dict[int, tuple[int, int]], out_shape: tuple[int, int]) -> dict:
    seen: set[tuple[int, int]] = set()
    has_overlap = False
    has_clip = False
    for shape in shapes:
        cells, clipped = _stamp_cells(shape["stamp_offsets"], anchors[int(shape["color"])], out_shape)
        has_clip |= clipped
        for cell in cells:
            if cell in seen:
                has_overlap = True
            seen.add(cell)
    return {"has_overlap": has_overlap, "has_clip": has_clip, "filled": len(seen)}


def _balanced_directions(n: int) -> list[str]:
    dirs = ["up", "down", "left", "right"]
    start = random.randint(0, len(dirs) - 1)
    seq = [dirs[(start + i) % len(dirs)] for i in range(n)]
    if n > 4:
        seq[-1] = random.choice(dirs)
    random.shuffle(seq)
    return seq


def _build_grid(
    shapes: list[dict],
    source_pos: dict[int, tuple[int, int]],
    anchors: dict[int, tuple[int, int]],
    top_h: int,
    width: int,
    out_shape: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = out_shape
    frame_h = out_h + 2
    frame_w = out_w + 2
    frame_left_margin = random.randint(*FRAME_MARGIN_RANGE)
    min_width = frame_w + frame_left_margin + 1
    assert width >= min_width
    frame_c = random.randint(frame_left_margin, width - frame_w - 1)
    bottom_h = frame_h + random.randint(*BOTTOM_EXTRA_H_RANGE)
    frame_r = random.randint(1, bottom_h - frame_h - 1)

    top = np.full((top_h, width), BLACK, dtype=int)
    for idx, shape in enumerate(shapes):
        r0, c0 = source_pos[idx]
        patch = shape["patch"]
        h, w = patch.shape
        mask = patch != BLACK
        top[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]

    bottom = np.full((bottom_h, width), BLACK, dtype=int)
    bottom[frame_r, frame_c : frame_c + frame_w] = GREY
    bottom[frame_r + frame_h - 1, frame_c : frame_c + frame_w] = GREY
    bottom[frame_r : frame_r + frame_h, frame_c] = GREY
    bottom[frame_r : frame_r + frame_h, frame_c + frame_w - 1] = GREY

    for color, (r, c) in anchors.items():
        bottom[frame_r + 1 + r, frame_c + 1 + c] = int(color)

    sep = np.full((1, width), MAROON, dtype=int)
    return np.vstack([top, sep, bottom])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_shapes = random.randint(*SHAPE_COUNT_RANGE)
        colors = random.sample(PALETTE, n_shapes)
        directions = _balanced_directions(n_shapes)

        shapes: list[dict] = []
        used_areas: set[int] = set()
        valid = True
        for color, direction in zip(colors, directions):
            made = None
            for _ in range(80):
                candidate = _make_shape(direction, color)
                if candidate["area"] not in used_areas:
                    used_areas.add(int(candidate["area"]))
                    made = candidate
                    break
            if made is None:
                valid = False
                break
            shapes.append(made)
        if not valid:
            continue

        out_shape = (random.randint(*OUTPUT_H_RANGE), random.randint(*OUTPUT_W_RANGE))
        anchors = _choose_anchor_positions(shapes, out_shape)
        if anchors is None:
            continue

        stats = _anchor_stats(shapes, anchors, out_shape)
        if stats["filled"] < max(8, n_shapes * 3):
            continue
        if stats["filled"] > out_shape[0] * out_shape[1] - 4:
            continue

        width = random.randint(max(GRID_W_RANGE[0], out_shape[1] + 6), GRID_W_RANGE[1])
        top_h = random.randint(*TOP_H_RANGE)
        source_pos = _place_sources(shapes, top_h, width)
        if source_pos is None:
            continue

        grid = _build_grid(shapes, source_pos, anchors, top_h, width, out_shape)
        expected = _render_output(shapes, anchors, out_shape)

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
