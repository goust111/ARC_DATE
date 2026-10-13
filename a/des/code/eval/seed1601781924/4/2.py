from __future__ import annotations

from collections import defaultdict, deque
import math
import random

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANT_ROTATIONS = {
    "ul": 0,
    "ur": 1,
    "lr": 2,
    "ll": 3,
}

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            groups.append(pts)
    return groups


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_points(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns) % 4).copy()


def _coords_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    trimmed = _trim_mask(mask)
    return tuple((int(r), int(c)) for r, c in np.argwhere(trimmed))


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_coords_signature(_rotate_cw(mask, k)) for k in range(4))


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _quadrant(points: list[tuple[int, int]], center_r: float, center_c: float) -> str:
    avg_r = sum(r for r, _ in points) / float(len(points))
    avg_c = sum(c for _, c in points) / float(len(points))
    if avg_r < center_r and avg_c < center_c:
        return "ul"
    if avg_r < center_r and avg_c > center_c:
        return "ur"
    if avg_r > center_r and avg_c > center_c:
        return "lr"
    if avg_r > center_r and avg_c < center_c:
        return "ll"
    raise AssertionError("fragment center lies on a template axis")


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = canvas[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place scattered colored fragments into template holes, rotating by quadrant.

    Input structure:
    - The background is black.
    - A large rectangular template sits near the center as one connected component.
    - The template contains background holes fully enclosed by the template.
    - Outside the template, each colored fragment is a separate 4-connected component.

    Rule:
    - Match each fragment to a hole by shape, allowing rotations only.
    - Determine the fragment's actual placement rotation from its quadrant around the
      template center: upper-left 0, upper-right 90 clockwise, lower-right 180,
      lower-left 270 clockwise.
    - Remove the outside fragments and fill the matching holes with their colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    comps = _components(g != int(bg))
    assert len(comps) >= 2
    template_points = max(comps, key=len)
    fragment_points = [pts for pts in comps if pts is not template_points]

    out = np.full_like(g, int(bg))
    for r, c in template_points:
        out[r, c] = int(g[r, c])

    tr0, tc0, tr1, tc1 = _bbox(template_points)
    center_r = (tr0 + tr1) / 2.0
    center_c = (tc0 + tc1) / 2.0

    template_crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    hole_mask = template_crop == int(bg)
    hole_records = []
    holes_by_signature: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)

    for pts in _components(hole_mask):
        if any(r == 0 or c == 0 or r == hole_mask.shape[0] - 1 or c == hole_mask.shape[1] - 1 for r, c in pts):
            continue
        mask, (hr0, hc0, hr1, hc1) = _mask_from_points(pts)
        record = {
            "mask": mask,
            "bbox": (tr0 + hr0, tc0 + hc0, tr0 + hr1, tc0 + hc1),
            "used": False,
        }
        hole_records.append(record)
        holes_by_signature[_canonical_signature(mask)].append(record)

    assert len(hole_records) == len(fragment_points)

    for points in fragment_points:
        colors = {int(g[r, c]) for r, c in points}
        assert len(colors) == 1
        color = next(iter(colors))
        frag_mask, _ = _mask_from_points(points)
        quadrant = _quadrant(points, center_r, center_c)
        rotated = _rotate_cw(frag_mask, QUADRANT_ROTATIONS[quadrant])

        candidates = []
        for hole in holes_by_signature[_canonical_signature(frag_mask)]:
            if bool(hole["used"]):
                continue
            hole_mask_local = hole["mask"]
            if hole_mask_local.shape == rotated.shape and np.array_equal(hole_mask_local, rotated):
                candidates.append(hole)

        assert len(candidates) == 1
        hole = candidates[0]
        hole["used"] = True
        hr0, hc0, hr1, hc1 = hole["bbox"]
        out[hr0 : hr1 + 1, hc0 : hc1 + 1][hole["mask"]] = int(color)

    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(3, 4)
        box_w = random.randint(3, 4)
        area = random.randint(4, min(6, box_h * box_w - 1))
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}

        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, box_h, box_w):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))

        if len(cells) != area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if min(mask.shape) < 2:
            continue
        if bool(np.all(mask)):
            continue
        rotations = {_coords_signature(_rotate_cw(mask, k)) for k in range(4)}
        if len(rotations) != 4:
            continue
        return mask
    return None


def _sample_shapes(count: int) -> list[np.ndarray] | None:
    used = set()
    shapes: list[np.ndarray] = []
    attempts = 0
    while len(shapes) < count and attempts < 2000:
        attempts += 1
        mask = _random_shape()
        if mask is None:
            continue
        sig = _canonical_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        shapes.append(mask)
    return shapes if len(shapes) == count else None


def _build_template(shapes: list[np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    count = len(shapes)
    max_h = max(int(shape.shape[0]) for shape in shapes)
    max_w = max(int(shape.shape[1]) for shape in shapes)
    extra_slots = random.randint(0, 2)
    slot_total = count + extra_slots

    row_options = [2, 3]
    random.shuffle(row_options)
    for slot_rows in row_options:
        slot_cols = int(math.ceil(slot_total / float(slot_rows)))
        slot_h = max_h + random.randint(2, 3)
        slot_w = max_w + random.randint(2, 3)
        template_h = 2 + slot_rows * slot_h
        template_w = 2 + slot_cols * slot_w
        if template_h > 24 or template_w > 24:
            continue

        template = np.full((template_h, template_w), TEMPLATE_COLOR, dtype=int)
        indices = random.sample(range(slot_rows * slot_cols), count)
        hole_infos = []
        for shape, idx in zip(shapes, indices):
            sr = idx // slot_cols
            sc = idx % slot_cols
            h, w = shape.shape
            top = 1 + sr * slot_h + random.randint(1, slot_h - h - 1)
            left = 1 + sc * slot_w + random.randint(1, slot_w - w - 1)
            patch = template[top : top + h, left : left + w]
            patch[shape] = BG
            hole_infos.append(
                {
                    "hole_mask": shape.copy(),
                    "hole_bbox": (top, left, top + h - 1, left + w - 1),
                }
            )
        return template, hole_infos
    return None


def _quadrant_counts() -> dict[str, int]:
    counts = {name: 1 for name in ("ul", "ur", "ll", "lr")}
    extras = random.randint(0, 2)
    for name in random.sample(list(counts), extras):
        counts[name] += 1
    return counts


def _build_quadrant_canvas(items: list[dict[str, object]]) -> np.ndarray:
    pieces = items[:]
    random.shuffle(pieces)
    if len(pieces) == 1:
        piece = pieces[0]
        mask = piece["fragment_mask"]
        color = piece["color"]
        h, w = mask.shape
        pad_t = random.randint(0, 2)
        pad_b = random.randint(0, 2)
        pad_l = random.randint(0, 2)
        pad_r = random.randint(0, 2)
        canvas = np.full((h + pad_t + pad_b, w + pad_l + pad_r), BG, dtype=int)
        _paint_mask(canvas, mask, pad_t, pad_l, color)
        return canvas

    assert len(pieces) == 2
    gap = random.randint(1, 2)
    a, b = pieces
    mask_a = a["fragment_mask"]
    mask_b = b["fragment_mask"]
    color_a = a["color"]
    color_b = b["color"]
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape

    if random.random() < 0.5:
        core_h = max(int(ha), int(hb))
        core_w = int(wa) + int(wb) + gap
        ra = random.randint(0, core_h - int(ha))
        rb = random.randint(0, core_h - int(hb))
        ca = 0
        cb = int(wa) + gap
    else:
        core_h = int(ha) + int(hb) + gap
        core_w = max(int(wa), int(wb))
        ra = 0
        rb = int(ha) + gap
        ca = random.randint(0, core_w - int(wa))
        cb = random.randint(0, core_w - int(wb))

    pad_t = random.randint(0, 2)
    pad_b = random.randint(0, 2)
    pad_l = random.randint(0, 2)
    pad_r = random.randint(0, 2)
    canvas = np.full((core_h + pad_t + pad_b, core_w + pad_l + pad_r), BG, dtype=int)
    _paint_mask(canvas, mask_a, pad_t + ra, pad_l + ca, color_a)
    _paint_mask(canvas, mask_b, pad_t + rb, pad_l + cb, color_b)
    return canvas


def _place_corner_sprite(
    grid: np.ndarray,
    sprite: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    keep_gap_bottom: bool,
    keep_gap_right: bool,
    keep_gap_top: bool,
    keep_gap_left: bool,
) -> None:
    h, w = sprite.shape
    min_r = int(row_start + (1 if keep_gap_top else 0))
    max_r = int(row_end - h - (1 if keep_gap_bottom else 0))
    min_c = int(col_start + (1 if keep_gap_left else 0))
    max_c = int(col_end - w - (1 if keep_gap_right else 0))
    assert min_r <= max_r and min_c <= max_c
    r0 = random.randint(min_r, max_r)
    c0 = random.randint(min_c, max_c)
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[sprite != BG] = sprite[sprite != BG]


def _gen_attempt() -> np.ndarray | None:
    counts = _quadrant_counts()
    total = sum(counts.values())

    shapes = _sample_shapes(total)
    if shapes is None:
        return None

    built = _build_template(shapes)
    if built is None:
        return None
    template, hole_infos = built

    colors = random.sample(FRAGMENT_COLORS, total)
    order = list(range(total))
    random.shuffle(order)

    pieces_by_quadrant: dict[str, list[dict[str, object]]] = {name: [] for name in counts}
    cursor = 0
    for quadrant in ("ul", "ur", "ll", "lr"):
        for _ in range(counts[quadrant]):
            hole_idx = order[cursor]
            cursor += 1
            color = colors[hole_idx]
            hole_mask = hole_infos[hole_idx]["hole_mask"]
            turns = (4 - QUADRANT_ROTATIONS[quadrant]) % 4
            fragment_mask = _rotate_cw(hole_mask, turns)
            pieces_by_quadrant[quadrant].append(
                {
                    "color": color,
                    "hole_mask": hole_mask,
                    "hole_bbox": hole_infos[hole_idx]["hole_bbox"],
                    "fragment_mask": fragment_mask,
                    "quadrant": quadrant,
                }
            )

    corner_canvases = {
        quadrant: _build_quadrant_canvas(items)
        for quadrant, items in pieces_by_quadrant.items()
    }

    top_band = max(
        int(corner_canvases["ul"].shape[0]),
        int(corner_canvases["ur"].shape[0]),
    ) + random.randint(2, 4)
    bottom_band = max(
        int(corner_canvases["ll"].shape[0]),
        int(corner_canvases["lr"].shape[0]),
    ) + random.randint(2, 4)
    left_band = max(
        int(corner_canvases["ul"].shape[1]),
        int(corner_canvases["ll"].shape[1]),
    ) + random.randint(2, 4)
    right_band = max(
        int(corner_canvases["ur"].shape[1]),
        int(corner_canvases["lr"].shape[1]),
    ) + random.randint(2, 4)

    th, tw = template.shape
    grid_h = top_band + th + bottom_band
    grid_w = left_band + tw + right_band
    if grid_h > 40 or grid_w > 40:
        return None

    input_grid = np.full((grid_h, grid_w), BG, dtype=int)
    output_grid = np.full((grid_h, grid_w), BG, dtype=int)

    tr0 = top_band
    tc0 = left_band
    input_grid[tr0 : tr0 + th, tc0 : tc0 + tw] = template
    output_grid[tr0 : tr0 + th, tc0 : tc0 + tw] = TEMPLATE_COLOR

    for quadrant, items in pieces_by_quadrant.items():
        for piece in items:
            hr0, hc0, hr1, hc1 = piece["hole_bbox"]
            hole_patch = output_grid[tr0 + hr0 : tr0 + hr1 + 1, tc0 + hc0 : tc0 + hc1 + 1]
            hole_patch[piece["hole_mask"]] = int(piece["color"])

    _place_corner_sprite(
        input_grid,
        corner_canvases["ul"],
        0,
        top_band,
        0,
        left_band,
        keep_gap_bottom=True,
        keep_gap_right=True,
        keep_gap_top=False,
        keep_gap_left=False,
    )
    _place_corner_sprite(
        input_grid,
        corner_canvases["ur"],
        0,
        top_band,
        tc0 + tw,
        grid_w,
        keep_gap_bottom=True,
        keep_gap_right=False,
        keep_gap_top=False,
        keep_gap_left=True,
    )
    _place_corner_sprite(
        input_grid,
        corner_canvases["ll"],
        tr0 + th,
        grid_h,
        0,
        left_band,
        keep_gap_bottom=False,
        keep_gap_right=True,
        keep_gap_top=True,
        keep_gap_left=False,
    )
    _place_corner_sprite(
        input_grid,
        corner_canvases["lr"],
        tr0 + th,
        grid_h,
        tc0 + tw,
        grid_w,
        keep_gap_bottom=False,
        keep_gap_right=False,
        keep_gap_top=True,
        keep_gap_left=True,
    )

    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
