import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
TEMPLATE_COLOR = 5
SWATCH_SHAPE = (2, 2)
GEN_RETRIES = 300

# Irregular 4-connected templates. Orientation matters.
SHAPE_LIBRARY = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _touches_edge(box: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    height = rel[-1][0] + 1
    width = max(c for _, c in rel) + 1
    return height, width, tuple(rel)


def _frame_holes(
    crop: np.ndarray,
    color: int,
    bg: int,
    offset_r: int,
    offset_c: int,
) -> list[dict]:
    if not np.all((crop == bg) | (crop == color)):
        return []
    if not np.all(crop[0, :] == color) or not np.all(crop[-1, :] == color):
        return []
    if not np.all(crop[:, 0] == color) or not np.all(crop[:, -1] == color):
        return []

    holes = []
    for comp in _components(crop == bg):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == crop.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == crop.shape[1] - 1):
            continue
        global_coords = comp + np.array([[offset_r, offset_c]])
        holes.append(
            {
                "coords": global_coords,
                "shape": _shape_key(comp),
            }
        )
    return holes


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    frame_mask = np.zeros_like(grid, dtype=bool)
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            box = _bbox(coords)
            if _touches_edge(box, h, w):
                continue
            if box[2] - box[0] + 1 < 9 or box[3] - box[1] + 1 < 9:
                continue
            r0, c0, r1, c1 = box
            crop = grid[r0 : r1 + 1, c0 : c1 + 1]
            rel = coords - np.array([[r0, c0]])
            comp_mask = np.zeros_like(crop, dtype=bool)
            comp_mask[rel[:, 0], rel[:, 1]] = True
            if not np.array_equal(comp_mask, crop == color):
                continue
            holes = _frame_holes(crop, color, bg, r0, c0)
            if not holes:
                continue
            frame_mask[coords[:, 0], coords[:, 1]] = True
            frames.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "holes": holes,
                }
            )
    assert frames, "expected at least one frame"
    return frames, frame_mask


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _find_legend(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> tuple[dict, np.ndarray]:
    h, w = grid.shape
    comps = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            if np.all(frame_mask[coords[:, 0], coords[:, 1]]):
                continue
            box = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "area": len(coords),
                }
            )

    left_anchors = [comp for comp in comps if comp["bbox"][1] == 0]
    right_anchors = [comp for comp in comps if comp["bbox"][3] == w - 1]
    if len(left_anchors) >= len(right_anchors):
        anchors = left_anchors
        strip_end = max(comp["bbox"][3] for comp in anchors) + 4
        legend_comps = [comp for comp in comps if comp["bbox"][1] <= strip_end]
    else:
        anchors = right_anchors
        strip_start = min(comp["bbox"][1] for comp in anchors) - 4
        legend_comps = [comp for comp in comps if comp["bbox"][3] >= strip_start]

    assert anchors, "expected edge-anchored legend templates"
    legend_comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    groups = []
    for comp in legend_comps:
        r0, _, r1, _ = comp["bbox"]
        placed = False
        for group in groups:
            gr0, gr1 = group["rows"]
            if _intervals_overlap(r0, r1, gr0, gr1):
                group["rows"] = (min(gr0, r0), max(gr1, r1))
                group["items"].append(comp)
                placed = True
                break
        if not placed:
            groups.append({"rows": (r0, r1), "items": [comp]})

    legend_mask = np.zeros_like(grid, dtype=bool)
    mapping = {}
    for group in groups:
        items = group["items"]
        assert len(items) == 2, "each legend row must contain template and swatch"
        solid = [item for item in items if item["area"] == _bbox_area(item["bbox"])]
        shape = [item for item in items if item["area"] < _bbox_area(item["bbox"])]
        assert len(solid) == 1 and len(shape) == 1
        swatch = solid[0]
        template = shape[0]
        mapping[_shape_key(template["coords"])] = swatch["color"]
        for item in items:
            legend_mask[item["coords"][:, 0], item["coords"][:, 1]] = True

    assert mapping, "expected at least one legend mapping"
    return mapping, legend_mask


def _draw_rays(out: np.ndarray, center: tuple[int, int], length: int, color: int, bg: int) -> None:
    h, w = out.shape
    cr, cc = center
    for dr, dc in DIR4:
        for step in range(1, length + 1):
            nr = cr + dr * step
            nc = cc + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(out[nr, nc]) != bg:
                break
            out[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's internal hole shape to the edge legend, then draw center rays whose length equals the hole count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    frames, frame_mask = _find_frames(g, bg)
    legend_map, legend_mask = _find_legend(g, bg, frame_mask)

    out = np.full_like(g, bg)
    keep_mask = frame_mask | legend_mask
    out[keep_mask] = g[keep_mask]

    for frame in frames:
        box = frame["bbox"]
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        hole_shapes = [hole["shape"] for hole in frame["holes"]]
        matched_colors = {legend_map[shape] for shape in hole_shapes if shape in legend_map}
        assert len(matched_colors) == 1, "frame must match exactly one legend template"
        ray_color = matched_colors.pop()
        _draw_rays(out, center, len(frame["holes"]), ray_color, bg)

    return out


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _distribute(total: int, slots: int, rng: random.Random) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[rng.randrange(slots)] += 1
    return parts


def _line_starts(
    count: int,
    obj_size: int,
    span: int,
    edge_margin: int,
    gap: int,
    rng: random.Random,
) -> list[int] | None:
    if count == 0:
        return []
    base = count * obj_size + (count - 1) * gap + 2 * edge_margin
    if base > span:
        return None
    extras = _distribute(span - base, count + 1, rng)
    starts = []
    cur = edge_margin + extras[0]
    for i in range(count):
        starts.append(cur)
        cur += obj_size
        if i < count - 1:
            cur += gap + extras[i + 1]
    return starts


def _build_frame_patch(
    shape_mask: np.ndarray,
    hole_count: int,
    frame_color: int,
    layout: str,
    rng: random.Random,
) -> np.ndarray | None:
    side_count = hole_count - 1
    sh, sw = shape_mask.shape
    center_side = 2 * hole_count + 1

    if layout == "tb":
        outer_h = center_side + 2 * (sh + 2)
        outer_w = max(center_side + 4, side_count * sw + max(0, side_count - 1) + 2)
    else:
        outer_h = max(center_side + 4, side_count * sh + max(0, side_count - 1) + 2)
        outer_w = center_side + 2 * (sw + 2)
    outer_h = _odd(outer_h)
    outer_w = _odd(outer_w)

    patch = np.full((outer_h, outer_w), frame_color, dtype=int)
    center_r0 = outer_h // 2 - center_side // 2
    center_c0 = outer_w // 2 - center_side // 2
    patch[center_r0 : center_r0 + center_side, center_c0 : center_c0 + center_side] = BG

    if layout == "tb":
        per_top = rng.randrange(side_count + 1)
        per_bottom = side_count - per_top
        if side_count == 1 and rng.random() < 0.5:
            per_top, per_bottom = 1, 0
        elif side_count == 1:
            per_top, per_bottom = 0, 1
        top_cols = _line_starts(per_top, sw, outer_w, 1, 1, rng)
        bottom_cols = _line_starts(per_bottom, sw, outer_w, 1, 1, rng)
        if top_cols is None or bottom_cols is None:
            return None
        top_r = center_r0 - sh - 1
        bot_r = center_r0 + center_side + 1
        for col in top_cols:
            sub = patch[top_r : top_r + sh, col : col + sw]
            sub[shape_mask] = BG
        for col in bottom_cols:
            sub = patch[bot_r : bot_r + sh, col : col + sw]
            sub[shape_mask] = BG
    else:
        per_left = rng.randrange(side_count + 1)
        per_right = side_count - per_left
        if side_count == 1 and rng.random() < 0.5:
            per_left, per_right = 1, 0
        elif side_count == 1:
            per_left, per_right = 0, 1
        left_rows = _line_starts(per_left, sh, outer_h, 1, 1, rng)
        right_rows = _line_starts(per_right, sh, outer_h, 1, 1, rng)
        if left_rows is None or right_rows is None:
            return None
        left_c = center_c0 - sw - 1
        right_c = center_c0 + center_side + 1
        for row in left_rows:
            sub = patch[row : row + sh, left_c : left_c + sw]
            sub[shape_mask] = BG
        for row in right_rows:
            sub = patch[row : row + sh, right_c : right_c + sw]
            sub[shape_mask] = BG

    holes = _frame_holes(patch, frame_color, BG, 0, 0)
    if len(holes) != hole_count:
        return None
    shape_hist = Counter(hole["shape"] for hole in holes)
    if shape_hist[_shape_key(np.argwhere(shape_mask))] != side_count:
        return None
    return patch


def _make_legend_row(shape_mask: np.ndarray, ray_color: int, side: str) -> np.ndarray:
    sh, sw = shape_mask.shape
    row_h = max(sh, SWATCH_SHAPE[0])
    row_w = sw + 2 + SWATCH_SHAPE[1]
    row = np.full((row_h, row_w), BG, dtype=int)
    tr = (row_h - sh) // 2
    sr = (row_h - SWATCH_SHAPE[0]) // 2
    if side == "left":
        row[tr : tr + sh, 0:sw][shape_mask] = TEMPLATE_COLOR
        row[sr : sr + SWATCH_SHAPE[0], sw + 2 : sw + 2 + SWATCH_SHAPE[1]] = ray_color
    else:
        tc = row_w - sw
        row[tr : tr + sh, tc : tc + sw][shape_mask] = TEMPLATE_COLOR
        row[sr : sr + SWATCH_SHAPE[0], 0 : SWATCH_SHAPE[1]] = ray_color
    return row


def _can_place(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(mask.shape[0], r0 + h + 1)
    cc1 = min(mask.shape[1], c0 + w + 1)
    return not np.any(mask[rr0:rr1, cc0:cc1])


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    use = patch != BG
    sub[use] = patch[use]


def _render_expected(
    grid: np.ndarray,
    keep_mask: np.ndarray,
    frames_meta: list[dict],
) -> np.ndarray:
    out = np.full_like(grid, BG)
    out[keep_mask] = grid[keep_mask]
    for meta in frames_meta:
        _draw_rays(out, meta["center"], meta["hole_count"], meta["ray_color"], BG)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(1, 3)
    legend_side = rng.choice(["left", "right"])
    shapes = rng.sample(SHAPE_LIBRARY, n_frames)
    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], 2 * n_frames)
    frame_colors = colors[:n_frames]
    ray_colors = colors[n_frames:]
    hole_counts = [rng.randint(2, 4) for _ in range(n_frames)]
    layouts = [rng.choice(["tb", "lr"]) for _ in range(n_frames)]

    frame_patches = []
    for shape_mask, hole_count, frame_color, layout in zip(shapes, hole_counts, frame_colors, layouts):
        patch = _build_frame_patch(shape_mask, hole_count, frame_color, layout, rng)
        if patch is None:
            return None
        frame_patches.append(patch)

    legend_rows = [_make_legend_row(shape_mask, ray_color, legend_side) for shape_mask, ray_color in zip(shapes, ray_colors)]
    row_gaps = [rng.randint(1, 2) for _ in range(max(0, n_frames - 1))]
    legend_h = sum(row.shape[0] for row in legend_rows) + sum(row_gaps)
    legend_w = max(row.shape[1] for row in legend_rows)

    min_h = max(legend_h + 2, max(patch.shape[0] for patch in frame_patches) + 2, 22 + 3 * (n_frames - 1))
    min_w = max(legend_w + 10, max(patch.shape[1] for patch in frame_patches) + legend_w + 6, 26 + 4 * (n_frames - 1))
    height = rng.randint(min_h, min_h + 8)
    width = rng.randint(min_w, min_w + 8)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    legend_top = rng.randint(1, height - legend_h - 1)
    current_r = legend_top
    keep_mask = np.zeros_like(grid, dtype=bool)
    for i, row in enumerate(legend_rows):
        row_h, row_w = row.shape
        col = 0 if legend_side == "left" else width - row_w
        _place_patch(grid, row, current_r, col)
        row_mask = row != BG
        keep_mask[current_r : current_r + row_h, col : col + row_w] |= row_mask
        occupied[max(0, current_r - 1) : min(height, current_r + row_h + 1), max(0, col - 1) : min(width, col + row_w + 1)] = True
        current_r += row_h
        if i < len(row_gaps):
            current_r += row_gaps[i]

    left_bound = legend_w + 3 if legend_side == "left" else 1
    right_bound = width - 2 if legend_side == "left" else width - legend_w - 4
    if right_bound <= left_bound:
        return None

    order = sorted(range(n_frames), key=lambda idx: frame_patches[idx].shape[0] * frame_patches[idx].shape[1], reverse=True)
    frames_meta = [None] * n_frames
    for idx in order:
        patch = frame_patches[idx]
        ph, pw = patch.shape
        r_lo = 1
        r_hi = height - ph - 1
        c_lo = left_bound
        c_hi = right_bound - pw + 1
        if r_hi < r_lo or c_hi < c_lo:
            return None
        placed = False
        for _ in range(150):
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            if not _can_place(occupied, r0, c0, ph, pw):
                continue
            _place_patch(grid, patch, r0, c0)
            patch_mask = patch != BG
            keep_mask[r0 : r0 + ph, c0 : c0 + pw] |= patch_mask
            occupied[max(0, r0 - 1) : min(height, r0 + ph + 1), max(0, c0 - 1) : min(width, c0 + pw + 1)] = True
            frames_meta[idx] = {
                "center": (r0 + ph // 2, c0 + pw // 2),
                "hole_count": hole_counts[idx],
                "ray_color": ray_colors[idx],
            }
            placed = True
            break
        if not placed:
            return None

    noise_count = rng.randint(3, 8)
    noise_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    placed_noise = 0
    for _ in range(200):
        if placed_noise >= noise_count:
            break
        r = rng.randint(1, height - 2)
        c = rng.randint(1, width - 2)
        if occupied[r, c]:
            continue
        rr0 = max(0, r - 1)
        rr1 = min(height, r + 2)
        cc0 = max(0, c - 1)
        cc1 = min(width, c + 2)
        if np.any(keep_mask[rr0:rr1, cc0:cc1]):
            continue
        if np.any(grid[rr0:rr1, cc0:cc1] != BG):
            continue
        grid[r, c] = rng.choice(noise_colors)
        occupied[r, c] = True
        placed_noise += 1
    if placed_noise < 3:
        return None

    expected = _render_expected(grid, keep_mask, frames_meta)  # type: ignore[arg-type]
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
