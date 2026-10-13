import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
AZURE = 8
PALETTE_COLORS = (2, 3, 4, 6)

LEGEND_PAD = 1
LEGEND_ROW_GAP = 1
OBJECT_PADDING = 1
GEN_RETRIES = 400


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(points: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(points[:, 0].min()),
        int(points[:, 1].min()),
        int(points[:, 0].max()),
        int(points[:, 1].max()),
    )


def crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = bbox_coords(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    mask = np.asarray(mask, dtype=bool)
    return tuple(mask.shape), tuple(int(v) for v in mask.ravel())


def scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=bool)
    return np.kron(np.asarray(mask, dtype=bool), block)


def matches_scaled(base: np.ndarray, obj: np.ndarray) -> bool:
    base = crop_mask(base)
    obj = crop_mask(obj)
    if base.size == 0 or obj.size == 0:
        return False
    bh, bw = base.shape
    oh, ow = obj.shape
    if oh % bh != 0 or ow % bw != 0:
        return False
    k_h = oh // bh
    k_w = ow // bw
    if k_h != k_w or k_h < 1:
        return False
    k = k_h
    blocks = obj.reshape(bh, k, bw, k).transpose(0, 2, 1, 3)
    block_any = blocks.any(axis=(2, 3))
    block_all = blocks.all(axis=(2, 3))
    return np.array_equal(block_any, block_all) and np.array_equal(block_any, base)


def paste_color(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w][patch] = int(color)


def reserve_bbox(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def free_positions(occupied: np.ndarray, patch_shape: tuple[int, int], pad: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = patch_shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        rr0 = max(0, r0 - pad)
        rr1 = min(h, r0 + ph + pad)
        for c0 in range(w - pw + 1):
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + pw + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                out.append((r0, c0))
    return out


def random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(*cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return crop_mask(mask)


def touches_all_sides(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def all_legend_shapes() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        if not touches_all_sides(mask):
            continue
        if len(components_mask(mask)) != 1:
            continue
        shapes.append(mask)
    return shapes


LEGEND_SHAPES = all_legend_shapes()


def build_legend(shape_rows: list[np.ndarray], colors: list[int]) -> np.ndarray:
    rows = len(shape_rows)
    height = 2 * LEGEND_PAD + rows * 3 + (rows - 1) * LEGEND_ROW_GAP
    width = 2 * LEGEND_PAD + 5
    legend = np.full((height, width), BG, dtype=int)
    legend[0, :] = AZURE
    legend[-1, :] = AZURE
    legend[:, 0] = AZURE
    legend[:, -1] = AZURE
    for idx, (shape, color) in enumerate(zip(shape_rows, colors)):
        r0 = LEGEND_PAD + idx * (3 + LEGEND_ROW_GAP)
        c0 = LEGEND_PAD
        legend[r0 : r0 + 3, c0 : c0 + 3][shape] = GREY
        legend[r0 + 1, c0 + 4] = int(color)
    return legend


def corner_origin(grid_shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    gh, gw = grid_shape
    ph, pw = patch_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, gw - pw
    if corner == "bl":
        return gh - ph, 0
    if corner == "br":
        return gh - ph, gw - pw
    raise ValueError(corner)


def find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in components_mask(grid == AZURE):
        r0, c0, r1, c1 = bbox_coords(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros_like(sub, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if not np.all(sub[frame] == AZURE):
            continue
        if np.any(sub[~frame] == AZURE):
            continue
        if int((sub == AZURE).sum()) != int(frame.sum()):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected a unique azure legend frame"
    return candidates[0]


def parse_legend(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[tuple[tuple[int, int], tuple[int, ...]], int]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    grey_comps = components_mask(interior == GREY)
    color_pixels = [
        (int(r), int(c), int(interior[r, c]))
        for r, c in np.argwhere(
            (interior != BG) & (interior != GREY) & (interior != AZURE)
        )
    ]
    mapping: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    used_swatches: set[int] = set()
    grey_comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    for comp in grey_comps:
        gr0, gc0, gr1, gc1 = bbox_coords(comp)
        mask = np.zeros((gr1 - gr0 + 1, gc1 - gc0 + 1), dtype=bool)
        for rr, cc in comp:
            mask[int(rr) - gr0, int(cc) - gc0] = True
        center_r = 0.5 * (gr0 + gr1)
        candidates = [
            (idx, pix)
            for idx, pix in enumerate(color_pixels)
            if idx not in used_swatches and pix[1] > gc1 and gr0 - 1 <= pix[0] <= gr1 + 1
        ]
        assert candidates, "legend row missing color swatch"
        idx, (_, _, color) = min(
            candidates,
            key=lambda item: (
                abs(item[1][0] - center_r),
                item[1][1] - gc1,
            ),
        )
        used_swatches.add(idx)
        key = mask_key(crop_mask(mask))
        assert key not in mapping, "legend shapes must be unique"
        mapping[key] = int(color)
    assert len(mapping) == len(grey_comps) and len(mapping) in (2, 3)
    return mapping


def object_components(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    mask = grid == GREY
    r0, c0, r1, c1 = legend_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return components_mask(mask)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_bbox = find_legend_bbox(g)
    shape_to_color = parse_legend(g, legend_bbox)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG
    out[out == GREY] = BG

    for comp in object_components(g, legend_bbox):
        r0, c0, r1, c1 = bbox_coords(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for rr, cc in comp:
            mask[int(rr) - r0, int(cc) - c0] = True

        matches = [
            color
            for key, color in shape_to_color.items()
            if matches_scaled(np.array(key[1], dtype=bool).reshape(key[0]), mask)
        ]
        assert len(matches) <= 1, "shape match must be unambiguous"
        if matches:
            paste_color(out, mask, r0, c0, matches[0])
    return out


def sample_unmatched_mask(legend_shapes: list[np.ndarray]) -> np.ndarray | None:
    legend_keys = {mask_key(crop_mask(shape)) for shape in legend_shapes}
    for _ in range(120):
        h = random.randint(3, 7)
        w = random.randint(3, 7)
        area = random.randint(max(3, (h * w) // 3), max(3, min(h * w - 1, (2 * h * w) // 3)))
        mask = random_connected_mask(h, w, area)
        if mask.size == 0:
            continue
        if len(components_mask(mask)) != 1:
            continue
        if any(matches_scaled(shape, mask) for shape in legend_shapes):
            continue
        if mask_key(mask) in legend_keys:
            continue
        return mask
    return None


def place_objects(
    grid_shape: tuple[int, int],
    legend_bbox: tuple[int, int, int, int],
    objects: list[dict],
) -> list[dict] | None:
    occupied = np.zeros(grid_shape, dtype=bool)
    reserve_bbox(
        occupied,
        legend_bbox[0],
        legend_bbox[1],
        legend_bbox[2] - legend_bbox[0] + 1,
        legend_bbox[3] - legend_bbox[1] + 1,
        OBJECT_PADDING,
    )

    placements: list[dict] = []
    order = sorted(range(len(objects)), key=lambda i: objects[i]["mask"].size, reverse=True)
    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        positions = free_positions(occupied, mask.shape, OBJECT_PADDING)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        reserve_bbox(occupied, r0, c0, mask.shape[0], mask.shape[1], OBJECT_PADDING)
        placed = dict(obj)
        placed["r0"] = r0
        placed["c0"] = c0
        placements.append(placed)
    return placements


def build_expected(grid_shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    expected = np.full(grid_shape, BG, dtype=int)
    for obj in placements:
        if obj["out_color"] is not None:
            paste_color(expected, obj["mask"], obj["r0"], obj["c0"], obj["out_color"])
    return expected


def build_input(
    grid_shape: tuple[int, int],
    legend: np.ndarray,
    legend_corner: str,
    placements: list[dict],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    grid = np.full(grid_shape, BG, dtype=int)
    lr0, lc0 = corner_origin(grid_shape, legend.shape, legend_corner)
    grid[lr0 : lr0 + legend.shape[0], lc0 : lc0 + legend.shape[1]] = legend
    for obj in placements:
        paste_color(grid, obj["mask"], obj["r0"], obj["c0"], GREY)
    return grid, (lr0, lc0, lr0 + legend.shape[0] - 1, lc0 + legend.shape[1] - 1)


def gen_attempt() -> np.ndarray | None:
    rows = random.choice([2, 3])
    legend_shapes = random.sample(LEGEND_SHAPES, rows)
    legend_colors = random.sample(PALETTE_COLORS, rows)
    legend = build_legend(legend_shapes, legend_colors)

    grid_h = random.randint(max(24, legend.shape[0] + 8), 30)
    grid_w = random.randint(max(24, legend.shape[1] + 10), 30)
    grid_shape = (grid_h, grid_w)
    legend_corner = random.choice(["tl", "tr", "bl", "br"])
    lr0, lc0 = corner_origin(grid_shape, legend.shape, legend_corner)
    legend_bbox = (lr0, lc0, lr0 + legend.shape[0] - 1, lc0 + legend.shape[1] - 1)

    total_objects = random.randint(3, 5)
    positive_unmatched = [u for u in range(1, min(2, total_objects - 1) + 1) if u < total_objects - u]
    if positive_unmatched and random.random() < 0.5:
        unmatched_count = random.choice(positive_unmatched)
    else:
        unmatched_count = 0
    matched_count = total_objects - unmatched_count

    objects: list[dict] = []
    for _ in range(matched_count):
        idx = random.randrange(rows)
        factor = random.choice([2, 3])
        mask = scale_mask(legend_shapes[idx], factor)
        objects.append({"mask": mask, "out_color": legend_colors[idx]})

    for _ in range(unmatched_count):
        mask = sample_unmatched_mask(legend_shapes)
        if mask is None:
            return None
        objects.append({"mask": mask, "out_color": None})

    random.shuffle(objects)
    placements = place_objects(grid_shape, legend_bbox, objects)
    if placements is None:
        return None

    inp, actual_legend_bbox = build_input(grid_shape, legend, legend_corner, placements)
    expected = build_expected(grid_shape, placements)

    if actual_legend_bbox != legend_bbox:
        return None
    if len(object_components(inp, legend_bbox)) != total_objects:
        return None
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
