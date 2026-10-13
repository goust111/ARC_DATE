import numpy as np


BG_COLOR = 7
PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]
GRID_HEIGHT_RANGE = (18, 31)
GRID_WIDTH_RANGE = (18, 31)
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_AREA_RANGE = (2, 6)
OBJECT_BBOX_SIDE_RANGE = (1, 4)
GENERATION_RETRIES = 400
RNG = np.random.default_rng()


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _square_hull(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    side = max(h, w)
    extra_h = side - h
    extra_w = side - w
    sr0 = r0 - extra_h // 2
    sr1 = r1 + (extra_h - extra_h // 2)
    sc0 = c0 - extra_w // 2
    sc1 = c1 + (extra_w - extra_w // 2)
    return int(sr0), int(sc0), int(sr1), int(sc1)


def _paint_square_ring(
    out: np.ndarray, rect: tuple[int, int, int, int], color: int
) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0)
    rr1 = min(h - 1, r1)
    cc0 = max(0, c0)
    cc1 = min(w - 1, c1)
    if rr0 > rr1 or cc0 > cc1:
        return
    out[rr0 : rr1 + 1, cc0] = color
    out[rr0 : rr1 + 1, cc1] = color
    out[rr0, cc0 : cc1 + 1] = color
    out[rr1, cc0 : cc1 + 1] = color


def _ring_rectangles(bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    side = max(h, w)
    sr0, sc0, sr1, sc1 = _square_hull(bbox)
    rings = []
    for layer in range(1, side + 1):
        delta = 2 * layer - 1
        rings.append((sr0 - delta, sc0 - delta, sr1 + delta, sc1 + delta))
    return rings


def _footprint_rect(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    rings = _ring_rectangles(bbox)
    r0 = min(bbox[0], rings[0][0])
    c0 = min(bbox[1], rings[0][1])
    r1 = max(bbox[2], rings[-1][2])
    c1 = max(bbox[3], rings[-1][3])
    return max(0, r0), max(0, c0), min(h - 1, r1), min(w - 1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw concentric square rings around each foreground object.

    For every 4-connected monochrome foreground object, compute its bounding-box
    width W and height H, let R = max(W, H), then draw R one-pixel-thick square
    rings of the same color. The rings are centered on the object's bounding-box
    center, start one cell outside the smallest centered square hull of the bbox,
    and expand outward with one background cell between consecutive rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    comps = _find_components(g, bg)
    out = g.copy()
    for comp in comps:
        for rect in _ring_rectangles(comp["bbox"]):
            _paint_square_ring(out, rect, int(comp["color"]))
    return out


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    start = cells[RNG.integers(len(cells))]
    chosen = {start}
    frontier = {start}
    while len(chosen) < area:
        candidates = []
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    candidates.append((nr, nc))
        if not candidates:
            return None
        nxt = candidates[RNG.integers(len(candidates))]
        chosen.add(nxt)
        frontier.add(nxt)
    mask = np.zeros((height, width), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    if rows[0] != 0 or rows[-1] != height - 1 or cols[0] != 0 or cols[-1] != width - 1:
        return None
    return mask


def _sample_object_mask() -> np.ndarray:
    for _ in range(200):
        height = int(RNG.integers(OBJECT_BBOX_SIDE_RANGE[0], OBJECT_BBOX_SIDE_RANGE[1] + 1))
        width = int(RNG.integers(OBJECT_BBOX_SIDE_RANGE[0], OBJECT_BBOX_SIDE_RANGE[1] + 1))
        if height * width < OBJECT_AREA_RANGE[0]:
            continue
        max_area = min(OBJECT_AREA_RANGE[1], height * width)
        area = int(RNG.integers(OBJECT_AREA_RANGE[0], max_area + 1))
        mask = _random_connected_mask(height, width, area)
        if mask is not None:
            return mask
    raise RuntimeError("failed to sample a connected object mask")


def _rectangles_separate(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _place_object(
    grid: np.ndarray,
    occupied: list[tuple[int, int, int, int]],
    mask: np.ndarray,
    force_clip: bool,
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    gh, gw = grid.shape
    mh, mw = mask.shape
    bbox0 = (0, 0, mh - 1, mw - 1)
    rings0 = _ring_rectangles(bbox0)
    outer0 = rings0[-1]
    vertical_modes = ["any"]
    horizontal_modes = ["any"]
    if force_clip:
        vertical_modes = ["top", "bottom", "any"]
        horizontal_modes = ["left", "right", "any"]
    for _ in range(250):
        vm = vertical_modes[int(RNG.integers(len(vertical_modes)))]
        hm = horizontal_modes[int(RNG.integers(len(horizontal_modes)))]
        if vm == "top":
            top_limit = -outer0[0]
            row = int(RNG.integers(0, min(top_limit, gh - mh) + 1))
        elif vm == "bottom":
            bottom_start = gh - 1 - outer0[2]
            row = int(RNG.integers(max(0, bottom_start), gh - mh + 1))
        else:
            row = int(RNG.integers(0, gh - mh + 1))
        if hm == "left":
            left_limit = -outer0[1]
            col = int(RNG.integers(0, min(left_limit, gw - mw) + 1))
        elif hm == "right":
            right_start = gw - 1 - outer0[3]
            col = int(RNG.integers(max(0, right_start), gw - mw + 1))
        else:
            col = int(RNG.integers(0, gw - mw + 1))

        bbox = (row, col, row + mh - 1, col + mw - 1)
        footprint = _footprint_rect(bbox, grid.shape)
        if not all(_rectangles_separate(footprint, prev, margin=1) for prev in occupied):
            continue
        outer = _ring_rectangles(bbox)[-1]
        clipped = outer[0] < 0 or outer[1] < 0 or outer[2] >= gh or outer[3] >= gw
        if force_clip and not clipped:
            continue
        return row, col, footprint
    return None


def _input_is_valid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    bg = _dominant_color(grid)
    if bg != BG_COLOR:
        return False
    comps = _find_components(grid, bg)
    if not (OBJECT_COUNT_RANGE[0] <= len(comps) <= OBJECT_COUNT_RANGE[1]):
        return False
    for comp in comps:
        if not (OBJECT_AREA_RANGE[0] <= comp["area"] <= OBJECT_AREA_RANGE[1]):
            return False
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 > OBJECT_BBOX_SIDE_RANGE[1] or c1 - c0 + 1 > OBJECT_BBOX_SIDE_RANGE[1]:
            return False
    out = main(grid)
    out_comps = _find_components(out, bg)
    if len(out_comps) < len(comps):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    gh = int(RNG.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    gw = int(RNG.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    grid = np.full((gh, gw), BG_COLOR, dtype=int)

    object_count = int(RNG.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))
    colors = list(RNG.permutation(PALETTE)[:object_count])
    masks = [_sample_object_mask() for _ in range(object_count)]

    force_clip = bool(RNG.integers(2))
    clip_index = int(RNG.integers(object_count)) if force_clip else -1
    order = list(np.argsort([-max(mask.shape) for mask in masks]))
    occupied: list[tuple[int, int, int, int]] = []

    placements: list[tuple[int, int]] = [(-1, -1)] * object_count
    for idx in order:
        placed = _place_object(
            grid,
            occupied,
            masks[idx],
            force_clip=(idx == clip_index),
        )
        if placed is None:
            return None
        row, col, footprint = placed
        placements[idx] = (row, col)
        occupied.append(footprint)

    for idx, mask in enumerate(masks):
        row, col = placements[idx]
        color = colors[idx]
        coords = np.argwhere(mask)
        for dr, dc in coords:
            grid[row + int(dr), col + int(dc)] = color

    if force_clip:
        bbox = (
            placements[clip_index][0],
            placements[clip_index][1],
            placements[clip_index][0] + masks[clip_index].shape[0] - 1,
            placements[clip_index][1] + masks[clip_index].shape[1] - 1,
        )
        outer = _ring_rectangles(bbox)[-1]
        if not (outer[0] < 0 or outer[1] < 0 or outer[2] >= gh or outer[3] >= gw):
            return None

    if not _input_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
