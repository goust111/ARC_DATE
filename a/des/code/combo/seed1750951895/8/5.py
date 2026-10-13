import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
FRAME_COUNT_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (5, 9)
INNER_WIDTH_RANGE = (5, 9)
INSET_RANGE = (3, 5)
OUTER_MARGIN_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return max(counts, key=counts.get)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for coords in _components_of_color(grid == color):
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            comp_mask = np.zeros_like(grid, dtype=bool)
            for r, c in coords:
                comp_mask[r, c] = True
            border = _frame_mask(grid.shape, bbox)
            if np.array_equal(comp_mask, border):
                frames.append({"color": color, "bbox": bbox})
    assert frames, "expected at least one rectangular frame"
    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _local_mask(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    descendant_bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    for dr0, dc0, dr1, dc1 in descendant_bboxes:
        mask[dr0 : dr1 + 1, dc0 : dc1 + 1] = False
    return mask


def _flip_local_region(
    out: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    coords = np.argwhere(mask)
    for r, c in coords:
        mc = c0 + c1 - c
        assert mask[r, mc], "local region must be horizontally closed under reflection"
    out[mask] = bg
    for r, c in coords:
        mc = c0 + c1 - c
        out[r, mc] = src[r, c]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort nested one-pixel-thick rectangular frames by containment depth and
    horizontally flip the local interior pattern of odd-depth frames.

    The local interior of a frame is its interior area with every contained
    frame bbox removed, so nested subproblems stay fixed while the annular
    pattern around them is transformed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    frames = _find_frames(g, bg)

    depths: list[int] = []
    descendants: list[list[tuple[int, int, int, int]]] = []
    for i, frame in enumerate(frames):
        bbox = frame["bbox"]
        container_count = 0
        smaller: list[tuple[int, int, int, int]] = []
        for j, other in enumerate(frames):
            if i == j:
                continue
            obox = other["bbox"]
            if _contains(obox, bbox):
                container_count += 1
            if _contains(bbox, obox):
                smaller.append(obox)
        depths.append(container_count)
        descendants.append(smaller)

    out = g.copy()
    for frame, depth, child_boxes in sorted(
        zip(frames, depths, descendants),
        key=lambda item: item[1],
    ):
        if depth % 2 == 0:
            continue
        bbox = frame["bbox"]
        mask = _local_mask(g.shape, bbox, child_boxes)
        if not np.any(mask):
            continue
        _flip_local_region(out, g, mask, bbox, bg)
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    grid[_frame_mask(grid.shape, bbox)] = color


def _band_rectangles(
    frame_bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int] | None,
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    if child_bbox is None:
        return [(ir0, ic0, ir1, ic1)]
    cr0, cc0, cr1, cc1 = child_bbox
    bands: list[tuple[int, int, int, int]] = []
    if ir0 <= cr0 - 1:
        bands.append((ir0, ic0, cr0 - 1, ic1))
    if cr1 + 1 <= ir1:
        bands.append((cr1 + 1, ic0, ir1, ic1))
    if ic0 <= cc0 - 1:
        bands.append((cr0, ic0, cr1, cc0 - 1))
    if cc1 + 1 <= ic1:
        bands.append((cr0, cc1 + 1, cr1, ic1))
    return [b for b in bands if b[0] <= b[2] and b[1] <= b[3]]


def _random_pattern_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    shape_type = rng.choice(["rect", "hbar", "vbar", "ell"])
    mask = np.zeros((height, width), dtype=bool)
    if shape_type == "rect":
        ph = rng.randint(1, max(1, min(height, 3)))
        pw = rng.randint(1, max(1, min(width, 3)))
        r0 = rng.randint(0, height - ph)
        c0 = rng.randint(0, width - pw)
        mask[r0 : r0 + ph, c0 : c0 + pw] = True
    elif shape_type == "hbar":
        ph = 1
        pw = rng.randint(2 if width >= 2 else 1, width)
        r0 = rng.randint(0, height - ph)
        c0 = rng.randint(0, width - pw)
        mask[r0 : r0 + ph, c0 : c0 + pw] = True
    elif shape_type == "vbar":
        ph = rng.randint(2 if height >= 2 else 1, height)
        pw = 1
        r0 = rng.randint(0, height - ph)
        c0 = rng.randint(0, width - pw)
        mask[r0 : r0 + ph, c0 : c0 + pw] = True
    else:
        ph = rng.randint(2 if height >= 2 else 1, min(height, 3))
        pw = rng.randint(2 if width >= 2 else 1, min(width, 3))
        r0 = rng.randint(0, height - ph)
        c0 = rng.randint(0, width - pw)
        mask[r0 : r0 + ph, c0] = True
        mask[r0 + ph - 1, c0 : c0 + pw] = True
    return mask


def _stamp_pattern(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    child_bbox: tuple[int, int, int, int] | None,
    color: int,
    depth: int,
    rng: random.Random,
) -> bool:
    candidates = _band_rectangles(frame_bbox, child_bbox)
    rng.shuffle(candidates)
    for r0, c0, r1, c1 in candidates:
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height <= 0 or width <= 0:
            continue
        tries = 20
        while tries > 0:
            tries -= 1
            mask = _random_pattern_mask(height, width, rng)
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue
            abs_cols = coords[:, 1] + c0
            if depth % 2 == 1:
                mirrored_cols = frame_bbox[1] + frame_bbox[3] - abs_cols
                if np.array_equal(np.sort(abs_cols), np.sort(mirrored_cols)):
                    continue
            grid[r0 : r1 + 1, c0 : c1 + 1][mask] = color
            return True
    return False


def _build_bboxes(rng: random.Random) -> list[tuple[int, int, int, int]]:
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    inner_h = rng.randint(*INNER_HEIGHT_RANGE)
    inner_w = rng.randint(*INNER_WIDTH_RANGE)
    sizes = [(inner_h, inner_w)]
    for _ in range(frame_count - 1):
        prev_h, prev_w = sizes[-1]
        inset_r = rng.randint(*INSET_RANGE)
        inset_c = rng.randint(*INSET_RANGE)
        sizes.append((prev_h + 2 * inset_r, prev_w + 2 * inset_c))
    sizes.reverse()

    outer_h, outer_w = sizes[0]
    margin_top = rng.randint(*OUTER_MARGIN_RANGE)
    margin_left = rng.randint(*OUTER_MARGIN_RANGE)
    margin_bottom = rng.randint(*OUTER_MARGIN_RANGE)
    margin_right = rng.randint(*OUTER_MARGIN_RANGE)
    total_h = outer_h + margin_top + margin_bottom
    total_w = outer_w + margin_left + margin_right

    bboxes: list[tuple[int, int, int, int]] = []
    r0 = margin_top
    c0 = margin_left
    for height, width in sizes:
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        bboxes.append((r0, c0, r1, c1))
        if (height, width) != sizes[-1]:
            next_h, next_w = sizes[bboxes.__len__()]
            dr = (height - next_h) // 2
            dc = (width - next_w) // 2
            r0 += dr
            c0 += dc
    assert bboxes[0] == (margin_top, margin_left, margin_top + outer_h - 1, margin_left + outer_w - 1)
    assert total_h >= bboxes[0][2] + 1 and total_w >= bboxes[0][3] + 1
    return bboxes, total_h, total_w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bboxes, total_h, total_w = _build_bboxes(rng)
    needed_colors = 2 * len(bboxes)
    if needed_colors > len(COLOR_POOL):
        return None
    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    frame_colors = colors[: len(bboxes)]
    pattern_colors = colors[len(bboxes) : needed_colors]

    grid = np.full((total_h, total_w), BG, dtype=int)
    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(grid, bbox, color)

    for depth, bbox in enumerate(bboxes):
        child_bbox = bboxes[depth + 1] if depth + 1 < len(bboxes) else None
        ok = _stamp_pattern(grid, bbox, child_bbox, pattern_colors[depth], depth, rng)
        if not ok:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    odd_depths = [depth for depth in range(len(bboxes)) if depth % 2 == 1]
    if not odd_depths:
        return None
    for depth in odd_depths:
        bbox = bboxes[depth]
        child_bbox = bboxes[depth + 1] if depth + 1 < len(bboxes) else None
        mask = _local_mask(grid.shape, bbox, [] if child_bbox is None else [child_bbox])
        if np.any(grid[mask] != out[mask]):
            break
    else:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
