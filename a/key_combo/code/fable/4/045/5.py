import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _extract_frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_mask = grid == GREY
    components = _find_components(grey_mask)
    assert len(components) >= 2, "expected at least two grey frames"

    boxes: list[tuple[int, int, int, int]] = []
    for comp in components:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each frame must enclose a black interior"
        expected = _border_mask(grid.shape, bbox)
        comp_mask = np.zeros_like(grey_mask)
        for r, c in comp:
            comp_mask[r, c] = True
        assert np.array_equal(comp_mask, expected), "grey components must be one-cell-thick hollow rectangles"
        boxes.append(bbox)

    boxes.sort(key=lambda b: ((b[2] - b[0] + 1) * (b[3] - b[1] + 1)), reverse=True)
    for outer, inner in zip(boxes, boxes[1:]):
        assert outer[0] < inner[0] < inner[2] < outer[2]
        assert outer[1] < inner[1] < inner[3] < outer[3]
    return boxes


def _band_mask(
    shape: tuple[int, int], outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> np.ndarray:
    r0o, c0o, r1o, c1o = outer
    r0i, c0i, r1i, c1i = inner
    mask = np.zeros(shape, dtype=bool)
    mask[r0o + 1 : r1o, c0o + 1 : c1o] = True
    mask[r0i : r1i + 1, c0i : c1i + 1] = False
    return mask


def _analyze_bands(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], list[int], list[np.ndarray], list[int]]:
    boxes = _extract_frame_bboxes(grid)
    band_count = len(boxes) - 1
    assert band_count >= 2, "this task needs at least two bands"

    widths: list[int] = []
    masks: list[np.ndarray] = []
    for outer, inner in zip(boxes, boxes[1:]):
        deltas = (
            inner[0] - outer[0] - 1,
            inner[1] - outer[1] - 1,
            outer[2] - inner[2] - 1,
            outer[3] - inner[3] - 1,
        )
        assert min(deltas) >= 1
        assert len(set(deltas)) == 1, "adjacent frames must stay equally spaced on all four sides"
        width = deltas[0]
        assert width <= band_count, "band width must point to an existing band index"
        widths.append(width)
        band = _band_mask(grid.shape, outer, inner)
        assert np.any(band), "band must contain black cells for its seed"
        masks.append(band)

    seed_mask = (grid != BLACK) & (grid != GREY)
    seed_coords = np.argwhere(seed_mask)
    assert len(seed_coords) == band_count, "expected exactly one colored seed per band"

    seed_colors: list[int] = []
    used_colors: set[int] = set()
    covered = np.zeros_like(seed_mask)
    for band in masks:
        seeds = np.argwhere(seed_mask & band)
        assert len(seeds) == 1, "each band must contain exactly one seed"
        r, c = map(int, seeds[0])
        color = int(grid[r, c])
        assert color not in used_colors, "seed colors must be distinct"
        used_colors.add(color)
        seed_colors.append(color)
        covered |= band

    assert not np.any(seed_mask & ~covered), "seeds must stay inside bands"
    return boxes, widths, masks, seed_colors


def _paint_bands(grid: np.ndarray, widths: list[int], bands: list[np.ndarray], seed_colors: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for width, band in zip(widths, bands):
        out[band] = seed_colors[width - 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each concentric band by following its width pointer to another band's seed color.

    Input structure:
    - Black background with several concentric one-cell-thick grey hollow rectangles.
    - Each region between two adjacent frames is a band, ordered from outermost to innermost.
    - Every band contains exactly one non-grey colored seed pixel, and the seed colors are all distinct.

    Rule:
    - Measure the width of each band as the equal gap between its outer and inner grey frames.
    - For band d with width w, repaint the whole band with the seed color found in band w.
    - Keep all grey frames unchanged, and leave the area outside the outermost frame and inside the innermost frame black.

    Key insight:
    - A band's own seed is usually a decoy: width acts as a visible pointer that redirects the band to another band's seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _boxes, widths, bands, seed_colors = _analyze_bands(g)
    return _paint_bands(g, widths, bands, seed_colors)


def _build_bboxes(widths: list[int], inner_h: int, inner_w: int) -> list[tuple[int, int, int, int]]:
    frames_rev = [(0, 0, inner_h - 1, inner_w - 1)]
    current = frames_rev[0]
    for width in reversed(widths):
        r0, c0, r1, c1 = current
        current = (r0 - width - 1, c0 - width - 1, r1 + width + 1, c1 + width + 1)
        frames_rev.append(current)
    frames = list(reversed(frames_rev))

    min_r = min(b[0] for b in frames)
    min_c = min(b[1] for b in frames)
    if min_r != 0 or min_c != 0:
        frames = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in frames]
    return frames


def _choose_widths() -> list[int] | None:
    band_count = random.randint(2, 5)
    widths = [random.randint(1, band_count) for _ in range(band_count)]
    if all(width == idx for idx, width in enumerate(widths, start=1)):
        j = random.randrange(band_count)
        choices = [v for v in range(1, band_count + 1) if v != j + 1]
        widths[j] = random.choice(choices)
    return widths


def _gen_attempt() -> np.ndarray | None:
    widths = _choose_widths()
    band_count = len(widths)
    max_sum = 12
    if sum(widths) + band_count > max_sum:
        return None

    inner_h = random.randint(3, 7)
    inner_w = random.randint(3, 7)
    frames = _build_bboxes(widths, inner_h, inner_w)
    outer = frames[0]
    outer_h = outer[2] - outer[0] + 1
    outer_w = outer[3] - outer[1] + 1

    pad_top = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    remaining_h = 30 - outer_h - pad_top
    remaining_w = 30 - outer_w - pad_left
    if remaining_h < 1 or remaining_w < 1:
        return None
    pad_bottom = random.randint(1, min(3, remaining_h))
    pad_right = random.randint(1, min(3, remaining_w))
    grid_h = outer_h + pad_top + pad_bottom
    grid_w = outer_w + pad_left + pad_right

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    shifted_frames = [
        (r0 + pad_top, c0 + pad_left, r1 + pad_top, c1 + pad_left) for r0, c0, r1, c1 in frames
    ]
    for bbox in shifted_frames:
        grid[_border_mask(grid.shape, bbox)] = GREY

    band_masks = [
        _band_mask(grid.shape, outer_bbox, inner_bbox)
        for outer_bbox, inner_bbox in zip(shifted_frames, shifted_frames[1:])
    ]
    if any(int(mask.sum()) < 2 for mask in band_masks):
        return None

    colors = random.sample(SEED_COLORS, band_count)
    for color, mask in zip(colors, band_masks):
        cells = np.argwhere(mask)
        r, c = map(int, cells[random.randrange(len(cells))])
        grid[r, c] = color

    if not any(width != idx for idx, width in enumerate(widths, start=1)):
        return None

    out = _paint_bands(grid, widths, band_masks, colors)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero((grid != BLACK) & (grid != GREY)) != band_count:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        _boxes, widths, bands, seed_colors = _analyze_bands(grid)
        expected = _paint_bands(grid, widths, bands, seed_colors)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
