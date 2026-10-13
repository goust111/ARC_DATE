import random
from collections import Counter

import numpy as np


GREY_COLOR = 5
LEGEND_ENTRIES = 5
LEGEND_BLOCK = 2
LEGEND_GAP = 1
LEGEND_HEIGHT = LEGEND_ENTRIES * LEGEND_BLOCK + (LEGEND_ENTRIES - 1) * LEGEND_GAP
LEGEND_WIDTH = LEGEND_BLOCK
AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
GEN_RETRIES = 400
SHAPE_RETRIES = 120
PLACE_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg_mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _component_submask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        sub[r - r0, c - c0] = True
    return sub


def _legend_noise(grid: np.ndarray, bg: int, r: int, c: int) -> int:
    h, w = grid.shape
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(h, r + LEGEND_HEIGHT + 1)
    c1 = min(w, c + LEGEND_WIDTH + 1)
    around = grid[r0:r1, c0:c1]
    total_non_bg = int(np.count_nonzero(around != bg))
    legend_non_bg = int(np.count_nonzero(grid[r : r + LEGEND_HEIGHT, c : c + LEGEND_WIDTH] != bg))
    return total_non_bg - legend_non_bg


def _corner_distance(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    return min(
        r + c,
        r + (w - (c + LEGEND_WIDTH)),
        (h - (r + LEGEND_HEIGHT)) + c,
        (h - (r + LEGEND_HEIGHT)) + (w - (c + LEGEND_WIDTH)),
    )


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates = []
    for r in range(h - LEGEND_HEIGHT + 1):
        for c in range(w - LEGEND_WIDTH + 1):
            colors = []
            ok = True
            for idx in range(LEGEND_ENTRIES):
                rr = r + idx * (LEGEND_BLOCK + LEGEND_GAP)
                block = grid[rr : rr + LEGEND_BLOCK, c : c + LEGEND_WIDTH]
                color = int(block[0, 0])
                if color == bg or not np.all(block == color):
                    ok = False
                    break
                colors.append(color)
                if idx < LEGEND_ENTRIES - 1:
                    gap = grid[rr + LEGEND_BLOCK, c : c + LEGEND_WIDTH]
                    if not np.all(gap == bg):
                        ok = False
                        break
            if not ok or len(set(colors)) != LEGEND_ENTRIES:
                continue
            score = (_corner_distance(grid, r, c), _legend_noise(grid, bg, r, c), r, c)
            candidates.append((score, r, c, colors))
    assert candidates, "legend not found"
    _, r, c, colors = min(candidates, key=lambda item: item[0])
    mask = np.zeros_like(grid, dtype=bool)
    for idx in range(LEGEND_ENTRIES):
        rr = r + idx * (LEGEND_BLOCK + LEGEND_GAP)
        mask[rr : rr + LEGEND_BLOCK, c : c + LEGEND_WIDTH] = True
    return {idx: int(color) for idx, color in enumerate(colors)}, mask


def _grey_color(grid: np.ndarray, bg: int, legend_colors: set[int], legend_mask: np.ndarray) -> int:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg or color in legend_colors:
            continue
        count = int(np.count_nonzero((grid == color) & (~legend_mask)))
        if count:
            candidates.append((count, color))
    assert candidates, "grey object color not found"
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey closed shapes using the corner legend indexed by hole count.

    Input structure:
    - The background is black, with several disconnected grey closed shapes drawn as 4-connected outlines or filled regions.
    - Each grey shape may contain 0 to 4 enclosed background holes.
    - One corner contains a vertical legend made of five 2x2 color blocks with one background row between consecutive blocks.

    Rule:
    - Read the legend from top to bottom so entry 0, 1, 2, 3, 4 gives the color for shapes with that many holes.
    - For each grey connected component, count the number of enclosed background regions inside its own bounding box.
    - Recolor every pixel of that grey component with the legend color for its hole count, while leaving the holes and the legend unchanged.

    Key insight:
    - The shapes are distinguished by topology rather than geometry: the enclosed-hole count selects a legend entry, and only the grey boundary/fill pixels are repainted.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend, legend_mask = _find_legend(g, bg)
    assert sorted(legend) == list(range(LEGEND_ENTRIES)), "legend must cover hole counts 0..4"

    grey = _grey_color(g, bg, set(legend.values()), legend_mask)
    grey_mask = (g == grey) & (~legend_mask)

    out = g.copy()
    for comp in _components(grey_mask):
        holes = _hole_count(_component_submask(comp))
        assert holes in legend, f"missing legend entry for hole count {holes}"
        color = int(legend[holes])
        for r, c in comp:
            out[r, c] = color
    return out


def _paint_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _attach_protrusions(mask: np.ndarray, max_count: int) -> np.ndarray:
    result = mask.copy()
    count = random.randint(0, max_count)
    for _ in range(count):
        pts = np.argwhere(result)
        r0 = int(pts[:, 0].min())
        c0 = int(pts[:, 1].min())
        r1 = int(pts[:, 0].max())
        c1 = int(pts[:, 1].max())
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            cols = [c for c in range(c0, c1 + 1) if result[r0, c]]
            if not cols or r0 == 0:
                continue
            seg = random.choice(_runs_from_line(cols))
            width = random.randint(1, min(3, seg[1] - seg[0] + 1))
            start = random.randint(seg[0], seg[1] - width + 1)
            depth = random.randint(1, min(2, r0))
            _paint_rect(result, r0 - depth, start, r0 - 1, start + width - 1)
        elif side == "bottom":
            cols = [c for c in range(c0, c1 + 1) if result[r1, c]]
            if not cols or r1 == result.shape[0] - 1:
                continue
            seg = random.choice(_runs_from_line(cols))
            width = random.randint(1, min(3, seg[1] - seg[0] + 1))
            start = random.randint(seg[0], seg[1] - width + 1)
            depth = random.randint(1, min(2, result.shape[0] - 1 - r1))
            _paint_rect(result, r1 + 1, start, r1 + depth, start + width - 1)
        elif side == "left":
            rows = [r for r in range(r0, r1 + 1) if result[r, c0]]
            if not rows or c0 == 0:
                continue
            seg = random.choice(_runs_from_line(rows))
            height = random.randint(1, min(3, seg[1] - seg[0] + 1))
            start = random.randint(seg[0], seg[1] - height + 1)
            depth = random.randint(1, min(2, c0))
            _paint_rect(result, start, c0 - depth, start + height - 1, c0 - 1)
        else:
            rows = [r for r in range(r0, r1 + 1) if result[r, c1]]
            if not rows or c1 == result.shape[1] - 1:
                continue
            seg = random.choice(_runs_from_line(rows))
            height = random.randint(1, min(3, seg[1] - seg[0] + 1))
            start = random.randint(seg[0], seg[1] - height + 1)
            depth = random.randint(1, min(2, result.shape[1] - 1 - c1))
            _paint_rect(result, start, c1 + 1, start + height - 1, c1 + depth)
    return result


def _runs_from_line(values: list[int]) -> list[tuple[int, int]]:
    runs = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _make_partition_shape(hole_count: int) -> np.ndarray:
    pad = 3
    if hole_count == 1:
        mode = random.choice(["vertical", "horizontal"])
    elif hole_count in (2, 3):
        mode = random.choice(["vertical", "horizontal"])
    else:
        mode = random.choices(["grid4", "vertical", "horizontal"], weights=[4, 1, 1], k=1)[0]

    if mode == "vertical":
        chamber_widths = [random.randint(1, 2 + (1 if hole_count == 1 else 0)) for _ in range(hole_count)]
        inner_h = random.randint(2, 4)
        width = 2 + sum(chamber_widths) + (hole_count - 1)
        height = inner_h + 2
        mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=bool)
        r0 = c0 = pad
        mask[r0, c0 : c0 + width] = True
        mask[r0 + height - 1, c0 : c0 + width] = True
        mask[r0 : r0 + height, c0] = True
        mask[r0 : r0 + height, c0 + width - 1] = True
        cursor = c0 + 1 + chamber_widths[0]
        for idx in range(hole_count - 1):
            mask[r0 : r0 + height, cursor] = True
            cursor += 1 + chamber_widths[idx + 1]
    elif mode == "horizontal":
        chamber_heights = [random.randint(1, 2 + (1 if hole_count == 1 else 0)) for _ in range(hole_count)]
        inner_w = random.randint(2, 4)
        height = 2 + sum(chamber_heights) + (hole_count - 1)
        width = inner_w + 2
        mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=bool)
        r0 = c0 = pad
        mask[r0, c0 : c0 + width] = True
        mask[r0 + height - 1, c0 : c0 + width] = True
        mask[r0 : r0 + height, c0] = True
        mask[r0 : r0 + height, c0 + width - 1] = True
        cursor = r0 + 1 + chamber_heights[0]
        for idx in range(hole_count - 1):
            mask[cursor, c0 : c0 + width] = True
            cursor += 1 + chamber_heights[idx + 1]
    else:
        left = random.randint(1, 3)
        right = random.randint(1, 3)
        top = random.randint(1, 3)
        bottom = random.randint(1, 3)
        width = 2 + left + 1 + right
        height = 2 + top + 1 + bottom
        mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=bool)
        r0 = c0 = pad
        mask[r0, c0 : c0 + width] = True
        mask[r0 + height - 1, c0 : c0 + width] = True
        mask[r0 : r0 + height, c0] = True
        mask[r0 : r0 + height, c0 + width - 1] = True
        mask[r0 : r0 + height, c0 + 1 + left] = True
        mask[r0 + 1 + top, c0 : c0 + width] = True

    return mask


def _make_zero_hole_shape() -> np.ndarray:
    pad = 3
    mode = random.choice(["rect", "plus", "diamond"])
    if mode == "rect":
        height = random.randint(3, 7)
        width = random.randint(3, 7)
        mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=bool)
        mask[pad : pad + height, pad : pad + width] = True
        return mask
    if mode == "plus":
        height = random.randint(5, 9)
        width = random.randint(5, 9)
        mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=bool)
        bar_h = random.randint(1, 2)
        bar_w = random.randint(1, 2)
        mid_r = pad + height // 2
        mid_c = pad + width // 2
        r0 = mid_r - bar_h // 2
        c0 = mid_c - bar_w // 2
        mask[r0 : r0 + bar_h, pad : pad + width] = True
        mask[pad : pad + height, c0 : c0 + bar_w] = True
        return mask
    radius = random.randint(2, 4)
    side = 2 * radius + 1
    mask = np.zeros((side + 2 * pad, side + 2 * pad), dtype=bool)
    center = pad + radius
    for r in range(side):
        for c in range(side):
            if abs(r - radius) + abs(c - radius) <= radius:
                mask[pad + r, pad + c] = True
    return mask


def _make_shape(hole_count: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        if hole_count == 0:
            mask = _make_zero_hole_shape()
            mask = _attach_protrusions(mask, max_count=3)
        else:
            mask = _make_partition_shape(hole_count)
            mask = _attach_protrusions(mask, max_count=2)
        if random.random() < 0.5:
            mask = np.rot90(mask, random.randint(0, 3))
        mask = _crop_mask(mask)
        if len(_components(mask)) == 1 and _hole_count(mask) == hole_count:
            return mask
    raise RuntimeError(f"failed to build shape with {hole_count} holes")


def _legend_origin(shape: tuple[int, int], corner: str, margin_r: int, margin_c: int) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return margin_r, margin_c
    if corner == "tr":
        return margin_r, w - margin_c - LEGEND_WIDTH
    if corner == "bl":
        return h - margin_r - LEGEND_HEIGHT, margin_c
    return h - margin_r - LEGEND_HEIGHT, w - margin_c - LEGEND_WIDTH


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _place_legend(grid: np.ndarray, blocked: np.ndarray, colors: list[int], corner: str) -> None:
    margin_r = random.randint(0, 2)
    margin_c = random.randint(0, 2)
    r0, c0 = _legend_origin(grid.shape, corner, margin_r, margin_c)
    for idx, color in enumerate(colors):
        rr = r0 + idx * (LEGEND_BLOCK + LEGEND_GAP)
        grid[rr : rr + LEGEND_BLOCK, c0 : c0 + LEGEND_WIDTH] = color
    _mark_blocked(blocked, r0, c0, LEGEND_HEIGHT, LEGEND_WIDTH, pad=2)


def _try_place_objects(grid: np.ndarray, blocked: np.ndarray, shapes: list[np.ndarray]) -> bool:
    h, w = grid.shape
    for mask in shapes:
        mh, mw = mask.shape
        placed = False
        min_r = 2
        min_c = 2
        max_r = h - mh - 2
        max_c = w - mw - 2
        if max_r < min_r or max_c < min_c:
            return False
        for _ in range(PLACE_RETRIES):
            top = random.randint(min_r, max_r)
            left = random.randint(min_c, max_c)
            if blocked[top - 1 : top + mh + 1, left - 1 : left + mw + 1].any():
                continue
            region = grid[top : top + mh, left : left + mw]
            region[mask] = GREY_COLOR
            _mark_blocked(blocked, top, left, mh, mw, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_generated(grid: np.ndarray, expected_holes: list[int], legend_colors: list[int]) -> None:
    bg = _background_color(grid)
    legend, legend_mask = _find_legend(grid, bg)
    assert [legend[i] for i in range(LEGEND_ENTRIES)] == legend_colors
    grey = _grey_color(grid, bg, set(legend.values()), legend_mask)
    comps = _components((grid == grey) & (~legend_mask))
    holes = sorted(_hole_count(_component_submask(comp)) for comp in comps)
    assert holes == sorted(expected_holes)
    out = main(grid)
    assert out.shape == grid.shape


def _gen_attempt() -> np.ndarray | None:
    object_count = 5 + random.randint(0, 1)
    hole_counts = list(range(LEGEND_ENTRIES))
    if object_count > LEGEND_ENTRIES:
        hole_counts.append(random.randint(0, LEGEND_ENTRIES - 1))
    random.shuffle(hole_counts)

    shapes = []
    for hole_count in hole_counts:
        shapes.append(_make_shape(hole_count))
    shapes.sort(key=lambda mask: mask.shape[0] * mask.shape[1], reverse=True)

    padded_area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in shapes) + (LEGEND_HEIGHT + 4) * (LEGEND_WIDTH + 4)
    base = max(
        GRID_SIDE_MIN,
        LEGEND_HEIGHT + 3,
        int(np.ceil(np.sqrt(padded_area))) + random.randint(2, 5),
    )
    if base > GRID_SIDE_MAX:
        return None

    grid_h = random.randint(base, GRID_SIDE_MAX)
    grid_w = random.randint(base, GRID_SIDE_MAX)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    legend_colors = random.sample(AVAILABLE_COLORS, LEGEND_ENTRIES)
    corner = random.choice(["tl", "tr", "bl", "br"])
    _place_legend(grid, blocked, legend_colors, corner)

    if not _try_place_objects(grid, blocked, shapes):
        return None

    try:
        _validate_generated(grid, hole_counts, legend_colors)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
