import random
from collections import deque

import numpy as np


BG = 0
DIR_ORDER = ("up", "right", "down", "left")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates = []
    for pts in _components_mask(grid != BG):
        if len(pts) != 8:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if (bh, bw) not in ((4, 2), (2, 4)):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] != BG):
            continue
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if touches_corner:
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one 2x4/4x2 corner legend"
    return candidates[0]


def _legend_mapping(legend: np.ndarray) -> dict[str, tuple[int, int]]:
    mapping: dict[str, tuple[int, int]] = {}
    if legend.shape == (4, 2):
        for i, direction in enumerate(DIR_ORDER):
            mapping[direction] = (int(legend[i, 0]), int(legend[i, 1]))
    elif legend.shape == (2, 4):
        for i, direction in enumerate(DIR_ORDER):
            mapping[direction] = (int(legend[0, i]), int(legend[1, i]))
    else:
        raise AssertionError("legend must be 4x2 or 2x4")
    return mapping


def _frame_info(grid: np.ndarray, points: list[tuple[int, int]]):
    r0, c0, r1, c1 = _bbox(points)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    present = sub != BG
    h, w = sub.shape

    if h < 3 or w < 3:
        return None
    if np.any(present[1:-1, 1:-1]):
        return None

    colors = np.unique(sub[present])
    if colors.size != 1:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    matches = []
    if w % 2 == 1:
        mid_c = w // 2
        top_gap = border.copy()
        top_gap[0, mid_c] = False
        if np.array_equal(present, top_gap):
            matches.append(("up", (r0, c0 + mid_c), (r0, c0, r1, c1)))
        bottom_gap = border.copy()
        bottom_gap[h - 1, mid_c] = False
        if np.array_equal(present, bottom_gap):
            matches.append(("down", (r1, c0 + mid_c), (r0, c0, r1, c1)))
    if h % 2 == 1:
        mid_r = h // 2
        left_gap = border.copy()
        left_gap[mid_r, 0] = False
        if np.array_equal(present, left_gap):
            matches.append(("left", (r0 + mid_r, c0), (r0, c0, r1, c1)))
        right_gap = border.copy()
        right_gap[mid_r, w - 1] = False
        if np.array_equal(present, right_gap):
            matches.append(("right", (r0 + mid_r, c1), (r0, c0, r1, c1)))

    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each frame's gap direction as a lookup key into the corner legend.

    The legend lists directions in fixed order: up, right, down, left.
    In a 4x2 legend each row is [marker, fill].
    In a 2x4 legend the top row is marker colors and the bottom row is fill colors.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    lr0, lc0, lr1, lc1 = _legend_bbox(g)
    legend = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
    mapping = _legend_mapping(legend)

    work = g.copy()
    work[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG

    frame_count = 0
    for pts in _components_mask(work != BG):
        info = _frame_info(work, pts)
        if info is None:
            continue
        direction, gap, (r0, c0, r1, c1) = info
        marker_color, fill_color = mapping[direction]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill_color
        out[gap] = marker_color
        frame_count += 1

    assert frame_count > 0, "expected at least one gapped rectangular frame"
    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
FRAME_COUNT_RANGE = (2, 4)
FRAME_HEIGHT_RANGE = (4, 8)
FRAME_WIDTH_RANGE = (4, 8)
PLACEMENT_RETRIES = 300
GEN_RETRIES = 400


def _legend_patch(mapping: dict[str, tuple[int, int]], orientation: str) -> np.ndarray:
    if orientation == "vertical":
        patch = np.zeros((4, 2), dtype=int)
        for i, direction in enumerate(DIR_ORDER):
            marker, fill = mapping[direction]
            patch[i, 0] = marker
            patch[i, 1] = fill
        return patch
    if orientation == "horizontal":
        patch = np.zeros((2, 4), dtype=int)
        for i, direction in enumerate(DIR_ORDER):
            marker, fill = mapping[direction]
            patch[0, i] = marker
            patch[1, i] = fill
        return patch
    raise ValueError(orientation)


def _corner_origin(shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    ph, pw = patch_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - pw
    if corner == "bl":
        return h - ph, 0
    if corner == "br":
        return h - ph, w - pw
    raise ValueError(corner)


def _mark_forbidden(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1):
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_frame_shape(direction: str) -> tuple[int, int]:
    if direction in ("up", "down"):
        height = random.randint(*FRAME_HEIGHT_RANGE)
        width_choices = [w for w in range(FRAME_WIDTH_RANGE[0], FRAME_WIDTH_RANGE[1] + 1) if w % 2 == 1]
        width = random.choice(width_choices)
    else:
        width = random.randint(*FRAME_WIDTH_RANGE)
        height_choices = [h for h in range(FRAME_HEIGHT_RANGE[0], FRAME_HEIGHT_RANGE[1] + 1) if h % 2 == 1]
        height = random.choice(height_choices)
    return height, width


def _frame_patch(height: int, width: int, direction: str, border_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    if direction == "up":
        gap = (0, width // 2)
    elif direction == "right":
        gap = (height // 2, width - 1)
    elif direction == "down":
        gap = (height - 1, width // 2)
    elif direction == "left":
        gap = (height // 2, 0)
    else:
        raise ValueError(direction)
    patch[gap] = BG
    return patch, gap


def _place_frame(
    grid: np.ndarray,
    expected: np.ndarray,
    forbidden: np.ndarray,
    direction: str,
    border_color: int,
    marker_color: int,
    fill_color: int,
) -> bool:
    h, w = grid.shape
    frame_h, frame_w = _sample_frame_shape(direction)
    patch, gap = _frame_patch(frame_h, frame_w, direction, border_color)

    if h - frame_h < 3 or w - frame_w < 3:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(1, h - frame_h - 1)
        c0 = random.randint(1, w - frame_w - 1)
        if forbidden[r0 : r0 + frame_h, c0 : c0 + frame_w].any():
            continue

        grid[r0 : r0 + frame_h, c0 : c0 + frame_w] = patch
        expected[r0 : r0 + frame_h, c0 : c0 + frame_w] = patch
        expected[r0 + 1 : r0 + frame_h - 1, c0 + 1 : c0 + frame_w - 1] = fill_color
        expected[r0 + gap[0], c0 + gap[1]] = marker_color
        _mark_forbidden(forbidden, (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1))
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    legend_colors = random.sample(range(1, 10), 8)
    markers = legend_colors[:4]
    fills = legend_colors[4:]
    mapping = {direction: (markers[i], fills[i]) for i, direction in enumerate(DIR_ORDER)}

    orientation = random.choice(["vertical", "horizontal"])
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend = _legend_patch(mapping, orientation)
    lr0, lc0 = _corner_origin((h, w), legend.shape, corner)
    lph, lpw = legend.shape
    grid[lr0 : lr0 + lph, lc0 : lc0 + lpw] = legend
    _mark_forbidden(forbidden, (lr0, lc0, lr0 + lph - 1, lc0 + lpw - 1))

    frame_count = random.randint(*FRAME_COUNT_RANGE)
    directions = [random.choice(DIR_ORDER) for _ in range(frame_count)]
    if len(set(directions)) == 1:
        directions[-1] = random.choice([d for d in DIR_ORDER if d != directions[0]])
    random.shuffle(directions)

    used_border_colors: set[int] = set()
    for direction in directions:
        marker_color, fill_color = mapping[direction]
        choices = [c for c in range(1, 10) if c not in {marker_color, fill_color} and c not in used_border_colors]
        if not choices:
            return None
        border_color = random.choice(choices)
        used_border_colors.add(border_color)
        ok = _place_frame(grid, expected, forbidden, direction, border_color, marker_color, fill_color)
        if not ok:
            return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
