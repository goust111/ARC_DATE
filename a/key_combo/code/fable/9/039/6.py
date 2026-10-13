import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairSpec:
    color: int
    r0: int
    c0: int
    h: int
    w: int
    marker: tuple[int, int]

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r0 + self.h - 1, self.c0 + self.w - 1


_PALETTE = tuple(range(10))
_GRID_H_RANGE = (18, 30)
_GRID_W_RANGE = (18, 30)
_RECT_H_RANGE = (2, 4)
_RECT_W_RANGE = (2, 5)
_DIAG_LEN_RANGE = (2, 5)
_STRAIGHT_LEN_RANGE = (1, 4)
_MAX_GENERATION_ATTEMPTS = 500
_QUADRANTS = {
    "nw": (-1, -1),
    "ne": (-1, 1),
    "sw": (1, -1),
    "se": (1, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(dst.shape[0], r0 + h)
    cc1 = min(dst.shape[1], c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - r0
    pc0 = cc0 - c0
    dst[rr0:rr1, cc0:cc1] = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _corner_candidates(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _nearest_corner(bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> tuple[int, int]:
    distances = []
    for corner in _corner_candidates(bbox):
        dist = abs(marker[0] - corner[0]) + abs(marker[1] - corner[1])
        distances.append((dist, corner))
    distances.sort()
    assert len(distances) >= 2 and distances[0][0] < distances[1][0], "nearest corner must be unique"
    return distances[0][1]


def _sign(x: int) -> int:
    return 1 if x > 0 else -1


def _decompose_vector(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    diag_len = min(abs_dr, abs_dc)
    d = (_sign(dr) * diag_len, _sign(dc) * diag_len)
    if abs_dr > abs_dc:
        a = (_sign(dr) * (abs_dr - abs_dc), 0)
    elif abs_dc > abs_dr:
        a = (0, _sign(dc) * (abs_dc - abs_dr))
    else:
        a = (0, 0)
    return d, a


def _offsets_until_exit(
    shape: tuple[int, int], top_left: tuple[int, int], sprite_shape: tuple[int, int], d: tuple[int, int], a: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = sprite_shape
    cur_r = 0
    cur_c = 0
    offsets: list[tuple[int, int]] = []
    if a == (0, 0):
        steps = [d]
    else:
        steps = [d, a]
    step_idx = 0
    for _ in range(64):
        step = steps[step_idx % len(steps)]
        step_idx += 1
        cur_r += step[0]
        cur_c += step[1]
        rr = top_left[0] + cur_r
        cc = top_left[1] + cur_c
        if rr >= shape[0] or cc >= shape[1] or rr + h <= 0 or cc + w <= 0:
            break
        offsets.append((cur_r, cur_c))
    return offsets


def _project_mask(shape: tuple[int, int], pair: PairSpec) -> tuple[np.ndarray, dict]:
    mask = np.zeros(shape, dtype=bool)
    mask[pair.r0 : pair.r0 + pair.h, pair.c0 : pair.c0 + pair.w] = True
    corner = _nearest_corner(pair.bbox, pair.marker)
    d, a = _decompose_vector(pair.marker[0] - corner[0], pair.marker[1] - corner[1])
    offsets = _offsets_until_exit(shape, (pair.r0, pair.c0), (pair.h, pair.w), d, a)
    clipped = 0
    for dr, dc in offsets:
        rr = pair.r0 + dr
        cc = pair.c0 + dc
        rr0 = max(0, rr)
        cc0 = max(0, cc)
        rr1 = min(shape[0], rr + pair.h)
        cc1 = min(shape[1], cc + pair.w)
        if rr0 <= rr or cc0 <= cc:
            pass
        if rr0 > rr or cc0 > cc or rr1 < rr + pair.h or cc1 < cc + pair.w:
            clipped += 1
        mask[rr0:rr1, cc0:cc1] = True
    stats = {"offsets": offsets, "d": d, "a": a, "clipped": clipped}
    return mask, stats


def _scene_from_grid(grid: np.ndarray) -> tuple[int, list[PairSpec]]:
    bg = _get_bg(grid)
    pairs: list[PairSpec] = []
    for color in _colors_present(grid, exclude={bg}):
        comps = _components_for_color(grid, color)
        assert len(comps) == 2, "each foreground color must form one rectangle and one marker"
        singles = [comp for comp in comps if len(comp) == 1]
        rects = [comp for comp in comps if len(comp) > 1]
        assert len(singles) == 1 and len(rects) == 1, "each color must have one singleton marker and one rectangle"
        rect = rects[0]
        r0, c0, r1, c1 = _bbox(rect)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert len(rect) == h * w, "foreground component must be a solid rectangle"
        marker = singles[0][0]
        corner = _nearest_corner((r0, c0, r1, c1), marker)
        dr = marker[0] - corner[0]
        dc = marker[1] - corner[1]
        assert dr != 0 and dc != 0, "marker must lie diagonally away from its matched corner"
        pairs.append(PairSpec(color=color, r0=r0, c0=c0, h=h, w=w, marker=marker))
    return bg, sorted(pairs, key=lambda pair: pair.color)


def _render_scene(shape: tuple[int, int], bg: int, pairs: list[PairSpec], mode: str = "correct") -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for pair in sorted(pairs, key=lambda item: item.color):
        patch = np.full((pair.h, pair.w), pair.color, dtype=int)
        out[pair.r0 : pair.r0 + pair.h, pair.c0 : pair.c0 + pair.w] = pair.color
        corner = _nearest_corner(pair.bbox, pair.marker)
        v = (pair.marker[0] - corner[0], pair.marker[1] - corner[1])
        d, a = _decompose_vector(v[0], v[1])
        if mode == "correct":
            offsets = _offsets_until_exit(shape, (pair.r0, pair.c0), (pair.h, pair.w), d, a)
        elif mode == "vector":
            offsets = _offsets_until_exit(shape, (pair.r0, pair.c0), (pair.h, pair.w), v, (0, 0))
        elif mode == "diagonal":
            offsets = _offsets_until_exit(shape, (pair.r0, pair.c0), (pair.h, pair.w), d, (0, 0))
        else:
            raise ValueError(mode)
        for dr, dc in offsets:
            _paste_clip(out, patch, pair.r0 + dr, pair.c0 + dc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each solid rectangle along an alternating diagonal-plus-axis staircase so its same-color marker lands on the corresponding corner after one full period.

    Input structure:
    - A dominant background contains several foreground colors.
    - Each foreground color appears in exactly two disconnected objects: one solid axis-aligned rectangle and one isolated marker pixel.
    - The marker sits diagonally away from the rectangle, so the nearest rectangle corner to that marker is unique and differs in both row and column.

    Rule:
    - For each color, match the singleton marker with the solid rectangle of the same color.
    - Let v be the vector from the rectangle corner nearest the marker to the marker itself. Decompose v into a 45-degree diagonal step d with length `min(|dr|, |dc|)` and a remaining horizontal or vertical step a.
    - Keep the original rectangle, then stamp full rectangle copies at offsets `d, d+a, 2d+a, 2d+2a, ...`, clipping any out-of-bounds parts and stopping when the next copy is fully outside the grid. In the pure-diagonal case `a=0`, this becomes repeated diagonal stamping.

    Key insight:
    - The marker does not mark the final destination of a simple whole-vector translation; it marks the corner reached after one diagonal beat and one straight beat of the alternating staircase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, pairs = _scene_from_grid(g)
    return _render_scene(g.shape, bg, pairs, mode="correct")


def _choose_modes(count: int) -> list[str]:
    if count == 2:
        modes = ["row", "col"]
    elif count == 3:
        modes = ["row", "col", "equal" if random.random() < 0.6 else random.choice(["row", "col"])]
    else:
        modes = ["row", "col", "equal", random.choice(["row", "col"])]
    random.shuffle(modes)
    return modes


def _position_ranges(h: int, w: int, rect_h: int, rect_w: int, quadrant: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    row_mid = h // 2
    col_mid = w // 2
    top_rows = (2, row_mid - rect_h - 1)
    bottom_rows = (row_mid + 1, h - rect_h - 3)
    left_cols = (2, col_mid - rect_w - 1)
    right_cols = (col_mid + 1, w - rect_w - 3)
    if quadrant == "nw":
        row_range, col_range = top_rows, left_cols
    elif quadrant == "ne":
        row_range, col_range = top_rows, right_cols
    elif quadrant == "sw":
        row_range, col_range = bottom_rows, left_cols
    else:
        row_range, col_range = bottom_rows, right_cols
    if row_range[0] > row_range[1] or col_range[0] > col_range[1]:
        return None
    return row_range, col_range


def _pick_pair(shape: tuple[int, int], quadrant: str, color: int, mode: str) -> PairSpec | None:
    h, w = shape
    sr, sc = _QUADRANTS[quadrant]
    for _ in range(120):
        rect_h = random.randint(*_RECT_H_RANGE)
        rect_w = random.randint(*_RECT_W_RANGE)
        ranges = _position_ranges(h, w, rect_h, rect_w, quadrant)
        if ranges is None:
            continue
        row_range, col_range = ranges
        r0 = random.randint(*row_range)
        c0 = random.randint(*col_range)
        corner_r = r0 if sr < 0 else r0 + rect_h - 1
        corner_c = c0 if sc < 0 else c0 + rect_w - 1
        row_space = corner_r if sr < 0 else h - 1 - corner_r
        col_space = corner_c if sc < 0 else w - 1 - corner_c
        diag_cap = min(row_space, col_space, _DIAG_LEN_RANGE[1])
        if diag_cap < _DIAG_LEN_RANGE[0]:
            continue
        diag_len = random.randint(_DIAG_LEN_RANGE[0], diag_cap)
        if mode == "equal":
            abs_dr = diag_len
            abs_dc = diag_len
        elif mode == "row":
            straight_cap = min(_STRAIGHT_LEN_RANGE[1], row_space - diag_len)
            if straight_cap < _STRAIGHT_LEN_RANGE[0]:
                continue
            abs_dr = diag_len + random.randint(_STRAIGHT_LEN_RANGE[0], straight_cap)
            abs_dc = diag_len
        else:
            straight_cap = min(_STRAIGHT_LEN_RANGE[1], col_space - diag_len)
            if straight_cap < _STRAIGHT_LEN_RANGE[0]:
                continue
            abs_dr = diag_len
            abs_dc = diag_len + random.randint(_STRAIGHT_LEN_RANGE[0], straight_cap)
        marker = (corner_r + sr * abs_dr, corner_c + sc * abs_dc)
        pair = PairSpec(color=color, r0=r0, c0=c0, h=rect_h, w=rect_w, marker=marker)
        _, stats = _project_mask(shape, pair)
        if len(stats["offsets"]) < 2:
            continue
        if mode != "equal" and len(stats["offsets"]) < 3:
            continue
        return pair
    return None


def _input_mask(pair: PairSpec, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[pair.r0 : pair.r0 + pair.h, pair.c0 : pair.c0 + pair.w] = True
    mask[pair.marker] = True
    return mask


def _well_separated(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            shifted = np.zeros_like(mask_a)
            r_src0 = max(0, -dr)
            r_src1 = min(mask_a.shape[0], mask_a.shape[0] - dr)
            c_src0 = max(0, -dc)
            c_src1 = min(mask_a.shape[1], mask_a.shape[1] - dc)
            shifted[r_src0 + dr : r_src1 + dr, c_src0 + dc : c_src1 + dc] = mask_a[r_src0:r_src1, c_src0:c_src1]
            if np.any(shifted & mask_b):
                return False
    return True


def _build_candidate_scene() -> np.ndarray | None:
    h = random.randint(*_GRID_H_RANGE)
    w = random.randint(*_GRID_W_RANGE)
    count = random.randint(2, 4)
    bg = random.choice(_PALETTE)
    fg_choices = [color for color in _PALETTE if color != bg]
    colors = random.sample(fg_choices, count)
    quadrants = random.sample(list(_QUADRANTS), count)
    modes = _choose_modes(count)
    pairs: list[PairSpec] = []
    input_union = np.zeros((h, w), dtype=bool)
    output_union = np.zeros((h, w), dtype=bool)
    clipped_total = 0
    for quadrant, color, mode in zip(quadrants, colors, modes):
        pair = _pick_pair((h, w), quadrant, color, mode)
        if pair is None:
            return None
        pair_input = _input_mask(pair, (h, w))
        pair_output, stats = _project_mask((h, w), pair)
        if np.any(pair_output & output_union):
            return None
        if not _well_separated(pair_input, input_union):
            return None
        clipped_total += int(stats["clipped"])
        input_union |= pair_input
        output_union |= pair_output
        pairs.append(pair)
    if clipped_total == 0:
        return None
    grid = np.full((h, w), bg, dtype=int)
    for pair in pairs:
        grid[pair.r0 : pair.r0 + pair.h, pair.c0 : pair.c0 + pair.w] = pair.color
        grid[pair.marker] = pair.color
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, pairs = _scene_from_grid(grid)
    except AssertionError:
        return False
    correct = _render_scene(grid.shape, bg, pairs, mode="correct")
    if not np.array_equal(main(grid), correct):
        return False
    if np.array_equal(correct, _render_scene(grid.shape, bg, pairs, mode="vector")):
        return False
    if np.array_equal(correct, _render_scene(grid.shape, bg, pairs, mode="diagonal")):
        return False
    seen_equal = False
    seen_row = False
    seen_col = False
    clipped_total = 0
    for pair in pairs:
        _, stats = _project_mask(grid.shape, pair)
        d = stats["d"]
        a = stats["a"]
        clipped_total += int(stats["clipped"])
        if a == (0, 0):
            seen_equal = True
        elif a[0] != 0:
            seen_row = True
        else:
            seen_col = True
        if len(stats["offsets"]) < 2:
            return False
        if a != (0, 0) and len(stats["offsets"]) < 3:
            return False
        v = (pair.marker[0] - _nearest_corner(pair.bbox, pair.marker)[0], pair.marker[1] - _nearest_corner(pair.bbox, pair.marker)[1])
        if v != (d[0] + a[0], d[1] + a[1]):
            return False
    if not (seen_row and seen_col):
        return False
    if clipped_total == 0:
        return False
    if seen_equal and len(pairs) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
