from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import Color, find_components, paste
except ModuleNotFoundError:
    if "/home/lixinyu/snap/new/solvers" not in sys.path:
        sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color, find_components, paste


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_INDICATOR_COLORS = (Color.RED, Color.GREEN)
_BG = Color.BLACK
_PATTERN_PALETTE = [
    Color.BLUE,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
_NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _rand_connected_region(
    h: int,
    w: int,
    anchor: tuple[int, int],
    axis_c: int,
    side: str,
    target_size: int,
    *,
    max_steps: int = 500,
) -> set[tuple[int, int]] | None:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    ar, ac = anchor
    if not (0 <= ar < h and 0 <= ac < w):
        return None
    if side == "left" and ac > axis_c:
        return None
    if side == "right" and ac < axis_c:
        return None

    def allowed(r: int, c: int) -> bool:
        return c <= axis_c if side == "left" else c >= axis_c

    cells: set[tuple[int, int]] = {anchor}
    frontier: list[tuple[int, int]] = [anchor]

    attempts = 0
    while len(cells) < target_size and attempts < max_steps:
        attempts += 1
        r, c = random.choice(frontier)
        dr, dc = random.choice(_NEIGHBORS_4)
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and allowed(nr, nc):
            nxt = (nr, nc)
            if nxt not in cells:
                cells.add(nxt)
                frontier.append(nxt)

    return cells if len(cells) == target_size else None


def _axis_has_missing_partner(patch: np.ndarray, axis_c: int) -> bool:
    h, w = map(int, patch.shape)
    for r in range(h):
        for c in range(w):
            col = int(patch[r, c])
            if col == _BG:
                continue
            if c == axis_c:
                continue
            mirrored_c = 2 * int(axis_c) - c
            if not (0 <= mirrored_c < w):
                # A pixel near border with a virtual mirror outside canvas is a clear asymmetry signal.
                return True
            if int(patch[r, mirrored_c]) != col:
                return True
    return False


def _generate_partial_pattern(indicator_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(120):
        h = random.randint(3, 6)
        w = random.randint(3, 8)
        axis_c = random.randint(0, w - 1)
        side = random.choice(["left", "right"])

        marker_r = random.randint(0, h - 1)
        marker = (marker_r, axis_c)

        allowed_area = 0
        for rr in range(h):
            for cc in range(w):
                if side == "left" and cc <= axis_c:
                    allowed_area += 1
                if side == "right" and cc >= axis_c:
                    allowed_area += 1
        if allowed_area < 4:
            continue

        target_size = random.randint(4, min(allowed_area, 12))
        region = _rand_connected_region(h, w, marker, axis_c, side, target_size)
        if region is None:
            continue

        patch = np.full((h, w), _BG, dtype=int)
        palette = [c for c in _PATTERN_PALETTE if c != indicator_color]
        fill_colors = random.sample(palette, k=random.randint(1, min(3, len(palette))))

        patch[marker] = indicator_color
        for r, c in region:
            if (r, c) == marker:
                continue
            patch[r, c] = random.choice(fill_colors)

        if _axis_has_missing_partner(patch, axis_c):
            return patch, marker

    # Fallback deterministic asymmetric one-side shape.
    h, w = 5, 6
    axis_c = random.randint(1, w - 2)
    marker = (2, axis_c)
    fill = random.choice([c for c in _PATTERN_PALETTE if c != indicator_color])
    patch = np.full((h, w), _BG, dtype=int)
    patch[marker] = indicator_color
    for rr in range(h):
        for cc in range(axis_c):
            if random.random() < 0.75:
                patch[rr, cc] = fill
    return patch, marker


def _bbox_for_coords(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _free_position(
    occupied: np.ndarray,
    shape_mask: np.ndarray,
    *,
    padding: int = 1,
    tries: int = 500,
) -> tuple[int, int] | None:
    H, W = occupied.shape
    ph, pw = map(int, shape_mask.shape)
    if ph > H or pw > W:
        return None

    for _ in range(tries):
        r0 = random.randint(0, H - ph)
        c0 = random.randint(0, W - pw)

        rr0 = max(0, r0 - padding)
        cc0 = max(0, c0 - padding)
        rr1 = min(H, r0 + ph + padding)
        cc1 = min(W, c0 + pw + padding)

        if np.any(occupied[rr0:rr1, cc0:cc1]):
            continue

        return r0, c0

    return None


def _place_isolated_marker(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    *,
    tries: int = 500,
) -> tuple[int, int] | None:
    H, W = grid.shape
    for _ in range(tries):
        r = random.randint(0, H - 1)
        c = random.randint(0, W - 1)
        if occupied[r, c]:
            continue

        rr0 = max(0, r - 1)
        cc0 = max(0, c - 1)
        rr1 = min(H, r + 2)
        cc1 = min(W, c + 2)
        if np.any(occupied[rr0:rr1, cc0:cc1]):
            continue

        grid[r, c] = int(color)
        occupied[r, c] = True
        return r, c

    return None


def _build_symmetric_stamp(
    grid: np.ndarray,
    component_coords: np.ndarray,
    marker_row: int,
    marker_col: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    H, W = grid.shape
    stamp_cells: dict[tuple[int, int], int] = {}

    for r, c in map(tuple, component_coords):
        r_i = int(r)
        c_i = int(c)
        col = int(grid[r_i, c_i])
        stamp_cells[(r_i, c_i)] = col

        mirrored_c = 2 * int(marker_col) - c_i
        if 0 <= mirrored_c < W:
            existing = stamp_cells.get((r_i, mirrored_c))
            if existing is not None and existing != col:
                # Degenerate/conflicted inputs are tolerated; keep original visible pixel.
                continue
            stamp_cells[(r_i, mirrored_c)] = col

    r0, c0, r1, c1 = _bbox_for_coords(set(stamp_cells.keys()))
    stamp = np.full((r1 - r0 + 1, c1 - c0 + 1), _BG, dtype=int)
    for (r, c), value in stamp_cells.items():
        stamp[r - r0, c - c0] = int(value)

    return stamp, (marker_row - r0, marker_col - c0)


# ---------------------------------------------------------------------------
# Main transform
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """以指示像素所在列对每个图案做左右镜像补全，再在同色孤立指示像素处盖印。

Input structure:
- 黑色为背景。
- 输入上有若干多色图案，每个图案内部恰有一个红色或绿色指示像素。
- 另有若干红色/绿色孤立像素，作为盖印目标（与图案内部像素不连通）。

Rule:
- 识别每个包含且仅包含一个指示色像素的多色图案，对该图案按指示像素所在列作左右镜像补全（对齐该列）。
- 用该对称全形替换该图案原位。
- 将该对称全形按其指示像素与每个同色孤立指示像素对齐后逐一盖印，背景其余位置不变。

Key insight:
- 指示像素同时决定了图案的补全对称轴、与目标配对的颜色以及盖印时的锚点位置。
"""

    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects a 2D grid")

    out = g.copy()
    H, W = g.shape

    components = find_components(g, bg=_BG, connectivity=8, monochromatic=False, return_format="dict")
    patterns: list[tuple[np.ndarray, tuple[int, int], int, tuple[int, int]]] = []
    targets: dict[int, list[tuple[int, int]]] = {Color.RED: [], Color.GREEN: []}

    for comp in components:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        values = g[coords[:, 0], coords[:, 1]]
        is_indicator = np.isin(values, _INDICATOR_COLORS)
        cnt_indicator = int(np.sum(is_indicator))

        # Size-1 components are isolated markers.
        if coords.shape[0] == 1 and cnt_indicator == 1:
            mr, mc = map(int, coords[0])
            col = int(g[mr, mc])
            targets[col].append((mr, mc))
            continue

        # Multi-cell patterns must include exactly one indicator pixel.
        if cnt_indicator != 1:
            continue

        marker_index = int(np.flatnonzero(is_indicator)[0])
        marker_row, marker_col = map(int, coords[marker_index])
        marker_color = int(values[marker_index])

        stamp, anchor = _build_symmetric_stamp(g, coords, marker_row, marker_col)
        patterns.append((stamp, anchor, marker_color, (marker_row, marker_col)))

    for stamp, anchor, marker_color, marker_global in patterns:
        ph, pw = map(int, stamp.shape)
        anchor_r, anchor_c = map(int, anchor)

        # Replace original pattern with its completed counterpart.
        top = marker_global[0] - anchor_r
        left = marker_global[1] - anchor_c
        if 0 <= top and 0 <= left and top + ph <= H and left + pw <= W:
            paste(out, stamp, top, left, transparent=_BG)

        # Stamp to each same-color isolated target.
        for tr, tc in targets.get(marker_color, []):
            top = int(tr) - anchor_r
            left = int(tc) - anchor_c
            if 0 <= top and 0 <= left and top + ph <= H and left + pw <= W:
                paste(out, stamp, top, left, transparent=_BG)

    return out


# ---------------------------------------------------------------------------
# Random generator
# ---------------------------------------------------------------------------


def _build_inputs_once() -> np.ndarray | None:
    H = random.randint(16, 30)
    W = random.randint(16, 30)
    grid = np.full((H, W), _BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    pattern_count = random.randint(1, 4)
    if random.random() < 0.65 and pattern_count >= 2:
        # Branch: force both marker colors to appear.
        marker_colors = [Color.RED, Color.GREEN] + [
            random.choice(_INDICATOR_COLORS) for _ in range(pattern_count - 2)
        ]
        random.shuffle(marker_colors)
    else:
        marker_colors = [random.choice(_INDICATOR_COLORS) for _ in range(pattern_count)]

    for marker_color in marker_colors:
        for _ in range(100):
            patch, marker = _generate_partial_pattern(indicator_color=marker_color)
            mask = patch != _BG

            pos = _free_position(occupied, mask, padding=1, tries=250)
            if pos is None:
                continue

            r0, c0 = pos
            paste(grid, patch, r0, c0, transparent=_BG)
            occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= (patch != _BG)
            break
        else:
            return None

    # One or more isolated target markers per pattern, with occasional duplicates.
    for marker_color in marker_colors:
        target_count = random.choice((1, 1, 2, 3))
        for _ in range(target_count):
            if _place_isolated_marker(grid, occupied, marker_color, tries=500) is None:
                return None

    # Optional extra random targets.
    if random.random() < 0.55:
        for marker_color in _INDICATOR_COLORS:
            if random.random() < 0.45:
                if _place_isolated_marker(grid, occupied, marker_color, tries=300) is None:
                    return None

    # Keep only valid cases in which main can parse the generated sample.
    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _build_inputs_once()
        if grid is not None:
            return grid

    raise RuntimeError("failed to generate valid puzzle input")
