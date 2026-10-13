import random
from typing import List, Tuple

import numpy as np


BG_COLOR = 0
RAY_COLOR = 3


def main(grid: np.ndarray) -> np.ndarray:
    """Infer a hidden symmetry axis and project one-side defects as rays.

    Input structure:
    A black background grid with one main foreground color forming a shape nearly symmetric across one horizontal or vertical axis. The symmetry axis is not drawn. A small red arrow strip lies on one border edge (pointing inward) and indicates ray direction.

    Rule:
    1. Find background, foreground, and marker color.
    2. Choose the best horizontal/vertical mirror axis by maximizing foreground symmetry.
    3. Collect difference pixels: foreground pixels whose mirror on the chosen axis is background.
    4. Determine ray direction from marker side (right edge -> left, left edge -> right, top edge -> down, bottom edge -> up).
    5. Remove marker pixels (replace with background). For each difference pixel, draw green pixels from the next cell in direction until the next step is out of bounds or is foreground.

    Key insight:
    The task is to recover the implicit axis through symmetry analysis first, then use edge-control direction to cast deterministic rays from asymmetric pixels.
    """

    if grid.ndim != 2:
        raise ValueError("grid must be a 2D ndarray")

    out = np.array(grid, copy=True)
    h, w = out.shape
    if h == 0 or w == 0:
        return out

    bg_color = _most_common_color(out)

    colors = np.setdiff1d(np.unique(out), np.array([bg_color], dtype=out.dtype))
    if colors.size == 0:
        return out

    counts = {int(c): int(np.sum(out == c)) for c in colors}
    fg_color = max(counts, key=counts.get)

    arrow_candidates = [int(c) for c in colors if int(c) != fg_color]
    if arrow_candidates:
        arrow_color = max(arrow_candidates, key=lambda c: _count_border_pixels(out, c))
    else:
        arrow_color = fg_color

    axis_orient, axis_a2 = _find_symmetry_axis(out, fg_color)
    dr, dc = _direction_from_markers(out, bg_color, arrow_color)

    # Remove markers.
    if np.any(out == arrow_color):
        out[out == arrow_color] = bg_color

    # Draw green rays.
    for r, c in _find_differences(out, fg_color, axis_orient, axis_a2):
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w:
            if out[rr, cc] == fg_color:
                break
            out[rr, cc] = RAY_COLOR
            rr += dr
            cc += dc

    return out


def generate_input() -> np.ndarray:
    """Generate a random ARC-like instance with hidden axis, asymmetric defects, and border arrows."""

    h = random.randint(12, 28)
    w = random.randint(12, 28)
    grid = np.full((h, w), BG_COLOR, dtype=np.int64)

    palette = [1, 2, 4, 5, 6, 7, 8, 9]
    fg_color = random.choice(palette)
    arrow_color = random.choice([c for c in palette if c != fg_color])

    axis_orient = random.choice(["v", "h"])
    direction_side = random.choice(["left", "right", "top", "bottom"])

    # Choose a stable interior axis.
    if axis_orient == "v":
        anchor = random.randint(4, w - 5)
        axis_a2 = 2 * anchor + (random.randint(0, 1))
    else:
        anchor = random.randint(4, h - 5)
        axis_a2 = 2 * anchor + (random.randint(0, 1))

    _draw_symmetric_pattern(
        grid,
        fg_color,
        axis_orient,
        axis_a2,
        random.choice(["negative", "positive"]),
    )

    # Guarantee pattern exists; retry once if needed.
    if np.sum(grid == fg_color) == 0:
        _draw_symmetric_pattern(
            grid,
            fg_color,
            axis_orient,
            axis_a2,
            "negative",
        )

    _inject_defects(grid, fg_color, axis_orient, axis_a2)

    if direction_side == "left":
        _place_left_arrows(grid, arrow_color)
    elif direction_side == "right":
        _place_right_arrows(grid, arrow_color)
    elif direction_side == "top":
        _place_top_arrows(grid, arrow_color)
    else:
        _place_bottom_arrows(grid, arrow_color)

    return grid


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnt = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnt)])


def _count_border_pixels(grid: np.ndarray, color: int) -> int:
    return int(np.sum(grid[0, :] == color) + np.sum(grid[-1, :] == color) + np.sum(grid[:, 0] == color) + np.sum(grid[:, -1] == color))


def _direction_from_markers(grid: np.ndarray, bg_color: int, arrow_color: int) -> Tuple[int, int]:
    h, w = grid.shape
    if not np.any(grid == arrow_color):
        return 0, 1

    side_counts = {
        "right": int(np.sum(grid[:, w - 1] == arrow_color)),
        "left": int(np.sum(grid[:, 0] == arrow_color)),
        "top": int(np.sum(grid[0, :] == arrow_color)),
        "bottom": int(np.sum(grid[h - 1, :] == arrow_color)),
    }

    # Order fixes deterministic tie handling.
    ordered = ["right", "left", "top", "bottom"]
    chosen = ordered[0]
    for s in ordered[1:]:
        if side_counts[s] > side_counts[chosen]:
            chosen = s

    if side_counts[chosen] == 0:
        return 0, 1

    if chosen == "right":
        return 0, -1
    if chosen == "left":
        return 0, 1
    if chosen == "top":
        return 1, 0
    return -1, 0


def _find_symmetry_axis(grid: np.ndarray, fg_color: int) -> Tuple[str, int]:
    h, w = grid.shape
    pts = np.argwhere(grid == fg_color)
    if len(pts) == 0:
        return "v", w - 1

    best = (10**9, 10**9, 10**9, 10**9)
    best_axis = ("v", w - 1)

    def eval_axis(orient: str, a2: int) -> Tuple[int, int, int, int]:
        bad = 0
        good = 0
        valid = 0
        for r, c in pts:
            if orient == "v":
                mr, mc = r, a2 - c
            else:
                mr, mc = a2 - r, c
            if 0 <= mr < h and 0 <= mc < w:
                valid += 1
                if grid[mr, mc] == fg_color:
                    good += 1
                else:
                    bad += 1
        if valid == 0:
            return (10**9, -1, -1, -1)
        mid = w - 1 if orient == "v" else h - 1
        # Lower bad is better, then fewer asymmetries, then more valid pairs, then center alignment.
        return (bad, -good, -valid, abs(a2 - mid))

    for orient, span in (("v", w), ("h", h)):
        for a2 in range(1, 2 * span - 2):
            cur = eval_axis(orient, a2)
            if cur < best:
                best = cur
                best_axis = (orient, a2)

    return best_axis


def _find_differences(
    grid: np.ndarray,
    fg_color: int,
    orient: str,
    a2: int,
) -> List[Tuple[int, int]]:
    h, w = grid.shape
    diffs: List[Tuple[int, int]] = []
    for r, c in np.argwhere(grid == fg_color):
        if orient == "v":
            mr, mc = r, a2 - c
        else:
            mr, mc = a2 - r, c
        if not (0 <= mr < h and 0 <= mc < w):
            continue
        if mr == r and mc == c:
            continue
        if grid[mr, mc] != fg_color:
            diffs.append((int(r), int(c)))
    return diffs


def _mirror_pos(orient: str, a2: int, r: int, c: int) -> Tuple[int, int]:
    if orient == "v":
        return r, a2 - c
    return a2 - r, c


def _draw_symmetric_pattern(
    grid: np.ndarray,
    fg_color: int,
    orient: str,
    a2: int,
    base_side: str,
) -> None:
    h, w = grid.shape
    n_shapes = random.randint(3, 7)

    for _ in range(n_shapes):
        rh = random.randint(1, 4)
        rw = random.randint(1, 4)
        if orient == "v":
            if a2 % 2 == 0:
                axis = a2 // 2
                if base_side == "negative":
                    lo, hi = 2, axis - 2
                else:
                    lo, hi = axis + 2, w - 3
            else:
                split = a2 // 2
                if base_side == "negative":
                    lo, hi = 2, split
                else:
                    lo, hi = split + 1, w - 3

            if lo > hi or lo >= w - 1 or hi < 2:
                continue
            rw = min(rw, max(1, hi - lo))
            r0 = random.randint(2, max(2, h - 3 - rh + 1))
            c0 = random.randint(lo, hi - rw + 1)
            for r in range(r0, r0 + rh):
                for c in range(c0, c0 + rw):
                    if random.random() < 0.75:
                        grid[r, c] = fg_color
                        mr, mc = _mirror_pos(orient, a2, r, c)
                        if 0 <= mr < h and 0 <= mc < w:
                            grid[mr, mc] = fg_color
        else:
            if a2 % 2 == 0:
                axis = a2 // 2
                if base_side == "negative":
                    lo, hi = 2, axis - 2
                else:
                    lo, hi = axis + 2, h - 3
            else:
                split = a2 // 2
                if base_side == "negative":
                    lo, hi = 2, split
                else:
                    lo, hi = split + 1, h - 3

            if lo > hi or lo >= h - 1 or hi < 2:
                continue

            rh = min(rh, max(1, hi - lo))
            c0 = random.randint(2, max(2, w - 3 - rw + 1))
            r0 = random.randint(lo, hi - rh + 1)
            for r in range(r0, r0 + rh):
                for c in range(c0, c0 + rw):
                    if random.random() < 0.75:
                        grid[r, c] = fg_color
                        mr, mc = _mirror_pos(orient, a2, r, c)
                        if 0 <= mr < h and 0 <= mc < w:
                            grid[mr, mc] = fg_color

    # Extra density to avoid degenerate tiny objects.
    if np.sum(grid == fg_color) < 24:
        for r in range(2, h - 2):
            for c in range(2, w - 2):
                if random.random() < 0.15:
                    grid[r, c] = fg_color
                    mr, mc = _mirror_pos(orient, a2, r, c)
                    if 0 <= mr < h and 0 <= mc < w:
                        grid[mr, mc] = fg_color


def _inject_defects(grid: np.ndarray, fg_color: int, orient: str, a2: int) -> None:
    h, w = grid.shape
    fg_cells = set(map(tuple, np.argwhere(grid == fg_color)))
    pairs = []

    for r, c in fg_cells:
        mr, mc = _mirror_pos(orient, a2, r, c)
        if not (0 <= mr < h and 0 <= mc < w):
            continue
        if (mr, mc) not in fg_cells:
            continue
        if orient == "v" and c < mc:
            pairs.append((r, c, mr, mc))
        if orient == "h" and r < mr:
            pairs.append((r, c, mr, mc))

    if not pairs:
        return

    random.shuffle(pairs)
    remove_count = max(1, random.randint(1, max(1, len(pairs) // 2)))
    remove_count = min(remove_count, len(pairs))

    for i in range(remove_count):
        r1, c1, r2, c2 = pairs[i]
        if random.random() < 0.5:
            grid[r1, c1] = BG_COLOR
        else:
            grid[r2, c2] = BG_COLOR


def _equal_spaced_indices(count: int, lo: int, hi: int) -> List[int]:
    if count <= 0 or lo > hi:
        return []
    available = hi - lo + 1
    count = min(count, available)
    if count <= 1:
        return [random.randint(lo, hi)]
    max_step = max(1, available // (count - 1))
    step = random.randint(1, max_step)
    max_start = hi - step * (count - 1)
    if max_start < lo:
        max_start = lo
    start = random.randint(lo, max_start)
    return [start + i * step for i in range(count)]


def _place_left_arrows(grid: np.ndarray, color: int) -> None:
    h, w = grid.shape
    if h < 5 or w < 4:
        return
    cnt = random.randint(1, max(1, (h - 6) // 2))
    rows = _equal_spaced_indices(cnt, 2, h - 3)
    for r in rows:
        if 1 <= r <= h - 2:
            grid[r, 1] = color
            grid[r - 1, 0] = color
            grid[r + 1, 0] = color


def _place_right_arrows(grid: np.ndarray, color: int) -> None:
    h, w = grid.shape
    if h < 5 or w < 4:
        return
    cnt = random.randint(1, max(1, (h - 6) // 2))
    rows = _equal_spaced_indices(cnt, 2, h - 3)
    for r in rows:
        if 1 <= r <= h - 2:
            grid[r, w - 2] = color
            grid[r - 1, w - 1] = color
            grid[r + 1, w - 1] = color


def _place_top_arrows(grid: np.ndarray, color: int) -> None:
    h, w = grid.shape
    if w < 5 or h < 4:
        return
    cnt = random.randint(1, max(1, (w - 6) // 2))
    cols = _equal_spaced_indices(cnt, 2, w - 3)
    for c in cols:
        if 1 <= c <= w - 2:
            grid[1, c] = color
            grid[0, c - 1] = color
            grid[0, c + 1] = color


def _place_bottom_arrows(grid: np.ndarray, color: int) -> None:
    h, w = grid.shape
    if w < 5 or h < 4:
        return
    cnt = random.randint(1, max(1, (w - 6) // 2))
    cols = _equal_spaced_indices(cnt, 2, w - 3)
    for c in cols:
        if 1 <= c <= w - 2:
            grid[h - 2, c] = color
            grid[h - 1, c - 1] = color
            grid[h - 1, c + 1] = color


__all__ = ["main", "generate_input"]
