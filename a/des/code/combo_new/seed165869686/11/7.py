import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _tight_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, r0, c0


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for pts in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        holes += 1
    return holes


def _contour_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out.append((r, c))
                    break
    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _flip_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _add_protrusions(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    pad = 4
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask
    for _ in range(rng.randint(0, 2)):
        pts = np.argwhere(canvas)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        side = rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            width = rng.randint(1, min(3, c1 - c0 + 1))
            start = rng.randint(c0, c1 - width + 1)
            depth = rng.randint(1, 2)
            if side == "top" and r0 - depth >= 0:
                canvas[r0 - depth : r0, start : start + width] = True
            if side == "bottom" and r1 + depth < canvas.shape[0]:
                canvas[r1 + 1 : r1 + depth + 1, start : start + width] = True
        else:
            height = rng.randint(1, min(3, r1 - r0 + 1))
            start = rng.randint(r0, r1 - height + 1)
            depth = rng.randint(1, 2)
            if side == "left" and c0 - depth >= 0:
                canvas[start : start + height, c0 - depth : c0] = True
            if side == "right" and c1 + depth < canvas.shape[1]:
                canvas[start : start + height, c1 + 1 : c1 + depth + 1] = True
    return _crop(canvas)


def _solid_mask(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        h = rng.randint(3, 7)
        w = rng.randint(3, 7)
        target = rng.randint(max(5, (h * w) // 3), min(h * w, 18))
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        frontier = set()
        for dr, dc in DIR4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while int(mask.sum()) < target and frontier:
            cell = rng.choice(list(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for dr, dc in DIR4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        mask = _crop(mask)
        mask = _add_protrusions(mask, rng)
        mask = _flip_mask(mask, rng)
        if _hole_count(mask) == 0 and int(mask.sum()) >= 6 and min(mask.shape) >= 2:
            return mask
    raise RuntimeError("failed to build solid mask")


def _holey_mask(hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(80):
        horizontal = rng.random() < 0.5
        if horizontal:
            hole_h = rng.randint(1, 3)
            top = rng.randint(1, 3)
            bottom = rng.randint(1, 3)
            left = rng.randint(1, 3)
            right = rng.randint(1, 3)
            widths = [rng.randint(1, 2) for _ in range(hole_count)]
            gaps = [rng.randint(1, 2) for _ in range(hole_count - 1)]
            h = top + hole_h + bottom
            w = left + sum(widths) + sum(gaps) + right
            mask = np.ones((h, w), dtype=bool)
            cur = left
            for idx, width in enumerate(widths):
                mask[top : top + hole_h, cur : cur + width] = False
                cur += width
                if idx < len(gaps):
                    cur += gaps[idx]
        else:
            hole_w = rng.randint(1, 3)
            top = rng.randint(1, 3)
            bottom = rng.randint(1, 3)
            left = rng.randint(1, 3)
            right = rng.randint(1, 3)
            heights = [rng.randint(1, 2) for _ in range(hole_count)]
            gaps = [rng.randint(1, 2) for _ in range(hole_count - 1)]
            h = top + sum(heights) + sum(gaps) + bottom
            w = left + hole_w + right
            mask = np.ones((h, w), dtype=bool)
            cur = top
            for idx, height in enumerate(heights):
                mask[cur : cur + height, left : left + hole_w] = False
                cur += height
                if idx < len(gaps):
                    cur += gaps[idx]
        mask = _add_protrusions(mask, rng)
        mask = _flip_mask(mask, rng)
        if _hole_count(mask) == hole_count and max(mask.shape) <= 16:
            return mask
    raise RuntimeError("failed to build holey mask")


def _make_object_mask(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        return _solid_mask(rng)
    return _holey_mask(hole_count, rng)


def _place_legend(
    grid: np.ndarray, corner: str, cell: int, colors_bottom_to_top: list[int]
) -> np.ndarray:
    h, w = grid.shape
    total_h = cell * len(colors_bottom_to_top)
    if corner == "tl":
        r_base, c0 = 0, 0
    elif corner == "tr":
        r_base, c0 = 0, w - cell
    elif corner == "bl":
        r_base, c0 = h - total_h, 0
    else:
        r_base, c0 = h - total_h, w - cell

    for idx, color in enumerate(colors_bottom_to_top):
        rr0 = r_base + total_h - (idx + 1) * cell
        rr1 = rr0 + cell
        grid[rr0:rr1, c0 : c0 + cell] = color

    mask = np.zeros_like(grid, dtype=bool)
    mask[r_base : r_base + total_h, c0 : c0 + cell] = True
    return mask


def _detect_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for cell in range(1, min(3, h, w) + 1):
        for corner in ("tl", "tr", "bl", "br"):
            if corner in ("tl", "bl"):
                c0 = 0
            else:
                c0 = w - cell
            colors: list[int] = []
            steps = h // cell
            for idx in range(steps):
                if corner in ("tl", "tr"):
                    r0 = idx * cell
                else:
                    r0 = h - (idx + 1) * cell
                block = g[r0 : r0 + cell, c0 : c0 + cell]
                vals = np.unique(block)
                if vals.size == 1 and int(vals[0]) != BG:
                    colors.append(int(vals[0]))
                else:
                    break
            if len(colors) < 2 or len(set(colors)) != len(colors):
                continue
            if corner in ("tl", "tr"):
                colors_bottom_to_top = colors[::-1]
                r_base = 0
            else:
                colors_bottom_to_top = colors
                r_base = h - len(colors) * cell
            mask = np.zeros_like(g, dtype=bool)
            mask[r_base : r_base + len(colors) * cell, c0 : c0 + cell] = True
            candidates.append((len(colors), cell, colors_bottom_to_top, mask))

    if not candidates:
        raise AssertionError("legend not found")

    _, _, colors, mask = max(candidates, key=lambda item: (item[0], item[1]))
    return colors, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object's contour using the legend color indexed by its hole count.

    Input structure:
    - The background is black.
    - One corner contains a vertical legend made of small same-sized colored squares stacked along the grid edge.
    - Elsewhere are separated single-color connected objects; some are solid and some contain black holes fully enclosed by the object.

    Rule:
    - Read the legend from bottom to top so square 0 is the lowest one, square 1 is above it, and so on.
    - For each non-legend object, count how many enclosed black hole regions it contains.
    - Use the legend color at that hole-count index to recolor exactly the object's contour pixels, where a contour pixel is any object pixel with at least one 4-neighbor background cell.
    - Keep non-contour object pixels unchanged, keep holes black, and leave the legend untouched.

    Key insight:
    - The legend encodes a visible lookup table from a topological property of each object, namely its hole count, to the color that should be applied only on that object's boundary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_colors, legend_mask = _detect_legend(g)
    object_mask = (g != BG) & (~legend_mask)
    for pts in _components(object_mask):
        local_mask, r0, c0 = _tight_mask(pts)
        holes = _hole_count(local_mask)
        if not (0 <= holes < len(legend_colors)):
            continue
        contour_color = int(legend_colors[holes])
        for lr, lc in _contour_cells(local_mask):
            out[r0 + lr, c0 + lc] = contour_color
    return out


GEN_RETRIES = 400


def _blocked_positions(blocked: np.ndarray, mask: np.ndarray, rng: random.Random) -> list[tuple[int, int]]:
    h, w = blocked.shape
    mh, mw = mask.shape
    positions = [
        (r0, c0)
        for r0 in range(1, h - mh)
        for c0 in range(1, w - mw)
        if not blocked[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1].any()
    ]
    rng.shuffle(positions)
    return positions


def _render_object(
    grid: np.ndarray,
    output: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    object_color: int,
    contour_color: int,
) -> None:
    sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    sub[mask] = object_color
    out_sub = output[top : top + mask.shape[0], left : left + mask.shape[1]]
    out_sub[mask] = object_color
    for r, c in _contour_cells(mask):
        out_sub[r, c] = contour_color


def _make_hole_counts(count: int, max_holes: int, rng: random.Random) -> list[int]:
    values = [0, rng.randint(1, max_holes)]
    while len(values) < count:
        values.append(rng.randint(0, max_holes))
    rng.shuffle(values)
    return values


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(16, 30)
    w = rng.randint(16, 30)
    cell = rng.randint(1, 3)
    legend_len = rng.randint(2, 4)
    if cell * legend_len > h:
        return None

    colors = list(range(1, 10))
    rng.shuffle(colors)
    legend_colors = colors[:legend_len]
    object_color = colors[legend_len]
    corner = rng.choice(["tl", "tr", "bl", "br"])

    grid = np.zeros((h, w), dtype=int)
    output = np.zeros((h, w), dtype=int)
    legend_mask = _place_legend(grid, corner, cell, legend_colors)
    _place_legend(output, corner, cell, legend_colors)

    blocked = np.zeros((h, w), dtype=bool)
    legend_pts = np.argwhere(legend_mask)
    lr0, lc0 = legend_pts.min(axis=0)
    lr1, lc1 = legend_pts.max(axis=0)
    blocked[max(0, lr0 - 1) : min(h, lr1 + 2), max(0, lc0 - 1) : min(w, lc1 + 2)] = True

    object_count = rng.randint(2, min(5, legend_len + 2))
    hole_counts = _make_hole_counts(object_count, legend_len - 1, rng)
    objects = []
    interiorful = False
    for hc in hole_counts:
        mask = _make_object_mask(hc, rng)
        objects.append((int(mask.sum()), hc, mask))
        if int(mask.sum()) > len(_contour_cells(mask)):
            interiorful = True
    if not interiorful:
        return None

    for _area, hc, mask in sorted(objects, reverse=True):
        positions = _blocked_positions(blocked, mask, rng)
        if not positions:
            return None
        top, left = positions[0]
        _render_object(grid, output, mask, top, left, object_color, legend_colors[hc])
        blocked[top - 1 : top + mask.shape[0] + 1, left - 1 : left + mask.shape[1] + 1] = True

    if int(np.bincount(grid.ravel(), minlength=10).argmax()) != BG:
        return None
    if np.array_equal(grid, output):
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
