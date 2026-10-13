from __future__ import annotations

import math
from collections import Counter, deque

import numpy as np


BLACK = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_SIDE = 30
GEN_RETRIES = 400
OBJECT_RETRIES = 200
PLACEMENT_RETRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components(mask)) == 1


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_border = True
                    queue.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _outer_contour(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask, 1, constant_values=False)
    ext = np.zeros_like(padded, dtype=bool)
    queue = deque([(0, 0)])
    ext[0, 0] = True
    while queue:
        rr, cc = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1]:
                if not padded[nr, nc] and not ext[nr, nc]:
                    ext[nr, nc] = True
                    queue.append((nr, nc))

    contour: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        pr, pc = int(r + 1), int(c + 1)
        if any(ext[pr + dr, pc + dc] for dr, dc in DIR4):
            contour.append((int(r), int(c)))
    return contour


def _legend_colors(grid: np.ndarray) -> list[int]:
    legend = [int(grid[r, 0]) for r in range(grid.shape[0]) if int(grid[r, 0]) != BLACK]
    assert legend, "expected a non-empty legend in the leftmost column"
    return legend


def _object_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = (grid != BLACK).copy()
    mask[:, 0] = False
    comps = _components(mask)
    assert comps, "expected at least one foreground object to the right of the legend"
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each object's outer contour using the legend color indexed by its hole count.

    Input structure:
    - The background is black.
    - The leftmost column contains a vertical legend of single colored cells separated by black rows; its top-to-bottom order encodes hole counts 0, 1, 2, ...
    - The remaining area contains several large 4-connected monochrome objects, and some of them enclose black holes.

    Rule:
    - Read the legend colors from the leftmost column in top-to-bottom order.
    - For each non-legend foreground object, count its enclosed black holes as 4-connected black regions inside the object's bounding box that do not touch that box border.
    - Use that hole count as an index into the legend, then recolor only the object's outer contour pixels to the selected legend color.
    - Keep the legend itself and every non-contour object pixel unchanged.

    Key insight:
    - The task combines a topological lookup and a geometric boundary test: the number of holes chooses the new color, but only pixels adjacent to the exterior black background are repainted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend = _legend_colors(g)
    out = g.copy()

    for comp in _object_components(g):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each object must be monochrome"
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True

        holes = _hole_count(mask)
        assert 0 <= holes < len(legend), "hole count must be representable by the legend"
        repaint = legend[holes]

        for rr, cc in _outer_contour(mask):
            out[r0 + rr, c0 + cc] = repaint

    return out


def _compact_hole_layout(hole_count: int, rng: np.random.Generator) -> tuple[int, int]:
    options: list[tuple[int, int, int, int]] = []
    for rows in range(1, hole_count + 1):
        cols = math.ceil(hole_count / rows)
        spare = rows * cols - hole_count
        options.append((abs(rows - cols), spare, rows * cols, rows))
    options.sort()
    best = options[: min(3, len(options))]
    _, _, _, rows = best[int(rng.integers(0, len(best)))]
    cols = math.ceil(hole_count / rows)
    return int(rows), int(cols)


def _add_attachment(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    canvas = np.pad(mask, 3, constant_values=False)
    coords = np.argwhere(canvas)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    choices = []
    if r0 > 0:
        choices.append("top")
    if r1 < canvas.shape[0] - 1:
        choices.append("bottom")
    if c0 > 0:
        choices.append("left")
    if c1 < canvas.shape[1] - 1:
        choices.append("right")
    if not choices:
        return mask

    side = choices[int(rng.integers(0, len(choices)))]
    if side in ("top", "bottom"):
        span = int(c1 - c0 + 1)
        length = int(rng.integers(1, max(2, min(span, span // 2 + 1))))
        start = int(rng.integers(int(c0), int(c1 - length + 2)))
        max_depth = int(r0) if side == "top" else int(canvas.shape[0] - 1 - r1)
        depth = int(rng.integers(1, min(3, max_depth) + 1))
        if side == "top":
            canvas[int(r0 - depth) : int(r0), start : start + length] = True
        else:
            canvas[int(r1 + 1) : int(r1 + depth + 1), start : start + length] = True
    else:
        span = int(r1 - r0 + 1)
        length = int(rng.integers(1, max(2, min(span, span // 2 + 1))))
        start = int(rng.integers(int(r0), int(r1 - length + 2)))
        max_depth = int(c0) if side == "left" else int(canvas.shape[1] - 1 - c1)
        depth = int(rng.integers(1, min(3, max_depth) + 1))
        if side == "left":
            canvas[start : start + length, int(c0 - depth) : int(c0)] = True
        else:
            canvas[start : start + length, int(c1 + 1) : int(c1 + depth + 1)] = True

    return _crop_bool(canvas)


def _try_add_notch(mask: np.ndarray, target_holes: int, rng: np.random.Generator) -> np.ndarray:
    candidate = mask.copy()
    h, w = candidate.shape
    side = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
    if side in ("top", "bottom"):
        if w < 3 or h < 2:
            return mask
        depth = int(rng.integers(1, min(2, h - 1) + 1))
        length = int(rng.integers(1, max(2, w // 3 + 1)))
        start = int(rng.integers(0, w - length + 1))
        if side == "top":
            candidate[:depth, start : start + length] = False
        else:
            candidate[h - depth :, start : start + length] = False
    else:
        if h < 3 or w < 2:
            return mask
        depth = int(rng.integers(1, min(2, w - 1) + 1))
        length = int(rng.integers(1, max(2, h // 3 + 1)))
        start = int(rng.integers(0, h - length + 1))
        if side == "left":
            candidate[start : start + length, :depth] = False
        else:
            candidate[start : start + length, w - depth :] = False

    if candidate.any() and _is_connected(candidate) and _hole_count(candidate) == target_holes:
        cropped = _crop_bool(candidate)
        if len(_outer_contour(cropped)) < int(cropped.sum()):
            return cropped
    return mask


def _make_object_mask(hole_count: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        if hole_count == 0:
            h = int(rng.integers(4, 8))
            w = int(rng.integers(4, 8))
            mask = np.ones((h, w), dtype=bool)
        else:
            rows, cols = _compact_hole_layout(hole_count, rng)
            hole_h = int(rng.integers(1, 3))
            hole_w = int(rng.integers(1, 3))
            gap_r = int(rng.integers(1, 3))
            gap_c = int(rng.integers(1, 3))
            top = int(rng.integers(1, 3))
            bottom = int(rng.integers(1, 3))
            left = int(rng.integers(1, 3))
            right = int(rng.integers(1, 3))

            h = top + bottom + rows * hole_h + (rows - 1) * gap_r
            w = left + right + cols * hole_w + (cols - 1) * gap_c
            mask = np.ones((h, w), dtype=bool)

            slots = [(sr, sc) for sr in range(rows) for sc in range(cols)]
            rng.shuffle(slots)
            for sr, sc in slots[:hole_count]:
                rr = top + sr * (hole_h + gap_r)
                cc = left + sc * (hole_w + gap_c)
                mask[rr : rr + hole_h, cc : cc + hole_w] = False

        for _ in range(int(rng.integers(0, 4))):
            mask = _add_attachment(mask, rng)
        for _ in range(int(rng.integers(0, 3))):
            mask = _try_add_notch(mask, hole_count, rng)

        k = int(rng.integers(0, 4))
        mask = np.rot90(mask, k)
        if bool(rng.integers(0, 2)):
            mask = np.fliplr(mask)
        mask = _crop_bool(mask)

        if not _is_connected(mask):
            continue
        if _hole_count(mask) != hole_count:
            continue
        if max(mask.shape) > 12:
            continue
        if int(mask.sum()) < 10:
            continue
        if len(_outer_contour(mask)) >= int(mask.sum()):
            continue
        return mask
    return None


def _can_place(
    occupancy: np.ndarray, mask: np.ndarray, top: int, left: int
) -> bool:
    h, w = occupancy.shape
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupancy[r, c]:
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupancy[r0:r1, c0:c1].any():
            return False
    return True


def _place_mask(
    grid: np.ndarray, occupancy: np.ndarray, mask: np.ndarray, color: int, top: int, left: int
) -> None:
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        grid[r, c] = int(color)
        occupancy[r, c] = True


def _validate_generated(grid: np.ndarray, required_counts: Counter[int]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > MAX_SIDE:
        return False
    legend = _legend_colors(grid)
    objects = _object_components(grid)
    found = Counter()
    for comp in objects:
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            return False
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        holes = _hole_count(mask)
        if holes >= len(legend):
            return False
        if len(_outer_contour(mask)) >= int(mask.sum()):
            return False
        found[holes] += 1
    if found != required_counts:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[:, 0], grid[:, 0]):
        return False
    return np.any(out != grid)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    legend_len = int(rng.integers(3, 6))
    legend_colors = [int(x) for x in rng.choice(np.arange(1, 10), size=legend_len, replace=False)]

    counts = list(range(legend_len))
    extra_cap = max(0, 6 - legend_len)
    extra = int(rng.integers(0, extra_cap + 1))
    counts.extend(int(rng.integers(0, legend_len)) for _ in range(extra))
    rng.shuffle(counts)

    masks: list[tuple[int, np.ndarray]] = []
    for hole_count in counts:
        mask = _make_object_mask(hole_count, rng)
        if mask is None:
            return None
        masks.append((hole_count, mask))
    masks.sort(key=lambda item: int(item[1].sum()), reverse=True)

    legend_span = 2 * legend_len - 1
    max_h = max(int(mask.shape[0]) for _, mask in masks)
    max_w = max(int(mask.shape[1]) for _, mask in masks)
    total_pack_area = sum((int(mask.shape[0]) + 2) * (int(mask.shape[1]) + 2) for _, mask in masks)

    min_h = max(legend_span + 1, max_h + 2, 12)
    min_w = max(max_w + 3, 12)

    grid_h = None
    grid_w = None
    for _ in range(50):
        h = int(rng.integers(min_h, MAX_SIDE + 1))
        w = int(rng.integers(min_w, MAX_SIDE + 1))
        if (h - 2) * (w - 3) >= total_pack_area * 2:
            grid_h, grid_w = h, w
            break
    if grid_h is None or grid_w is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    legend_top = int(rng.integers(0, grid_h - legend_span + 1))
    for idx, color in enumerate(legend_colors):
        grid[legend_top + 2 * idx, 0] = int(color)

    occupancy = np.zeros((grid_h, grid_w), dtype=bool)
    occupancy[:, 0] = True

    remaining = [color for color in range(1, 10) if color not in legend_colors]
    palette_size = int(rng.integers(2, min(4, len(remaining)) + 1))
    object_palette = [int(x) for x in rng.choice(np.array(remaining), size=palette_size, replace=False)]

    for hole_count, mask in masks:
        obj_h, obj_w = map(int, mask.shape)
        if grid_h - obj_h <= 1 or grid_w - obj_w <= 2:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(rng.integers(1, grid_h - obj_h))
            left = int(rng.integers(2, grid_w - obj_w))
            if not _can_place(occupancy, mask, top, left):
                continue
            color = object_palette[int(rng.integers(0, len(object_palette)))]
            _place_mask(grid, occupancy, mask, color, top, left)
            placed = True
            break
        if not placed:
            return None

    required_counts = Counter(counts)
    if not _validate_generated(grid, required_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
