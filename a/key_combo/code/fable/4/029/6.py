from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BAR = 5
RECT_COLORS = tuple(color for color in range(10) if color not in (BG, BAR))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    pieces = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([piece.reshape(-1) for piece in pieces if piece.size])
    colors = np.unique(boundary)
    assert colors.size == 1, "each exposed boundary must be monochrome"
    return int(colors[0])


def _peel_inner_bounds(rect: np.ndarray, depth: int) -> tuple[int, int, int, int]:
    top, left = 0, 0
    bottom, right = map(int, rect.shape)
    for _ in range(int(depth)):
        if top >= bottom or left >= right:
            break
        color = _boundary_color(rect, top, bottom, left, right)
        previous = (top, bottom, left, right)

        while top < bottom and np.all(rect[top, left:right] == color):
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            right -= 1

        assert (top, bottom, left, right) != previous, "each peel must remove one full color layer"
    return top, bottom, left, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each gray bar and grind away that many outer color layers from its attached nested rectangle.

    Input structure:
    - The background is the most common color.
    - One or more solid foreground rectangles are filled by inward nested rectangular color layers.
    - Some rectangles have a one-cell-thick gray bar touching exactly one outer side by 4-neighbor contact.

    Rule:
    - Count the pixels in every gray bar and erase the bar to background.
    - For the unique rectangle touched by that bar, peel away that many outer color layers.
    - A color layer is one maximal monochrome rectangular boundary, so a two-cell-thick ring still counts as one layer; untouched rectangles stay unchanged.

    Key insight:
    - The gray segment length encodes peel depth, and peeling follows color-layer boundaries rather than one-cell geometric shells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()

    rect_mask = (g != bg) & (g != BAR)
    rect_components = _components(rect_mask)
    rect_id = np.full(g.shape, -1, dtype=int)
    rectangles: list[tuple[int, int, int, int, np.ndarray]] = []
    for idx, coords in enumerate(rect_components):
        r0, c0, r1, c1 = _bbox_coords(coords)
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "rectangle must fill its bbox"
        rect = g[r0 : r1 + 1, c0 : c1 + 1]
        rectangles.append((r0, c0, r1, c1, rect.copy()))
        rect_id[coords[:, 0], coords[:, 1]] = idx

    peeled_depths: dict[int, int] = {}
    for bar_coords in _components(g == BAR):
        r0, c0, r1, c1 = _bbox_coords(bar_coords)
        assert min(r1 - r0 + 1, c1 - c0 + 1) == 1, "gray marker must be a one-cell-thick bar"

        touched = set()
        for r, c in bar_coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and rect_id[nr, nc] != -1:
                    touched.add(int(rect_id[nr, nc]))
        assert len(touched) == 1, "each gray bar must touch exactly one rectangle"
        rect_idx = touched.pop()
        assert rect_idx not in peeled_depths, "each rectangle may have at most one gray bar"
        peeled_depths[rect_idx] = int(bar_coords.shape[0])
        out[bar_coords[:, 0], bar_coords[:, 1]] = bg

    for rect_idx, depth in peeled_depths.items():
        r0, c0, r1, c1, rect = rectangles[rect_idx]
        top, bottom, left, right = _peel_inner_bounds(rect, depth)
        local = np.full(rect.shape, bg, dtype=int)
        if top < bottom and left < right:
            local[top:bottom, left:right] = rect[top:bottom, left:right]
        out[r0 : r1 + 1, c0 : c1 + 1] = local

    return out


def _build_rectangle(force_thick: bool) -> tuple[np.ndarray, int]:
    for _ in range(200):
        n_layers = random.choice([2, 3, 3, 4, 4, 5])
        if n_layers == 1:
            thicknesses: list[int] = []
        else:
            thicknesses = [2 if random.random() < 0.35 else 1 for _ in range(n_layers - 1)]
            if force_thick and 2 not in thicknesses:
                thicknesses[random.randrange(len(thicknesses))] = 2

        inner_h = random.randint(1, 4)
        inner_w = random.randint(1, 5)
        total_border = 2 * sum(thicknesses)
        h = inner_h + total_border
        w = inner_w + total_border
        if not (5 <= h <= 13 and 5 <= w <= 13):
            continue

        colors = random.sample(RECT_COLORS, n_layers)
        rect = np.full((h, w), colors[0], dtype=int)
        top, left = 0, 0
        bottom, right = h, w
        for layer, color in enumerate(colors):
            rect[top:bottom, left:right] = color
            if layer < len(thicknesses):
                t = thicknesses[layer]
                top += t
                bottom -= t
                left += t
                right -= t
        return rect, n_layers
    raise RuntimeError("failed to build a legal nested rectangle")


def _attach_bar(rect: np.ndarray, depth: int) -> np.ndarray:
    h, w = rect.shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        start = random.randint(0, w - depth)
        patch = np.full((h + 1, w), BG, dtype=int)
        if side == "top":
            patch[1:, :] = rect
            patch[0, start : start + depth] = BAR
        else:
            patch[:-1, :] = rect
            patch[h, start : start + depth] = BAR
        return patch

    start = random.randint(0, h - depth)
    patch = np.full((h, w + 1), BG, dtype=int)
    if side == "left":
        patch[:, 1:] = rect
        patch[start : start + depth, 0] = BAR
    else:
        patch[:, :-1] = rect
        patch[start : start + depth, w] = BAR
    return patch


def _make_object(force_thick: bool, touched: bool) -> tuple[np.ndarray, int, int]:
    rect, n_layers = _build_rectangle(force_thick=force_thick)
    if not touched:
        return rect, n_layers, 0
    depth = random.randint(1, n_layers)
    return _attach_bar(rect, depth), n_layers, depth


def _fits(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False

    patch_coords = np.argwhere(patch != BG)
    for pr, pc in patch_coords:
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        if canvas[rr, cc] != BG:
            return False
        for nr in range(rr - padding, rr + padding + 1):
            for nc in range(cc - padding, cc + padding + 1):
                if 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] != BG:
                    return False
    return True


def _place_patch(canvas: np.ndarray, patch: np.ndarray, padding: int = 1) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    positions = [
        (r0, c0)
        for r0 in range(h - ph + 1)
        for c0 in range(w - pw + 1)
        if _fits(canvas, patch, r0, c0, padding)
    ]
    if not positions:
        return False
    r0, c0 = random.choice(positions)
    sub = canvas[r0 : r0 + ph, c0 : c0 + pw]
    sub[patch != BG] = patch[patch != BG]
    return True


def _bg_is_unique_max(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts[BG]
    return bg_count == max(counts.values()) and sum(count == bg_count for count in counts.values()) == 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rect_count = random.choice([1, 2, 2, 3])
        if rect_count == 1:
            touched_count = 1
        elif random.random() < 0.7:
            touched_count = random.randint(1, rect_count - 1)
        else:
            touched_count = rect_count

        touched_flags = [True] * touched_count + [False] * (rect_count - touched_count)
        random.shuffle(touched_flags)

        objects: list[tuple[np.ndarray, int, int]] = []
        for index, touched in enumerate(touched_flags):
            objects.append(_make_object(force_thick=index == 0, touched=touched))

        if touched_count == rect_count and all(depth == n_layers for _patch, n_layers, depth in objects if depth > 0):
            continue

        patches = [patch for patch, _n_layers, _depth in objects]
        max_h = max(patch.shape[0] for patch in patches)
        max_w = max(patch.shape[1] for patch in patches)
        total_area = sum(int(np.count_nonzero(patch != BG)) for patch in patches)

        min_h = max(14, max_h + 2)
        min_w = max(14, max_w + 2)
        max_h_limit = min(30, max(min_h, 18 + rect_count * 3))
        max_w_limit = min(30, max(min_w, 18 + rect_count * 3))
        if min_h > 30 or min_w > 30:
            continue

        grid_h = random.randint(min_h, max_h_limit)
        grid_w = random.randint(min_w, max_w_limit)
        if total_area > (grid_h * grid_w) // 2:
            continue

        canvas = np.full((grid_h, grid_w), BG, dtype=int)
        placed = True
        for patch in sorted(patches, key=lambda arr: int(np.count_nonzero(arr != BG)), reverse=True):
            if not _place_patch(canvas, patch, padding=1):
                placed = False
                break
        if not placed or not _bg_is_unique_max(canvas):
            continue

        out = main(canvas)
        if np.array_equal(out, canvas):
            continue
        assert max(canvas.shape) <= 30 and max(out.shape) <= 30
        return canvas

    raise RuntimeError("failed to generate a valid input grid")
