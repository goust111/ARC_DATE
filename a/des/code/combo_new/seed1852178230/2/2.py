from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

Coord = tuple[int, int]
Shape = frozenset[Coord]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_coords(coords: list[Coord] | set[Coord] | Shape) -> tuple[Shape, tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    if not pts:
        raise ValueError("cannot normalize an empty shape")
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    norm = frozenset((r - r0, c - c0) for r, c in pts)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, (int(h), int(w))


def _shape_to_patch(shape: Shape, color: int, bg: int) -> np.ndarray:
    _, (h, w) = _normalize_coords(shape)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _ordered_factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _can_fill_rectangle(
    shape_a: Shape,
    bbox_a: tuple[int, int],
    shape_b: Shape,
    bbox_b: tuple[int, int],
    rect_h: int,
    rect_w: int,
) -> bool:
    ha, wa = bbox_a
    hb, wb = bbox_b
    if rect_h < ha or rect_w < wa or rect_h < hb or rect_w < wb:
        return False

    coords_a = tuple(shape_a)
    for ra in range(rect_h - ha + 1):
        for ca in range(rect_w - wa + 1):
            occ = np.zeros((rect_h, rect_w), dtype=bool)
            for r, c in coords_a:
                occ[ra + r, ca + c] = True
            complement = [(int(r), int(c)) for r, c in np.argwhere(~occ)]
            comp_norm, _ = _normalize_coords(complement)
            if comp_norm == shape_b:
                return True
    return False


def _reconstruct_rectangle(piece_a: Shape, piece_b: Shape) -> tuple[int, int]:
    total_area = len(piece_a) + len(piece_b)
    shape_a, bbox_a = _normalize_coords(piece_a)
    shape_b, bbox_b = _normalize_coords(piece_b)
    candidates: list[tuple[int, int]] = []
    for rect_h, rect_w in _ordered_factor_pairs(total_area):
        if _can_fill_rectangle(shape_a, bbox_a, shape_b, bbox_b, rect_h, rect_w):
            candidates.append((int(rect_h), int(rect_w)))

    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise AssertionError("expected at least one valid rectangle reconstruction")
    if len(unique) != 1:
        raise AssertionError(f"expected a unique rectangle reconstruction, got {unique}")
    return unique[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Merge each same-color fragment pair into its full rectangle, then center-stack the rectangles from largest area to smallest.

    Input structure:
    - The background is the dominant color.
    - Every non-background color appears in exactly two separate monochrome 4-connected fragments.
    - The two fragments of one color are complementary translation-only pieces of one solid axis-aligned rectangle.
    - The completed rectangles fit inside the largest completed rectangle.

    Rule:
    - For each color, find its two fragments and recover the unique rectangle that they can form by translation only.
    - Measure each recovered rectangle by its area.
    - Create a new canvas whose size equals the largest recovered rectangle.
    - Draw the rectangles in descending area order, centered on that canvas; later rectangles overwrite earlier ones, and odd offsets round toward the top-left.

    Key insight:
    - The input pieces are not independent objects to sort directly; each color first has to be reassembled into a solid rectangle, and only those hidden rectangle sizes determine the nested output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    rectangles: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = _find_components(g == int(color))
        if len(comps) != 2:
            raise AssertionError(f"color {color} should have exactly two components")
        piece_a, _ = _normalize_coords(comps[0])
        piece_b, _ = _normalize_coords(comps[1])
        rect_h, rect_w = _reconstruct_rectangle(piece_a, piece_b)
        rectangles.append((int(rect_h * rect_w), int(rect_h), int(rect_w), int(color)))

    if not rectangles:
        return g.copy()

    rectangles.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    out_h = rectangles[0][1]
    out_w = rectangles[0][2]
    out = np.full((out_h, out_w), int(bg), dtype=int)

    for _area, rect_h, rect_w, color in rectangles:
        r0 = (out_h - rect_h) // 2
        c0 = (out_w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(color)

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    if random.random() < 0.5:
        arr = np.flipud(arr)
    return arr.copy()


def _split_rectangle(rect_h: int, rect_w: int) -> tuple[Shape, Shape, tuple[int, int]] | None:
    for _ in range(80):
        if random.random() < 0.5:
            cuts = [random.randint(0, rect_w - 2) for _ in range(rect_h)]
            if len(set(cuts)) == 1 and random.random() < 0.8:
                continue
            mask = np.zeros((rect_h, rect_w), dtype=bool)
            for r, cut in enumerate(cuts):
                mask[r, : cut + 1] = True
        else:
            cuts = [random.randint(0, rect_h - 2) for _ in range(rect_w)]
            if len(set(cuts)) == 1 and random.random() < 0.8:
                continue
            mask = np.zeros((rect_h, rect_w), dtype=bool)
            for c, cut in enumerate(cuts):
                mask[: cut + 1, c] = True

        area_a = int(mask.sum())
        area_b = int(mask.size - area_a)
        if min(area_a, area_b) < 3:
            continue

        mask = _random_transform(mask)
        rect_dims = (int(mask.shape[0]), int(mask.shape[1]))
        shape_a, _ = _normalize_coords([(int(r), int(c)) for r, c in np.argwhere(mask)])
        shape_b, _ = _normalize_coords([(int(r), int(c)) for r, c in np.argwhere(~mask)])
        try:
            if _reconstruct_rectangle(shape_a, shape_b) != rect_dims:
                continue
        except AssertionError:
            continue
        return shape_a, shape_b, rect_dims
    return None


def _center_stack(rectangles: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: (item["area"], item["h"], item["w"]), reverse=True)
    out = np.full((ordered[0]["h"], ordered[0]["w"]), int(bg), dtype=int)
    for rect in ordered:
        r0 = (out.shape[0] - rect["h"]) // 2
        c0 = (out.shape[1] - rect["w"]) // 2
        out[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = int(rect["color"])
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    dil = src.copy()
    dil[1:, :] |= src[:-1, :]
    dil[:-1, :] |= src[1:, :]
    dil[:, 1:] |= src[:, :-1]
    dil[:, :-1] |= src[:, 1:]
    dil[1:, 1:] |= src[:-1, :-1]
    dil[1:, :-1] |= src[:-1, 1:]
    dil[:-1, 1:] |= src[1:, :-1]
    dil[:-1, :-1] |= src[1:, 1:]
    return dil


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False

    mask = patch != int(bg)
    dilated = _dilate8(occupied)
    row_lo = 1 if h - ph >= 2 else 0
    row_hi = h - ph - 1 if h - ph >= 2 else h - ph
    col_lo = 1 if w - pw >= 2 else 0
    col_hi = w - pw - 1 if w - pw >= 2 else w - pw

    spots = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(spots)
    for r0, c0 in spots:
        if np.any(dilated[r0 : r0 + ph, c0 : c0 + pw] & mask):
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
        return True
    return False


def _sample_rectangles(bg: int) -> list[dict] | None:
    n_rects = random.randint(2, 4)
    palette = [c for c in range(10) if c != int(bg)]
    colors = random.sample(palette, n_rects)

    outer_h = random.randint(5, 10)
    outer_w = random.randint(5, 10)

    candidates = [(h, w) for h in range(3, outer_h + 1) for w in range(3, outer_w + 1)]
    random.shuffle(candidates)
    dims: list[tuple[int, int]] = [(outer_h, outer_w)]
    used_areas = {outer_h * outer_w}

    for h, w in candidates:
        if (h, w) == (outer_h, outer_w) or h * w in used_areas:
            continue
        dims.append((int(h), int(w)))
        used_areas.add(h * w)
        if len(dims) == n_rects:
            break
    if len(dims) != n_rects:
        return None

    rectangles: list[dict] = []
    for color, (rect_h, rect_w) in zip(colors, dims):
        split = _split_rectangle(rect_h, rect_w)
        if split is None:
            return None
        piece_a, piece_b, (actual_h, actual_w) = split
        if actual_h > outer_h or actual_w > outer_w:
            return None
        rectangles.append(
            {
                "color": int(color),
                "h": int(actual_h),
                "w": int(actual_w),
                "area": int(actual_h * actual_w),
                "piece_a": piece_a,
                "piece_b": piece_b,
            }
        )

    outer = max(rectangles, key=lambda item: item["area"])
    if any(rect["h"] > outer["h"] or rect["w"] > outer["w"] for rect in rectangles):
        return None
    return rectangles


def _build_input(rectangles: list[dict], bg: int) -> np.ndarray | None:
    total_fg = sum(rect["area"] for rect in rectangles)
    min_side = max(18, max(max(rect["h"], rect["w"]) for rect in rectangles) + 8)
    max_side = 30
    for _ in range(50):
        h = random.randint(min_side, max_side)
        w = random.randint(min_side, max_side)
        if total_fg * 2 >= h * w:
            continue

        grid = np.full((h, w), int(bg), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        patches: list[np.ndarray] = []
        for rect in rectangles:
            if random.random() < 0.5:
                first, second = rect["piece_a"], rect["piece_b"]
            else:
                first, second = rect["piece_b"], rect["piece_a"]
            patches.append(_shape_to_patch(first, rect["color"], bg))
            patches.append(_shape_to_patch(second, rect["color"], bg))

        patches.sort(key=lambda patch: (patch.shape[0] * patch.shape[1], patch.shape[0], patch.shape[1]), reverse=True)
        ok = True
        for patch in patches:
            if not _place_patch(grid, occupied, patch, bg):
                ok = False
                break
        if ok:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        bg = random.randint(0, 9)
        rectangles = _sample_rectangles(bg)
        if rectangles is None:
            continue
        expected = _center_stack(rectangles, bg)
        inp = _build_input(rectangles, bg)
        if inp is None:
            continue
        try:
            out = main(inp)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
