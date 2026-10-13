from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _pattern_is_asymmetric(patch: np.ndarray) -> bool:
    patch = np.asarray(patch, dtype=int)
    variants = [
        np.flipud(patch),
        np.fliplr(patch),
        np.flipud(np.fliplr(patch)),
    ]
    if patch.shape[0] == patch.shape[1]:
        variants.extend([np.rot90(patch, 1), np.rot90(patch, 3), patch.T, np.fliplr(patch.T)])
    return all(not np.array_equal(patch, variant) for variant in variants)


def _rows_for_color(g: np.ndarray, color: int, bg: int) -> list[int]:
    h, w = g.shape
    rows: list[int] = []
    for r in range(h):
        row = g[r, :]
        if int(np.count_nonzero(row == color)) != w - 1:
            continue
        if int(np.count_nonzero(row == bg)) != 1:
            continue
        if np.all((row == color) | (row == bg)):
            rows.append(int(r))
    return rows


def _cols_for_color(g: np.ndarray, color: int, bg: int) -> list[int]:
    h, w = g.shape
    cols: list[int] = []
    for c in range(w):
        col = g[:, c]
        if int(np.count_nonzero(col == color)) != h - 1:
            continue
        if int(np.count_nonzero(col == bg)) != 1:
            continue
        if np.all((col == color) | (col == bg)):
            cols.append(int(c))
    return cols


def _find_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    h, w = g.shape

    candidates: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        rows = _rows_for_color(g, color, bg)
        cols = _cols_for_color(g, color, bg)
        if len(rows) == 2 and len(cols) == 0:
            rows = sorted(rows)
            holes = [(rows[0], int(np.where(g[rows[0], :] == bg)[0][0])), (rows[1], int(np.where(g[rows[1], :] == bg)[0][0]))]
            candidates.append({"orient": "h", "wall_color": int(color), "lines": rows, "holes_sorted": holes})
        if len(cols) == 2 and len(rows) == 0:
            cols = sorted(cols)
            holes = [(int(np.where(g[:, cols[0]] == bg)[0][0]), cols[0]), (int(np.where(g[:, cols[1]] == bg)[0][0]), cols[1])]
            candidates.append({"orient": "v", "wall_color": int(color), "lines": cols, "holes_sorted": holes})

    assert len(candidates) == 1, "expected exactly one pair of parallel single-hole walls"
    layout = candidates[0]
    wall_color = int(layout["wall_color"])
    source_mask = (g != bg) & (g != wall_color)
    assert bool(source_mask.any()), "expected one source pattern away from the walls"

    if layout["orient"] == "h":
        r0, r1 = layout["lines"]
        top_has = bool(source_mask[:r0, :].any())
        middle_has = bool(source_mask[r0 + 1 : r1, :].any())
        bottom_has = bool(source_mask[r1 + 1 :, :].any())
        assert not middle_has, "middle band must be empty in the input"
        assert top_has ^ bottom_has, "source pattern must occupy exactly one outer band"
        if top_has:
            source_side = "top"
            hole1, hole2 = layout["holes_sorted"][0], layout["holes_sorted"][1]
        else:
            source_side = "bottom"
            hole1, hole2 = layout["holes_sorted"][1], layout["holes_sorted"][0]
    else:
        c0, c1 = layout["lines"]
        left_has = bool(source_mask[:, :c0].any())
        middle_has = bool(source_mask[:, c0 + 1 : c1].any())
        right_has = bool(source_mask[:, c1 + 1 :].any())
        assert not middle_has, "middle band must be empty in the input"
        assert left_has ^ right_has, "source pattern must occupy exactly one outer band"
        if left_has:
            source_side = "left"
            hole1, hole2 = layout["holes_sorted"][0], layout["holes_sorted"][1]
        else:
            source_side = "right"
            hole1, hole2 = layout["holes_sorted"][1], layout["holes_sorted"][0]

    layout = dict(layout)
    layout["bg"] = int(bg)
    layout["shape"] = (int(h), int(w))
    layout["hole1"] = tuple(map(int, hole1))
    layout["hole2"] = tuple(map(int, hole2))
    layout["source_side"] = source_side
    layout["source_mask"] = source_mask
    return layout


def _is_middle_cell(layout: dict, r: int, c: int) -> bool:
    lo, hi = sorted(layout["lines"])
    if layout["orient"] == "h":
        return lo < int(r) < hi
    return lo < int(c) < hi


def _is_third_cell(layout: dict, r: int, c: int) -> bool:
    lo, hi = sorted(layout["lines"])
    side = layout["source_side"]
    if layout["orient"] == "h":
        return int(r) > hi if side == "top" else int(r) < lo
    return int(c) > hi if side == "left" else int(c) < lo


def _double_image_positions(grid: np.ndarray) -> tuple[dict, list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    layout = _find_layout(g)
    source_mask = layout["source_mask"]
    source_colors = set(map(int, g[source_mask]))
    assert len(source_colors) >= 2, "source pattern must be multicolor"
    assert _is_connected(source_mask), "source pattern must form one connected object"

    h1r, h1c = layout["hole1"]
    h2r, h2c = layout["hole2"]

    first_full: list[tuple[int, int, int]] = []
    first_visible: list[tuple[int, int, int]] = []
    second_full: list[tuple[int, int, int]] = []

    for r0, c0 in np.argwhere(source_mask):
        r = int(r0)
        c = int(c0)
        color = int(g[r, c])

        rr = 2 * h1r - r
        cc = 2 * h1c - c
        first_full.append((rr, cc, color))
        if _in_bounds(g.shape, rr, cc) and _is_middle_cell(layout, rr, cc):
            first_visible.append((rr, cc, color))

        r2 = r + 2 * (h2r - h1r)
        c2 = c + 2 * (h2c - h1c)
        second_full.append((r2, c2, color))

    return layout, first_full, first_visible, second_full


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade a multicolor source through two single-cell pinholes to draw a clipped first image and an upright translated second image.

    Input structure:
    - Two parallel full wall lines of one color divide the grid into three bands, and each wall has exactly one background hole.
    - A single connected multicolor source pattern lies in exactly one outer band; the middle band and far band start as background.
    - The hole positions usually differ along the wall direction, so the cascade is not an axis mirror.

    Rule:
    - Detect which outer band contains the source; the adjacent wall is the first pinhole wall and the other wall is the second.
    - Reflect every source pixel through the first hole as a point reflection. Draw only the reflected pixels that land inside the middle band; pixels on walls, outside the grid, or beyond the middle band stay hidden.
    - Reflect the complete geometric first image through the second hole. Equivalently, translate every source pixel by twice the hole-to-hole offset and draw the result in the far band.

    Key insight:
    - Each pinhole performs point reflection around the hole cell, not line reflection across the wall. Even first-step pixels clipped from view still participate in the second step, so the final image is a translated upright copy of the source.
    """
    g = np.asarray(grid, dtype=int)
    layout, first_full, first_visible, second_full = _double_image_positions(g)
    out = g.copy()

    for r, c, color in first_visible:
        out[r, c] = int(color)

    for r, c, color in second_full:
        assert _in_bounds(g.shape, r, c), "second image must stay inside the grid"
        assert _is_third_cell(layout, r, c), "second image must lie in the far band"
        out[r, c] = int(color)

    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < h and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < w and mask[r, c + 1])
            ):
                cells.append((r, c))
    return cells


def _grow_connected_mask(h: int, w: int, count: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < count:
        mask[random.choice(_frontier(mask))] = True
    return mask


def _split_counts(total: int, parts: int) -> list[int]:
    counts = [1] * parts
    for _ in range(total - parts):
        counts[random.randrange(parts)] += 1
    random.shuffle(counts)
    return counts


def _generate_patch(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 3 or max_w < 3:
        return None
    for _ in range(200):
        h = random.randint(3, max_h)
        w = random.randint(3, max_w)
        area = h * w
        count_lo = max(5, (area + 1) // 2)
        count_hi = min(area - 1, 10)
        if count_lo > count_hi:
            continue
        mask = _grow_connected_mask(h, w, random.randint(count_lo, count_hi))
        colors = random.sample(PALETTE, random.randint(2, min(4, int(mask.sum()))))
        counts = _split_counts(int(mask.sum()), len(colors))
        patch = np.full((h, w), BG, dtype=int)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)
        idx = 0
        for color, size in zip(colors, counts):
            for r, c in coords[idx : idx + size]:
                patch[r, c] = int(color)
            idx += size
        if _pattern_is_asymmetric(patch):
            return patch
    return None


def _place_source_cols(width: int, patch: np.ndarray, clipped: bool) -> tuple[int, int, int] | None:
    occupied = np.argwhere(patch != BG)
    _, occ_c0, _, occ_c1 = _bbox(patch != BG)

    if clipped:
        choices: list[tuple[int, int, int]] = []
        for hole1_c in range(1, min(4, width - 3)):
            boundary = 2 * hole1_c
            for delta in range(1, min(4, width - hole1_c)):
                hole2_c = hole1_c + delta
                lo = max(0, boundary - occ_c1 + 1, -occ_c0 - 2 * delta)
                hi = min(width - patch.shape[1], boundary - occ_c0, width - 1 - occ_c1 - 2 * delta)
                if lo > hi:
                    continue
                for c0 in range(lo, hi + 1):
                    source_cols = {c0 + int(c) for _r, c in occupied}
                    reflected_cols = {2 * hole1_c - c for c in source_cols}
                    if any(col < 0 for col in reflected_cols) and any(0 <= col < width for col in reflected_cols):
                        choices.append((hole1_c, hole2_c, c0))
        return random.choice(choices) if choices else None

    choices = []
    for hole1_c in range(2, width - 2):
        for delta in (-3, -2, -1, 1, 2, 3):
            hole2_c = hole1_c + delta
            if not 0 <= hole2_c < width:
                continue
            lo = max(0, 2 * hole1_c - (width - 1) - occ_c0, -occ_c0 - 2 * delta)
            hi = min(width - patch.shape[1], 2 * hole1_c - occ_c1, width - 1 - occ_c1 - 2 * delta)
            if lo > hi:
                continue
            choices.append((hole1_c, hole2_c, random.randint(lo, hi)))
    return random.choice(choices) if choices else None


def _canonical_case(clipped: bool) -> np.ndarray | None:
    for _ in range(200):
        middle = random.randint(4, 8)
        top = random.randint(middle, min(10, middle + 4))
        bottom = random.randint(middle, min(10, middle + 4))
        height = top + middle + bottom + 2
        width = random.randint(16, 24)

        patch = _generate_patch(min(5, top, middle, bottom), min(7, width - 6))
        if patch is None:
            continue
        ph, pw = patch.shape
        row_lo = top - middle
        row_hi = top - ph
        if row_lo > row_hi:
            continue
        r0 = random.randint(row_lo, row_hi)

        cols = _place_source_cols(width, patch, clipped)
        if cols is None:
            continue
        hole1_c, hole2_c, c0 = cols

        g = np.full((height, width), BG, dtype=int)
        wall1_r = top
        wall2_r = top + middle + 1
        g[wall1_r, :] = WALL
        g[wall2_r, :] = WALL
        g[wall1_r, hole1_c] = BG
        g[wall2_r, hole2_c] = BG

        source_mask = patch != BG
        g[r0 : r0 + ph, c0 : c0 + pw][source_mask] = patch[source_mask]
        return g
    return None


def _random_global_transform(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if random.choice([False, True]):
        out = np.flipud(out)
    if random.choice([False, True]):
        out = np.fliplr(out)
    if random.choice([False, True]):
        out = out.T
    return out.copy()


def _validate_semantics(inp: np.ndarray, expect_clipped: bool) -> bool:
    layout, first_full, first_visible, second_full = _double_image_positions(inp)
    out = main(inp)

    source_patch = _crop_bbox(inp, layout["source_mask"])
    if not _pattern_is_asymmetric(source_patch):
        return False

    source_coords = [(int(r), int(c), int(inp[r, c])) for r, c in np.argwhere(layout["source_mask"])]
    if len(source_coords) < 5:
        return False
    if len(set(color for _r, _c, color in source_coords)) < 2:
        return False

    if layout["orient"] == "h":
        tangential = layout["hole2"][1] - layout["hole1"][1]
    else:
        tangential = layout["hole2"][0] - layout["hole1"][0]
    if tangential == 0:
        return False

    expected = np.asarray(inp, dtype=int).copy()
    for r, c, color in first_visible:
        expected[r, c] = int(color)
    for r, c, color in second_full:
        if not _in_bounds(inp.shape, r, c):
            return False
        if not _is_third_cell(layout, r, c):
            return False
        expected[r, c] = int(color)
    if not np.array_equal(out, expected):
        return False

    if len(first_visible) == 0 or len(second_full) == 0:
        return False

    first_visible_set = {(r, c, color) for r, c, color in first_visible}
    hidden = [triple for triple in first_full if triple not in first_visible_set]
    if expect_clipped != bool(hidden):
        return False

    second_from_visible = []
    h2r, h2c = layout["hole2"]
    for r, c, color in first_visible:
        second_from_visible.append((2 * h2r - r, 2 * h2c - c, color))
    full_second = set(second_full)
    visible_second = set(second_from_visible)
    if expect_clipped:
        if len(hidden) == 0:
            return False
        if not visible_second < full_second:
            return False
    else:
        if len(hidden) != 0:
            return False
        if full_second != visible_second:
            return False

    dr = 2 * (layout["hole2"][0] - layout["hole1"][0])
    dc = 2 * (layout["hole2"][1] - layout["hole1"][1])
    translated = {(r + dr, c + dc, color) for r, c, color in source_coords}
    if translated != full_second:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        clipped = random.random() < 0.6
        base = _canonical_case(clipped)
        if base is None:
            continue
        grid = _random_global_transform(base)
        try:
            if _validate_semantics(grid, expect_clipped=clipped):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid double-pinhole cascade input")
