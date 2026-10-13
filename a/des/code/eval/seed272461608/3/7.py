from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 200
GEN_PLACE_ATTEMPTS = 300
GEN_COLORS = list(range(10))
GEN_GRID_RANGE = (22, 31)
GEN_OBJECT_COUNTS = [2, 3, 4]
GEN_PADDING = 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(cells)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    comps = _components(bg)
    holes = 0
    for comp in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return int(holes)


def _is_connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    return cells > 0 and len(_components(mask)) == 1


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _classify_l(mask: np.ndarray) -> int:
    h, w = mask.shape
    assert h >= 2 and w >= 2
    patterns: list[np.ndarray] = []

    p0 = np.zeros((h, w), dtype=bool)
    p0[-1, :] = True
    p0[:, 0] = True
    patterns.append(p0)

    p1 = np.zeros((h, w), dtype=bool)
    p1[0, :] = True
    p1[:, 0] = True
    patterns.append(p1)

    p2 = np.zeros((h, w), dtype=bool)
    p2[0, :] = True
    p2[:, -1] = True
    patterns.append(p2)

    p3 = np.zeros((h, w), dtype=bool)
    p3[-1, :] = True
    p3[:, -1] = True
    patterns.append(p3)

    for idx, pat in enumerate(patterns):
        if np.array_equal(mask, pat):
            return int(idx)
    raise AssertionError("legend component must be an L mark")


def _legend_rotation_map(
    grid: np.ndarray, bg: int
) -> tuple[tuple[int, int, int, int], np.ndarray, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, tuple[int, int, int, int], dict[int, int]]] = []

    for color0 in np.unique(g):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components(g == color):
            bbox = _bbox(comp)
            if not _touches_corner(bbox, g.shape):
                continue
            r0, c0, r1, c1 = bbox
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue

            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(sub == color, border):
                continue

            inner = sub[1:-1, 1:-1]
            inner_mask = (inner != bg) & (inner != color)
            parts = _components(inner_mask)
            if len(parts) != 4:
                continue

            try:
                parts_sorted = sorted(parts, key=lambda cells: _bbox(cells)[1])
                orientations = []
                for cells in parts_sorted:
                    pr0, pc0, pr1, pc1 = _bbox(cells)
                    local = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
                    for rr, cc in cells:
                        local[rr - pr0, cc - pc0] = True
                    orientations.append(_classify_l(local))
            except AssertionError:
                continue

            turns = {idx: int((ori - orientations[0]) % 4) for idx, ori in enumerate(orientations)}
            if {int(v) for v in turns.values()} != {0, 1, 2, 3}:
                continue
            candidates.append((int(border.sum()), bbox, turns))

    assert candidates, "expected one corner legend frame"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _area, bbox, turns = candidates[0]
    r0, c0, r1, c1 = bbox
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return bbox, legend_mask, turns


def _paste_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == mask.shape
    target[mask] = int(color)


def _rotated_top_left(
    bbox: tuple[int, int, int, int], rotated_shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    hh, ww = rotated_shape
    nr0 = (int(r0 + r1) - (int(hh) - 1)) // 2
    nc0 = (int(c0 + c1) - (int(ww) - 1)) // 2
    return int(nr0), int(nc0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored shape in place according to the number of enclosed holes,
    using the corner legend to decode 0/1/2/3-hole -> rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    _legend_bbox, legend_mask, turns = _legend_rotation_map(g, bg)

    out = g.copy()
    object_cells = (g != bg) & (~legend_mask)
    out[object_cells] = bg

    objects: list[tuple[int, list[tuple[int, int]]]] = []
    for color0 in np.unique(g[~legend_mask]):
        color = int(color0)
        if color == bg:
            continue
        for comp in _components((g == color) & (~legend_mask)):
            objects.append((color, comp))

    assert objects, "expected at least one non-legend object"

    for color, cells in objects:
        r0, c0, r1, c1 = _bbox(cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for rr, cc in cells:
            patch[rr - r0, cc - c0] = color

        assert patch.shape[0] % 2 == patch.shape[1] % 2, "bbox center must support quarter turns"
        hole_count = _count_holes(patch == color)
        assert 0 <= hole_count <= 3
        turn = int(turns[hole_count])
        rotated = np.rot90(patch, -turn)
        nr0, nc0 = _rotated_top_left((r0, c0, r1, c1), rotated.shape)

        target = out[nr0 : nr0 + rotated.shape[0], nc0 : nc0 + rotated.shape[1]]
        assert target.shape == rotated.shape, "rotated shape must stay within the grid"
        mask = rotated != bg
        assert not np.any(target[mask] != bg), "rotated shapes must not overlap"
        target[mask] = rotated[mask]

    return out


def _build_l_patch(orientation: int, size: int = 3) -> np.ndarray:
    patch = np.zeros((size, size), dtype=bool)
    if orientation == 0:
        patch[-1, :] = True
        patch[:, 0] = True
    elif orientation == 1:
        patch[0, :] = True
        patch[:, 0] = True
    elif orientation == 2:
        patch[0, :] = True
        patch[:, -1] = True
    elif orientation == 3:
        patch[-1, :] = True
        patch[:, -1] = True
    else:
        raise AssertionError("invalid orientation")
    return patch


def _build_legend(bg: int, frame_color: int, glyph_color: int, base_orientation: int) -> np.ndarray:
    legend = np.full((5, 17), bg, dtype=int)
    legend[0, :] = frame_color
    legend[-1, :] = frame_color
    legend[:, 0] = frame_color
    legend[:, -1] = frame_color

    for idx in range(4):
        c0 = idx * 4
        patch = _build_l_patch((base_orientation + idx) % 4)
        legend[1:4, 1 + c0 : 1 + c0 + 3][patch] = glyph_color
    return legend


def _legend_position(shape: tuple[int, int], corner: str, legend_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    lh, lw = legend_shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - lw
    if corner == "bl":
        return h - lh, 0
    return h - lh, w - lw


def _count_holes_and_validate(mask: np.ndarray) -> int:
    assert _is_connected(mask)
    assert _touches_all_edges(mask)
    holes = _count_holes(mask)
    return int(holes)


def _try_add_holes(mask: np.ndarray, hole_count: int, rng: np.random.Generator) -> bool:
    h, w = mask.shape
    blocked = np.zeros_like(mask, dtype=bool)
    holes_done = 0
    tries = 0
    while holes_done < hole_count and tries < 200:
        tries += 1
        hh = int(rng.integers(1, 3))
        ww = int(rng.integers(1, 3))
        max_r = h - hh - 1
        max_c = w - ww - 1
        if max_r <= 1 or max_c <= 1:
            return False
        r0 = int(rng.integers(1, max_r))
        c0 = int(rng.integers(1, max_c))
        r1 = r0 + hh - 1
        c1 = c0 + ww - 1
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        if bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
            continue
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        holes_done += 1
    return holes_done == hole_count


def _random_notches(mask: np.ndarray, hole_count: int, rng: np.random.Generator) -> None:
    h, w = mask.shape
    notch_tries = int(rng.integers(2, 6))
    for _ in range(notch_tries):
        candidate = mask.copy()
        side = str(rng.choice(["top", "bottom", "left", "right"]))
        if side in {"top", "bottom"}:
            depth = int(rng.integers(1, max(2, h // 2)))
            width = int(rng.integers(1, max(2, w - 1)))
            c0 = int(rng.integers(0, w - width + 1))
            if side == "top":
                candidate[:depth, c0 : c0 + width] = False
            else:
                candidate[h - depth :, c0 : c0 + width] = False
        else:
            depth = int(rng.integers(1, max(2, w // 2)))
            height = int(rng.integers(1, max(2, h - 1)))
            r0 = int(rng.integers(0, h - height + 1))
            if side == "left":
                candidate[r0 : r0 + height, :depth] = False
            else:
                candidate[r0 : r0 + height, w - depth :] = False

        if not _touches_all_edges(candidate):
            continue
        if not _is_connected(candidate):
            continue
        if _count_holes(candidate) != hole_count:
            continue
        if int(candidate.sum()) < max(8, hole_count * 6 + 6):
            continue
        mask[:, :] = candidate


def _shape_changes_under_rotation(mask: np.ndarray, turn: int) -> bool:
    if turn % 4 == 0:
        return True
    rotated = np.rot90(mask, -turn)
    if mask.shape != rotated.shape:
        return True
    return not np.array_equal(mask, rotated)


def _generate_shape(hole_count: int, turn: int, rng: np.random.Generator) -> np.ndarray | None:
    min_side = 7 if hole_count >= 2 else 5
    if hole_count == 3:
        min_side = 9

    for _ in range(GEN_SHAPE_ATTEMPTS):
        h = int(rng.choice([v for v in range(min_side, 12) if v % 2 == 1]))
        w = int(rng.choice([v for v in range(min_side, 12) if v % 2 == 1]))
        mask = np.ones((h, w), dtype=bool)
        if not _try_add_holes(mask, hole_count, rng):
            continue
        _random_notches(mask, hole_count, rng)
        if _count_holes_and_validate(mask) != hole_count:
            continue
        if not _shape_changes_under_rotation(mask, turn):
            continue
        return mask
    return None


def _can_place(
    occupied: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int, padding: int
) -> bool:
    h, w = occupied.shape
    ph, pw = patch_mask.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    for rr, cc in np.argwhere(patch_mask):
        r = int(r0 + rr)
        c = int(c0 + cc)
        rr0 = max(0, r - padding)
        cc0 = max(0, c - padding)
        rr1 = min(h, r + padding + 1)
        cc1 = min(w, c + padding + 1)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            return False
    return True


def _mark_occupied(occupied: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch_mask.shape
    occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch_mask


def _place_object(
    inp: np.ndarray,
    expected: np.ndarray,
    occupied_inp: np.ndarray,
    occupied_out: np.ndarray,
    mask: np.ndarray,
    color: int,
    turn: int,
    rng: np.random.Generator,
) -> bool:
    for _ in range(GEN_PLACE_ATTEMPTS):
        input_turn = int(rng.integers(0, 4))
        input_mask = np.rot90(mask, -input_turn)
        output_mask = np.rot90(input_mask, -turn)
        in_h, in_w = input_mask.shape
        out_h, out_w = output_mask.shape
        max_h = max(in_h, out_h)
        max_w = max(in_w, out_w)
        if max_h > inp.shape[0] or max_w > inp.shape[1]:
            return False

        center_r = int(rng.integers(max_h // 2, inp.shape[0] - max_h // 2 + 1))
        center_c = int(rng.integers(max_w // 2, inp.shape[1] - max_w // 2 + 1))
        in_r0 = center_r - in_h // 2
        in_c0 = center_c - in_w // 2
        out_r0 = center_r - out_h // 2
        out_c0 = center_c - out_w // 2

        if not _can_place(occupied_inp, input_mask, in_r0, in_c0, GEN_PADDING):
            continue
        if not _can_place(occupied_out, output_mask, out_r0, out_c0, GEN_PADDING):
            continue

        _paste_mask(inp, input_mask, color, in_r0, in_c0)
        _paste_mask(expected, output_mask, color, out_r0, out_c0)
        _mark_occupied(occupied_inp, input_mask, in_r0, in_c0)
        _mark_occupied(occupied_out, output_mask, out_r0, out_c0)
        return True
    return False


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GEN_GRID_RANGE))
    w = int(rng.integers(*GEN_GRID_RANGE))
    bg = int(rng.integers(0, 10))

    object_count = int(rng.choice(GEN_OBJECT_COUNTS))
    hole_counts = list(map(int, rng.choice([0, 1, 2, 3], size=object_count, replace=False)))

    used = {bg}
    if len(used) + object_count + 2 > len(GEN_COLORS):
        return None
    pool = [c for c in GEN_COLORS if c not in used]
    rng.shuffle(pool)
    frame_color = int(pool.pop())
    glyph_color = int(pool.pop())
    object_colors = [int(pool.pop()) for _ in range(object_count)]

    legend = _build_legend(bg, frame_color, glyph_color, int(rng.integers(0, 4)))
    corner = str(rng.choice(["tl", "tr", "bl", "br"]))
    lr0, lc0 = _legend_position((h, w), corner, legend.shape)

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    inp[lr0 : lr0 + legend.shape[0], lc0 : lc0 + legend.shape[1]] = legend
    expected[lr0 : lr0 + legend.shape[0], lc0 : lc0 + legend.shape[1]] = legend

    occupied_inp = np.zeros((h, w), dtype=bool)
    occupied_out = np.zeros((h, w), dtype=bool)
    occupied_inp[lr0 : lr0 + legend.shape[0], lc0 : lc0 + legend.shape[1]] = legend != bg
    occupied_out[lr0 : lr0 + legend.shape[0], lc0 : lc0 + legend.shape[1]] = legend != bg

    for hole_count, color in zip(hole_counts, object_colors, strict=False):
        turn = int(hole_count)
        mask = _generate_shape(hole_count, turn, rng)
        if mask is None:
            return None
        if not _place_object(inp, expected, occupied_inp, occupied_out, mask, color, turn, rng):
            return None

    if np.count_nonzero(inp == bg) <= (h * w) // 2:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
