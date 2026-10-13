from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            part: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                part.add((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(part)
    return parts


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _axis_reflect(
    r: int,
    c: int,
    bbox: tuple[int, int, int, int],
    axis: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if axis == "vertical":
        return r, c0 + c1 - c
    return r0 + r1 - r, c


def _target_from_marker(grid: np.ndarray) -> tuple[tuple[int, int, int, int], str, int]:
    red_pts = np.argwhere(grid == RED)
    assert red_pts.shape == (1, 2), "expected exactly one red marker"
    rr, rc = map(int, red_pts[0])

    neighbors: list[tuple[str, int, int, int]] = []
    side_by_dir = {
        (-1, 0): "bottom",
        (1, 0): "top",
        (0, -1): "right",
        (0, 1): "left",
    }
    for dr, dc in DIRS4:
        nr, nc = rr + dr, rc + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            color = int(grid[nr, nc])
            if color not in (BG, RED):
                neighbors.append((side_by_dir[(dr, dc)], color, nr, nc))

    candidates: list[tuple[tuple[int, int, int, int], str, int]] = []
    for side, border_color, nr, nc in neighbors:
        border_components = _components(grid == border_color)
        target = next(part for part in border_components if (nr, nc) in part)
        bbox = _bbox(target)

        r0, c0, r1, c1 = bbox
        if side == "left" and nc != c0:
            continue
        if side == "right" and nc != c1:
            continue
        if side == "top" and nr != r0:
            continue
        if side == "bottom" and nr != r1:
            continue

        try:
            _covered_span(grid, bbox, side, border_color)
        except AssertionError:
            continue
        candidates.append((bbox, side, border_color))

    assert len(candidates) == 1, "marker must identify exactly one partially covered target"
    return candidates[0]


def _covered_span(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    border_color: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "left":
        coords = [r for r in range(r0, r1 + 1) if int(grid[r, c0]) != border_color]
    elif side == "right":
        coords = [r for r in range(r0, r1 + 1) if int(grid[r, c1]) != border_color]
    elif side == "top":
        coords = [c for c in range(c0, c1 + 1) if int(grid[r0, c]) != border_color]
    else:
        coords = [c for c in range(c0, c1 + 1) if int(grid[r1, c]) != border_color]

    assert coords, "target border must contain a hidden segment"
    assert coords[-1] - coords[0] + 1 == len(coords), "hidden border segment must be contiguous"
    return coords[0], coords[-1]


def _covered_box(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    span0: int,
    span1: int,
) -> tuple[int, int, int, int, str]:
    r0, c0, r1, c1 = bbox
    axis = "vertical" if side in {"left", "right"} else "horizontal"

    if side in {"left", "right"}:
        depths = []
        for r in range(span0, span1 + 1):
            depth = 0
            cols = range(c0, c1 + 1) if side == "left" else range(c1, c0 - 1, -1)
            for c in cols:
                mr, mc = _axis_reflect(r, c, bbox, axis)
                if int(grid[r, c]) != int(grid[mr, mc]):
                    depth += 1
                else:
                    break
            depths.append(depth)
        depth = max(depths)
        assert depth > 0 and all(d == depth for d in depths), "covered strip width must be constant"
        if side == "left":
            return span0, c0, span1, c0 + depth - 1, axis
        return span0, c1 - depth + 1, span1, c1, axis

    depths = []
    for c in range(span0, span1 + 1):
        depth = 0
        rows = range(r0, r1 + 1) if side == "top" else range(r1, r0 - 1, -1)
        for r in rows:
            mr, mc = _axis_reflect(r, c, bbox, axis)
            if int(grid[r, c]) != int(grid[mr, mc]):
                depth += 1
            else:
                break
        depths.append(depth)
    depth = max(depths)
    assert depth > 0 and all(d == depth for d in depths), "covered strip height must be constant"
    if side == "top":
        return r0, span0, r0 + depth - 1, span1, axis
    return r1 - depth + 1, span0, r1, span1, axis


def _validate_visible_symmetry(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    covered_box: tuple[int, int, int, int],
    axis: str,
) -> None:
    r0, c0, r1, c1 = bbox
    cr0, cc0, cr1, cc1 = covered_box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            in_cover = cr0 <= r <= cr1 and cc0 <= c <= cc1
            mr, mc = _axis_reflect(r, c, bbox, axis)
            mirror_in_cover = cr0 <= mr <= cr1 and cc0 <= mc <= cc1
            if in_cover or mirror_in_cover:
                continue
            assert int(grid[r, c]) == int(grid[mr, mc]), "visible target must respect a single mirror axis"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover only the hidden part of a marked symmetric rectangle from the visible half.

    Input structure:
    - Black background with several axis-aligned colored rectangles.
    - Exactly one rectangle is partially covered by another rectangle, and a single red pixel marks one visible endpoint of the covered side.
    - The marked target rectangle keeps a one-color border and an interior pattern that is symmetric about either its vertical or horizontal midline.

    Rule:
    - Use the red marker to locate the target rectangle and determine which side is covered.
    - Read the target rectangle's full bounding box from its visible border, then find the contiguous covered strip on that side.
    - Reflect visible target pixels across the rectangle's symmetry axis to reconstruct the covered strip, and output only that recovered strip cropped to its bounding box.

    Key insight:
    - The marker identifies the damaged rectangle and the damaged side; because the target keeps its full outer box and has a single mirror symmetry, the missing content is exactly the mirror image of the visible opposite side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bbox, side, border_color = _target_from_marker(g)
    span0, span1 = _covered_span(g, bbox, side, border_color)
    cr0, cc0, cr1, cc1, axis = _covered_box(g, bbox, side, span0, span1)
    _validate_visible_symmetry(g, bbox, (cr0, cc0, cr1, cc1), axis)

    out = np.zeros((cr1 - cr0 + 1, cc1 - cc0 + 1), dtype=int)
    for r in range(cr0, cr1 + 1):
        for c in range(cc0, cc1 + 1):
            mr, mc = _axis_reflect(r, c, bbox, axis)
            out[r - cr0, c - cc0] = int(g[mr, mc])
    return out


def _is_symmetric(patch: np.ndarray, axis: str) -> bool:
    if axis == "vertical":
        return np.array_equal(patch, np.fliplr(patch))
    return np.array_equal(patch, np.flipud(patch))


def _build_target_patch(
    height: int,
    width: int,
    axis: str,
    border_color: int,
    base_color: int,
    accent_color: int,
) -> np.ndarray:
    patch = np.full((height, width), base_color, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color

    for _ in range(200):
        out = patch.copy()
        accent_cells: set[tuple[int, int]] = set()
        block_count = random.randint(2, 4)
        for _block in range(block_count):
            if axis == "vertical":
                max_half_w = width // 2 - 1
                if max_half_w < 1:
                    continue
                bh = random.randint(1, min(2, height - 2))
                bw = random.randint(1, min(2, max_half_w))
                top = random.randint(1, height - bh - 1)
                left = random.randint(1, max_half_w - bw + 1)
                for dr in range(bh):
                    for dc in range(bw):
                        r = top + dr
                        c = left + dc
                        accent_cells.add((r, c))
                        accent_cells.add((r, width - 1 - c))
            else:
                max_half_h = height // 2 - 1
                if max_half_h < 1:
                    continue
                bh = random.randint(1, min(2, max_half_h))
                bw = random.randint(1, min(2, width - 2))
                top = random.randint(1, max_half_h - bh + 1)
                left = random.randint(1, width - bw - 1)
                for dr in range(bh):
                    for dc in range(bw):
                        r = top + dr
                        c = left + dc
                        accent_cells.add((r, c))
                        accent_cells.add((height - 1 - r, c))

        if len(accent_cells) < 4:
            continue
        for r, c in accent_cells:
            out[r, c] = accent_color

        other_axis = "horizontal" if axis == "vertical" else "vertical"
        if _is_symmetric(out, axis) and not _is_symmetric(out, other_axis):
            return out

    raise RuntimeError("failed to build an asymmetric single-axis target pattern")


def _inflate(mask: np.ndarray, gap: int) -> np.ndarray:
    if gap <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for r, c in zip(*np.where(mask), strict=False):
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    blocked: np.ndarray,
    gap: int,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            ok = True
            for pr in range(ph):
                for pc in range(pw):
                    if patch[pr, pc] == BG:
                        continue
                    if blocked[r0 + pr, c0 + pc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    for pr in range(ph):
        for pc in range(pw):
            if patch[pr, pc] != BG:
                grid[r0 + pr, c0 + pc] = int(patch[pr, pc])

    occupied = np.zeros_like(blocked)
    for pr in range(ph):
        for pc in range(pw):
            if patch[pr, pc] != BG:
                occupied[r0 + pr, c0 + pc] = True
    blocked |= _inflate(occupied, gap)
    return r0, c0


def _solid_patch(height: int, width: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=int)


def _sample_design() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    grid = np.zeros((height, width), dtype=int)

    axis = random.choice(["vertical", "horizontal"])
    if axis == "vertical":
        side = random.choice(["left", "right"])
        th = random.randint(7, 12)
        tw = random.randint(8, 14)
    else:
        side = random.choice(["top", "bottom"])
        th = random.randint(8, 14)
        tw = random.randint(7, 12)

    palette = [1, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(palette)
    border_color, base_color, accent_color, occluder_color = palette[:4]
    target = _build_target_patch(th, tw, axis, border_color, base_color, accent_color)

    if side in {"left", "right"}:
        occ_h = random.randint(2, th - 2)
        start_r = random.randint(1, th - occ_h - 1)
        max_depth = max(1, tw // 2 - 1)
        occ_w = random.randint(1, max_depth)
        ext_w = random.randint(1, 3)
        occ_r0, occ_r1 = start_r, start_r + occ_h - 1
        if side == "left":
            occ_c0, occ_c1 = 0, occ_w - 1
            red_local_c = -1
            occ_patch_c0 = -ext_w
        else:
            occ_c0, occ_c1 = tw - occ_w, tw - 1
            red_local_c = tw
            occ_patch_c0 = tw - occ_w
        red_local_r = random.choice([occ_r0 - 1, occ_r1 + 1])
        occ_patch_h = occ_h
        occ_patch_w = occ_w + ext_w
        occ_patch = np.full((occ_patch_h, occ_patch_w), occluder_color, dtype=int)
        need_left = ext_w if side == "left" else 0
        need_right = ext_w if side == "right" else 0
        need_top = 0
        need_bottom = 0
    else:
        occ_w = random.randint(2, tw - 2)
        start_c = random.randint(1, tw - occ_w - 1)
        max_depth = max(1, th // 2 - 1)
        occ_h = random.randint(1, max_depth)
        ext_h = random.randint(1, 3)
        occ_c0, occ_c1 = start_c, start_c + occ_w - 1
        if side == "top":
            occ_r0, occ_r1 = 0, occ_h - 1
            red_local_r = -1
            occ_patch_r0 = -ext_h
        else:
            occ_r0, occ_r1 = th - occ_h, th - 1
            red_local_r = th
            occ_patch_r0 = th - occ_h
        red_local_c = random.choice([occ_c0 - 1, occ_c1 + 1])
        occ_patch_h = occ_h + ext_h
        occ_patch_w = occ_w
        occ_patch = np.full((occ_patch_h, occ_patch_w), occluder_color, dtype=int)
        need_left = 0
        need_right = 0
        need_top = ext_h if side == "top" else 0
        need_bottom = ext_h if side == "bottom" else 0

    min_r = max(need_top, 1 if red_local_r < 0 else 0)
    max_r = height - th - max(need_bottom, 1 if red_local_r >= th else 0)
    min_c = max(need_left, 1 if red_local_c < 0 else 0)
    max_c = width - tw - max(need_right, 1 if red_local_c >= tw else 0)
    if min_r > max_r or min_c > max_c:
        raise RuntimeError("target does not fit")

    top = random.randint(min_r, max_r)
    left = random.randint(min_c, max_c)

    visible_target = target.copy()
    visible_target[occ_r0 : occ_r1 + 1, occ_c0 : occ_c1 + 1] = BG
    grid[top : top + th, left : left + tw] = visible_target

    if side == "left":
        occ_top = top + occ_r0
        occ_left = left - ext_w
    elif side == "right":
        occ_top = top + occ_r0
        occ_left = left + tw - occ_w
    elif side == "top":
        occ_top = top - ext_h
        occ_left = left + occ_c0
    else:
        occ_top = top + th - occ_h
        occ_left = left + occ_c0
    grid[occ_top : occ_top + occ_patch.shape[0], occ_left : occ_left + occ_patch.shape[1]] = occ_patch

    red_r = top + red_local_r
    red_c = left + red_local_c
    grid[red_r, red_c] = RED

    blocked = _inflate(grid != BG, 1)
    free_colors = [color for color in palette[4:] if color != RED]
    distractor_count = random.randint(1, min(3, len(free_colors)))
    for color in free_colors[:distractor_count]:
        dh = random.randint(2, 5)
        dw = random.randint(2, 6)
        patch = _solid_patch(dh, dw, color)
        placed = _place_patch(grid, patch, blocked, gap=1)
        if placed is None:
            raise RuntimeError("failed to place distractor")

    expected = target[occ_r0 : occ_r1 + 1, occ_c0 : occ_c1 + 1].copy()
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            grid, expected = _sample_design()
        except RuntimeError:
            continue
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
