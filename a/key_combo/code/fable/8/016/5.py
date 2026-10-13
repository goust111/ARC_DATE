from __future__ import annotations

import random

import numpy as np


_RNG = np.random.default_rng()
_COLORS = tuple(range(10))
_MAX_GENERATION_ATTEMPTS = 400


def _majority_color(arr: np.ndarray) -> int:
    flat = np.asarray(arr, dtype=int).ravel()
    assert flat.size > 0
    return int(np.bincount(flat, minlength=int(flat.max()) + 1).argmax())


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_foreground(panel: np.ndarray, bg: int) -> np.ndarray:
    b = _bbox(panel != bg)
    assert b is not None, "patch panel must contain foreground"
    r0, c0, r1, c1 = b
    return np.array(panel[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _separator_row(grid: np.ndarray) -> int:
    h = int(grid.shape[0])
    candidates: list[int] = []
    for r in range(1, h - 1):
        if not bool(np.all(grid[r] == grid[r, 0])):
            continue
        color = int(grid[r, 0])
        if not bool(np.any(np.delete(grid, r, axis=0) == color)):
            candidates.append(r)
    assert len(candidates) == 1, "expected one separator row"
    return candidates[0]


def _rigid_transforms(sprite: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for mirrored in (False, True):
        base = np.fliplr(sprite) if mirrored else sprite
        for rot in range(4):
            variant = np.rot90(base, k=rot)
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(np.array(variant, dtype=int))
    return variants


def _best_registration(
    canvas: np.ndarray, patch: np.ndarray, bg: int
) -> tuple[int, int, np.ndarray, int, int]:
    ch, cw = map(int, canvas.shape)
    best_overlap = -1
    best_row = -1
    best_col = -1
    best_pose: np.ndarray | None = None
    best_count = 0

    for pose in _rigid_transforms(patch):
        ph, pw = map(int, pose.shape)
        if ph > ch or pw > cw:
            continue
        fg = np.argwhere(pose != bg)
        assert fg.size > 0
        for r0 in range(ch - ph + 1):
            for c0 in range(cw - pw + 1):
                overlap = 0
                legal = True
                for dr, dc in fg:
                    rr = int(r0 + dr)
                    cc = int(c0 + dc)
                    pv = int(pose[dr, dc])
                    cv = int(canvas[rr, cc])
                    if cv == bg:
                        continue
                    if cv != pv:
                        legal = False
                        break
                    overlap += 1
                if not legal:
                    continue
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_row = int(r0)
                    best_col = int(c0)
                    best_pose = pose
                    best_count = 1
                elif overlap == best_overlap:
                    if (
                        best_pose is None
                        or best_row != int(r0)
                        or best_col != int(c0)
                        or best_pose.shape != pose.shape
                        or not np.array_equal(best_pose, pose)
                    ):
                        best_count += 1

    assert best_pose is not None, "expected at least one legal placement"
    return best_row, best_col, best_pose, int(best_overlap), int(best_count)


def _overlay(canvas: np.ndarray, pose: np.ndarray, r0: int, c0: int, bg: int) -> np.ndarray:
    out = np.array(canvas, copy=True)
    ph, pw = map(int, pose.shape)
    region = out[r0 : r0 + ph, c0 : c0 + pw]
    mask = pose != bg
    conflict = (region != bg) & mask & (region != pose)
    assert not bool(conflict.any())
    region[mask] = pose[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete the separator row, register the lower patch by maximal consistent overlap, and fill the missing canvas area.

    Input structure:
    - A single full-width monochrome separator row splits the grid into an upper canvas panel and a lower patch panel.
    - The upper panel is a larger canvas with some foreground already present and some missing cells left as background.
    - The lower panel contains the same patch content in an unknown rigid pose (rotation and/or mirror), surrounded only by panel background.

    Rule:
    - Remove the separator row and crop the lower panel to its foreground bbox to obtain the patch.
    - Try all rigid patch poses and all translations inside the upper canvas; a placement is legal iff every overlapping foreground color agrees.
    - Among legal placements, choose the unique one with the largest number of overlapping foreground pixels, then transparently overlay the patch to fill the canvas background holes.

    Key insight:
    - The patch is not aligned by panel borders: its pose and position are recovered only by maximizing color-consistent foreground overlap with the existing canvas trace.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    upper = np.array(g[:sep], dtype=int)
    lower = np.array(g[sep + 1 :], dtype=int)
    assert upper.size > 0 and lower.size > 0

    panels = np.concatenate([upper.ravel(), lower.ravel()])
    bg = _majority_color(panels)
    patch = _crop_to_foreground(lower, bg)

    row, col, pose, _overlap, winner_count = _best_registration(upper, patch, bg)
    assert winner_count == 1, "expected a unique best registration"
    return _overlay(upper, pose, row, col, bg)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = map(int, mask.shape)
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if not bool(mask[rr, cc]) or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _make_patch_mask(h: int, w: int) -> np.ndarray | None:
    area_lo = max(6, (h * w) // 3)
    area_hi = min(h * w - 1, (2 * h * w) // 3 + 1)
    if area_lo > area_hi:
        return None

    target = int(_RNG.integers(area_lo, area_hi + 1))
    mask = np.zeros((h, w), dtype=bool)
    start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            frontier.update(_neighbors4(int(r), int(c), h, w))
        frontier = {coord for coord in frontier if not bool(mask[coord])}
        if not frontier:
            return None
        rr, cc = list(frontier)[int(_RNG.integers(len(frontier)))]
        mask[rr, cc] = True

    if not (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    ):
        return None
    if not _connected(mask):
        return None
    return mask


def _make_patch(h: int, w: int, bg: int, sep: int) -> np.ndarray | None:
    available = [color for color in _COLORS if color not in {bg, sep}]
    assert len(available) >= 3

    for _ in range(100):
        mask = _make_patch_mask(h, w)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        color_count = int(_RNG.integers(2, min(5, len(available), len(coords)) + 1))
        palette = list(map(int, _RNG.choice(available, size=color_count, replace=False)))
        patch = np.full((h, w), bg, dtype=int)

        order = _RNG.permutation(len(coords))
        for i, idx in enumerate(order):
            r, c = map(int, coords[int(idx)])
            if i < color_count:
                color = palette[i]
            else:
                color = int(palette[int(_RNG.integers(color_count))])
            patch[r, c] = color

        if len(_rigid_transforms(patch)) == 8:
            return patch
    return None


def _add_context(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    border_cells: list[tuple[int, int, tuple[int, int]]] = []
    ph, pw = map(int, patch.shape)
    for r in range(ph):
        for c in range(pw):
            if int(patch[r, c]) == bg:
                continue
            if r == 0:
                border_cells.append((r, c, (-1, 0)))
            if r == ph - 1:
                border_cells.append((r, c, (1, 0)))
            if c == 0:
                border_cells.append((r, c, (0, -1)))
            if c == pw - 1:
                border_cells.append((r, c, (0, 1)))

    random.shuffle(border_cells)
    target = int(_RNG.integers(1, 4))
    placed = 0
    h, w = map(int, canvas.shape)

    for r, c, (dr, dc) in border_cells:
        length = int(_RNG.integers(1, 4))
        cells: list[tuple[int, int]] = []
        ok = True
        for step in range(1, length + 1):
            rr = int(top + r + dr * step)
            cc = int(left + c + dc * step)
            if not (0 <= rr < h and 0 <= cc < w):
                ok = False
                break
            if left <= cc < left + pw and top <= rr < top + ph:
                ok = False
                break
            if int(canvas[rr, cc]) != bg:
                ok = False
                break
            cells.append((rr, cc))
        if not ok or not cells:
            continue
        for rr, cc in cells:
            canvas[rr, cc] = int(patch[r, c])
        placed += 1
        if placed >= target:
            return


def _place_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    region = canvas[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.integers(0, 10))
    sep = int(_RNG.choice([color for color in _COLORS if color != bg]))

    canvas_h = int(_RNG.integers(10, 19))
    canvas_w = int(_RNG.integers(12, 23))
    patch_h = int(_RNG.integers(4, min(8, canvas_h)))
    patch_w = int(_RNG.integers(4, min(8, canvas_w)))

    patch = _make_patch(patch_h, patch_w, bg, sep)
    if patch is None:
        return None

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    src_top = int(_RNG.integers(1, canvas_h - patch_h + 1))
    src_left = int(_RNG.integers(1, canvas_w - patch_w + 1))
    _place_patch(canvas, patch, src_top, src_left, bg)
    _add_context(canvas, patch, src_top, src_left, bg)
    full_canvas = np.array(canvas, copy=True)

    fg_coords = [tuple(map(int, coord)) for coord in np.argwhere(patch != bg)]
    if len(fg_coords) < 6:
        return None

    keep_min = max(3, len(fg_coords) // 3)
    keep_max = len(fg_coords) - 2
    if keep_min > keep_max:
        return None
    keep_count = int(_RNG.integers(keep_min, keep_max + 1))
    keep_idx = set(map(int, _RNG.choice(len(fg_coords), size=keep_count, replace=False)))
    kept_colors = {int(patch[fg_coords[i]]) for i in keep_idx}
    if len(kept_colors) < 2:
        return None

    holey_canvas = np.array(full_canvas, copy=True)
    for i, (r, c) in enumerate(fg_coords):
        if i in keep_idx:
            continue
        holey_canvas[src_top + r, src_left + c] = bg

    poses = _rigid_transforms(patch)
    pose = poses[int(_RNG.integers(len(poses)))]
    ph, pw = map(int, pose.shape)
    bottom_h = int(ph + _RNG.integers(0, 3))
    patch_row = int(_RNG.integers(0, bottom_h - ph + 1))
    patch_col = int(_RNG.integers(0, canvas_w - pw + 1))
    lower = np.full((bottom_h, canvas_w), bg, dtype=int)
    _place_patch(lower, pose, patch_row, patch_col, bg)

    inp = np.vstack([holey_canvas, np.full((1, canvas_w), sep, dtype=int), lower])
    try:
        row, col, best_pose, overlap, winner_count = _best_registration(
            holey_canvas, _crop_to_foreground(lower, bg), bg
        )
    except AssertionError:
        return None

    if winner_count != 1 or overlap < 2:
        return None

    solved = _overlay(holey_canvas, best_pose, row, col, bg)
    if not np.array_equal(solved, full_canvas):
        return None
    if not np.array_equal(main(inp), full_canvas):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid patch-registration input")
