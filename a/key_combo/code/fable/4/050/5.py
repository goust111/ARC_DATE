from __future__ import annotations

import random

import numpy as np


BG = 0
MOVE_PAD = 1
GENERATE_ATTEMPTS = 400
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(int(v) for v in np.asarray(grid, dtype=int).ravel()))
    if exclude is None:
        return colors
    banned = set(int(v) for v in exclude)
    return [color for color in colors if color not in banned]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _trim_bool(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None, "mask must be non-empty"
    return crop_bbox(mask.astype(bool), box)


def _is_vertical_sym(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _is_horizontal_sym(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)))


def _is_double_sym(mask: np.ndarray) -> bool:
    return _is_vertical_sym(mask) and _is_horizontal_sym(mask)


def _render_mask(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[np.asarray(mask, dtype=bool)] = int(color)
    return out


def _cropped_color_mask(grid: np.ndarray, color: int, bg: int) -> np.ndarray:
    mask = np.asarray(grid == int(color), dtype=bool)
    box = bbox(mask)
    assert box is not None, "color must be present"
    crop = np.full((box[2] - box[0] + 1, box[3] - box[1] + 1), int(bg), dtype=int)
    crop[mask[box[0] : box[2] + 1, box[1] : box[3] + 1]] = int(color)
    return crop


def _repair_variants(mask: np.ndarray, *, need_vertical: bool, need_horizontal: bool) -> list[np.ndarray]:
    base = np.pad(np.asarray(mask, dtype=bool), MOVE_PAD)
    occupied = [tuple(map(int, coord)) for coord in np.argwhere(base)]
    empty = [tuple(map(int, coord)) for coord in np.argwhere(~base)]
    found: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for sr, sc in occupied:
        base[sr, sc] = False
        for dr, dc in empty:
            if dr == sr and dc == sc:
                continue
            base[dr, dc] = True
            trimmed = _trim_bool(base)
            if need_vertical and not _is_vertical_sym(trimmed):
                base[dr, dc] = False
                continue
            if need_horizontal and not _is_horizontal_sym(trimmed):
                base[dr, dc] = False
                continue
            key = (trimmed.shape, trimmed.tobytes())
            found[key] = trimmed.copy()
            base[dr, dc] = False
        base[sr, sc] = True

    return list(found.values())


def _target_repairs_for_grid(grid: np.ndarray) -> list[tuple[int, list[np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    candidates: list[tuple[int, list[np.ndarray]]] = []

    for color in colors_present(g, exclude={bg}):
        crop = _cropped_color_mask(g, color, bg)
        mask = crop == int(color)
        if _is_double_sym(mask):
            continue
        repairs = _repair_variants(mask, need_vertical=True, need_horizontal=True)
        if repairs:
            candidates.append((int(color), repairs))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique color mask whose one-pixel move repairs both horizontal and vertical symmetry at once.

    Input structure:
    - The grid has a dominant background and several non-background color masks.
    - Some masks are already axis-symmetric, while several others are only one move away from a single-axis repair.
    - Exactly one color mask has a single-pixel defect whose move can restore both left-right and up-down symmetry.

    Rule:
    - For each non-background color, crop its full mask to the bounding box and try every one-pixel move inside a small padded search window.
    - Ignore masks that are already doubly symmetric, and ignore masks whose one-move repairs never achieve both axes simultaneously.
    - Select the unique mask with a doubly symmetric one-move repair and return that repaired mask cropped to its bounding box.

    Key insight:
    - The target defect is a shared break point for both axes, so one moved pixel fixes the entire color mask in both directions at once.
    """
    candidates = _target_repairs_for_grid(np.asarray(grid, dtype=int))
    assert len(candidates) == 1, f"expected one double-axis-repairable mask, got {len(candidates)}"
    color, repairs = candidates[0]
    assert len(repairs) == 1, f"expected a unique repaired crop for color {color}, got {len(repairs)}"
    return _render_mask(repairs[0], color)


def _orbit4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = {(int(r), int(c)), (int(r), int(w - 1 - c)), (int(h - 1 - r), int(c)), (int(h - 1 - r), int(w - 1 - c))}
    return sorted(cells)


def _orbit_touches(mask: np.ndarray, orbit: list[tuple[int, int]]) -> bool:
    h, w = mask.shape
    for r, c in orbit:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                return True
    return False


def _build_double_base() -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h = random.choice([5, 7, 9])
    w = random.choice([5, 7, 9])
    cr, cc = h // 2, w // 2
    mask = np.zeros((h, w), dtype=bool)
    mask[cr, :] = True
    mask[:, cc] = True

    candidates = [(dr, dc) for dr in range(1, min(3, cr) + 1) for dc in range(1, min(3, cc) + 1)]
    random.shuffle(candidates)
    use = candidates[: random.randint(1, max(1, min(3, len(candidates))))]
    added: list[list[tuple[int, int]]] = []
    for dr, dc in use:
        orbit = _orbit4(cr - dr, cc - dc, h, w)
        for rr, cc2 in orbit:
            mask[rr, cc2] = True
        added.append(orbit)

    return mask, added


def _build_vertical_base() -> np.ndarray:
    h = random.randint(5, 8)
    w = random.choice([5, 7, 9])
    cc = w // 2
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, cc] = True

    bottom_len = random.choice([k for k in range(1, w, 2)])
    half = bottom_len // 2
    mask[h - 1, cc - half : cc + half + 1] = True

    extras = random.randint(0, 3)
    for _ in range(extras):
        r_choices = [r for r in range(1, h - 1) if r != h // 2]
        if not r_choices:
            break
        r = random.choice(r_choices)
        c = random.randint(0, cc - 1)
        mask[r, c] = True
        mask[r, w - 1 - c] = True

    assert _is_vertical_sym(mask)
    assert not _is_horizontal_sym(mask)
    return mask


def _build_horizontal_base() -> np.ndarray:
    return _build_vertical_base().T.copy()


def _strict_symmetric_mask() -> np.ndarray:
    kind = random.choice(["double", "vertical", "horizontal"])
    if kind == "double":
        return _build_double_base()[0]
    if kind == "vertical":
        return _build_vertical_base()
    return _build_horizontal_base()


def _target_broken_and_output() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        base, orbits = _build_double_base()
        trimmed_base = _trim_bool(base)
        orbit_cells = [cell for orbit in orbits for cell in orbit]
        corners = [(0, 0), (0, base.shape[1] - 1), (base.shape[0] - 1, 0), (base.shape[0] - 1, base.shape[1] - 1)]
        edge_strays = [(1, 0), (0, 1), (1, base.shape[1] - 1), (0, base.shape[1] - 2)]
        stray_options = [cell for cell in corners + edge_strays if not base[cell]]
        random.shuffle(orbit_cells)
        random.shuffle(stray_options)

        for missing in orbit_cells:
            for stray in stray_options:
                broken = base.copy()
                broken[missing] = False
                broken[stray] = True
                if _is_double_sym(broken):
                    continue
                repairs = _repair_variants(broken, need_vertical=True, need_horizontal=True)
                if len(repairs) == 1 and np.array_equal(repairs[0], trimmed_base):
                    return broken, trimmed_base
    raise RuntimeError("failed to construct double-axis target")


def _can_place(occupied: np.ndarray, h: int, w: int, r0: int, c0: int, padding: int) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    r1 = max(0, r0 - padding)
    c1 = max(0, c0 - padding)
    r2 = min(H, r0 + h + padding)
    c2 = min(W, c0 + w + padding)
    return not bool(np.any(occupied[r1:r2, c1:c2]))


def _place_patches(patches: list[tuple[np.ndarray, int]], H: int, W: int, padding: int) -> np.ndarray | None:
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    order = sorted(patches, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)

    for patch, color in order:
        h, w = patch.shape
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1) if _can_place(occupied, h, w, r, c, padding)]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        colored = _render_mask(patch, color)
        paste(grid, colored, r0, c0, transparent=BG)
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return grid


def _build_single_axis_decoy(axis: str) -> np.ndarray:
    for _ in range(120):
        base = _build_vertical_base() if axis == "vertical" else _build_horizontal_base()
        h, w = base.shape
        if axis == "vertical":
            missing_options = [(0, c) for c in range(w // 2) if base[0, c]]
            stray_options = [(h - 2, 0), (h - 2, 1), (h // 2 - 1, 0), (1, 0)]
            need_vertical, need_horizontal = True, False
        else:
            missing_options = [(r, 0) for r in range(h // 2) if base[r, 0]]
            stray_options = [(0, w - 2), (1, w - 2), (0, 1), (0, w // 2 - 1)]
            need_vertical, need_horizontal = False, True

        missing_options = [cell for cell in missing_options if base[cell]]
        stray_options = [cell for cell in stray_options if 0 <= cell[0] < h and 0 <= cell[1] < w and not base[cell]]
        random.shuffle(missing_options)
        random.shuffle(stray_options)

        for missing in missing_options:
            for stray in stray_options:
                broken = base.copy()
                broken[missing] = False
                broken[stray] = True
                axis_repairs = _repair_variants(broken, need_vertical=need_vertical, need_horizontal=need_horizontal)
                double_repairs = _repair_variants(broken, need_vertical=True, need_horizontal=True)
                if double_repairs:
                    continue
                if axis_repairs and any(np.array_equal(repair, _trim_bool(base)) for repair in axis_repairs):
                    return broken
    raise RuntimeError(f"failed to build {axis} decoy")


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        target_inp, target_out_mask = _target_broken_and_output()
        perfect = _strict_symmetric_mask()
        vertical_decoy = _build_single_axis_decoy("vertical")
        horizontal_decoy = _build_single_axis_decoy("horizontal")

        masks = [target_inp, perfect, vertical_decoy, horizontal_decoy]
        if random.random() < 0.5:
            masks.append(_strict_symmetric_mask())

        colors = random.sample(PALETTE, len(masks))
        target_color = colors[0]
        zipped = list(zip(masks, colors))
        random.shuffle(zipped)

        max_h = max(mask.shape[0] for mask in masks)
        max_w = max(mask.shape[1] for mask in masks)
        H = random.randint(max(20, max_h + 6), 30)
        W = random.randint(max(20, max_w + 6), 30)
        grid = _place_patches(zipped, H, W, padding=2)
        if grid is None:
            continue

        expected = _render_mask(target_out_mask, target_color)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
