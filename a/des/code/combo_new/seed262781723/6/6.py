from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
STAMP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _canonical_rot_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
    return min(variants)


def _legend_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 4 or ww < 4:
            continue
        if r1 != h - 1 and c1 != w - 1:
            continue
        local = np.zeros((hh, ww), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        frame = np.ones((hh, ww), dtype=bool)
        frame[1:-1, 1:-1] = False
        if not np.array_equal(local, frame):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not np.any((inner != BG) & (inner != GREY)):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one legend frame"
    return candidates[0]


def _stamp_mapping(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> dict[tuple[int, int, bytes], int]:
    r0, c0, r1, c1 = legend_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mapping: dict[tuple[int, int, bytes], int] = {}
    for color in np.unique(inner):
        color = int(color)
        if color in (BG, GREY):
            continue
        comps = _components(inner == color)
        assert len(comps) == 1, "each legend color must form one stamp"
        key = _canonical_rot_key(_tight_mask(comps[0]))
        assert key not in mapping, "legend shapes must be unique up to rotation"
        mapping[key] = color
    assert mapping, "legend must contain at least one stamp"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each object's black hole to a legend stamp shape and color the hole with that stamp's color.

    Input structure:
    - The background is black, and the main region contains several separate grey rectangular objects.
    - Each grey object encloses one black hole whose tight shape matches exactly one colored stamp from the legend, up to rotation.
    - A single grey rectangular frame on the right edge or bottom edge contains the colored legend stamps on a black interior.

    Rule:
    - Find the grey framed legend area and read every colored stamp inside it as a shape-to-color entry.
    - For every enclosed black hole inside each non-legend grey object, match its tight shape against the legend under 0/90/180/270-degree rotation.
    - Fill the hole cells with the matched stamp color, and clear the entire legend area to black while leaving the grey objects unchanged.

    Key insight:
    - The holes are negative-space copies of the legend stamps, so solving requires shape matching on enclosed black regions rather than on the grey objects themselves.
    """
    g = np.asarray(grid, dtype=int)
    legend_bbox = _legend_frame_bbox(g)
    mapping = _stamp_mapping(g, legend_bbox)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG

    for comp in _components(g == GREY):
        bbox = _bbox(comp)
        if bbox == legend_bbox:
            continue
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == GREY) | (patch == BG)), "grey objects should only contain grey and black"

        for hole in _components(patch == BG):
            if (
                np.any(hole[:, 0] == 0)
                or np.any(hole[:, 1] == 0)
                or np.any(hole[:, 0] == patch.shape[0] - 1)
                or np.any(hole[:, 1] == patch.shape[1] - 1)
            ):
                continue
            key = _canonical_rot_key(_tight_mask(hole))
            assert key in mapping, "each hole must match a legend stamp"
            color = mapping[key]
            out[r0 + hole[:, 0], c0 + hole[:, 1]] = color

    return out


def _shape_library() -> list[np.ndarray]:
    shapes = [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    ]
    keys = [_canonical_rot_key(shape) for shape in shapes]
    assert len(set(keys)) == len(shapes), "shape library must be unique up to rotation"
    return shapes


SHAPE_LIBRARY = _shape_library()


def _build_object_patch(hole_mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    hh, hw = hole_mask.shape
    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    patch = np.full((hh + top + bottom, hw + left + right), GREY, dtype=int)
    region = patch[top : top + hh, left : left + hw]
    region[hole_mask] = BG
    hole_cells = np.argwhere(hole_mask) + np.array([top, left], dtype=int)
    return patch, hole_cells


def _build_legend_patch(
    display_specs: list[dict[str, np.ndarray | int]],
    side: str,
    rng: random.Random,
) -> np.ndarray:
    masks = [spec["display"] for spec in display_specs]
    heights = [int(mask.shape[0]) for mask in masks]
    widths = [int(mask.shape[1]) for mask in masks]
    gap = 1

    if side == "right":
        total_h = sum(heights) + gap * (len(masks) - 1)
        inner_h = total_h + rng.randint(2, 4)
        inner_w = max(widths) + rng.randint(2, 4)
        legend = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
        legend[0, :] = GREY
        legend[-1, :] = GREY
        legend[:, 0] = GREY
        legend[:, -1] = GREY
        inner = legend[1:-1, 1:-1]
        start_r = rng.randint(1, inner_h - total_h - 1)
        cur_r = start_r
        for spec, mask in zip(display_specs, masks):
            mh, mw = mask.shape
            c0 = rng.randint(1, inner_w - mw - 1)
            inner[cur_r : cur_r + mh, c0 : c0 + mw][mask] = int(spec["color"])
            cur_r += mh + gap
        return legend

    total_w = sum(widths) + gap * (len(masks) - 1)
    inner_h = max(heights) + rng.randint(2, 4)
    inner_w = total_w + rng.randint(2, 4)
    legend = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY
    inner = legend[1:-1, 1:-1]
    start_c = rng.randint(1, inner_w - total_w - 1)
    cur_c = start_c
    for spec, mask in zip(display_specs, masks):
        mh, mw = mask.shape
        r0 = rng.randint(1, inner_h - mh - 1)
        inner[r0 : r0 + mh, cur_c : cur_c + mw][mask] = int(spec["color"])
        cur_c += mw + gap
    return legend


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sample_grid_shape(
    side: str,
    legend_shape: tuple[int, int],
    object_patches: list[np.ndarray],
    rng: random.Random,
) -> tuple[int, int] | None:
    legend_h, legend_w = legend_shape
    max_obj_h = max(int(p.shape[0]) for p in object_patches)
    max_obj_w = max(int(p.shape[1]) for p in object_patches)
    need = sum((int(p.shape[0]) + 1) * (int(p.shape[1]) + 1) for p in object_patches) + 12

    for _ in range(60):
        if side == "right":
            h_min = max(legend_h, max_obj_h + 2, 10)
            if h_min > 30:
                return None
            h = rng.randint(h_min, 30)
            main_h = max(3, h - 2)
            main_w_min = max(max_obj_w + 2, _ceil_div(need, main_h))
            w_min = legend_w + main_w_min
            if w_min > 30:
                continue
            w = rng.randint(max(w_min, 12), 30)
            return h, w
        w_min = max(legend_w, max_obj_w + 2, 10)
        if w_min > 30:
            return None
        w = rng.randint(w_min, 30)
        main_w = max(3, w - 2)
        main_h_min = max(max_obj_h + 2, _ceil_div(need, main_w))
        h_min = legend_h + main_h_min
        if h_min > 30:
            continue
        h = rng.randint(max(h_min, 12), 30)
        return h, w
    return None


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    mask[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = rng.choice(candidates)
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    _reserve(occupied, (r0, c0, r0 + ph - 1, c0 + pw - 1))
    return r0, c0


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(("right", "bottom"))
    n_stamps = rng.randint(2, 4)
    n_objects = rng.randint(1, n_stamps)

    shapes = rng.sample(SHAPE_LIBRARY, n_stamps)
    colors = rng.sample(STAMP_COLORS, n_stamps)
    specs = []
    for shape, color in zip(shapes, colors):
        specs.append(
            {
                "base": shape,
                "display": np.rot90(shape, rng.randint(0, 3)),
                "color": int(color),
                "key": _canonical_rot_key(shape),
            }
        )
    rng.shuffle(specs)

    object_specs = rng.sample(specs, n_objects)
    object_patches: list[dict[str, np.ndarray | int]] = []
    for spec in object_specs:
        hole_mask = np.rot90(np.asarray(spec["base"], dtype=bool), rng.randint(0, 3))
        patch, hole_cells = _build_object_patch(hole_mask, rng)
        object_patches.append(
            {
                "patch": patch,
                "hole_cells": hole_cells,
                "color": int(spec["color"]),
            }
        )

    legend_patch = _build_legend_patch(specs, side, rng)
    grid_shape = _sample_grid_shape(
        side,
        (int(legend_patch.shape[0]), int(legend_patch.shape[1])),
        [np.asarray(info["patch"], dtype=int) for info in object_patches],
        rng,
    )
    if grid_shape is None:
        return None
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    output = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    lh, lw = legend_patch.shape
    if side == "right":
        lr0 = rng.randint(0, h - lh)
        lc0 = w - lw
    else:
        lr0 = h - lh
        lc0 = rng.randint(0, w - lw)
    grid[lr0 : lr0 + lh, lc0 : lc0 + lw] = legend_patch
    _reserve(occupied, (lr0, lc0, lr0 + lh - 1, lc0 + lw - 1))

    output[:] = grid
    output[lr0 : lr0 + lh, lc0 : lc0 + lw] = BG

    object_patches.sort(key=lambda item: -int(np.count_nonzero(item["patch"] == GREY)))
    for info in object_patches:
        patch = np.asarray(info["patch"], dtype=int)
        placed = _place_patch(grid, occupied, patch, rng)
        if placed is None:
            return None
        r0, c0 = placed
        output[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        hole_cells = np.asarray(info["hole_cells"], dtype=int)
        output[r0 + hole_cells[:, 0], c0 + hole_cells[:, 1]] = int(info["color"])

    try:
        if not np.array_equal(main(grid), output):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
