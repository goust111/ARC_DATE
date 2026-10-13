from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BLACK = 0
GREY = 5
PANEL_SIZE = 10
MAX_GENERATION_ATTEMPTS = 500
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    offsets = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), offsets


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, _r1, _c1 = _bbox_coords(coords)
    return np.array(sorted((int(r - r0), int(c - c0)) for r, c in coords), dtype=int)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(cells), dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_mask(grid == color):
            comps.append({"color": color, "coords": coords})
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(int(r) in (0, h - 1) or int(c) in (0, w - 1) for r, c in coords)


def _enclosed_holes(left: np.ndarray) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    h, w = left.shape
    for coords in _components_mask(left == BLACK):
        if _touches_border(coords, (h, w)):
            continue
        hole = {(int(r), int(c)) for r, c in coords}
        enclosed = True
        for r, c in hole:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    enclosed = False
                    break
                if (nr, nc) not in hole and int(left[nr, nc]) != GREY:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            holes.append(coords)
    return holes


def _mirror_coord(cell: tuple[int, int], axis_type: str, k: int) -> tuple[int, int]:
    r, c = cell
    if axis_type == "v":
        return r, int(k - c)
    return int(k - r), c


def _paired_subset(local_coords: np.ndarray, axis_type: str, k: int) -> np.ndarray:
    pts = {(int(r), int(c)) for r, c in local_coords}
    kept = [cell for cell in pts if _mirror_coord(cell, axis_type, k) in pts]
    if not kept:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(kept), dtype=int)


def _denoise_object(coords: np.ndarray) -> dict:
    local = _normalize_coords(coords)
    raw_key = _shape_key(local)
    h = int(local[:, 0].max()) + 1
    w = int(local[:, 1].max()) + 1
    candidates: list[dict] = []

    for k in range(0, 2 * (w - 1) + 1):
        kept = _paired_subset(local, "v", k)
        if len(kept) > 0:
            candidates.append(
                {
                    "axis_type": "v",
                    "k": k,
                    "score": int(len(kept)),
                    "clean_coords": kept,
                    "clean_key": _shape_key(kept),
                }
            )
    for k in range(0, 2 * (h - 1) + 1):
        kept = _paired_subset(local, "h", k)
        if len(kept) > 0:
            candidates.append(
                {
                    "axis_type": "h",
                    "k": k,
                    "score": int(len(kept)),
                    "clean_coords": kept,
                    "clean_key": _shape_key(kept),
                }
            )

    assert candidates, "object must admit a symmetry explanation"
    best_score = max(c["score"] for c in candidates)
    best = [c for c in candidates if c["score"] == best_score]
    clean_keys = {c["clean_key"] for c in best}
    assert len(clean_keys) == 1, "best denoised shape must be unique"
    chosen = best[0]
    return {
        "raw_key": raw_key,
        "clean_key": chosen["clean_key"],
        "clean_coords": chosen["clean_coords"],
        "axis_type": chosen["axis_type"],
        "axis_k": chosen["k"],
        "best_score": best_score,
        "n_best": len(best),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each left-side hole with the color of the matching right-side object after symmetry denoising.

    Input structure:
    - The grid is split into two equal panels side by side.
    - The left panel contains a connected grey template with 2-3 enclosed black hole-regions.
    - The right panel contains the same number of monochromatic 4-connected objects on black.
    - Each right-side object is nearly symmetric horizontally or vertically, except for 1-2 same-color noise pixels.

    Rule:
    - Find every enclosed black hole in the left panel and normalize each hole mask to its bbox.
    - For each right-side object, find the symmetry axis that best explains the object and delete pixels that lack a mirrored partner across that axis.
    - Normalize the denoised object mask to its bbox, match it to the unique left hole with the same normalized shape, and fill that hole with the object's color.
    - Return only the completed left panel.

    Key insight:
    - The right objects do not match the holes until their asymmetric noise is removed, so symmetry-based denoising is the prerequisite for shape matching and hole filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] % 2 == 0
    mid = g.shape[1] // 2
    left = g[:, :mid]
    right = g[:, mid:]

    holes = _enclosed_holes(left)
    assert 2 <= len(holes) <= 3
    hole_to_coords: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], np.ndarray] = {}
    for coords in holes:
        key = _shape_key(coords)
        assert key not in hole_to_coords, "hole shapes must be unique"
        hole_to_coords[key] = coords

    shape_to_color: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {}
    for obj in _color_components(right, bg=BLACK):
        info = _denoise_object(obj["coords"])
        assert info["raw_key"] != info["clean_key"], "object must contain removable noise"
        key = info["clean_key"]
        assert key not in shape_to_color, "denoised object shapes must be unique"
        shape_to_color[key] = int(obj["color"])

    assert len(shape_to_color) == len(hole_to_coords)
    out = left.copy()
    for key, coords in hole_to_coords.items():
        assert key in shape_to_color, "each hole must have a unique denoised match"
        out[coords[:, 0], coords[:, 1]] = shape_to_color[key]
    return out


def _is_connected_mask(mask: np.ndarray) -> bool:
    return len(_components_mask(mask)) == 1


@lru_cache(None)
def _shape_catalog() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    vertical: list[np.ndarray] = []
    horizontal: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()

    for h in range(2, 5):
        for w in range(2, 5):
            n = h * w
            for bits in range(1 << n):
                area = int(bits.bit_count())
                if area < 3 or area > 6:
                    continue
                mask = np.array([(bits >> i) & 1 for i in range(n)], dtype=bool).reshape(h, w)
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if not _is_connected_mask(mask):
                    continue
                if area == h * w:
                    continue
                vsym = bool(np.array_equal(mask, np.fliplr(mask)))
                hsym = bool(np.array_equal(mask, np.flipud(mask)))
                if vsym == hsym:
                    continue
                key = _shape_key(np.argwhere(mask))
                if key in seen:
                    continue
                seen.add(key)
                if vsym:
                    vertical.append(mask.astype(bool))
                else:
                    horizontal.append(mask.astype(bool))
    assert len(vertical) >= 3 and len(horizontal) >= 3
    return tuple(vertical), tuple(horizontal)


def _mask_with_offset(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _coords_clear(
    occupied: np.ndarray,
    coords: list[tuple[int, int]],
    *,
    border_margin: int,
    gap8: bool,
) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (border_margin <= r < h - border_margin and border_margin <= c < w - border_margin):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not gap8 and abs(dr) + abs(dc) == 2:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _place_masks(
    masks: list[np.ndarray],
    panel_h: int,
    panel_w: int,
    *,
    border_margin: int,
    gap8: bool,
) -> list[tuple[int, int]] | None:
    order = list(range(len(masks)))
    random.shuffle(order)
    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        anchors = [
            (r0, c0)
            for r0 in range(border_margin, panel_h - border_margin - h + 1)
            for c0 in range(border_margin, panel_w - border_margin - w + 1)
        ]
        random.shuffle(anchors)
        placed = False
        for r0, c0 in anchors:
            coords = _mask_with_offset(mask, r0, c0)
            if not _coords_clear(occupied, coords, border_margin=border_margin, gap8=gap8):
                continue
            for r, c in coords:
                occupied[r, c] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None]


def _noise_candidates(mask: np.ndarray, axis_type: str, k: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if not any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in DIR4):
                continue
            mr, mc = _mirror_coord((r, c), axis_type, k)
            if (mr, mc) == (r, c):
                continue
            if 0 <= mr < h and 0 <= mc < w and mask[mr, mc]:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _make_noisy_patch(clean_mask: np.ndarray) -> np.ndarray | None:
    axis_type = "v" if np.array_equal(clean_mask, np.fliplr(clean_mask)) else "h"
    h, w = clean_mask.shape
    canvas = np.zeros((h + 2, w + 2), dtype=bool)
    canvas[1 : 1 + h, 1 : 1 + w] = clean_mask
    clean_key = _shape_key(np.argwhere(clean_mask))
    true_k = int((w - 1) + 2) if axis_type == "v" else int((h - 1) + 2)

    for _ in range(80):
        noisy = canvas.copy()
        candidate_pool = _noise_candidates(canvas, axis_type, true_k)
        noise_target = random.randint(1, 2)
        chosen: list[tuple[int, int]] = []
        forbidden: set[tuple[int, int]] = set()
        for cell in candidate_pool:
            if cell in forbidden:
                continue
            chosen.append(cell)
            mr, mc = _mirror_coord(cell, axis_type, true_k)
            forbidden.add(cell)
            forbidden.add((mr, mc))
            if len(chosen) == noise_target:
                break
        if len(chosen) != noise_target:
            continue

        for r, c in chosen:
            noisy[r, c] = True
        patch = _crop_mask(noisy)
        try:
            info = _denoise_object(np.argwhere(patch))
        except AssertionError:
            continue
        if info["clean_key"] != clean_key:
            continue
        if info["raw_key"] == clean_key:
            continue
        if info["n_best"] != 1:
            continue
        return patch
    return None


def _choose_shapes(n_shapes: int) -> list[np.ndarray]:
    vertical, horizontal = _shape_catalog()
    if n_shapes == 2:
        pattern = random.choice([(1, 1), (2, 0), (0, 2)])
    else:
        pattern = random.choice([(1, 2), (2, 1)])
    v_count, h_count = pattern
    chosen_v = random.sample(list(vertical), v_count)
    chosen_h = random.sample(list(horizontal), h_count)
    shapes = [m.copy() for m in chosen_v + chosen_h]
    random.shuffle(shapes)
    return shapes


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    n_shapes = random.choice([2, 3])
    hole_masks = _choose_shapes(n_shapes)
    hole_keys = {_shape_key(np.argwhere(mask)) for mask in hole_masks}
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n_shapes)

    left_input = np.full((PANEL_SIZE, PANEL_SIZE), GREY, dtype=int)
    output = left_input.copy()

    hole_placements = _place_masks(hole_masks, PANEL_SIZE, PANEL_SIZE, border_margin=1, gap8=True)
    if hole_placements is None:
        return None
    for mask, color, (r0, c0) in zip(hole_masks, colors, hole_placements):
        rr, cc = np.where(mask)
        left_input[r0 + rr, c0 + cc] = BLACK
        output[r0 + rr, c0 + cc] = color

    noisy_masks: list[np.ndarray] = []
    for mask in hole_masks:
        noisy = _make_noisy_patch(mask)
        if noisy is None:
            return None
        if _shape_key(np.argwhere(noisy)) in hole_keys:
            return None
        noisy_masks.append(noisy)

    right_panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    object_placements = _place_masks(noisy_masks, PANEL_SIZE, PANEL_SIZE, border_margin=0, gap8=True)
    if object_placements is None:
        return None
    for noisy_mask, color, (r0, c0) in zip(noisy_masks, colors, object_placements):
        rr, cc = np.where(noisy_mask)
        right_panel[r0 + rr, c0 + cc] = color

    if len(_color_components(right_panel, bg=BLACK)) != n_shapes:
        return None
    inp = np.concatenate([left_input, right_panel], axis=1)
    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        inp, out = built
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid symmetry-denoising match example")
