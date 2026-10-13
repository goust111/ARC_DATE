from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
FRAME_THICKNESS = 2
MAX_GENERATION_RETRIES = 400

BOX_COUNT_CHOICES = (2, 3)
GRID_SIDE_RANGE = {
    2: (22, 28),
    3: (26, 30),
}
BOX_H_RANGE = (8, 11)
BOX_W_RANGE = (8, 12)
BOX_PADDING = 2
KEY_PADDING = 1
PLACEMENT_RETRIES = 250

SHAPE_LIBRARY = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
)
SIDES = ("top", "bottom", "left", "right")


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_array(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            parts.append(coords)
    return parts


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    cid = 0
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components(grid == color):
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "id": cid,
                    "color": int(color),
                    "coords": coords,
                    "coords_arr": arr,
                    "bbox": _bbox_from_array(arr),
                    "area": int(len(coords)),
                }
            )
            cid += 1
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert len(rows) and len(cols)
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = _bbox_from_array(arr)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
    return mask


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            cropped = _crop_mask(candidate.astype(bool))
            key = (int(cropped.shape[0]), int(cropped.shape[1]), cropped.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cropped)
    return variants


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (int(v.shape[0]), int(v.shape[1]), v.tobytes())
        for v in _dihedral_variants(mask.astype(bool))
    )


def _ring_mask(h: int, w: int, t: int) -> np.ndarray:
    ring = np.zeros((h, w), dtype=bool)
    ring[:t, :] = True
    ring[h - t :, :] = True
    ring[:, :t] = True
    ring[:, w - t :] = True
    return ring


def _parse_lock_box(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    local = np.zeros((h, w), dtype=bool)
    coords = comp["coords_arr"]
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    for t in range(1, min(3, min(h, w) // 2) + 1):
        if h < 2 * t + 3 or w < 2 * t + 3:
            continue
        ring = _ring_mask(h, w, t)
        if np.any(local & ~ring):
            continue

        gap_mask = ring & ~local
        gap_area = int(gap_mask.sum())
        if gap_area < 2 or gap_area > 8:
            continue
        if np.any(patch[gap_mask] != bg):
            continue

        gap_parts = _components(gap_mask)
        if len(gap_parts) != 1:
            continue
        gap_local = gap_parts[0]
        if not any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in gap_local):
            continue

        interior = ~ring
        inner_fg = (patch != bg) & interior
        if int(inner_fg.sum()) != 1:
            continue
        dot_r, dot_c = map(int, np.argwhere(inner_fg)[0])
        if (dot_r, dot_c) not in {
            (t, t),
            (t, w - t - 1),
            (h - t - 1, t),
            (h - t - 1, w - t - 1),
        }:
            continue
        dot_color = int(patch[dot_r, dot_c])
        if dot_color in {bg, color}:
            continue

        gap_global = [(r0 + rr, c0 + cc) for rr, cc in gap_local]
        gap_shape = _mask_from_coords(gap_local)
        return {
            "component_id": int(comp["id"]),
            "border_color": color,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "dot_cell": (int(r0 + dot_r), int(c0 + dot_c)),
            "dot_color": dot_color,
            "gap_cells": gap_global,
            "shape_key": _shape_key(gap_shape),
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lock-box gap with the key sprite of the same shape, then recolor
    the inserted key using the box's hidden interior dot.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _find_monochrome_components(g, bg)

    boxes: list[dict] = []
    box_ids: set[int] = set()
    for comp in comps:
        parsed = _parse_lock_box(g, comp, bg)
        if parsed is not None:
            boxes.append(parsed)
            box_ids.add(int(comp["id"]))
    assert boxes, "expected at least one lock box"

    dot_cells = {box["dot_cell"] for box in boxes}
    keys_by_shape: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for comp in comps:
        if int(comp["id"]) in box_ids:
            continue
        if int(comp["area"]) == 1 and tuple(comp["coords"][0]) in dot_cells:
            continue
        keys_by_shape[_shape_key(_mask_from_coords(comp["coords"]))].append(comp)

    for key_list in keys_by_shape.values():
        key_list.sort(key=lambda comp: (*comp["bbox"], int(comp["color"])))

    boxes.sort(key=lambda box: (*box["bbox"], box["dot_color"]))
    assignments: list[tuple[dict, dict]] = []
    for box in boxes:
        key_list = keys_by_shape.get(box["shape_key"], [])
        assert key_list, "every gap needs a matching key"
        assignments.append((box, key_list.pop(0)))

    assert all(not leftover for leftover in keys_by_shape.values()), "unexpected extra key components"

    out = g.copy()
    for box, key_comp in assignments:
        for r, c in key_comp["coords"]:
            out[int(r), int(c)] = bg
        dr, dc = box["dot_cell"]
        out[int(dr), int(dc)] = bg
        for r, c in box["gap_cells"]:
            out[int(r), int(c)] = int(box["dot_color"])
    return out


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    H, W = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _side_variants(shape: np.ndarray, side: str) -> list[np.ndarray]:
    if side in {"top", "bottom"}:
        return [v for v in _dihedral_variants(shape) if int(v.shape[0]) <= FRAME_THICKNESS]
    return [v for v in _dihedral_variants(shape) if int(v.shape[1]) <= FRAME_THICKNESS]


def _choose_gap_origin(side: str, box_h: int, box_w: int, gap_h: int, gap_w: int) -> tuple[int, int] | None:
    if side in {"top", "bottom"}:
        min_c = FRAME_THICKNESS
        max_c = box_w - gap_w - FRAME_THICKNESS
        if max_c < min_c:
            return None
        c0 = random.randint(min_c, max_c)
        r0 = 0 if side == "top" else box_h - gap_h
        return int(r0), int(c0)
    min_r = FRAME_THICKNESS
    max_r = box_h - gap_h - FRAME_THICKNESS
    if max_r < min_r:
        return None
    r0 = random.randint(min_r, max_r)
    c0 = 0 if side == "left" else box_w - gap_w
    return int(r0), int(c0)


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _make_box_patch(spec: dict) -> np.ndarray:
    h = int(spec["box_h"])
    w = int(spec["box_w"])
    patch = np.full((h, w), BG, dtype=int)
    patch[_ring_mask(h, w, FRAME_THICKNESS)] = int(spec["border_color"])
    gr, gc = spec["gap_origin"]
    gap_mask = spec["gap_mask"]
    gap_view = patch[gr : gr + gap_mask.shape[0], gc : gc + gap_mask.shape[1]]
    gap_view[gap_mask] = BG
    dr, dc = spec["dot_local"]
    patch[dr, dc] = int(spec["dot_color"])
    return patch


def _apply_expected_transform(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        kr, kc = spec["key_top_left"]
        key_mask = spec["key_mask"]
        key_view = out[kr : kr + key_mask.shape[0], kc : kc + key_mask.shape[1]]
        key_view[key_mask] = BG

        br, bc = spec["box_top_left"]
        dr, dc = spec["dot_local"]
        out[br + dr, bc + dc] = BG

        gr, gc = spec["gap_origin"]
        gap_mask = spec["gap_mask"]
        gap_view = out[
            br + gr : br + gr + gap_mask.shape[0],
            bc + gc : bc + gc + gap_mask.shape[1],
        ]
        gap_view[gap_mask] = int(spec["dot_color"])
    return out


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_boxes = random.choice(BOX_COUNT_CHOICES)
    side_lo, side_hi = GRID_SIDE_RANGE[n_boxes]
    H = random.randint(side_lo, side_hi)
    W = random.randint(side_lo, side_hi)

    colors = list(range(1, 10))
    random.shuffle(colors)
    border_colors = colors[:n_boxes]
    dot_colors = colors[n_boxes : 2 * n_boxes]
    key_colors = colors[2 * n_boxes : 3 * n_boxes]
    base_shapes = random.sample(list(SHAPE_LIBRARY), n_boxes)

    specs: list[dict] = []
    for idx in range(n_boxes):
        side = random.choice(SIDES)
        gap_options = _side_variants(base_shapes[idx], side)
        if not gap_options:
            return None
        gap_mask = random.choice(gap_options)
        gap_h, gap_w = map(int, gap_mask.shape)

        min_h = max(BOX_H_RANGE[0], gap_h + 2 * FRAME_THICKNESS + 1)
        min_w = max(BOX_W_RANGE[0], gap_w + 2 * FRAME_THICKNESS + 1)
        box_h = random.randint(min_h, BOX_H_RANGE[1])
        box_w = random.randint(min_w, BOX_W_RANGE[1])
        gap_origin = _choose_gap_origin(side, box_h, box_w, gap_h, gap_w)
        if gap_origin is None:
            return None

        t = FRAME_THICKNESS
        dot_local = random.choice(
            [
                (t, t),
                (t, box_w - t - 1),
                (box_h - t - 1, t),
                (box_h - t - 1, box_w - t - 1),
            ]
        )
        key_mask = random.choice(_dihedral_variants(base_shapes[idx]))

        specs.append(
            {
                "border_color": int(border_colors[idx]),
                "dot_color": int(dot_colors[idx]),
                "key_color": int(key_colors[idx]),
                "box_h": int(box_h),
                "box_w": int(box_w),
                "gap_mask": gap_mask,
                "gap_origin": gap_origin,
                "dot_local": dot_local,
                "key_mask": key_mask,
                "shape_key": _shape_key(base_shapes[idx]),
            }
        )

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for spec in sorted(specs, key=lambda item: item["box_h"] * item["box_w"], reverse=True):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, H - int(spec["box_h"]))
            left = random.randint(0, W - int(spec["box_w"]))
            if not _can_place(occupied, top, left, int(spec["box_h"]), int(spec["box_w"]), BOX_PADDING):
                continue
            spec["box_top_left"] = (int(top), int(left))
            _mark_occupied(occupied, top, left, int(spec["box_h"]), int(spec["box_w"]), BOX_PADDING)
            placed = True
            break
        if not placed:
            return None

    for spec in sorted(specs, key=lambda item: int(item["key_mask"].sum()), reverse=True):
        key_h, key_w = map(int, spec["key_mask"].shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, H - key_h)
            left = random.randint(0, W - key_w)
            if not _can_place(occupied, top, left, key_h, key_w, KEY_PADDING):
                continue
            spec["key_top_left"] = (int(top), int(left))
            _mark_occupied(occupied, top, left, key_h, key_w, KEY_PADDING)
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        br, bc = spec["box_top_left"]
        _paste_patch(grid, _make_box_patch(spec), br, bc)

    for spec in specs:
        kr, kc = spec["key_top_left"]
        key_patch = np.full(spec["key_mask"].shape, BG, dtype=int)
        key_patch[spec["key_mask"]] = int(spec["key_color"])
        _paste_patch(grid, key_patch, kr, kc)

    expected = _apply_expected_transform(grid, specs)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        sampled = _sample_case()
        if sampled is None:
            continue
        inp, expected = sampled
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid lock-and-key puzzle")

