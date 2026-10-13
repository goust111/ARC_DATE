from __future__ import annotations

import random

import numpy as np


BG_COLOR = 5
SHAPE_COLOR = 1
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _find_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))

    return components


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    norm = _normalize_mask(mask)
    return norm.shape, tuple(int(v) for v in norm.reshape(-1))


def _enclosed_cells(barrier: np.ndarray) -> np.ndarray:
    barrier = np.asarray(barrier, dtype=bool)
    h, w = map(int, barrier.shape)
    outside = np.zeros_like(barrier, dtype=bool)
    stack: list[tuple[int, int]] = []

    def seed(r: int, c: int) -> None:
        if not barrier[r, c] and not outside[r, c]:
            outside[r, c] = True
            stack.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not barrier[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return (~barrier) & (~outside)


def _describe_component(coords: np.ndarray) -> dict:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    enclosed = _enclosed_cells(mask)
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "enclosed": enclosed,
        "filled": mask | enclosed,
    }


def _find_shape_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        components = _find_components(grid == color, connectivity=8)
        if len(components) < 2:
            continue
        hole_count = 0
        solid_count = 0
        for coords in components:
            desc = _describe_component(coords)
            if bool(desc["enclosed"].any()):
                hole_count += 1
            elif len(coords) > 1:
                solid_count += 1
        if hole_count > 0 and solid_count > 0:
            candidates.append(color)

    assert len(candidates) == 1, "expected one color shared by hollow and solid objects"
    return int(candidates[0])


def _scale_factor(small_mask: np.ndarray, big_filled: np.ndarray) -> int | None:
    sh, sw = map(int, small_mask.shape)
    bh, bw = map(int, big_filled.shape)
    if bh % sh != 0 or bw % sw != 0:
        return None
    scale_r = bh // sh
    scale_c = bw // sw
    if scale_r != scale_c or scale_r < 2:
        return None
    expanded = np.kron(
        np.asarray(small_mask, dtype=np.uint8),
        np.ones((scale_r, scale_r), dtype=np.uint8),
    ).astype(bool)
    return scale_r if np.array_equal(expanded, big_filled) else None


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    transforms: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            norm = _normalize_mask(variant)
            if not bool(norm[0, 0]):
                continue
            sig = _mask_signature(norm)
            if sig not in seen:
                seen.add(sig)
                transforms.append(norm)
    return transforms


def _outline_from_filled(filled: np.ndarray) -> np.ndarray:
    filled = np.asarray(filled, dtype=bool)
    h, w = map(int, filled.shape)
    outline = np.zeros_like(filled, dtype=bool)
    for r, c in np.argwhere(filled):
        rr = int(r)
        cc = int(c)
        if rr in (0, h - 1) or cc in (0, w - 1):
            outline[rr, cc] = True
            continue
        if any(not filled[rr + dr, cc + dc] for dr, dc in DIR4):
            outline[rr, cc] = True
    return outline


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each small solid shape with the geometrically similar larger outline,
    stamp the small shape inside the larger outline at every colored marker, and
    erase the original small shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(np.bincount(g.reshape(-1)).argmax())
    shape_color = _find_shape_color(g, bg)

    components = [_describe_component(coords) for coords in _find_components(g == shape_color, connectivity=8)]
    bigs = [comp for comp in components if bool(comp["enclosed"].any())]
    smalls = [comp for comp in components if not bool(comp["enclosed"].any())]
    assert bigs and smalls

    small_masks: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    for comp in smalls:
        small_masks.setdefault(_mask_signature(comp["mask"]), comp["mask"])

    for big in bigs:
        matches = [mask for mask in small_masks.values() if _scale_factor(mask, big["filled"]) is not None]
        assert len(matches) == 1, "expected a unique small-shape match for each large outline"
        big["stamp_mask"] = matches[0]

    out = g.copy()
    for comp in smalls:
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = bg

    h, w = map(int, g.shape)
    for big in bigs:
        r0, c0, _r1, _c1 = big["bbox"]
        canvas = big["enclosed"]
        markers: list[tuple[int, int, int]] = []
        for lr, lc in np.argwhere(canvas):
            rr = r0 + int(lr)
            cc = c0 + int(lc)
            color = int(g[rr, cc])
            if color not in (bg, shape_color):
                markers.append((rr, cc, color))
        assert markers

        stamp_cells = np.argwhere(big["stamp_mask"])
        for mr, mc, color in markers:
            for dr, dc in stamp_cells:
                rr = mr + int(dr)
                cc = mc + int(dc)
                lr = rr - r0
                lc = cc - c0
                if 0 <= rr < h and 0 <= cc < w and 0 <= lr < canvas.shape[0] and 0 <= lc < canvas.shape[1]:
                    if canvas[lr, lc] and out[rr, cc] == bg:
                        out[rr, cc] = color

    return out


BASE_SHAPES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
]

SHAPE_POOL: list[np.ndarray] = []
_POOL_SEEN: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
for _base in BASE_SHAPES:
    for _variant in _all_transforms(_base):
        _sig = _mask_signature(_variant)
        if _sig not in _POOL_SEEN:
            _POOL_SEEN.add(_sig)
            SHAPE_POOL.append(_variant)


GRID_MIN = 24
GRID_MAX = 36
PAIR_COUNTS = (1, 2, 3)
SCALE_OPTIONS = (2, 3)
MAX_MARKERS = 4
OBJECT_PAD = 1
GEN_ATTEMPTS = 400


def _anchor_candidates(small_mask: np.ndarray, canvas: np.ndarray) -> list[dict]:
    stamp_cells = [tuple(map(int, pt)) for pt in np.argwhere(small_mask)]
    sh, sw = map(int, small_mask.shape)
    ch, cw = map(int, canvas.shape)
    candidates: list[dict] = []

    for r in range(ch - sh + 1):
        for c in range(cw - sw + 1):
            if not canvas[r, c]:
                continue
            cells = frozenset((r + dr, c + dc) for dr, dc in stamp_cells)
            if all(canvas[rr, cc] for rr, cc in cells):
                candidates.append({"anchor": (r, c), "cells": cells})

    return candidates


def _candidates_conflict(a: dict, b: dict) -> bool:
    if a["cells"] & b["cells"]:
        return True
    ar, ac = a["anchor"]
    br, bc = b["anchor"]
    return max(abs(ar - br), abs(ac - bc)) <= 1


def _choose_anchor_set(candidates: list[dict], target: int) -> list[dict] | None:
    ordered = candidates[:]
    random.shuffle(ordered)
    chosen: list[dict] = []

    def backtrack(start: int) -> bool:
        if len(chosen) == target:
            return True
        if len(ordered) - start < target - len(chosen):
            return False
        for idx in range(start, len(ordered)):
            cand = ordered[idx]
            if any(_candidates_conflict(cand, prev) for prev in chosen):
                continue
            chosen.append(cand)
            if backtrack(idx + 1):
                return True
            chosen.pop()
        return False

    if backtrack(0):
        return chosen[:]
    return None


def _build_pair_spec(small_mask: np.ndarray) -> dict | None:
    scale = random.choice(SCALE_OPTIONS)
    filled = np.kron(
        np.asarray(small_mask, dtype=np.uint8),
        np.ones((scale, scale), dtype=np.uint8),
    ).astype(bool)
    outline = _outline_from_filled(filled)
    canvas = filled & (~outline)
    candidates = _anchor_candidates(small_mask, canvas)
    if not candidates:
        return None

    feasible: list[tuple[int, list[dict]]] = []
    for marker_count in range(1, min(MAX_MARKERS, len(candidates)) + 1):
        anchors = _choose_anchor_set(candidates, marker_count)
        if anchors is not None:
            feasible.append((marker_count, anchors))
    if not feasible:
        return None

    marker_count, anchors = random.choice(feasible)
    marker_colors = [random.choice(MARKER_COLORS) for _ in range(marker_count)]
    markers = [
        (int(anchor["anchor"][0]), int(anchor["anchor"][1]), int(color))
        for anchor, color in zip(anchors, marker_colors)
    ]

    return {
        "small_mask": np.asarray(small_mask, dtype=bool),
        "scale": scale,
        "filled": filled,
        "outline": outline,
        "canvas": canvas,
        "markers": markers,
    }


def _pairings_are_unambiguous(pair_specs: list[dict]) -> bool:
    small_masks = [spec["small_mask"] for spec in pair_specs]
    for spec in pair_specs:
        matches = sum(_scale_factor(mask, spec["filled"]) is not None for mask in small_masks)
        if matches != 1:
            return False
    return True


def _box_free(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_box(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _place_objects(shape: tuple[int, int], pair_specs: list[dict]) -> dict | None:
    occupied = np.zeros(shape, dtype=bool)
    objects: list[tuple[str, int, int, int, int]] = []
    for idx, spec in enumerate(pair_specs):
        bh, bw = map(int, spec["outline"].shape)
        sh, sw = map(int, spec["small_mask"].shape)
        objects.append(("big", idx, bh * bw, bh, bw))
        objects.append(("small", idx, sh * sw, sh, sw))
    random.shuffle(objects)
    objects.sort(key=lambda item: item[2], reverse=True)

    positions: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, _area, height, width in objects:
        spots = [
            (r, c)
            for r in range(shape[0] - height + 1)
            for c in range(shape[1] - width + 1)
            if _box_free(occupied, r, c, height, width, OBJECT_PAD)
        ]
        if not spots:
            return None
        top, left = random.choice(spots)
        positions[(kind, idx)] = (int(top), int(left))
        _mark_box(occupied, top, left, height, width, OBJECT_PAD)

    return positions


def _render_input(shape: tuple[int, int], pair_specs: list[dict], positions: dict) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)

    for idx, spec in enumerate(pair_specs):
        top, left = positions[("big", idx)]
        h, w = spec["outline"].shape
        patch = grid[top : top + h, left : left + w]
        patch[spec["outline"]] = SHAPE_COLOR
        for mr, mc, color in spec["markers"]:
            grid[top + mr, left + mc] = int(color)

    for idx, spec in enumerate(pair_specs):
        top, left = positions[("small", idx)]
        h, w = spec["small_mask"].shape
        patch = grid[top : top + h, left : left + w]
        patch[spec["small_mask"]] = SHAPE_COLOR

    return grid


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNTS)
    indices = random.sample(range(len(SHAPE_POOL)), pair_count)
    small_masks = [np.asarray(SHAPE_POOL[idx], dtype=bool) for idx in indices]
    pair_specs = []
    for mask in small_masks:
        spec = _build_pair_spec(mask)
        if spec is None:
            return None
        pair_specs.append(spec)

    if not _pairings_are_unambiguous(pair_specs):
        return None

    total_area = 0
    min_h = 0
    min_w = 0
    for spec in pair_specs:
        bh, bw = map(int, spec["outline"].shape)
        sh, sw = map(int, spec["small_mask"].shape)
        total_area += (bh + 2 * OBJECT_PAD + 1) * (bw + 2 * OBJECT_PAD + 1)
        total_area += (sh + 2 * OBJECT_PAD + 1) * (sw + 2 * OBJECT_PAD + 1)
        min_h = max(min_h, bh + 2, sh + 2)
        min_w = max(min_w, bw + 2, sw + 2)

    side_floor = max(GRID_MIN, int(np.ceil(total_area ** 0.5)))
    if side_floor > GRID_MAX:
        return None
    height = random.randint(max(min_h, side_floor), GRID_MAX)
    width = random.randint(max(min_w, side_floor), GRID_MAX)

    positions = _place_objects((height, width), pair_specs)
    if positions is None:
        return None

    grid = _render_input((height, width), pair_specs, positions)
    if int(np.bincount(grid.reshape(-1)).argmax()) != BG_COLOR:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
