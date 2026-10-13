from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, bbox_h, bbox_w, find_components, get_bg, paste


BG = Color.BLACK
PALETTE = tuple(color for color in Color.NOT_BLACK if color != BG)
NEIGHBOR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
)
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 120


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(np.asarray(mask, dtype=bool))
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(mask, dtype=bool)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_mask(component: dict) -> np.ndarray:
    coords = np.asarray(component["coords"], dtype=int)
    box = tuple(map(int, component["bbox"]))
    mask = np.zeros((bbox_h(box), bbox_w(box)), dtype=bool)
    rel = coords - np.array([[box[0], box[1]]], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _crop_mask(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for reflected in (base, np.fliplr(base)):
        for k in range(4):
            variant = np.rot90(reflected, k).copy()
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "expected a black background"

    components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    assert len(components) == 8, "expected exactly eight monochromatic sprites"

    base_mask = _component_mask(components[0])
    variants = _dihedral_variants(base_mask)
    assert len(variants) == 8, "the mother sprite must have eight distinct orientations"
    key_to_variant = {_mask_key(variant): idx for idx, variant in enumerate(variants)}

    orientation_counts = [0] * len(variants)
    orientation_colors: list[list[int]] = [[] for _ in variants]
    for component in components:
        color = int(component["color"])
        mask = _component_mask(component)
        key = _mask_key(mask)
        assert key in key_to_variant, "every object must match the shared orientation orbit"
        idx = key_to_variant[key]
        orientation_counts[idx] += 1
        orientation_colors[idx].append(color)

    missing = [idx for idx, count in enumerate(orientation_counts) if count == 0]
    repeated = [idx for idx, count in enumerate(orientation_counts) if count == 2]
    singles = [idx for idx, count in enumerate(orientation_counts) if count == 1]
    assert len(missing) == 1, "expected one absent orientation"
    assert len(repeated) == 1, "expected one duplicated orientation"
    assert len(singles) == 6, "expected the remaining six orientations exactly once"

    color_counts = Counter(int(component["color"]) for component in components)
    repeated_colors = [color for color, count in color_counts.items() if count == 2]
    assert len(repeated_colors) == 1, "expected one shared color for the duplicated orientation"
    repeated_color = int(repeated_colors[0])
    assert orientation_colors[repeated[0]] == [repeated_color, repeated_color], "duplicate orientation must provide the output color"
    assert all(count == 1 for color, count in color_counts.items() if color != repeated_color), "all other colors must be unique"

    return {
        "bg": bg,
        "variants": variants,
        "missing_idx": missing[0],
        "repeated_idx": repeated[0],
        "repeated_color": repeated_color,
        "orientation_counts": tuple(orientation_counts),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enumerate the sprite's full rotation-reflection orbit and draw the missing orientation in the duplicate orientation's color.

    Input structure:
    - The input is a black background containing eight non-touching monochromatic sprite objects.
    - Every sprite is one rotation or reflection of the same asymmetric mother shape.
    - Exactly seven of the eight dihedral orientations appear, and one of the seen orientations appears twice.

    Rule:
    - Normalize any one sprite to its tight bounding-box mask and generate all eight rotation-reflection variants.
    - Match every input sprite against that orientation orbit and count how many times each normalized variant occurs.
    - Output the unique missing variant as a tight sprite patch, colored with the shared color of the duplicated orientation.

    Key insight:
    - The repeated orientation is a decoy for shape, but it uniquely identifies the output color; the answer shape is the orbit member that never appears.
    """
    info = _analyze_scene(np.asarray(grid, dtype=int))
    missing_mask = info["variants"][info["missing_idx"]]
    out = np.full(missing_mask.shape, info["bg"], dtype=int)
    out[missing_mask] = int(info["repeated_color"])
    return out


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        area_low = max(4, int(np.ceil(0.35 * h * w)))
        area_high = min(h * w - 1, int(np.floor(0.7 * h * w)))
        if area_low > area_high:
            continue
        target = int(rng.integers(area_low, area_high + 1))

        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(h)), int(rng.integers(w)))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = tuple(frontier)[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        box = bbox(mask)
        if box is None or bbox_h(box) != h or bbox_w(box) != w:
            continue
        variants = _dihedral_variants(mask)
        if len(variants) != 8:
            continue
        return mask

    raise RuntimeError("failed to generate an asymmetric connected mother sprite")


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBOR8:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_scene(
    rng: np.random.Generator,
    masks: list[np.ndarray],
    colors: list[int],
) -> np.ndarray | None:
    max_h = max(int(mask.shape[0]) for mask in masks)
    max_w = max(int(mask.shape[1]) for mask in masks)

    for _ in range(LAYOUT_ATTEMPTS):
        height = int(rng.integers(max(16, max_h + 4), 31))
        width = int(rng.integers(max(16, max_w + 4), 31))
        grid = np.full((height, width), BG, dtype=int)
        blocked = np.zeros((height, width), dtype=bool)

        order = list(range(len(masks)))
        rng.shuffle(order)
        order.sort(key=lambda idx: int(masks[idx].sum()), reverse=True)

        ok = True
        for idx in order:
            mask = masks[idx]
            coords = np.argwhere(mask)
            h, w = map(int, mask.shape)
            candidates: list[tuple[int, int]] = []
            row_start = 0 if rng.random() < 0.35 else 1
            col_start = 0 if rng.random() < 0.35 else 1
            row_end = height - h + 1 if rng.random() < 0.35 else height - h
            col_end = width - w + 1 if rng.random() < 0.35 else width - w
            if row_end <= row_start or col_end <= col_start:
                row_start, col_start = 0, 0
                row_end, col_end = height - h + 1, width - w + 1
            for r0 in range(row_start, row_end):
                for c0 in range(col_start, col_end):
                    placed = coords + np.array([[r0, c0]], dtype=int)
                    if not blocked[placed[:, 0], placed[:, 1]].any():
                        candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            r0, c0 = candidates[int(rng.integers(len(candidates)))]
            patch = _mask_to_patch(mask, colors[idx])
            paste(grid, patch, r0, c0, transparent=BG)
            _mark_blocked(blocked, coords + np.array([[r0, c0]], dtype=int))

        if ok:
            return grid

    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    mother = _random_connected_mask(rng)
    variants = _dihedral_variants(mother)
    missing_idx = int(rng.integers(8))
    repeated_idx = int(rng.integers(7))
    if repeated_idx >= missing_idx:
        repeated_idx += 1

    present_indices = [idx for idx in range(8) if idx != missing_idx]
    present_indices.append(repeated_idx)
    rng.shuffle(present_indices)

    sampled_colors = list(map(int, rng.choice(PALETTE, size=7, replace=False)))
    repeated_color = sampled_colors[0]
    singleton_colors = sampled_colors[1:]
    color_map: dict[int, int] = {}
    color_map[repeated_idx] = repeated_color
    singleton_iter = iter(singleton_colors)
    for idx in range(8):
        if idx != repeated_idx and idx != missing_idx:
            color_map[idx] = int(next(singleton_iter))

    masks = [variants[idx] for idx in present_indices]
    colors = [color_map[idx] for idx in present_indices]
    grid = _place_scene(rng, masks, colors)
    if grid is None:
        return None

    info = _analyze_scene(grid)
    if _mask_key(info["variants"][info["missing_idx"]]) != _mask_key(variants[missing_idx]):
        return None
    if _mask_key(info["variants"][info["repeated_idx"]]) != _mask_key(variants[repeated_idx]):
        return None
    if info["repeated_color"] != repeated_color:
        return None
    expected = np.full(variants[missing_idx].shape, BG, dtype=int)
    expected[variants[missing_idx]] = repeated_color
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-orientation ARC input")
