from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
IDENTITY = (0, False)
TRANSFORMS = [(k, flip) for k in range(4) for flip in (False, True)]
SYMMETRY_FAMILIES = {
    "rot180": (2, False),
    "mirror_v": (0, True),
    "mirror_h": (2, True),
}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    H, W = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            values = [int(g[rr, cc]) for rr, cc in coords]
            counts = Counter(values)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                    "colors": set(counts),
                    "color_counts": counts,
                }
            )
    return comps


def _indicator_pos(patch: np.ndarray, indicator_color: int) -> tuple[int, int] | None:
    coords = np.argwhere(patch == indicator_color)
    if len(coords) != 1:
        return None
    return tuple(map(int, coords[0]))


def _matching_transforms(
    source_patch: np.ndarray,
    target_patch: np.ndarray,
    indicator_color: int,
    bg: int,
    *,
    require_indicator: bool,
) -> list[tuple[tuple[int, bool], np.ndarray]]:
    target_mask = target_patch != bg
    target_indicator = _indicator_pos(target_patch, indicator_color)
    if require_indicator and target_indicator is None:
        return []

    matches: list[tuple[tuple[int, bool], np.ndarray]] = []
    for transform in TRANSFORMS:
        transformed = _apply_transform(source_patch, *transform)
        if transformed.shape != target_patch.shape:
            continue
        if not np.array_equal(transformed != bg, target_mask):
            continue
        if require_indicator and _indicator_pos(transformed, indicator_color) != target_indicator:
            continue
        matches.append((transform, transformed))
    return matches


def _analyze_indicator_candidate(grid: np.ndarray, indicator_color: int, bg: int) -> dict | None:
    components = _find_components(grid, bg)
    if len(components) < 3:
        return None
    if not all(component["color_counts"].get(indicator_color, 0) == 1 for component in components):
        return None

    source_candidates = [component for component in components if len(component["colors"]) >= 3]
    if len(source_candidates) != 1:
        return None
    source = source_candidates[0]
    targets = [component for component in components if component is not source]
    if len(targets) < 2:
        return None
    if not all(len(component["colors"]) == 2 for component in targets):
        return None

    source_patch = _crop_bbox(grid, source["bbox"])
    if _indicator_pos(source_patch, indicator_color) is None:
        return None

    analyzed_targets = []
    for component in targets:
        target_patch = _crop_bbox(grid, component["bbox"])
        mask_matches = _matching_transforms(
            source_patch, target_patch, indicator_color, bg, require_indicator=False
        )
        if len(mask_matches) < 2:
            return None
        exact_matches = _matching_transforms(
            source_patch, target_patch, indicator_color, bg, require_indicator=True
        )
        if len(exact_matches) != 1:
            return None

        analyzed_targets.append(
            {
                "bbox": component["bbox"],
                "coords": component["coords"],
                "transform": exact_matches[0][0],
                "paint_patch": exact_matches[0][1],
                "mask_transforms": [match[0] for match in mask_matches],
            }
        )

    return {
        "bg": bg,
        "indicator_color": indicator_color,
        "source": source,
        "source_patch": source_patch,
        "targets": analyzed_targets,
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    candidates = []
    for color in sorted(set(map(int, g.flat)) - {bg}):
        analyzed = _analyze_indicator_candidate(g, int(color), bg)
        if analyzed is not None:
            candidates.append(analyzed)

    assert len(candidates) == 1, "expected a unique indicator color / source-target decomposition"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the multicolor source sprite onto each symmetric placeholder using the indicator pixel to choose its orientation.

    Input structure:
    - The dominant background surrounds one connected multicolor source sprite whose mask contains exactly one indicator-color pixel.
    - Several separate placeholder objects have the same mask up to rotation or reflection, and each placeholder is monochrome except for one indicator-color pixel.
    - The shared mask has a nontrivial dihedral symmetry, so shape matching alone leaves at least two valid orientations.

    Rule:
    - Infer the unique source sprite and the shared indicator color.
    - For each placeholder, enumerate the source-mask rotations/reflections that fit the placeholder mask, then keep only the transform whose indicator pixel lands on the placeholder indicator.
    - Replace that placeholder by the fully colored transformed source sprite, and finally erase the original source sprite to background.

    Key insight:
    - Because the mask is symmetric, the object outline underdetermines orientation; only the lone indicator pixel breaks that symmetry and selects the correct stamp transform.
    """
    g = np.asarray(grid, dtype=int)
    analyzed = _analyze_grid(g)
    bg = analyzed["bg"]

    out = g.copy()
    for r, c in analyzed["source"]["coords"]:
        out[r, c] = bg

    for target in analyzed["targets"]:
        for r, c in target["coords"]:
            out[r, c] = bg
        r0, c0, _r1, _c1 = target["bbox"]
        patch = target["paint_patch"]
        mask = patch != bg
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]

    return out


def _mask_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)))


def _mask_automorphisms(mask: np.ndarray) -> list[tuple[int, bool]]:
    autos = []
    as_int = mask.astype(int)
    for transform in TRANSFORMS:
        transformed = _apply_transform(as_int, *transform)
        if transformed.shape == as_int.shape and np.array_equal(transformed, as_int):
            autos.append(transform)
    return autos


def _indicator_choices(mask: np.ndarray, symmetry: tuple[int, bool]) -> list[tuple[int, int]]:
    choices = []
    for coord in map(tuple, np.argwhere(mask)):
        singleton = np.zeros(mask.shape, dtype=int)
        singleton[coord] = 1
        moved = np.argwhere(_apply_transform(singleton, *symmetry) == 1)
        if len(moved) == 1 and tuple(map(int, moved[0])) != coord:
            choices.append(tuple(map(int, coord)))
    return choices


def _build_mask_catalog() -> dict[str, list[np.ndarray]]:
    catalog = {name: [] for name in SYMMETRY_FAMILIES}
    dims = [(3, 3), (3, 4), (4, 3), (4, 4)]
    for h, w in dims:
        cells = list(range(h * w))
        for count in range(max(5, max(h, w)), min(9, h * w - 1) + 1):
            for combo in combinations(cells, count):
                mask = np.zeros((h, w), dtype=bool)
                for idx in combo:
                    mask[idx // w, idx % w] = True
                if not _mask_has_full_bbox(mask) or not _is_connected_mask(mask):
                    continue
                autos = _mask_automorphisms(mask)
                if len(autos) != 2 or IDENTITY not in autos:
                    continue
                symmetry = [t for t in autos if t != IDENTITY][0]
                family = None
                for name, fam_symmetry in SYMMETRY_FAMILIES.items():
                    if symmetry == fam_symmetry:
                        family = name
                        break
                if family is None:
                    continue
                if not _indicator_choices(mask, symmetry):
                    continue
                catalog[family].append(mask.astype(int))
    assert all(catalog.values()), "mask catalog unexpectedly empty"
    return catalog


MASK_CATALOG = _build_mask_catalog()


def _patch_is_asymmetric(patch: np.ndarray) -> bool:
    for transform in TRANSFORMS:
        if transform == IDENTITY:
            continue
        transformed = _apply_transform(patch, *transform)
        if transformed.shape == patch.shape and np.array_equal(transformed, patch):
            return False
    return True


def _make_source_patch(
    mask: np.ndarray,
    indicator_color: int,
    body_colors: list[int],
    indicator_coord: tuple[int, int],
) -> np.ndarray | None:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    others = [coord for coord in coords if coord != indicator_coord]
    if len(others) < len(body_colors):
        return None

    for _ in range(200):
        patch = np.zeros(mask.shape, dtype=int)
        patch[indicator_coord] = indicator_color
        shuffled = others[:]
        random.shuffle(shuffled)
        for coord, color in zip(shuffled, body_colors):
            patch[coord] = color
        for coord in shuffled[len(body_colors) :]:
            patch[coord] = random.choice(body_colors)
        if len(set(int(patch[r, c]) for r, c in others)) < 2:
            continue
        if _patch_is_asymmetric(patch):
            return patch
    return None


def _make_placeholder_patch(
    source_patch: np.ndarray,
    transform: tuple[int, bool],
    indicator_color: int,
    placeholder_color: int,
) -> np.ndarray:
    transformed = _apply_transform(source_patch, *transform)
    patch = np.zeros(transformed.shape, dtype=int)
    mask = transformed != BG
    patch[mask] = placeholder_color
    ir, ic = _indicator_pos(transformed, indicator_color)
    patch[ir, ic] = indicator_color
    return patch


def _choose_target_transforms(n_targets: int) -> list[tuple[int, bool]]:
    for _ in range(200):
        transforms = [random.choice(TRANSFORMS) for _ in range(n_targets)]
        if all(transform == IDENTITY for transform in transforms):
            continue
        if len(set(transforms)) == 1:
            continue
        if n_targets >= 3:
            has_reflection = any(flip for _k, flip in transforms)
            has_rotation = any((k != 0) and (not flip) for k, flip in transforms)
            if not (has_reflection and has_rotation):
                continue
        return transforms
    raise RuntimeError("failed to sample diverse target transforms")


def _can_place(
    rects: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    gap: int,
) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for ar0, ac0, ar1, ac1 in rects:
        if not (r1 + gap < ar0 or ar1 + gap < r0 or c1 + gap < ac0 or ac1 + gap < c0):
            return False
    return True


def _place_patches(patches: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]] | None:
    min_h = max(max(patch.shape[0] for patch in patches) + 2, 12)
    min_w = max(max(patch.shape[1] for patch in patches) + 2, 12)

    for _ in range(200):
        H = random.randint(min_h, 22)
        W = random.randint(min_w, 22)
        order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
        rects: list[tuple[int, int, int, int]] = []
        positions: list[tuple[int, int] | None] = [None] * len(patches)

        ok = True
        for idx in order:
            h, w = patches[idx].shape
            candidates = [
                (r, c)
                for r in range(H - h + 1)
                for c in range(W - w + 1)
                if _can_place(rects, r, c, h, w, gap=1)
            ]
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            positions[idx] = (r0, c0)
            rects.append((r0, c0, r0 + h - 1, c0 + w - 1))

        if ok:
            return H, W, [pos for pos in positions if pos is not None]
    return None


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    dst[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _semantic_checks(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    try:
        analyzed = _analyze_grid(inp)
        solved = main(inp)
    except Exception:
        return False

    if not np.array_equal(solved, expected_out):
        return False

    source_patch = analyzed["source_patch"]
    indicator_color = analyzed["indicator_color"]
    transforms = [target["transform"] for target in analyzed["targets"]]
    if all(transform == IDENTITY for transform in transforms):
        return False

    for target in analyzed["targets"]:
        if len(target["mask_transforms"]) < 2:
            return False
        chosen = target["transform"]
        chosen_patch = target["paint_patch"]
        for alt in target["mask_transforms"]:
            if alt == chosen:
                continue
            alt_patch = _apply_transform(source_patch, *alt)
            if np.array_equal(alt_patch, chosen_patch):
                return False
        if _indicator_pos(chosen_patch, indicator_color) is None:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    family = random.choice(sorted(MASK_CATALOG))
    mask = random.choice(MASK_CATALOG[family]).astype(int)
    symmetry = SYMMETRY_FAMILIES[family]
    indicator_coord = random.choice(_indicator_choices(mask.astype(bool), symmetry))

    n_targets = random.randint(2, 4)
    max_body_colors = min(4, int(mask.sum()) - 1, 8 - n_targets)
    if max_body_colors < 2:
        return None
    n_body_colors = random.randint(2, max_body_colors)

    palette = random.sample(list(range(1, 10)), 1 + n_body_colors + n_targets)
    indicator_color = palette[0]
    body_colors = palette[1 : 1 + n_body_colors]
    placeholder_colors = palette[1 + n_body_colors :]

    source_patch = _make_source_patch(mask, indicator_color, body_colors, indicator_coord)
    if source_patch is None:
        return None

    target_transforms = _choose_target_transforms(n_targets)
    placeholder_patches = [
        _make_placeholder_patch(source_patch, transform, indicator_color, placeholder_color)
        for transform, placeholder_color in zip(target_transforms, placeholder_colors)
    ]

    placement = _place_patches([source_patch] + placeholder_patches)
    if placement is None:
        return None
    H, W, positions = placement

    inp = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)

    source_pos = positions[0]
    _paste(inp, source_patch, *source_pos)

    for patch, pos in zip(placeholder_patches, positions[1:]):
        _paste(inp, patch, *pos)

    out.fill(BG)
    for transform, pos in zip(target_transforms, positions[1:]):
        _paste(out, _apply_transform(source_patch, *transform), *pos)

    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamped-indicator puzzle input")
