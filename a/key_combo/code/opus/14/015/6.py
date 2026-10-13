from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
OFFSETS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 400
BOX_PAD = 2


def _neighbors4(r: int, c: int):
    for dr, dc in OFFSETS4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    origin = coords.min(axis=0)
    norm = coords - origin
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    cells = tuple(sorted((int(r), int(c)) for r, c in norm))
    return h, w, cells


def _shape_key_from_mask(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    return _shape_key(np.argwhere(mask))


def _mirror_key(key: tuple[int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    h, w, cells = key
    mirrored = tuple(sorted((int(r), int(w - 1 - c)) for r, c in cells))
    return int(h), int(w), mirrored


def _mask_from_key(key: tuple[int, int, tuple[tuple[int, int], ...]]) -> np.ndarray:
    h, w, cells = key
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _component_touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_holes(crop: np.ndarray, color: int) -> list[np.ndarray] | None:
    if np.any((crop != BG) & (crop != int(color))):
        return None
    obj_mask = crop == int(color)
    if len(_components(obj_mask)) != 1:
        return None
    holes = []
    for comp in _components(crop == BG):
        if not _component_touches_edge(comp, crop.shape):
            holes.append(comp)
    return holes


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _analyze_component(grid: np.ndarray, comp: dict) -> dict | None:
    bbox = comp["bbox"]
    r0, c0, _r1, _c1 = bbox
    crop = _crop_bbox(grid, bbox)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    obj_mask = np.zeros(crop.shape, dtype=bool)
    obj_mask[rel[:, 0], rel[:, 1]] = True
    if not np.array_equal(crop == int(comp["color"]), obj_mask):
        return None
    holes = _extract_holes(crop, int(comp["color"]))
    if holes is None:
        return None
    hole_keys = [_shape_key(hole) for hole in holes]
    info = dict(comp)
    info["crop"] = crop
    info["shape_key"] = _shape_key(rel)
    info["holes"] = holes
    info["hole_keys"] = hole_keys
    return info


def _candidate_match(obj: dict, markers: list[dict], *, mirror_markers: bool) -> dict | None:
    needed = Counter(obj["hole_keys"])
    fill_by_shape = {}
    used_markers = []
    for hole_key, count in needed.items():
        matches = []
        for marker in markers:
            marker_key = _mirror_key(marker["shape_key"]) if mirror_markers else marker["shape_key"]
            if marker_key == hole_key:
                matches.append(marker)
        if len(matches) != count:
            return None
        colors = {int(marker["color"]) for marker in matches}
        if len(colors) != 1:
            return None
        fill_by_shape[hole_key] = next(iter(colors))
        used_markers.extend(matches)
    if len({id(marker) for marker in used_markers}) != len(used_markers):
        return None
    return {"object": obj, "fill_by_shape": fill_by_shape}


def _find_matches(grid: np.ndarray, *, mirror_markers: bool) -> list[dict]:
    comps = _find_monochrome_components(grid, BG)
    objects = []
    markers = []
    for comp in comps:
        info = _analyze_component(grid, comp)
        if info is None:
            continue
        if info["holes"]:
            objects.append(info)
        else:
            markers.append(info)
    matches = []
    for obj in objects:
        match = _candidate_match(obj, markers, mirror_markers=mirror_markers)
        if match is not None:
            matches.append(match)
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the object whose hole shapes match the horizontally mirrored markers, then fill each hole with its marker color.

    Input structure:
    - The grid has a black background, 2-3 large monochrome objects with internal black holes, and 2-3 small monochrome marker shapes placed elsewhere.
    - Each large object is a single 4-connected component, and each marker is a separate 4-connected component.
    - Exactly one large object has all of its hole shapes represented by the markers after the marker shapes are reflected left-to-right.

    Rule:
    - Extract every large object's internal hole components as normalized shapes inside that object's bounding box.
    - Reflect every marker shape horizontally, normalize it, and match those mirrored shapes against object-hole shapes with a complete bijection by shape multiplicity.
    - Select the unique object with a complete mirrored-shape match, crop its bounding box, and fill each hole with the color of the corresponding matched marker shape.

    Key insight:
    - The markers do not match the holes directly: each marker must be mirrored first, and only then does its geometry identify which cropped object's holes should be filled.
    """
    g = np.asarray(grid, dtype=int)
    matches = _find_matches(g, mirror_markers=True)
    assert len(matches) == 1
    match = matches[0]
    obj = match["object"]
    out = obj["crop"].copy()
    for hole in obj["holes"]:
        out[hole[:, 0], hole[:, 1]] = int(match["fill_by_shape"][_shape_key(hole)])
    return out


def _has_internal_hole(mask: np.ndarray) -> bool:
    for comp in _components(~mask):
        if not _component_touches_edge(comp, mask.shape):
            return True
    return False


def _make_shape_catalog() -> list[np.ndarray]:
    catalog: dict[tuple[int, int, tuple[tuple[int, int], ...]], np.ndarray] = {}
    for h in range(2, 5):
        for w in range(2, 5):
            cells = [(r, c) for r in range(h) for c in range(w)]
            for area in range(3, min(6, h * w)):
                for chosen in combinations(cells, area):
                    mask = np.zeros((h, w), dtype=bool)
                    for r, c in chosen:
                        mask[r, c] = True
                    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                        continue
                    if len(_components(mask)) != 1:
                        continue
                    if np.array_equal(mask, np.fliplr(mask)):
                        continue
                    if _has_internal_hole(mask):
                        continue
                    key = _shape_key_from_mask(mask)
                    canonical = min(key, _mirror_key(key))
                    catalog.setdefault(canonical, mask.copy())
    return [catalog[key] for key in sorted(catalog)]


SHAPE_CATALOG = _make_shape_catalog()


def _choose_hole_shapes(count: int) -> list[np.ndarray]:
    assert len(SHAPE_CATALOG) >= count
    shapes = []
    for base in random.sample(SHAPE_CATALOG, count):
        mask = base.copy()
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        shapes.append(mask)
    return shapes


def _object_patch(color: int, hole_masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]] | None:
    min_h = max(6, max(mask.shape[0] for mask in hole_masks) + 3)
    min_w = max(6, max(mask.shape[1] for mask in hole_masks) + 3)
    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(min_h, min(10, min_h + 3))
        w = random.randint(min_w, min(10, min_w + 4))
        occupied = np.zeros((h, w), dtype=bool)
        placements = []
        order = list(range(len(hole_masks)))
        random.shuffle(order)
        ok = True
        for idx in order:
            mask = hole_masks[idx]
            hh, ww = mask.shape
            candidates = [
                (r, c)
                for r in range(1, h - hh)
                for c in range(1, w - ww)
            ]
            random.shuffle(candidates)
            placed = None
            for r0, c0 in candidates:
                cells = np.argwhere(mask)
                fits = True
                for rr, cc in cells:
                    ar = int(r0 + rr)
                    ac = int(c0 + cc)
                    if occupied[ar, ac]:
                        fits = False
                        break
                    for nr, nc in _neighbors4(ar, ac):
                        if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                            fits = False
                            break
                    if not fits:
                        break
                if fits:
                    placed = (r0, c0)
                    break
            if placed is None:
                ok = False
                break
            placements.append((idx, placed[0], placed[1]))
            for rr, cc in np.argwhere(mask):
                occupied[placed[0] + rr, placed[1] + cc] = True
        if not ok:
            continue
        patch = np.full((h, w), int(color), dtype=int)
        hole_coords = [None] * len(hole_masks)
        for idx, r0, c0 in placements:
            mask = hole_masks[idx]
            coords = np.argwhere(mask)
            abs_coords = coords + np.array([r0, c0], dtype=int)
            patch[abs_coords[:, 0], abs_coords[:, 1]] = BG
            hole_coords[idx] = abs_coords
        holes = _extract_holes(patch, int(color))
        if holes is None or len(holes) != len(hole_masks):
            continue
        if Counter(_shape_key(hole) for hole in holes) != Counter(_shape_key(np.argwhere(mask)) for mask in hole_masks):
            continue
        return patch, hole_coords
    return None


def _marker_patch(color: int, hole_mask: np.ndarray) -> np.ndarray:
    mirrored = np.fliplr(hole_mask)
    patch = np.zeros(mirrored.shape, dtype=int)
    patch[mirrored] = int(color)
    return patch


def _boxes_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = BOX_PAD) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_patch(grid: np.ndarray, patch: np.ndarray, placed: list[tuple[int, int, int, int]]) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    coords = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(coords)
    mask = patch != BG
    for r0, c0 in coords:
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if all(_boxes_far(box, other) for other in placed):
            if np.any((grid[r0 : r0 + ph, c0 : c0 + pw] != BG) & mask):
                continue
            view = grid[r0 : r0 + ph, c0 : c0 + pw]
            view[mask] = patch[mask]
            placed.append(box)
            return True
    return False


def _build_expected(target_patch: np.ndarray, target_holes: list[np.ndarray], marker_colors: list[int]) -> np.ndarray:
    out = target_patch.copy()
    for hole, color in zip(target_holes, marker_colors):
        out[hole[:, 0], hole[:, 1]] = int(color)
    return out


def _scene_config() -> tuple[int, int, int, list[int]]:
    target_holes = random.choice([1, 2])
    marker_count = random.choice([count for count in [2, 3] if count >= target_holes])
    extra_markers = marker_count - target_holes
    min_distractors = max(1, extra_markers)
    distractor_count = random.randint(min_distractors, 2)
    hole_counts = []
    for idx in range(distractor_count):
        if idx < extra_markers:
            hole_counts.append(2)
        else:
            hole_counts.append(random.choice([1, 2]))
    random.shuffle(hole_counts)
    return target_holes, marker_count, distractor_count, hole_counts


def _validate_scene(inp: np.ndarray, expected: np.ndarray, *, marker_count: int, target_hole_count: int) -> bool:
    mirror_matches = _find_matches(inp, mirror_markers=True)
    direct_matches = _find_matches(inp, mirror_markers=False)
    if len(mirror_matches) != 1:
        return False
    if direct_matches:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    comps = [_analyze_component(inp, comp) for comp in _find_monochrome_components(inp, BG)]
    infos = [info for info in comps if info is not None]
    holes = [key for info in infos if info["holes"] for key in info["hole_keys"]]
    markers = [info for info in infos if not info["holes"]]
    marker_keys = [info["shape_key"] for info in markers]
    if len(markers) != marker_count:
        return False
    if len(holes) < target_hole_count + 1:
        return False
    if set(holes) & set(marker_keys):
        return False
    target = mirror_matches[0]["object"]
    if len(target["holes"]) != target_hole_count:
        return False
    if np.any(expected == BG):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        target_hole_count, marker_count, distractor_count, distractor_holes = _scene_config()
        total_holes = target_hole_count + sum(distractor_holes)
        shape_masks = _choose_hole_shapes(total_holes)
        offset = 0

        target_hole_masks = shape_masks[offset : offset + target_hole_count]
        offset += target_hole_count
        distractor_shape_groups = []
        for count in distractor_holes:
            distractor_shape_groups.append(shape_masks[offset : offset + count])
            offset += count

        total_colors = 1 + distractor_count + marker_count
        colors = random.sample(PALETTE, total_colors)
        target_color = colors[0]
        distractor_colors = colors[1 : 1 + distractor_count]
        marker_colors = colors[1 + distractor_count :]

        target_built = _object_patch(target_color, target_hole_masks)
        if target_built is None:
            continue
        target_patch, target_holes = target_built

        distractor_patches = []
        distractor_hole_coords = []
        ok = True
        for color, masks in zip(distractor_colors, distractor_shape_groups):
            built = _object_patch(color, masks)
            if built is None:
                ok = False
                break
            patch, holes = built
            distractor_patches.append(patch)
            distractor_hole_coords.append(holes)
        if not ok:
            continue

        source_masks = list(target_hole_masks)
        required_extras = marker_count - target_hole_count
        sourced = 0
        for masks in distractor_shape_groups:
            if sourced >= required_extras:
                break
            if len(masks) >= 2:
                source_masks.append(random.choice(masks))
                sourced += 1
        if sourced != required_extras:
            continue
        if len(source_masks) != marker_count:
            continue

        expected = _build_expected(target_patch, target_holes, marker_colors[:target_hole_count])
        canvas_h = random.randint(20, 28)
        canvas_w = random.randint(20, 28)
        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        placed: list[tuple[int, int, int, int]] = []

        patches = [target_patch, *distractor_patches]
        marker_patches = [
            _marker_patch(color, mask)
            for color, mask in zip(marker_colors, source_masks)
        ]

        random.shuffle(marker_patches)

        place_ok = True
        for patch in patches:
            if not _place_patch(grid, patch, placed):
                place_ok = False
                break
        if not place_ok:
            continue
        for patch in marker_patches:
            if not _place_patch(grid, patch, placed):
                place_ok = False
                break
        if not place_ok:
            continue

        if _validate_scene(
            grid,
            expected,
            marker_count=marker_count,
            target_hole_count=target_hole_count,
        ):
            return grid

    raise RuntimeError("failed to generate a valid mirrored-hole ARC input")
