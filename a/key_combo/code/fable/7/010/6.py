from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
ALL_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 800


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def d4_transforms(arr: np.ndarray) -> list[np.ndarray]:
    flip = np.fliplr(arr)
    return [np.rot90(arr, k).copy() for k in range(4)] + [np.rot90(flip, k).copy() for k in range(4)]


def shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return ()
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def shape_diff_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    return len(set(shape_key(mask_a)) ^ set(shape_key(mask_b)))


def component_colors(grid: np.ndarray, coords: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted({int(grid[r, c]) for r, c in coords}))


def corner_name(bbox: tuple[int, int, int, int], grid_shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    h, w = grid_shape
    cr = (r0 + r1) / 2
    cc = (c0 + c1) / 2
    return ("T" if cr < h / 2 else "B") + ("L" if cc < w / 2 else "R")


def analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == 5, "expected one center sprite and four corner silhouettes"

    objects = []
    for coords in comps:
        bb = bbox_coords(coords)
        patch = crop_bbox(g, bb)
        colors = component_colors(g, coords)
        objects.append(
            {
                "coords": coords,
                "bbox": bb,
                "patch": patch,
                "mask": patch != bg,
                "colors": colors,
                "corner": corner_name(bb, g.shape),
            }
        )

    sprite_candidates = [obj for obj in objects if len(obj["colors"]) > 1]
    assert len(sprite_candidates) == 1, "expected exactly one multicolor sprite"
    sprite = sprite_candidates[0]

    corners = [obj for obj in objects if obj is not sprite]
    assert len(corners) == 4
    assert {obj["corner"] for obj in corners} == {"TL", "TR", "BL", "BR"}
    for obj in corners:
        assert len(obj["colors"]) == 1, "corner silhouettes must be monochrome"

    sprite_variants = d4_transforms(sprite["patch"])
    sprite_mask_variants = [variant != bg for variant in sprite_variants]
    matches: list[tuple[dict, int]] = []
    for obj in corners:
        for t_idx, mask in enumerate(sprite_mask_variants):
            if obj["mask"].shape == mask.shape and np.array_equal(obj["mask"], mask):
                matches.append((obj, t_idx))
    assert len(matches) == 1, "expected exactly one matching corner and one matching orientation"

    return {
        "grid": g,
        "bg": bg,
        "sprite": sprite,
        "corners": sorted(corners, key=lambda obj: obj["corner"]),
        "match_corner": matches[0][0],
        "transform_index": matches[0][1],
        "sprite_variants": sprite_variants,
        "sprite_mask_variants": sprite_mask_variants,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the central multicolor sprite onto the uniquely matching corner silhouette, using that silhouette's orientation.

    Input structure:
    - A black-background grid contains one multicolor connected sprite away from the corners.
    - Each corner holds one monochrome silhouette component.
    - Exactly one corner silhouette has the same mask as the sprite under one rotation/reflection; the other three are near-miss decoys.

    Rule:
    - Compare the central sprite mask against each corner silhouette under all eight rotation/reflection transforms.
    - Find the unique corner and transform whose silhouette mask matches exactly.
    - Apply that same transform to the full multicolor sprite, stamp it onto the matched corner silhouette location, and erase the original central sprite.

    Key insight:
    - The correct corner tells both where the sprite goes and which rotation/reflection to apply, so the internal color pattern must move with the same transform as the outer mask.
    """
    info = analyze_scene(grid)
    out = info["grid"].copy()
    sprite_coords = info["sprite"]["coords"]
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = info["bg"]

    patch = info["sprite_variants"][info["transform_index"]]
    r0, c0, _, _ = info["match_corner"]["bbox"]
    paste(out, patch, r0, c0, transparent=info["bg"])
    return out


def random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        target_area = random.randint(5, min(h * w - 1, 9))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target_area:
            r, c = random.choice(tuple(cells))
            nbrs = [(nr, nc) for nr, nc in neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells]
            if not nbrs:
                continue
            cells.add(random.choice(nbrs))

        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        keys = {shape_key(variant) for variant in d4_transforms(mask.astype(int))}
        if len(keys) != 8:
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric connected mask")


def colorize_mask(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    coords = [tuple(x) for x in np.argwhere(mask)]
    k = random.randint(2, min(4, len(palette), len(coords)))
    use_colors = random.sample(palette, k=k)
    while True:
        assigned = [random.choice(use_colors) for _ in coords]
        if len(set(assigned)) >= 2:
            break
    for (r, c), color in zip(coords, assigned):
        sprite[r, c] = color
    return sprite


def connected_from_shape_key(key: tuple[tuple[int, int], ...]) -> bool:
    cells = set(key)
    if not cells:
        return False
    q = deque([next(iter(cells))])
    seen = set()
    while q:
        cell = q.popleft()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nbr in neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                q.append(nbr)
    return len(seen) == len(cells)


def mutate_shape(mask: np.ndarray, forbidden_keys: set[tuple[tuple[int, int], ...]]) -> np.ndarray:
    base = set(shape_key(mask))
    assert base
    for _ in range(400):
        current = set(base)
        diff_goal = random.choice((1, 2))
        for _step in range(diff_goal):
            addable = set()
            for r, c in current:
                for nr, nc in neighbors4(r, c):
                    if nr >= -1 and nc >= -1:
                        addable.add((nr, nc))
            addable -= current

            removable = []
            if len(current) > 4:
                for cell in list(current):
                    trial = tuple(sorted(current - {cell}))
                    if connected_from_shape_key(trial):
                        removable.append(cell)

            ops = []
            if addable:
                ops.append(("add", random.choice(tuple(addable))))
            if removable:
                ops.append(("remove", random.choice(removable)))
            if not ops:
                break
            op, cell = random.choice(ops)
            if op == "add":
                current.add(cell)
            else:
                current.remove(cell)

        if current == base:
            continue
        if not connected_from_shape_key(tuple(sorted(current))):
            continue

        arr = np.array(sorted(current), dtype=int)
        arr = arr - arr.min(axis=0)
        out = np.zeros((int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1), dtype=bool)
        out[arr[:, 0], arr[:, 1]] = True
        key = shape_key(out)
        if key in forbidden_keys:
            continue
        diff = shape_diff_count(mask, out)
        if diff not in (1, 2):
            continue
        return out
    raise RuntimeError("failed to create a near-shape decoy")


def silhouette_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def build_expected_output(info: dict) -> np.ndarray:
    out = info["grid"].copy()
    sprite_coords = info["sprite"]["coords"]
    out[sprite_coords[:, 0], sprite_coords[:, 1]] = info["bg"]
    patch = info["sprite_variants"][info["transform_index"]]
    r0, c0, _, _ = info["match_corner"]["bbox"]
    paste(out, patch, r0, c0, transparent=info["bg"])
    return out


def validate_instance(grid: np.ndarray) -> None:
    info = analyze_scene(grid)
    sprite_colors = set(info["sprite"]["colors"])
    assert len(sprite_colors) >= 2
    assert info["transform_index"] != 0, "generator must require a real reorientation"

    exact_keys = {shape_key(mask) for mask in info["sprite_mask_variants"]}
    for corner in info["corners"]:
        key = shape_key(corner["mask"])
        if corner is info["match_corner"]:
            assert key in exact_keys
        else:
            assert key not in exact_keys
            diff = min(shape_diff_count(corner["mask"], mask) for mask in info["sprite_mask_variants"])
            assert diff in (1, 2), "decoys must differ by one or two cells from some true transform"

    expected = build_expected_output(info)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, grid)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        base_mask = random_connected_mask()
        sprite_palette = random.sample(list(ALL_COLORS), k=random.randint(4, 5))
        base_sprite = colorize_mask(base_mask, sprite_palette)
        center_idx = random.randrange(8)
        center_sprite = d4_transforms(base_sprite)[center_idx]
        center_mask = center_sprite != BG

        sprite_variants = d4_transforms(center_sprite)
        sprite_masks = [variant != BG for variant in sprite_variants]
        match_idx = random.randint(1, 7)
        target_corner = random.choice(("TL", "TR", "BL", "BR"))

        remaining_colors = [color for color in ALL_COLORS if color not in set(sprite_palette)]
        if len(remaining_colors) < 4:
            continue
        corner_colors = dict(zip(("TL", "TR", "BL", "BR"), random.sample(remaining_colors, 4)))

        exact_keys = {shape_key(mask) for mask in sprite_masks}
        used_decoys: set[tuple[tuple[int, int], ...]] = set()
        corner_masks: dict[str, np.ndarray] = {}
        for name in ("TL", "TR", "BL", "BR"):
            if name == target_corner:
                corner_masks[name] = sprite_masks[match_idx]
            else:
                ref_mask = random.choice(sprite_masks)
                decoy = mutate_shape(ref_mask, exact_keys | used_decoys)
                used_decoys.add(shape_key(decoy))
                corner_masks[name] = decoy

        corner_patches = {name: silhouette_from_mask(mask, corner_colors[name]) for name, mask in corner_masks.items()}

        top_band = max(corner_patches["TL"].shape[0], corner_patches["TR"].shape[0]) + 2
        bottom_band = max(corner_patches["BL"].shape[0], corner_patches["BR"].shape[0]) + 2
        left_band = max(corner_patches["TL"].shape[1], corner_patches["BL"].shape[1]) + 2
        right_band = max(corner_patches["TR"].shape[1], corner_patches["BR"].shape[1]) + 2

        min_h = top_band + center_sprite.shape[0] + bottom_band
        min_w = left_band + center_sprite.shape[1] + right_band
        if min_h > 30 or min_w > 30:
            continue

        h = random.randint(min_h, min(30, min_h + 6))
        w = random.randint(min_w, min(30, min_w + 6))
        center_r = random.randint(top_band, h - bottom_band - center_sprite.shape[0])
        center_c = random.randint(left_band, w - right_band - center_sprite.shape[1])

        grid = np.zeros((h, w), dtype=int)
        paste(grid, corner_patches["TL"], 0, 0, transparent=BG)
        paste(grid, corner_patches["TR"], 0, w - corner_patches["TR"].shape[1], transparent=BG)
        paste(grid, corner_patches["BL"], h - corner_patches["BL"].shape[0], 0, transparent=BG)
        paste(grid, corner_patches["BR"], h - corner_patches["BR"].shape[0], w - corner_patches["BR"].shape[1], transparent=BG)
        paste(grid, center_sprite, center_r, center_c, transparent=BG)

        try:
            validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
