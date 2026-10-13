from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import crop_bbox, find_components, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(cells, dtype=int))
        return comps

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        return_format: str = "dict",
        **_: object,
    ) -> list[dict]:
        assert connectivity == 4
        assert return_format == "dict"
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        out: list[dict] = []
        if monochromatic:
            for color in sorted(set(map(int, grid.flatten())) - {bg}):
                for coords in _components_from_mask(grid == color):
                    r0, c0 = coords.min(axis=0)
                    r1, c1 = coords.max(axis=0)
                    out.append(
                        {
                            "color": int(color),
                            "coords": coords,
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "area": int(coords.shape[0]),
                        }
                    )
            return out
        for coords in _components_from_mask(grid != bg):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            out.append(
                {
                    "color": None,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
        return out


def _dihedral_variants(arr: np.ndarray) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    for k in range(4):
        rotated = np.rot90(arr, k)
        variants.append({"name": f"rot{k}", "mirrored": False, "array": rotated})
        flipped = np.fliplr(rotated)
        variants.append({"name": f"rot{k}_fliplr", "mirrored": True, "array": flipped})
    return variants


def _mask_variants_with_ids(mask: np.ndarray) -> list[dict[str, object]]:
    ids = np.arange(mask.size, dtype=int).reshape(mask.shape)
    variants: list[dict[str, object]] = []
    for mask_variant, ids_variant in zip(_dihedral_variants(mask), _dihedral_variants(ids)):
        variants.append(
            {
                "name": mask_variant["name"],
                "mirrored": mask_variant["mirrored"],
                "mask": np.asarray(mask_variant["array"], dtype=bool),
                "ids": np.asarray(ids_variant["array"], dtype=int),
            }
        )
    return variants


def _occupied_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(mask)]


def _sprite_from_component(grid: np.ndarray, bg: int, comp: dict) -> dict[str, object]:
    bbox = tuple(map(int, comp["bbox"]))
    sub = crop_bbox(grid, bbox)
    mask = sub != bg
    colors = sorted(set(map(int, sub[mask])))
    return {"bbox": bbox, "sub": sub, "mask": mask, "colors": colors}


def _match_source_to_target(source_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, object]:
    matches = [
        variant
        for variant in _mask_variants_with_ids(source_mask)
        if variant["mask"].shape == target_mask.shape and np.array_equal(variant["mask"], target_mask)
    ]
    assert len(matches) == 1, "each sprite must match the source by exactly one dihedral transform"
    return matches[0]


def _color_swap_pair(colors: list[int]) -> dict[int, int]:
    assert len(colors) == 2 and colors[0] != colors[1]
    return {int(colors[0]): int(colors[1]), int(colors[1]): int(colors[0])}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source sprite's two decoration colors onto every rotated or mirrored copy, swapping the pair on mirrored copies.

    Input structure:
    - The background is the most common color.
    - Several separated 4-connected sprites share one occupancy mask up to rotation or reflection and one common body color.
    - Exactly one sprite is the source: some of its occupied cells are recolored with exactly two decoration colors, while every other sprite is still monochrome.
    - The shared sprite mask has no non-trivial dihedral self-symmetry, so every copy's transform relative to the source is unique.

    Rule:
    - Find the unique decorated source sprite and use its occupied-cell mask as the reference orientation.
    - For each undecorated sprite, determine the unique dihedral transform that maps the source mask onto that sprite mask.
    - Copy every decorated source cell to the corresponding cell of that sprite.
    - If the transform includes a reflection, swap the two decoration colors before painting; if it is a pure rotation, keep the colors unchanged.

    Key insight:
    - The decorations label template cells of the sprite, and mirror parity is the hidden switch: the same marked cells propagate to every copy, but mirrored copies invert the two decoration colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert len(comps) >= 2, "expected multiple sprite components"

    sprites = [_sprite_from_component(g, bg, comp) for comp in comps]
    source_candidates = [sprite for sprite in sprites if len(sprite["colors"]) > 1]
    assert len(source_candidates) == 1, "expected exactly one decorated source sprite"
    source = source_candidates[0]
    others = [sprite for sprite in sprites if sprite is not source]
    assert others, "expected at least one undecorated copy"
    assert all(len(sprite["colors"]) == 1 for sprite in others), "non-source sprites must be monochrome"

    body_colors = {int(sprite["colors"][0]) for sprite in others}
    assert len(body_colors) == 1, "all undecorated copies must share one body color"
    body_color = next(iter(body_colors))
    source_colors = [int(color) for color in source["colors"]]
    assert body_color in source_colors and len(source_colors) == 3, "source must contain body color plus two decoration colors"

    decoration_colors = [color for color in source_colors if color != body_color]
    swap = _color_swap_pair(decoration_colors)

    source_sub = np.asarray(source["sub"], dtype=int)
    source_mask = np.asarray(source["mask"], dtype=bool)
    source_ids = np.arange(source_mask.size, dtype=int).reshape(source_mask.shape)
    decorated_cells: list[tuple[int, int, int]] = []
    for r, c in _occupied_coords(source_mask):
        color = int(source_sub[r, c])
        if color != body_color:
            decorated_cells.append((r, c, color))
    assert decorated_cells, "source must contain decorations"
    assert {color for _r, _c, color in decorated_cells} == set(decoration_colors), "both decoration colors must appear"

    out = g.copy()
    for sprite in others:
        match = _match_source_to_target(source_mask, np.asarray(sprite["mask"], dtype=bool))
        target_ids = np.asarray(match["ids"], dtype=int)
        target_map = {
            int(target_ids[r, c]): (int(r), int(c))
            for r, c in _occupied_coords(np.asarray(match["mask"], dtype=bool))
        }
        r0, c0, _r1, _c1 = map(int, sprite["bbox"])
        for sr, sc, src_color in decorated_cells:
            src_id = int(source_ids[sr, sc])
            tr, tc = target_map[src_id]
            paint = swap[src_color] if bool(match["mirrored"]) else src_color
            assert int(out[r0 + tr, c0 + tc]) == body_color, "decorations must land on undecorated body cells"
            out[r0 + tr, c0 + tc] = int(paint)

    return out


def _tight_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _unique_variant_count(mask: np.ndarray) -> int:
    seen: list[np.ndarray] = []
    for variant in _dihedral_variants(mask):
        arr = np.asarray(variant["array"], dtype=bool)
        if not any(prev.shape == arr.shape and np.array_equal(prev, arr) for prev in seen):
            seen.append(arr)
    return len(seen)


def _random_shape() -> np.ndarray:
    for _ in range(400):
        target_area = random.randint(6, 10)
        cells = {(0, 0)}
        while len(cells) < target_area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            nr, nc = r + dr, c + dc
            if abs(nr) > 5 or abs(nc) > 6:
                continue
            cells.add((nr, nc))
        mask = _tight_mask(cells)
        h, w = mask.shape
        if h < 3 or w < 3 or h > 6 or w > 7 or h == w:
            continue
        if int(mask.sum()) == h * w:
            continue
        if _unique_variant_count(mask) != 8:
            continue
        if np.any(mask.sum(axis=0) == h) or np.any(mask.sum(axis=1) == w):
            continue
        return mask
    raise AssertionError("failed to generate a non-symmetric sprite shape")


def _choose_variants(mask: np.ndarray) -> tuple[dict[str, object], list[dict[str, object]]]:
    variants = _mask_variants_with_ids(mask)
    src_idx = random.randrange(len(variants))
    source_variant = variants[src_idx]
    same_indices = [i for i, variant in enumerate(variants) if i != src_idx and bool(variant["mirrored"]) == bool(source_variant["mirrored"])]
    opposite_indices = [i for i, variant in enumerate(variants) if bool(variant["mirrored"]) != bool(source_variant["mirrored"])]
    assert same_indices and opposite_indices

    sprite_count = random.randint(3, 5)
    chosen_indices = [random.choice(same_indices), random.choice(opposite_indices)]
    remaining_indices = [i for i in range(len(variants)) if i != src_idx and i not in chosen_indices]
    random.shuffle(remaining_indices)
    chosen_indices.extend(remaining_indices[: sprite_count - 3])
    chosen = [variants[i] for i in chosen_indices]
    random.shuffle(chosen)
    return source_variant, chosen


def _pick_decorations(source_variant: dict[str, object]) -> list[tuple[int, int, int]]:
    mask = np.asarray(source_variant["mask"], dtype=bool)
    coords = _occupied_coords(mask)
    area = len(coords)
    deco_count = random.randint(2, min(5, area - 1))
    chosen = random.sample(coords, deco_count)
    colors = random.sample(range(1, 10), 2)
    deco_a, deco_b = colors
    assignments: list[tuple[int, int, int]] = []
    assignments.append((chosen[0][0], chosen[0][1], deco_a))
    assignments.append((chosen[1][0], chosen[1][1], deco_b))
    for r, c in chosen[2:]:
        assignments.append((r, c, random.choice((deco_a, deco_b))))
    return assignments


def _can_place(boxes: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int, H: int, W: int, gap: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    nr0, nc0, nr1, nc1 = r0 - gap, c0 - gap, r0 + h - 1 + gap, c0 + w - 1 + gap
    for br0, bc0, br1, bc1 in boxes:
        if not (nr1 < br0 or br1 < nr0 or nc1 < bc0 or bc1 < nc0):
            return False
    return True


def _place_variants(source_variant: dict[str, object], targets: list[dict[str, object]]) -> tuple[int, int, dict[str, tuple[int, int]], list[tuple[int, int, int, int]]]:
    items = [("source", source_variant)] + [(f"target{i}", variant) for i, variant in enumerate(targets)]
    random.shuffle(items)

    max_h = max(int(np.asarray(variant["mask"]).shape[0]) for _name, variant in items)
    max_w = max(int(np.asarray(variant["mask"]).shape[1]) for _name, variant in items)
    min_h = max(16, max_h * 2 + 6)
    min_w = max(16, max_w * 2 + 6)

    for _ in range(250):
        H = random.randint(min_h, 30)
        W = random.randint(min_w, 30)
        boxes: list[tuple[int, int, int, int]] = []
        placements: dict[str, tuple[int, int]] = {}
        okay = True
        for name, variant in items:
            h, w = np.asarray(variant["mask"]).shape
            placed = False
            for _attempt in range(200):
                r0 = random.randint(0, H - h)
                c0 = random.randint(0, W - w)
                if _can_place(boxes, r0, c0, h, w, H, W, gap=2):
                    placements[name] = (r0, c0)
                    boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
                    placed = True
                    break
            if not placed:
                okay = False
                break
        if okay:
            return H, W, placements, boxes
    raise AssertionError("failed to place sprite variants")


def _render_case(
    H: int,
    W: int,
    bg: int,
    body: int,
    source_variant: dict[str, object],
    targets: list[dict[str, object]],
    placements: dict[str, tuple[int, int]],
    decorations: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((H, W), bg, dtype=int)
    source_mask = np.asarray(source_variant["mask"], dtype=bool)
    target_specs = [("source", source_variant)] + [(f"target{i}", variant) for i, variant in enumerate(targets)]
    for name, variant in target_specs:
        r0, c0 = placements[name]
        mask = np.asarray(variant["mask"], dtype=bool)
        coords = np.argwhere(mask)
        grid[r0 + coords[:, 0], c0 + coords[:, 1]] = body

    deco_pair = sorted({int(color) for _r, _c, color in decorations})
    swap = _color_swap_pair(deco_pair)
    source_ids = np.asarray(source_variant["ids"], dtype=int)
    input_grid = grid.copy()
    for r, c, color in decorations:
        sr0, sc0 = placements["source"]
        input_grid[sr0 + r, sc0 + c] = int(color)

    output_grid = input_grid.copy()
    for i, target in enumerate(targets):
        target_ids = np.asarray(target["ids"], dtype=int)
        target_map = {
            int(target_ids[r, c]): (int(r), int(c))
            for r, c in _occupied_coords(np.asarray(target["mask"], dtype=bool))
        }
        tr0, tc0 = placements[f"target{i}"]
        mirrored = bool(source_variant["mirrored"]) != bool(target["mirrored"])
        for r, c, color in decorations:
            target_r, target_c = target_map[int(source_ids[r, c])]
            output_grid[tr0 + target_r, tc0 + target_c] = swap[color] if mirrored else int(color)

    return input_grid, output_grid


def _naive_no_swap_output(
    H: int,
    W: int,
    bg: int,
    body: int,
    source_variant: dict[str, object],
    targets: list[dict[str, object]],
    placements: dict[str, tuple[int, int]],
    decorations: list[tuple[int, int, int]],
) -> np.ndarray:
    input_grid, _correct = _render_case(H, W, bg, body, source_variant, targets, placements, decorations)
    output = input_grid.copy()
    source_ids = np.asarray(source_variant["ids"], dtype=int)
    for i, target in enumerate(targets):
        target_ids = np.asarray(target["ids"], dtype=int)
        target_map = {
            int(target_ids[r, c]): (int(r), int(c))
            for r, c in _occupied_coords(np.asarray(target["mask"], dtype=bool))
        }
        tr0, tc0 = placements[f"target{i}"]
        for r, c, color in decorations:
            target_r, target_c = target_map[int(source_ids[r, c])]
            output[tr0 + target_r, tc0 + target_c] = int(color)
    return output


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    bg, body = random.sample(range(10), 2)
    mask = _random_shape()
    source_variant, targets = _choose_variants(mask)

    occupied = int(np.asarray(source_variant["mask"], dtype=bool).sum())
    assert occupied >= 3
    while True:
        decorations = _pick_decorations(source_variant)
        deco_colors = {int(color) for _r, _c, color in decorations}
        if bg not in deco_colors and body not in deco_colors:
            break

    H, W, placements, _boxes = _place_variants(source_variant, targets)
    inp, expected = _render_case(H, W, bg, body, source_variant, targets, placements, decorations)

    same_parity = sum(bool(target["mirrored"]) == bool(source_variant["mirrored"]) for target in targets)
    opp_parity = len(targets) - same_parity
    assert same_parity >= 1 and opp_parity >= 1
    assert _unique_variant_count(mask) == 8
    assert np.array_equal(main(inp), expected)
    assert not np.array_equal(_naive_no_swap_output(H, W, bg, body, source_variant, targets, placements, decorations), expected)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            grid, _expected = _build_case()
            return grid
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid chirality-propagation puzzle")
