from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


Shape = tuple[tuple[int, int], ...]

_SHAPE_BOX = 4
_SHAPE_SIZES = (4, 5, 6)
_MAX_ATTEMPTS = 400
_PALETTE = tuple(range(10))


def _normalize_coords(coords: np.ndarray) -> Shape:
    pts = np.asarray(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_to_mask(shape: Shape) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _shape_signature(shape: Shape) -> Shape:
    return min(variant["shape"] for variant in _shape_variants(shape))


def _apply_transform(arr: np.ndarray, rotations: int, reflect: bool) -> np.ndarray:
    out = np.rot90(arr, int(rotations))
    if reflect:
        out = np.fliplr(out)
    return out.copy()


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    bbox = crop_bbox(arr, bbox_coords(np.argwhere(arr != bg)))
    return bbox


def _shape_variants(shape: Shape) -> list[dict]:
    base = _shape_to_mask(shape)
    seen: set[Shape] = set()
    out: list[dict] = []
    for rotations in range(4):
        for reflect in (False, True):
            mask = _apply_transform(base, rotations, reflect)
            norm = _normalize_coords(np.argwhere(mask))
            if norm in seen:
                continue
            seen.add(norm)
            out.append(
                {
                    "shape": norm,
                    "rotations": int(rotations),
                    "reflect": bool(reflect),
                }
            )
    return out


def _patch_variants(patch: np.ndarray, bg: int) -> list[dict]:
    seen: set[bytes] = set()
    out: list[dict] = []
    for rotations in range(4):
        for reflect in (False, True):
            transformed = _apply_transform(patch, rotations, reflect)
            transformed = _crop_non_bg(transformed, bg)
            key = transformed.tobytes() + str(transformed.shape).encode()
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "patch": transformed,
                    "shape": _normalize_coords(np.argwhere(transformed != bg)),
                    "rotations": int(rotations),
                    "reflect": bool(reflect),
                }
            )
    return out


def _build_shape_catalog() -> dict[int, list[Shape]]:
    catalog: dict[int, list[Shape]] = {area: [] for area in _SHAPE_SIZES}
    seen: set[Shape] = set()
    for area in _SHAPE_SIZES:
        for cells in combinations(range(_SHAPE_BOX * _SHAPE_BOX), area):
            coords = np.array([(cell // _SHAPE_BOX, cell % _SHAPE_BOX) for cell in cells], dtype=int)
            mask = np.zeros((_SHAPE_BOX, _SHAPE_BOX), dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            if len(components_mask(mask, connectivity=4)) != 1:
                continue
            shape = _normalize_coords(coords)
            variants = _shape_variants(shape)
            if len(variants) != 8:
                continue
            signature = min(variant["shape"] for variant in variants)
            if signature in seen:
                continue
            seen.add(signature)
            catalog[area].append(signature)
    return catalog


_SHAPE_CATALOG = _build_shape_catalog()


def _candidate_colors(bg: int, block_color: int) -> list[int]:
    return [color for color in _PALETTE if color not in (bg, block_color)]


def _make_colored_patch(shape: Shape, bg: int, palette: list[int]) -> np.ndarray:
    coords = list(shape)
    random.shuffle(coords)
    color_count = random.randint(2, min(4, len(coords)))
    colors = random.sample(palette, color_count)
    assignments = colors[:]
    while len(assignments) < len(coords):
        assignments.append(random.choice(colors))
    random.shuffle(assignments)
    patch = np.full(_shape_to_mask(shape).shape, bg, dtype=int)
    for (r, c), color in zip(sorted(coords), assignments):
        patch[r, c] = int(color)
    return patch


def _random_transform(shape: Shape) -> dict:
    return random.choice(_shape_variants(shape))


def _place_shape_in_block(
    rect_h: int,
    rect_w: int,
    shape: Shape,
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    mask = _shape_to_mask(shape)
    h, w = mask.shape
    anchors = [
        (r0, c0)
        for r0 in range(1, rect_h - h)
        for c0 in range(1, rect_w - w)
    ]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        cells = {(r0 + r, c0 + c) for r, c in shape}
        halo = set()
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rect_h and 0 <= cc < rect_w:
                        halo.add((rr, cc))
        if occupied & halo:
            continue
        occupied.update(cells)
        occupied.update(halo)
        return r0, c0
    return None


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    occ = np.argwhere(patch != bg)
    occupied = {(int(r0 + r), int(c0 + c)) for r, c in occ}
    for rr, cc in occupied:
        if int(grid[rr, cc]) != bg:
            return False
    for rr, cc in occupied:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if (nr, nc) not in occupied and int(grid[nr, nc]) != bg:
                        return False
    return True


def _place_patch_anywhere(grid: np.ndarray, patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    anchors = [
        (r0, c0)
        for r0 in range(grid.shape[0] - patch.shape[0] + 1)
        for c0 in range(grid.shape[1] - patch.shape[1] + 1)
    ]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        if _can_place_patch(grid, patch, r0, c0, bg):
            paste(grid, patch, r0, c0, transparent=bg)
            return r0, c0
    return None


def _extract_cavities(grid: np.ndarray, block_bbox: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, c0, r1, c1 = block_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    cavities: list[dict] = []
    for coords in components_mask(sub == bg, connectivity=4):
        if any(
            rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1
            for rr, cc in coords
        ):
            continue
        local_bbox = bbox_coords(coords)
        mask = crop_bbox(sub == bg, local_bbox)
        cavities.append(
            {
                "bbox": (r0 + local_bbox[0], c0 + local_bbox[1], r0 + local_bbox[2], c0 + local_bbox[3]),
                "shape": _normalize_coords(np.argwhere(mask)),
                "area": int(mask.sum()),
            }
        )
    cavities.sort(key=lambda item: item["bbox"])
    return cavities


def _find_block(grid: np.ndarray, bg: int) -> dict:
    objects = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected at least one foreground component"
    block = max(objects, key=lambda item: item["area"])
    return block


def _extract_candidates(grid: np.ndarray, bg: int, block_color: int) -> tuple[list[dict], np.ndarray]:
    mask = (grid != bg) & (grid != block_color)
    candidates: list[dict] = []
    for coords in components_mask(mask, connectivity=8):
        bbox = bbox_coords(coords)
        patch = crop_bbox(grid, bbox)
        candidates.append(
            {
                "bbox": bbox,
                "patch": patch,
                "shape": _normalize_coords(np.argwhere(patch != bg)),
                "area": int(np.sum(patch != bg)),
            }
        )
    candidates.sort(key=lambda item: item["bbox"])
    return candidates, mask


def _solve_matches(cavities: list[dict], candidates: list[dict], bg: int) -> list[dict]:
    options: list[list[dict]] = []
    for cavity in cavities:
        cavity_options: list[dict] = []
        for candidate_index, candidate in enumerate(candidates):
            for variant in _patch_variants(candidate["patch"], bg):
                if variant["shape"] != cavity["shape"]:
                    continue
                cavity_options.append(
                    {
                        "candidate_index": candidate_index,
                        "patch": variant["patch"],
                    }
                )
        assert cavity_options, "each cavity must have at least one matching candidate"
        options.append(cavity_options)

    order = sorted(range(len(cavities)), key=lambda idx: len(options[idx]))
    solution: list[dict] | None = None
    solution_count = 0

    def search(pos: int, used: set[int], chosen: dict[int, dict]) -> None:
        nonlocal solution, solution_count
        if solution_count > 1:
            return
        if pos == len(order):
            solution_count += 1
            if solution is None:
                solution = [chosen[idx] for idx in range(len(cavities))]
            return
        cavity_idx = order[pos]
        for item in options[cavity_idx]:
            candidate_index = item["candidate_index"]
            if candidate_index in used:
                continue
            chosen[cavity_idx] = item
            used.add(candidate_index)
            search(pos + 1, used, chosen)
            used.remove(candidate_index)
            del chosen[cavity_idx]

    search(0, set(), {})
    assert solution_count == 1 and solution is not None, "scene must admit one unique assignment"
    return solution


def _render_solution(grid: np.ndarray, cavities: list[dict], matches: list[dict], candidate_mask: np.ndarray, bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[candidate_mask] = bg
    for cavity, match in zip(cavities, matches):
        r0, c0, r1, c1 = cavity["bbox"]
        patch = match["patch"]
        assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
        paste(out, patch, r0, c0, transparent=bg)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill each cavity inside the large solid block with the uniquely matching transformed multicolor key, then erase every candidate sprite.

    Input structure:
    - A dominant background surrounds one large monochrome block whose interior contains several irregular background-colored cavities.
    - Outside the block are several separated multicolor candidate sprites on the background.
    - Exactly one sprite per cavity has an occupied shape that matches that cavity under one rotation or reflection; other sprites are distractors.

    Rule:
    - Find the large monochrome block and extract every enclosed cavity shape from its background holes.
    - For each cavity, locate the unique outside sprite whose occupied cells can be rotated or reflected to exactly match the cavity.
    - Stamp that whole transformed multicolor sprite into the cavity cell-for-cell, erase all outside sprites to background, and keep the rest of the block unchanged.

    Key insight:
    - Shape alone identifies the correct key, but the cavity-filling colors come from the key after applying its uniquely determined orientation.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = int(get_bg(grid))
    block = _find_block(grid, bg)
    cavities = _extract_cavities(grid, block["bbox"], bg)
    candidates, candidate_mask = _extract_candidates(grid, bg, block["color"])
    matches = _solve_matches(cavities, candidates, bg)
    return _render_solution(grid, cavities, matches, candidate_mask, bg)


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    bg = random.randrange(10)
    block_color = random.choice([color for color in _PALETTE if color != bg])
    grid_h = random.randint(21, 28)
    grid_w = random.randint(21, 28)
    block_h = random.randint(10, min(15, grid_h - 8))
    block_w = random.randint(11, min(16, grid_w - 8))
    block_r0 = random.randint(2, grid_h - block_h - 3)
    block_c0 = random.randint(2, grid_w - block_w - 3)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[block_r0 : block_r0 + block_h, block_c0 : block_c0 + block_w] = block_color

    num_cavities = random.randint(2, 3)
    cavity_areas = [random.choice(_SHAPE_SIZES) for _ in range(num_cavities)]
    chosen_shapes: list[Shape] = []
    used_signatures: set[Shape] = set()
    for area in cavity_areas:
        options = [shape for shape in _SHAPE_CATALOG[area] if _shape_signature(shape) not in used_signatures]
        if not options:
            raise RuntimeError("shape catalog exhausted")
        shape = random.choice(options)
        chosen_shapes.append(shape)
        used_signatures.add(_shape_signature(shape))

    occupied: set[tuple[int, int]] = set()
    cavity_specs: list[dict] = []
    palettes = _candidate_colors(bg, block_color)
    transform_mismatch = False
    reflection_seen = False
    for shape in chosen_shapes:
        variants = _shape_variants(shape)
        cavity_variant = random.choice(variants)
        key_variant = random.choice(variants)
        cavity_shape = cavity_variant["shape"]
        cavity_anchor = _place_shape_in_block(block_h, block_w, cavity_shape, occupied)
        if cavity_anchor is None:
            raise RuntimeError("failed to place cavity")
        canonical_patch = _make_colored_patch(shape, bg, palettes)
        cavity_patch = _crop_non_bg(
            _apply_transform(canonical_patch, cavity_variant["rotations"], cavity_variant["reflect"]),
            bg,
        )
        key_patch = _crop_non_bg(
            _apply_transform(canonical_patch, key_variant["rotations"], key_variant["reflect"]),
            bg,
        )
        transform_mismatch |= (
            cavity_variant["rotations"] != key_variant["rotations"]
            or cavity_variant["reflect"] != key_variant["reflect"]
        )
        reflection_seen |= cavity_variant["reflect"] or key_variant["reflect"]

        local_r0, local_c0 = cavity_anchor
        for dr, dc in cavity_shape:
            grid[block_r0 + local_r0 + dr, block_c0 + local_c0 + dc] = bg
        cavity_specs.append(
            {
                "shape": shape,
                "canonical_patch": canonical_patch,
                "cavity_variant": cavity_variant,
                "cavity_shape": cavity_shape,
                "cavity_patch": cavity_patch,
                "key_patch": key_patch,
                "bbox": (
                    block_r0 + local_r0,
                    block_c0 + local_c0,
                    block_r0 + local_r0 + cavity_patch.shape[0] - 1,
                    block_c0 + local_c0 + cavity_patch.shape[1] - 1,
                ),
                "area": len(shape),
            }
        )

    if not transform_mismatch:
        idx = random.randrange(len(cavity_specs))
        shape = cavity_specs[idx]["shape"]
        variants = _shape_variants(shape)
        cavity_variant = cavity_specs[idx]["cavity_variant"]
        alt = random.choice(
            [
                item
                for item in variants
                if item["shape"] != cavity_specs[idx]["cavity_shape"]
            ]
        )
        canonical_patch = cavity_specs[idx]["canonical_patch"]
        cavity_specs[idx]["key_patch"] = _crop_non_bg(
            _apply_transform(canonical_patch, alt["rotations"], alt["reflect"]),
            bg,
        )
        reflection_seen |= cavity_variant["reflect"] or alt["reflect"]

    if not reflection_seen and random.random() < 0.7:
        idx = random.randrange(len(cavity_specs))
        shape = cavity_specs[idx]["shape"]
        reflected = [
            item
            for item in _shape_variants(shape)
            if item["reflect"] and item["shape"] != cavity_specs[idx]["cavity_shape"]
        ]
        if reflected:
            alt = random.choice(reflected)
            canonical_patch = cavity_specs[idx]["canonical_patch"]
            cavity_specs[idx]["key_patch"] = _crop_non_bg(
                _apply_transform(canonical_patch, alt["rotations"], alt["reflect"]),
                bg,
            )

    true_patches = [spec["key_patch"] for spec in cavity_specs]
    distractor_count = random.randint(1, 3)
    distractors: list[np.ndarray] = []
    for idx in range(distractor_count):
        if idx == 0:
            area = random.choice([spec["area"] for spec in cavity_specs])
        else:
            area = random.choice(_SHAPE_SIZES)
        forbidden = {_shape_signature(spec["shape"]) for spec in cavity_specs}
        options = [shape for shape in _SHAPE_CATALOG[area] if _shape_signature(shape) not in forbidden]
        if not options:
            raise RuntimeError("failed to find distractor")
        shape = random.choice(options)
        distractors.append(_make_colored_patch(shape, bg, palettes))

    for patch in true_patches + distractors:
        if _place_patch_anywhere(grid, patch, bg) is None:
            raise RuntimeError("failed to place candidate")

    expected = np.full_like(grid, bg)
    expected[block_r0 : block_r0 + block_h, block_c0 : block_c0 + block_w] = block_color
    for spec in cavity_specs:
        paste(expected, spec["cavity_patch"], spec["bbox"][0], spec["bbox"][1], transparent=bg)
    return grid, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> None:
    bg = int(get_bg(inp))
    block = _find_block(inp, bg)
    cavities = _extract_cavities(inp, block["bbox"], bg)
    candidates, candidate_mask = _extract_candidates(inp, bg, block["color"])
    assert 2 <= len(cavities) <= 3
    assert len(candidates) > len(cavities)
    cavity_signatures = {_shape_signature(cavity["shape"]) for cavity in cavities}
    assert len(cavity_signatures) == len(cavities)
    matches = _solve_matches(cavities, candidates, bg)
    solved = _render_solution(inp, cavities, matches, candidate_mask, bg)
    assert np.array_equal(solved, expected)
    assert np.any(candidate_mask)
    assert np.any(inp != expected)
    for cavity, match in zip(cavities, matches):
        assert cavity["area"] == int(np.sum(match["patch"] != bg))
        unique_colors = {int(v) for v in match["patch"].flatten() if int(v) != bg}
        assert len(unique_colors) >= 2


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        try:
            inp, expected = _build_sample()
        except RuntimeError:
            continue
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            _semantic_checks(inp, expected)
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
