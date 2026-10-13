from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
SHAPE_COLORS = (1, 2, 3, 4, 5)
MARKER_COLORS = (7, 8, 9)
TILE_SIZE = 5
GRID_SIZE = 17
DIVIDERS = (5, 11)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CLASS_SIZE_OPTIONS = {
    2: ((4, 5),),
    3: ((3, 3, 3), (2, 3, 4), (2, 2, 5)),
}
MAX_GEN_ATTEMPTS = 500


def _non_divider_runs(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    divider_set = set(int(i) for i in divider_indices)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in divider_set:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected non-empty mask")
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _tight_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_mask(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    for k in range(4):
        rotated = np.rot90(base, k)
        for candidate in (rotated, np.fliplr(rotated)):
            tight = _tight_mask(candidate)
            key = _mask_key(tight)
            if key not in seen:
                seen.add(key)
                variants.append(tight)
    return variants


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_mask_key(variant) for variant in _unique_variants(mask))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _marker_candidates(tile_shape_mask: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    shape_coords = np.argwhere(tile_shape_mask)
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            if tile_shape_mask[r, c]:
                continue
            if any(max(abs(r - sr), abs(c - sc)) <= 1 for sr, sc in shape_coords):
                continue
            candidates.append((r, c))
    return candidates


def _placement_options(oriented_mask: np.ndarray, require_marker_space: bool) -> list[dict]:
    h, w = oriented_mask.shape
    options: list[dict] = []
    for r0 in range(TILE_SIZE - h + 1):
        for c0 in range(TILE_SIZE - w + 1):
            tile_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
            tile_mask[r0 : r0 + h, c0 : c0 + w] = oriented_mask
            marker_spots = _marker_candidates(tile_mask)
            if require_marker_space and not marker_spots:
                continue
            options.append(
                {
                    "shape_coords": [tuple(map(int, xy)) for xy in np.argwhere(tile_mask)],
                    "marker_spots": marker_spots,
                }
            )
    return options


def _has_any_marked_placement(mask: np.ndarray) -> bool:
    return any(_placement_options(variant, require_marker_space=True) for variant in _unique_variants(mask))


def _random_connected_mask(area: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(1, min(4, area))
        w = random.randint(1, 4)
        if area > h * w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            candidates = list(frontier)
            weights = []
            for r, c in candidates:
                touch_count = sum(
                    0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in NEIGHBORS4
                )
                weights.append(1 + int(touch_count))
            chosen = random.choices(candidates, weights=weights, k=1)[0]
            mask[chosen] = True
        if int(mask.sum()) != area:
            continue
        tight = _tight_mask(mask)
        if tight.shape[0] > 4 or tight.shape[1] > 4:
            continue
        if len(_unique_variants(tight)) < 2:
            continue
        if not _has_any_marked_placement(tight):
            continue
        return tight
    return None


def _generate_class_masks(class_count: int) -> list[np.ndarray] | None:
    if class_count == 2:
        shared_area = random.randint(4, 6)
        area_plan = [shared_area, shared_area]
    else:
        shared_area = random.randint(4, 6)
        area_plan = [shared_area, shared_area, random.randint(3, 6)]

    masks: list[np.ndarray] = []
    signatures: set[tuple[str, ...]] = set()
    for area in area_plan:
        for _ in range(200):
            candidate = _random_connected_mask(area)
            if candidate is None:
                continue
            signature = _shape_signature(candidate)
            if signature in signatures:
                continue
            signatures.add(signature)
            masks.append(candidate)
            break
        else:
            return None
    return masks


def _tile_bounds(tile_index: int) -> tuple[int, int]:
    tile_r, tile_c = divmod(tile_index, 3)
    return tile_r * (TILE_SIZE + 1), tile_c * (TILE_SIZE + 1)


def _render_input_from_plan(
    class_masks: list[np.ndarray],
    class_sizes: list[int],
    class_targets: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    tile_ids = list(range(9))
    random.shuffle(tile_ids)
    metadata: list[dict] = []
    cursor = 0

    for class_id, (mask, size, target_color) in enumerate(zip(class_masks, class_sizes, class_targets)):
        member_tiles = tile_ids[cursor : cursor + size]
        cursor += size
        marked_tile = random.choice(member_tiles)
        variants = _unique_variants(mask)
        variant_keys = [_mask_key(variant) for variant in variants]
        if size > 1 and len(set(variant_keys)) < 2:
            return None

        chosen_variants = [random.randrange(len(variants)) for _ in range(size)]
        if size > 1:
            chosen_variants[0] = random.randrange(len(variants))
            different = [idx for idx, key in enumerate(variant_keys) if key != variant_keys[chosen_variants[0]]]
            if not different:
                return None
            chosen_variants[1] = random.choice(different)
            random.shuffle(chosen_variants)

        for tile_index, variant_idx in zip(member_tiles, chosen_variants):
            oriented = variants[variant_idx]
            needs_marker = tile_index == marked_tile
            options = _placement_options(oriented, require_marker_space=needs_marker)
            if not options:
                return None
            option = random.choice(options)
            shape_color = random.choice(SHAPE_COLORS)
            local_tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
            for r, c in option["shape_coords"]:
                local_tile[r, c] = shape_color

            marker_pos = None
            if needs_marker:
                marker_pos = random.choice(option["marker_spots"])
                local_tile[marker_pos] = target_color

            r0, c0 = _tile_bounds(tile_index)
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = local_tile
            metadata.append(
                {
                    "tile_index": int(tile_index),
                    "class_id": int(class_id),
                    "signature": _shape_signature(mask),
                    "shape_color": int(shape_color),
                    "shape_area": int(len(option["shape_coords"])),
                    "raw_key": _mask_key(oriented),
                    "target_color": int(target_color),
                    "marker_pos": marker_pos,
                    "shape_coords": option["shape_coords"],
                }
            )
    return grid, metadata


def _extract_tiles(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    row_dividers = [r for r in range(h) if np.all(grid[r, :] == BG)]
    col_dividers = [c for c in range(w) if np.all(grid[:, c] == BG)]
    row_runs = _non_divider_runs(h, row_dividers)
    col_runs = _non_divider_runs(w, col_dividers)
    assert len(row_runs) == 3 and len(col_runs) == 3, "expected a 3x3 tile lattice"
    assert len({r1 - r0 + 1 for r0, r1 in row_runs}) == 1, "tile rows must be equal height"
    assert len({c1 - c0 + 1 for c0, c1 in col_runs}) == 1, "tile cols must be equal width"
    assert {r1 - r0 + 1 for r0, r1 in row_runs} == {TILE_SIZE}
    assert {c1 - c0 + 1 for c0, c1 in col_runs} == {TILE_SIZE}

    infos: list[dict] = []
    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            shape_mask = np.isin(tile, SHAPE_COLORS)
            shape_coords = np.argwhere(shape_mask)
            assert 3 <= len(shape_coords) <= 6, "tile must contain one 3-6 cell shape"
            assert _is_connected(shape_mask), "shape must be 4-connected"
            shape_colors = set(map(int, tile[shape_mask]))
            assert len(shape_colors) == 1, "shape must be monochromatic"

            marker_coords = np.argwhere(np.isin(tile, MARKER_COLORS))
            assert len(marker_coords) <= 1, "each tile may have at most one marker"
            if len(marker_coords) == 1:
                marker_r, marker_c = map(int, marker_coords[0])
                for sr, sc in shape_coords:
                    assert max(abs(marker_r - int(sr)), abs(marker_c - int(sc))) > 1, "marker touches shape"
                marker = (marker_r, marker_c)
                marker_color = int(tile[marker_r, marker_c])
            else:
                marker = None
                marker_color = None

            used_mask = shape_mask | np.isin(tile, MARKER_COLORS)
            assert not np.any((tile != BG) & ~used_mask), "unexpected extra colors in tile"

            tight = _tight_mask(shape_mask)
            infos.append(
                {
                    "tile_index": 3 * tile_r + tile_c,
                    "origin": (int(r0), int(c0)),
                    "shape_color": int(next(iter(shape_colors))),
                    "shape_area": int(len(shape_coords)),
                    "shape_coords_local": [tuple(map(int, xy)) for xy in shape_coords],
                    "shape_coords_global": [(int(r0 + rr), int(c0 + cc)) for rr, cc in shape_coords],
                    "marker_local": marker,
                    "marker_global": None if marker is None else (int(r0 + marker[0]), int(c0 + marker[1])),
                    "marker_color": marker_color,
                    "raw_key": _mask_key(tight),
                    "signature": _shape_signature(tight),
                }
            )
    return infos


def _expected_output_from_infos(grid: np.ndarray, infos: list[dict]) -> tuple[np.ndarray, dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for info in infos:
        groups[info["signature"]].append(info)

    assert 2 <= len(groups) <= 3, "there must be 2-3 equivalence classes"
    out = np.array(grid, dtype=int, copy=True)
    for info in infos:
        if info["marker_global"] is not None:
            mr, mc = info["marker_global"]
            out[mr, mc] = BG

    for group in groups.values():
        marker_colors = [info["marker_color"] for info in group if info["marker_color"] is not None]
        assert len(marker_colors) == 1, "each class must have exactly one marker"
        target_color = int(marker_colors[0])
        for info in group:
            for r, c in info["shape_coords_global"]:
                out[r, c] = target_color
    return out, groups


def _validate_instance(grid: np.ndarray, metadata: list[dict] | None = None) -> None:
    infos = _extract_tiles(grid)
    expected, groups = _expected_output_from_infos(grid, infos)

    class_sizes = sorted(len(group) for group in groups.values())
    assert all(size >= 2 for size in class_sizes), "all classes must have at least two members"

    marker_colors = [info["marker_color"] for info in infos if info["marker_color"] is not None]
    assert len(marker_colors) == len(groups), "marker count must equal class count"
    assert len(set(marker_colors)) == len(marker_colors), "class target colors must be unique"

    group_areas = [group[0]["shape_area"] for group in groups.values()]
    assert any(group_areas[i] == group_areas[j] for i in range(len(group_areas)) for j in range(i + 1, len(group_areas))), (
        "need same-area shapes across classes to block area shortcuts"
    )
    assert any(len({info["raw_key"] for info in group}) >= 2 for group in groups.values()), (
        "need an equivalence class with multiple visible orientations"
    )

    shape_color_to_signatures: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    for info in infos:
        shape_color_to_signatures[info["shape_color"]].add(info["signature"])
    mixed_within_group = any(len({info["shape_color"] for info in group}) >= 2 for group in groups.values())
    shared_across_groups = any(len(sigs) >= 2 for sigs in shape_color_to_signatures.values())
    assert mixed_within_group or shared_across_groups, "shape colors should not reveal the grouping"

    out = main(grid)
    assert np.array_equal(out, expected), "main() must match the puzzle semantics"
    for info in infos:
        if info["marker_global"] is not None:
            mr, mc = info["marker_global"]
            assert out[mr, mc] == BG, "marker pixels must be removed"
    assert any(
        info["marker_global"] is None and any(out[r, c] != info["shape_color"] for r, c in info["shape_coords_global"])
        for info in infos
    ), "at least one unmarked tile must be recolored via class propagation"

    if metadata is not None:
        by_tile = {info["tile_index"]: info for info in infos}
        for item in metadata:
            parsed = by_tile[item["tile_index"]]
            assert parsed["signature"] == item["signature"]
            assert parsed["shape_area"] == item["shape_area"]
            assert parsed["marker_color"] == (None if item["marker_pos"] is None else item["target_color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each tile shape by the marker color assigned to its rotation/reflection equivalence class.

    Input structure:
    - The grid is a 3x3 lattice of equal 5x5 tiles separated by full black divider rows and columns.
    - Each tile contains one 4-connected monochromatic shape using colors 1-5, and some tiles also contain one isolated marker pixel using colors 7-9.
    - Shapes in different tiles may differ in position, orientation, reflection, and source color.

    Rule:
    - For every tile, crop the shape to its tight mask and compare masks up to any rotation and horizontal/vertical reflection.
    - Group tiles whose normalized masks are dihedrally equivalent.
    - In each group, the unique marker pixel names the target color for that whole group; recolor every shape pixel in the group to that marker color.
    - Remove all marker pixels and keep the dividers, background, shape positions, and orientations unchanged.

    Key insight:
    - The marker does not label a location; it labels a hidden geometry-defined class, so the solver must normalize shapes across tiles before any recoloring is determined.
    """
    g = np.asarray(grid, dtype=int)
    infos = _extract_tiles(g)
    out, _groups = _expected_output_from_infos(g, infos)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        class_count = random.choice((2, 3))
        class_sizes = list(random.choice(CLASS_SIZE_OPTIONS[class_count]))
        random.shuffle(class_sizes)
        class_masks = _generate_class_masks(class_count)
        if class_masks is None:
            continue
        class_targets = random.sample(MARKER_COLORS, class_count)
        rendered = _render_input_from_plan(class_masks, class_sizes, class_targets)
        if rendered is None:
            continue
        grid, metadata = rendered
        try:
            _validate_instance(grid, metadata)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
