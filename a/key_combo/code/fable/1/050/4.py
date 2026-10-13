from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


BG = Color.BLACK
PIECE_COLOR = Color.GREY
ANSWER_COLOR = Color.GREEN
CONTAINER_COLORS = [c for c in Color.ALL_COLORS if c not in {BG, PIECE_COLOR, ANSWER_COLOR}]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask.astype(bool))
    assert b is not None, "mask must be non-empty"
    return crop_bbox(mask.astype(bool), b).astype(bool)


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    base = _normalize_mask(mask)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        variant = np.fliplr(base) if flipped else base
        for k in range(4):
            transformed = _normalize_mask(np.rot90(variant, k))
            key = (tuple(map(int, transformed.shape)), transformed.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(transformed)
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = [
        (tuple(map(int, transformed.shape)), transformed.tobytes())
        for transformed in _unique_transforms(mask)
    ]
    return min(keys)


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    base = _normalize_mask(mask)
    keys: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = _normalize_mask(np.rot90(base, k))
        keys.add((tuple(map(int, rotated.shape)), rotated.tobytes()))
    return keys


def _requires_mirror(display_mask: np.ndarray, placed_mask: np.ndarray) -> bool:
    placed_key = (tuple(map(int, _normalize_mask(placed_mask).shape)), _normalize_mask(placed_mask).tobytes())
    return placed_key not in _rotation_keys(display_mask)


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return any(
        int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1
        for r, c in coords
    )


def _extract_piece_masks(grid: np.ndarray) -> tuple[int, list[np.ndarray], np.ndarray]:
    container_colors = sorted(set(map(int, grid.flatten())) - {BG, PIECE_COLOR})
    assert len(container_colors) == 1, f"expected exactly one container color, got {container_colors}"
    container_color = container_colors[0]

    container_components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[container_color],
        return_format="dict",
    )
    assert len(container_components) == 1, "expected exactly one monochrome container"
    container = container_components[0]
    container_mask = np.zeros(grid.shape, dtype=bool)
    coords = container["coords"]
    container_mask[coords[:, 0], coords[:, 1]] = True

    piece_components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[PIECE_COLOR],
        return_format="dict",
    )
    assert piece_components, "expected at least one gray puzzle piece"
    piece_masks: list[np.ndarray] = []
    for piece in piece_components:
        r0, c0, r1, c1 = piece["bbox"]
        piece_masks.append((grid[r0 : r1 + 1, c0 : c1 + 1] == PIECE_COLOR))
    return container_color, piece_masks, container_mask


def _hole_mask(grid: np.ndarray) -> np.ndarray:
    holes = np.zeros(grid.shape, dtype=bool)
    for coords in components_mask(grid == BG, connectivity=4):
        if _touches_boundary(coords, grid.shape):
            continue
        holes[coords[:, 0], coords[:, 1]] = True
    assert holes.any(), "expected at least one enclosed hole"
    return holes


def _candidate_sets_for_piece(piece_mask: np.ndarray, hole_mask: np.ndarray) -> list[frozenset[tuple[int, int]]]:
    h_grid, w_grid = hole_mask.shape
    candidates: dict[frozenset[tuple[int, int]], None] = {}
    for transformed in _unique_transforms(piece_mask):
        h, w = map(int, transformed.shape)
        if h > h_grid or w > w_grid:
            continue
        rr, cc = np.nonzero(transformed)
        rr = rr.astype(int)
        cc = cc.astype(int)
        for r0 in range(h_grid - h + 1):
            for c0 in range(w_grid - w + 1):
                patch = hole_mask[r0 : r0 + h, c0 : c0 + w]
                if not bool(np.all(patch[transformed])):
                    continue
                cells = frozenset((r0 + int(r), c0 + int(c)) for r, c in zip(rr, cc))
                candidates[cells] = None
    return list(candidates)


def _solve_piece_layout(
    piece_masks: list[np.ndarray],
    hole_mask: np.ndarray,
    *,
    max_solutions: int = 2,
) -> tuple[list[list[frozenset[tuple[int, int]]]], list[list[frozenset[tuple[int, int]]]]]:
    candidate_lists = [_candidate_sets_for_piece(piece_mask, hole_mask) for piece_mask in piece_masks]
    if any(len(candidates) == 0 for candidates in candidate_lists):
        return [], candidate_lists

    order = sorted(
        range(len(piece_masks)),
        key=lambda idx: (len(candidate_lists[idx]), -int(np.count_nonzero(piece_masks[idx]))),
    )
    placed: list[frozenset[tuple[int, int]] | None] = [None] * len(piece_masks)
    used: set[tuple[int, int]] = set()
    solutions: list[list[frozenset[tuple[int, int]]]] = []

    def backtrack(depth: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if depth == len(order):
            assert all(candidate is not None for candidate in placed)
            solutions.append([candidate for candidate in placed if candidate is not None])
            return

        piece_idx = order[depth]
        remaining = order[depth + 1 :]
        for candidate in candidate_lists[piece_idx]:
            if not used.isdisjoint(candidate):
                continue
            placed[piece_idx] = candidate
            used.update(candidate)
            if all(any(used.isdisjoint(option) for option in candidate_lists[other]) for other in remaining):
                backtrack(depth + 1)
            for cell in candidate:
                used.remove(cell)
            placed[piece_idx] = None

    backtrack(0)

    return solutions, candidate_lists


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed each gray piece into the container's enclosed holes, then mark the uncovered remainder green.

    Input structure:
    - A black background contains one large monochrome container and several separate gray pieces.
    - The container is solid except for enclosed black hole regions inside its bounding area.
    - Every gray piece is a connected sprite outside the container.

    Rule:
    - Treat the enclosed black regions as the hole space to be tiled.
    - For each gray piece, consider all rotations and mirror variants, and find the unique collision-free placement of all pieces inside the holes.
    - Remove the exterior gray pieces, recolor the placed cells to the container color, and recolor every still-uncovered hole cell green.

    Key insight:
    - The visible output is only the leftover cavity after a hidden exact-fit packing; the fitted pieces disappear because they inherit the container's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, f"expected a 2D ARC grid, got {g.shape}"

    container_color, piece_masks, _container_mask = _extract_piece_masks(g)
    hole_mask = _hole_mask(g)
    solutions, candidate_lists = _solve_piece_layout(piece_masks, hole_mask, max_solutions=2)
    assert all(candidate_lists), "every piece must have at least one valid placement"
    assert len(solutions) == 1, "expected a unique global placement of all pieces"

    placed_cells = set().union(*solutions[0])

    out = g.copy()
    out[out == PIECE_COLOR] = BG
    out[hole_mask] = ANSWER_COLOR
    for r, c in placed_cells:
        out[r, c] = container_color
    return out


GEN_MAX_ATTEMPTS = 400
GEN_LOCAL_PATCH_ATTEMPTS = 120
GEN_GROUP_PATCH_ATTEMPTS = 140
GEN_PIECE_ATTEMPTS = 120

PIECE_COUNT_RANGE = (2, 4)
PURE_HOLE_COUNT_RANGE = (0, 2)
CONTAINER_H_RANGE = (12, 18)
CONTAINER_W_RANGE = (13, 20)
GRID_MARGIN_MIN = 2
GRID_MARGIN_MAX = 5
EXTRA_CELL_MAX = 4
PURE_HOLE_AREA_MAX = 2


def _random_connected_mask(min_h: int = 2, max_h: int = 4, min_w: int = 2, max_w: int = 5) -> np.ndarray:
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    target = random.randint(3, min(8, h * w - 1))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _normalize_mask(mask)
    if min(mask.shape) < 2:
        raise ValueError("reject thin degenerate piece")
    if int(mask.sum()) < 3 or bool(mask.all()):
        raise ValueError("reject tiny or solid-rectangle piece")
    if len(_unique_transforms(mask)) < 2:
        raise ValueError("reject fully symmetric piece")
    return mask


def _choose_distinct_pieces() -> list[np.ndarray]:
    target = random.randint(*PIECE_COUNT_RANGE)
    pieces: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    attempts = 0
    while len(pieces) < target:
        attempts += 1
        if attempts > GEN_PIECE_ATTEMPTS:
            raise RuntimeError("failed to choose distinct pieces")
        try:
            piece = _random_connected_mask()
        except ValueError:
            continue
        key = _canonical_shape_key(piece)
        if key in seen:
            continue
        seen.add(key)
        pieces.append(piece)
    return pieces


def _adjacent_empty_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    filled = set(map(tuple, np.argwhere(mask)))
    for r, c in filled:
        for dr, dc in DIRS4:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                cells.add((nr, nc))
    return list(cells)


def _add_connected_extras(mask: np.ndarray, *, min_extra: int = 1, max_extra: int = EXTRA_CELL_MAX) -> np.ndarray:
    out = mask.copy()
    extra_target = random.randint(min_extra, max_extra)
    for _ in range(extra_target):
        candidates = _adjacent_empty_cells(out)
        if not candidates:
            break
        out[random.choice(candidates)] = True
    return out


def _coords_from_mask(mask: np.ndarray, r0: int = 0, c0: int = 0) -> frozenset[tuple[int, int]]:
    rr, cc = np.nonzero(mask)
    return frozenset((r0 + int(r), c0 + int(c)) for r, c in zip(rr, cc))


def _path_between(a: tuple[int, int], b: tuple[int, int], first_axis: str) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    pts: list[tuple[int, int]] = []
    if first_axis == "h":
        c_step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + c_step, c_step):
            pts.append((r0, c))
        r_step = 1 if r1 >= r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            pts.append((r, c1))
        return pts
    r_step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + r_step, r_step):
        pts.append((r, c0))
    c_step = 1 if c1 >= c0 else -1
    for c in range(c0 + c_step, c1 + c_step, c_step):
        pts.append((r1, c))
    return pts


def _single_piece_patch(piece_mask: np.ndarray) -> tuple[np.ndarray, list[frozenset[tuple[int, int]]], np.ndarray]:
    variants = _unique_transforms(piece_mask)
    for _ in range(GEN_LOCAL_PATCH_ATTEMPTS):
        placed = random.choice(variants)
        h, w = map(int, placed.shape)
        patch_h = h + random.randint(1, 3)
        patch_w = w + random.randint(1, 3)
        r0 = random.randint(0, patch_h - h)
        c0 = random.randint(0, patch_w - w)
        patch = np.zeros((patch_h, patch_w), dtype=bool)
        patch[r0 : r0 + h, c0 : c0 + w] = placed
        intended = _coords_from_mask(placed, r0, c0)
        patch = _add_connected_extras(patch, min_extra=1)
        solutions, _ = _solve_piece_layout([piece_mask], patch, max_solutions=2)
        if len(solutions) == 1 and solutions[0][0] == intended and int(patch.sum()) > len(intended):
            return patch, [intended], placed
    raise RuntimeError("failed to build a unique single-piece hole patch")


def _two_piece_patch(
    piece_a: np.ndarray, piece_b: np.ndarray
) -> tuple[np.ndarray, list[frozenset[tuple[int, int]]], list[np.ndarray]]:
    variants_a = _unique_transforms(piece_a)
    variants_b = _unique_transforms(piece_b)
    for _ in range(GEN_GROUP_PATCH_ATTEMPTS):
        placed_a = random.choice(variants_a)
        placed_b = random.choice(variants_b)
        h1, w1 = map(int, placed_a.shape)
        h2, w2 = map(int, placed_b.shape)
        horizontal = random.random() < 0.5
        if horizontal:
            patch_h = max(h1, h2) + random.randint(1, 4)
            gap = random.randint(1, 3)
            patch_w = w1 + w2 + gap + random.randint(1, 4)
            r1 = random.randint(0, patch_h - h1)
            r2 = random.randint(0, patch_h - h2)
            c1 = random.randint(0, 1)
            c2 = random.randint(c1 + w1 + gap, patch_w - w2)
        else:
            patch_w = max(w1, w2) + random.randint(1, 4)
            gap = random.randint(1, 3)
            patch_h = h1 + h2 + gap + random.randint(1, 4)
            c1 = random.randint(0, patch_w - w1)
            c2 = random.randint(0, patch_w - w2)
            r1 = random.randint(0, 1)
            r2 = random.randint(r1 + h1 + gap, patch_h - h2)

        patch = np.zeros((patch_h, patch_w), dtype=bool)
        slice_a = (slice(r1, r1 + h1), slice(c1, c1 + w1))
        slice_b = (slice(r2, r2 + h2), slice(c2, c2 + w2))
        if bool(np.any(patch[slice_a] & placed_a)) or bool(np.any(patch[slice_b] & placed_b)):
            continue
        patch[slice_a] |= placed_a
        if bool(np.any(patch[slice_b] & placed_b)):
            continue
        patch[slice_b] |= placed_b

        intended_a = _coords_from_mask(placed_a, r1, c1)
        intended_b = _coords_from_mask(placed_b, r2, c2)
        best_pair = None
        best_dist = None
        for cell_a in intended_a:
            for cell_b in intended_b:
                dist = abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_pair = (cell_a, cell_b)
        assert best_pair is not None

        connector_ok = False
        for axis in ("h", "v"):
            path = _path_between(best_pair[0], best_pair[1], axis)[1:-1]
            if any(patch[r, c] for r, c in path):
                continue
            for r, c in path:
                patch[r, c] = True
            connector_ok = True
            break
        if not connector_ok:
            continue

        patch = _add_connected_extras(patch, min_extra=0, max_extra=2)
        solutions, _ = _solve_piece_layout([piece_a, piece_b], patch, max_solutions=2)
        if len(solutions) != 1:
            continue
        if solutions[0][0] == intended_a and solutions[0][1] == intended_b:
            return patch, [intended_a, intended_b], [placed_a, placed_b]
    raise RuntimeError("failed to build a unique shared hole patch")


def _pure_leftover_patch() -> np.ndarray:
    area = random.randint(1, PURE_HOLE_AREA_MAX)
    if area == 1:
        return np.array([[True]], dtype=bool)
    if random.random() < 0.5:
        return np.array([[True, True]], dtype=bool)
    return np.array([[True], [True]], dtype=bool)


def _build_hole_groups(piece_masks: list[np.ndarray]) -> tuple[list[dict], bool]:
    indices = list(range(len(piece_masks)))
    random.shuffle(indices)
    groups: list[list[int]] = []
    if len(indices) >= 2 and random.random() < 0.45:
        groups.append(indices[:2])
        indices = indices[2:]
    groups.extend([[idx] for idx in indices])
    random.shuffle(groups)

    hole_groups: list[dict] = []
    orientation_changed = False
    mirror_used = False
    for group in groups:
        if len(group) == 1:
            patch, intended_sets, placed_variant = _single_piece_patch(piece_masks[group[0]])
            placed_variants = [placed_variant]
        else:
            patch, intended_sets, placed_variants = _two_piece_patch(piece_masks[group[0]], piece_masks[group[1]])

        display_variants: list[np.ndarray] = []
        for piece_idx, placed_variant in zip(group, placed_variants):
            options = _unique_transforms(piece_masks[piece_idx])
            display_variant = random.choice(options)
            if (tuple(display_variant.shape), display_variant.tobytes()) != (
                tuple(placed_variant.shape),
                placed_variant.tobytes(),
            ):
                orientation_changed = True
            if _requires_mirror(display_variant, placed_variant):
                mirror_used = True
            display_variants.append(display_variant)

        hole_groups.append(
            {
                "piece_indices": group,
                "hole_patch": patch,
                "intended_sets": intended_sets,
                "placed_variants": placed_variants,
                "display_variants": display_variants,
            }
        )

    return hole_groups, bool(orientation_changed and (mirror_used or random.random() < 0.5))


def _place_hole_patches(inner_shape: tuple[int, int], patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    canvas = np.zeros(inner_shape, dtype=int)
    positions: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: int(np.count_nonzero(patches[idx])), reverse=True)
    for idx in order:
        patch = patches[idx].astype(int)
        try:
            r0, c0 = random_free_location(canvas, patch, transparent=0, padding=1)
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=0)
        positions[idx] = (int(r0), int(c0))
    return positions


def _render_container_patch(
    container_h: int,
    container_w: int,
    container_color: int,
    hole_groups: list[dict],
    pure_patches: list[np.ndarray],
    positions: list[tuple[int, int]],
) -> tuple[np.ndarray, list[set[tuple[int, int]]], np.ndarray]:
    inner_h = container_h - 2
    inner_w = container_w - 2
    hole_inner = np.zeros((inner_h, inner_w), dtype=bool)
    placed_sets_global: list[set[tuple[int, int]]] = []

    for group, (r0, c0) in zip(hole_groups, positions[: len(hole_groups)]):
        patch = group["hole_patch"]
        hole_inner[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch
        for intended in group["intended_sets"]:
            placed_sets_global.append({(r0 + r, c0 + c) for r, c in intended})

    for patch, (r0, c0) in zip(pure_patches, positions[len(hole_groups) :]):
        hole_inner[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] |= patch

    container_patch = np.full((container_h, container_w), int(container_color), dtype=int)
    inner = np.full((inner_h, inner_w), int(container_color), dtype=int)
    inner[hole_inner] = BG
    container_patch[1:-1, 1:-1] = inner
    return container_patch, placed_sets_global, hole_inner


def _place_container_and_pieces(
    container_patch: np.ndarray,
    hole_groups: list[dict],
    placed_sets_inner: list[set[tuple[int, int]]],
    hole_inner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    container_h, container_w = container_patch.shape
    total_piece_extent = sum(max(mask.shape) for group in hole_groups for mask in group["display_variants"])

    h_min = container_h + GRID_MARGIN_MIN * 2
    h_max = min(30, container_h + GRID_MARGIN_MAX * 2 + max(4, total_piece_extent // 2))
    w_min = container_w + GRID_MARGIN_MIN * 2
    w_max = min(30, container_w + GRID_MARGIN_MAX * 2 + max(5, total_piece_extent))
    if h_min > h_max or w_min > w_max:
        return None

    H = random.randint(h_min, h_max)
    W = random.randint(w_min, w_max)
    if H > 30 or W > 30:
        return None

    inp = np.full((H, W), BG, dtype=int)
    container_r = random.randint(1, H - container_h - 1)
    container_c = random.randint(1, W - container_w - 1)
    paste(inp, container_patch, container_r, container_c)

    blocked = np.zeros((H, W), dtype=int)
    blocked[
        container_r - 1 : container_r + container_h + 1,
        container_c - 1 : container_c + container_w + 1,
    ] = 1

    piece_locations: list[tuple[np.ndarray, int, int]] = []
    for group in hole_groups:
        for display_mask in group["display_variants"]:
            patch = display_mask.astype(int)
            try:
                r0, c0 = random_free_location(blocked, patch, transparent=0, padding=1)
            except ValueError:
                return None
            paste(blocked, patch, r0, c0, transparent=0)
            piece_locations.append((display_mask, int(r0), int(c0)))

    for display_mask, r0, c0 in piece_locations:
        patch = np.where(display_mask, PIECE_COLOR, BG)
        paste(inp, patch, r0, c0, transparent=BG)

    out = inp.copy()
    out[out == PIECE_COLOR] = BG

    hole_global = np.zeros((H, W), dtype=bool)
    hole_global[
        container_r + 1 : container_r + 1 + hole_inner.shape[0],
        container_c + 1 : container_c + 1 + hole_inner.shape[1],
    ] = hole_inner
    out[hole_global] = ANSWER_COLOR

    placed_global = [
        {(container_r + 1 + r, container_c + 1 + c) for r, c in placed_set}
        for placed_set in placed_sets_inner
    ]
    for placed_set in placed_global:
        for r, c in placed_set:
            out[r, c] = int(container_patch[0, 0])

    return inp, out


def _generator_has_required_variation(hole_groups: list[dict], allow_soft_pass: bool) -> bool:
    changed = any(
        (tuple(display.shape), display.tobytes()) != (tuple(placed.shape), placed.tobytes())
        for group in hole_groups
        for display, placed in zip(group["display_variants"], group["placed_variants"])
    )
    mirror = any(
        _requires_mirror(display, placed)
        for group in hole_groups
        for display, placed in zip(group["display_variants"], group["placed_variants"])
    )
    return changed and (mirror or allow_soft_pass)


def _gen_attempt() -> np.ndarray | None:
    try:
        piece_masks = _choose_distinct_pieces()
        hole_groups, allow_soft_pass = _build_hole_groups(piece_masks)
    except RuntimeError:
        return None
    if not _generator_has_required_variation(hole_groups, allow_soft_pass):
        return None

    pure_patches = [_pure_leftover_patch() for _ in range(random.randint(*PURE_HOLE_COUNT_RANGE))]
    all_patches = [group["hole_patch"] for group in hole_groups] + pure_patches

    min_inner_h = max(patch.shape[0] for patch in all_patches) + 1
    min_inner_w = max(patch.shape[1] for patch in all_patches) + 1
    container_h_lo = max(CONTAINER_H_RANGE[0], min_inner_h + 2)
    container_w_lo = max(CONTAINER_W_RANGE[0], min_inner_w + 2)
    if container_h_lo > CONTAINER_H_RANGE[1] or container_w_lo > CONTAINER_W_RANGE[1]:
        return None

    container_h = random.randint(container_h_lo, CONTAINER_H_RANGE[1])
    container_w = random.randint(container_w_lo, CONTAINER_W_RANGE[1])
    inner_shape = (container_h - 2, container_w - 2)
    positions = _place_hole_patches(inner_shape, all_patches)
    if positions is None:
        return None

    container_color = random.choice(CONTAINER_COLORS)
    container_patch, placed_sets_inner, hole_inner = _render_container_patch(
        container_h,
        container_w,
        container_color,
        hole_groups,
        pure_patches,
        positions,
    )
    if int(np.count_nonzero(hole_inner)) <= sum(len(placed) for placed in placed_sets_inner):
        return None

    rendered = _place_container_and_pieces(container_patch, hole_groups, placed_sets_inner, hole_inner)
    if rendered is None:
        return None
    inp, out = rendered

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
