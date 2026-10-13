from __future__ import annotations

import math
from collections import deque

import numpy as np


BG_COLOR = 0
ANCHOR_COLOR = 2
PALETTE = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)

DIR_COUNT = 8
DIR_STEP = math.pi / 4.0
DIR_HALF_STEP = math.pi / 8.0
INPUT_ANGLE_MARGIN = 0.05
MAX_GENERATION_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 200

_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _extract_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue

            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            patch_h = int(r1 - r0 + 1)
            patch_w = int(c1 - c0 + 1)
            mask = np.zeros((patch_h, patch_w), dtype=bool)
            for rr, cc in coords:
                mask[int(rr - r0), int(cc - c0)] = True
            patch = np.full((patch_h, patch_w), int(bg), dtype=int)
            patch[mask] = int(color)

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "patch": patch,
                    "area": int(len(coords)),
                }
            )

    return components


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (0.5 * (r0 + r1), 0.5 * (c0 + c1))


def _choose_anchor(components: list[dict], shape: tuple[int, int]) -> tuple[int, dict]:
    h, w = shape
    mid_r = 0.5 * (h - 1)
    mid_c = 0.5 * (w - 1)

    candidates: list[tuple[float, int, int, int, int]] = []
    for idx, comp in enumerate(components):
        r0, c0, r1, c1 = comp["bbox"]
        if int(comp["area"]) != 4:
            continue
        if int(r1 - r0 + 1) != 2 or int(c1 - c0 + 1) != 2:
            continue
        if not bool(comp["mask"].all()):
            continue
        ctr_r, ctr_c = _bbox_center(comp["bbox"])
        dist = abs(ctr_r - mid_r) + abs(ctr_c - mid_c)
        color_penalty = 0 if int(comp["color"]) == ANCHOR_COLOR else 1
        candidates.append((float(dist), int(color_penalty), int(r0), int(c0), int(idx)))

    assert candidates, "expected a central 2x2 anchor block"
    _, _, _, _, best_idx = min(candidates)
    return int(best_idx), components[int(best_idx)]


def _angle_from_anchor(
    bbox: tuple[int, int, int, int], anchor_center: tuple[float, float]
) -> float:
    center_r, center_c = _bbox_center(bbox)
    dr = float(center_r - anchor_center[0])
    dc = float(center_c - anchor_center[1])
    assert not (abs(dr) < 1e-9 and abs(dc) < 1e-9), "object center must differ from anchor center"
    angle = math.atan2(dc, -dr)
    if angle < 0:
        angle += 2.0 * math.pi
    return float(angle)


def _direction_index(
    bbox: tuple[int, int, int, int], anchor_center: tuple[float, float]
) -> int:
    angle = _angle_from_anchor(bbox, anchor_center)
    return int(math.floor(angle / DIR_STEP + 0.5)) % DIR_COUNT


def _angle_margin(angle: float, direction: int) -> float:
    target = float(direction) * DIR_STEP
    delta = (angle - target + math.pi) % (2.0 * math.pi) - math.pi
    return float(DIR_HALF_STEP - abs(delta))


def _center_aligned_start(anchor_center: float, size: int) -> int:
    return int(math.floor(anchor_center - (size - 1) / 2.0))


def _slot_top_left(
    slot_idx: int,
    patch_shape: tuple[int, int],
    anchor_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    anchor_r0, anchor_c0, anchor_r1, anchor_c1 = anchor_bbox
    patch_h, patch_w = patch_shape
    anchor_center_r, anchor_center_c = _bbox_center(anchor_bbox)

    if int(slot_idx) == 0:
        return (
            int(anchor_r0 - patch_h - 1),
            _center_aligned_start(anchor_center_c, int(patch_w)),
        )
    if int(slot_idx) == 1:
        return (int(anchor_r0 - patch_h - 1), int(anchor_c1 + 2))
    if int(slot_idx) == 2:
        return (
            _center_aligned_start(anchor_center_r, int(patch_h)),
            int(anchor_c1 + 2),
        )
    if int(slot_idx) == 3:
        return (int(anchor_r1 + 2), int(anchor_c1 + 2))
    if int(slot_idx) == 4:
        return (
            int(anchor_r1 + 2),
            _center_aligned_start(anchor_center_c, int(patch_w)),
        )
    if int(slot_idx) == 5:
        return (int(anchor_r1 + 2), int(anchor_c0 - patch_w - 1))
    if int(slot_idx) == 6:
        return (
            _center_aligned_start(anchor_center_r, int(patch_h)),
            int(anchor_c0 - patch_w - 1),
        )
    assert int(slot_idx) == 7
    return (int(anchor_r0 - patch_h - 1), int(anchor_c0 - patch_w - 1))


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    patch_h, patch_w = patch.shape
    assert 0 <= int(r0) <= int(grid.shape[0] - patch_h)
    assert 0 <= int(c0) <= int(grid.shape[1] - patch_w)
    mask = patch != int(bg)
    view = grid[int(r0) : int(r0) + patch_h, int(c0) : int(c0) + patch_w]
    assert not bool((view[mask] != int(bg)).any()), "objects must not overlap"
    view[mask] = patch[mask]


def _render_rearranged(
    shape: tuple[int, int],
    bg: int,
    anchor: dict,
    ordered_objects: list[dict],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    anchor_r0, anchor_c0, anchor_r1, anchor_c1 = anchor["bbox"]
    out[int(anchor_r0) : int(anchor_r1) + 1, int(anchor_c0) : int(anchor_c1) + 1] = int(
        anchor["color"]
    )

    for slot_idx, obj in enumerate(ordered_objects):
        top, left = _slot_top_left(int(slot_idx), obj["patch"].shape, anchor["bbox"])
        _paste_patch(out, obj["patch"], int(top), int(left), int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort foreground objects by their angular position around the central 2x2 anchor,
    then redraw them in canonical clockwise slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    components = _extract_components(g, bg)
    assert components, "expected foreground components"

    anchor_idx, anchor = _choose_anchor(components, g.shape)
    anchor_center = _bbox_center(anchor["bbox"])

    objects: list[dict] = []
    for idx, comp in enumerate(components):
        if int(idx) == int(anchor_idx):
            continue
        direction = _direction_index(comp["bbox"], anchor_center)
        objects.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "patch": comp["patch"].copy(),
                "direction": int(direction),
            }
        )

    assert 1 <= len(objects) <= DIR_COUNT, "expected between 1 and 8 objects"
    directions = [int(obj["direction"]) for obj in objects]
    assert len(set(directions)) == len(directions), "objects must occupy distinct compass sectors"

    ordered = sorted(objects, key=lambda obj: int(obj["direction"]))
    return _render_rearranged(g.shape, int(bg), anchor, ordered)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h = max(rows) + 1
    w = max(cols) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _random_shape_mask() -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target_area = int(_RNG.integers(2, 7))
        cells = {(0, 0)}

        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    frontier.add((int(r + dr), int(c + dc)))
            frontier.difference_update(cells)
            pick_list = list(frontier)
            chosen = pick_list[int(_RNG.integers(len(pick_list)))]
            cells.add(chosen)

        norm = _normalize_cells(cells)
        mask = _cells_to_mask(norm)
        if bool(_RNG.integers(0, 2)):
            mask = np.fliplr(mask)
        mask = np.rot90(mask, int(_RNG.integers(0, 4)))

        rows, cols = np.where(mask)
        cropped = mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]
        if cropped.shape[0] > 4 or cropped.shape[1] > 4:
            continue
        if cropped.shape == (2, 2) and bool(cropped.all()):
            continue
        return cropped.astype(bool)

    raise RuntimeError("failed to generate a bounded 4-connected shape")


def _patch_from_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _can_place_mask(occupied_dilated: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    return not bool((occupied_dilated[int(r0) : int(r0) + h, int(c0) : int(c0) + w] & mask).any())


def _mark_mask(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = mask.shape
    occupied[int(r0) : int(r0) + h, int(c0) : int(c0) + w] |= mask


def _sample_object_count() -> int:
    if float(_RNG.random()) < 0.5:
        return 8
    return int(_RNG.integers(2, 8))


def _sample_direction_set(count: int) -> list[int]:
    if int(count) == DIR_COUNT:
        return list(range(DIR_COUNT))

    require_compression = bool(_RNG.integers(0, 2))
    fallback: list[int] | None = None
    for _ in range(100):
        dirs = sorted(int(v) for v in _RNG.choice(DIR_COUNT, size=int(count), replace=False))
        fallback = dirs
        if not require_compression or dirs != list(range(int(count))):
            return dirs
    assert fallback is not None
    return fallback


def _make_object_specs(count: int, bg: int) -> list[dict]:
    colors = [int(v) for v in _RNG.choice(PALETTE, size=int(count), replace=False)]
    masks = [_random_shape_mask() for _ in range(int(count))]
    perm = [int(v) for v in _RNG.permutation(int(count))]
    directions = _sample_direction_set(int(count))

    specs: list[dict] = []
    for idx, direction in enumerate(directions):
        mask = masks[int(perm[idx])]
        color = colors[int(idx)]
        specs.append(
            {
                "color": int(color),
                "direction": int(direction),
                "mask": mask,
                "patch": _patch_from_mask(mask, int(color), int(bg)),
                "area": int(mask.sum()),
            }
        )
    return specs


def _output_layout_ok(
    shape: tuple[int, int],
    bg: int,
    anchor: dict,
    ordered_objects: list[dict],
) -> bool:
    try:
        _render_rearranged(shape, int(bg), anchor, ordered_objects)
    except AssertionError:
        return False
    return True


def _candidate_positions(
    grid_shape: tuple[int, int],
    anchor_center: tuple[float, float],
    occupied_dilated: np.ndarray,
    spec: dict,
) -> list[tuple[int, int]]:
    grid_h, grid_w = grid_shape
    mask = spec["mask"]
    patch_h, patch_w = mask.shape
    direction = int(spec["direction"])

    candidates: list[tuple[int, int]] = []
    for r0 in range(int(grid_h - patch_h + 1)):
        for c0 in range(int(grid_w - patch_w + 1)):
            if not _can_place_mask(occupied_dilated, mask, int(r0), int(c0)):
                continue
            bbox = (int(r0), int(c0), int(r0 + patch_h - 1), int(c0 + patch_w - 1))
            angle = _angle_from_anchor(bbox, anchor_center)
            if int(_direction_index(bbox, anchor_center)) != direction:
                continue
            if _angle_margin(angle, direction) < INPUT_ANGLE_MARGIN:
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _place_input_objects(
    grid_shape: tuple[int, int],
    anchor: dict,
    specs: list[dict],
) -> list[dict] | None:
    occupied = np.zeros(grid_shape, dtype=bool)
    anchor_r0, anchor_c0, anchor_r1, anchor_c1 = anchor["bbox"]
    occupied[int(anchor_r0) : int(anchor_r1) + 1, int(anchor_c0) : int(anchor_c1) + 1] = True
    anchor_center = _bbox_center(anchor["bbox"])

    remaining = [dict(spec) for spec in specs]
    placed: list[dict] = []

    while remaining:
        occupied_dilated = _dilate8(occupied)
        best_idx = None
        best_positions: list[tuple[int, int]] | None = None

        for idx, spec in enumerate(remaining):
            positions = _candidate_positions(grid_shape, anchor_center, occupied_dilated, spec)
            if not positions:
                return None
            if best_positions is None or len(positions) < len(best_positions):
                best_idx = int(idx)
                best_positions = positions

        assert best_idx is not None and best_positions is not None
        spec = remaining.pop(int(best_idx))
        choice = best_positions[int(_RNG.integers(len(best_positions)))]
        r0, c0 = choice
        _mark_mask(occupied, spec["mask"], int(r0), int(c0))
        spec["bbox"] = (
            int(r0),
            int(c0),
            int(r0 + spec["mask"].shape[0] - 1),
            int(c0 + spec["mask"].shape[1] - 1),
        )
        placed.append(spec)

    return placed


def _render_input(shape: tuple[int, int], bg: int, anchor: dict, placed_objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    anchor_r0, anchor_c0, anchor_r1, anchor_c1 = anchor["bbox"]
    grid[int(anchor_r0) : int(anchor_r1) + 1, int(anchor_c0) : int(anchor_c1) + 1] = int(
        anchor["color"]
    )
    for spec in placed_objects:
        r0, c0, _, _ = spec["bbox"]
        _paste_patch(grid, spec["patch"], int(r0), int(c0), int(bg))
    return grid


def _expected_output(shape: tuple[int, int], bg: int, anchor: dict, placed_objects: list[dict]) -> np.ndarray:
    ordered = sorted(placed_objects, key=lambda obj: int(obj["direction"]))
    return _render_rearranged(shape, int(bg), anchor, ordered)


def _build_anchor(shape: tuple[int, int]) -> dict:
    h, w = shape
    r0 = int(h // 2 - 1)
    c0 = int(w // 2 - 1)
    return {
        "color": int(ANCHOR_COLOR),
        "bbox": (int(r0), int(c0), int(r0 + 1), int(c0 + 1)),
        "patch": np.full((2, 2), int(ANCHOR_COLOR), dtype=int),
    }


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(17, 26))
    grid_w = int(_RNG.integers(17, 26))
    shape = (int(grid_h), int(grid_w))
    bg = int(BG_COLOR)
    anchor = _build_anchor(shape)

    count = _sample_object_count()
    specs = _make_object_specs(int(count), int(bg))
    ordered_specs = sorted(specs, key=lambda obj: int(obj["direction"]))
    if not _output_layout_ok(shape, int(bg), anchor, ordered_specs):
        return None

    placed = _place_input_objects(shape, anchor, specs)
    if placed is None:
        return None

    grid = _render_input(shape, int(bg), anchor, placed)
    expected = _expected_output(shape, int(bg), anchor, placed)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
