from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 0
PALETTE = np.arange(1, 10, dtype=int)
NEI4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_SHAPE_SIDE = 5
MIN_SHAPE_AREA = 4
MAX_SHAPE_AREA = 7
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


@dataclass(frozen=True)
class Obj:
    r0: int
    c0: int
    mask: np.ndarray
    color: int


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    tight = _trim_mask(np.asarray(mask, dtype=bool))
    return (int(tight.shape[0]), int(tight.shape[1]), tight.astype(np.uint8).tobytes())


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    arr = _trim_mask(np.asarray(mask, dtype=bool))
    variants = []
    for base in (arr, np.fliplr(arr)):
        for k in range(4):
            variants.append(_trim_mask(np.rot90(base, k)).copy())
    return variants


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[int, int, bytes]] = set()
    out: list[np.ndarray] = []
    for variant in _all_transforms(mask):
        key = _mask_key(variant)
        if key not in seen:
            seen.add(key)
            out.append(variant.copy())
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_key(variant) for variant in _all_transforms(mask))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    fg = np.asarray(mask, dtype=bool)
    h, w = fg.shape
    seen = np.zeros_like(fg, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEI4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _extract_objects(grid: np.ndarray) -> list[Obj]:
    g = np.asarray(grid, dtype=int)
    objects: list[Obj] = []
    for coords in _connected_components(g != BG):
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        colors = set()
        for r, c in coords:
            mask[int(r) - r0, int(c) - c0] = True
            colors.add(int(g[int(r), int(c)]))
        assert len(colors) == 1, "objects must be monochrome"
        objects.append(Obj(r0=r0, c0=c0, mask=mask, color=next(iter(colors))))
    return objects


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    target[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _extract_objects(g)
    if not objects:
        return np.zeros_like(g)

    grouped: dict[tuple[int, int, bytes], list[Obj]] = {}
    for obj in objects:
        key = _canonical_shape_key(obj.mask)
        grouped.setdefault(key, []).append(obj)

    groups = []
    for members in grouped.values():
        leader = min(members, key=lambda obj: (obj.r0, obj.c0))
        groups.append(
            {
                "leader_pos": (leader.r0, leader.c0),
                "color": int(leader.color),
                "members": sorted(members, key=lambda obj: (obj.c0, obj.r0)),
            }
        )

    groups.sort(key=lambda group: group["leader_pos"])
    band_edges = [h * i // len(groups) for i in range(len(groups) + 1)]

    out = np.full_like(g, BG)
    for band_idx, group in enumerate(groups):
        band_r0 = int(band_edges[band_idx])
        band_r1 = int(band_edges[band_idx + 1])
        band_h = int(band_r1 - band_r0)
        members = group["members"]
        compact_w = sum(int(obj.mask.shape[1]) for obj in members) + len(members) - 1
        assert compact_w <= w, "group does not fit horizontally"
        start_c = (w - compact_w) // 2
        cursor = int(start_c)
        for obj in members:
            obj_h, obj_w = map(int, obj.mask.shape)
            assert obj_h <= band_h, "object does not fit inside its band"
            dest_r0 = int(band_r0 + (band_h - obj_h) // 2)
            _paint_mask(out, obj.mask, dest_r0, cursor, int(group["color"]))
            cursor += obj_w + 1

    return out


def _random_shape(area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEI4:
                    candidate = (r + dr, c + dc)
                    if candidate not in cells:
                        frontier.append(candidate)
            if not frontier:
                break
            nr, nc = frontier[int(_RNG.integers(len(frontier)))]
            cells.add((nr, nc))

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r_min, c_min = min(rs), min(cs)
        normalized = {(r - r_min, c - c_min) for r, c in cells}
        h = max(r for r, _ in normalized) + 1
        w = max(c for _, c in normalized) + 1
        if h < 2 or w < 2 or h > MAX_SHAPE_SIDE or w > MAX_SHAPE_SIDE:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in normalized:
            mask[r, c] = True
        if mask.sum() != area:
            continue
        if area == h * w and _RNG.random() < 0.85:
            continue
        if len(_unique_transforms(mask)) < 2:
            continue
        return mask
    return None


def _choose_orientation_masks(base_mask: np.ndarray, count: int) -> list[np.ndarray]:
    variants = _unique_transforms(base_mask)
    indices = [int(_RNG.integers(len(variants))) for _ in range(count)]
    if count > 1 and len(variants) > 1 and len(set(indices)) == 1:
        indices[-1] = (indices[-1] + 1) % len(variants)
    return [variants[idx].copy() for idx in indices]


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    rows, cols = mask.shape
    for dr in range(rows):
        for dc in range(cols):
            if not mask[dr, dc]:
                continue
            rr, cc = r0 + dr, c0 + dc
            for nr in range(max(0, rr - 1), min(blocked.shape[0], rr + 2)):
                for nc in range(max(0, cc - 1), min(blocked.shape[1], cc + 2)):
                    blocked[nr, nc] = True


def _can_place(mask: np.ndarray, blocked: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > blocked.shape[0] or c0 + w > blocked.shape[1]:
        return False
    target = blocked[r0 : r0 + h, c0 : c0 + w]
    return not np.any(target & mask)


def _candidate_positions(
    mask: np.ndarray,
    blocked: np.ndarray,
    row_min: int,
    row_max: int,
    condition=None,
) -> list[tuple[int, int]]:
    h, w = mask.shape
    upper = min(row_max, blocked.shape[0] - h)
    if upper < row_min:
        return []
    candidates = []
    for r0 in range(max(0, row_min), upper + 1):
        for c0 in range(0, blocked.shape[1] - w + 1):
            if condition is not None and not condition(r0, c0):
                continue
            if _can_place(mask, blocked, r0, c0):
                candidates.append((r0, c0))
    return candidates


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    row_min: int,
    row_max: int,
    condition=None,
) -> tuple[int, int] | None:
    candidates = _candidate_positions(mask, blocked, row_min, row_max, condition=condition)
    if not candidates:
        return None
    r0, c0 = candidates[int(_RNG.integers(len(candidates)))]
    _paint_mask(grid, mask, r0, c0, color)
    _mark_blocked(blocked, mask, r0, c0)
    return int(r0), int(c0)


def _build_group_specs() -> list[dict] | None:
    group_count = int(_RNG.integers(2, 5))
    object_counts = [int(_RNG.integers(2, 5)) for _ in range(group_count)]

    duplicate_area = int(_RNG.integers(MIN_SHAPE_AREA, MAX_SHAPE_AREA + 1))
    areas = [duplicate_area, duplicate_area]
    while len(areas) < group_count:
        areas.append(int(_RNG.integers(MIN_SHAPE_AREA, MAX_SHAPE_AREA + 1)))
    _RNG.shuffle(areas)

    leader_colors = list(_RNG.choice(PALETTE, size=group_count, replace=False))
    specs = []
    seen_keys: set[tuple[int, int, bytes]] = set()
    for area, count, leader_color in zip(areas, object_counts, leader_colors):
        base_mask = _random_shape(area)
        if base_mask is None:
            return None
        key = _canonical_shape_key(base_mask)
        tries = 0
        while key in seen_keys:
            base_mask = _random_shape(area)
            if base_mask is None:
                return None
            key = _canonical_shape_key(base_mask)
            tries += 1
            if tries > 100:
                return None
        seen_keys.add(key)

        masks = _choose_orientation_masks(base_mask, count)
        other_pool = [int(color) for color in PALETTE if int(color) != int(leader_color)]
        other_colors = list(_RNG.choice(other_pool, size=count - 1, replace=True))
        objects = [{"mask": masks[0], "color": int(leader_color), "leader": True}]
        for mask, color in zip(masks[1:], other_colors):
            objects.append({"mask": mask, "color": int(color), "leader": False})
        specs.append({"area": area, "objects": objects})

    _RNG.shuffle(specs)
    return specs


def _choose_grid_shape(specs: list[dict]) -> tuple[int, int] | None:
    group_count = len(specs)
    total_objects = sum(len(spec["objects"]) for spec in specs)
    max_obj_h = max(int(obj["mask"].shape[0]) for spec in specs for obj in spec["objects"])
    max_compact_w = max(
        sum(int(obj["mask"].shape[1]) for obj in spec["objects"]) + len(spec["objects"]) - 1
        for spec in specs
    )

    band_base = max_obj_h + int(_RNG.integers(2, 5))
    h = max(band_base * group_count + int(_RNG.integers(0, group_count + 1)), 12 + total_objects)
    h = min(h, 32)
    w = max(max_compact_w + int(_RNG.integers(2, 8)), 8 + total_objects)
    w = min(w, 32)

    if h // group_count < max_obj_h or w < max_compact_w:
        return None
    return int(h), int(w)


def _validate_generated(grid: np.ndarray) -> bool:
    objects = _extract_objects(grid)
    grouped: dict[tuple[int, int, bytes], list[Obj]] = {}
    for obj in objects:
        grouped.setdefault(_canonical_shape_key(obj.mask), []).append(obj)

    if not (2 <= len(grouped) <= 4):
        return False
    if not all(2 <= len(members) <= 4 for members in grouped.values()):
        return False

    areas = [int(next(iter(members)).mask.sum()) for members in grouped.values()]
    if len(set(areas)) == len(areas):
        return False

    has_orientation_variation = False
    for members in grouped.values():
        colors = {int(obj.color) for obj in members}
        if len(colors) < 2:
            return False
        local_shapes = {_mask_key(obj.mask) for obj in members}
        if len(local_shapes) > 1:
            has_orientation_variation = True
        leader = min(members, key=lambda obj: (obj.r0, obj.c0))
        for obj in members:
            if (obj.r0, obj.c0) == (leader.r0, leader.c0):
                continue
            if obj.r0 < leader.r0:
                return False
            if obj.r0 == leader.r0 and obj.c0 < leader.c0:
                return False
    if not has_orientation_variation:
        return False

    out = main(grid)
    return out.shape == grid.shape and np.any(out != grid)


def _gen_attempt() -> np.ndarray | None:
    specs = _build_group_specs()
    if specs is None:
        return None
    grid_shape = _choose_grid_shape(specs)
    if grid_shape is None:
        return None
    h, w = grid_shape

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    leader_order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["objects"][0]["mask"].sum()),
        reverse=True,
    )
    leader_positions: dict[int, tuple[int, int]] = {}
    for idx in leader_order:
        leader = specs[idx]["objects"][0]
        mask = leader["mask"]
        max_row = max(0, h // 2 - int(mask.shape[0]))
        placed = _place_object(
            grid,
            blocked,
            mask,
            int(leader["color"]),
            row_min=0,
            row_max=max_row,
        )
        if placed is None:
            return None
        leader_positions[idx] = placed

    pending = []
    for idx, spec in enumerate(specs):
        leader_r, leader_c = leader_positions[idx]
        for obj in spec["objects"][1:]:
            pending.append(
                (
                    -int(obj["mask"].sum()),
                    leader_r,
                    leader_c,
                    idx,
                    obj,
                )
            )
    pending.sort(key=lambda item: item[:3])

    for _neg_area, leader_r, leader_c, idx, obj in pending:
        mask = obj["mask"]

        def after_leader(r0: int, c0: int, base_r: int = leader_r, base_c: int = leader_c) -> bool:
            return (r0 > base_r) or (r0 == base_r and c0 > base_c)

        placed = _place_object(
            grid,
            blocked,
            mask,
            int(obj["color"]),
            row_min=leader_r,
            row_max=h - int(mask.shape[0]),
            condition=after_leader,
        )
        if placed is None:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
