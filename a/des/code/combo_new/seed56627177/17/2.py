from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
MAX_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    assert points.size > 0
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    points = tuple(tuple(map(int, pt)) for pt in np.argwhere(mask))
    return tuple(map(int, mask.shape)), points


def _apply_transform(mask: np.ndarray, op: int) -> np.ndarray:
    out = np.rot90(mask, int(op % 4))
    if op >= 4:
        out = np.fliplr(out)
    return _crop_mask(out)


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return min(_mask_key(_apply_transform(mask, op)) for op in range(8))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return components


def _recolor_component(out: np.ndarray, comp: dict[str, object], first: int, second: int) -> None:
    r0, c0, _, _ = comp["bbox"]
    mask = comp["mask"]
    index = 0
    for rr in range(mask.shape[0]):
        for cc in range(mask.shape[1]):
            if not bool(mask[rr, cc]):
                continue
            out[int(r0) + rr, int(c0) + cc] = int(first if index % 2 == 0 else second)
            index += 1
    assert index == int(mask.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair matching scattered shapes and weave each matched color pair through both outlines.

    Input structure:
    - The background is a single dominant color.
    - Several separated 4-connected monochrome regions are scattered on the grid.
    - Each geometric shape appears exactly twice, with the two copies using different colors and possibly different rotations or mirror flips.

    Rule:
    - Group regions into pairs by shape, treating rotations and mirror flips as equivalent.
    - For each paired region A/B with colors cA and cB, keep both regions in place.
    - Refill region A by scanning only its own pixels in row-major order inside its bounding box, alternating cA then cB.
    - Refill region B the same way, but start with cB and then alternate cB/cA.
    - Leave every background cell unchanged.

    Key insight:
    - The task is two-stage: first recover the pairing by geometry rather than color or position, then apply a scan-order two-color interleave inside each matched silhouette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    components = _find_components(g, bg)
    assert components and len(components) % 2 == 0

    groups: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        groups[_canonical_key(comp["mask"])].append(comp)

    assert all(len(group) == 2 for group in groups.values())

    out = g.copy()
    for pair in groups.values():
        first, second = pair
        c_first = int(first["color"])
        c_second = int(second["color"])
        assert c_first != c_second
        _recolor_component(out, first, c_first, c_second)
        _recolor_component(out, second, c_second, c_first)

    return out


def _random_shape_mask() -> np.ndarray:
    for _ in range(300):
        height = int(RNG.integers(2, 6))
        width = int(RNG.integers(2, 6))
        full_area = height * width
        area_min = max(4, (full_area + 2) // 3)
        area_max = min(full_area - 1, 9)
        if area_min > area_max:
            continue

        target_area = int(RNG.integers(area_min, area_max + 1))
        start = (int(RNG.integers(height)), int(RNG.integers(width)))
        cells = {start}

        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            weights: list[int] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        if (nr, nc) not in frontier:
                            frontier.append((nr, nc))
                            adjacency = 0
                            for ar, ac in NEIGHBORS4:
                                if (nr + ar, nc + ac) in cells:
                                    adjacency += 1
                            weights.append(adjacency * adjacency + 1)
            if not frontier:
                break

            probs = np.asarray(weights, dtype=float)
            probs /= probs.sum()
            pick = int(RNG.choice(len(frontier), p=probs))
            cells.add(frontier[pick])

        if len(cells) != target_area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2 or bool(mask.all()):
            continue

        rotation_keys = {_mask_key(_apply_transform(mask, op)) for op in range(4)}
        transform_keys = {_mask_key(_apply_transform(mask, op)) for op in range(8)}
        if len(rotation_keys) < 2 or len(transform_keys) < 4:
            continue

        return mask

    raise RuntimeError("failed to generate a varied shape")


def _choose_pair_masks(mask: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    variants = {op: _apply_transform(mask, op) for op in range(8)}
    keys = {op: _mask_key(var) for op, var in variants.items()}
    op1 = int(RNG.integers(0, 8))
    key1 = keys[op1]

    if mode == "same":
        candidates = [op for op in range(8) if keys[op] == key1]
    elif mode == "rotate":
        candidates = [op for op in range(8) if op // 4 == op1 // 4 and keys[op] != key1]
    else:
        candidates = [op for op in range(8) if op // 4 != op1 // 4 and keys[op] != key1]

    if not candidates:
        candidates = [op for op in range(8) if keys[op] != key1]
    if not candidates:
        candidates = [op1]

    op2 = int(candidates[int(RNG.integers(len(candidates)))])
    return variants[op1], variants[op2]


def _pair_modes(pair_count: int) -> list[str]:
    base = ["same", "rotate", "reflect"]
    if pair_count == 2:
        modes = list(RNG.choice(base, size=2, replace=False))
    else:
        modes = list(RNG.permutation(base))
        while len(modes) < pair_count:
            modes.append(base[int(RNG.integers(0, len(base)))])
        RNG.shuffle(modes)
    return modes


def _can_place(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > blocked.shape[0] or left + w > blocked.shape[1]:
        return False
    return not bool(np.any(blocked[top : top + h, left : left + w] & mask))


def _stamp_mask(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)
    grid[top : top + h, left : left + w] = patch
    blocked[max(0, top - 1) : min(blocked.shape[0], top + h + 1), max(0, left - 1) : min(blocked.shape[1], left + w + 1)] = True


def _validate_generated(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    if bg != 0:
        return False

    components = _find_components(grid, bg)
    if len(components) < 4 or len(components) % 2 != 0:
        return False

    colors = [int(comp["color"]) for comp in components]
    if len(set(colors)) != len(colors):
        return False

    groups: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        groups[_canonical_key(comp["mask"])].append(comp)
    if any(len(group) != 2 for group in groups.values()):
        return False

    labels = np.full(grid.shape, -1, dtype=int)
    for idx, comp in enumerate(components):
        for r, c in comp["coords"]:
            labels[int(r), int(c)] = int(idx)
    for r, c in np.argwhere(labels >= 0):
        here = int(labels[r, c])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    other = int(labels[nr, nc])
                    if other >= 0 and other != here:
                        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out != grid):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.integers(2, 5))
    colors = list(map(int, RNG.choice(np.arange(1, 10), size=2 * pair_count, replace=False)))
    modes = _pair_modes(pair_count)

    used_keys: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    items: list[dict[str, object]] = []

    for pair_id in range(pair_count):
        for _ in range(200):
            base_mask = _random_shape_mask()
            key = _canonical_key(base_mask)
            if key in used_keys:
                continue
            used_keys.add(key)

            mask_a, mask_b = _choose_pair_masks(base_mask, modes[pair_id])
            color_a = colors[2 * pair_id]
            color_b = colors[2 * pair_id + 1]
            items.append({"mask": mask_a, "color": color_a, "area": int(mask_a.sum()), "tie": float(RNG.random())})
            items.append({"mask": mask_b, "color": color_b, "area": int(mask_b.sum()), "tie": float(RNG.random())})
            break
        else:
            return None

    max_h = max(int(item["mask"].shape[0]) for item in items)
    max_w = max(int(item["mask"].shape[1]) for item in items)
    min_side = max(12 + 2 * pair_count, max_h + 4, max_w + 4)
    if min_side > 30:
        return None

    height = int(RNG.integers(min_side, 31))
    width = int(RNG.integers(min_side, 31))
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    place_order = sorted(items, key=lambda item: (-int(item["area"]), -max(item["mask"].shape), float(item["tie"])))
    for item in place_order:
        mask = item["mask"]
        h, w = mask.shape
        placed = False
        for _ in range(400):
            top = int(RNG.integers(0, height - h + 1))
            left = int(RNG.integers(0, width - w + 1))
            if not _can_place(blocked, mask, top, left):
                continue
            _stamp_mask(grid, blocked, mask, top, left, int(item["color"]))
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
