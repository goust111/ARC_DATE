from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

PAIR_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (3, 6)
GRID_SIDE_MAX = 30

MASK_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 220
GRID_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400

DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _mask_connected8(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and bool(mask[nr, nc])
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _random_complementary_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    total = h * w
    min_area = max(4, (total * 3) // 10)
    max_area = min(total - 4, (total * 7) // 10)
    if min_area > max_area:
        return None

    for _ in range(MASK_ATTEMPTS):
        fill_prob = random.uniform(0.35, 0.65)
        mask = np.array(
            [[random.random() < fill_prob for _ in range(w)] for _ in range(h)],
            dtype=bool,
        )
        area = int(mask.sum())
        if not (min_area <= area <= max_area):
            continue
        comp = ~mask
        if not _touches_all_sides(mask) or not _touches_all_sides(comp):
            continue
        if not _mask_connected8(mask) or not _mask_connected8(comp):
            continue
        return mask, comp
    return None


def _extract_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or bool(seen[r, c]):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS8:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments with the same bbox dimensions, assemble each pair
    into a two-color rectangle, then sort the assembled rectangles by area.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG

    components = _extract_components(g, bg)
    assert components and len(components) % 2 == 0

    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        mask = comp["mask"]
        groups[tuple(map(int, mask.shape))].append(comp)

    assembled: list[dict[str, object]] = []
    for dims, group in groups.items():
        index_by_key: dict[bytes, list[int]] = defaultdict(list)
        for idx, comp in enumerate(group):
            index_by_key[np.asarray(comp["mask"], dtype=bool).tobytes()].append(idx)

        used: set[int] = set()
        order = sorted(
            range(len(group)),
            key=lambda idx: (
                int(group[idx]["bbox"][0]),
                int(group[idx]["bbox"][1]),
                int(group[idx]["color"]),
            ),
        )

        for idx in order:
            if idx in used:
                continue

            first = group[idx]
            first_mask = np.asarray(first["mask"], dtype=bool)
            partner_key = (~first_mask).tobytes()
            candidates = [
                j
                for j in index_by_key.get(partner_key, [])
                if j not in used and j != idx
            ]
            assert len(candidates) == 1

            partner = group[candidates[0]]
            assert int(first["color"]) != int(partner["color"])

            h, w = dims
            patch = np.full((h, w), bg, dtype=int)
            patch[first_mask] = int(first["color"])
            patch[np.asarray(partner["mask"], dtype=bool)] = int(partner["color"])

            source_key = min(
                (int(first["bbox"][0]), int(first["bbox"][1])),
                (int(partner["bbox"][0]), int(partner["bbox"][1])),
            )
            assembled.append(
                {
                    "patch": patch,
                    "h": h,
                    "w": w,
                    "area": h * w,
                    "source_key": source_key,
                }
            )
            used.add(idx)
            used.add(candidates[0])

        assert len(used) == len(group)

    assert len(assembled) * 2 == len(components)

    assembled.sort(
        key=lambda item: (
            int(item["area"]),
            int(item["h"]),
            int(item["w"]),
            item["source_key"],
        )
    )

    out_h = max(int(item["h"]) for item in assembled)
    out_w = sum(int(item["w"]) for item in assembled) + len(assembled) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for item in assembled:
        patch = np.asarray(item["patch"], dtype=int)
        h, w = patch.shape
        out[out_h - h : out_h, c : c + w] = patch
        c += w + 1

    return out


def _component_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _assembled_patch(mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    patch = np.full(mask.shape, int(color_b), dtype=int)
    patch[mask] = int(color_a)
    return patch


def _place_components(
    patches: list[np.ndarray],
    height: int,
    width: int,
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(
        key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1],
        reverse=True,
    )

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            r1, c1 = r0 + ph, c0 + pw
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r1 + 1)
            cc1 = min(width, c1 + 1)
            if bool(blocked[rr0:rr1, cc0:cc1].any()):
                continue

            mask = patch != BG
            grid_slice = grid[r0:r1, c0:c1]
            grid_slice[mask] = patch[mask]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _expected_output(pair_specs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(
        pair_specs,
        key=lambda item: (
            int(item["area"]),
            int(item["h"]),
            int(item["w"]),
        ),
    )
    out_h = max(int(item["h"]) for item in ordered)
    out_w = sum(int(item["w"]) for item in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for item in ordered:
        patch = np.asarray(item["assembled"], dtype=int)
        h, w = patch.shape
        out[out_h - h : out_h, c : c + w] = patch
        c += w + 1
    return out


def _sample_pair_specs() -> list[dict[str, object]] | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    used_areas: set[int] = set()
    colors = random.sample(PALETTE, 2 * pair_count)
    pair_specs: list[dict[str, object]] = []

    for pair_idx in range(pair_count):
        built = False
        for _ in range(MASK_ATTEMPTS):
            h = random.randint(*RECT_SIDE_RANGE)
            w = random.randint(*RECT_SIDE_RANGE)
            area = h * w
            if area in used_areas:
                continue
            masks = _random_complementary_masks(h, w)
            if masks is None:
                continue

            color_a = colors[2 * pair_idx]
            color_b = colors[2 * pair_idx + 1]
            mask_a, mask_b = masks
            pair_specs.append(
                {
                    "h": h,
                    "w": w,
                    "area": area,
                    "component_a": _component_patch(mask_a, color_a),
                    "component_b": _component_patch(mask_b, color_b),
                    "assembled": _assembled_patch(mask_a, color_a, color_b),
                }
            )
            used_areas.add(area)
            built = True
            break
        if not built:
            return None

    return pair_specs


def _generate_attempt() -> np.ndarray | None:
    pair_specs = _sample_pair_specs()
    if pair_specs is None:
        return None

    component_patches: list[np.ndarray] = []
    for spec in pair_specs:
        component_patches.append(np.asarray(spec["component_a"], dtype=int))
        component_patches.append(np.asarray(spec["component_b"], dtype=int))

    pair_count = len(pair_specs)
    side_low = 12 + 3 * pair_count
    max_patch_h = max(int(patch.shape[0]) for patch in component_patches)
    max_patch_w = max(int(patch.shape[1]) for patch in component_patches)
    side_low = max(side_low, max_patch_h + 4, max_patch_w + 4)
    if side_low > GRID_SIDE_MAX:
        return None

    expected = _expected_output(pair_specs)
    if expected.shape[0] > GRID_SIDE_MAX or expected.shape[1] > GRID_SIDE_MAX:
        return None

    for _ in range(GRID_ATTEMPTS):
        height = random.randint(side_low, GRID_SIDE_MAX)
        width = random.randint(side_low, GRID_SIDE_MAX)
        grid = _place_components(component_patches, height, width)
        if grid is None:
            continue
        if np.array_equal(main(grid), expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
