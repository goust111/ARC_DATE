from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

ARC_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 300
SLOT_SIZE = 7
PAIR_COUNT_OPTIONS = (2, 3)
AREA_CHOICES = (16, 20, 25)
AREA_TO_BASE_DIMS = {
    16: (4, 4),
    20: (4, 5),
    25: (5, 5),
}


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected(mask: np.ndarray, connectivity: int) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    h, w = mask.shape
    start = tuple(map(int, points[0]))
    stack = [start]
    seen = {start}

    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _touches_full_bbox(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _extract_components(
    grid: np.ndarray, bg: int, connectivity: int
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "anchor": (r0, c0),
                    "mask": mask,
                    "shape": mask.shape,
                }
            )

    return components


def _pair_components(components: list[dict[str, object]]) -> list[dict[str, object]]:
    assert components and len(components) % 2 == 0
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for comp in components:
        shape = tuple(map(int, comp["shape"]))
        grouped.setdefault(shape, []).append(comp)

    pairs: list[dict[str, object]] = []
    for shape, group in grouped.items():
        assert len(group) % 2 == 0
        remaining = list(range(len(group)))
        while remaining:
            i = remaining.pop(0)
            first = group[i]
            first_mask = np.asarray(first["mask"], dtype=bool)
            wanted = ~first_mask
            matches = [
                j
                for j in remaining
                if np.array_equal(np.asarray(group[j]["mask"], dtype=bool), wanted)
            ]
            assert len(matches) == 1
            j = matches[0]
            remaining.remove(j)
            second = group[j]
            second_mask = np.asarray(second["mask"], dtype=bool)
            h, w = shape
            patch = np.full((h, w), -1, dtype=int)
            patch[first_mask] = int(first["color"])
            patch[second_mask] = int(second["color"])
            assert np.all(patch >= 0)
            pair_anchor = min(
                tuple(map(int, first["anchor"])), tuple(map(int, second["anchor"]))
            )
            pairs.append(
                {
                    "area": int(h * w),
                    "h": int(h),
                    "w": int(w),
                    "patch": patch,
                    "anchor": pair_anchor,
                }
            )

    assert len(pairs) * 2 == len(components)
    pairs.sort(
        key=lambda item: (
            int(item["area"]),
            int(item["h"]),
            int(item["w"]),
            tuple(map(int, item["anchor"])),
        )
    )
    return pairs


def _render_output(pairs: list[dict[str, object]], bg: int) -> np.ndarray:
    out_h = sum(int(pair["h"]) for pair in pairs) + len(pairs) - 1
    out_w = max(int(pair["w"]) for pair in pairs)
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for pair in pairs:
        patch = np.asarray(pair["patch"], dtype=int)
        h, w = patch.shape
        out[row : row + h, :w] = patch
        row += h + 1

    return out


def _solve_with_connectivity(grid: np.ndarray, bg: int, connectivity: int) -> np.ndarray:
    components = _extract_components(grid, bg=bg, connectivity=connectivity)
    assert components and len(components) % 2 == 0
    return _render_output(_pair_components(components), bg=bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments into full rectangles and stack the completed rectangles by increasing area.

    Input structure:
    - A dominant background color fills most of the grid.
    - The non-background cells form an even number of separate monochrome fragments.
    - Each fragment has a bounding box, and paired fragments share the same bounding-box size.

    Rule:
    - Extract all colored fragments and compare only fragments whose bounding-box dimensions match.
    - A valid pair is the unique same-sized pair whose binary masks are exact complements inside that box, so together they fill a solid rectangle while keeping their original colors.
    - Reconstruct every paired rectangle, sort the rectangles by area from small to large, and place them top-to-bottom with one background row between rectangles and with left alignment.

    Key insight:
    - The pairing signal is geometric complementarity inside a shared bounding box, not color identity or original position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)

    last_error: AssertionError | None = None
    for connectivity in (4, 8):
        try:
            return _solve_with_connectivity(g, bg=bg, connectivity=connectivity)
        except AssertionError as exc:
            last_error = exc

    assert last_error is not None
    raise last_error


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {
        (start[0] + dr, start[1] + dc)
        for dr, dc in NEIGHBORS8
        if 0 <= start[0] + dr < h and 0 <= start[1] + dc < w
    }

    while int(mask.sum()) < area:
        if not frontier:
            return mask
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        r, c = cell
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int) -> tuple[np.ndarray, ...]:
    found: dict[bytes, np.ndarray] = {}
    min_area = max(4, (h * w) // 3)
    max_area = min(h * w - 4, (2 * h * w) // 3)

    for _ in range(12000):
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(h, w, area)
        comp = ~mask
        if not _touches_full_bbox(mask) or not _touches_full_bbox(comp):
            continue
        if not _connected(mask, 8) or not _connected(comp, 8):
            continue
        key = min(mask.tobytes(), comp.tobytes())
        if key not in found:
            found[key] = mask.copy()
        if len(found) >= 24:
            break

    if not found:
        raise RuntimeError(f"failed to find valid complementary masks for {(h, w)}")
    return tuple(found.values())


def _sample_complementary_masks(base_h: int, base_w: int) -> tuple[np.ndarray, np.ndarray]:
    mask = random.choice(_valid_masks(base_h, base_w)).copy()
    for _ in range(random.randint(0, 3)):
        mask = np.rot90(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return mask, ~mask


def _paint_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _build_expected(pair_patches: list[np.ndarray], bg: int) -> np.ndarray:
    ordered = sorted(
        pair_patches,
        key=lambda patch: (int(patch.shape[0] * patch.shape[1]), int(patch.shape[0]), int(patch.shape[1])),
    )
    out_h = sum(patch.shape[0] for patch in ordered) + len(ordered) - 1
    out_w = max(patch.shape[1] for patch in ordered)
    out = np.full((out_h, out_w), bg, dtype=int)
    row = 0
    for patch in ordered:
        h, w = patch.shape
        out[row : row + h, :w] = patch
        row += h + 1
    return out


def _choose_layout(component_count: int) -> tuple[int, int]:
    options = []
    for rows in (2, 3):
        cols = (component_count + rows - 1) // rows
        if rows * SLOT_SIZE + 4 <= ARC_LIMIT and cols * SLOT_SIZE + 4 <= ARC_LIMIT:
            options.append((rows, cols))
    return random.choice(options)


def _place_components(component_patches: list[np.ndarray], bg: int) -> np.ndarray:
    count = len(component_patches)
    rows, cols = _choose_layout(count)
    top = random.randint(1, 2)
    left = random.randint(1, 2)
    bottom = random.randint(1, 2)
    right = random.randint(1, 2)
    grid_h = top + rows * SLOT_SIZE + bottom
    grid_w = left + cols * SLOT_SIZE + right
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    slot_ids = list(range(rows * cols))
    chosen_slots = random.sample(slot_ids, count)
    random.shuffle(component_patches)

    for patch, slot_id in zip(component_patches, chosen_slots):
        slot_r = slot_id // cols
        slot_c = slot_id % cols
        h, w = patch.shape
        max_off_r = SLOT_SIZE - h - 1
        max_off_c = SLOT_SIZE - w - 1
        off_r = random.randint(0, max_off_r)
        off_c = random.randint(0, max_off_c)
        r0 = top + slot_r * SLOT_SIZE + off_r
        c0 = left + slot_c * SLOT_SIZE + off_c
        mask = patch != bg
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]

    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    areas = random.sample(list(AREA_CHOICES), random.choice(PAIR_COUNT_OPTIONS))
    colors = random.sample([c for c in range(10) if c != bg], 2 * len(areas))

    pair_patches: list[np.ndarray] = []
    components: list[np.ndarray] = []
    color_idx = 0
    for area in areas:
        base_h, base_w = AREA_TO_BASE_DIMS[area]
        mask_a, mask_b = _sample_complementary_masks(base_h, base_w)
        color_a = colors[color_idx]
        color_b = colors[color_idx + 1]
        color_idx += 2

        comp_a = _paint_mask(mask_a, color_a, bg)
        comp_b = _paint_mask(mask_b, color_b, bg)
        pair_patch = np.full(mask_a.shape, bg, dtype=int)
        pair_patch[mask_a] = color_a
        pair_patch[mask_b] = color_b

        pair_patches.append(pair_patch)
        components.extend([comp_a, comp_b])

    random.shuffle(components)
    inp = _place_components(components, bg=bg)
    expected = _build_expected(pair_patches, bg=bg)

    if inp.ndim != 2 or expected.ndim != 2:
        return None
    if max(inp.shape) > ARC_LIMIT or max(expected.shape) > ARC_LIMIT:
        return None
    if _dominant_color(inp) != bg:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
