import random
from collections import deque

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CONTAINER_PALETTE = [1, 2, 3, 6, 7, 8, 9]
DISTRACTOR_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SMALL_SHAPES = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords), "area": len(cells)})
    return comps


def _u_patch(height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[:, 0] = color
    patch[:, -1] = color
    patch[-1, :] = color
    return patch


def _is_container(grid: np.ndarray, comp: dict) -> bool:
    h, _w = grid.shape
    r0, c0, r1, c1 = comp["bbox"]
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    if box_h < 3 or box_w < 3:
        return False
    if r1 < h // 2:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = _u_patch(box_h, box_w, int(comp["color"]))
    return bool(np.array_equal(patch, expected))


def _detect_containers(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    containers = []
    others = []
    for comp in _find_components(grid):
        if _is_container(grid, comp):
            r0, c0, r1, c1 = comp["bbox"]
            containers.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "inner_height": int(r1 - r0),
                    "inner_width": int(c1 - c0 - 1),
                }
            )
        else:
            others.append(comp)
    containers.sort(key=lambda item: item["bbox"][1])
    return containers, others


def _counts_by_color(objects: list[dict], colors: set[int]) -> dict[int, int]:
    counts = {color: 0 for color in colors}
    for obj in objects:
        color = int(obj["color"])
        if color in counts:
            counts[color] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count color-matched small objects and encode each count as water height inside the same-colored container.

    Input structure:
    - Black (0) background with two or three monochromatic U-shaped containers in the lower half.
    - Each container is open upward, has wall thickness 1, and encloses an empty black interior.
    - The upper area contains several small 4-connected monochromatic objects, some sharing container colors and some using distractor colors.

    Rule:
    - Detect every lower-half U-shaped container and read its color.
    - Count how many non-container 4-connected objects use each container color.
    - For each container, fill that many interior rows from the bottom upward with the container's own color.
    - Erase every non-container object whose color matches any container color, while leaving distractor-color objects unchanged.

    Key insight:
    - A container's color plays two coupled roles at once: it selects which upper objects to count and also determines where that count is spatially encoded as a vertical fill height.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers, other_objects = _detect_containers(g)
    if not containers:
        return out

    container_colors = {int(item["color"]) for item in containers}
    counts = _counts_by_color(other_objects, container_colors)

    for obj in other_objects:
        if int(obj["color"]) not in container_colors:
            continue
        rr = obj["coords"][:, 0]
        cc = obj["coords"][:, 1]
        out[rr, cc] = BG

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        color = int(container["color"])
        fill_rows = min(int(counts[color]), int(container["inner_height"]))
        for k in range(fill_rows):
            row = r1 - 1 - k
            out[row, c0 + 1 : c1] = color

    return out


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(gh, r0 + h + padding)
    cc1 = min(gw, c0 + w + padding)
    return not bool((grid[rr0:rr1, cc0:cc1] != BG).any())


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    h, w = patch.shape
    coords = []
    for dr in range(h):
        for dc in range(w):
            if int(patch[dr, dc]) == BG:
                continue
            grid[r0 + dr, c0 + dc] = int(patch[dr, dc])
            coords.append((r0 + dr, c0 + dc))
    return coords


def _random_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _sample_counts(capacities: list[int]) -> list[int] | None:
    n = len(capacities)
    total_limit = 6 if n == 2 else 7
    for _ in range(120):
        counts = [random.randint(1, cap) for cap in capacities]
        if random.random() < 0.45:
            idx = random.randrange(n)
            counts[idx] = capacities[idx]
        if sum(counts) > total_limit:
            continue
        if len(set(counts)) == 1:
            continue
        if max(counts) <= 1:
            continue
        return counts
    return None


def _container_layout() -> tuple[int, int, list[dict], list[int]] | None:
    n = random.choice([2, 2, 3])
    widths = [random.randint(3, 5) for _ in range(n)]
    heights = [random.randint(3, 5) for _ in range(n)]
    capacities = [h - 1 for h in heights]
    counts = _sample_counts(capacities)
    if counts is None:
        return None

    base_width = sum(widths) + 2 * (n - 1) + 2
    width = base_width + random.randint(0, 6)
    if width > 28:
        return None

    upper_height = random.randint(7, 11)
    height = upper_height + max(heights)
    if height > 24:
        return None

    extras = _random_partition(width - base_width, n + 1)
    gaps = [1 + extras[0]]
    for idx in range(n - 1):
        gaps.append(2 + extras[idx + 1])
    right_margin = 1 + extras[-1]

    containers = []
    col = gaps[0]
    colors = random.sample(CONTAINER_PALETTE, n)
    for idx in range(n):
        box_w = widths[idx]
        box_h = heights[idx]
        r0 = height - box_h
        c0 = col
        c1 = c0 + box_w - 1
        r1 = height - 1
        containers.append(
            {
                "color": colors[idx],
                "bbox": (r0, c0, r1, c1),
                "count": counts[idx],
                "inner_height": box_h - 1,
                "center_c": c0 + box_w // 2,
            }
        )
        if idx < n - 1:
            col = c1 + 1 + gaps[idx + 1]
    if containers[-1]["bbox"][3] + right_margin >= width:
        return None
    return height, width, containers, counts


def _object_patch(color: int) -> np.ndarray:
    shape = random.choice(SMALL_SHAPES)
    return (shape * color).astype(int)


def _candidate_positions(
    grid: np.ndarray,
    patch: np.ndarray,
    row_max: int,
    predicate=None,
) -> list[tuple[int, int]]:
    h, w = patch.shape
    gh, gw = grid.shape
    top_limit = min(row_max, gh - h - 1)
    candidates = []
    for r0 in range(1, top_limit + 1):
        for c0 in range(1, gw - w):
            if not _can_place(grid, patch, r0, c0, padding=1):
                continue
            if predicate is not None and not predicate(r0, c0):
                continue
            candidates.append((r0, c0))
    random.shuffle(candidates)
    return candidates


def _center_column(patch: np.ndarray, r0: int, c0: int) -> float:
    coords = np.argwhere(patch != BG)
    return float(c0 + coords[:, 1].mean())


def _metadata_output(grid: np.ndarray, containers: list[dict], objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    container_colors = {int(item["color"]) for item in containers}
    for obj in objects:
        if obj["matched"]:
            for r, c in obj["cells"]:
                out[r, c] = BG
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        color = int(container["color"])
        count = int(container["count"])
        for k in range(count):
            row = r1 - 1 - k
            out[row, c0 + 1 : c1] = color
        assert color in container_colors
    return out


def _misaligned_object_exists(objects: list[dict], containers: list[dict]) -> bool:
    by_color = {int(item["color"]): item for item in containers}
    centers = [float(item["center_c"]) for item in containers]
    for obj in objects:
        if not obj["matched"]:
            continue
        container = by_color[int(obj["color"])]
        center = float(np.mean([c for _r, c in obj["cells"]]))
        c0 = int(container["bbox"][1])
        c1 = int(container["bbox"][3])
        if not (c0 <= center <= c1):
            return True
        own_dist = abs(center - float(container["center_c"]))
        other_dist = min(abs(center - c) for c in centers if c != float(container["center_c"]))
        if other_dist < own_dist:
            return True
    return False


def _validate_scene(grid: np.ndarray, containers: list[dict], objects: list[dict]) -> bool:
    detected, others = _detect_containers(grid)
    if len(detected) != len(containers):
        return False

    placed = {(int(item["color"]), tuple(item["bbox"])) for item in containers}
    found = {(int(item["color"]), tuple(item["bbox"])) for item in detected}
    if placed != found:
        return False

    counts = _counts_by_color(others, {int(item["color"]) for item in detected})
    expected_counts = {int(item["color"]): int(item["count"]) for item in containers}
    if counts != expected_counts:
        return False

    if not any(not obj["matched"] for obj in objects):
        return False
    if not _misaligned_object_exists(objects, containers):
        return False

    out = main(grid)
    want = _metadata_output(grid, containers, objects)
    if not np.array_equal(out, want):
        return False

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        count = int(container["count"])
        color = int(container["color"])
        for row in range(r0, r1):
            filled = row >= r1 - count
            interior = out[row, c0 + 1 : c1]
            if filled and not np.all(interior == color):
                return False
            if not filled and not np.all(interior == BG):
                return False

    for obj in objects:
        cells = tuple(obj["cells"])
        vals = [int(out[r, c]) for r, c in cells]
        if obj["matched"]:
            if any(v != BG for v in vals):
                return False
        else:
            if any(v != int(obj["color"]) for v in vals):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _container_layout()
    if layout is None:
        return None
    height, width, containers, _counts = layout
    grid = np.zeros((height, width), dtype=int)

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = _u_patch(r1 - r0 + 1, c1 - c0 + 1, int(container["color"]))

    upper_row_max = min(item["bbox"][0] for item in containers) - 2
    if upper_row_max < 2:
        return None

    distractor_colors = [c for c in DISTRACTOR_PALETTE if c not in {int(item["color"]) for item in containers}]
    random.shuffle(distractor_colors)
    distractor_object_count = random.randint(1, 3)

    specs = []
    for container in containers:
        for _ in range(int(container["count"])):
            specs.append({"color": int(container["color"]), "matched": True})
    for _ in range(distractor_object_count):
        specs.append({"color": distractor_colors[random.randrange(min(2, len(distractor_colors)))], "matched": False})

    random.shuffle(specs)
    misaligned_color = random.choice(containers)["color"]
    misaligned_idx = next((i for i, spec in enumerate(specs) if spec["matched"] and spec["color"] == misaligned_color), None)
    if misaligned_idx is None:
        return None
    specs[0], specs[misaligned_idx] = specs[misaligned_idx], specs[0]

    objects = []
    container_by_color = {int(item["color"]): item for item in containers}
    centers = {int(item["color"]): float(item["center_c"]) for item in containers}

    for idx, spec in enumerate(specs):
        patch = _object_patch(int(spec["color"]))
        predicate = None
        if idx == 0 and spec["matched"] and len(containers) >= 2:
            own = container_by_color[int(spec["color"])]
            own_center = centers[int(spec["color"])]
            other_centers = [float(item["center_c"]) for item in containers if int(item["color"]) != int(spec["color"])]

            def predicate(r0: int, c0: int, patch=patch, own=own, own_center=own_center, other_centers=other_centers) -> bool:
                center = _center_column(patch, r0, c0)
                if own["bbox"][1] <= center <= own["bbox"][3]:
                    return False
                return min(abs(center - c) for c in other_centers) <= abs(center - own_center)

        candidates = _candidate_positions(grid, patch, upper_row_max, predicate=predicate)
        if not candidates:
            return None
        r0, c0 = candidates[0]
        cells = _place_patch(grid, patch, r0, c0)
        objects.append({"color": int(spec["color"]), "matched": bool(spec["matched"]), "cells": cells})

    if not _validate_scene(grid, containers, objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-counting scene")
