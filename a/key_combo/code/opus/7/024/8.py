from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
CARDINALS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LOCAL_BOUND = 4
INPUT_ROWS = (14, 20)
INPUT_COLS = (16, 22)
MAX_GENERATION_ATTEMPTS = 400
MIN_ANGLE_GAP = math.radians(18)

ENDPOINTS = [
    (-3, 0),
    (-2, 2),
    (0, 3),
    (2, 2),
    (3, 0),
    (2, -2),
    (0, -3),
    (-2, -2),
]

COMPATIBLE_ENDPOINTS = {
    (-1, 0): [(-3, 0), (-2, 2), (-2, -2)],
    (0, 1): [(0, 3), (-2, 2), (2, 2)],
    (1, 0): [(3, 0), (2, 2), (2, -2)],
    (0, -1): [(0, -3), (-2, -2), (2, -2)],
}


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _angle_from_anchor(anchor: tuple[float, float], centroid: tuple[float, float]) -> float:
    dr = float(centroid[0] - anchor[0])
    dc = float(centroid[1] - anchor[1])
    return math.atan2(dc, -dr) % (2 * math.pi)


def _in_local_bounds(cell: tuple[int, int]) -> bool:
    return abs(int(cell[0])) <= LOCAL_BOUND and abs(int(cell[1])) <= LOCAL_BOUND


def _local_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_sprite_connected(local_cells: set[tuple[int, int]]) -> bool:
    nodes = {(0, 0), *local_cells}
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        cell = stack.pop()
        for nxt in _local_neighbors(cell):
            if nxt in nodes and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == nodes


def _patch_from_local_cells(local_cells: set[tuple[int, int]], color: int) -> tuple[np.ndarray, tuple[int, int]]:
    coords = [(0, 0), *sorted(local_cells)]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), Color.BLACK, dtype=int)
    anchor = (-min_r, -min_c)
    patch[anchor[0], anchor[1]] = Color.GREY
    for r, c in local_cells:
        patch[r - min_r, c - min_c] = int(color)
    return patch, anchor


def _build_path(endpoint: tuple[int, int], entry: tuple[int, int]) -> list[tuple[int, int]]:
    path = [entry]
    cur_r, cur_c = entry
    end_r, end_c = endpoint
    while (cur_r, cur_c) != (end_r, end_c):
        candidates = []
        if cur_r != end_r:
            candidates.append((cur_r + _sign(end_r - cur_r), cur_c))
        if cur_c != end_c:
            candidates.append((cur_r, cur_c + _sign(end_c - cur_c)))
        random.shuffle(candidates)
        cur_r, cur_c = candidates[0]
        path.append((cur_r, cur_c))
    return path


def _secondary_arm(entry: tuple[int, int], used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    choices = [d for d in CARDINALS if d != entry]
    random.shuffle(choices)
    for dr, dc in choices:
        step1 = (dr, dc)
        if step1 in used or not _in_local_bounds(step1):
            continue
        arm = [step1]
        step2 = (dr * 2, dc * 2)
        if random.random() < 0.55 and step2 not in used and _in_local_bounds(step2):
            arm.append(step2)
        return arm
    return []


def _grow_object(
    area: int,
    endpoint: tuple[int, int],
    entry: tuple[int, int],
    *,
    force_split: bool,
) -> set[tuple[int, int]] | None:
    cells = set(_build_path(endpoint, entry))
    if len(cells) > area:
        return None

    if force_split and len(cells) < area:
        for cell in _secondary_arm(entry, cells):
            if len(cells) >= area:
                break
            cells.add(cell)

    while len(cells) < area:
        frontier = set()
        for src in {(0, 0), *cells}:
            for nxt in _local_neighbors(src):
                if nxt != (0, 0) and nxt not in cells and _in_local_bounds(nxt):
                    frontier.add(nxt)
        if not frontier:
            return None

        scored = []
        for cell in frontier:
            dist_to_end = abs(cell[0] - endpoint[0]) + abs(cell[1] - endpoint[1])
            dist_from_anchor = abs(cell[0]) + abs(cell[1])
            score = (-dist_to_end, dist_from_anchor)
            scored.append((score, random.random(), cell))
        scored.sort(reverse=True)
        cells.add(scored[0][2])

    if not _is_sprite_connected(cells):
        return None
    return cells


def _overlay_order(sprites: list[dict]) -> list[int]:
    return sorted(
        range(len(sprites)),
        key=lambda i: (
            -int(sprites[i]["area"]),
            int(sprites[i].get("bbox_rc", (0, 0))[0]),
            int(sprites[i].get("bbox_rc", (0, 0))[1]),
        ),
    )


def _analyze_sprites(sprites: list[dict]) -> dict:
    coverage: dict[tuple[int, int], list[int]] = {}
    visible_owner: dict[tuple[int, int], int] = {}
    visible_color: dict[tuple[int, int], int] = {}

    for idx, sprite in enumerate(sprites):
        for cell in sprite["local_cells"]:
            coverage.setdefault(cell, []).append(idx)

    for idx in _overlay_order(sprites):
        sprite = sprites[idx]
        for cell in sprite["local_cells"]:
            visible_owner[cell] = idx
            visible_color[cell] = int(sprite["color"])

    visible_counts = [0] * len(sprites)
    for owner in visible_owner.values():
        visible_counts[int(owner)] += 1

    rows = [r for r, _ in visible_color]
    cols = [c for _, c in visible_color]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    composite = np.full((max_r - min_r + 1, max_c - min_c + 1), Color.BLACK, dtype=int)
    for (r, c), color in visible_color.items():
        composite[r - min_r, c - min_c] = int(color)

    anchor = (-min_r, -min_c)
    regions = find_components(
        composite,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(regions) >= 3, "expected at least three regions in composite"

    ordered = []
    for region in regions:
        coords = region["coords"]
        centroid = tuple(coords.mean(axis=0))
        angle = _angle_from_anchor(anchor, centroid)
        ordered.append((angle, region))
    ordered.sort(key=lambda item: item[0])

    recolored = composite.copy()
    ordered_regions = [region for _angle, region in ordered]
    ordered_colors = [int(region["color"]) for region in ordered_regions]
    for i, region in enumerate(ordered_regions):
        new_color = ordered_colors[(i + 1) % len(ordered_regions)]
        coords = region["coords"]
        recolored[coords[:, 0], coords[:, 1]] = int(new_color)

    return {
        "coverage": coverage,
        "visible_counts": visible_counts,
        "composite": composite,
        "anchor": anchor,
        "ordered": ordered,
        "output": recolored,
    }


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert 3 <= len(components) <= 4, "expected 3-4 sprites"

    sprites = []
    seen_colors = set()
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        patch = crop_bbox(grid, component["bbox"])
        anchor_coords = np.argwhere(patch == Color.GREY)
        assert anchor_coords.shape == (1, 2), "each sprite must contain exactly one gray anchor"
        anchor = tuple(map(int, anchor_coords[0]))

        colors = sorted(set(map(int, patch.flatten())) - {Color.BLACK, Color.GREY})
        assert len(colors) == 1, "each sprite must use exactly one non-gray color"
        color = int(colors[0])
        assert color in PALETTE, "unexpected sprite color"
        assert color not in seen_colors, "sprite colors must be distinct"
        seen_colors.add(color)

        local_cells = set()
        for rr, cc in np.argwhere(patch == color):
            local_cells.add((int(rr - anchor[0]), int(cc - anchor[1])))

        assert 4 <= (len(local_cells) + 1) <= 8, "sprite size must be 4-8 pixels including anchor"
        assert _is_sprite_connected(local_cells), "sprite must be 4-connected through its anchor"
        sprites.append(
            {
                "color": color,
                "area": len(local_cells),
                "bbox_rc": (int(r0), int(c0)),
                "local_cells": local_cells,
            }
        )

    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay anchor-marked sprites, then cyclically recolor the emergent regions by their angle around the shared anchor.

    Input structure:
    - Black background contains 3-4 separated 4-connected sprites.
    - Each sprite has exactly one gray anchor pixel and all remaining pixels share one unique non-gray color.
    - Sprites are disjoint in the input and are interpreted in the grid's row/column coordinate system.

    Rule:
    - Crop each sprite to its bounding box and identify the gray anchor.
    - Choose the sprite with the largest non-anchor area as the base (ties by smaller input bbox row, then col), and align every other sprite so all gray anchors coincide.
    - When aligned non-anchor pixels overlap, the smaller-area sprite is visible above the larger one; then delete the shared gray anchor, crop the composite to its non-black bbox, find every 4-connected monochrome region, sort those regions by centroid angle around the original anchor position (up = 0 degrees, clockwise increasing), and recolor each region with the next region's original color in that cyclic order.

    Key insight:
    - Anchor alignment does not directly produce the answer; it creates a composite whose post-overlap color regions and anchor-centered angular order determine the final cyclic recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sprites = _extract_sprites(g)
    analysis = _analyze_sprites(sprites)
    return analysis["output"]


def _sample_endpoint_plan(n_objects: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    shared_dir = random.choice(CARDINALS)
    shared_endpoints = random.sample(COMPATIBLE_ENDPOINTS[shared_dir], 2)
    selected = shared_endpoints[:]

    remaining = [ep for ep in ENDPOINTS if ep not in selected]
    random.shuffle(remaining)
    for endpoint in remaining:
        candidate = [*selected, endpoint]
        angles = sorted(_angle_from_anchor((0.0, 0.0), ep) for ep in candidate)
        gaps = [
            (angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi)
            for i in range(len(angles))
        ]
        if min(gaps) >= math.radians(30):
            selected.append(endpoint)
        if len(selected) == n_objects:
            break

    if len(selected) != n_objects:
        raise RuntimeError("failed to pick endpoints")

    plans = []
    forced = set(shared_endpoints)
    for endpoint in selected:
        options = []
        er, ec = endpoint
        if er < 0:
            options.append((-1, 0))
        if er > 0:
            options.append((1, 0))
        if ec < 0:
            options.append((0, -1))
        if ec > 0:
            options.append((0, 1))
        entry = shared_dir if endpoint in forced else random.choice(options)
        plans.append((endpoint, entry))
    random.shuffle(plans)
    return plans


def _semantic_checks(sprites: list[dict]) -> dict | None:
    analysis = _analyze_sprites(sprites)
    overlap_cells = [cell for cell, owners in analysis["coverage"].items() if len(owners) >= 2]
    if not overlap_cells:
        return None

    if any(count <= 0 for count in analysis["visible_counts"]):
        return None

    ordered = analysis["ordered"]
    if len(ordered) <= len(sprites):
        return None

    angles = [angle for angle, _region in ordered]
    gaps = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi) for i in range(len(angles))]
    if min(gaps) < MIN_ANGLE_GAP:
        return None

    if np.array_equal(analysis["composite"], analysis["output"]):
        return None

    changed = int(np.sum(analysis["composite"] != analysis["output"]))
    if changed < 3:
        return None

    return analysis


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 2) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != Color.BLACK):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if np.any(grid[r1:r2, c1:c2] != Color.BLACK):
            return False
    return True


def _place_sprites(sprites: list[dict], H: int, W: int) -> np.ndarray | None:
    grid = np.full((H, W), Color.BLACK, dtype=int)
    order = sorted(range(len(sprites)), key=lambda i: sprites[i]["patch"].size, reverse=True)
    for idx in order:
        patch = sprites[idx]["patch"]
        h, w = patch.shape
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if _can_place(grid, patch, r0, c0, padding=2):
                paste(grid, patch, int(r0), int(c0), transparent=Color.BLACK)
                sprites[idx]["bbox_rc"] = (int(r0), int(c0))
                placed = True
                break
        if not placed:
            return None
    return grid


def _build_random_sprite(area: int, color: int, endpoint: tuple[int, int], entry: tuple[int, int], force_split: bool) -> dict | None:
    local_cells = _grow_object(area, endpoint, entry, force_split=force_split)
    if local_cells is None:
        return None
    patch, anchor = _patch_from_local_cells(local_cells, color)
    return {
        "color": int(color),
        "area": int(area),
        "bbox_rc": (0, 0),
        "local_cells": local_cells,
        "patch": patch,
        "anchor_local": anchor,
    }


def _generate_attempt() -> np.ndarray | None:
    n_objects = random.choice([3, 4])
    areas = random.sample([4, 5, 6, 7], n_objects)
    colors = random.sample(PALETTE, n_objects)
    plans = _sample_endpoint_plan(n_objects)

    sprites = []
    for i, ((endpoint, entry), area, color) in enumerate(zip(plans, areas, colors)):
        sprite = _build_random_sprite(area, color, endpoint, entry, force_split=(i < 2))
        if sprite is None:
            return None
        sprites.append(sprite)

    analysis = _semantic_checks(sprites)
    if analysis is None:
        return None

    H = random.randint(*INPUT_ROWS)
    W = random.randint(*INPUT_COLS)
    grid = _place_sprites(sprites, H, W)
    if grid is None:
        return None

    expected = analysis["output"]
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
