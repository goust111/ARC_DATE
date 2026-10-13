from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
FG_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200
GRID_MIN = 15
GRID_MAX = 30
OBJ_SIZE_MIN = 2
OBJ_SIZE_MAX = 6
GROUP_PADDING = 2


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    if np.any(values == 0):
        return 0
    return int(np.bincount(values).argmax())


def _label_components(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1 or int(grid[r, c]) == int(bg):
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            labels[r, c] = len(components)
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if labels[nr, nc] == -1 and int(grid[nr, nc]) == color:
                            labels[nr, nc] = len(components)
                            stack.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "id": len(components),
                    "color": color,
                    "coords": coord_arr,
                    "area": int(len(coords)),
                }
            )

    return labels, components


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _build_gap_pairs(labels: np.ndarray, bg_mask: np.ndarray) -> dict[tuple[int, int], dict]:
    h, w = labels.shape
    pairs: dict[tuple[int, int], dict] = {}

    for r in range(h):
        for c in range(w):
            if not bool(bg_mask[r, c]):
                continue

            if 0 < c < w - 1:
                left = int(labels[r, c - 1])
                right = int(labels[r, c + 1])
                if left != -1 and right != -1 and left != right:
                    key = _pair_key(left, right)
                    info = pairs.setdefault(key, {"cells": set(), "contact": 0})
                    info["cells"].add((r, c))
                    info["contact"] += 1

            if 0 < r < h - 1:
                up = int(labels[r - 1, c])
                down = int(labels[r + 1, c])
                if up != -1 and down != -1 and up != down:
                    key = _pair_key(up, down)
                    info = pairs.setdefault(key, {"cells": set(), "contact": 0})
                    info["cells"].add((r, c))
                    info["contact"] += 1

    return pairs


def _choose_targets(
    components: list[dict],
    pairs: dict[tuple[int, int], dict],
) -> dict[int, int]:
    areas = {int(comp["id"]): int(comp["area"]) for comp in components}
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for (a, b), info in pairs.items():
        area_a = areas[a]
        area_b = areas[b]
        contact = int(info["contact"])

        if area_a < area_b:
            candidates[a].append((b, contact))
        elif area_b < area_a:
            candidates[b].append((a, contact))

    targets: dict[int, int] = {}
    for comp_id, options in candidates.items():
        best = max(contact for _neighbor, contact in options)
        winners = [neighbor for neighbor, contact in options if contact == best]
        if len(winners) == 1:
            targets[int(comp_id)] = int(winners[0])

    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each foreground object across a one-cell gap into its larger neighboring object.

    Input structure:
    - The background is black.
    - The grid contains several 4-connected monochrome foreground objects, with each object
      using its own visible color.
    - Some object pairs are separated only by a single row or column of background cells, so
      their boundaries line up across a one-pixel gap.

    Rule:
    - Detect every pair of foreground objects whose boundaries face each other across exactly
      one background cell, and measure the shared boundary length along that gap.
    - For such a neighboring pair, compare the two object areas.
    - Recolor the smaller object to the larger object's color and also fill the one-pixel gap
      between them with that larger color.
    - If the two areas are equal, leave both objects and the gap unchanged; objects with no
      one-pixel-gap neighbor also stay unchanged.

    Key insight:
    The meaningful relation is not direct touching but alignment across a single black spacer;
    that spacer identifies the neighbor pair, while area comparison determines the merge color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    labels, components = _label_components(g, bg)
    if not components:
        return g.copy()

    pairs = _build_gap_pairs(labels, g == bg)
    if not pairs:
        return g.copy()

    targets = _choose_targets(components, pairs)
    if not targets:
        return g.copy()

    comp_by_id = {int(comp["id"]): comp for comp in components}
    out = g.copy()

    for comp_id, target_id in targets.items():
        coords = comp_by_id[comp_id]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(comp_by_id[target_id]["color"])

    for comp_id, target_id in targets.items():
        color = int(comp_by_id[target_id]["color"])
        gap_cells = pairs[_pair_key(comp_id, target_id)]["cells"]
        for r, c in gap_cells:
            out[int(r), int(c)] = color

    return out


def _random_rect(color: int) -> np.ndarray:
    h = random.randint(OBJ_SIZE_MIN, OBJ_SIZE_MAX)
    w = random.randint(OBJ_SIZE_MIN, OBJ_SIZE_MAX)
    return np.full((h, w), int(color), dtype=int)


def _sample_pair_dims(relation: str) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        h1 = random.randint(OBJ_SIZE_MIN, OBJ_SIZE_MAX)
        w1 = random.randint(OBJ_SIZE_MIN, OBJ_SIZE_MAX)
        if relation == "equal":
            if random.random() < 0.5 and h1 != w1:
                h2, w2 = w1, h1
            else:
                h2, w2 = h1, w1
        else:
            h2 = random.randint(OBJ_SIZE_MIN, OBJ_SIZE_MAX)
            w2 = random.randint(OBJ_SIZE_MIN, OBJ_SIZE_MAX)

        area1 = h1 * w1
        area2 = h2 * w2
        if relation == "equal" and area1 == area2:
            return (h1, w1), (h2, w2)
        if relation == "first_smaller" and area1 < area2:
            return (h1, w1), (h2, w2)
        if relation == "second_smaller" and area2 < area1:
            return (h1, w1), (h2, w2)


def _make_horizontal_pair(color1: int, color2: int, relation: str) -> np.ndarray:
    (h1, w1), (h2, w2) = _sample_pair_dims(relation)
    overlap = random.randint(1, min(h1, h2))
    if random.random() < 0.5:
        top1 = 0
        top2 = h1 - overlap
    else:
        top1 = h2 - overlap
        top2 = 0

    height = max(top1 + h1, top2 + h2)
    width = w1 + 1 + w2
    patch = np.zeros((height, width), dtype=int)
    patch[top1 : top1 + h1, 0:w1] = int(color1)
    patch[top2 : top2 + h2, w1 + 1 : w1 + 1 + w2] = int(color2)
    return patch


def _make_vertical_pair(color1: int, color2: int, relation: str) -> np.ndarray:
    (h1, w1), (h2, w2) = _sample_pair_dims(relation)
    overlap = random.randint(1, min(w1, w2))
    if random.random() < 0.5:
        left1 = 0
        left2 = w1 - overlap
    else:
        left1 = w2 - overlap
        left2 = 0

    height = h1 + 1 + h2
    width = max(left1 + w1, left2 + w2)
    patch = np.zeros((height, width), dtype=int)
    patch[0:h1, left1 : left1 + w1] = int(color1)
    patch[h1 + 1 : h1 + 1 + h2, left2 : left2 + w2] = int(color2)
    return patch


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False

    for _ in range(PLACE_ATTEMPTS):
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        r0 = max(0, top - GROUP_PADDING)
        r1 = min(h, top + ph + GROUP_PADDING)
        c0 = max(0, left - GROUP_PADDING)
        c1 = min(w, left + pw + GROUP_PADDING)
        if reserved[r0:r1, c0:c1].any():
            continue

        mask = patch != 0
        grid[top : top + ph, left : left + pw][mask] = patch[mask]
        reserved[top : top + ph, left : left + pw] = True
        return True

    return False


def _make_layout() -> tuple[list[np.ndarray], bool, bool] | None:
    pair_count = random.randint(2, 3)
    force_equal = random.random() < 0.6
    isolated_count = random.randint(1, 2) if random.random() < 0.6 else random.randint(0, 1)

    relations = [random.choice(("first_smaller", "second_smaller"))]
    if pair_count >= 2:
        relations.append("equal" if force_equal else random.choice(("first_smaller", "second_smaller")))
    while len(relations) < pair_count:
        relations.append(random.choice(("first_smaller", "second_smaller", "equal")))
    if all(relation == "equal" for relation in relations):
        relations[0] = random.choice(("first_smaller", "second_smaller"))

    orientations = []
    if pair_count == 2 and random.random() < 0.8:
        orientations = ["horizontal", "vertical"]
        random.shuffle(orientations)
    else:
        for _ in range(pair_count):
            orientations.append(random.choice(("horizontal", "vertical")))

    total_objects = pair_count * 2 + isolated_count
    if total_objects > len(FG_COLORS):
        return None
    colors = random.sample(FG_COLORS, total_objects)

    patches: list[np.ndarray] = []
    color_index = 0
    for orientation, relation in zip(orientations, relations):
        color1 = colors[color_index]
        color2 = colors[color_index + 1]
        color_index += 2
        if orientation == "horizontal":
            patches.append(_make_horizontal_pair(color1, color2, relation))
        else:
            patches.append(_make_vertical_pair(color1, color2, relation))

    for _ in range(isolated_count):
        patches.append(_random_rect(colors[color_index]))
        color_index += 1

    random.shuffle(patches)
    return patches, any(relation == "equal" for relation in relations), isolated_count > 0


def _valid_generated_grid(
    grid: np.ndarray,
    expected_pairs: int,
    require_equal: bool,
    require_isolated: bool,
) -> bool:
    labels, components = _label_components(grid, 0)
    if len(components) < 4:
        return False

    component_colors = [int(comp["color"]) for comp in components]
    if len(component_colors) != len(set(component_colors)):
        return False

    pairs = _build_gap_pairs(labels, grid == 0)
    if len(pairs) != expected_pairs:
        return False

    degree = defaultdict(int)
    equal_pairs = 0
    for (a, b), info in pairs.items():
        if int(info["contact"]) <= 0:
            return False
        degree[a] += 1
        degree[b] += 1
        if int(components[a]["area"]) == int(components[b]["area"]):
            equal_pairs += 1

    if max(degree.values(), default=0) > 1:
        return False

    if require_equal and equal_pairs == 0:
        return False

    isolated = sum(1 for comp in components if degree[int(comp["id"])] == 0)
    if require_isolated and isolated == 0:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out != grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        layout = _make_layout()
        if layout is None:
            continue
        patches, require_equal, require_isolated = layout

        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        grid = np.zeros((h, w), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)

        placed = True
        for patch in patches:
            if not _place_patch(grid, reserved, patch):
                placed = False
                break
        if not placed:
            continue

        expected_pairs = sum(1 for patch in patches if np.count_nonzero(patch == 0) > 0)
        if not _valid_generated_grid(grid, expected_pairs, require_equal, require_isolated):
            continue

        return grid

    raise RuntimeError("generate_input failed to create a valid sample")
