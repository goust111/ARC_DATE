from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
KEY_RING_OFFSETS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            labels[r, c] = next_id
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in N4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
    return labels, next_id


def _ring_coords(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in KEY_RING_OFFSETS]


def _detect_keys(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    keys: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(g[r0 + 1, c0 + 1]) != BG:
                continue
            coords = _ring_coords(r0, c0)
            vals = [int(g[r, c]) for r, c in coords]
            if any(v in (BG, GREY) for v in vals):
                continue
            colors = set(vals)
            if len(colors) != 3:
                continue
            counts = Counter(vals).most_common()
            if len(counts) != 3 or counts[0][1] == counts[1][1]:
                continue
            keys.append(
                {
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "ring": coords,
                    "colors": frozenset(colors),
                    "activation": int(counts[0][0]),
                }
            )

    seen_boxes = set()
    unique_keys = []
    for key in keys:
        if key["bbox"] in seen_boxes:
            continue
        seen_boxes.add(key["bbox"])
        unique_keys.append(key)
    return unique_keys


def _check_bg_band(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w and int(grid[r, c]) != BG:
            return False
    return True


def _detect_doors(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    doors: list[dict] = []
    used = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(2, w - 3):
            door_color = int(g[r, c])
            if door_color in (BG, GREY) or door_color != int(g[r, c + 1]):
                continue
            if used[r, c] or used[r, c + 1]:
                continue
            left_color = int(g[r, c - 1])
            right_color = int(g[r, c + 2])
            if left_color in (BG, GREY) or right_color in (BG, GREY):
                continue
            if len({door_color, left_color, right_color}) != 3:
                continue
            if int(g[r, c - 2]) != GREY or int(g[r, c + 3]) != GREY:
                continue
            side_band = [(r - 1, cc) for cc in range(c - 1, c + 3)] + [(r + 1, cc) for cc in range(c - 1, c + 3)]
            if not _check_bg_band(g, side_band):
                continue
            used[r, c : c + 2] = True
            doors.append(
                {
                    "orientation": "h",
                    "door_cells": [(r, c), (r, c + 1)],
                    "open_cells": [(r, c - 1), (r, c), (r, c + 1), (r, c + 2)],
                    "outer_cells": [(r, c - 2), (r, c + 3)],
                    "colors": frozenset({door_color, left_color, right_color}),
                }
            )

    for r in range(2, h - 3):
        for c in range(w):
            door_color = int(g[r, c])
            if door_color in (BG, GREY) or door_color != int(g[r + 1, c]):
                continue
            if used[r, c] or used[r + 1, c]:
                continue
            top_color = int(g[r - 1, c])
            bottom_color = int(g[r + 2, c])
            if top_color in (BG, GREY) or bottom_color in (BG, GREY):
                continue
            if len({door_color, top_color, bottom_color}) != 3:
                continue
            if int(g[r - 2, c]) != GREY or int(g[r + 3, c]) != GREY:
                continue
            side_band = [(rr, c - 1) for rr in range(r - 1, r + 3)] + [(rr, c + 1) for rr in range(r - 1, r + 3)]
            if not _check_bg_band(g, side_band):
                continue
            used[r : r + 2, c] = True
            doors.append(
                {
                    "orientation": "v",
                    "door_cells": [(r, c), (r + 1, c)],
                    "open_cells": [(r - 1, c), (r, c), (r + 1, c), (r + 2, c)],
                    "outer_cells": [(r - 2, c), (r + 3, c)],
                    "colors": frozenset({door_color, top_color, bottom_color}),
                }
            )

    return doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock corridor doors by matching each door's three-color lock set with a key's three-color ring.

    Input structure:
    - Background is black.
    - Grey one-cell-wide straight corridors are interrupted by colored 2-cell doors.
    - Each door has one colored cell on each side along the corridor direction.
    - Keys are 3x3 rings with a black center near the edges of the grid.

    Rule:
    - Collect each door's three colors as the door color plus the two adjacent side colors.
    - Match a key whose outer ring contains exactly the same three colors.
    - Open matched doors by turning the door cells and the two side cells grey.
    - Recolor the newly connected grey corridor component with the key's majority ring color.
    - Clear all keys to black. Unmatched doors stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    keys = _detect_keys(g)
    key_by_colors: dict[frozenset[int], dict] = {}
    for key in keys:
        assert key["colors"] not in key_by_colors, "keys must have unique three-color signatures"
        key_by_colors[key["colors"]] = key

    doors = _detect_doors(g)

    out = g.copy()
    for key in keys:
        r0, c0, r1, c1 = key["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    matched: list[tuple[dict, dict]] = []
    for door in doors:
        key = key_by_colors.get(door["colors"])
        if key is None:
            continue
        matched.append((door, key))
        for r, c in door["open_cells"]:
            out[r, c] = GREY

    pre_labels, _ = _label_components(g == GREY)
    post_labels, _ = _label_components(out == GREY)

    comp_to_color: dict[int, int] = {}
    for door, key in matched:
        comp_id = int(post_labels[door["door_cells"][0]])
        assert comp_id > 0, "opened door must belong to a grey component"
        left_id = int(pre_labels[door["outer_cells"][0]])
        right_id = int(pre_labels[door["outer_cells"][1]])
        if left_id == 0 or right_id == 0 or left_id == right_id:
            continue
        color = int(key["activation"])
        prev = comp_to_color.get(comp_id)
        if prev is None:
            comp_to_color[comp_id] = color
        else:
            assert prev == color, "one corridor component cannot receive conflicting key colors"

    for comp_id, color in comp_to_color.items():
        out[post_labels == comp_id] = color

    return out


def _reserve_box(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _sample_triplets(count: int) -> list[tuple[int, int, int]]:
    combos = list(itertools.combinations(PALETTE, 3))
    random.shuffle(combos)
    return [tuple(map(int, combo)) for combo in combos[:count]]


def _draw_corridor(
    grid: np.ndarray,
    occupied: np.ndarray,
    triplet: tuple[int, int, int],
) -> bool:
    h, w = grid.shape
    for _ in range(120):
        orientation = random.choice(("h", "v"))
        left_len = random.randint(2, 6)
        right_len = random.randint(2, 6)
        total = left_len + right_len + 4

        colors = list(triplet)
        door_color = random.choice(colors)
        side_colors = [color for color in colors if color != door_color]
        random.shuffle(side_colors)

        if orientation == "h":
            if total > w - 8 or h < 9:
                continue
            r = random.randint(4, h - 5)
            c0 = random.randint(4, w - total - 4)
            bbox = (r, c0, r, c0 + total - 1)
            if not _reserve_box(occupied, bbox, pad=1):
                continue

            grid[r, c0 : c0 + left_len] = GREY
            grid[r, c0 + left_len] = side_colors[0]
            grid[r, c0 + left_len + 1 : c0 + left_len + 3] = door_color
            grid[r, c0 + left_len + 3] = side_colors[1]
            grid[r, c0 + left_len + 4 : c0 + total] = GREY
            return True

        if total > h - 8 or w < 9:
            continue
        r0 = random.randint(4, h - total - 4)
        c = random.randint(4, w - 5)
        bbox = (r0, c, r0 + total - 1, c)
        if not _reserve_box(occupied, bbox, pad=1):
            continue

        grid[r0 : r0 + left_len, c] = GREY
        grid[r0 + left_len, c] = side_colors[0]
        grid[r0 + left_len + 1 : r0 + left_len + 3, c] = door_color
        grid[r0 + left_len + 3, c] = side_colors[1]
        grid[r0 + left_len + 4 : r0 + total, c] = GREY
        return True

    return False


def _make_key_patch(triplet: tuple[int, int, int]) -> tuple[np.ndarray, int]:
    activation = random.choice(triplet)
    other = [color for color in triplet if color != activation]

    major_count = random.choice((4, 5))
    if major_count == 5:
        minor = [2, 1]
    else:
        minor = random.choice(([3, 1], [2, 2]))
    random.shuffle(other)

    ring_colors = [activation] * major_count + [other[0]] * minor[0] + [other[1]] * minor[1]
    random.shuffle(ring_colors)

    patch = np.full((3, 3), BG, dtype=int)
    for (dr, dc), color in zip(KEY_RING_OFFSETS, ring_colors):
        patch[dr, dc] = int(color)
    return patch, int(activation)


def _place_key(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
) -> bool:
    h, w = grid.shape
    for _ in range(160):
        side = random.choice(("top", "bottom", "left", "right"))
        if side == "top":
            r0 = random.choice((0, 1, 2))
            c0 = random.randint(0, w - 3)
        elif side == "bottom":
            r0 = random.choice((h - 3, h - 4, h - 5))
            c0 = random.randint(0, w - 3)
        elif side == "left":
            r0 = random.randint(0, h - 3)
            c0 = random.choice((0, 1, 2))
        else:
            r0 = random.randint(0, h - 3)
            c0 = random.choice((w - 3, w - 4, w - 5))

        bbox = (r0, c0, r0 + 2, c0 + 2)
        if not _reserve_box(occupied, bbox, pad=1):
            continue
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
        return True
    return False


def _build_scene() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    n_matched = random.randint(1, 3)
    n_unmatched = random.randint(0, 1)
    total_doors = n_matched + n_unmatched

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    triplets = _sample_triplets(total_doors)
    matched_triplets = triplets[:n_matched]

    for triplet in triplets:
        if not _draw_corridor(grid, occupied, triplet):
            return None

    for triplet in matched_triplets:
        patch, _activation = _make_key_patch(triplet)
        if not _place_key(grid, occupied, patch):
            return None

    return grid


def _is_valid_sample(grid: np.ndarray) -> bool:
    try:
        keys = _detect_keys(grid)
        doors = _detect_doors(grid)
        if not keys or not doors:
            return False

        key_sigs = {key["colors"] for key in keys}
        matched_doors = [door for door in doors if door["colors"] in key_sigs]
        unmatched_doors = [door for door in doors if door["colors"] not in key_sigs]
        if len(matched_doors) != len(keys):
            return False

        pre_labels, _ = _label_components(grid == GREY)
        out = main(grid)
        if out.shape != grid.shape:
            return False

        door_components = set()
        post_labels, _ = _label_components(out != BG)
        for door in matched_doors:
            left_id = int(pre_labels[door["outer_cells"][0]])
            right_id = int(pre_labels[door["outer_cells"][1]])
            if left_id == 0 or right_id == 0 or left_id == right_id:
                return False
            door_components.add(tuple(sorted((left_id, right_id))))
        if len(door_components) != len(matched_doors):
            return False

        if unmatched_doors and np.array_equal(out, grid):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_scene()
        if grid is None:
            continue
        if _is_valid_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid key-lock corridor puzzle")
