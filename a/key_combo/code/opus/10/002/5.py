from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BASE_SIDE = 12
GEN_RETRY_LIMIT = 200


TEMPLATES = [
    {
        "name": "feedback_3color",
        "mode": "non_tie",
        "coords": [(0, 0), (2, 3), (2, 6), (4, 4), (4, 7), (6, 5), (9, 1), (9, 4), (11, 8)],
        "slots": [0, 0, 0, 1, 1, 1, 2, 2, 2],
    },
    {
        "name": "feedback_2color",
        "mode": "non_tie",
        "coords": [(0, 0), (2, 3), (2, 6), (4, 4), (4, 7), (6, 5), (9, 1), (9, 4), (11, 8)],
        "slots": [0, 0, 0, 1, 1, 1, 1, 1, 1],
    },
    {
        "name": "tie_3color",
        "mode": "tie",
        "coords": [(0, 0), (2, 3), (2, 6), (4, 1), (4, 4), (10, 10), (8, 1), (8, 4)],
        "slots": [0, 0, 0, 1, 1, 1, 2, 2],
    },
    {
        "name": "tie_2color",
        "mode": "tie",
        "coords": [(0, 0), (2, 3), (2, 6), (4, 1), (4, 4), (10, 10), (8, 1), (8, 4)],
        "slots": [0, 0, 0, 1, 1, 1, 1, 1],
    },
]


def _halo_cells(shape: tuple[int, int], r: int, c: int) -> tuple[tuple[int, int], ...]:
    h, w = shape
    return tuple(
        (rr, cc)
        for rr in range(max(0, r - 1), min(h, r + 2))
        for cc in range(max(0, c - 1), min(w, c + 2))
    )


def _halos_touch(halo_a: tuple[tuple[int, int], ...], halo_b_set: set[tuple[int, int]]) -> bool:
    for r, c in halo_a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in halo_b_set:
                return True
    return False


def _contact_pixel_count(halo: tuple[tuple[int, int], ...], other_union: set[tuple[int, int]]) -> int:
    count = 0
    for r, c in halo:
        if any((r + dr, c + dc) in other_union for dr, dc in DIR4):
            count += 1
    return count


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    markers = []
    for r, c in np.argwhere(g != 0):
        markers.append(
            {
                "row": int(r),
                "col": int(c),
                "color": int(g[r, c]),
            }
        )

    for marker in markers:
        halo = _halo_cells(g.shape, marker["row"], marker["col"])
        marker["halo"] = halo
        marker["halo_set"] = set(halo)

    n = len(markers)
    touch = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            is_touching = _halos_touch(markers[i]["halo"], markers[j]["halo_set"]) or _halos_touch(
                markers[j]["halo"], markers[i]["halo_set"]
            )
            touch[i][j] = is_touching
            touch[j][i] = is_touching

    contact_pixels = [0] * n
    pre_neighbor_counts = [0] * n
    for i, marker in enumerate(markers):
        same = [j for j, other in enumerate(markers) if j != i and other["color"] == marker["color"]]
        other_union: set[tuple[int, int]] = set()
        for j in same:
            other_union.update(markers[j]["halo_set"])
            if touch[i][j]:
                pre_neighbor_counts[i] += 1
        contact_pixels[i] = _contact_pixel_count(marker["halo"], other_union)

    survivors = [count >= 2 for count in contact_pixels]

    post_neighbor_counts = [0] * n
    for i, marker in enumerate(markers):
        if not survivors[i]:
            continue
        post_neighbor_counts[i] = sum(
            1
            for j, other in enumerate(markers)
            if j != i and survivors[j] and other["color"] == marker["color"] and touch[i][j]
        )

    overlap_pixels: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, marker in enumerate(markers):
        if not survivors[i]:
            continue
        for cell in marker["halo"]:
            overlap_pixels[cell].append(i)

    winners: dict[tuple[int, int], int] = {}
    for cell, indices in overlap_pixels.items():
        winners[cell] = max(
            indices,
            key=lambda idx: (
                post_neighbor_counts[idx],
                -markers[idx]["row"],
                -markers[idx]["col"],
            ),
        )

    out = np.zeros_like(g)
    for cell, winner in winners.items():
        out[cell] = markers[winner]["color"]
    for i, marker in enumerate(markers):
        if survivors[i]:
            out[marker["row"], marker["col"]] = marker["color"]

    return {
        "grid": g,
        "markers": markers,
        "touch": touch,
        "contact_pixels": contact_pixels,
        "pre_neighbor_counts": pre_neighbor_counts,
        "survivors": survivors,
        "post_neighbor_counts": post_neighbor_counts,
        "overlap_pixels": overlap_pixels,
        "winners": winners,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand every colored marker into a one-cell halo, delete weak same-color contacts, then arbitrate overlapping survivor halos.

    Input structure:
    - The grid is black background with 8-12 colored single-pixel markers using 2-3 colors.
    - Markers may lie near edges, so their clipped 3x3 halos can have different sizes.
    - Some same-color markers are close enough that their halos touch by 4-neighbor adjacency, while others are isolated.

    Rule:
    - Treat each non-black pixel as a marker and dilate it to its clipped 3x3 halo in the marker's own color.
    - For each marker, count how many pixels in its halo have a 4-neighbor belonging to another same-color halo; keep the marker iff that contact-pixel count is at least 2, otherwise erase both the marker and its whole halo.
    - After all deletions, color each pixel covered by surviving halos with the surviving marker whose same-color surviving-neighbor count is largest; break ties by smaller marker row, then smaller marker column, and finally restore every surviving center pixel to its original color.

    Key insight:
    - The decisive adjacency is not between the original points but between their imagined post-dilation halos, and deleting weak halos changes the surviving same-color neighbor counts that later resolve cross-color overlaps.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _transform_point(r: int, c: int, code: int, side: int = BASE_SIDE) -> tuple[int, int]:
    last = side - 1
    if code == 0:
        return r, c
    if code == 1:
        return c, last - r
    if code == 2:
        return last - r, last - c
    if code == 3:
        return last - c, r
    if code == 4:
        return r, last - c
    if code == 5:
        return last - r, c
    if code == 6:
        return c, r
    if code == 7:
        return last - c, last - r
    raise ValueError(f"bad transform code: {code}")


def _place_template(template: dict, height: int, width: int) -> list[tuple[int, int]]:
    code = random.randrange(8)
    transformed = [_transform_point(r, c, code) for r, c in template["coords"]]
    weak_r, weak_c = transformed[0]
    add_r = 0 if weak_r == 0 else height - 1 - weak_r
    add_c = 0 if weak_c == 0 else width - 1 - weak_c
    placed = [(r + add_r, c + add_c) for r, c in transformed]
    assert all(0 <= r < height and 0 <= c < width for r, c in placed)
    return placed


def _branch_condition(meta: dict, mode: str) -> bool:
    markers = meta["markers"]
    counts = meta["post_neighbor_counts"]
    winners = meta["winners"]
    overlap_pixels = meta["overlap_pixels"]

    for cell, indices in overlap_pixels.items():
        colors = {markers[i]["color"] for i in indices}
        if len(colors) < 2:
            continue
        color_best = Counter()
        for i in indices:
            color_best[markers[i]["color"]] = max(color_best[markers[i]["color"]], counts[i])
        top = max(color_best.values())
        top_colors = [color for color, value in color_best.items() if value == top]
        winner_color = markers[winners[cell]]["color"]
        if mode == "tie":
            if len(top_colors) >= 2:
                tied = [i for i in indices if counts[i] == top]
                expected = min(tied, key=lambda idx: (markers[idx]["row"], markers[idx]["col"]))
                if winner_color == markers[expected]["color"]:
                    return True
        else:
            if len(top_colors) == 1 and winner_color == top_colors[0]:
                return True
    return False


def _validate_generated(grid: np.ndarray, mode: str) -> bool:
    meta = _analyze(grid)
    markers = meta["markers"]
    colors = Counter(marker["color"] for marker in markers)
    if not (8 <= len(markers) <= 12):
        return False
    if not (2 <= len(colors) <= 3):
        return False
    if min(colors.values()) < 2:
        return False
    if sum(not alive for alive in meta["survivors"]) < 2:
        return False

    overlap_pairs = set()
    for indices in meta["overlap_pixels"].values():
        for i in indices:
            for j in indices:
                if i < j and markers[i]["color"] != markers[j]["color"]:
                    overlap_pairs.add((i, j))
    if len(overlap_pairs) < 2:
        return False

    feedback = any(
        meta["survivors"][i] and meta["pre_neighbor_counts"][i] > meta["post_neighbor_counts"][i]
        for i in range(len(markers))
    )
    if not feedback:
        return False

    if not _branch_condition(meta, mode):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out, meta["output"]):
        return False
    return True


def _template_colors(template: dict) -> list[int]:
    slot_count = max(template["slots"]) + 1
    return random.sample(PALETTE, slot_count)


def _build_grid(template: dict) -> np.ndarray:
    height = random.randint(12, 15)
    width = random.randint(12, 15)
    coords = _place_template(template, height, width)
    colors = _template_colors(template)

    grid = np.zeros((height, width), dtype=int)
    for (r, c), slot in zip(coords, template["slots"]):
        grid[r, c] = colors[slot]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        template = random.choice(TEMPLATES)
        grid = _build_grid(template)
        if _validate_generated(grid, template["mode"]):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
