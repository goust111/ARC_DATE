from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _label_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _analyze_blocks(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    non_bg = [int(color) for color in np.unique(g) if int(color) != bg]
    assert non_bg

    fg = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))
    components = _label_components(g == fg)
    assert 4 <= len(components) <= 5

    records: list[dict] = []
    for coords in components:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        bh, bw = patch.shape
        assert bh == bw and bh in (5, 6)

        marker_mask = (patch != bg) & (patch != fg)
        marker_coords = np.argwhere(marker_mask)
        assert marker_coords.shape[0] == 1
        mr, mc = map(int, marker_coords[0])
        marker_color = int(patch[mr, mc])

        records.append(
            {
                "bbox": bbox,
                "patch": patch.copy(),
                "hole_count": int(np.count_nonzero(patch == bg)),
                "marker_color": marker_color,
                "marker_coord": (r0 + mr, c0 + mc),
                "area": len(coords),
            }
        )

    hole_hist = Counter(record["hole_count"] for record in records)
    unique_holes = [holes for holes, count in hole_hist.items() if count == 1]
    assert len(unique_holes) == 1

    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        record["selected"] = record["hole_count"] == unique_holes[0]
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == record["patch"])

    assert sum(int(record["selected"]) for record in records) == 1
    return bg, fg, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the foreground block whose number of internal background cells is unique, and fill that block's holes with its marker color.

    Input structure:
    - The background is black, while one dominant foreground color forms 4-5 disjoint 4-connected blocks.
    - Each block fits inside its own 5x5 or 6x6 bounding box and contains exactly one non-background marker pixel of a different color.
    - Background cells inside a block's bounding box act as that block's holes; all non-selected blocks share one hole count and exactly one block has a different count.

    Rule:
    - Find the dominant foreground color and its 4-connected components.
    - For each component, count the background cells inside its own bounding box and identify the unique outlier count.
    - Erase every non-outlier block completely.
    - Keep the outlier block in place and recolor every background cell inside its bounding box to the color of its single marker pixel, while leaving the marker and foreground pixels unchanged.

    Key insight:
    - The same local statistic, the number of background cells inside each block's bbox, both selects the surviving block and decides which operation applies: the outlier is filled, while all others are erased.
    """
    g = np.asarray(grid, dtype=int)
    bg, _fg, records = _analyze_blocks(g)

    out = np.full_like(g, bg)
    target = next(record for record in records if record["selected"])
    filled = target["patch"].copy()
    filled[filled == bg] = target["marker_color"]
    r0, c0, r1, c1 = target["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = filled
    return out


GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (18, 24)
BLOCK_COUNT_CHOICES = (4, 5)
BLOCK_SIZE_CHOICES = (5, 6)
HOLE_COUNT_CHOICES = (0, 1, 2, 3, 4)
FG_COLOR_CHOICES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MIN_BBOX_GAP = 1
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 500
BLOCK_ATTEMPTS = 200


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_BBOX_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    return not separated


def _hole_positions(size: int, count: int) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(BLOCK_ATTEMPTS):
        holes = random.sample(candidates, count)
        if all(abs(r1 - r2) + abs(c1 - c2) != 1 for i, (r1, c1) in enumerate(holes) for r2, c2 in holes[i + 1 :]):
            return holes
    raise RuntimeError("failed to sample non-adjacent hole positions")


def _is_connected(mask: np.ndarray) -> bool:
    comps = _label_components(mask)
    return len(comps) == 1


def _make_block(size: int, fg: int, marker_color: int, hole_count: int) -> np.ndarray:
    for _ in range(BLOCK_ATTEMPTS):
        block = np.full((size, size), fg, dtype=int)
        for r, c in _hole_positions(size, hole_count):
            block[r, c] = 0

        interior_fg = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if block[r, c] == fg]
        random.shuffle(interior_fg)
        for r, c in interior_fg:
            block[r, c] = marker_color
            if _is_connected(block == fg):
                return block
            block[r, c] = fg
    raise RuntimeError("failed to construct a valid block")


def _place_bboxes(height: int, width: int, sizes: list[int]) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    order = list(range(len(sizes)))
    random.shuffle(order)
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(sizes)

    for idx in order:
        size = sizes[idx]
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - size)
            c0 = random.randint(0, width - size)
            bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
            if all(not _boxes_too_close(bbox, other) for other in placed):
                placed.append(bbox)
                boxes[idx] = bbox
                break
        else:
            return None

    return [box for box in boxes if box is not None]


def _expected_output_from_records(shape: tuple[int, int], bg: int, records: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    target = next(record for record in records if record["selected"])
    r0, c0, r1, c1 = target["bbox"]
    patch = target["patch"].copy()
    patch[patch == bg] = target["marker_color"]
    out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, fg, records = _analyze_blocks(grid)
    except AssertionError:
        return False

    if len({record["marker_color"] for record in records}) != len(records):
        return False

    hole_hist = Counter(record["hole_count"] for record in records)
    if sorted(hole_hist.values()) != [1, len(records) - 1]:
        return False

    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.count_nonzero((patch != bg) & (patch != fg)) != 1:
            return False

    expected = _expected_output_from_records(grid.shape, bg, records)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    for record in records:
        r0, c0, r1, c1 = record["bbox"]
        out_patch = actual[r0 : r1 + 1, c0 : c1 + 1]
        if record["selected"]:
            wanted = record["patch"].copy()
            wanted[wanted == bg] = record["marker_color"]
            if not np.array_equal(out_patch, wanted):
                return False
        elif np.any(out_patch != bg):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    block_count = random.choice(BLOCK_COUNT_CHOICES)
    fg = random.choice(FG_COLOR_CHOICES)
    marker_colors = random.sample([color for color in FG_COLOR_CHOICES if color != fg], block_count)

    common_holes = random.choice(HOLE_COUNT_CHOICES)
    unique_holes = random.choice([count for count in HOLE_COUNT_CHOICES if count != common_holes])
    unique_index = random.randrange(block_count)

    sizes = [random.choice(BLOCK_SIZE_CHOICES) for _ in range(block_count)]
    boxes = _place_bboxes(height, width, sizes)
    if boxes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for idx, bbox in enumerate(boxes):
        r0, c0, r1, c1 = bbox
        holes = unique_holes if idx == unique_index else common_holes
        block = _make_block(sizes[idx], fg, marker_colors[idx], holes)
        grid[r0 : r1 + 1, c0 : c1 + 1] = block

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
