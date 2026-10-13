from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

MARKER_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _detect_blocks(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != BACKGROUND)
    blocks: list[dict] = []
    for idx, coords in enumerate(comps):
        assert len(coords) == 4, "every non-background component must be one 2x2 marker block"
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        assert r1 == r0 + 1 and c1 == c0 + 1, "marker block must have bbox 2x2"
        block = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        vals = [int(v) for v in block.flat]
        assert BACKGROUND not in vals
        assert len(set(vals)) == 4, "marker block cells must have four distinct colors"
        assert set(vals).issubset(MARKER_COLORS)
        blocks.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "block": block,
            }
        )
    blocks.sort(key=lambda item: item["bbox"])
    return blocks


def _expand_blocks(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    h, w = grid.shape
    out = grid.copy()
    owner = np.full((h, w), -1, dtype=int)

    blocks = _detect_blocks(grid)
    regions: list[dict] = []
    region_id = 0
    for block in blocks:
        for r in range(block["bbox"][0], block["bbox"][2] + 1):
            for c in range(block["bbox"][1], block["bbox"][3] + 1):
                owner[r, c] = -2 - block["id"]

    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        marker = block["block"]
        up = min(2, r0)
        down = min(2, h - r1 - 1)
        left = min(2, c0)
        right = min(2, w - c1 - 1)

        specs = [
            (int(marker[0, 0]), r1 + 1, r1 + down, c1 + 1, c1 + right, "br"),
            (int(marker[0, 1]), r1 + 1, r1 + down, c0 - left, c0 - 1, "bl"),
            (int(marker[1, 0]), r0 - up, r0 - 1, c1 + 1, c1 + right, "tr"),
            (int(marker[1, 1]), r0 - up, r0 - 1, c0 - left, c0 - 1, "tl"),
        ]

        for color, rr0, rr1, cc0, cc1, quadrant in specs:
            actual: list[tuple[int, int]] = []
            if rr0 <= rr1 and cc0 <= cc1:
                for r in range(rr0, rr1 + 1):
                    for c in range(cc0, cc1 + 1):
                        if out[r, c] == BACKGROUND:
                            out[r, c] = color
                            owner[r, c] = region_id
                            actual.append((r, c))
            if actual:
                regions.append(
                    {
                        "id": region_id,
                        "block_id": block["id"],
                        "color": color,
                        "bbox": _bbox_from_coords(actual),
                        "cells": actual,
                        "quadrant": quadrant,
                    }
                )
                region_id += 1

    return out, regions, owner


def _analyze_pairs(grid: np.ndarray) -> dict:
    expanded, regions, owner = _expand_blocks(grid)
    pairs: list[dict] = []
    for i in range(len(regions)):
        a = regions[i]
        ar0, ac0, ar1, ac1 = a["bbox"]
        for j in range(i + 1, len(regions)):
            b = regions[j]
            if a["block_id"] == b["block_id"] or a["color"] != b["color"]:
                continue
            br0, bc0, br1, bc1 = b["bbox"]
            row0 = max(ar0, br0)
            row1 = min(ar1, br1)
            if row0 > row1:
                continue
            if ac1 < bc0:
                gap0, gap1 = ac1 + 1, bc0 - 1
                left_id, right_id = a["id"], b["id"]
            elif bc1 < ac0:
                gap0, gap1 = bc1 + 1, ac0 - 1
                left_id, right_id = b["id"], a["id"]
            else:
                continue
            if gap0 > gap1:
                continue
            gap = expanded[row0 : row1 + 1, gap0 : gap1 + 1]
            blocked = np.any(gap != BACKGROUND)
            blockers = sorted(
                {
                    int(region)
                    for region in np.unique(owner[row0 : row1 + 1, gap0 : gap1 + 1])
                    if region >= 0 and region not in {left_id, right_id}
                }
            )
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "rows": (row0, row1),
                    "cols": (gap0, gap1),
                    "blocked": bool(blocked),
                    "blockers": blockers,
                }
            )
    return {"expanded": expanded, "regions": regions, "pairs": pairs, "owner": owner}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand every 2x2 marker block diagonally and bridge unobstructed horizontal gaps between same-colored expanded regions.

    Input structure:
    - The grid is a black background containing several separated 2x2 marker blocks.
    - Every marker block uses four distinct colors chosen from blue, red, green, yellow, and orange.
    - Different blocks may reuse colors, which is what can later create same-colored horizontal alignments.

    Rule:
    - Keep every marker block unchanged.
    - For each marker block, fill the diagonally opposite outside rectangle of each marker cell, extending at most two rows and two columns and clipping at the grid boundary; only background cells may be painted.
    - After all expansions are finished, examine every pair of expanded regions from different blocks with the same color, overlapping row spans, and disjoint column spans; if the full horizontal gap rectangle between them is still all background, fill that gap rectangle with grey.

    Key insight:
    - The grey bridges are not marked directly by the input blocks: they appear only after diagonal expansion creates same-colored regions whose shared row band is horizontally clear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert GREY not in set(map(int, np.unique(g))), "grey is output-only"

    analysis = _analyze_pairs(g)
    base = analysis["expanded"]
    out = base.copy()
    for pair in analysis["pairs"]:
        if pair["blocked"]:
            continue
        row0, row1 = pair["rows"]
        col0, col1 = pair["cols"]
        out[row0 : row1 + 1, col0 : col1 + 1] = GREY
    return out


def _random_distinct_colors(rng: random.Random) -> list[int]:
    colors = list(MARKER_COLORS)
    rng.shuffle(colors)
    return colors[:4]


def _make_primary_blocks(rng: random.Random, bridge_color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    others = [c for c in MARKER_COLORS if c != bridge_color]
    rng.shuffle(others)
    a_tr, b_tl, c_tl = others[:3]

    def finish_block(first_row: tuple[int, int]) -> np.ndarray:
        remaining = [c for c in MARKER_COLORS if c not in first_row]
        rng.shuffle(remaining)
        return np.array([[first_row[0], first_row[1]], [remaining[0], remaining[1]]], dtype=int)

    block_a = finish_block((bridge_color, a_tr))
    block_b = finish_block((b_tl, bridge_color))
    block_c = finish_block((c_tl, bridge_color))
    return block_a, block_b, block_c


def _construct_case(rng: random.Random) -> np.ndarray:
    bridge_color = rng.choice(MARKER_COLORS)
    gap1 = rng.randint(1, 4)
    gap2 = rng.randint(1, 4)
    c_a = rng.randint(0, 2)
    c_b = c_a + 6 + gap1
    c_c = c_b + 6 + gap2

    min_w = c_c + 2
    width = rng.randint(min_w + 1, min(30, min_w + 6))
    height = rng.randint(12, 18)
    grid = np.zeros((height, width), dtype=int)

    block_a, block_b, block_c = _make_primary_blocks(rng, bridge_color)
    grid[0:2, c_a : c_a + 2] = block_a
    grid[0:2, c_b : c_b + 2] = block_b
    grid[0:2, c_c : c_c + 2] = block_c

    add_fourth = rng.random() < 0.55
    if add_fourth:
        c_d = rng.randint(0, width - 2)
        grid[height - 2 : height, c_d : c_d + 2] = np.array(_random_distinct_colors(rng), dtype=int).reshape(2, 2)

    if rng.random() < 0.5:
        grid = np.fliplr(grid)
    if rng.random() < 0.5:
        grid = np.flipud(grid)
    return grid


def _validate_case(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    analysis = _analyze_pairs(grid)
    blocks = _detect_blocks(grid)
    if len(blocks) not in {3, 4}:
        return False

    solved = main(grid)
    grey_mask = solved == GREY
    if not np.any(grey_mask):
        return False
    if np.any(grid == GREY):
        return False

    unblocked_pairs = [pair for pair in analysis["pairs"] if not pair["blocked"]]
    blocked_pairs = [pair for pair in analysis["pairs"] if pair["blocked"]]
    if len(unblocked_pairs) != 1:
        return False
    if len(blocked_pairs) < 1:
        return False

    bridge_pair = unblocked_pairs[0]
    row0, row1 = bridge_pair["rows"]
    col0, col1 = bridge_pair["cols"]
    if not np.all(solved[row0 : row1 + 1, col0 : col1 + 1] == GREY):
        return False

    grey_boxes = _components(grey_mask)
    if len(grey_boxes) != 1:
        return False

    # Require a genuinely blocked same-color pair whose gap contains a third expansion region.
    if not any(pair["blockers"] for pair in blocked_pairs):
        return False

    # Marker blocks must remain separated components in the input.
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        if (r1 - r0, c1 - c0) != (1, 1):
            return False

    # The intended bridge should be horizontal and nontrivial.
    if col0 > col1 or row0 > row1:
        return False
    if (col1 - col0 + 1) < 1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rng = random.Random()
        grid = _construct_case(rng)
        if _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
