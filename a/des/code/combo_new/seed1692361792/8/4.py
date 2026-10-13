from __future__ import annotations

import random
import numpy as np


BLACK = 0
GREY = 5
PORTAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRIANGLE_MASKS: dict[str, np.ndarray] = {
    "ul": np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    "ur": np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool),
    "dl": np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool),
    "dr": np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=bool),
}
SIDES = ["up", "down", "left", "right"]
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIR4:
        rr, cc = int(r + dr), int(c + dc)
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _triangle_info(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(coords) != 6:
        return None

    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return None

    patch = np.zeros((3, 3), dtype=bool)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = True

    for name, mask in TRIANGLE_MASKS.items():
        if np.array_equal(patch, mask):
            return {
                "name": name,
                "top_left": (int(r0), int(c0)),
                "cells": tuple(sorted((int(r), int(c)) for r, c in coords)),
            }
    return None


def _portal_pairs(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    pairs: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(coords) == 2:
            pairs[color] = (coords[0], coords[1])
    return pairs


def _mask_cells(mask: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    return {
        (int(r0 + r), int(c0 + c))
        for r in range(3)
        for c in range(3)
        if bool(mask[r, c])
    }


def _portal_offsets(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    if side == "up":
        return [(-1, int(c)) for c in range(3) if bool(mask[0, c])]
    if side == "down":
        return [(3, int(c)) for c in range(3) if bool(mask[2, c])]
    if side == "left":
        return [(int(r), -1) for r in range(3) if bool(mask[r, 0])]
    return [(int(r), 3) for r in range(3) if bool(mask[r, 2])]


def _valid_top_lefts(height: int, width: int, side: str) -> list[tuple[int, int]]:
    if side == "up":
        rows = range(1, height - 2)
        cols = range(0, width - 2)
    elif side == "down":
        rows = range(0, height - 3)
        cols = range(0, width - 2)
    elif side == "left":
        rows = range(0, height - 2)
        cols = range(1, width - 2)
    else:
        rows = range(0, height - 2)
        cols = range(0, width - 3)
    return [(int(r), int(c)) for r in rows for c in cols]


def _expand_cells(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    expanded.add((int(rr), int(cc)))
    return expanded


def _zones_separated(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for ar, ac in a:
        for br, bc in b:
            if max(abs(int(ar - br)), abs(int(ac - bc))) <= 1:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each grey triangular arrow from its touching portal to the matching portal endpoint.

    Input structure:
    - The background is black.
    - Each portal pair consists of exactly two isolated single pixels of the same non-black, non-grey color.
    - Each arrow is a grey 3x3 right triangle made of 6 cells, with one of the four diagonal-facing orientations.
    - Every arrow is 4-neighbor adjacent to exactly one portal pixel, while the rest of the grid stays empty.

    Rule:
    - Match every portal pixel to the other pixel of the same color.
    - For each grey triangle, find the unique portal pixel touching it and compute the translation from that portal to its partner.
    - Remove the original triangle, then redraw the whole triangle after applying that translation vector.
    - Keep all portal pixels unchanged; only the grey arrow shapes move.

    Key insight:
    - The portal is an anchor, so the arrow keeps its full shape, orientation, and local offset from the portal; the puzzle is solved by translating the entire component by the vector between portal endpoints.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = g.copy()

    pairs = _portal_pairs(g)
    portal_lookup: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for color, (a, b) in pairs.items():
        portal_lookup[(int(a[0]), int(a[1]))] = (int(color), (int(b[0]), int(b[1])))
        portal_lookup[(int(b[0]), int(b[1]))] = (int(color), (int(a[0]), int(a[1])))

    moved_cells: list[list[tuple[int, int]]] = []
    for comp in _components4(g == GREY):
        info = _triangle_info(comp)
        if info is None:
            continue

        adjacent_portals: set[tuple[int, int]] = set()
        for r, c in info["cells"]:
            for rr, cc in _neighbors4(int(r), int(c), height, width):
                if (int(rr), int(cc)) in portal_lookup:
                    adjacent_portals.add((int(rr), int(cc)))

        if len(adjacent_portals) != 1:
            continue

        source_portal = next(iter(adjacent_portals))
        target_portal = portal_lookup[source_portal][1]
        dr = int(target_portal[0] - source_portal[0])
        dc = int(target_portal[1] - source_portal[1])

        for r, c in info["cells"]:
            out[int(r), int(c)] = BLACK
        moved_cells.append([(int(r + dr), int(c + dc)) for r, c in info["cells"]])

    for cells in moved_cells:
        for r, c in cells:
            if 0 <= r < height and 0 <= c < width:
                out[int(r), int(c)] = GREY

    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(16, 24)
    arrow_count = random.randint(2, 4)
    colors = random.sample(PORTAL_COLORS, arrow_count)

    grid = np.full((height, width), BLACK, dtype=int)
    blocked: set[tuple[int, int]] = set()
    groups: list[dict[str, object]] = []

    for color in colors:
        side = random.choice(SIDES)
        mask_name = random.choice(list(TRIANGLE_MASKS))
        mask = TRIANGLE_MASKS[mask_name]
        top_lefts = _valid_top_lefts(height, width, side)
        random.shuffle(top_lefts)
        placed = False

        for src_top in top_lefts[:160]:
            src_cells = _mask_cells(mask, src_top)
            offsets = _portal_offsets(mask, side)
            random.shuffle(offsets)

            for off_r, off_c in offsets:
                portal_a = (int(src_top[0] + off_r), int(src_top[1] + off_c))
                source_zone = set(src_cells) | {portal_a}
                if source_zone & blocked:
                    continue

                dst_tops = _valid_top_lefts(height, width, side)
                random.shuffle(dst_tops)
                for dst_top in dst_tops[:220]:
                    if dst_top == src_top:
                        continue

                    dst_cells = _mask_cells(mask, dst_top)
                    portal_b = (int(dst_top[0] + off_r), int(dst_top[1] + off_c))
                    target_zone = set(dst_cells) | {portal_b}
                    reserved = source_zone | target_zone

                    if reserved & blocked:
                        continue
                    if not _zones_separated(source_zone, target_zone):
                        continue

                    # Prevent the unused portal endpoint from touching the source arrow.
                    bad_touch = False
                    for r, c in src_cells:
                        for rr, cc in _neighbors4(r, c, height, width):
                            if (int(rr), int(cc)) == portal_b:
                                bad_touch = True
                                break
                        if bad_touch:
                            break
                    if bad_touch:
                        continue

                    for r, c in src_cells:
                        grid[int(r), int(c)] = GREY
                    grid[portal_a] = int(color)
                    grid[portal_b] = int(color)

                    groups.append(
                        {
                            "source": tuple(sorted(source_zone - {portal_a})),
                            "target": tuple(sorted(dst_cells)),
                        }
                    )
                    blocked |= _expand_cells(source_zone | target_zone, height, width)
                    placed = True
                    break

                if placed:
                    break
            if placed:
                break

        if not placed:
            return None

    expected = grid.copy()
    for group in groups:
        for r, c in group["source"]:
            expected[int(r), int(c)] = BLACK
    for group in groups:
        for r, c in group["target"]:
            expected[int(r), int(c)] = GREY

    triangle_count = sum(
        1
        for comp in _components4(grid == GREY)
        if _triangle_info(comp) is not None
    )
    if triangle_count != arrow_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-arrow input")
