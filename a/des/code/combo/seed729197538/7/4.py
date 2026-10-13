from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MAGENTA = 6
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 7, 8, 9]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_isolated_anchor(g: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = g.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each marked shape in the direction exposed by its magenta pixel until that
    magenta position lands on the matching same-color anchor pixel.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = BG

    comps = _components(g != bg)
    objects: list[dict[str, object]] = []
    out = g.copy()

    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        if MAGENTA not in colors:
            continue
        assert len(colors) == 2, "marked object must have exactly one body color plus magenta"
        magenta_cells = [(r, c) for r, c in comp if int(g[r, c]) == MAGENTA]
        assert len(magenta_cells) == 1, "each object must contain exactly one magenta pixel"
        body_colors = [color for color in colors if color != MAGENTA]
        assert len(body_colors) == 1
        body = int(body_colors[0])
        marker = magenta_cells[0]
        comp_set = set(comp)

        exposed: list[tuple[int, int]] = []
        for dr, dc in DIRS:
            nr, nc = marker[0] + dr, marker[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in comp_set and int(g[nr, nc]) == bg:
                exposed.append((dr, dc))
        assert len(exposed) == 1, "magenta marker must expose exactly one direction"
        dr, dc = exposed[0]

        anchor = None
        rr, cc = marker[0] + dr, marker[1] + dc
        while 0 <= rr < h and 0 <= cc < w:
            val = int(g[rr, cc])
            if val != bg:
                if val == body and (rr, cc) not in comp_set and _is_isolated_anchor(g, rr, cc, bg):
                    anchor = (rr, cc)
                break
            rr += dr
            cc += dc
        assert anchor is not None, "no matching anchor found on the exposed ray"
        objects.append({"coords": comp, "body": body, "marker": marker, "anchor": anchor})

    assert objects, "expected at least one marked object"

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = bg

    for obj in objects:
        body = int(obj["body"])
        marker_r, marker_c = obj["marker"]
        anchor_r, anchor_c = obj["anchor"]
        dr = anchor_r - marker_r
        dc = anchor_c - marker_c
        for r, c in obj["coords"]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = body
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                pt = (nr, nc)
                if pt not in seen:
                    seen.add(pt)
                    q.append(pt)
    return len(seen) == int(mask.sum())


def _symmetric(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int, tuple[int, int]]]:
    h, w = mask.shape
    out: list[tuple[int, int, tuple[int, int]]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        exposed: list[tuple[int, int]] = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not bool(mask[nr, nc]):
                exposed.append((dr, dc))
        if len(exposed) != 1:
            continue
        dr, dc = exposed[0]
        rr, cc = r + dr, c + dc
        blocked = False
        while 0 <= rr < h and 0 <= cc < w:
            if bool(mask[rr, cc]):
                blocked = True
                break
            rr += dr
            cc += dc
        if not blocked:
            out.append((int(r), int(c), (dr, dc)))
    return out


def _random_blob_mask() -> np.ndarray:
    for _ in range(400):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        max_area = min(h * w - 1, 11)
        min_area = max(5, (11 * h * w + 19) // 20)
        if min_area > max_area:
            continue
        target = random.randint(min_area, max_area)

        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if min(mask.shape) < 3:
            continue
        if not _mask_connected(mask):
            continue
        if _symmetric(mask):
            continue
        if _marker_candidates(mask):
            return mask
    raise RuntimeError("failed to generate a valid blob mask")


def _oriented_shape(direction: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(400):
        base = _random_blob_mask()
        variants = [np.rot90(base, k).copy() for k in range(4)]
        random.shuffle(variants)
        for mask in variants:
            candidates = [(r, c) for r, c, d in _marker_candidates(mask) if d == direction]
            if candidates:
                return mask, random.choice(candidates)
    raise RuntimeError("failed to orient a blob for the requested direction")


def _dilate_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                out.add((nr, nc))
    return out


def _translated_visible(
    cells: list[tuple[int, int]], shift: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = shift
    out = []
    for r, c in cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _render_input(shape: tuple[int, int], placements: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    g = np.full((h, w), BG, dtype=int)
    for spec in placements:
        color = int(spec["color"])
        origin_r, origin_c = spec["origin"]
        marker_r, marker_c = spec["marker"]
        for rr, cc in spec["cells"]:
            g[origin_r + rr, origin_c + cc] = color
        g[origin_r + marker_r, origin_c + marker_c] = MAGENTA
        anchor_r, anchor_c = spec["anchor"]
        g[anchor_r, anchor_c] = color
    return g


def _render_output(shape: tuple[int, int], placements: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    g = np.full((h, w), BG, dtype=int)
    for spec in placements:
        color = int(spec["color"])
        origin_r, origin_c = spec["origin"]
        marker_r, marker_c = spec["marker"]
        anchor_r, anchor_c = spec["anchor"]
        shift = (anchor_r - (origin_r + marker_r), anchor_c - (origin_c + marker_c))
        for rr, cc in spec["cells"]:
            nr, nc = origin_r + rr + shift[0], origin_c + cc + shift[1]
            if 0 <= nr < h and 0 <= nc < w:
                g[nr, nc] = color
    return g


def _placement_options(
    board_shape: tuple[int, int],
    mask: np.ndarray,
    marker: tuple[int, int],
    direction: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    reserved_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
    want_clipped: bool | None,
) -> list[dict[str, object]]:
    h, w = board_shape
    mh, mw = mask.shape
    cells = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    area = len(cells)
    dr, dc = direction
    options: list[dict[str, object]] = []

    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            global_cells = {(r0 + rr, c0 + cc) for rr, cc in cells}
            if global_cells & reserved_input:
                continue

            marker_global = (r0 + marker[0], c0 + marker[1])
            if dr == -1:
                max_dist = marker_global[0]
            elif dr == 1:
                max_dist = h - 1 - marker_global[0]
            elif dc == -1:
                max_dist = marker_global[1]
            else:
                max_dist = w - 1 - marker_global[1]
            if max_dist < 2:
                continue

            distances = list(range(2, max_dist + 1))
            random.shuffle(distances)
            for dist in distances:
                anchor = (marker_global[0] + dr * dist, marker_global[1] + dc * dist)
                if anchor in reserved_input:
                    continue

                path_cells = {
                    (marker_global[0] + dr * step, marker_global[1] + dc * step)
                    for step in range(1, dist)
                }
                blocked = False
                for pt in path_cells:
                    if pt in reserved_input or pt in global_cells:
                        blocked = True
                        break
                if blocked:
                    continue

                current_occupied = occupied_input | global_cells
                for nr, nc in _neighbors4(*anchor):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in current_occupied:
                        blocked = True
                        break
                if blocked:
                    continue

                visible = _translated_visible(
                    [(r0 + rr, c0 + cc) for rr, cc in cells], (dr * dist, dc * dist), board_shape
                )
                if len(visible) < max(3, area // 2):
                    continue
                clipped = len(visible) < area
                if want_clipped is not None and clipped != want_clipped:
                    continue
                visible_set = set(visible)
                if visible_set & reserved_output:
                    continue

                options.append(
                    {
                        "cells": cells,
                        "origin": (r0, c0),
                        "marker": marker,
                        "anchor": anchor,
                        "direction": direction,
                        "path": path_cells,
                        "visible": visible_set,
                    }
                )
    random.shuffle(options)
    return options


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(11, 18)
        w = random.randint(11, 18)
        n_objects = random.randint(1, 4)
        colors = random.sample(PALETTE, n_objects)

        placements: list[dict[str, object]] = []
        occupied_input: set[tuple[int, int]] = set()
        reserved_input: set[tuple[int, int]] = set()
        occupied_output: set[tuple[int, int]] = set()
        reserved_output: set[tuple[int, int]] = set()

        ok = True
        for color in colors:
            direction = random.choice(DIRS)
            mask, marker = _oriented_shape(direction)
            want_clipped = random.random() < 0.35
            options = _placement_options(
                (h, w),
                mask,
                marker,
                direction,
                occupied_input,
                reserved_input,
                occupied_output,
                reserved_output,
                want_clipped,
            )
            if not options:
                options = _placement_options(
                    (h, w),
                    mask,
                    marker,
                    direction,
                    occupied_input,
                    reserved_input,
                    occupied_output,
                    reserved_output,
                    None,
                )
            if not options:
                ok = False
                break

            spec = random.choice(options)
            spec["color"] = color
            placements.append(spec)

            input_cells = {
                (spec["origin"][0] + rr, spec["origin"][1] + cc)
                for rr, cc in spec["cells"]
            }
            input_cells.add(spec["anchor"])
            occupied_input |= input_cells
            reserved_input |= _dilate_cells(input_cells, (h, w))
            reserved_input |= spec["path"]
            occupied_output |= spec["visible"]
            reserved_output |= _dilate_cells(spec["visible"], (h, w))

        if not ok:
            continue

        inp = _render_input((h, w), placements)
        expected = _render_output((h, w), placements)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid input grid")
