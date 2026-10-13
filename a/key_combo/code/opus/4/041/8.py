from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    clip: bool = False,
) -> None:
    h, w = patch.shape
    rr0 = int(r0)
    cc0 = int(c0)
    rr1 = rr0 + h
    cc1 = cc0 + w
    if not clip:
        if rr0 < 0 or cc0 < 0 or rr1 > dst.shape[0] or cc1 > dst.shape[1]:
            raise ValueError("patch out of bounds")
        view = dst[rr0:rr1, cc0:cc1]
        if transparent is None:
            view[:] = patch
        else:
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return

    dr0 = max(0, rr0)
    dc0 = max(0, cc0)
    dr1 = min(dst.shape[0], rr1)
    dc1 = min(dst.shape[1], cc1)
    if dr0 >= dr1 or dc0 >= dc1:
        return
    pr0 = dr0 - rr0
    pc0 = dc0 - cc0
    pr1 = pr0 + (dr1 - dr0)
    pc1 = pc0 + (dc1 - dc0)
    sub = patch[pr0:pr1, pc0:pc1]
    view = dst[dr0:dr1, dc0:dc1]
    if transparent is None:
        view[:] = sub
    else:
        mask = sub != int(transparent)
        view[mask] = sub[mask]


def find_components(
    grid: np.ndarray,
    *,
    bg: int = BG,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    offsets = DIR4 if connectivity == 4 else DIR8
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in offsets:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) == int(bg):
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            out.append(
                {
                    "color": color if monochromatic else None,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each matched sprite and duplicate it by mirroring along an axis encoded by its border legend.

    Input structure:
    - The background is black.
    - The interior contains 2-3 disconnected 4-connected multi-color sprites, each using exactly two foreground colors.
    - The outer border contains 2-3 separate two-cell domino legends; horizontal dominoes lie on the top or bottom edge, and vertical dominoes lie on the left or right edge.

    Rule:
    - Read each legend as [replacement color, source color], using left-to-right order for horizontal legends and top-to-bottom order for vertical legends.
    - Match a legend to the unique sprite that contains the source color.
    - Recolor every source-color pixel of that sprite to the replacement color.
    - If the legend is on the top or bottom edge, keep the recolored sprite in place and append its vertical mirror directly below its bbox.
    - If the legend is on the left or right edge, keep the recolored sprite in place and append its horizontal mirror directly to the right of its bbox.
    - Erase all legend pixels to black and clip any mirrored pixels that would leave the grid.

    Key insight:
    - A single border domino carries two independent signals at once: its ordered colors define the recoloring map, and the edge it sits on chooses whether the matched sprite mirrors downward or rightward.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=False)
    legends: list[dict] = []
    legend_cells: list[tuple[int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        if int(comp["area"]) != 2:
            continue
        pts = [tuple(map(int, p)) for p in coords]
        (r1, c1), (r2, c2) = pts
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            continue
        if r1 == r2 and r1 in (0, h - 1):
            ordered = sorted(pts, key=lambda p: p[1])
            axis = "down"
        elif c1 == c2 and c1 in (0, w - 1):
            ordered = sorted(pts, key=lambda p: p[0])
            axis = "right"
        else:
            continue
        colors = [int(g[r, c]) for r, c in ordered]
        legends.append(
            {
                "cells": ordered,
                "replacement": colors[0],
                "source": colors[1],
                "axis": axis,
            }
        )
        legend_cells.extend(ordered)

    work = g.copy()
    for r, c in legend_cells:
        work[r, c] = BG

    sprites = find_components(work, bg=BG, connectivity=4, monochromatic=False)
    sprite_colors = []
    for sprite in sprites:
        vals = {int(work[r, c]) for r, c in sprite["coords"]}
        sprite_colors.append(vals)

    out = work.copy()
    matched_ids: set[int] = set()
    transformed: list[tuple[tuple[int, int], np.ndarray]] = []

    for legend in legends:
        matches = [i for i, vals in enumerate(sprite_colors) if int(legend["source"]) in vals]
        if len(matches) != 1:
            continue
        idx = matches[0]
        matched_ids.add(idx)
        sprite = sprites[idx]
        r0, c0, r1, c1 = sprite["bbox"]
        patch = crop_bbox(work, sprite["bbox"])
        patch[patch == int(legend["source"])] = int(legend["replacement"])
        if legend["axis"] == "down":
            patch = np.vstack([patch, np.flipud(patch)])
        else:
            patch = np.hstack([patch, np.fliplr(patch)])
        transformed.append(((r0, c0), patch))

    for idx in matched_ids:
        coords = sprites[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for pos, patch in transformed:
        paste(out, patch, pos[0], pos[1], transparent=BG, clip=True)

    return out


def _normalize_patch(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_polyomino(area: int) -> np.ndarray:
    for _ in range(100):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        if box_h * box_w < area:
            continue
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_patch(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        k = random.randrange(4)
        mask = np.rot90(mask, k)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = _normalize_patch(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        return mask
    raise RuntimeError("failed to sample polyomino")


def _make_sprite(primary: int, source: int) -> np.ndarray:
    for _ in range(100):
        area = random.randint(5, 9)
        mask = _random_polyomino(area)
        cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
        max_source = min(3, max(1, (area - 1) // 2))
        source_count = random.randint(1, max_source)
        source_cells = set(random.sample(cells, source_count))
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = int(primary)
        for r, c in source_cells:
            patch[r, c] = int(source)
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    raise RuntimeError("failed to sample sprite")


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _axis_plan(count: int) -> list[str]:
    if count == 2:
        axes = ["down", "right"]
    else:
        axes = ["down", "right", random.choice(["down", "right"])]
    random.shuffle(axes)
    return axes


def _place_sprites(specs: list[dict], h: int, w: int) -> bool:
    order = sorted(range(len(specs)), key=lambda i: specs[i]["out_h"] * specs[i]["out_w"], reverse=True)
    occupied: list[tuple[int, int, int, int]] = []
    for idx in order:
        spec = specs[idx]
        max_r = h - spec["out_h"] - 2
        max_c = w - spec["out_w"] - 2
        if max_r < 2 or max_c < 2:
            return False
        positions = [(r, c) for r in range(2, max_r + 1) for c in range(2, max_c + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            box = (r0, c0, r0 + spec["out_h"] - 1, c0 + spec["out_w"] - 1)
            if any(_rect_overlap(box, other, pad=1) for other in occupied):
                continue
            spec["pos"] = (r0, c0)
            occupied.append(box)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_legends(specs: list[dict], h: int, w: int) -> bool:
    used: set[tuple[int, int]] = set()
    used_adjacent: set[tuple[int, int]] = set()

    for spec in specs:
        sides = ["top", "bottom"] if spec["axis"] == "down" else ["left", "right"]
        random.shuffle(sides)
        placed = False
        for side in sides:
            if side in ("top", "bottom"):
                row = 0 if side == "top" else h - 1
                starts = list(range(1, w - 2))
                random.shuffle(starts)
                for c0 in starts:
                    cells = [(row, c0), (row, c0 + 1)]
                    if any(cell in used_adjacent for cell in cells):
                        continue
                    spec["legend"] = cells
                    placed = True
                    break
            else:
                col = 0 if side == "left" else w - 1
                starts = list(range(1, h - 2))
                random.shuffle(starts)
                for r0 in starts:
                    cells = [(r0, col), (r0 + 1, col)]
                    if any(cell in used_adjacent for cell in cells):
                        continue
                    spec["legend"] = cells
                    placed = True
                    break
            if placed:
                for r, c in spec["legend"]:
                    used.add((r, c))
                    used_adjacent.add((r, c))
                    for nr, nc in neighbors4(r, c):
                        if 0 <= nr < h and 0 <= nc < w:
                            used_adjacent.add((nr, nc))
                break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.choice([2, 3, 3])
    if sprite_count == 2:
        h = random.randint(15, 20)
        w = random.randint(15, 20)
    else:
        h = random.randint(18, 24)
        w = random.randint(18, 24)

    axes = _axis_plan(sprite_count)
    colors = random.sample(PALETTE, 3 * sprite_count)
    specs: list[dict] = []
    for i, axis in enumerate(axes):
        primary, source, replacement = colors[3 * i : 3 * i + 3]
        patch = _make_sprite(primary, source)
        out_h = patch.shape[0] * (2 if axis == "down" else 1)
        out_w = patch.shape[1] * (2 if axis == "right" else 1)
        specs.append(
            {
                "axis": axis,
                "primary": primary,
                "source": source,
                "replacement": replacement,
                "patch": patch,
                "out_h": out_h,
                "out_w": out_w,
            }
        )

    if not _place_sprites(specs, h, w):
        return None
    if not _place_legends(specs, h, w):
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0 = spec["pos"]
        paste(grid, spec["patch"], r0, c0, transparent=BG)
    for spec in specs:
        (r1, c1), (r2, c2) = spec["legend"]
        grid[r1, c1] = int(spec["replacement"])
        grid[r2, c2] = int(spec["source"])

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
