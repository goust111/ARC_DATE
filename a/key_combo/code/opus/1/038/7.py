from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEP = 4
TOP_COLORS = (1, 2, 3)
BOTTOM_COLORS = (5, 7, 8)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------- solver helpers ---------------


def _monochrome_components(grid: np.ndarray, bg: int = BG) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    labels = np.full((h, w), -1, dtype=int)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            comp_id = len(components)
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                labels[rr, cc] = comp_id
                for dr, dc in N4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                }
            )

    return components, labels


def _find_separator_row(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    rows = np.where(np.all(g == SEP, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full separator row"
    row = int(rows[0])
    assert 0 < row < g.shape[0] - 1, "separator row must split the grid into two non-empty halves"
    return row


def _stitch_halves(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    top = np.asarray(top, dtype=int)
    bottom = np.asarray(bottom, dtype=int)
    top_components, top_labels = _monochrome_components(top, bg=BG)
    bottom_components, bottom_labels = _monochrome_components(bottom, bg=BG)

    out_h = max(int(top.shape[0]), int(bottom.shape[0]) + 1)
    out_w = max(int(top.shape[1]), int(bottom.shape[1]) + 1)
    out = np.full((out_h, out_w), SEP, dtype=int)

    for r in range(out_h):
        for c in range(out_w):
            top_here = r < top.shape[0] and c < top.shape[1] and int(top[r, c]) != BG
            bottom_here = (
                r >= 1
                and c >= 1
                and (r - 1) < bottom.shape[0]
                and (c - 1) < bottom.shape[1]
                and int(bottom[r - 1, c - 1]) != BG
            )

            if top_here and bottom_here:
                top_id = int(top_labels[r, c])
                bottom_id = int(bottom_labels[r - 1, c - 1])
                top_area = int(top_components[top_id]["area"])
                bottom_area = int(bottom_components[bottom_id]["area"])
                assert top_area != bottom_area, "overlap tie is outside the puzzle definition"
                out[r, c] = int(top[r, c]) if top_area < bottom_area else int(bottom[r - 1, c - 1])
            elif top_here:
                out[r, c] = int(top[r, c])
            elif bottom_here:
                out[r, c] = int(bottom[r - 1, c - 1])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diagonally stitch the two separator-split halves and recolor each overlap cell by the smaller original component area.

    Input structure:
    - A single full yellow row divides the grid into an upper half and a lower half.
    - Each half has black background and several single-color 4-connected objects.
    - Upper-half colors and lower-half colors come from disjoint palettes, so every cross-half overlap is color-distinguishable.

    Rule:
    - Remove the yellow separator row and detect 4-connected monochromatic components separately in the upper and lower halves.
    - Build a yellow output canvas in the common stitched coordinate system: place the upper half at (0, 0) and the lower half one row lower
      and one column to the right, treating black as transparent.
    - For any output cell covered by both halves, trace each pixel back to its original component and keep the color of the component with
      the smaller area; non-overlapping foreground pixels keep their original colors.

    Key insight:
    - The overlap is created only by the diagonal offset, but each overlapped pixel must be resolved using a global property of its source
      component before stitching: the original connected-component area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_row = _find_separator_row(g)
    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    return _stitch_halves(top, bottom)


# --------------- generator helpers ---------------


GEN_RETRIES = 500
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 80
MIN_OVERLAPS = 2
MAX_OVERLAPS = 8


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    cells = _normalize_cells(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_connected_shape(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in N4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)

            options = list(frontier)
            random.shuffle(options)
            placed = False
            for nxt in options:
                trial = _normalize_cells(cells | {nxt})
                height = max(r for r, _ in trial) + 1
                width = max(c for _, c in trial) + 1
                if height <= max_h and width <= max_w:
                    cells = trial
                    placed = True
                    break
            if not placed:
                break

        if len(cells) != area:
            continue

        mask = _cells_to_mask(cells)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        return mask.astype(bool)

    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - pad)
        r2 = min(h, rr + pad + 1)
        c1 = max(0, cc - pad)
        c2 = min(w, cc + pad + 1)
        if bool(occupied[r1:r2, c1:c2].any()):
            return False
    return True


def _place_masks(height: int, width: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        candidates = [(r, c) for r in range(height - mh + 1) for c in range(width - mw + 1)]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            if not _can_place(occupied, mask, int(r0), int(c0), pad=1):
                continue
            placements[idx] = (int(r0), int(c0))
            coords = np.argwhere(mask)
            occupied[coords[:, 0] + int(r0), coords[:, 1] + int(c0)] = True
            placed = True
            break

        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if (r, c) is not None] if all(p is not None for p in placements) else None


def _render_half(height: int, width: int, masks: list[np.ndarray], placements: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for mask, (r0, c0), color in zip(masks, placements, colors):
        target = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        assert not bool(np.any(target[mask] != BG))
        target[mask] = int(color)
    return grid


def _assign_colors(palette: tuple[int, ...], count: int) -> list[int]:
    colors = list(palette)
    random.shuffle(colors)
    assigned = colors[: min(len(colors), count)]
    while len(assigned) < count:
        assigned.append(random.choice(palette))
    random.shuffle(assigned)
    return assigned


def _sample_half_areas(n_top: int, n_bottom: int) -> tuple[list[int], list[int]]:
    total = n_top + n_bottom
    all_areas = sorted(random.sample(range(2, 10), total))

    top_areas = [all_areas[0], all_areas[-2]]
    bottom_areas = [all_areas[1], all_areas[-1]]
    middle = all_areas[2:-2]
    random.shuffle(middle)

    while len(top_areas) < n_top:
        top_areas.append(middle.pop())
    while len(bottom_areas) < n_bottom:
        bottom_areas.append(middle.pop())

    random.shuffle(top_areas)
    random.shuffle(bottom_areas)
    return top_areas, bottom_areas


def _build_half(height: int, width: int, areas: list[int], palette: tuple[int, ...]) -> np.ndarray | None:
    max_h = min(4, height)
    max_w = min(5, width)
    masks: list[np.ndarray] = []

    for area in areas:
        mask = _random_connected_shape(int(area), max_h=max_h, max_w=max_w)
        if mask is None:
            return None
        masks.append(mask)

    placements = _place_masks(height, width, masks)
    if placements is None:
        return None

    colors = _assign_colors(palette, len(areas))
    return _render_half(height, width, masks, placements, colors)


def _analyze_case(inp: np.ndarray) -> dict:
    sep_row = _find_separator_row(inp)
    top = np.asarray(inp[:sep_row, :], dtype=int)
    bottom = np.asarray(inp[sep_row + 1 :, :], dtype=int)
    top_components, top_labels = _monochrome_components(top, bg=BG)
    bottom_components, bottom_labels = _monochrome_components(bottom, bg=BG)

    out_h = max(int(top.shape[0]), int(bottom.shape[0]) + 1)
    out_w = max(int(top.shape[1]), int(bottom.shape[1]) + 1)

    overlap_pairs: dict[tuple[int, int], int] = {}
    overlap_cells = 0
    top_wins = 0
    bottom_wins = 0
    top_only = 0
    bottom_only = 0

    for r in range(out_h):
        for c in range(out_w):
            top_here = r < top.shape[0] and c < top.shape[1] and int(top[r, c]) != BG
            bottom_here = (
                r >= 1
                and c >= 1
                and (r - 1) < bottom.shape[0]
                and (c - 1) < bottom.shape[1]
                and int(bottom[r - 1, c - 1]) != BG
            )
            if top_here and bottom_here:
                overlap_cells += 1
                top_id = int(top_labels[r, c])
                bottom_id = int(bottom_labels[r - 1, c - 1])
                overlap_pairs[(top_id, bottom_id)] = overlap_pairs.get((top_id, bottom_id), 0) + 1
                top_area = int(top_components[top_id]["area"])
                bottom_area = int(bottom_components[bottom_id]["area"])
                if top_area < bottom_area:
                    top_wins += 1
                elif bottom_area < top_area:
                    bottom_wins += 1
            elif top_here:
                top_only += 1
            elif bottom_here:
                bottom_only += 1

    return {
        "top": top,
        "bottom": bottom,
        "top_components": top_components,
        "bottom_components": bottom_components,
        "overlap_cells": overlap_cells,
        "overlap_pairs": overlap_pairs,
        "top_wins": top_wins,
        "bottom_wins": bottom_wins,
        "top_only": top_only,
        "bottom_only": bottom_only,
    }


def _valid_generated_case(inp: np.ndarray) -> bool:
    try:
        info = _analyze_case(inp)
    except AssertionError:
        return False

    top = info["top"]
    bottom = info["bottom"]
    top_components = info["top_components"]
    bottom_components = info["bottom_components"]

    if not (3 <= len(top_components) <= 4 and 3 <= len(bottom_components) <= 4):
        return False

    if set(int(v) for v in np.unique(top) if int(v) != BG) != set(TOP_COLORS):
        return False
    if set(int(v) for v in np.unique(bottom) if int(v) != BG) != set(BOTTOM_COLORS):
        return False

    all_areas = [int(comp["area"]) for comp in top_components + bottom_components]
    if len(all_areas) != len(set(all_areas)):
        return False

    if not (MIN_OVERLAPS <= int(info["overlap_cells"]) <= MAX_OVERLAPS):
        return False
    if len(info["overlap_pairs"]) < 2:
        return False
    if int(info["top_wins"]) == 0 or int(info["bottom_wins"]) == 0:
        return False
    if int(info["top_only"]) == 0 or int(info["bottom_only"]) == 0:
        return False

    try:
        out = main(inp)
    except AssertionError:
        return False

    if out.shape != (max(top.shape[0], bottom.shape[0] + 1), top.shape[1] + 1):
        return False
    return bool(np.all((out == SEP) | (out == BG) | np.isin(out, TOP_COLORS + BOTTOM_COLORS)))


def _generate_candidate() -> np.ndarray | None:
    top_h, bottom_h = random.choice(((5, 6), (6, 5), (6, 7), (7, 6)))
    width = random.randint(10, 12)
    n_top = random.randint(3, 4)
    n_bottom = random.randint(3, 4)
    top_areas, bottom_areas = _sample_half_areas(n_top, n_bottom)

    for _ in range(PLACEMENT_RETRIES):
        top = _build_half(top_h, width, top_areas, TOP_COLORS)
        bottom = _build_half(bottom_h, width, bottom_areas, BOTTOM_COLORS)
        if top is None or bottom is None:
            continue

        grid = np.full((top_h + bottom_h + 1, width), SEP, dtype=int)
        grid[:top_h, :] = top
        grid[top_h + 1 :, :] = bottom

        if _valid_generated_case(grid):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid diagonal-overlap stitching input")
