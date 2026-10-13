from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _touches_local_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _fill_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = int(color)


def _can_place_bbox(used: np.ndarray, top: int, left: int, h: int, w: int, margin: int = 1) -> bool:
    hh, ww = used.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(hh, top + h + margin)
    c1 = min(ww, left + w + margin)
    return not bool(np.any(used[r0:r1, c0:c1]))


def _mark_used(used: np.ndarray, top: int, left: int, h: int, w: int, margin: int = 1) -> None:
    hh, ww = used.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(hh, top + h + margin)
    c1 = min(ww, left + w + margin)
    used[r0:r1, c0:c1] = True


def _pick_bbox_slot(used: np.ndarray, h: int, w: int, tries: int = 120) -> tuple[int, int] | None:
    hh, ww = used.shape
    if h > hh or w > ww:
        return None
    for _ in range(tries):
        top = random.randint(0, hh - h)
        left = random.randint(0, ww - w)
        if _can_place_bbox(used, top, left, h, w):
            return top, left
    return None


def _rotation_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {(m.shape, m.tobytes()) for m in (np.rot90(mask, k) for k in range(4))}


def _random_shape(area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in DIR4
                if (r + dr, c + dc) not in cells
            }
            cells.add(random.choice(tuple(frontier)))
        coords = np.array(sorted(cells), dtype=int)
        mask = _coords_to_mask(coords)
        if mask.shape[0] > 4 or mask.shape[1] > 4:
            continue
        if bool(np.all(mask)):
            continue
        if len(_rotation_keys(mask)) != 4:
            continue
        return mask
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinsert scattered same-color fragments into the missing holes of their template rectangles.

    Input structure:
    - The background is the most common color.
    - Several large monochrome rectangles of different colors appear on the grid.
    - Each large rectangle contains one small connected background hole that does not touch the rectangle border.
    - Elsewhere on the background, one small monochrome fragment of the same color is scattered for each rectangle.

    Rule:
    - Group objects by color, and treat the largest same-color component as the template rectangle.
    - Extract the unique interior background hole from that template rectangle.
    - Find the smaller same-color fragment whose shape matches the hole after one of the four right-angle rotations.
    - Fill the hole with the rectangle color and erase the fragment's original location back to background.

    Key insight:
    - Color tells which fragment belongs to which rectangle, but the actual placement is determined by the only rotation that makes the fragment mask equal the hole mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in (int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        if len(comps) < 2:
            continue
        comps.sort(key=len, reverse=True)
        template = comps[0]
        tr0, tc0, tr1, tc1 = _bbox(template)
        sub = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
        if np.any((sub != bg) & (sub != color)):
            continue

        bg_regions = _components(sub == bg)
        hole_regions = [pts for pts in bg_regions if not _touches_local_border(pts, sub.shape)]
        if len(hole_regions) != 1:
            continue

        hole_coords = hole_regions[0]
        hole_mask = _coords_to_mask(hole_coords)
        matches: list[np.ndarray] = []
        for frag in comps[1:]:
            if len(frag) != len(hole_coords):
                continue
            frag_mask = _coords_to_mask(frag)
            rotations = [k for k in range(4) if np.array_equal(np.rot90(frag_mask, k), hole_mask)]
            if len(rotations) == 1:
                matches.append(frag)

        if len(matches) != 1:
            continue

        frag = matches[0]
        out[tr0 + hole_coords[:, 0], tc0 + hole_coords[:, 1]] = color
        out[frag[:, 0], frag[:, 1]] = bg

    return out


GEN_ATTEMPTS = 400
NUM_TEMPLATE_CHOICES = (2, 3, 4)
RECT_H_RANGE = (6, 10)
RECT_W_RANGE = (6, 10)
HOLE_AREA_RANGE = (3, 6)


def _bg_is_most_common(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)]) == int(bg)


def _gen_attempt() -> np.ndarray | None:
    num_templates = random.choice(NUM_TEMPLATE_CHOICES)
    min_side = min(30, 16 + 2 * num_templates)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    bg = random.randint(0, 9)
    colors = random.sample([c for c in range(10) if c != bg], num_templates)

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    for color in colors:
        hole_area = random.randint(HOLE_AREA_RANGE[0], HOLE_AREA_RANGE[1])
        base_shape = _random_shape(hole_area)
        if base_shape is None:
            return None

        hole_mask = np.rot90(base_shape, random.randint(0, 3))
        hh, hw = hole_mask.shape

        rect_h_min = max(RECT_H_RANGE[0], hh + 2)
        rect_h_max = min(RECT_H_RANGE[1], h)
        rect_w_min = max(RECT_W_RANGE[0], hw + 2)
        rect_w_max = min(RECT_W_RANGE[1], w)
        if rect_h_min > rect_h_max or rect_w_min > rect_w_max:
            return None

        rect_h = random.randint(rect_h_min, rect_h_max)
        rect_w = random.randint(rect_w_min, rect_w_max)
        rect_slot = _pick_bbox_slot(used, rect_h, rect_w)
        if rect_slot is None:
            return None
        rect_top, rect_left = rect_slot
        _mark_used(used, rect_top, rect_left, rect_h, rect_w)

        out[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = color
        inp[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = color

        hole_top = random.randint(1, rect_h - hh - 1)
        hole_left = random.randint(1, rect_w - hw - 1)
        hole_patch = inp[
            rect_top + hole_top : rect_top + hole_top + hh,
            rect_left + hole_left : rect_left + hole_left + hw,
        ]
        hole_patch[hole_mask] = bg

        frag_mask = np.rot90(hole_mask, random.randint(0, 3))
        fh, fw = frag_mask.shape
        frag_slot = _pick_bbox_slot(used, fh, fw)
        if frag_slot is None:
            return None
        frag_top, frag_left = frag_slot
        _mark_used(used, frag_top, frag_left, fh, fw)
        _fill_mask(inp, frag_mask, frag_top, frag_left, color)

    if np.array_equal(inp, out):
        return None
    if not _bg_is_most_common(inp, bg):
        return None
    if not _bg_is_most_common(out, bg):
        return None

    for color in colors:
        comps = _components(inp == color)
        if len(comps) != 2:
            return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
