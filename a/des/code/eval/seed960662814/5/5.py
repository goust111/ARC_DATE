import math
import random
from collections import Counter, deque

import numpy as np


OUTLINE_COLOR = 5
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int, forbidden: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if forbidden is None:
        forbidden = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or forbidden[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or forbidden[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "size": int(coords.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _flood_reachable(not_blocked: np.ndarray) -> np.ndarray:
    h, w = not_blocked.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not_blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not_blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and not_blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _outline_and_interior(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    bg = _background_color(grid)
    best = None
    for comp in _monochrome_components(grid, bg):
        comp_mask = np.zeros_like(grid, dtype=bool)
        comp_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True
        reachable = _flood_reachable(~comp_mask)
        interior = (grid == bg) & ~reachable
        if not interior.any():
            continue
        score = (int(interior.sum()), int(comp["size"]))
        if best is None or score > best[0]:
            best = (score, comp_mask, interior, int(comp["color"]))
    assert best is not None, "failed to locate the closed outline"
    return best[1], best[2], best[3]


def _outline_anchor_map(outline: np.ndarray, interior: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    h, w = outline.shape
    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    coords = np.argwhere(outline)
    for r, c in coords:
        inside = []
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and interior[nr, nc]:
                inside.append((int(nr), int(nc)))
        if len(inside) == 1:
            anchors[(int(r), int(c))] = inside[0]
    return anchors


def _nearest_outline_pixel(seed_coords: np.ndarray, outline_candidates: np.ndarray) -> tuple[int, int]:
    best_dist = None
    best_points: list[tuple[int, int]] = []
    for orow, ocol in outline_candidates:
        dist = int(np.min(np.abs(seed_coords[:, 0] - int(orow)) + np.abs(seed_coords[:, 1] - int(ocol))))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_points = [(int(orow), int(ocol))]
        elif dist == best_dist:
            best_points.append((int(orow), int(ocol)))
    assert best_points, "seed must have a nearest outline pixel"
    best_points = sorted(set(best_points))
    return best_points[0]


def _nearest_seed_pixel(seed_coords: np.ndarray, outline_pixel: tuple[int, int]) -> tuple[int, int]:
    pr, pc = outline_pixel
    dists = np.abs(seed_coords[:, 0] - pr) + np.abs(seed_coords[:, 1] - pc)
    best = int(np.min(dists))
    pts = [tuple(map(int, xy)) for xy in seed_coords[dists == best]]
    pts = sorted(set(pts))
    assert pts, "seed must have a nearest seed pixel"
    return pts[0]


def _k_range(anchor_coord: int, offsets: np.ndarray, low: int, high: int, period: int) -> range:
    lower = (low - anchor_coord - int(np.max(offsets))) / period
    upper = (high - anchor_coord - int(np.min(offsets))) / period
    kmin = math.ceil(lower)
    kmax = math.floor(upper)
    return range(kmin, kmax + 1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    outline, interior, outline_color = _outline_and_interior(g)
    assert outline_color != bg

    seed_forbidden = outline.copy()
    seeds = _monochrome_components(g, bg, forbidden=seed_forbidden)
    anchors = _outline_anchor_map(outline, interior)
    anchor_coords = np.array(list(anchors.keys()), dtype=int)
    assert anchor_coords.size > 0, "outline must expose anchorable edge pixels"

    out = g.copy()
    for seed in seeds:
        out[seed["coords"][:, 0], seed["coords"][:, 1]] = bg

    interior_points = np.argwhere(interior)
    r0, c0 = interior_points.min(axis=0)
    r1, c1 = interior_points.max(axis=0)

    seed_infos = []
    for seed in seeds:
        outline_pixel = _nearest_outline_pixel(seed["coords"], anchor_coords)
        anchor = anchors[outline_pixel]
        ref_pixel = _nearest_seed_pixel(seed["coords"], outline_pixel)
        bbox_r0, bbox_c0, bbox_r1, bbox_c1 = seed["bbox"]
        row_period = int(bbox_r1 - bbox_r0 + 2)
        col_period = int(bbox_c1 - bbox_c0 + 2)
        offsets = seed["coords"] - np.array(ref_pixel, dtype=int)
        seed_infos.append(
            {
                "color": int(seed["color"]),
                "size": int(seed["size"]),
                "row_period": row_period,
                "col_period": col_period,
                "anchor": anchor,
                "offsets": offsets,
            }
        )

    seed_infos.sort(key=lambda item: (-item["size"], item["color"], item["anchor"]))

    for seed in seed_infos:
        anchor_r, anchor_c = seed["anchor"]
        offsets = seed["offsets"]
        row_range = _k_range(anchor_r, offsets[:, 0], int(r0), int(r1), int(seed["row_period"]))
        col_range = _k_range(anchor_c, offsets[:, 1], int(c0), int(c1), int(seed["col_period"]))
        for kr in row_range:
            base_r = anchor_r + kr * int(seed["row_period"])
            for kc in col_range:
                base_c = anchor_c + kc * int(seed["col_period"])
                for dr, dc in offsets:
                    rr = int(base_r + int(dr))
                    cc = int(base_c + int(dc))
                    if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and interior[rr, cc] and out[rr, cc] == bg:
                        out[rr, cc] = int(seed["color"])

    return out


LOCAL_SHAPES_BY_SIZE = {
    2: [
        [(1, 0), (2, 0)],
    ],
    3: [
        [(1, 0), (2, 0), (3, 0)],
        [(1, 0), (2, 0), (2, 1)],
        [(1, 0), (2, 0), (2, -1)],
    ],
    4: [
        [(1, 0), (2, 0), (3, 0), (4, 0)],
        [(1, 0), (2, 0), (3, 0), (2, 1)],
        [(1, 0), (2, 0), (3, 0), (2, -1)],
        [(1, 0), (2, 0), (2, 1), (3, 1)],
        [(1, 0), (2, 0), (2, -1), (3, -1)],
    ],
    5: [
        [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1)],
        [(1, 0), (2, 0), (3, 0), (2, -1), (3, -1)],
        [(1, 0), (2, 0), (3, 0), (4, 0), (3, 1)],
        [(1, 0), (2, 0), (3, 0), (4, 0), (3, -1)],
        [(1, 0), (2, 0), (2, 1), (2, -1), (3, 0)],
    ],
}


def _outline_from_fill(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    outline = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not filled[nr, nc]:
                    outline[r, c] = True
                    break
    return outline


def _draw_segment(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        mask[ar, c0 : c1 + 1] = True
    else:
        r0, r1 = sorted((ar, br))
        mask[r0 : r1 + 1, ac] = True


def _append_vertex(vertices: list[tuple[int, int]], point: tuple[int, int]) -> None:
    if not vertices or vertices[-1] != point:
        vertices.append(point)


def _build_canvas(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    min_margin = 4
    outer_h = random.randint(10, h - 2 * min_margin)
    outer_w = random.randint(10, w - 2 * min_margin)
    top = random.randint(min_margin, h - min_margin - outer_h)
    left = random.randint(min_margin, w - min_margin - outer_w)
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    top_notch = None
    if random.random() < 0.75:
        x1 = random.randint(left + 2, right - 5)
        x2 = random.randint(x1 + 2, right - 2)
        d = random.randint(2, min(4, outer_h - 4))
        top_notch = (x1, x2, d)
    bottom_notch = None
    if random.random() < 0.35:
        x1 = random.randint(left + 2, right - 5)
        x2 = random.randint(x1 + 2, right - 2)
        d = random.randint(2, min(4, outer_h - 4))
        if top_notch is None or top_notch[2] + d <= outer_h - 4:
            bottom_notch = (x1, x2, d)
    left_notch = None
    if random.random() < 0.55:
        y1 = random.randint(top + 2, bottom - 5)
        y2 = random.randint(y1 + 2, bottom - 2)
        d = random.randint(2, min(4, outer_w - 4))
        left_notch = (y1, y2, d)
    right_notch = None
    if random.random() < 0.35:
        y1 = random.randint(top + 2, bottom - 5)
        y2 = random.randint(y1 + 2, bottom - 2)
        d = random.randint(2, min(4, outer_w - 4))
        if left_notch is None or left_notch[2] + d <= outer_w - 4:
            right_notch = (y1, y2, d)
    if top_notch is None and bottom_notch is None and left_notch is None and right_notch is None:
        top_notch = (left + 3, right - 3, 2)

    vertices: list[tuple[int, int]] = []
    _append_vertex(vertices, (top, left))
    if top_notch is None:
        _append_vertex(vertices, (top, right))
    else:
        x1, x2, d = top_notch
        _append_vertex(vertices, (top, x1))
        _append_vertex(vertices, (top + d, x1))
        _append_vertex(vertices, (top + d, x2))
        _append_vertex(vertices, (top, x2))
        _append_vertex(vertices, (top, right))

    if right_notch is None:
        _append_vertex(vertices, (bottom, right))
    else:
        y1, y2, d = right_notch
        _append_vertex(vertices, (y1, right))
        _append_vertex(vertices, (y1, right - d))
        _append_vertex(vertices, (y2, right - d))
        _append_vertex(vertices, (y2, right))
        _append_vertex(vertices, (bottom, right))

    if bottom_notch is None:
        _append_vertex(vertices, (bottom, left))
    else:
        x1, x2, d = bottom_notch
        _append_vertex(vertices, (bottom, x2))
        _append_vertex(vertices, (bottom - d, x2))
        _append_vertex(vertices, (bottom - d, x1))
        _append_vertex(vertices, (bottom, x1))
        _append_vertex(vertices, (bottom, left))

    if left_notch is None:
        _append_vertex(vertices, (top, left))
    else:
        y1, y2, d = left_notch
        _append_vertex(vertices, (y2, left))
        _append_vertex(vertices, (y2, left + d))
        _append_vertex(vertices, (y1, left + d))
        _append_vertex(vertices, (y1, left))
        _append_vertex(vertices, (top, left))

    outline = np.zeros((h, w), dtype=bool)
    for a, b in zip(vertices, vertices[1:]):
        _draw_segment(outline, a, b)

    reachable = _flood_reachable(~outline)
    interior = (~reachable) & (~outline)
    filled = interior | outline
    if int(interior.sum()) < 45:
        return None
    anchorable = _outline_anchor_map(outline, interior)
    if len(anchorable) < 20:
        return None
    if len(_mask_components(outline)) != 1:
        return None
    return filled, outline


def _candidate_edges(outline: np.ndarray, interior: np.ndarray) -> list[dict]:
    candidates = []
    anchors = _outline_anchor_map(outline, interior)
    for pixel, anchor in anchors.items():
        pr, pc = pixel
        ar, ac = anchor
        outward = (pr - ar, pc - ac)
        candidates.append({"pixel": pixel, "anchor": anchor, "outward": outward})
    random.shuffle(candidates)
    return candidates


def _shape_to_cells(pixel: tuple[int, int], outward: tuple[int, int], shape: list[tuple[int, int]], tangent_flip: int) -> list[tuple[int, int]]:
    pr, pc = pixel
    orow, ocol = outward
    if orow != 0:
        trow, tcol = 0, tangent_flip
    else:
        trow, tcol = tangent_flip, 0
    cells = []
    for depth, shift in shape:
        rr = pr + depth * orow + shift * trow
        cc = pc + depth * ocol + shift * tcol
        cells.append((int(rr), int(cc)))
    return cells


def _unique_contact(seed_coords: np.ndarray, outline_coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    dist_matrix = np.abs(seed_coords[:, None, 0] - outline_coords[None, :, 0]) + np.abs(seed_coords[:, None, 1] - outline_coords[None, :, 1])
    best = int(np.min(dist_matrix))
    seed_ids, outline_ids = np.where(dist_matrix == best)
    outline_pts = sorted({tuple(map(int, outline_coords[idx])) for idx in outline_ids})
    if len(outline_pts) != 1:
        return None
    contact = outline_pts[0]
    seed_dists = np.abs(seed_coords[:, 0] - contact[0]) + np.abs(seed_coords[:, 1] - contact[1])
    nearest_seed_pts = sorted({tuple(map(int, seed_coords[idx])) for idx in np.where(seed_dists == int(np.min(seed_dists)))[0]})
    if len(nearest_seed_pts) != 1:
        return None
    return contact, nearest_seed_pts[0]


def _well_separated(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    canvas = _build_canvas(h, w)
    if canvas is None:
        return None
    filled, outline = canvas
    interior = filled & ~outline
    candidates = _candidate_edges(outline, interior)
    if len(candidates) < 12:
        return None

    bg_choices = [c for c in range(10) if c != OUTLINE_COLOR]
    bg = random.choice(bg_choices)
    seed_colors = [c for c in range(10) if c not in (bg, OUTLINE_COLOR)]

    seed_count = random.randint(2, 4)
    sizes = random.sample([2, 3, 4, 5], seed_count)
    colors = random.sample(seed_colors, seed_count)

    grid = np.full((h, w), bg, dtype=int)
    grid[outline] = OUTLINE_COLOR

    occupied: set[tuple[int, int]] = set()
    used_contacts: set[tuple[int, int]] = set()
    outline_coords = np.argwhere(outline)

    placed = []
    by_dir: dict[tuple[int, int], list[dict]] = {}
    for cand in candidates:
        by_dir.setdefault(tuple(cand["outward"]), []).append(cand)

    dirs = list(by_dir.keys())
    random.shuffle(dirs)
    dir_cycle = dirs + dirs

    for idx, (size, color) in enumerate(zip(sizes, colors)):
        size_ok = False
        preferred_dir = dir_cycle[idx] if idx < len(dir_cycle) else None
        pool = candidates[:]
        random.shuffle(pool)
        if preferred_dir is not None:
            pool.sort(key=lambda item: tuple(item["outward"]) != preferred_dir)
        for cand in pool:
            pixel = tuple(cand["pixel"])
            if pixel in used_contacts:
                continue
            for shape in random.sample(LOCAL_SHAPES_BY_SIZE[size], len(LOCAL_SHAPES_BY_SIZE[size])):
                for tangent_flip in (-1, 1):
                    cells = _shape_to_cells(pixel, tuple(cand["outward"]), shape, tangent_flip)
                    if len(set(cells)) != len(cells):
                        continue
                    if not all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in cells):
                        continue
                    if any(filled[r, c] for r, c in cells):
                        continue
                    if not _well_separated(cells, occupied):
                        continue

                    seed_coords = np.array(cells, dtype=int)
                    contact = _unique_contact(seed_coords, outline_coords)
                    if contact is None:
                        continue
                    if contact[0] != pixel:
                        continue

                    for r, c in cells:
                        grid[r, c] = color
                    occupied.update(cells)
                    used_contacts.add(pixel)
                    placed.append({"color": color, "cells": cells, "size": size})
                    size_ok = True
                    break
                if size_ok:
                    break
            if size_ok:
                break
        if not size_ok:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    for info in placed:
        if int((out == info["color"]).sum()) == 0:
            return None
    if int((out != bg).sum()) <= int(outline.sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
