from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 800


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
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = _components(grid != BG)
    objects: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask = np.zeros_like(patch, dtype=bool)
        colors = set()
        for r, c in coords:
            mask[r - r0, c - c0] = True
            colors.add(int(grid[r, c]))
        assert len(colors) == 1, "each fragment must be monochrome"
        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "color": int(next(iter(colors))),
                "area": int(mask.sum()),
            }
        )
    return objects


def _is_palette_block(obj: dict) -> bool:
    mask = obj["mask"]
    return mask.shape == (3, 3) and bool(mask.all())


def _piece_struct(mask: np.ndarray) -> dict:
    tight = np.asarray(mask, dtype=bool)
    cells = [(int(r), int(c)) for r, c in np.argwhere(tight)]
    cell_set = set(cells)
    boundary: dict[tuple[int, int], list[tuple[int, int]]] = {d: [] for d in DIR4}
    for r, c in cells:
        for dr, dc in DIR4:
            if (r + dr, c + dc) not in cell_set:
                boundary[(dr, dc)].append((r, c))
    return {"mask": tight, "cells": cells, "cell_set": cell_set, "boundary": boundary}


def _translate_cells(cells: list[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _contact_count(a_cells: set[tuple[int, int]], b_cells: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in a_cells:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b_cells:
                total += 1
    return int(total)


def _pair_offsets(piece_a: dict, piece_b: dict) -> list[tuple[tuple[int, int], int]]:
    best: dict[tuple[int, int], int] = {}
    a_cells = piece_a["cell_set"]
    for dr, dc in DIR4:
        opp = (-dr, -dc)
        for ar, ac in piece_a["boundary"][(dr, dc)]:
            touch_r = ar + dr
            touch_c = ac + dc
            for br, bc in piece_b["boundary"][opp]:
                offset = (int(touch_r - br), int(touch_c - bc))
                if offset in best:
                    continue
                b_trans = _translate_cells(piece_b["cells"], offset)
                if a_cells & b_trans:
                    continue
                contact = _contact_count(a_cells, b_trans)
                if contact > 0:
                    best[offset] = int(contact)
    return sorted(best.items(), key=lambda item: (-item[1], abs(item[0][0]) + abs(item[0][1]), item[0]))


def _cells_to_mask(cells: set[tuple[int, int]]) -> tuple[np.ndarray, tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    h = int(max_r - min_r + 1)
    w = int(max_c - min_c + 1)
    mask = np.zeros((h, w), dtype=bool)
    norm: list[tuple[int, int]] = []
    for r, c in cells:
        rr = int(r - min_r)
        cc = int(c - min_c)
        mask[rr, cc] = True
        norm.append((rr, cc))
    norm.sort()
    return mask, tuple(norm), h, w


def _assemble_piece_masks(masks: list[np.ndarray]) -> np.ndarray:
    pieces = [_piece_struct(mask) for mask in masks]
    pieces.sort(key=lambda piece: (-len(piece["cells"]), piece["mask"].shape[0] * piece["mask"].shape[1]))
    n = len(pieces)
    assert 2 <= n <= 4, "expected 2 to 4 pieces"

    pair_options: list[list[list[tuple[tuple[int, int], int]]]] = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                pair_options[i][j] = _pair_offsets(pieces[i], pieces[j])

    placed_offsets: dict[int, tuple[int, int]] = {0: (0, 0)}
    placed_cells: dict[int, set[tuple[int, int]]] = {0: set(pieces[0]["cell_set"])}
    union_cells = set(pieces[0]["cell_set"])

    best_score: tuple[int, int, int] | None = None
    best_norm: tuple[tuple[int, int], ...] | None = None
    best_mask: np.ndarray | None = None
    memo: dict[tuple[tuple[int, int, int], ...], int] = {}

    def dfs(current_contact: int) -> None:
        nonlocal best_score, best_norm, best_mask

        state = tuple(sorted((idx, off[0], off[1]) for idx, off in placed_offsets.items()))
        if memo.get(state, -1) >= current_contact:
            return
        memo[state] = int(current_contact)

        if len(placed_offsets) == n:
            mask, norm, h, w = _cells_to_mask(union_cells)
            score = (int(-(h * w)), int(current_contact), int(-(h + w)))
            if best_score is None or score > best_score or (score == best_score and (best_norm is None or norm < best_norm)):
                best_score = score
                best_norm = norm
                best_mask = mask
            return

        remaining = [idx for idx in range(n) if idx not in placed_offsets]
        next_piece: int | None = None
        next_candidates: list[tuple[tuple[int, int], set[tuple[int, int]], int]] | None = None

        for idx in remaining:
            offsets: dict[tuple[int, int], None] = {}
            for placed_idx, placed_off in placed_offsets.items():
                for rel_off, _contact in pair_options[placed_idx][idx]:
                    offsets[(placed_off[0] + rel_off[0], placed_off[1] + rel_off[1])] = None

            feasible: list[tuple[tuple[int, int], set[tuple[int, int]], int]] = []
            for offset in offsets:
                trans = _translate_cells(pieces[idx]["cells"], offset)
                if trans & union_cells:
                    continue
                add_contact = 0
                for placed_idx in placed_offsets:
                    add_contact += _contact_count(placed_cells[placed_idx], trans)
                if add_contact > 0:
                    feasible.append((offset, trans, int(add_contact)))

            feasible.sort(key=lambda item: (-item[2], abs(item[0][0]) + abs(item[0][1]), item[0]))
            if not feasible:
                continue
            if next_candidates is None or len(feasible) < len(next_candidates):
                next_piece = idx
                next_candidates = feasible

        if next_piece is None or next_candidates is None:
            return

        for offset, trans, add_contact in next_candidates:
            placed_offsets[next_piece] = offset
            placed_cells[next_piece] = trans
            union_cells.update(trans)
            dfs(current_contact + add_contact)
            for cell in trans:
                union_cells.remove(cell)
            del placed_cells[next_piece]
            del placed_offsets[next_piece]

    dfs(0)
    assert best_mask is not None, "failed to assemble pieces"
    return best_mask


def _count_holes(mask: np.ndarray) -> int:
    bg_mask = ~np.asarray(mask, dtype=bool)
    h, w = bg_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = r == 0 or r == h - 1 or c == 0 or c == w - 1
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                            touches_edge = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return int(holes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble monochrome puzzle fragments by translation-only edge complement,
    count enclosed holes in the assembled silhouette, then recolor the silhouette
    by the H-th 3x3 palette block from the bottom strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    palette = [obj for obj in objects if _is_palette_block(obj)]
    pieces = [obj for obj in objects if not _is_palette_block(obj)]
    assert 2 <= len(pieces) <= 4, "expected 2 to 4 puzzle pieces"
    assert 2 <= len(palette) <= 4, "expected 2 to 4 palette blocks"

    palette.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    palette_colors = [int(obj["color"]) for obj in palette]
    assembled = _assemble_piece_masks([obj["mask"] for obj in pieces])

    holes = _count_holes(assembled)
    assert 1 <= holes <= len(palette_colors), "hole count must index the palette"

    out = np.full(assembled.shape, BG, dtype=int)
    out[assembled] = int(palette_colors[holes - 1])
    return out


def _make_holey_shape(hole_count: int) -> np.ndarray:
    layouts: list[tuple[int, int]] = []
    for rows in (1, 2):
        cols = (hole_count + rows - 1) // rows
        if rows * cols >= hole_count:
            layouts.append((rows, cols))
    rows, cols = random.choice(layouts)
    hole_h = random.randint(1, 2)
    hole_w = random.randint(1, 3)

    mask = np.ones((rows * hole_h + rows + 1, cols * hole_w + cols + 1), dtype=bool)
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = random.sample(slots, hole_count)
    for sr, sc in chosen:
        r0 = 1 + sr * (hole_h + 1)
        c0 = 1 + sc * (hole_w + 1)
        mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False

    for _ in range(random.randint(0, 2)):
        h, w = mask.shape
        side = random.choice(["top", "bottom", "left", "right"])
        if side in {"top", "bottom"}:
            span = random.randint(1, max(1, w // 2))
            start = random.randint(0, w - span)
            band = np.zeros((1, w), dtype=bool)
            band[0, start : start + span] = True
            mask = np.vstack([band, mask]) if side == "top" else np.vstack([mask, band])
        else:
            span = random.randint(1, max(1, h // 2))
            start = random.randint(0, h - span)
            band = np.zeros((h, 1), dtype=bool)
            band[start : start + span, 0] = True
            mask = np.hstack([band, mask]) if side == "left" else np.hstack([mask, band])
    return mask


def _neighbors(cell: tuple[int, int], cell_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIR4 if (r + dr, c + dc) in cell_set]


def _partition_shape(mask: np.ndarray, piece_count: int) -> list[np.ndarray] | None:
    cells = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    cell_set = set(cells)
    if len(cells) < piece_count * 3:
        return None

    seeds = random.sample(cells, piece_count)
    pieces = [set([seed]) for seed in seeds]
    assigned = set(seeds)
    frontiers = []
    for seed in seeds:
        frontiers.append(set(_neighbors(seed, cell_set)) - assigned)

    while len(assigned) < len(cells):
        choices: list[tuple[int, tuple[int, int]]] = []
        weights: list[float] = []
        for idx, frontier in enumerate(frontiers):
            if not frontier:
                continue
            weight = 1.0 / float(len(pieces[idx]) + 1)
            for cell in frontier:
                if cell in assigned:
                    continue
                choices.append((idx, cell))
                weights.append(weight)
        if not choices:
            return None
        pick = random.choices(range(len(choices)), weights=weights, k=1)[0]
        idx, cell = choices[pick]
        pieces[idx].add(cell)
        assigned.add(cell)
        for frontier in frontiers:
            frontier.discard(cell)
        for nb in _neighbors(cell, cell_set):
            if nb not in assigned:
                frontiers[idx].add(nb)

    masks: list[np.ndarray] = []
    for piece in pieces:
        coords = sorted(piece)
        if len(coords) < 3:
            return None
        piece_mask = _mask_from_coords(coords)
        if piece_mask.shape == (3, 3) and bool(piece_mask.all()):
            return None
        masks.append(piece_mask)

    adj_graph = {idx: set() for idx in range(piece_count)}
    owner = {}
    for idx, piece in enumerate(pieces):
        for cell in piece:
            owner[cell] = idx
    for cell, idx in owner.items():
        r, c = cell
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in owner and owner[nb] != idx:
                adj_graph[idx].add(owner[nb])
    if any(not neighbors for neighbors in adj_graph.values()):
        return None
    return masks


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_patches(
    patches: list[np.ndarray], region_h: int, region_w: int
) -> list[tuple[np.ndarray, int, int]] | None:
    blocked = np.zeros((region_h, region_w), dtype=bool)
    placements: list[tuple[np.ndarray, int, int]] = []
    for patch in sorted(patches, key=lambda arr: -int(np.count_nonzero(arr))):
        h, w = patch.shape
        options: list[tuple[int, int]] = []
        for r in range(region_h - h + 1):
            for c in range(region_w - w + 1):
                rr0 = max(0, r - 1)
                rr1 = min(region_h, r + h + 1)
                cc0 = max(0, c - 1)
                cc1 = min(region_w, c + w + 1)
                if not blocked[rr0:rr1, cc0:cc1].any():
                    options.append((int(r), int(c)))
        if not options:
            return None
        r, c = random.choice(options)
        placements.append((patch, r, c))
        blocked[max(0, r - 1) : min(region_h, r + h + 1), max(0, c - 1) : min(region_w, c + w + 1)] = True
    return placements


def _expected_output(mask: np.ndarray, palette_colors: list[int], hole_count: int) -> np.ndarray:
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = int(palette_colors[hole_count - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.randint(1, 4)
    final_mask = _make_holey_shape(hole_count)
    area = int(final_mask.sum())

    feasible_piece_counts = [count for count in range(2, 5) if area >= count * 3]
    if not feasible_piece_counts:
        return None
    piece_count = random.choice(feasible_piece_counts)

    piece_masks = _partition_shape(final_mask, piece_count)
    if piece_masks is None:
        return None

    palette_count = random.randint(max(2, hole_count), 4)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], piece_count + palette_count)
    piece_colors = colors[:piece_count]
    palette_colors = colors[piece_count:]
    piece_patches = [_mask_to_patch(mask, color) for mask, color in zip(piece_masks, piece_colors)]

    gap = random.randint(1, 3)
    palette_width = palette_count * 3 + (palette_count - 1) * gap
    top_h = max(max(patch.shape[0] for patch in piece_patches) + random.randint(3, 6), final_mask.shape[0] + 3, 8)
    bottom_margin = random.randint(0, 2)
    grid_h = top_h + 1 + 3 + bottom_margin
    grid_w = max(
        16,
        final_mask.shape[1] + random.randint(6, 10),
        sum(patch.shape[1] for patch in piece_patches) + piece_count + 2,
        palette_width + 2,
    )

    palette_row = grid_h - 3 - bottom_margin
    placements = _place_patches(piece_patches, palette_row - 1, grid_w)
    if placements is None:
        return None

    start_c = random.randint(0, grid_w - palette_width)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, r0, c0 in placements:
        h, w = patch.shape
        sel = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][sel] = patch[sel]

    col = start_c
    for color in palette_colors:
        grid[palette_row : palette_row + 3, col : col + 3] = int(color)
        col += 3 + gap

    expected = _expected_output(final_mask, palette_colors, hole_count)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
