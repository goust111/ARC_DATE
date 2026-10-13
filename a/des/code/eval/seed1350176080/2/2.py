from __future__ import annotations

import random

import numpy as np


RED = 2
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

SYMMETRIC_MASKS = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[int(r - r0), int(c - c0)] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _crop_to_non_bg(patch: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(patch != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _template_holes(
    grid: np.ndarray,
    bg: int,
    template_mask: np.ndarray,
    template_color: int,
) -> list[dict]:
    h, w = grid.shape
    holes: list[dict] = []
    for cells in _components(grid == int(bg)):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        cell_set = set(cells)
        boundary: set[tuple[int, int]] = set()
        valid = True
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_set:
                    continue
                if int(grid[nr, nc]) == int(bg):
                    valid = False
                    break
                if not bool(template_mask[nr, nc]):
                    valid = False
                    break
                boundary.add((int(nr), int(nc)))
            if not valid:
                break
        if not valid or not boundary:
            continue

        marker_cells = {(r, c) for r, c in boundary if int(grid[r, c]) != int(template_color)}
        if len(marker_cells) != 1:
            continue

        marker = next(iter(marker_cells))
        mask, bbox = _mask_from_cells(cells)
        holes.append(
            {
                "cells": set(cells),
                "mask": mask,
                "bbox": bbox,
                "marker": marker,
                "color": int(grid[marker]),
            }
        )
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outer fragment to the slot marker of the same color, rotate it so that its red
    anchor lands on the slot marker, then insert the fragment body into the matching hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    fg_comps = _components(g != int(bg))
    assert fg_comps, "expected a template and at least one fragment"

    comp_sizes = [len(comp) for comp in fg_comps]
    template_idx = int(np.argmax(comp_sizes))
    template_cells = fg_comps[template_idx]

    template_mask = np.zeros(g.shape, dtype=bool)
    for r, c in template_cells:
        template_mask[int(r), int(c)] = True

    template_counts: dict[int, int] = {}
    for r, c in template_cells:
        color = int(g[r, c])
        template_counts[color] = template_counts.get(color, 0) + 1
    template_color = max(template_counts, key=template_counts.get)

    holes = _template_holes(g, bg, template_mask, template_color)
    holes_by_color = {hole["color"]: hole for hole in holes}
    assert holes_by_color, "expected at least one marked hole"
    assert len(holes_by_color) == len(holes), "hole marker colors must be unique"

    pieces: dict[int, dict] = {}
    piece_cells: list[list[tuple[int, int]]] = []
    for idx, cells in enumerate(fg_comps):
        if idx == template_idx:
            continue
        piece_cells.append(cells)
        colors = {int(g[r, c]) for r, c in cells}
        assert int(RED) in colors and len(colors) == 2, "each fragment must contain one red marker and one body color"
        body_colors = [color for color in colors if color != int(RED)]
        assert len(body_colors) == 1
        body_color = int(body_colors[0])
        assert body_color not in pieces, "fragment colors must be unique"

        r0, c0, r1, c1 = _bbox(cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in cells:
            patch[int(r - r0), int(c - c0)] = int(g[r, c])
        pieces[body_color] = {"patch": patch}

    assert set(pieces) == set(holes_by_color), "slot colors and fragment colors must match one-to-one"

    out = g.copy()
    for cells in piece_cells:
        for r, c in cells:
            out[int(r), int(c)] = int(bg)

    for color, hole in holes_by_color.items():
        patch = pieces[color]["patch"]
        matches: list[set[tuple[int, int]]] = []
        for rot in range(4):
            rotated = np.rot90(patch, rot)
            body = np.argwhere(rotated == int(color))
            red = np.argwhere(rotated == int(RED))
            if body.size == 0 or red.shape[0] != 1:
                continue

            br0, bc0 = body.min(axis=0)
            br1, bc1 = body.max(axis=0)
            shape = np.zeros((int(br1 - br0 + 1), int(bc1 - bc0 + 1)), dtype=bool)
            for rr, cc in body:
                shape[int(rr - br0), int(cc - bc0)] = True
            if shape.shape != hole["mask"].shape or not bool(np.array_equal(shape, hole["mask"])):
                continue

            shift_r = int(hole["bbox"][0] - br0)
            shift_c = int(hole["bbox"][1] - bc0)
            red_abs = (int(red[0, 0]) + shift_r, int(red[0, 1]) + shift_c)
            body_abs = {(int(rr) + shift_r, int(cc) + shift_c) for rr, cc in body}
            if red_abs != hole["marker"] or body_abs != hole["cells"]:
                continue
            matches.append(body_abs)

        assert len(matches) == 1, "each fragment should have a unique valid rotation"
        for r, c in matches[0]:
            out[int(r), int(c)] = int(color)

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _has_rotation_symmetry(mask: np.ndarray) -> bool:
    for rot in (1, 2, 3):
        rotated = np.rot90(mask, rot)
        if rotated.shape == mask.shape and bool(np.array_equal(rotated, mask)):
            return True
    return False


def _grow_polyomino(area: int, height: int, width: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return _normalize_mask(mask)


def _random_shape() -> np.ndarray:
    if random.random() < 0.5:
        return np.rot90(random.choice(SYMMETRIC_MASKS), random.randrange(4)).copy()

    for _ in range(100):
        area = random.randint(4, 7)
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        if height * width < area:
            continue
        mask = _grow_polyomino(area, height, width)
        if mask is None or int(mask.sum()) != area:
            continue
        if _has_rotation_symmetry(mask):
            continue
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return np.rot90(mask, random.randrange(4)).copy()
    raise AssertionError("failed to build an asymmetric shape")


def _expanded_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                out.add((int(rr), int(cc)))
    return out


def _marker_options(mask: np.ndarray, top: int, left: int) -> dict[str, list[tuple[int, int]]]:
    h, w = mask.shape
    options = {
        "top": [(top - 1, left + c) for c in range(w) if bool(mask[0, c])],
        "bottom": [(top + h, left + c) for c in range(w) if bool(mask[h - 1, c])],
        "left": [(top + r, left - 1) for r in range(h) if bool(mask[r, 0])],
        "right": [(top + r, left + w) for r in range(h) if bool(mask[r, w - 1])],
    }
    return {side: [(int(r), int(c)) for r, c in coords] for side, coords in options.items()}


def _build_piece_patch(mask: np.ndarray, color: int, marker_abs: tuple[int, int], hole_bbox: tuple[int, int, int, int], bg: int) -> np.ndarray:
    hr0, hc0, _hr1, _hc1 = hole_bbox
    red_rel = (int(marker_abs[0] - hr0), int(marker_abs[1] - hc0))
    h, w = mask.shape
    min_r = min(0, red_rel[0])
    max_r = max(h - 1, red_rel[0])
    min_c = min(0, red_rel[1])
    max_c = max(w - 1, red_rel[1])
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                patch[int(r - min_r), int(c - min_c)] = int(color)
    patch[int(red_rel[0] - min_r), int(red_rel[1] - min_c)] = int(RED)
    return patch


def _can_place_piece(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    bg: int,
    template_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top < 1 or left < 1 or top + ph > h - 1 or left + pw > w - 1:
        return False

    tr0, tc0, tr1, tc1 = template_bbox
    if not (top + ph - 1 < tr0 - 1 or top > tr1 + 1 or left + pw - 1 < tc0 - 1 or left > tc1 + 1):
        return False

    for dr, dc in np.argwhere(patch != int(bg)):
        rr = int(top + dr)
        cc = int(left + dc)
        if int(grid[rr, cc]) != int(bg):
            return False
        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        if bool(np.any(grid[r0:r1, c0:c1] != int(bg))):
            return False
    return True


def _random_piece_position(
    grid: np.ndarray,
    patch: np.ndarray,
    bg: int,
    template_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    tr0, tc0, tr1, tc1 = template_bbox
    for _ in range(250):
        side = random.choice(SIDES)
        if side == "top":
            r_lo, r_hi = 1, tr0 - ph - 1
            c_lo, c_hi = 1, w - pw - 1
        elif side == "bottom":
            r_lo, r_hi = tr1 + 2, h - ph - 1
            c_lo, c_hi = 1, w - pw - 1
        elif side == "left":
            r_lo, r_hi = 1, h - ph - 1
            c_lo, c_hi = 1, tc0 - pw - 1
        else:
            r_lo, r_hi = 1, h - ph - 1
            c_lo, c_hi = tc1 + 2, w - pw - 1

        if r_lo > r_hi or c_lo > c_hi:
            continue
        top = random.randint(r_lo, r_hi)
        left = random.randint(c_lo, c_hi)
        if _can_place_piece(grid, patch, top, left, bg, template_bbox):
            return top, left
    return None


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    palette = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    bg = random.choice(palette)
    remaining = [color for color in palette if color != bg]
    template_color = random.choice(remaining)
    remaining = [color for color in remaining if color != template_color]

    n_slots = random.randint(2, 4)
    slot_colors = random.sample(remaining, n_slots)

    height = random.randint(22 + n_slots, 30)
    width = random.randint(22 + n_slots, 30)
    template_h = random.randint(max(10, n_slots * 2 + 4), min(16, height - 6))
    template_w = random.randint(max(10, n_slots * 3 + 2), min(18, width - 6))

    center_r = (height - template_h) // 2
    center_c = (width - template_w) // 2
    top = min(max(2, center_r + random.randint(-2, 2)), height - template_h - 2)
    left = min(max(2, center_c + random.randint(-2, 2)), width - template_w - 2)
    bottom = top + template_h - 1
    right = left + template_w - 1
    template_bbox = (top, left, bottom, right)

    grid = np.full((height, width), int(bg), dtype=int)
    grid[top : bottom + 1, left : right + 1] = int(template_color)
    out = grid.copy()

    reserved: set[tuple[int, int]] = set()
    pieces_to_place: list[np.ndarray] = []
    hole_specs: list[dict] = []

    for color in slot_colors:
        placed = False
        for _ in range(200):
            mask = _random_shape()
            mh, mw = mask.shape
            if mh > template_h - 2 or mw > template_w - 2:
                continue
            hole_top = random.randint(top + 1, bottom - mh)
            hole_left = random.randint(left + 1, right - mw)
            cells = {
                (int(hole_top + r), int(hole_left + c))
                for r in range(mh)
                for c in range(mw)
                if bool(mask[r, c])
            }
            if not cells:
                continue
            if any(cell in reserved for cell in cells):
                continue

            options = _marker_options(mask, hole_top, hole_left)
            random_sides = list(SIDES)
            random.shuffle(random_sides)
            marker = None
            for side in random_sides:
                candidates = [
                    pos
                    for pos in options[side]
                    if top <= pos[0] <= bottom
                    and left <= pos[1] <= right
                    and pos not in cells
                    and pos not in reserved
                ]
                if not candidates:
                    continue
                marker = random.choice(candidates)
                break
            if marker is None:
                continue

            footprint = _expanded_cells(set(cells) | {marker}, height, width, pad=1)
            if any(pos in reserved for pos in footprint):
                continue
            reserved |= footprint

            hole_mask, hole_bbox = _mask_from_cells(cells)
            for r, c in cells:
                grid[int(r), int(c)] = int(bg)
                out[int(r), int(c)] = int(color)
            grid[marker] = int(color)
            out[marker] = int(color)

            solved_patch = _build_piece_patch(hole_mask, color, marker, hole_bbox, bg)
            input_patch = _crop_to_non_bg(np.rot90(solved_patch, random.randrange(4)), bg)
            pieces_to_place.append(input_patch)
            hole_specs.append({"color": color, "cells": cells, "marker": marker})
            placed = True
            break
        assert placed, "failed to place all template holes"

    place_order = list(range(len(pieces_to_place)))
    random.shuffle(place_order)
    for idx in place_order:
        patch = pieces_to_place[idx]
        pos = _random_piece_position(grid, patch, bg, template_bbox)
        assert pos is not None, "failed to place a fragment outside the template"
        pr, pc = pos
        ph, pw = patch.shape
        sub = grid[pr : pr + ph, pc : pc + pw]
        sub[patch != int(bg)] = patch[patch != int(bg)]
        grid[pr : pr + ph, pc : pc + pw] = sub

    values, counts = np.unique(grid, return_counts=True)
    assert int(values[int(np.argmax(counts))]) == int(bg), "background must remain the dominant color"
    assert len(hole_specs) == n_slots
    assert bool(np.array_equal(main(grid), out))
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, _out = _build_instance()
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
