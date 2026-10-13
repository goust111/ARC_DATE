from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
FRAGMENT_COLOR = 8
TOP_COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 500
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(sorted(coords))
    components.sort(key=lambda comp: _bbox(comp))
    return components


def _piece_from_coords(coords: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return {
        "coords": tuple(sorted(coords)),
        "cells": rel,
        "shape": (r1 - r0 + 1, c1 - c0 + 1),
        "area": len(rel),
        "bbox": (r0, c0, r1, c1),
    }


def _dominant_color(
    base: np.ndarray, cells: tuple[tuple[int, int], ...], top: int, left: int, strict: bool
) -> int | None:
    colors = [int(base[top + r, left + c]) for r, c in cells]
    counts = Counter(colors)
    best = max(counts.values())
    winners = {color for color, count in counts.items() if count == best}
    if strict and len(winners) != 1:
        return None
    if len(winners) == 1:
        return next(iter(winners))
    for r, c in sorted(cells):
        color = int(base[top + r, left + c])
        if color in winners:
            return color
    assert False, "tie winner must exist"


def _solve_tiling(
    pieces: list[dict[str, object]], target_shape: tuple[int, int], limit: int = 2
) -> list[list[tuple[int, int]]]:
    h, w = target_shape
    n = len(pieces)
    total_cells = h * w
    assert sum(int(piece["area"]) for piece in pieces) == total_cells
    all_mask = (1 << total_cells) - 1

    placements_by_piece: list[list[tuple[int, int, int]]] = []
    cover_lookup: list[list[tuple[int, int]]] = [[] for _ in range(total_cells)]
    piece_areas = [int(piece["area"]) for piece in pieces]

    for piece_index, piece in enumerate(pieces):
        cells = piece["cells"]
        ph, pw = piece["shape"]
        piece_placements: list[tuple[int, int, int]] = []
        for top in range(h - ph + 1):
            for left in range(w - pw + 1):
                mask = 0
                covered: list[int] = []
                for r, c in cells:
                    idx = (top + r) * w + (left + c)
                    mask |= 1 << idx
                    covered.append(idx)
                placement_index = len(piece_placements)
                piece_placements.append((mask, top, left))
                for idx in covered:
                    cover_lookup[idx].append((piece_index, placement_index))
        placements_by_piece.append(piece_placements)

    solutions: list[list[tuple[int, int]]] = []

    def choose_cell(occupied_mask: int, used_mask: int) -> tuple[int | None, list[tuple[int, int]]]:
        remaining = all_mask ^ occupied_mask
        best_idx = None
        best_options = None
        while remaining:
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            options: list[tuple[int, int]] = []
            for piece_index, placement_index in cover_lookup[idx]:
                if used_mask & (1 << piece_index):
                    continue
                placement_mask, _top, _left = placements_by_piece[piece_index][placement_index]
                if occupied_mask & placement_mask:
                    continue
                options.append((piece_index, placement_index))
            if not options:
                return idx, []
            if best_options is None or len(options) < len(best_options):
                best_idx = idx
                best_options = options
                if len(best_options) == 1:
                    break
            remaining ^= lsb
        return best_idx, best_options or []

    def dfs(occupied_mask: int, used_mask: int, chosen: list[tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        if occupied_mask == all_mask:
            if used_mask == (1 << n) - 1:
                offsets: list[tuple[int, int] | None] = [None] * n
                for piece_index, placement_index in chosen:
                    _mask, top, left = placements_by_piece[piece_index][placement_index]
                    offsets[piece_index] = (top, left)
                solutions.append([offset for offset in offsets if offset is not None])
            return

        _idx, options = choose_cell(occupied_mask, used_mask)
        if not options:
            return
        options.sort(key=lambda entry: (-piece_areas[entry[0]], entry[0], entry[1]))
        for piece_index, placement_index in options:
            placement_mask, _top, _left = placements_by_piece[piece_index][placement_index]
            dfs(
                occupied_mask | placement_mask,
                used_mask | (1 << piece_index),
                chosen + [(piece_index, placement_index)],
            )
            if len(solutions) >= limit:
                return

    dfs(0, 0, [])
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the lower azure fragments into the top rectangle in thought, then
    recolor each fragment in place by the top block color it covers most.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_rows = np.where(np.all(g == SEPARATOR, axis=1))[0]
    assert len(separator_rows) == 1, "expected exactly one grey separator row"
    split = int(separator_rows[0])
    assert 0 < split < g.shape[0] - 1, "separator must divide top and bottom regions"

    top = g[:split]
    bottom = g[split + 1 :]

    top_coords = np.argwhere(top != BACKGROUND)
    assert len(top_coords) > 0, "top region must contain the base rectangle"
    r0, c0, r1, c1 = _bbox(top_coords)
    base = top[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(base != BACKGROUND), "base image must be a dense rectangle"
    assert not np.any(base == SEPARATOR), "separator color cannot appear in the base image"

    fragment_components = _find_components(bottom == FRAGMENT_COLOR)
    assert fragment_components, "bottom region must contain azure fragments"
    pieces = [_piece_from_coords(component) for component in fragment_components]

    solutions = _solve_tiling(pieces, base.shape, limit=2)
    assert len(solutions) == 1, "fragments must tile the target rectangle uniquely"
    placements = solutions[0]

    out = np.full(bottom.shape, BACKGROUND, dtype=int)
    for piece, (top_off, left_off) in zip(pieces, placements):
        color = _dominant_color(base, piece["cells"], top_off, left_off, strict=False)
        assert color is not None
        for r, c in piece["coords"]:
            out[r, c] = color
    return out


def _random_composition(total: int, parts: int, minimum: int) -> list[int]:
    values = [minimum] * parts
    remaining = total - minimum * parts
    for _ in range(remaining):
        weights = [1.0 / (value + 1) for value in values]
        index = random.choices(range(parts), weights=weights, k=1)[0]
        values[index] += 1
    random.shuffle(values)
    return values


def _grow_partition(h: int, w: int, targets: list[int]) -> np.ndarray | None:
    piece_count = len(targets)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)

    seeds: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - sr) + abs(cell[1] - sc) >= 2 for sr, sc in seeds):
            seeds.append(cell)
        if len(seeds) == piece_count:
            break
    if len(seeds) < piece_count:
        seeds = random.sample(cells, piece_count)

    owner = np.full((h, w), -1, dtype=int)
    regions: list[set[tuple[int, int]]] = []
    sizes: list[int] = []
    for index, (r, c) in enumerate(seeds):
        owner[r, c] = index
        regions.append({(r, c)})
        sizes.append(1)

    while any(size < target for size, target in zip(sizes, targets)):
        progress = False
        order = list(range(piece_count))
        random.shuffle(order)
        order.sort(key=lambda idx: targets[idx] - sizes[idx], reverse=True)
        for index in order:
            if sizes[index] >= targets[index]:
                continue
            frontier = set()
            for r, c in regions[index]:
                for nr, nc in _neighbors4(r, c, h, w):
                    if owner[nr, nc] == -1:
                        frontier.add((nr, nc))
            if not frontier:
                continue
            cell = random.choice(tuple(frontier))
            owner[cell] = index
            regions[index].add(cell)
            sizes[index] += 1
            progress = True
        if not progress:
            return None

    assert np.all(owner >= 0)
    return owner


def _build_piece_partition(h: int, w: int) -> list[dict[str, object]] | None:
    area = h * w
    max_pieces = min(6, area // 3)
    if max_pieces < 3:
        return None
    for _ in range(200):
        piece_count = random.randint(3, max_pieces)
        targets = _random_composition(area, piece_count, 3)
        if max(targets) > area - 3 * (piece_count - 1):
            continue
        owner = _grow_partition(h, w, targets)
        if owner is None:
            continue

        pieces: list[dict[str, object]] = []
        for index in range(piece_count):
            coords = [tuple(map(int, cell)) for cell in np.argwhere(owner == index)]
            piece = _piece_from_coords(coords)
            piece["target_offset"] = piece["bbox"][:2]
            pieces.append(piece)

        if len({piece["cells"] for piece in pieces}) != piece_count:
            continue
        irregular = sum(int(piece["area"]) != int(piece["shape"][0] * piece["shape"][1]) for piece in pieces)
        if irregular < max(1, piece_count // 2):
            continue

        solutions = _solve_tiling(pieces, (h, w), limit=2)
        if len(solutions) != 1:
            continue
        return pieces
    return None


def _split_rectangles(h: int, w: int, block_count: int) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, h, w)]
    while len(rects) < block_count:
        candidates = [
            idx
            for idx, (r0, c0, r1, c1) in enumerate(rects)
            if (r1 - r0 >= 2) or (c1 - c0 >= 2)
        ]
        if not candidates:
            return None
        idx = random.choice(candidates)
        r0, c0, r1, c1 = rects.pop(idx)
        height = r1 - r0
        width = c1 - c0
        axes = []
        if height >= 2:
            axes.append("h")
        if width >= 2:
            axes.append("v")
        axis = random.choice(axes)
        if axis == "h":
            cut = random.randint(1, height - 1)
            rects.extend([(r0, c0, r0 + cut, c1), (r0 + cut, c0, r1, c1)])
        else:
            cut = random.randint(1, width - 1)
            rects.extend([(r0, c0, r1, c0 + cut), (r0, c0 + cut, r1, c1)])
    return rects


def _sample_base(
    rect_shape: tuple[int, int], pieces: list[dict[str, object]]
) -> tuple[np.ndarray, list[int]] | tuple[None, None]:
    h, w = rect_shape
    for _ in range(120):
        block_count = random.randint(3, min(len(TOP_COLORS), 6))
        rects = _split_rectangles(h, w, block_count)
        if rects is None:
            continue
        colors = random.sample(TOP_COLORS, block_count)
        base = np.full((h, w), BACKGROUND, dtype=int)
        for color, (r0, c0, r1, c1) in zip(colors, rects):
            base[r0:r1, c0:c1] = color

        dominant: list[int] = []
        ok = True
        for piece in pieces:
            top_off, left_off = piece["target_offset"]
            color = _dominant_color(base, piece["cells"], top_off, left_off, strict=True)
            if color is None:
                ok = False
                break
            dominant.append(color)
        if ok and len(set(dominant)) >= 2:
            return base, dominant
    return None, None


def _can_place_scattered(
    occupied: np.ndarray, cells: tuple[tuple[int, int], ...], top: int, left: int, shape: tuple[int, int]
) -> bool:
    h, w = shape
    for r, c in cells:
        rr = top + r
        cc = left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _scatter_pieces(
    pieces: list[dict[str, object]], bottom_shape: tuple[int, int]
) -> list[dict[str, object]] | None:
    bottom_h, bottom_w = bottom_shape
    occupied = np.zeros((bottom_h, bottom_w), dtype=bool)
    order = sorted(range(len(pieces)), key=lambda idx: (-int(pieces[idx]["area"]), idx))
    scattered: list[dict[str, object] | None] = [None] * len(pieces)

    for index in order:
        piece = pieces[index]
        ph, pw = piece["shape"]
        positions = []
        for top in range(bottom_h - ph + 1):
            for left in range(bottom_w - pw + 1):
                if _can_place_scattered(occupied, piece["cells"], top, left, bottom_shape):
                    positions.append((top, left))
        if not positions:
            return None
        top, left = random.choice(positions)
        coords = tuple(sorted((top + r, left + c) for r, c in piece["cells"]))
        for r, c in coords:
            occupied[r, c] = True
        scattered_piece = {
            "coords": coords,
            "cells": piece["cells"],
            "shape": piece["shape"],
            "area": piece["area"],
            "bbox": (top, left, top + ph - 1, left + pw - 1),
        }
        scattered[index] = scattered_piece

    assert all(piece is not None for piece in scattered)
    return [piece for piece in scattered if piece is not None]


def _build_expected(bottom_shape: tuple[int, int], pieces: list[dict[str, object]], colors: list[int]) -> np.ndarray:
    out = np.full(bottom_shape, BACKGROUND, dtype=int)
    for piece, color in zip(pieces, colors):
        for r, c in piece["coords"]:
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rect_h = random.randint(4, 6)
    rect_w = random.randint(5, 7)
    pieces = _build_piece_partition(rect_h, rect_w)
    if pieces is None:
        return None

    base, piece_colors = _sample_base((rect_h, rect_w), pieces)
    if base is None or piece_colors is None:
        return None

    padded_area = sum((int(piece["shape"][0]) + 2) * (int(piece["shape"][1]) + 2) for piece in pieces)
    side = int(np.ceil(np.sqrt(padded_area)))
    bottom_h_min = max(rect_h + 4, side - 1, max(int(piece["shape"][0]) for piece in pieces) + 3)
    bottom_w_min = max(rect_w + 6, side + 2, max(int(piece["shape"][1]) for piece in pieces) + 4)
    bottom_h = random.randint(bottom_h_min, bottom_h_min + 4)
    grid_w = random.randint(bottom_w_min, bottom_w_min + 6)

    scattered = _scatter_pieces(pieces, (bottom_h, grid_w))
    if scattered is None:
        return None

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    top_h = top_pad + rect_h + bottom_pad
    top_left = random.randint(1, grid_w - rect_w - 1)

    grid = np.full((top_h + 1 + bottom_h, grid_w), BACKGROUND, dtype=int)
    grid[top_pad : top_pad + rect_h, top_left : top_left + rect_w] = base
    grid[top_h, :] = SEPARATOR
    for piece in scattered:
        for r, c in piece["coords"]:
            grid[top_h + 1 + r, c] = FRAGMENT_COLOR

    expected = _build_expected((bottom_h, grid_w), scattered, piece_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
