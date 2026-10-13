from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_coords(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    return pts - pts.min(axis=0)


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = _crop_coords(coords)
    return tuple(sorted(map(tuple, pts.tolist())))


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    pts = _crop_coords(coords)
    h, w = pts.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _ordered_orientations(coords: np.ndarray) -> list[np.ndarray]:
    mask = _coords_to_mask(coords)
    variants = [
        mask,
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.rot90(np.fliplr(mask), 1),
        np.rot90(np.fliplr(mask), 2),
        np.rot90(np.fliplr(mask), 3),
    ]
    out: list[np.ndarray] = []
    seen = set()
    for variant in variants:
        pts = _crop_coords(_mask_to_coords(variant))
        key = _shape_signature(pts)
        if key not in seen:
            seen.add(key)
            out.append(pts)
    return out


def _find_components(grid: np.ndarray, *, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(coords)),
                }
            )
    return components


def _component_sizes(cells: set[tuple[int, int]]) -> list[int]:
    remaining = set(cells)
    sizes = []
    while remaining:
        start = remaining.pop()
        q = [start]
        size = 1
        while q:
            r, c = q.pop()
            for nb in _neighbors4(r, c):
                if nb in remaining:
                    remaining.remove(nb)
                    q.append(nb)
                    size += 1
        sizes.append(size)
    return sizes


def _capacities_fit(cells: set[tuple[int, int]], sizes: tuple[int, ...]) -> bool:
    capacities = tuple(sorted(_component_sizes(cells), reverse=True))
    needs = tuple(sorted(map(int, sizes), reverse=True))
    if sum(capacities) != sum(needs):
        return False
    if any(cap == 1 for cap in capacities):
        return False

    def rec(i: int, caps: tuple[int, ...]) -> bool:
        if i == len(needs):
            return all(cap == 0 for cap in caps)
        need = needs[i]
        tried = set()
        for j, cap in enumerate(caps):
            if cap < need or cap in tried:
                continue
            tried.add(cap)
            nxt = list(caps)
            nxt[j] -= need
            nxt = tuple(sorted(nxt, reverse=True))
            if rec(i + 1, nxt):
                return True
        return False

    return rec(0, capacities)


def _piece_sort_key(piece: dict) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = piece["bbox"]
    return (-int(piece["area"]), int(c0), int(r0))


def _first_placement(
    input_coords: np.ndarray,
    n: int,
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    for orient in _ordered_orientations(input_coords):
        h, w = orient.max(axis=0) + 1
        for r0 in range(n - int(h) + 1):
            for c0 in range(n - int(w) + 1):
                cells = tuple(sorted((int(r0 + r), int(c0 + c)) for r, c in orient))
                if all(cell not in occupied for cell in cells):
                    return cells
    return None


def _pack_greedily(
    ordered_pieces: list[dict],
    n: int,
    *,
    include_diagonal_barrier: bool,
) -> list[tuple[int, tuple[tuple[int, int], ...]]] | None:
    occupied = {(i, i) for i in range(n)} if include_diagonal_barrier else set()
    placed: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for idx, piece in enumerate(ordered_pieces):
        cells = _first_placement(piece["coords"], n, occupied)
        if cells is None:
            return None
        occupied.update(cells)
        placed.append((idx, cells))
    return placed


def _render_output(ordered_pieces: list[dict], n: int, placed: list[tuple[int, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    out = np.full((n, n), BG, dtype=int)
    for i in range(n):
        out[i, i] = GREY
    for idx, cells in placed:
        color = int(ordered_pieces[idx]["color"])
        for r, c in cells:
            out[r, c] = color
    return out


def _ordered_scene_pieces(grid: np.ndarray) -> list[dict]:
    scene = np.asarray(grid[:, :-1], dtype=int)
    pieces = []
    for comp in _find_components(scene, bg=BG):
        r0, c0, _r1, _c1 = comp["bbox"]
        pieces.append(
            {
                "color": int(comp["color"]),
                "area": int(comp["area"]),
                "bbox": comp["bbox"],
                "coords": comp["coords"] - np.array([r0, c0], dtype=int),
            }
        )
    return sorted(pieces, key=_piece_sort_key)


def _base_shape_catalog() -> dict[int, list[np.ndarray]]:
    raw = {
        2: (
            ((0, 0), (0, 1)),
        ),
        3: (
            ((0, 0), (0, 1), (0, 2)),
            ((0, 0), (1, 0), (1, 1)),
        ),
        4: (
            ((0, 0), (0, 1), (0, 2), (0, 3)),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            ((0, 0), (1, 0), (2, 0), (2, 1)),
            ((0, 0), (0, 1), (0, 2), (1, 1)),
            ((0, 1), (0, 2), (1, 0), (1, 1)),
        ),
        5: (
            ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
            ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
            ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
            ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
            ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
            ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ),
    }
    out: dict[int, list[np.ndarray]] = {}
    for size, shapes in raw.items():
        unique = []
        seen = set()
        for shape in shapes:
            arr = np.asarray(shape, dtype=int)
            for orient in _ordered_orientations(arr):
                sig = _shape_signature(orient)
                if sig not in seen:
                    seen.add(sig)
                    unique.append(orient)
        out[size] = unique
    return out


SHAPE_CATALOG = _base_shape_catalog()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the gray counter, reserve the main diagonal, and greedily pack the scattered fragments into the remaining square cells.

    Input structure:
    - The background is black, and the rightmost column contains a top-aligned gray counter whose length is between 4 and 6.
    - Every other foreground object is a monochromatic 4-connected fragment of size 2-5, with fragments separated from one another.
    - The total fragment area equals the non-diagonal area of the output square, so the counter is the only size instruction.

    Rule:
    - Count the gray pixels in the rightmost column to obtain N, then create an N x N black output grid.
    - Fill the main diagonal of that square with gray; those diagonal cells are forbidden to every fragment.
    - Sort fragments by decreasing area, breaking ties by the top-left corner of the input bbox from left to right and then top to bottom.
    - For each fragment in order, try its original orientation, its 90/180/270 rotations, then the horizontally mirrored versions of those orientations; for the first orientation that fits somewhere, place it at the first legal top-left scan position that avoids occupied cells and the diagonal.

    Key insight:
    - The gray counter does not survive as an object; it simultaneously sets the square size and the exact forbidden diagonal, so the packing step is constrained by information extracted earlier.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 2, "expected scene columns plus one counter column"

    counter = g[:, -1]
    n = int(np.count_nonzero(counter == GREY))
    assert 4 <= n <= 6, "counter must encode an output side length in [4, 6]"
    assert np.all(counter[:n] == GREY), "counter must be top-aligned"
    assert np.all(counter[n:] == BG), "only the top segment of the counter column may be gray"
    assert not np.any(g[:, :-1] == GREY), "gray is reserved for the counter and output diagonal"

    ordered_pieces = _ordered_scene_pieces(g)
    assert ordered_pieces, "expected at least one fragment"
    total_area = sum(int(piece["area"]) for piece in ordered_pieces)
    assert total_area == n * n - n, "fragment area must match the square minus its diagonal"

    placed = _pack_greedily(ordered_pieces, n, include_diagonal_barrier=True)
    assert placed is not None, "every fragment must have a legal greedy placement"

    out = _render_output(ordered_pieces, n, placed)
    assert np.count_nonzero(out != BG) == n * n, "output should contain diagonal cells plus a full packing"
    return out


def _sample_sizes(total_area: int) -> list[int] | None:
    min_pieces = max(3, (total_area + 4) // 5)
    max_pieces = min(len(PALETTE), total_area // 2)
    want_tie = random.random() < 0.7

    for _ in range(200):
        piece_count = random.randint(min_pieces, max_pieces)
        counts = Counter()
        sizes = []
        remaining = total_area
        for idx in range(piece_count):
            slots_left = piece_count - idx - 1
            options = []
            for size in range(2, 6):
                if counts[size] >= 4:
                    continue
                rem = remaining - size
                if 2 * slots_left <= rem <= 5 * slots_left:
                    options.append(size)
            if not options:
                break

            tied_options = [size for size in options if counts[size] > 0]
            large_options = [size for size in options if size >= 3]
            if want_tie and tied_options and random.random() < 0.55:
                size = random.choice(tied_options)
            elif large_options and random.random() < 0.7:
                size = random.choice(large_options)
            else:
                size = random.choice(options)

            sizes.append(size)
            counts[size] += 1
            remaining -= size
        if remaining != 0:
            continue
        if max(sizes) < 3:
            continue
        if want_tie and not any(count >= 2 for count in counts.values()):
            continue
        sizes.sort(reverse=True)
        return sizes
    return None


def _free_cells(n: int, occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied}


def _build_piece_sequence(n: int, sizes: list[int]) -> list[dict] | None:
    occupied0 = {(i, i) for i in range(n)}
    require_reorientation = random.random() < 0.8
    memo = set()

    def rec(
        idx: int,
        occupied: set[tuple[int, int]],
        did_reorient: bool,
    ) -> list[dict] | None:
        key = (idx, tuple(sorted(occupied)), did_reorient)
        if key in memo:
            return None

        remaining_sizes = tuple(sizes[idx:])
        free = _free_cells(n, occupied)
        if not _capacities_fit(free, remaining_sizes):
            memo.add(key)
            return None

        if idx == len(sizes):
            if len(occupied) == n * n and (did_reorient or not require_reorientation):
                return []
            memo.add(key)
            return None

        size = sizes[idx]
        candidates = SHAPE_CATALOG[size][:]
        random.shuffle(candidates)
        if require_reorientation and not did_reorient:
            candidates.sort(key=lambda arr: len(_ordered_orientations(arr)) > 1, reverse=True)

        for shape in candidates:
            cells = _first_placement(shape, n, occupied)
            if cells is None:
                continue
            new_occupied = set(occupied)
            new_occupied.update(cells)
            placed_coords = np.asarray(cells, dtype=int)
            reoriented = _shape_signature(shape) != _shape_signature(placed_coords)
            tail = rec(idx + 1, new_occupied, did_reorient or reoriented)
            if tail is not None:
                return [
                    {
                        "area": size,
                        "coords": shape.copy(),
                        "output_coords": placed_coords,
                        "reoriented": did_reorient or reoriented,
                    }
                ] + tail

        memo.add(key)
        return None

    return rec(0, occupied0, False)


def _assign_colors(pieces: list[dict]) -> None:
    colors = random.sample(PALETTE, len(pieces))
    for piece, color in zip(pieces, colors):
        piece["color"] = int(color)


def _layout_input(pieces: list[dict], n: int) -> np.ndarray | None:
    groups: dict[int, list[dict]] = {}
    for piece in pieces:
        groups.setdefault(int(piece["area"]), []).append(piece)

    group_items = sorted(groups.items(), reverse=True)
    band_heights = []
    band_widths = []
    for _area, group in group_items:
        heights = []
        widths = []
        for piece in group:
            mask = _coords_to_mask(piece["coords"])
            heights.append(int(mask.shape[0]))
            widths.append(int(mask.shape[1]))
            piece["input_patch"] = mask.astype(int) * int(piece["color"])
        band_heights.append(max(heights))
        band_widths.append(sum(widths) + 2 * (len(widths) - 1))

    scene_h = max(12, 2 + sum(band_heights) + 2 * (len(band_heights) - 1) + random.randint(0, 2))
    scene_w = max(13, 3 + max(band_widths) + random.randint(0, 2))
    if scene_h > 29 or scene_w > 29:
        return None

    grid = np.full((scene_h, scene_w + 1), BG, dtype=int)
    top = 1 + random.randint(0, max(0, scene_h - (sum(band_heights) + 2 * (len(band_heights) - 1) + 2)))

    current_r = top
    for (area, group), band_h, band_w in zip(group_items, band_heights, band_widths):
        slack = scene_w - band_w - 2
        if slack < 0:
            return None
        current_c = 1 + random.randint(0, slack)
        for piece in group:
            patch = piece["input_patch"]
            h, w = patch.shape
            r0 = current_r + random.randint(0, band_h - h)
            c0 = current_c
            grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]
            piece["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
            current_c += w + 2
        current_r += band_h + 2

    grid[:n, -1] = GREY
    return grid


def _build_expected_output(pieces: list[dict], n: int) -> np.ndarray:
    out = np.full((n, n), BG, dtype=int)
    for i in range(n):
        out[i, i] = GREY
    for piece in pieces:
        rr, cc = piece["output_coords"].T
        out[rr, cc] = int(piece["color"])
    return out


def _diagonal_barrier_matters(ordered_pieces: list[dict], expected: np.ndarray, n: int) -> bool:
    no_barrier = _pack_greedily(ordered_pieces, n, include_diagonal_barrier=False)
    if no_barrier is None:
        return True
    occupied = set()
    for _idx, cells in no_barrier:
        occupied.update(cells)
    if any((i, i) in occupied for i in range(n)):
        return True

    alt = np.full((n, n), BG, dtype=int)
    for idx, cells in no_barrier:
        color = int(ordered_pieces[idx]["color"])
        for r, c in cells:
            alt[r, c] = color
    alt[np.arange(n), np.arange(n)] = GREY
    return not np.array_equal(alt, expected)


def _validate_instance(inp: np.ndarray, expected: np.ndarray, n: int) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if int(np.count_nonzero(inp[:, -1] == GREY)) != n:
        return False
    if not np.all(inp[:n, -1] == GREY) or not np.all(inp[n:, -1] == BG):
        return False
    if np.any(inp[:, :-1] == GREY):
        return False

    ordered_pieces = _ordered_scene_pieces(inp)
    if len(ordered_pieces) == 0 or len(ordered_pieces) > len(PALETTE):
        return False
    if any(piece["area"] < 2 or piece["area"] > 5 for piece in ordered_pieces):
        return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False
    if np.count_nonzero(out == GREY) != n:
        return False
    if np.any(np.diag(out) != GREY):
        return False

    if len({int(piece["color"]) for piece in ordered_pieces}) != len(ordered_pieces):
        return False
    if sum(int(piece["area"]) for piece in ordered_pieces) != n * n - n:
        return False
    if not _diagonal_barrier_matters(ordered_pieces, expected, n):
        return False

    orientation_changed = False
    for piece in ordered_pieces:
        one_piece = _pack_greedily([piece], n, include_diagonal_barrier=True)
        if one_piece is None:
            return False
        placed_coords = np.asarray(one_piece[0][1], dtype=int)
        if _shape_signature(piece["coords"]) != _shape_signature(placed_coords):
            orientation_changed = True
            break
    if not orientation_changed:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        n = random.randint(4, 6)
        sizes = _sample_sizes(n * n - n)
        if sizes is None:
            continue

        pieces = _build_piece_sequence(n, sizes)
        if pieces is None:
            continue

        _assign_colors(pieces)
        inp = _layout_input(pieces, n)
        if inp is None:
            continue

        expected = _build_expected_output(pieces, n)
        if _validate_instance(inp, expected, n):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
