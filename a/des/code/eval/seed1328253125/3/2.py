from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
import random

import numpy as np


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_NAMES = ["top", "right", "bottom", "left"]
MIN_PIECES = 2
MAX_PIECES = 4
MIN_PIECE_AREA = 5
GEN_RETRIES = 400
PLACE_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _extract_patch(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
        mask[r - r0, c - c0] = True
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "mask": mask,
    }


def _boundary_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _detect_frame(grid: np.ndarray, bg: int, comps: list[list[tuple[int, int]]]) -> dict:
    best = None
    for idx, pts in enumerate(comps):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        boundary = _boundary_coords(r0, c0, r1, c1)
        if len(boundary) != len(pts) or set(pts) != boundary:
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior != int(bg)):
            continue

        values = [int(grid[r, c]) for r, c in pts]
        counts = Counter(values)
        frame_color = int(counts.most_common(1)[0][0])
        anchors = [(r, c, int(grid[r, c])) for r, c in pts if int(grid[r, c]) != frame_color]
        if len(anchors) != 4 or len({col for _, _, col in anchors}) != 4:
            continue

        side_by_color: dict[int, int] = {}
        seen_sides: set[int] = set()
        ok = True
        for r, c, col in anchors:
            side = None
            if r == r0 and c0 < c < c1:
                side = 0
            elif c == c1 and r0 < r < r1:
                side = 1
            elif r == r1 and c0 < c < c1:
                side = 2
            elif c == c0 and r0 < r < r1:
                side = 3
            if side is None or side in seen_sides:
                ok = False
                break
            seen_sides.add(side)
            side_by_color[int(col)] = int(side)
        if not ok or len(side_by_color) != 4:
            continue

        cand = {
            "component_index": int(idx),
            "coords": pts,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "frame_color": int(frame_color),
            "side_by_color": side_by_color,
        }
        if best is None or len(pts) > len(best["coords"]):
            best = cand

    if best is None:
        raise AssertionError("frame not found")
    return best


def _anchor_side(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    exposed = []
    for side, (dr, dc) in enumerate(DIRS4):
        nr, nc = int(r + dr), int(c + dc)
        if nr < 0 or nc < 0 or nr >= h or nc >= w or not bool(mask[nr, nc]):
            exposed.append(int(side))
    if len(exposed) != 1:
        raise AssertionError("anchor side is ambiguous")
    return int(exposed[0])


def _orient_piece(
    patch: np.ndarray,
    mask: np.ndarray,
    anchors: list[tuple[int, int, int]],
    side_by_color: dict[int, int],
) -> dict:
    needed = []
    for r, c, col in anchors:
        cur_side = _anchor_side(mask, int(r), int(c))
        target_side = int(side_by_color[int(col)])
        needed.append((int(col), int(cur_side), int(target_side)))

    rot = (needed[0][1] - needed[0][2]) % 4
    for _col, cur_side, target_side in needed[1:]:
        if (cur_side - rot) % 4 != target_side:
            raise AssertionError("anchor colors do not determine a unique rotation")

    patch_rot = np.rot90(patch, int(rot))
    mask_rot = np.rot90(mask, int(rot))
    cells = [(int(r), int(c), int(patch_rot[r, c])) for r, c in np.argwhere(mask_rot)]
    return {
        "patch": patch_rot,
        "mask": mask_rot,
        "cells": cells,
        "area": int(mask_rot.sum()),
    }


def _solve_tiling(interior_shape: tuple[int, int], pieces: list[dict]) -> tuple[list[tuple[int, int]], list[list[dict]]]:
    ih, iw = map(int, interior_shape)
    full_area = int(ih * iw)
    if sum(int(piece["area"]) for piece in pieces) != full_area:
        raise AssertionError("pieces do not cover the frame interior")

    placements_by_piece: list[list[dict]] = []
    cell_to_options: list[list[tuple[int, int]]] = [[] for _ in range(full_area)]

    for piece_idx, piece in enumerate(pieces):
        mask = np.asarray(piece["mask"], dtype=bool)
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        ph, pw = mask.shape
        placements: list[dict] = []
        for dr in range(ih - ph + 1):
            for dc in range(iw - pw + 1):
                bits = 0
                for rr, cc in coords:
                    bits |= 1 << int((dr + rr) * iw + (dc + cc))
                placement_idx = len(placements)
                placements.append({"bits": int(bits), "dr": int(dr), "dc": int(dc)})
                temp = bits
                while temp:
                    lsb = temp & -temp
                    cell_idx = int(lsb.bit_length() - 1)
                    cell_to_options[cell_idx].append((int(piece_idx), int(placement_idx)))
                    temp ^= lsb
        if not placements:
            raise AssertionError("piece has no legal placements")
        placements_by_piece.append(placements)

    full_mask = (1 << full_area) - 1

    @lru_cache(maxsize=None)
    def search(remaining_mask: int, occupied_bits: int):
        if remaining_mask == 0:
            return () if occupied_bits == full_mask else None

        uncovered = full_mask ^ occupied_bits
        best_options = None
        temp = uncovered
        while temp:
            lsb = temp & -temp
            cell_idx = int(lsb.bit_length() - 1)
            options = []
            for piece_idx, placement_idx in cell_to_options[cell_idx]:
                if not ((remaining_mask >> piece_idx) & 1):
                    continue
                bits = int(placements_by_piece[piece_idx][placement_idx]["bits"])
                if bits & occupied_bits:
                    continue
                options.append((int(piece_idx), int(placement_idx)))
            if not options:
                return None
            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break
            temp ^= lsb

        assert best_options is not None
        best_options.sort(key=lambda item: -pieces[item[0]]["area"])
        for piece_idx, placement_idx in best_options:
            bits = int(placements_by_piece[piece_idx][placement_idx]["bits"])
            result = search(int(remaining_mask ^ (1 << piece_idx)), int(occupied_bits | bits))
            if result is not None:
                return ((int(piece_idx), int(placement_idx)),) + result
        return None

    placement_choice = search((1 << len(pieces)) - 1, 0)
    if placement_choice is None:
        raise AssertionError("tiling solution not found")
    return list(placement_choice), placements_by_piece


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered pieces into the anchored frame.

    Each piece carries two colored anchor pixels. The anchor colors identify which
    frame sides those marked edge pixels must face after rotation. Once every piece
    is oriented, the rotated shapes translate into the frame interior and tile it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _components(g != int(bg))

    frame = _detect_frame(g, int(bg), comps)
    fr0, fc0, fr1, fc1 = frame["bbox"]
    side_by_color = frame["side_by_color"]

    pieces = []
    for comp_idx, pts in enumerate(comps):
        if comp_idx == int(frame["component_index"]):
            continue
        extracted = _extract_patch(g, pts, int(bg))
        patch = np.asarray(extracted["patch"], dtype=int)
        mask = np.asarray(extracted["mask"], dtype=bool)

        values = [int(g[r, c]) for r, c in pts]
        counts = Counter(values)
        base_color = int(counts.most_common(1)[0][0])
        r0, c0, _r1, _c1 = extracted["bbox"]
        anchors = [(int(r - r0), int(c - c0), int(g[r, c])) for r, c in pts if int(g[r, c]) != base_color]
        if len(anchors) != 2 or len({col for _, _, col in anchors}) != 2:
            raise AssertionError("each piece must have exactly two anchor colors")
        for _r, _c, col in anchors:
            if int(col) not in side_by_color:
                raise AssertionError("piece anchor does not match any frame side")

        pieces.append(_orient_piece(patch, mask, anchors, side_by_color))

    solution, placements_by_piece = _solve_tiling((fr1 - fr0 - 1, fc1 - fc0 - 1), pieces)

    out = np.full_like(g, int(bg))
    for r, c in frame["coords"]:
        out[r, c] = int(g[r, c])

    for piece_idx, placement_idx in solution:
        placement = placements_by_piece[piece_idx][placement_idx]
        dr = int(placement["dr"])
        dc = int(placement["dc"])
        for rr, cc, val in pieces[piece_idx]["cells"]:
            out[fr0 + 1 + dr + rr, fc0 + 1 + dc + cc] = int(val)

    return out


def _random_palette(piece_count: int) -> dict:
    colors = list(range(10))
    random.shuffle(colors)
    return {
        "bg": int(colors[0]),
        "frame": int(colors[1]),
        "side_colors": {
            0: int(colors[2]),
            1: int(colors[3]),
            2: int(colors[4]),
            3: int(colors[5]),
        },
        "piece_colors": [int(c) for c in colors[6 : 6 + piece_count]],
    }


def _random_partition(h: int, w: int, piece_count: int) -> list[set[tuple[int, int]]] | None:
    total = int(h * w)
    if total < piece_count * MIN_PIECE_AREA:
        return None

    targets = [MIN_PIECE_AREA] * piece_count
    for _ in range(total - piece_count * MIN_PIECE_AREA):
        targets[random.randrange(piece_count)] += 1

    seeds = random.sample([(r, c) for r in range(h) for c in range(w)], piece_count)
    owner = -np.ones((h, w), dtype=int)
    frontiers = [set() for _ in range(piece_count)]
    sizes = [0] * piece_count

    for idx, (r, c) in enumerate(seeds):
        owner[r, c] = int(idx)
        sizes[idx] = 1
    for idx, (r, c) in enumerate(seeds):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and int(owner[nr, nc]) == -1:
                frontiers[idx].add((int(nr), int(nc)))

    cells_left = total - piece_count
    while cells_left > 0:
        candidates = [idx for idx in range(piece_count) if frontiers[idx]]
        if not candidates:
            return None
        growable = [idx for idx in candidates if sizes[idx] < targets[idx]]
        pool = growable if growable else candidates
        weights = [max(1, targets[idx] - sizes[idx]) for idx in pool]
        idx = random.choices(pool, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[idx]))
        frontiers[idx].discard(cell)
        r, c = cell
        if int(owner[r, c]) != -1:
            continue
        owner[r, c] = int(idx)
        sizes[idx] += 1
        cells_left -= 1
        for other in range(piece_count):
            frontiers[other].discard(cell)
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and int(owner[nr, nc]) == -1:
                frontiers[idx].add((int(nr), int(nc)))

    regions: list[set[tuple[int, int]]] = []
    for idx in range(piece_count):
        cells = {(int(r), int(c)) for r, c in np.argwhere(owner == int(idx))}
        if len(cells) < MIN_PIECE_AREA:
            return None
        regions.append(cells)
    return regions


def _anchor_candidates(cells: set[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for r, c in cells:
        exposed = []
        for side, (dr, dc) in enumerate(DIRS4):
            if (int(r + dr), int(c + dc)) not in cells:
                exposed.append(int(side))
        if len(exposed) == 1:
            groups.setdefault(int(exposed[0]), []).append((int(r), int(c)))
    return groups


def _build_pieces(interior_h: int, interior_w: int, piece_count: int):
    palette = _random_palette(piece_count)
    regions = _random_partition(interior_h, interior_w, piece_count)
    if regions is None:
        return None

    interior = np.full((interior_h, interior_w), int(palette["bg"]), dtype=int)
    solved_pieces = []

    for idx, cells in enumerate(regions):
        candidates = _anchor_candidates(cells)
        available_sides = [side for side, pts in candidates.items() if pts]
        if len(available_sides) < 2:
            return None

        side_a, side_b = random.choice(list(combinations(available_sides, 2)))
        pos_a = random.choice(candidates[int(side_a)])
        pos_b = random.choice(candidates[int(side_b)])
        if pos_a == pos_b:
            return None

        color_map = {cell: int(palette["piece_colors"][idx]) for cell in cells}
        color_map[pos_a] = int(palette["side_colors"][int(side_a)])
        color_map[pos_b] = int(palette["side_colors"][int(side_b)])

        for r, c in cells:
            interior[r, c] = int(color_map[(r, c)])

        coords = list(cells)
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(palette["bg"]), dtype=int)
        mask = np.zeros_like(patch, dtype=bool)
        for r, c in cells:
            patch[r - r0, c - c0] = int(color_map[(r, c)])
            mask[r - r0, c - c0] = True

        solved_pieces.append({"patch": patch, "mask": mask, "area": int(mask.sum())})

    return {
        "palette": palette,
        "interior": interior,
        "solved_pieces": solved_pieces,
    }


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, bg: int, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for dr, dc in np.argwhere(patch != int(bg)):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - pad)
        r2 = min(H, rr + pad + 1)
        c1 = max(0, cc - pad)
        c2 = min(W, cc + pad + 1)
        if bool(np.any(occupied[r1:r2, c1:c2])):
            return False
    return True


def _stamp_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, bg: int, r0: int, c0: int) -> None:
    for dr, dc in np.argwhere(patch != int(bg)):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        grid[rr, cc] = int(patch[dr, dc])
        occupied[rr, cc] = True


def _scatter_pieces(
    canvas: np.ndarray,
    occupied: np.ndarray,
    pieces: list[np.ndarray],
    bg: int,
) -> bool:
    order = list(range(len(pieces)))
    order.sort(key=lambda idx: -int(np.count_nonzero(pieces[idx] != int(bg))))
    for idx in order:
        patch = pieces[idx]
        h, w = patch.shape
        H, W = canvas.shape
        positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions[:PLACE_RETRIES]:
            if _can_place_patch(occupied, patch, int(bg), int(r0), int(c0), pad=1):
                _stamp_patch(canvas, occupied, patch, int(bg), int(r0), int(c0))
                placed = True
                break
        if not placed:
            return False
    return True


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = random.randint(MIN_PIECES, MAX_PIECES)
    interior_h = random.randint(5 if piece_count <= 3 else 6, 8)
    interior_w = random.randint(5 if piece_count <= 3 else 6, 8)
    if interior_h * interior_w < piece_count * MIN_PIECE_AREA:
        return None

    built = _build_pieces(interior_h, interior_w, piece_count)
    if built is None:
        return None
    palette = built["palette"]
    bg = int(palette["bg"])
    frame = int(palette["frame"])

    frame_h = interior_h + 2
    frame_w = interior_w + 2
    H = random.randint(max(16, frame_h + 6), 24)
    W = random.randint(max(16, frame_w + 6), 24)

    if H - frame_h < 4 or W - frame_w < 4:
        return None

    fr0 = random.randint(2, H - frame_h - 2)
    fc0 = random.randint(2, W - frame_w - 2)
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    out = np.full((H, W), int(bg), dtype=int)
    inp = np.full((H, W), int(bg), dtype=int)

    for r, c in _boundary_coords(fr0, fc0, fr1, fc1):
        out[r, c] = int(frame)
        inp[r, c] = int(frame)

    top_c = random.randint(fc0 + 1, fc1 - 1)
    right_r = random.randint(fr0 + 1, fr1 - 1)
    bottom_c = random.randint(fc0 + 1, fc1 - 1)
    left_r = random.randint(fr0 + 1, fr1 - 1)
    side_positions = {
        0: (fr0, top_c),
        1: (right_r, fc1),
        2: (fr1, bottom_c),
        3: (left_r, fc0),
    }
    for side, pos in side_positions.items():
        color = int(palette["side_colors"][int(side)])
        out[pos] = color
        inp[pos] = color

    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = built["interior"]

    scattered = []
    for piece in built["solved_pieces"]:
        rot = random.randint(0, 3)
        scattered.append(np.rot90(piece["patch"], int(rot)))

    occupied = np.zeros((H, W), dtype=bool)
    r0 = max(0, fr0 - 1)
    r1 = min(H, fr1 + 2)
    c0 = max(0, fc0 - 1)
    c1 = min(W, fc1 + 2)
    occupied[r0:r1, c0:c1] = True

    if not _scatter_pieces(inp, occupied, scattered, int(bg)):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
