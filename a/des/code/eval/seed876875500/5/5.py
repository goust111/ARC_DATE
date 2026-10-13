from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _crop_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _border_mode(grid: np.ndarray) -> int:
    border = list(map(int, grid[0, :]))
    border.extend(map(int, grid[-1, :]))
    border.extend(map(int, grid[1:-1, 0]))
    border.extend(map(int, grid[1:-1, -1]))
    return Counter(border).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _border_mode(g)

    comp_infos = []
    for coords in _components(g != bg):
        comp_infos.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "area": len(coords),
            }
        )
    if not comp_infos:
        return g.copy()

    main_idx = max(range(len(comp_infos)), key=lambda i: comp_infos[i]["area"])
    main_coords = comp_infos[main_idx]["coords"]
    main_mask = np.zeros((h, w), dtype=bool)
    for r, c in main_coords:
        main_mask[r, c] = True

    grey = Counter(int(g[r, c]) for r, c in main_coords).most_common(1)[0][0]

    holes = []
    for coords in _components(g == bg):
        if _touches_border(coords, h, w):
            continue
        anchors: dict[int, tuple[int, int]] = {}
        touches_main = False
        for r, c in coords:
            for nr, nc in _neighbors4(r, c, h, w):
                if not main_mask[nr, nc]:
                    continue
                touches_main = True
                color = int(g[nr, nc])
                if color != grey:
                    anchors[color] = (nr, nc)
        if not touches_main or not anchors:
            continue
        bbox = _bbox(coords)
        holes.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": _crop_mask(coords, bbox),
                "anchors": anchors,
            }
        )

    pieces = []
    for idx, info in enumerate(comp_infos):
        if idx == main_idx:
            continue
        patch = _crop_component(g, info["coords"], bg)
        markers = {}
        for rr, cc in np.argwhere((patch != bg) & (patch != grey)):
            rr = int(rr)
            cc = int(cc)
            markers[int(patch[rr, cc])] = (rr, cc)
        pieces.append({"patch": patch, "markers": markers})

    hole_by_signature = {frozenset(hole["anchors"]): hole for hole in holes}
    out = np.full_like(g, bg)
    out[main_mask] = g[main_mask]

    for piece in pieces:
        sig = frozenset(piece["markers"])
        if sig not in hole_by_signature:
            continue
        hole = hole_by_signature[sig]
        matches = []
        for rot_k in range(4):
            rotated = np.rot90(piece["patch"], rot_k)
            occ = rotated != bg
            if rotated.shape != hole["mask"].shape or not np.array_equal(occ, hole["mask"]):
                continue
            hr0, hc0, _hr1, _hc1 = hole["bbox"]
            ok = True
            for color, (ar, ac) in hole["anchors"].items():
                pos = np.argwhere(rotated == color)
                if pos.shape[0] != 1:
                    ok = False
                    break
                pr, pc = map(int, pos[0])
                if abs((hr0 + pr) - ar) + abs((hc0 + pc) - ac) != 1:
                    ok = False
                    break
            if ok:
                matches.append(rot_k)
        if len(matches) != 1:
            continue
        for r, c in hole["coords"]:
            out[r, c] = grey

    return out


def _boundary_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                cells.append((r, c))
    return cells


def _anchor_dirs(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    dirs = []
    if r == 0:
        dirs.append((-1, 0))
    if r == height - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == width - 1:
        dirs.append((0, 1))
    return dirs


def _rotation_candidates(
    piece: np.ndarray, hole_shape: tuple[int, int], anchor_rels: dict[int, tuple[int, int]]
) -> list[int]:
    candidates = []
    for rot_k in range(4):
        rotated = np.rot90(piece, rot_k)
        if rotated.shape != hole_shape or not np.all(rotated != BG):
            continue
        ok = True
        for color, (ar, ac) in anchor_rels.items():
            pos = np.argwhere(rotated == color)
            if pos.shape[0] != 1:
                ok = False
                break
            pr, pc = map(int, pos[0])
            if abs(pr - ar) + abs(pc - ac) != 1:
                ok = False
                break
        if ok:
            candidates.append(rot_k)
    return candidates


def _make_piece_spec(
    height: int, width: int, colors: list[int], rng: random.Random
) -> dict[str, object] | None:
    piece = np.full((height, width), GREY, dtype=int)
    boundary = _boundary_cells(height, width)
    for _ in range(200):
        anchor_rels: dict[int, tuple[int, int]] = {}
        piece[:] = GREY
        chosen_cells = rng.sample(boundary, len(colors))
        for color, (r, c) in zip(colors, chosen_cells):
            dr, dc = rng.choice(_anchor_dirs(r, c, height, width))
            piece[r, c] = int(color)
            anchor_rels[int(color)] = (r + dr, c + dc)
        display_rot = rng.randrange(4)
        scattered = np.rot90(piece, display_rot)
        if len(_rotation_candidates(scattered, (height, width), anchor_rels)) == 1:
            return {
                "hole_h": height,
                "hole_w": width,
                "piece_base": piece.copy(),
                "piece_input": scattered,
                "anchor_rels": anchor_rels,
                "display_rot": display_rot,
            }
    return None


def _place_holes(
    main_h: int, main_w: int, specs: list[dict[str, object]], rng: random.Random
) -> bool:
    reserved = np.zeros((main_h, main_w), dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda i: int(specs[i]["hole_h"]) * int(specs[i]["hole_w"]),
        reverse=True,
    )
    for idx in order:
        hole_h = int(specs[idx]["hole_h"])
        hole_w = int(specs[idx]["hole_w"])
        if main_h - hole_h < 3 or main_w - hole_w < 3:
            return False
        placed = False
        for _ in range(300):
            r0 = rng.randint(1, main_h - hole_h - 1)
            c0 = rng.randint(1, main_w - hole_w - 1)
            rr0 = max(0, r0 - 1)
            rr1 = min(main_h, r0 + hole_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(main_w, c0 + hole_w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            reserved[rr0:rr1, cc0:cc1] = True
            specs[idx]["hole_pos"] = (r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, ph: int, pw: int) -> bool:
    h, w = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + ph > h - 1 or c0 + pw > w - 1:
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + pw + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_piece(
    occupied: np.ndarray,
    patch: np.ndarray,
    side: str,
    main_bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    grid_h, grid_w = occupied.shape
    ph, pw = map(int, patch.shape)
    mr0, mc0, mr1, mc1 = main_bbox
    if side == "top":
        row_min, row_max = 1, mr0 - ph - 1
        col_min, col_max = 1, grid_w - pw - 1
    elif side == "bottom":
        row_min, row_max = mr1 + 2, grid_h - ph - 1
        col_min, col_max = 1, grid_w - pw - 1
    elif side == "left":
        row_min, row_max = 1, grid_h - ph - 1
        col_min, col_max = 1, mc0 - pw - 1
    else:
        row_min, row_max = 1, grid_h - ph - 1
        col_min, col_max = mc1 + 2, grid_w - pw - 1
    if row_min > row_max or col_min > col_max:
        return None
    for _ in range(300):
        r0 = rng.randint(row_min, row_max)
        c0 = rng.randint(col_min, col_max)
        if _can_place_bbox(occupied, r0, c0, ph, pw):
            return r0, c0
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    hole_count = rng.randint(2, 4)
    anchor_counts = [rng.choice([1, 2]) for _ in range(hole_count)]
    total_anchor_count = sum(anchor_counts)
    colors = rng.sample(list(ANCHOR_COLORS), total_anchor_count)

    specs: list[dict[str, object]] = []
    color_idx = 0
    for anchor_count in anchor_counts:
        hole_h = rng.randint(2, 4)
        hole_w = rng.randint(2, 4)
        if rng.random() < 0.35:
            hole_w = hole_h
        piece_colors = colors[color_idx : color_idx + anchor_count]
        color_idx += anchor_count
        spec = _make_piece_spec(hole_h, hole_w, piece_colors, rng)
        if spec is None:
            return None
        specs.append(spec)

    top_need = 2
    bottom_need = 2
    left_need = 2
    right_need = 2
    sides = rng.sample(["top", "bottom", "left", "right"], hole_count)
    for spec, side in zip(specs, sides):
        spec["side"] = side
        ph, pw = map(int, spec["piece_input"].shape)
        if side == "top":
            top_need = max(top_need, ph + 2)
        elif side == "bottom":
            bottom_need = max(bottom_need, ph + 2)
        elif side == "left":
            left_need = max(left_need, pw + 2)
        else:
            right_need = max(right_need, pw + 2)

    main_h = rng.randint(max(10, max(int(s["hole_h"]) for s in specs) + 6), 15)
    main_w = rng.randint(max(10, max(int(s["hole_w"]) for s in specs) + 6), 15)
    if not _place_holes(main_h, main_w, specs, rng):
        return None

    top_margin = top_need + rng.randint(0, 2)
    bottom_margin = bottom_need + rng.randint(0, 2)
    left_margin = left_need + rng.randint(0, 2)
    right_margin = right_need + rng.randint(0, 2)
    grid_h = top_margin + main_h + bottom_margin
    grid_w = left_margin + main_w + right_margin
    if not (18 <= grid_h <= 30 and 18 <= grid_w <= 30):
        return None

    main_r0 = top_margin
    main_c0 = left_margin
    main_r1 = main_r0 + main_h - 1
    main_c1 = main_c0 + main_w - 1
    main_bbox = (main_r0, main_c0, main_r1, main_c1)

    solved = np.full((grid_h, grid_w), BG, dtype=int)
    solved[main_r0 : main_r1 + 1, main_c0 : main_c1 + 1] = GREY

    inp = solved.copy()
    for spec in specs:
        hole_h = int(spec["hole_h"])
        hole_w = int(spec["hole_w"])
        hole_r, hole_c = spec["hole_pos"]
        abs_r = main_r0 + hole_r
        abs_c = main_c0 + hole_c
        inp[abs_r : abs_r + hole_h, abs_c : abs_c + hole_w] = BG
        for color, (ar, ac) in spec["anchor_rels"].items():
            solved[abs_r + ar, abs_c + ac] = int(color)
            inp[abs_r + ar, abs_c + ac] = int(color)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[main_r0 : main_r1 + 1, main_c0 : main_c1 + 1] = True
    for spec in specs:
        patch = np.asarray(spec["piece_input"], dtype=int)
        placed = _place_piece(occupied, patch, str(spec["side"]), main_bbox, rng)
        if placed is None:
            return None
        pr0, pc0 = placed
        ph, pw = patch.shape
        inp[pr0 : pr0 + ph, pc0 : pc0 + pw] = patch
        occupied[pr0 : pr0 + ph, pc0 : pc0 + pw] = True

    if not np.array_equal(main(inp), solved):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
