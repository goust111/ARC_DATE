from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
AZURE = 8
COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_RETRIES = 400


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return r0, c0, r1, c1, mask


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot)
    return out


def _find_template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    frame = np.argwhere(grid == AZURE)
    assert frame.size > 0, "template frame not found"
    r0, c0 = frame.min(axis=0)
    r1, c1 = frame.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch[0, :] == AZURE)
    assert np.all(patch[-1, :] == AZURE)
    assert np.all(patch[:, 0] == AZURE)
    assert np.all(patch[:, -1] == AZURE)
    return r0, c0, r1, c1


def _extract_holes(template: np.ndarray) -> list[dict]:
    h, w = template.shape
    holes: list[dict] = []
    for coords in _components(template == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        r0, c0, r1, c1, mask = _tight_mask(coords)
        adjacent_colors: set[int] = set()
        for r, c in coords:
            for dr, dc in DIRS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and template[nr, nc] != BG:
                    adjacent_colors.add(int(template[nr, nc]))
        holes.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "mask": mask,
                "adjacent": adjacent_colors,
            }
        )
    holes.sort(key=lambda item: (item["r0"], item["c0"]))
    return holes


def _extract_pieces(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[dict]:
    search = grid.copy()
    r0, c0, r1, c1 = template_bbox
    search[r0 : r1 + 1, c0 : c1 + 1] = BG

    pieces: list[dict] = []
    for coords in _components(search != BG):
        pr0, pc0, pr1, pc1 = _bbox(coords)
        patch = search[pr0 : pr1 + 1, pc0 : pc1 + 1]
        grey = np.argwhere(patch == GREY)
        assert grey.size > 0, "piece interior must contain grey cells"
        gr0, gc0 = grey.min(axis=0)
        gr1, gc1 = grey.max(axis=0)
        gr0 = int(gr0)
        gc0 = int(gc0)
        gr1 = int(gr1)
        gc1 = int(gc1)
        mask = patch[gr0 : gr1 + 1, gc0 : gc1 + 1] == GREY
        colors = sorted(set(map(int, np.unique(patch))) - {BG, GREY})
        assert len(colors) == 1, "piece border color must be unique"
        pieces.append({"color": colors[0], "mask": mask})
    pieces.sort(key=lambda item: item["color"])
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered grey-centered fragments into the azure-framed template.

    The hole position is resolved by shape match under rotation and confirmed by
    the uniquely matching marker color adjacent to the hole.
    """
    g = np.asarray(grid, dtype=int)
    template_bbox = _find_template_bbox(g)
    tr0, tc0, tr1, tc1 = template_bbox
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()

    holes = _extract_holes(template)
    pieces = _extract_pieces(g, template_bbox)
    assert holes and pieces and len(holes) == len(pieces)

    used_holes: set[int] = set()
    for piece in pieces:
        matches: list[int] = []
        for idx, hole in enumerate(holes):
            if idx in used_holes or piece["color"] not in hole["adjacent"]:
                continue
            if any(
                rot.shape == hole["mask"].shape and np.array_equal(rot, hole["mask"])
                for rot in _rotations(piece["mask"])
            ):
                matches.append(idx)
        assert len(matches) == 1, "each piece must match exactly one hole"
        hole = holes[matches[0]]
        used_holes.add(matches[0])
        region = template[hole["r0"] : hole["r1"] + 1, hole["c0"] : hole["c1"] + 1]
        region[hole["mask"]] = int(piece["color"])

    assert len(used_holes) == len(holes)
    return template


def _choose_shapes(rng: random.Random, n_holes: int) -> list[tuple[int, int]]:
    canonical = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4)]
    bases: list[tuple[int, int]] = []
    if n_holes >= 3 and rng.random() < 0.5:
        dup = rng.choice(canonical)
        bases.extend([dup, dup])
    while len(bases) < n_holes:
        bases.append(rng.choice(canonical))
    rng.shuffle(bases)
    dims: list[tuple[int, int]] = []
    for a, b in bases[:n_holes]:
        if a != b and rng.random() < 0.5:
            dims.append((b, a))
        else:
            dims.append((a, b))
    return dims


def _place_holes(
    rng: random.Random, template_h: int, template_w: int, dims: list[tuple[int, int]]
) -> list[dict] | None:
    occ = np.zeros((template_h, template_w), dtype=bool)
    holes: list[dict] = []
    for h, w in sorted(dims, key=lambda item: item[0] * item[1], reverse=True):
        if template_h - h - 2 < 2 or template_w - w - 2 < 2:
            return None
        positions = [
            (r0, c0)
            for r0 in range(2, template_h - h - 1)
            for c0 in range(2, template_w - w - 1)
            if not occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any()
        ]
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        occ[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
        holes.append({"r0": r0, "c0": c0, "h": h, "w": w, "mask": np.ones((h, w), dtype=bool)})
    holes.sort(key=lambda item: (item["r0"], item["c0"]))
    return holes


def _pattern_value(r: int, c: int, colors: list[int], mode: int) -> int:
    n = len(colors)
    if mode == 0:
        idx = r + c
    elif mode == 1:
        idx = r + 2 * c
    elif mode == 2:
        idx = (r // 2) + c
    else:
        idx = 2 * (r % 2) + c
    return int(colors[idx % n])


def _build_template(
    rng: random.Random, template_h: int, template_w: int, holes: list[dict], piece_colors: list[int], fill_colors: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((template_h, template_w), BG, dtype=int)
    out = np.full((template_h, template_w), BG, dtype=int)
    inp[0, :] = inp[-1, :] = inp[:, 0] = inp[:, -1] = AZURE
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = AZURE

    mode = rng.randrange(4)
    for r in range(1, template_h - 1):
        for c in range(1, template_w - 1):
            color = _pattern_value(r, c, fill_colors, mode)
            inp[r, c] = color
            out[r, c] = color

    for hole, piece_color in zip(holes, piece_colors):
        r0 = int(hole["r0"])
        c0 = int(hole["c0"])
        h = int(hole["h"])
        w = int(hole["w"])
        inp[r0 : r0 + h, c0 : c0 + w] = BG
        out[r0 : r0 + h, c0 : c0 + w] = int(piece_color)

        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            mr, mc = r0 - 1, rng.randint(c0, c0 + w - 1)
        elif side == "bottom":
            mr, mc = r0 + h, rng.randint(c0, c0 + w - 1)
        elif side == "left":
            mr, mc = rng.randint(r0, r0 + h - 1), c0 - 1
        else:
            mr, mc = rng.randint(r0, r0 + h - 1), c0 + w
        inp[mr, mc] = int(piece_color)
        out[mr, mc] = int(piece_color)

    return inp, out


def _make_piece_patch(rng: random.Random, h: int, w: int, color: int) -> np.ndarray:
    patch = np.full((h + 2, w + 2), int(color), dtype=int)
    patch[1:-1, 1:-1] = GREY
    return np.rot90(patch, rng.randrange(4))


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    oh, ow = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > oh or c0 + w > ow:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(oh, r0 + h + pad)
    cc1 = min(ow, c0 + w + pad)
    return not occ[rr0:rr1, cc0:cc1].any()


def _place_pieces(
    rng: random.Random,
    grid: np.ndarray,
    occ: np.ndarray,
    pieces: list[np.ndarray],
    min_col: int,
) -> bool:
    h, w = grid.shape
    order = sorted(pieces, key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in order:
        ph, pw = map(int, patch.shape)
        positions = [
            (r0, c0)
            for r0 in range(1, h - ph)
            for c0 in range(min_col, w - pw)
            if _can_place(occ, r0, c0, ph, pw)
        ]
        if not positions:
            return False
        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_holes = rng.randint(2, 4)
    dims = _choose_shapes(rng, n_holes)
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)

    template_h = rng.randint(max(9, max_h + 6, 2 * n_holes + 3), 15)
    template_w = rng.randint(max(10, max_w + 7, 2 * n_holes + 4), 16)
    holes = _place_holes(rng, template_h, template_w, dims)
    if holes is None:
        return None

    colors = COLOR_POOL[:]
    rng.shuffle(colors)
    piece_colors = colors[:n_holes]
    remain = colors[n_holes:]
    fill_count = rng.randint(2, min(3, len(remain)))
    fill_colors = remain[:fill_count]

    template_in, template_out = _build_template(rng, template_h, template_w, holes, piece_colors, fill_colors)
    pieces = [
        _make_piece_patch(rng, int(hole["h"]), int(hole["w"]), int(color))
        for hole, color in zip(holes, piece_colors)
    ]

    left_margin = rng.randint(1, 2)
    right_margin = rng.randint(1, 2)
    gap = rng.randint(3, 5)
    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    max_piece_h = max(int(piece.shape[0]) for piece in pieces)
    total_piece_w = sum(int(piece.shape[1]) for piece in pieces) + 2 * (len(pieces) - 1)

    grid_h_min = max(template_h + top_margin + bottom_margin, max_piece_h + 4, 14)
    grid_h = rng.randint(grid_h_min, max(grid_h_min, template_h + 8, 22))
    right_zone_w_min = max(template_w + 2, total_piece_w // 2 + 4, max(int(piece.shape[1]) for piece in pieces) + 3)
    right_zone_w = rng.randint(right_zone_w_min, max(right_zone_w_min, total_piece_w + 8))
    grid_w = left_margin + template_w + gap + right_zone_w + right_margin

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    template_r0 = rng.randint(1, grid_h - template_h - 1)
    template_c0 = left_margin
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = template_in
    occ[
        max(0, template_r0 - 1) : min(grid_h, template_r0 + template_h + 1),
        max(0, template_c0 - 1) : min(grid_w, template_c0 + template_w + 1),
    ] = True

    if template_c0 + template_w > grid_w // 2:
        return None

    right_start = template_c0 + template_w + gap
    if not _place_pieces(rng, grid, occ, pieces, right_start):
        return None

    try:
        if not np.array_equal(main(grid), template_out):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
