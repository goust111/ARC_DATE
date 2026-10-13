import random
from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    assert not np.any((view != transparent) & mask)
    view[mask] = patch[mask]


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def connector_dir(sprite: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = sprite.shape
    dirs = []
    if r == 0:
        dirs.append((-1, 0))
    if r == h - 1:
        dirs.append((1, 0))
    if c == 0:
        dirs.append((0, -1))
    if c == w - 1:
        dirs.append((0, 1))
    assert len(dirs) == 1
    return dirs[0]


def rot180(arr: np.ndarray) -> np.ndarray:
    return arr[::-1, ::-1].copy()


def internal_hole_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    bg_mask = grid == bg
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return bg_mask & (~outside)


def assemble_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg)
    assert len(components) >= 2

    sprites = []
    by_color: dict[int, list[tuple[int, tuple[int, int], tuple[int, int]]]] = {}
    for idx, comp in enumerate(components):
        sprite = crop_bbox(g, comp["bbox"])
        sprites.append(sprite)
        for rr, cc in np.argwhere(sprite != bg):
            r = int(rr)
            c = int(cc)
            color = int(sprite[r, c])
            try:
                direction = connector_dir(sprite, r, c)
            except AssertionError:
                continue
            by_color.setdefault(color, []).append((idx, (r, c), direction))

    port_occurrences = {}
    for color, occ in by_color.items():
        if len(occ) != 2:
            continue
        (a_idx, _a_pos, _a_dir), (b_idx, _b_pos, _b_dir) = occ
        if a_idx == b_idx:
            continue
        port_occurrences[color] = occ
    assert port_occurrences

    adjacency = [[] for _ in sprites]
    for occ in port_occurrences.values():
        (a_idx, a_pos, a_dir), (b_idx, b_pos, b_dir) = occ
        assert b_dir == (-a_dir[0], -a_dir[1])
        delta = (a_pos[0] + a_dir[0] - b_pos[0], a_pos[1] + a_dir[1] - b_pos[1])
        adjacency[a_idx].append((b_idx, delta))
        adjacency[b_idx].append((a_idx, (-delta[0], -delta[1])))

    placements = {0: (0, 0)}
    q = deque([0])
    while q:
        idx = q.popleft()
        br, bc = placements[idx]
        for nbr, (dr, dc) in adjacency[idx]:
            pos = (br + dr, bc + dc)
            if nbr in placements:
                assert placements[nbr] == pos
            else:
                placements[nbr] = pos
                q.append(nbr)
    assert len(placements) == len(sprites)

    all_fg = []
    for idx, sprite in enumerate(sprites):
        br, bc = placements[idx]
        for rr, cc in np.argwhere(sprite != bg):
            all_fg.append((br + int(rr), bc + int(cc)))
    r0 = min(r for r, _ in all_fg)
    c0 = min(c for _, c in all_fg)
    r1 = max(r for r, _ in all_fg)
    c1 = max(c for _, c in all_fg)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for idx, sprite in enumerate(sprites):
        br, bc = placements[idx]
        paste(out, sprite, br - r0, bc - c0, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Dock equal-colored ports to assemble the pieces, then fill holes by whole-assembly central symmetry.

    Input structure:
    A dominant background contains multiple 4-connected multicolor fragments. Rare port
    colors appear exactly twice, once on each of two different fragment bbox edges, and
    the fragments may also contain internal background holes.

    Rule:
    Translate the cropped fragments so every equal-colored port pair touches edge-to-edge
    on opposite exposed sides, preserving all colors and orientations. In the assembled
    figure, fill each internal background hole cell with the color at its point-symmetric
    partner across the assembled tight crop's center, then return that tight crop.

    Key insight:
    The ports determine the translation-only assembly, and the missing colors are only
    recoverable after assembly because the required symmetric partners live on other pieces.
    """
    assembled = assemble_only(grid)
    bg = get_bg(assembled)
    out = assembled.copy()
    holes = internal_hole_mask(out, bg)
    h, w = out.shape
    for rr, cc in np.argwhere(holes):
        r = int(rr)
        c = int(cc)
        sr = h - 1 - r
        sc = w - 1 - c
        assert int(out[sr, sc]) != bg
        out[r, c] = int(out[sr, sc])
    return out


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    n = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            n += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return n


def _interior_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                out.append((r, c))
    return out


def _carve_notch(mask: np.ndarray, side: str, rng: random.Random) -> bool:
    h, w = mask.shape
    trial = mask.copy()
    if side == "T":
        depth = rng.randint(1, min(2, h - 3))
        width = rng.randint(1, max(1, w - 3))
        c0 = rng.randint(0, w - width - 1)
        trial[:depth, c0 : c0 + width] = False
    elif side == "B":
        depth = rng.randint(1, min(2, h - 3))
        width = rng.randint(1, max(1, w - 3))
        c0 = rng.randint(1, w - width)
        trial[h - depth :, c0 : c0 + width] = False
    elif side == "L":
        depth = rng.randint(1, min(2, w - 3))
        height = rng.randint(1, max(1, h - 3))
        r0 = rng.randint(1, h - height)
        trial[r0 : r0 + height, :depth] = False
    elif side == "R":
        depth = rng.randint(1, min(2, w - 3))
        height = rng.randint(1, max(1, h - 3))
        r0 = rng.randint(0, h - height - 1)
        trial[r0 : r0 + height, w - depth :] = False
    else:
        return False
    if not np.any(trial):
        return False
    if trial[0].sum() == 0 or trial[-1].sum() == 0 or trial[:, 0].sum() == 0 or trial[:, -1].sum() == 0:
        return False
    if _component_count(trial) != 1:
        return False
    if len(_interior_candidates(trial)) < 6:
        return False
    mask[:, :] = trial
    return True


def _dock_side(axis: str) -> tuple[str, str]:
    return ("R", "L") if axis == "h" else ("B", "T")


def _port_positions(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    if side == "R":
        c = w - 1
        for r in range(1, h - 1):
            if mask[r, c] and mask[r, c - 1]:
                out.append((r, c))
    elif side == "L":
        c = 0
        for r in range(1, h - 1):
            if mask[r, c] and mask[r, c + 1]:
                out.append((r, c))
    elif side == "B":
        r = h - 1
        for c in range(1, w - 1):
            if mask[r, c] and mask[r - 1, c]:
                out.append((r, c))
    else:
        r = 0
        for c in range(1, w - 1):
            if mask[r, c] and mask[r + 1, c]:
                out.append((r, c))
    return out


def _make_base_mask(h: int, w: int, axis: str, rng: random.Random) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    notch_sides = ["T", "B", "L"] if axis == "h" else ["L", "R", "T"]
    for _ in range(rng.randint(0, 3)):
        rng.shuffle(notch_sides)
        for side in notch_sides:
            if _carve_notch(mask, side, rng):
                break
    return mask


def _paint_piece(mask: np.ndarray, axis: str, port_rows: list[tuple[int, int]], colors: dict, rng: random.Random) -> np.ndarray:
    bg = colors["bg"]
    piece = np.full(mask.shape, bg, dtype=int)
    piece[mask] = colors["body"]

    h, w = mask.shape
    accent_pool = [(r, c) for r, c in _interior_candidates(mask) if (r, c) not in port_rows]
    rng.shuffle(accent_pool)

    accent_count = rng.randint(3, min(7, max(3, len(accent_pool))))
    for r, c in accent_pool[:accent_count]:
        piece[r, c] = colors["accent1"]

    if len(accent_pool) >= 10 and rng.random() < 0.55:
        tail = accent_pool[accent_count:]
        second_count = rng.randint(3, min(5, len(tail)))
        for r, c in tail[:second_count]:
            piece[r, c] = colors["accent2"]

    if axis == "h":
        band_col = rng.randint(1, max(1, w // 2))
        for r in range(1, h - 1):
            if mask[r, band_col]:
                piece[r, band_col] = colors["accent1"]
    else:
        band_row = rng.randint(1, max(1, h // 2))
        for c in range(1, w - 1):
            if mask[band_row, c]:
                piece[band_row, c] = colors["accent1"]

    return piece


def _nonadjacent_subset(candidates: list[tuple[int, int]], need: int, rng: random.Random) -> list[tuple[int, int]] | None:
    rng.shuffle(candidates)
    chosen = []
    for cell in candidates:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 2 for other in chosen):
            chosen.append(cell)
            if len(chosen) == need:
                return chosen
    return None


def _hole_cells(piece: np.ndarray, bg: int) -> set[tuple[int, int]]:
    holes = internal_hole_mask(piece, bg)
    return {tuple(map(int, rc)) for rc in np.argwhere(holes)}


def _scatter_positions(shape: tuple[int, int], patch_shape: tuple[int, int], rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = patch_shape
    H, W = shape
    positions = []
    for _ in range(200):
        positions.clear()
        for _k in range(2):
            r = rng.randint(1, H - h - 1)
            c = rng.randint(1, W - w - 1)
            positions.append((r, c))
        (r0, c0), (r1, c1) = positions
        row_gap = max(0, max(r0, r1) - min(r0 + h - 1, r1 + h - 1) - 1)
        col_gap = max(0, max(c0, c1) - min(c0 + w - 1, c1 + w - 1) - 1)
        if row_gap >= 2 or col_gap >= 2:
            return positions[0], positions[1]
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(list(range(10)))
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    axis = rng.choice(["h", "v"])
    h = rng.randint(6, 10)
    w = rng.randint(6, 10)
    mask_a = _make_base_mask(h, w, axis, rng)

    dock_a, dock_b = _dock_side(axis)
    port_a = _port_positions(mask_a, dock_a)
    if len(port_a) < 1:
        return None

    k_ports = 1
    if len(port_a) < k_ports:
        return None

    colors = {
        "bg": bg,
        "body": palette[0],
        "accent1": palette[1],
        "accent2": palette[2],
    }
    chosen_ports = rng.sample(port_a, k_ports)
    piece_a_full = _paint_piece(mask_a, axis, chosen_ports, colors, rng)
    piece_b_full = rot180(piece_a_full)

    port_colors = palette[3 : 3 + k_ports]
    if len(port_colors) < k_ports:
        return None
    for color, (r, c) in zip(port_colors, chosen_ports):
        piece_a_full[r, c] = color
        piece_b_full[h - 1 - r, w - 1 - c] = color

    if np.array_equal(piece_a_full, rot180(piece_a_full)):
        return None

    candidate_holes = []
    for r, c in _interior_candidates(mask_a):
        rr, cc = h - 1 - r, w - 1 - c
        if int(piece_a_full[r, c]) == bg or int(piece_a_full[rr, cc]) == bg:
            continue
        if int(piece_a_full[r, c]) in port_colors or int(piece_a_full[rr, cc]) in port_colors:
            continue
        if int(piece_a_full[r, c]) == int(piece_a_full[rr, cc]):
            continue
        candidate_holes.append((r, c))
    if len(candidate_holes) < 2:
        return None

    hole_mode = rng.choice(["A", "B", "both"])
    hole_count = rng.choice([1, 2, 2, 3, 4])
    chosen = _nonadjacent_subset(candidate_holes, hole_count, rng)
    if chosen is None:
        return None

    piece_a = piece_a_full.copy()
    piece_b = piece_b_full.copy()
    hole_targets_a = set()
    hole_targets_b = set()
    if hole_mode == "A":
        assign = ["A"] * hole_count
    elif hole_mode == "B":
        assign = ["B"] * hole_count
    else:
        assign = ["A"] * ((hole_count + 1) // 2) + ["B"] * (hole_count // 2)
        rng.shuffle(assign)

    for cell, side in zip(chosen, assign):
        r, c = cell
        rr, cc = h - 1 - r, w - 1 - c
        if side == "A":
            piece_a[r, c] = bg
            hole_targets_a.add((r, c))
        else:
            piece_b[rr, cc] = bg
            hole_targets_b.add((rr, cc))

    if _component_count(piece_a != bg) != 1 or _component_count(piece_b != bg) != 1:
        return None
    if _hole_cells(piece_a, bg) != hole_targets_a:
        return None
    if _hole_cells(piece_b, bg) != hole_targets_b:
        return None

    local_shortcut_fail = False
    for r, c in hole_targets_a:
        if int(piece_a_full[h - 1 - r, w - 1 - c]) != int(piece_b_full[h - 1 - r, w - 1 - c]):
            local_shortcut_fail = True
    for r, c in hole_targets_b:
        if int(piece_b_full[h - 1 - r, w - 1 - c]) != int(piece_a_full[h - 1 - r, w - 1 - c]):
            local_shortcut_fail = True
    if not local_shortcut_fail:
        return None

    min_h = max(h + 4, 18)
    min_w = max(w + 4, 18)
    if min_h > 30 or min_w > 30:
        return None
    H = rng.randint(min_h, 30)
    W = rng.randint(min_w, 30)
    pos = _scatter_positions((H, W), (h, w), rng)
    if pos is None:
        return None
    (r0, c0), (r1, c1) = pos

    inp = np.full((H, W), bg, dtype=int)
    paste(inp, piece_a, r0, c0, bg)
    paste(inp, piece_b, r1, c1, bg)
    full_inp = np.full((H, W), bg, dtype=int)
    paste(full_inp, piece_a_full, r0, c0, bg)
    paste(full_inp, piece_b_full, r1, c1, bg)

    try:
        expected = assemble_only(full_inp)
        assembled = assemble_only(inp)
        out = main(inp)
    except Exception:
        return None
    if max(expected.shape) > 30:
        return None
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(assembled, expected):
        return None

    seen_ports = []
    for color in port_colors:
        locs = np.argwhere(inp == color)
        if locs.shape[0] != 2:
            return None
        seen_ports.append(int(color))
    if len(seen_ports) != k_ports:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
