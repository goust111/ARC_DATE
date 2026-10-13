from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_PAIR_OFFSETS = 12
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                out.append((rr, cc))
                break
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    mask = src != BG
    dst[r0 : r0 + h, c0 : c0 + w][mask] = src[mask]


def _extract_legend(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    candidates = []
    for coords in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != r1:
            continue
        vals = {int(grid[r, c]) for r, c in coords}
        if len(vals) < 2:
            continue
        if r0 not in (0, h - 1):
            continue
        if c0 != 0 and c1 != w - 1:
            continue
        full = grid[r0, c0 : c1 + 1]
        if np.any(full == BG):
            continue
        candidates.append((c1 - c0 + 1, r0, c0, c1))
    if not candidates:
        raise ValueError("legend strip not found")
    _, r0, c0, c1 = max(candidates)
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    colors = [int(x) for x in grid[r0, c0 : c1 + 1]]
    return mask, colors


def _extract_pieces(grid: np.ndarray, blocked: np.ndarray) -> list[dict]:
    pieces = []
    for coords in _components((grid != BG) & (~blocked)):
        mask = _mask_from_coords(coords)
        color = int(grid[coords[0][0], coords[0][1]])
        pieces.append(
            {
                "color": color,
                "mask": mask,
                "coords": [(int(r), int(c)) for r, c in np.argwhere(mask)],
                "set": {(int(r), int(c)) for r, c in np.argwhere(mask)},
                "boundary": _boundary_cells(mask),
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "area": int(mask.sum()),
            }
        )
    if not pieces:
        raise ValueError("no pieces found")
    pieces.sort(key=lambda p: (-p["area"], p["h"], p["w"]))
    return pieces


def _overlap(set_a: set[tuple[int, int]], coords_b: list[tuple[int, int]], dr: int, dc: int) -> bool:
    for r, c in coords_b:
        if (r + dr, c + dc) in set_a:
            return True
    return False


def _contact_count(set_a: set[tuple[int, int]], coords_b: list[tuple[int, int]], dr: int, dc: int) -> int:
    total = 0
    for r, c in coords_b:
        rr = r + dr
        cc = c + dc
        for sr, sc in DIRS4:
            if (rr + sr, cc + sc) in set_a:
                total += 1
    return total


def _pair_offsets(piece_a: dict, piece_b: dict) -> list[tuple[int, int, int]]:
    scores: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()
    for ar, ac in piece_a["boundary"]:
        for br, bc in piece_b["boundary"]:
            for dr, dc in DIRS4:
                off = (ar + dr - br, ac + dc - bc)
                if off in seen:
                    continue
                seen.add(off)
                if _overlap(piece_a["set"], piece_b["coords"], off[0], off[1]):
                    continue
                contacts = _contact_count(piece_a["set"], piece_b["coords"], off[0], off[1])
                if contacts >= 2:
                    scores[off] = max(scores.get(off, 0), contacts)
    if not scores:
        return []
    best = max(scores.values())
    keep = [
        (dr, dc, score)
        for (dr, dc), score in scores.items()
        if score >= max(2, best - 1)
    ]
    keep.sort(key=lambda item: (-item[2], abs(item[0]) + abs(item[1]), item[0], item[1]))
    return keep[:MAX_PAIR_OFFSETS]


def _assemble_positions(pieces: list[dict]) -> dict[int, tuple[int, int]]:
    n = len(pieces)
    total_area = sum(piece["area"] for piece in pieces)
    offsets = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                offsets[i][j] = _pair_offsets(pieces[i], pieces[j])

    anchor = 0
    anchor_piece = pieces[anchor]
    positions: dict[int, tuple[int, int]] = {anchor: (0, 0)}
    occupied = set(anchor_piece["coords"])
    placed_area = anchor_piece["area"]

    def dfs(
        pos_map: dict[int, tuple[int, int]],
        occ: set[tuple[int, int]],
        min_r: int,
        max_r: int,
        min_c: int,
        max_c: int,
        occ_area: int,
    ) -> dict[int, tuple[int, int]] | None:
        if len(pos_map) == n:
            if (max_r - min_r + 1) * (max_c - min_c + 1) != total_area:
                return None
            return {idx: (r - min_r, c - min_c) for idx, (r, c) in pos_map.items()}

        best_piece = None
        best_options = None

        for j in range(n):
            if j in pos_map:
                continue
            pos_scores: dict[tuple[int, int], int] = defaultdict(int)
            for i, (ir, ic) in pos_map.items():
                for dr, dc, score in offsets[i][j]:
                    pos_scores[(ir + dr, ic + dc)] += score

            options = []
            piece = pieces[j]
            for (pr, pc), score in pos_scores.items():
                if _overlap(occ, piece["coords"], pr, pc):
                    continue

                touches = False
                for r, c in piece["coords"]:
                    rr = pr + r
                    cc = pc + c
                    for dr, dc in DIRS4:
                        if (rr + dr, cc + dc) in occ:
                            touches = True
                            break
                    if touches:
                        break
                if not touches:
                    continue

                new_min_r = min(min_r, pr)
                new_max_r = max(max_r, pr + piece["h"] - 1)
                new_min_c = min(min_c, pc)
                new_max_c = max(max_c, pc + piece["w"] - 1)
                bbox_area = (new_max_r - new_min_r + 1) * (new_max_c - new_min_c + 1)
                if bbox_area > total_area:
                    continue
                options.append((pr, pc, score, bbox_area))

            options.sort(key=lambda item: (-item[2], item[3], item[0], item[1]))
            if not options:
                continue
            if best_options is None or len(options) < len(best_options):
                best_piece = j
                best_options = options

        if best_piece is None or best_options is None:
            return None

        piece = pieces[best_piece]
        for pr, pc, _score, _bbox_area in best_options:
            new_pos = dict(pos_map)
            new_pos[best_piece] = (pr, pc)
            new_occ = set(occ)
            for r, c in piece["coords"]:
                new_occ.add((pr + r, pc + c))
            result = dfs(
                new_pos,
                new_occ,
                min(min_r, pr),
                max(max_r, pr + piece["h"] - 1),
                min(min_c, pc),
                max(max_c, pc + piece["w"] - 1),
                occ_area + piece["area"],
            )
            if result is not None:
                return result
        return None

    result = dfs(
        positions,
        occupied,
        0,
        anchor_piece["h"] - 1,
        0,
        anchor_piece["w"] - 1,
        placed_area,
    )
    if result is None:
        raise ValueError("assembly failed")
    return result


def _neighbor_sets_from_positions(
    pieces: list[dict], positions: dict[int, tuple[int, int]]
) -> tuple[dict[int, set[int]], tuple[int, int]]:
    owner: dict[tuple[int, int], int] = {}
    for idx, (r0, c0) in positions.items():
        for r, c in pieces[idx]["coords"]:
            owner[(r0 + r, c0 + c)] = idx

    rows = [r for r, _ in owner]
    cols = [c for _, c in owner]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1

    neighbors = {idx: set() for idx in positions}
    for (r, c), idx in owner.items():
        for dr, dc in DIRS4:
            other = owner.get((r + dr, c + dc))
            if other is not None and other != idx:
                neighbors[idx].add(other)
    return neighbors, (h, w)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_mask, legend_colors = _extract_legend(g)
    pieces = _extract_pieces(g, legend_mask)
    positions = _assemble_positions(pieces)
    neighbors, (h, w) = _neighbor_sets_from_positions(pieces, positions)

    out = np.zeros((h, w), dtype=int)
    for idx, piece in enumerate(pieces):
        neighbor_count = len(neighbors[idx])
        if neighbor_count < 1 or neighbor_count > len(legend_colors):
            raise ValueError("neighbor count outside legend range")
        color = int(legend_colors[neighbor_count - 1])
        r0, c0 = positions[idx]
        for r, c in piece["coords"]:
            out[r0 + r, c0 + c] = color
    return out


def _neighbors_from_owner(owner: np.ndarray) -> dict[int, set[int]]:
    ids = [int(x) for x in np.unique(owner)]
    out = {idx: set() for idx in ids}
    h, w = owner.shape
    for r in range(h):
        for c in range(w):
            cur = int(owner[r, c])
            if r + 1 < h and owner[r + 1, c] != cur:
                nxt = int(owner[r + 1, c])
                out[cur].add(nxt)
                out[nxt].add(cur)
            if c + 1 < w and owner[r, c + 1] != cur:
                nxt = int(owner[r, c + 1])
                out[cur].add(nxt)
                out[nxt].add(cur)
    return out


def _oracle_output(owner: np.ndarray, legend_colors: list[int]) -> np.ndarray:
    neighbors = _neighbors_from_owner(owner)
    out = np.zeros_like(owner)
    for idx, nbs in neighbors.items():
        out[owner == idx] = int(legend_colors[len(nbs) - 1])
    return out


def _connected_owner(owner: np.ndarray) -> bool:
    for idx in np.unique(owner):
        mask = owner == idx
        comps = _components(mask)
        if len(comps) != 1:
            return False
    return True


def _rand_parts(n: int, lo: int, hi: int) -> list[int]:
    return [random.randint(lo, hi) for _ in range(n)]


def _family_strip() -> np.ndarray:
    n = random.randint(3, 5)
    vertical = random.choice([True, False])
    if vertical:
        widths = _rand_parts(n, 3, 5)
        h = random.randint(6, 10)
        w = sum(widths)
        owner = np.zeros((h, w), dtype=int)
        c = 0
        for idx, width in enumerate(widths):
            owner[:, c : c + width] = idx
            c += width
    else:
        heights = _rand_parts(n, 3, 5)
        h = sum(heights)
        w = random.randint(6, 10)
        owner = np.zeros((h, w), dtype=int)
        r = 0
        for idx, height in enumerate(heights):
            owner[r : r + height, :] = idx
            r += height
    return owner


def _family_t() -> np.ndarray:
    mode = random.choice(["top", "bottom", "left", "right"])
    if mode in ("top", "bottom"):
        widths = _rand_parts(3, 3, 5)
        bar_h = random.randint(3, 5)
        rest_h = random.randint(4, 7)
        h = bar_h + rest_h
        w = sum(widths)
        owner = np.zeros((h, w), dtype=int)
        if mode == "top":
            owner[:bar_h, :] = 0
            r0 = bar_h
        else:
            owner[h - bar_h :, :] = 0
            r0 = 0
        c = 0
        for idx, width in enumerate(widths, start=1):
            owner[r0 : r0 + rest_h, c : c + width] = idx
            c += width
    else:
        heights = _rand_parts(3, 3, 5)
        bar_w = random.randint(3, 5)
        rest_w = random.randint(4, 7)
        h = sum(heights)
        w = bar_w + rest_w
        owner = np.zeros((h, w), dtype=int)
        if mode == "left":
            owner[:, :bar_w] = 0
            c0 = bar_w
        else:
            owner[:, w - bar_w :] = 0
            c0 = 0
        r = 0
        for idx, height in enumerate(heights, start=1):
            owner[r : r + height, c0 : c0 + rest_w] = idx
            r += height
    return owner


def _family_plus() -> np.ndarray:
    left_w, center_w, right_w = _rand_parts(3, 3, 5)
    top_h, center_h, bottom_h = _rand_parts(3, 3, 5)
    h = top_h + center_h + bottom_h
    w = left_w + center_w + right_w
    owner = np.zeros((h, w), dtype=int)
    owner[:, :left_w] = 0
    owner[:, left_w + center_w :] = 1
    owner[:top_h, left_w : left_w + center_w] = 2
    owner[top_h : top_h + center_h, left_w : left_w + center_w] = 3
    owner[top_h + center_h :, left_w : left_w + center_w] = 4
    return owner


def _make_owner() -> np.ndarray:
    family = random.choice(["strip", "t", "plus"])
    if family == "strip":
        owner = _family_strip()
    elif family == "t":
        owner = _family_t()
    else:
        owner = _family_plus()
    perm = list(range(int(owner.max()) + 1))
    random.shuffle(perm)
    remapped = owner.copy()
    for old, new in enumerate(perm):
        remapped[owner == old] = new
    return remapped


def _boundary_segments(owner: np.ndarray) -> list[tuple[str, int, int, int, int, int]]:
    h, w = owner.shape
    segments: list[tuple[str, int, int, int, int, int]] = []

    for c in range(1, w):
        r = 0
        while r < h:
            left = int(owner[r, c - 1])
            right = int(owner[r, c])
            if left == right:
                r += 1
                continue
            r0 = r
            while r + 1 < h and int(owner[r + 1, c - 1]) == left and int(owner[r + 1, c]) == right:
                r += 1
            segments.append(("v", c, r0, r, left, right))
            r += 1

    for r in range(1, h):
        c = 0
        while c < w:
            top = int(owner[r - 1, c])
            bottom = int(owner[r, c])
            if top == bottom:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(owner[r - 1, c + 1]) == top and int(owner[r, c + 1]) == bottom:
                c += 1
            segments.append(("h", r, c0, c, top, bottom))
            c += 1

    return segments


def _apply_tabs(owner: np.ndarray) -> np.ndarray | None:
    out = owner.copy()
    for kind, fixed, start, end, a, b in _boundary_segments(owner):
        if end - start + 1 < 3:
            return None
        pivot = random.randint(start + 1, end - 1)
        if kind == "v":
            if random.choice([True, False]):
                out[pivot, fixed] = a
            else:
                out[pivot, fixed - 1] = b
        else:
            if random.choice([True, False]):
                out[fixed, pivot] = a
            else:
                out[fixed - 1, pivot] = b
    if not _connected_owner(out):
        return None
    return out


def _piece_patches(owner: np.ndarray, colors: list[int]) -> list[np.ndarray]:
    patches = []
    for idx, color in enumerate(colors):
        mask = owner == idx
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        sub = mask[r0 : r1 + 1, c0 : c1 + 1]
        patch[sub] = int(color)
        patches.append(patch)
    return patches


def _place_scene(patches: list[np.ndarray], legend_colors: list[int]) -> np.ndarray | None:
    max_piece_h = max(p.shape[0] for p in patches)
    max_piece_w = max(p.shape[1] for p in patches)
    total_bbox_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    legend_len = len(legend_colors)

    base_h = max(18, max_piece_h + 6, int(np.ceil(np.sqrt(total_bbox_area * 2.2))))
    base_w = max(18, max_piece_w + 6, int(np.ceil(np.sqrt(total_bbox_area * 2.2))))
    canvas_h = min(30, base_h + random.randint(0, max(0, 30 - base_h)))
    canvas_w = min(30, base_w + random.randint(0, max(0, 30 - base_w)))
    if canvas_h < max_piece_h + 4 or canvas_w < max_piece_w + 4 or canvas_w < legend_len + 2:
        return None

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)

    corner = random.choice(["tl", "tr", "bl", "br"])
    legend = np.array(legend_colors, dtype=int)[None, :]
    if corner == "tl":
        lr, lc = 0, 0
    elif corner == "tr":
        lr, lc = 0, canvas_w - legend_len
    elif corner == "bl":
        lr, lc = canvas_h - 1, 0
    else:
        lr, lc = canvas_h - 1, canvas_w - legend_len
    _paste(grid, legend, lr, lc)
    blocked[lr, lc : lc + legend_len] = True
    blocked = _dilate8(blocked)

    occupied = blocked.copy()
    for patch in sorted(patches, key=lambda p: -int(np.count_nonzero(p))):
        mask = patch != BG
        ph, pw = patch.shape
        positions = []
        for r0 in range(1, canvas_h - ph):
            for c0 in range(1, canvas_w - pw):
                if not (occupied[r0 : r0 + ph, c0 : c0 + pw] & mask).any():
                    positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paste(grid, patch, r0, c0)
        used = np.zeros_like(grid, dtype=bool)
        used[r0 : r0 + ph, c0 : c0 + pw] = mask
        occupied |= _dilate8(used)
    return grid


def _valid_count_profile(owner: np.ndarray) -> bool:
    counts = sorted(len(v) for v in _neighbors_from_owner(owner).values())
    if len(set(counts)) < 2:
        return False
    return max(counts) >= 2


def _gen_attempt() -> np.ndarray | None:
    owner0 = _make_owner()
    owner = _apply_tabs(owner0)
    if owner is None or not _valid_count_profile(owner):
        return None

    neighbors = _neighbors_from_owner(owner)
    max_count = max(len(v) for v in neighbors.values())
    piece_count = len(neighbors)

    if piece_count + max_count > 9:
        return None

    palette = list(range(1, 10))
    legend_colors = random.sample(palette, max_count)
    remaining = [c for c in palette if c not in legend_colors]
    piece_colors = random.sample(remaining, piece_count)
    patches = _piece_patches(owner, piece_colors)
    grid = _place_scene(patches, legend_colors)
    if grid is None:
        return None

    oracle = _oracle_output(owner, legend_colors)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, oracle):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid sample")
