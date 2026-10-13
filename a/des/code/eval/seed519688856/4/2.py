from __future__ import annotations

from collections import deque

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mask_is_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(mask.sum()) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _extract_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append({"color": color, "mask": mask, "area": int(mask.sum())})
    return comps


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    rots = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k).copy()
        key = (tuple(int(v) for v in rot.shape), rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        rots.append(rot)
    return rots


def _pair_candidates(comp_a: dict, comp_b: dict) -> list[np.ndarray]:
    mask_a = comp_a["mask"]
    mask_b = comp_b["mask"]
    color_a = int(comp_a["color"])
    color_b = int(comp_b["color"])

    coords_a = [tuple(int(v) for v in rc) for rc in np.argwhere(mask_a)]
    cells_a = set(coords_a)
    ha, wa = (int(v) for v in mask_a.shape)
    candidates = {}

    for rot_b in _unique_rotations(mask_b):
        coords_b = [tuple(int(v) for v in rc) for rc in np.argwhere(rot_b)]
        hb, wb = (int(v) for v in rot_b.shape)
        for dr in range(-hb, ha + 1):
            for dc in range(-wb, wa + 1):
                shifted_b = [(r + dr, c + dc) for r, c in coords_b]
                if any(cell in cells_a for cell in shifted_b):
                    continue

                union = coords_a + shifted_b
                rows = [p[0] for p in union]
                cols = [p[1] for p in union]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                rect_h = r1 - r0 + 1
                rect_w = c1 - c0 + 1
                if rect_h * rect_w != len(union):
                    continue

                block = np.zeros((rect_h, rect_w), dtype=int)
                for r, c in coords_a:
                    block[r - r0, c - c0] = color_a
                for r, c in shifted_b:
                    block[r - r0, c - c0] = color_b
                key = (rect_h, rect_w, block.tobytes())
                candidates[key] = block

    return [
        candidates[key]
        for key in sorted(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


def _stack_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    ordered = sorted(
        blocks,
        key=lambda block: (
            int(block.shape[0] * block.shape[1]),
            int(block.shape[0]),
            int(block.shape[1]),
            block.tobytes(),
        ),
    )
    width = max(int(block.shape[1]) for block in ordered)
    height = sum(int(block.shape[0]) for block in ordered)
    out = np.zeros((height, width), dtype=int)
    cursor = height
    for block in ordered:
        bh, bw = (int(v) for v in block.shape)
        cursor -= bh
        c0 = (width - bw) // 2
        out[cursor : cursor + bh, c0 : c0 + bw] = block
    return out


def _all_outputs(grid: np.ndarray) -> list[np.ndarray]:
    comps = _extract_components(grid, bg=0)
    n = len(comps)
    if n == 0 or n % 2 == 1:
        return []

    pair_map: dict[tuple[int, int], list[np.ndarray]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            candidates = _pair_candidates(comps[i], comps[j])
            if candidates:
                pair_map[(i, j)] = candidates

    outputs = {}

    def backtrack(remaining: tuple[int, ...], blocks: list[np.ndarray]) -> None:
        if not remaining:
            out = _stack_blocks(blocks)
            outputs[(tuple(int(v) for v in out.shape), out.tobytes())] = out
            return
        i = remaining[0]
        for pos in range(1, len(remaining)):
            j = remaining[pos]
            cands = pair_map.get((i, j))
            if not cands:
                continue
            nxt = remaining[1:pos] + remaining[pos + 1 :]
            for block in cands:
                backtrack(nxt, blocks + [block])

    backtrack(tuple(range(n)), [])
    return [
        outputs[key]
        for key in sorted(outputs, key=lambda item: (item[0][0], item[0][1], item[1]))
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary monochrome fragments into filled rectangles, then stack
    the reconstructed rectangles from smallest area at the bottom to largest at
    the top.
    """
    g = np.asarray(grid, dtype=int)
    outputs = _all_outputs(g)
    if not outputs:
        raise ValueError("no valid complementary pairing found")
    return outputs[0]


# --------------- generator ---------------

_RNG = np.random.default_rng()

PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_HEIGHT_RANGE = (2, 6)
RECT_WIDTH_RANGE = (2, 7)
MIN_RECT_AREA = 6
MAX_RECT_AREA = 20
GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 28
PAIR_GEN_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 300
GEN_ATTEMPTS = 300
COLOR_POOL = tuple(range(1, 10))


def _random_connected_subset(h: int, w: int, target_area: int) -> np.ndarray | None:
    start = (int(_RNG.integers(h)), int(_RNG.integers(w)))
    chosen = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))
    while len(chosen) < target_area:
        frontier = {cell for cell in frontier if cell not in chosen}
        if not frontier:
            return None
        options = list(frontier)
        cell = options[int(_RNG.integers(len(options)))]
        chosen.add(cell)
        for nbr in _neighbors4(cell[0], cell[1], h, w):
            if nbr not in chosen:
                frontier.add(nbr)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _generate_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    area = h * w
    lo = 2
    hi = area - 2
    if lo > hi:
        return None

    for _ in range(PAIR_GEN_ATTEMPTS):
        target = int(_RNG.integers(lo, hi + 1))
        mask_a = _random_connected_subset(h, w, target)
        if mask_a is None:
            continue
        mask_b = ~mask_a
        if not _mask_connected(mask_b):
            continue
        if _mask_is_rectangle(mask_a) and _mask_is_rectangle(mask_b):
            continue
        return mask_a, mask_b
    return None


def _relative_rotation(mask: np.ndarray, require_turn: bool) -> int:
    if not require_turn:
        return 0
    options = [k for k in (1, 2, 3) if not np.array_equal(mask, np.rot90(mask, k))]
    if not options:
        return 0
    return int(options[int(_RNG.integers(len(options)))])


def _rectangle_catalog() -> dict[int, list[tuple[int, int]]]:
    by_area: dict[int, list[tuple[int, int]]] = {}
    for h in range(*RECT_HEIGHT_RANGE):
        for w in range(*RECT_WIDTH_RANGE):
            area = h * w
            if area < MIN_RECT_AREA or area > MAX_RECT_AREA:
                continue
            by_area.setdefault(area, []).append((h, w))
    return by_area


_RECTS_BY_AREA = _rectangle_catalog()


def _place_pieces(pieces: list[dict], h: int, w: int) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    order = list(range(len(pieces)))
    order.sort(
        key=lambda idx: (
            int(pieces[idx]["mask"].shape[0] * pieces[idx]["mask"].shape[1]),
            int(pieces[idx]["mask"].shape[0]),
            int(pieces[idx]["mask"].shape[1]),
        ),
        reverse=True,
    )

    for idx in order:
        piece = pieces[idx]
        mask = piece["mask"]
        color = int(piece["color"])
        ph, pw = (int(v) for v in mask.shape)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(_RNG.integers(0, h - ph + 1))
            c0 = int(_RNG.integers(0, w - pw + 1))
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            patch = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any((patch != 0) & mask):
                continue
            patch[mask] = color
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(PAIR_COUNT_CHOICES))
    all_areas = list(_RECTS_BY_AREA)
    chosen_areas = _RNG.choice(all_areas, size=pair_count, replace=False)

    pieces = []
    color_perm = list(_RNG.permutation(COLOR_POOL))
    color_idx = 0

    for area in chosen_areas:
        dims_options = _RECTS_BY_AREA[int(area)]
        h, w = dims_options[int(_RNG.integers(len(dims_options)))]
        partition = _generate_partition(h, w)
        if partition is None:
            return None
        mask_a, mask_b = partition

        turn_branch = bool(_RNG.integers(0, 2))
        base_rot = int(_RNG.integers(4))
        diff = _relative_rotation(mask_b, require_turn=turn_branch)
        rot_a = base_rot
        rot_b = (base_rot + diff) % 4

        pieces.append(
            {
                "mask": np.rot90(mask_a, rot_a).copy(),
                "color": int(color_perm[color_idx]),
            }
        )
        color_idx += 1
        pieces.append(
            {
                "mask": np.rot90(mask_b, rot_b).copy(),
                "color": int(color_perm[color_idx]),
            }
        )
        color_idx += 1

    bbox_budget = sum((piece["mask"].shape[0] + 2) * (piece["mask"].shape[1] + 2) for piece in pieces)
    base_side = int(np.ceil(np.sqrt(bbox_budget)))
    h = int(_RNG.integers(max(GRID_SIDE_MIN, base_side), GRID_SIDE_MAX + 1))
    w = int(_RNG.integers(max(GRID_SIDE_MIN, base_side), GRID_SIDE_MAX + 1))
    grid = _place_pieces(pieces, h, w)
    if grid is None:
        return None

    if len(_extract_components(grid, bg=0)) != len(pieces):
        return None

    outputs = _all_outputs(grid)
    if len(outputs) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid complementary-fragment sample")
