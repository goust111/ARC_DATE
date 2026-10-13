import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
PIECE_COLORS = [1, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 33)
PAIR_COUNTS = [2, 3, 4]
INTERIOR_SIDE_RANGE = (3, 7)
GEN_RETRIES = 500
PATCH_RETRIES = 200


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
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    mask = patch != BLACK
    region[mask] = patch[mask]


def _find_containers(grid: np.ndarray) -> list[dict]:
    red_lookup: dict[tuple[int, int], int] = {}
    red_sizes: dict[int, int] = {}
    for idx, comp in enumerate(_components(grid == RED)):
        red_sizes[idx] = len(comp)
        for cell in comp:
            red_lookup[cell] = idx

    containers = []
    for comp in _components(grid == GREEN):
        r0, c0, r1, c1 = _bbox(comp)
        gap_cells = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r not in (r0, r1) and c not in (c0, c1):
                    continue
                if int(grid[r, c]) != GREEN:
                    gap_cells.append((r, c))
        assert len(gap_cells) == 1
        gr, gc = gap_cells[0]
        if gr == r0:
            anchor = (r0 - 1, gc)
        elif gr == r1:
            anchor = (r1 + 1, gc)
        elif gc == c0:
            anchor = (gr, c0 - 1)
        else:
            anchor = (gr, c1 + 1)
        assert anchor in red_lookup
        containers.append(
            {
                "outer_bbox": (r0, c0, r1, c1),
                "interior_top_left": (r0 + 1, c0 + 1),
                "interior_shape": (r1 - r0 - 1, c1 - c0 - 1),
                "lock": int(red_sizes[red_lookup[anchor]]),
            }
        )
    return containers


def _find_pieces(grid: np.ndarray) -> list[dict]:
    piece_mask = (grid != BLACK) & (grid != RED) & (grid != GREEN)
    pieces = []
    for comp in _components(piece_mask):
        patch, bbox = _crop_patch(grid, comp)
        pieces.append({"bbox": bbox, "patch": patch, "shape": tuple(map(int, patch.shape))})
    return pieces


def _match_pieces_to_containers(pieces: list[dict], containers: list[dict]) -> dict[int, int]:
    candidates: dict[int, list[int]] = {}
    for pi, piece in enumerate(pieces):
        ph, pw = piece["shape"]
        cand = []
        for ci, container in enumerate(containers):
            ih, iw = container["interior_shape"]
            if (ph, pw) == (ih, iw) or (ph, pw) == (iw, ih):
                cand.append(ci)
        assert cand
        candidates[pi] = cand

    order = sorted(candidates, key=lambda i: (len(candidates[i]), pieces[i]["shape"], pieces[i]["bbox"]))
    used: set[int] = set()
    assignment: dict[int, int] = {}

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        pi = order[k]
        for ci in candidates[pi]:
            if ci in used:
                continue
            used.add(ci)
            assignment[pi] = ci
            if dfs(k + 1):
                return True
            used.remove(ci)
            del assignment[pi]
        return False

    assert dfs(0)
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    containers = _find_containers(grid)
    pieces = _find_pieces(grid)
    assert len(containers) == len(pieces)

    assignment = _match_pieces_to_containers(pieces, containers)
    for piece_idx, container_idx in assignment.items():
        piece = pieces[piece_idx]
        container = containers[container_idx]
        lock = int(container["lock"])
        rotated = np.rot90(piece["patch"], -lock)
        ih, iw = container["interior_shape"]
        assert rotated.shape == (ih, iw)

        pr0, pc0, pr1, pc1 = piece["bbox"]
        region = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        region[piece["patch"] != BLACK] = BLACK

        ir0, ic0 = container["interior_top_left"]
        _paste_nonzero(out, rotated, ir0, ic0)
    return out


def _choose_locks(count: int) -> list[int]:
    if count == 2:
        locks = [2, random.choice([1, 3])]
    elif count == 3:
        locks = [1, 2, 3]
    else:
        locks = [1, 2, 3, random.choice([1, 2, 3])]
    random.shuffle(locks)
    return locks


def _container_size_pool() -> list[tuple[int, int]]:
    pairs = []
    lo, hi = INTERIOR_SIDE_RANGE
    for a in range(lo, hi):
        for b in range(a + 1, hi):
            pairs.append((a, b))
    return pairs


def _connect_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    if random.random() < 0.5:
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
    else:
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
    return path


def _boundary_anchors(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]


def _piece_signature(arr: np.ndarray) -> bytes:
    return arr.tobytes() + str(arr.shape).encode()


def _generate_piece_patch(h: int, w: int) -> np.ndarray:
    area = h * w
    target_min = max(h + w - 2, (area * 2 + 4) // 5)
    target_max = max(target_min, area - 1)

    for _ in range(PATCH_RETRIES):
        target = random.randint(target_min, target_max)
        cells = set(_boundary_anchors(h, w))
        anchors = list(cells)
        random.shuffle(anchors)
        cur = anchors[0]
        for nxt in anchors[1:]:
            for cell in _connect_path(cur, nxt):
                cells.add(cell)
            cur = nxt

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) < target_min:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if len(_components(mask)) != 1:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        color_count = 2 if len(cells) < 7 else random.choice([2, 3])
        palette = random.sample(PIECE_COLORS, color_count)
        patch = np.zeros((h, w), dtype=int)
        for r, c in sorted(cells):
            patch[r, c] = random.choice(palette)
        used_colors = {int(v) for v in patch[patch != BLACK]}
        if len(used_colors) < 2:
            continue
        if np.array_equal(patch, np.rot90(patch, 2)):
            continue
        sigs = {_piece_signature(np.rot90(patch, k)) for k in range(4)}
        if len(sigs) < 3:
            continue
        return patch
    raise RuntimeError("failed to generate a non-degenerate piece patch")


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _box_free(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    hh, ww = blocked.shape
    if r0 - pad < 0 or c0 - pad < 0 or r1 + pad >= hh or c1 + pad >= ww:
        return False
    return not bool(blocked[r0 - pad : r1 + pad + 1, c0 - pad : c1 + pad + 1].any())


def _container_layout(
    r0: int,
    c0: int,
    interior_h: int,
    interior_w: int,
    side: str,
    gap_index: int,
    lock: int,
) -> dict:
    outer_h = interior_h + 2
    outer_w = interior_w + 2
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    if side == "top":
        gap = (r0, c0 + gap_index)
        marker_cells = [(r0 - k, c0 + gap_index) for k in range(1, lock + 1)]
    elif side == "bottom":
        gap = (r1, c0 + gap_index)
        marker_cells = [(r1 + k, c0 + gap_index) for k in range(1, lock + 1)]
    elif side == "left":
        gap = (r0 + gap_index, c0)
        marker_cells = [(r0 + gap_index, c0 - k) for k in range(1, lock + 1)]
    else:
        gap = (r0 + gap_index, c1)
        marker_cells = [(r0 + gap_index, c1 + k) for k in range(1, lock + 1)]

    rr0 = min([r0] + [r for r, _ in marker_cells])
    cc0 = min([c0] + [c for _, c in marker_cells])
    rr1 = max([r1] + [r for r, _ in marker_cells])
    cc1 = max([c1] + [c for _, c in marker_cells])

    return {
        "outer_bbox": (r0, c0, r1, c1),
        "interior_top_left": (r0 + 1, c0 + 1),
        "interior_shape": (interior_h, interior_w),
        "lock": lock,
        "side": side,
        "gap_index": gap_index,
        "gap": gap,
        "marker_cells": marker_cells,
        "reserve_bbox": (rr0, cc0, rr1, cc1),
    }


def _render_container(grid: np.ndarray, container: dict) -> None:
    r0, c0, r1, c1 = container["outer_bbox"]
    for r in range(r0, r1 + 1):
        grid[r, c0] = GREEN
        grid[r, c1] = GREEN
    for c in range(c0, c1 + 1):
        grid[r0, c] = GREEN
        grid[r1, c] = GREEN
    gr, gc = container["gap"]
    grid[gr, gc] = BLACK
    for r, c in container["marker_cells"]:
        grid[r, c] = RED


def _place_container(grid: np.ndarray, blocked: np.ndarray, interior_h: int, interior_w: int, lock: int) -> dict | None:
    for _ in range(30):
        side = random.choice(["top", "bottom", "left", "right"])
        gap_index = random.randint(1, interior_w if side in ("top", "bottom") else interior_h)
        candidates = []
        for r0 in range(grid.shape[0] - (interior_h + 2) + 1):
            for c0 in range(grid.shape[1] - (interior_w + 2) + 1):
                container = _container_layout(r0, c0, interior_h, interior_w, side, gap_index, lock)
                rr0, cc0, rr1, cc1 = container["reserve_bbox"]
                if _box_free(blocked, rr0, cc0, rr1, cc1, pad=1):
                    candidates.append(container)
        if candidates:
            container = random.choice(candidates)
            _render_container(grid, container)
            rr0, cc0, rr1, cc1 = container["reserve_bbox"]
            _mark_blocked(blocked, rr0, cc0, rr1, cc1, pad=1)
            return container
    return None


def _place_piece(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates = []
    for r0 in range(grid.shape[0] - h + 1):
        for c0 in range(grid.shape[1] - w + 1):
            if _box_free(blocked, r0, c0, r0 + h - 1, c0 + w - 1, pad=1):
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    _paste_nonzero(grid, patch, r0, c0)
    _mark_blocked(blocked, r0, c0, r0 + h - 1, c0 + w - 1, pad=1)
    return r0, c0


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    pair_count = random.choice(PAIR_COUNTS)
    size_pool = random.sample(_container_size_pool(), pair_count)
    locks = _choose_locks(pair_count)

    final_patches: list[np.ndarray] = []
    containers: list[dict] = []
    for (a, b), lock in zip(size_pool, locks):
        interior_h, interior_w = (a, b) if random.random() < 0.5 else (b, a)
        container = _place_container(grid, blocked, interior_h, interior_w, lock)
        if container is None:
            return None
        final_patches.append(_generate_piece_patch(interior_h, interior_w))
        containers.append(container)

    expected = np.zeros_like(grid)
    for container, final_patch in zip(containers, final_patches):
        _render_container(expected, container)
        ir0, ic0 = container["interior_top_left"]
        _paste_nonzero(expected, final_patch, ir0, ic0)

    for container, final_patch in zip(containers, final_patches):
        input_patch = np.rot90(final_patch, container["lock"])
        placed = _place_piece(grid, blocked, input_patch)
        if placed is None:
            return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
