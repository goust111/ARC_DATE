import random
from collections import Counter, deque

import numpy as np


BLACK = 0
QUADRANT_ORDER = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _quadrant_of(r: int, c: int, mid: int) -> int:
    if r < mid and c < mid:
        return 0
    if r < mid and c >= mid:
        return 1
    if r >= mid and c < mid:
        return 2
    return 3


def _local_coord(q: int, r: int, c: int, n: int) -> tuple[int, int]:
    if q == 0:
        return int(r), int(c)
    if q == 1:
        return int(r), int(n - 1 - c)
    if q == 2:
        return int(n - 1 - r), int(c)
    return int(n - 1 - r), int(n - 1 - c)


def _orbit_positions(n: int, lr: int, lc: int) -> list[tuple[int, int]]:
    return [
        (int(lr), int(lc)),
        (int(lr), int(n - 1 - lc)),
        (int(n - 1 - lr), int(lc)),
        (int(n - 1 - lr), int(n - 1 - lc)),
    ]


def _global_bbox_from_local(
    n: int, q: int, lr0: int, lc0: int, h: int, w: int
) -> tuple[int, int, int, int]:
    if q == 0:
        return int(lr0), int(lc0), int(lr0 + h - 1), int(lc0 + w - 1)
    if q == 1:
        return int(lr0), int(n - lc0 - w), int(lr0 + h - 1), int(n - 1 - lc0)
    if q == 2:
        return int(n - lr0 - h), int(lc0), int(n - 1 - lr0), int(lc0 + w - 1)
    return int(n - lr0 - h), int(n - lc0 - w), int(n - 1 - lr0), int(n - 1 - lc0)


def _find_missing_rectangle(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    n = grid.shape[0]
    mid = n // 2
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for coords in _component_coords(grid == BLACK):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        if area != coords.shape[0]:
            continue
        q0 = _quadrant_of(box[0], box[1], mid)
        q1 = _quadrant_of(box[2], box[3], mid)
        if q0 != q1:
            continue
        candidates.append((box, q0))
    assert len(candidates) == 1, "expected exactly one black missing rectangle"
    return candidates[0]


def _inside_bbox(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _learn_pair_maps(
    grid: np.ndarray, bg: int, missing_box: tuple[int, int, int, int]
) -> dict[tuple[int, int], dict[int, int]]:
    n = grid.shape[0]
    mid = n // 2
    pair_maps = {(a, b): {} for a in range(4) for b in range(4) if a != b}

    for lr in range(mid):
        for lc in range(mid):
            positions = _orbit_positions(n, lr, lc)
            if any(_inside_bbox(r, c, missing_box) for r, c in positions):
                continue
            colors = [int(grid[r, c]) for r, c in positions]
            for qa in range(4):
                for qb in range(4):
                    if qa == qb:
                        continue
                    ca = colors[qa]
                    cb = colors[qb]
                    if ca == bg or cb == bg:
                        continue
                    known = pair_maps[(qa, qb)].get(ca)
                    if known is None:
                        pair_maps[(qa, qb)][ca] = cb
                    else:
                        assert known == cb, "inconsistent quadrant color permutation"

    return pair_maps


def _translate_color(
    color: int,
    src_q: int,
    dst_q: int,
    bg: int,
    pair_maps: dict[tuple[int, int], dict[int, int]],
) -> int:
    color = int(color)
    if color == bg or src_q == dst_q:
        return color

    queue = deque([(int(src_q), color)])
    seen = {(int(src_q), color)}
    while queue:
        q, cur = queue.popleft()
        if q == dst_q:
            return int(cur)
        for nq in range(4):
            if nq == q:
                continue
            nxt = pair_maps[(q, nq)].get(cur)
            if nxt is None:
                continue
            state = (nq, int(nxt))
            if state not in seen:
                seen.add(state)
                queue.append(state)
    raise AssertionError("failed to translate color into the missing quadrant")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the blacked-out rectangle by learning quadrant color permutations from intact symmetric orbits.

    Input structure:
    - The grid is a square whose non-background shape is mirror-symmetric across both the vertical and horizontal midlines.
    - Each quadrant uses its own color permutation on the same symmetric orbit structure, so matching positions need not share a color.
    - One solid black rectangle lies wholly inside a single quadrant and overwrites the original cells there.

    Rule:
    - Locate the unique black rectangle and identify its quadrant.
    - From every symmetric orbit that does not touch the missing rectangle, learn the fixed color correspondences between quadrants.
    - For each missing cell, take any intact mirrored counterpart from the same orbit, convert its color through the learned quadrant mapping, and output the recovered rectangle crop only.

    Key insight:
    - Shape symmetry alone is insufficient because colors are permuted by quadrant; the missing value is a translated symmetric color, not a direct copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    n = int(g.shape[0])
    assert n % 2 == 0, "expected an even side length"

    bg = _get_bg(g)
    missing_box, missing_q = _find_missing_rectangle(g)
    pair_maps = _learn_pair_maps(g, bg, missing_box)

    r0, c0, r1, c1 = missing_box
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            lr, lc = _local_coord(missing_q, r, c, n)
            orbit = _orbit_positions(n, lr, lc)
            fill = None
            for src_q in range(4):
                if src_q == missing_q:
                    continue
                sr, sc = orbit[src_q]
                src_color = int(g[sr, sc])
                if src_color == BLACK and src_color != bg:
                    continue
                fill = _translate_color(src_color, src_q, missing_q, bg, pair_maps)
                break
            assert fill is not None, "expected an intact symmetric source for every missing cell"
            out[r - r0, c - c0] = int(fill)

    return out


def _can_place(local: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    qh, qw = local.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(qh, r0 + h + 1)
    cc1 = min(qw, c0 + w + 1)
    return np.all(local[rr0:rr1, cc0:cc1] < 0)


def _place_block(local: np.ndarray, color_idx: int) -> bool:
    qh, qw = local.shape
    height = random.randint(1, min(4, qh - 1))
    width = random.randint(1, min(4, qw - 1))
    for _ in range(80):
        r0 = random.randint(0, qh - height)
        c0 = random.randint(0, qw - width)
        if not _can_place(local, r0, c0, height, width):
            continue
        local[r0 : r0 + height, c0 : c0 + width] = int(color_idx)
        return True
    return False


def _build_local_template(qsize: int, n_colors: int) -> np.ndarray | None:
    local = np.full((qsize, qsize), -1, dtype=int)
    min_reps = 2
    max_reps = 4
    for color_idx in range(n_colors):
        placed = 0
        target = random.randint(min_reps, max_reps)
        for _ in range(target):
            if _place_block(local, color_idx):
                placed += 1
        if placed == 0:
            return None

    extras = random.randint(2, 5)
    for _ in range(extras):
        _place_block(local, random.randrange(n_colors))

    filled = int(np.count_nonzero(local >= 0))
    total = qsize * qsize
    if filled < max(8, total // 6) or filled > (total * 3) // 5:
        return None

    counts = Counter(map(int, local[local >= 0].tolist()))
    if len(counts) != n_colors or min(counts.values()) < 2:
        return None
    return local


def _pick_quadrant_permutations(palette: list[int]) -> list[list[int]]:
    perms = [palette.copy()]
    for _ in range(3):
        perm = palette.copy()
        random.shuffle(perm)
        perms.append(perm)
    if all(perm == perms[0] for perm in perms[1:]):
        random.shuffle(perms[1])
    return perms


def _render_full_grid(local: np.ndarray, bg: int, perms: list[list[int]]) -> np.ndarray:
    qsize = local.shape[0]
    n = qsize * 2
    grid = np.full((n, n), int(bg), dtype=int)
    for lr in range(qsize):
        for lc in range(qsize):
            color_idx = int(local[lr, lc])
            if color_idx < 0:
                continue
            orbit = _orbit_positions(n, lr, lc)
            for q in range(4):
                r, c = orbit[q]
                grid[r, c] = int(perms[q][color_idx])
    return grid


def _candidate_patch_boxes(local: np.ndarray) -> list[tuple[int, int, int, int]]:
    qsize = local.shape[0]
    boxes = []
    r_starts = range(1, qsize - 2) if qsize >= 6 else range(0, qsize - 1)
    c_starts = range(1, qsize - 2) if qsize >= 6 else range(0, qsize - 1)
    for r0 in r_starts:
        for c0 in c_starts:
            max_h = min(5, qsize - r0 - (1 if qsize >= 6 else 0))
            max_w = min(5, qsize - c0 - (1 if qsize >= 6 else 0))
            for h in range(2, max_h + 1):
                for w in range(2, max_w + 1):
                    patch = local[r0 : r0 + h, c0 : c0 + w]
                    fg = patch[patch >= 0]
                    if fg.size < 2:
                        continue
                    colors = set(map(int, fg.tolist()))
                    if any(np.count_nonzero(local == col) == np.count_nonzero(patch == col) for col in colors):
                        continue
                    if len(colors) < 2 and not np.any(patch < 0):
                        continue
                    boxes.append((int(r0), int(c0), int(h), int(w)))
    return boxes


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(list(range(12, 25, 2)))
    qsize = n // 2

    bg = random.choice([c for c in range(1, 10) if c != BLACK])
    available = [c for c in range(1, 10) if c != bg]
    n_colors = random.randint(3, min(5, len(available)))
    palette = random.sample(available, n_colors)

    local = _build_local_template(qsize, n_colors)
    if local is None:
        return None

    boxes = _candidate_patch_boxes(local)
    if not boxes:
        return None

    perms = _pick_quadrant_permutations(palette)
    if not any(perms[q] != perms[0] for q in range(1, 4)):
        return None

    full = _render_full_grid(local, bg, perms)

    missing_q = random.randrange(4)
    lr0, lc0, h, w = random.choice(boxes)
    gr0, gc0, gr1, gc1 = _global_bbox_from_local(n, missing_q, lr0, lc0, h, w)
    truth = full[gr0 : gr1 + 1, gc0 : gc1 + 1].copy()

    inp = full.copy()
    inp[gr0 : gr1 + 1, gc0 : gc1 + 1] = BLACK

    if np.count_nonzero(inp == BLACK) != truth.size:
        return None
    if np.array_equal(truth, inp[gr0 : gr1 + 1, gc0 : gc1 + 1]):
        return None

    try:
        pred = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(pred, truth):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
