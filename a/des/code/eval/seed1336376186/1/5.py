import random
from typing import Iterable

import numpy as np


BG = 0
PIPE = 5
TARGET_FRAME = 7
SOURCE_FRAME = 8
PATTERN_COLORS = (1, 2, 3, 4, 6, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEFT_TURN = {
    (-1, 0): (0, -1),
    (0, -1): (1, 0),
    (1, 0): (0, 1),
    (0, 1): (-1, 0),
}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}


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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _rect_area_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _find_frames(grid: np.ndarray, color: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    frames: list[dict] = []
    lookup: dict[tuple[int, int], int] = {}
    for comp in _components(grid == color):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = _rect_border_cells(r0, c0, r1, c1)
        if set(comp) != border:
            continue
        frame_id = len(frames)
        frames.append({"bbox": (r0, c0, r1, c1)})
        for cell in border:
            lookup[cell] = frame_id
    return frames, lookup


def _count_bends(pipe_cells: Iterable[tuple[int, int]]) -> int:
    cells = set(pipe_cells)
    bends = 0
    for r, c in cells:
        neigh = [(dr, dc) for dr, dc in DIR4 if (r + dr, c + dc) in cells]
        if len(neigh) == 2 and neigh[0][0] != neigh[1][0] and neigh[0][1] != neigh[1][1]:
            bends += 1
    return bends


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each source-frame interior into its connected target frame after rotating
    clockwise once per pipe bend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    sources, source_lookup = _find_frames(g, SOURCE_FRAME)
    targets, target_lookup = _find_frames(g, TARGET_FRAME)

    for pipe in _components(g == PIPE):
        source_ids: set[int] = set()
        target_ids: set[int] = set()
        for r, c in pipe:
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in source_lookup:
                    source_ids.add(source_lookup[nb])
                if nb in target_lookup:
                    target_ids.add(target_lookup[nb])

        assert len(source_ids) == 1 and len(target_ids) == 1
        source = sources[next(iter(source_ids))]
        target = targets[next(iter(target_ids))]

        sr0, sc0, sr1, sc1 = source["bbox"]
        tr0, tc0, tr1, tc1 = target["bbox"]
        src_patch = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        rotated = np.rot90(src_patch, -(_count_bends(pipe) % 4))

        target_patch = out[tr0 + 1 : tr1, tc0 + 1 : tc1]
        th, tw = target_patch.shape
        rh, rw = rotated.shape
        assert rh <= th and rw <= tw
        target_patch[:, :] = BG
        target_patch[:rh, :rw] = rotated

    return out


MIN_INTERIOR = 3
MAX_INTERIOR = 6
LOCAL_ATTEMPTS = 250
PACK_ATTEMPTS = 250
GEN_ATTEMPTS = 300
PAIR_GAP = 2
GRID_EXTRA = 10


def _sample_connected_mask(h: int, w: int) -> np.ndarray | None:
    total = h * w
    min_cells = max(3, total // 4)
    max_cells = min(total - 1, max(min_cells, (2 * total) // 3))
    if min_cells > max_cells:
        return None
    target = random.randint(min_cells, max_cells)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < target and frontier:
        cr, cc = random.choice(tuple(frontier))
        options = []
        for dr, dc in DIR4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                options.append((nr, nc))
        if not options:
            frontier.remove((cr, cc))
            continue
        chosen = random.choice(options)
        cells.add(chosen)
        frontier.add(chosen)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _nontrivial_pattern(mask: np.ndarray) -> bool:
    if mask.sum() == 0 or mask.all():
        return False
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return False
    for k in (1, 2, 3):
        rot = np.rot90(mask, k)
        if rot.shape == mask.shape and np.array_equal(rot, mask):
            return False
    return True


def _sample_pattern(h: int, w: int, color: int) -> np.ndarray | None:
    for _ in range(80):
        mask = _sample_connected_mask(h, w)
        if mask is None or not _nontrivial_pattern(mask):
            continue
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = color
        return patch
    return None


def _turn_sequence(bends: int) -> list[tuple[int, int]]:
    direction = random.choice(DIR4)
    seq = [direction]
    for _ in range(bends):
        direction = LEFT_TURN[direction] if random.random() < 0.5 else RIGHT_TURN[direction]
        seq.append(direction)
    return seq


def _build_path(directions: list[tuple[int, int]], steps: list[int]) -> list[tuple[int, int]]:
    r = c = 0
    cells = [(r, c)]
    for (dr, dc), count in zip(directions, steps):
        for _ in range(count):
            r += dr
            c += dc
            cells.append((r, c))
    return cells


def _is_simple_path(cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    degs = []
    for r, c in cell_set:
        deg = sum((r + dr, c + dc) in cell_set for dr, dc in DIR4)
        if deg not in (1, 2):
            return False
        degs.append(deg)
    return degs.count(1) == 2 and degs.count(2) == len(cells) - 2


def _attached_frame_bbox(endpoint: tuple[int, int], side: tuple[int, int], fh: int, fw: int) -> tuple[int, int, int, int]:
    r, c = endpoint
    if side in ((0, -1), (0, 1)):
        offset = random.randint(1, fh - 2)
        r0 = r - offset
        c0 = c - fw if side == (0, -1) else c + 1
    else:
        offset = random.randint(1, fw - 2)
        c0 = c - offset
        r0 = r - fh if side == (-1, 0) else r + 1
    return (r0, c0, r0 + fh - 1, c0 + fw - 1)


def _adjacent_count(cell: tuple[int, int], other: set[tuple[int, int]]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in other for dr, dc in DIR4)


def _pair_geometry_ok(
    path: list[tuple[int, int]],
    source_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
) -> bool:
    path_set = set(path)
    src_border = _rect_border_cells(*source_bbox)
    dst_border = _rect_border_cells(*target_bbox)
    src_area = _rect_area_cells(*source_bbox)
    dst_area = _rect_area_cells(*target_bbox)
    if path_set & src_area or path_set & dst_area or src_area & dst_area:
        return False
    if any(_adjacent_count(cell, dst_area) for cell in src_area):
        return False
    if _adjacent_count(path[0], src_border) != 1 or _adjacent_count(path[-1], dst_border) != 1:
        return False
    for cell in path[1:]:
        if _adjacent_count(cell, src_area):
            return False
    for cell in path[:-1]:
        if _adjacent_count(cell, dst_area):
            return False
    return True


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _render_pair(
    source_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    path: list[tuple[int, int]],
    source_patch: np.ndarray,
    turns: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = list(path)
    for bbox in (source_bbox, target_bbox):
        r0, c0, r1, c1 = bbox
        points.extend([(r0, c0), (r0, c1), (r1, c0), (r1, c1)])
    min_r = min(r for r, _ in points)
    max_r = max(r for r, _ in points)
    min_c = min(c for _, c in points)
    max_c = max(c for _, c in points)
    pad = 1
    dr = pad - min_r
    dc = pad - min_c
    h = max_r - min_r + 1 + 2 * pad
    w = max_c - min_c + 1 + 2 * pad

    src_box = _shift_bbox(source_bbox, dr, dc)
    dst_box = _shift_bbox(target_bbox, dr, dc)
    shifted_path = [(r + dr, c + dc) for r, c in path]

    inp = np.zeros((h, w), dtype=int)
    _draw_frame(inp, src_box, SOURCE_FRAME)
    _draw_frame(inp, dst_box, TARGET_FRAME)
    for r, c in shifted_path:
        inp[r, c] = PIPE
    sr0, sc0, sr1, sc1 = src_box
    inp[sr0 + 1 : sr1, sc0 + 1 : sc1] = source_patch

    out = inp.copy()
    rotated = np.rot90(source_patch, -turns)
    tr0, tc0, tr1, tc1 = dst_box
    out[tr0 + 1 : tr1, tc0 + 1 : tc1] = rotated
    return inp, out


def _rotate_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randint(0, 3)
    return np.rot90(inp, k), np.rot90(out, k)


def _gen_pair(color: int, bends: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(LOCAL_ATTEMPTS):
        ih = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        iw = random.randint(MIN_INTERIOR, MAX_INTERIOR)
        source_patch = _sample_pattern(ih, iw, color)
        if source_patch is None:
            continue

        directions = _turn_sequence(bends)
        steps = [random.randint(2, 5) for _ in range(bends + 1)]
        path = _build_path(directions, steps)
        if not _is_simple_path(path):
            continue

        target_patch = np.rot90(source_patch, -bends)
        source_bbox = _attached_frame_bbox(path[0], (-directions[0][0], -directions[0][1]), ih + 2, iw + 2)
        target_bbox = _attached_frame_bbox(path[-1], directions[-1], target_patch.shape[0] + 2, target_patch.shape[1] + 2)
        if not _pair_geometry_ok(path, source_bbox, target_bbox):
            continue

        inp, out = _render_pair(source_bbox, target_bbox, path, source_patch, bends)
        inp, out = _rotate_pair(inp, out)
        try:
            if np.array_equal(main(inp), out):
                return inp, out
        except Exception:
            continue
    return None


def _pack_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    patches = [pair[0] for pair in pairs]
    padded_area = sum((p.shape[0] + 2 * PAIR_GAP) * (p.shape[1] + 2 * PAIR_GAP) for p in patches)
    side = int(np.ceil(np.sqrt(padded_area)))
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    min_h = max(max_h + 2 * PAIR_GAP + 2, side)
    min_w = max(max_w + 2 * PAIR_GAP + 2, side)

    for _ in range(PACK_ATTEMPTS):
        h = random.randint(min_h, min_h + GRID_EXTRA)
        w = random.randint(min_w, min_w + GRID_EXTRA)
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        order = list(range(len(pairs)))
        order.sort(key=lambda i: pairs[i][0].shape[0] * pairs[i][0].shape[1], reverse=True)
        ok = True
        for idx in order:
            pinp, pout = pairs[idx]
            ph, pw = pinp.shape
            placed = False
            for _ in range(160):
                r = random.randint(1, h - ph - 1)
                c = random.randint(1, w - pw - 1)
                r0 = max(0, r - PAIR_GAP)
                r1 = min(h, r + ph + PAIR_GAP)
                c0 = max(0, c - PAIR_GAP)
                c1 = min(w, c + pw + PAIR_GAP)
                if blocked[r0:r1, c0:c1].any():
                    continue
                region_in = inp[r : r + ph, c : c + pw]
                region_out = out[r : r + ph, c : c + pw]
                mask_in = pinp != BG
                mask_out = pout != BG
                region_in[mask_in] = pinp[mask_in]
                region_out[mask_out] = pout[mask_out]
                blocked[r0:r1, c0:c1] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return inp, out
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 3)
    bends = random.sample([0, 1, 2, 3], pair_count)
    colors = random.sample(PATTERN_COLORS, pair_count)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for color, bend_count in zip(colors, bends):
        pair = _gen_pair(color, bend_count)
        if pair is None:
            return None
        pairs.append(pair)

    packed = _pack_pairs(pairs)
    if packed is None:
        return None
    inp, expected = packed
    if np.count_nonzero(inp == BG) <= inp.size // 2:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
