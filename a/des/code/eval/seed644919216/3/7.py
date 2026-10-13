from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


AZURE = 8
BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7]

H_ONLY = (True, False)
V_ONLY = (False, True)
BOTH = (True, True)
NONE = (False, False)
SYM_TYPES = [H_ONLY, V_ONLY, BOTH, NONE]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    pts = np.asarray(points, dtype=int)
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def _split_spans(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for cut in sorted(cuts):
        if start < cut:
            spans.append((start, cut))
        start = cut + 1
    if start < length:
        spans.append((start, length))
    return spans


def _symmetry_type(mask: np.ndarray) -> tuple[bool, bool]:
    return bool(np.array_equal(mask, np.flipud(mask))), bool(np.array_equal(mask, np.fliplr(mask)))


def _warehouse_symbol_type(mask: np.ndarray) -> tuple[bool, bool]:
    h, w = mask.shape
    count = int(mask.sum())
    if count == 1:
        return NONE
    if h == 1 and w == 3 and bool(mask.all()):
        return H_ONLY
    if h == 3 and w == 1 and bool(mask.all()):
        return V_ONLY
    if (h, w) == (3, 3) and count == 5:
        plus = np.zeros((3, 3), dtype=bool)
        plus[1, :] = True
        plus[:, 1] = True
        if np.array_equal(mask, plus):
            return BOTH
    raise AssertionError("unrecognized warehouse symbol")


def _is_frame_component(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h != w or h < 3:
        return False
    if not (mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return False
    return not bool(mask[1:-1, 1:-1].any())


def _rect_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return int(dr + dc)


def _warehouse_mapping(grid: np.ndarray, building_bbox: tuple[int, int, int, int]) -> dict[tuple[bool, bool], np.ndarray]:
    _, _, _, build_c1 = building_bbox
    maroon = grid == MAROON
    maroon[:, : build_c1 + 1] = False
    comps = _components(maroon)

    frames: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    symbols: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        box = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for rr, cc in pts:
            box[rr - r0, cc - c0] = True
        if _is_frame_component(box):
            frames.append(((r0, c0, r1, c1), box))
        else:
            symbols.append(((r0, c0, r1, c1), box))

    assert len(frames) == 4
    assert len(symbols) == 4

    best_perm = None
    best_cost = None
    for perm in itertools.permutations(range(4)):
        cost = sum(_rect_distance(frames[i][0], symbols[perm[i]][0]) for i in range(4))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    assert best_perm is not None

    mapping: dict[tuple[bool, bool], np.ndarray] = {}
    inner_sizes = set()
    for frame_idx, sym_idx in enumerate(best_perm):
        (r0, c0, r1, c1), _ = frames[frame_idx]
        sym_type = _warehouse_symbol_type(symbols[sym_idx][1])
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(v) for v in np.unique(inner) if int(v) != BLACK]
        assert len(colors) == 1
        mapping[sym_type] = inner == colors[0]
        inner_sizes.add(inner.shape)

    assert set(mapping) == set(SYM_TYPES)
    assert len(inner_sizes) == 1
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each window's motif with the warehouse decoration selected by the motif's reflection symmetries.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    building_bbox = _bbox(np.argwhere(g == GREY))
    br0, bc0, br1, bc1 = building_bbox

    warehouse = _warehouse_mapping(g, building_bbox)

    inner = g[br0 + 1 : br1, bc0 + 1 : bc1]
    sep_rows = [r for r in range(inner.shape[0]) if bool(np.all(inner[r, :] == MAROON))]
    sep_cols = [c for c in range(inner.shape[1]) if bool(np.all(inner[:, c] == MAROON))]
    row_spans = _split_spans(inner.shape[0], sep_rows)
    col_spans = _split_spans(inner.shape[1], sep_cols)

    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1
    cell_size = next(iter(heights))
    assert widths == {cell_size}

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            window = inner[r0:r1, c0:c1]
            colors = [int(v) for v in np.unique(window) if int(v) != BLACK]
            assert len(colors) == 1
            color = colors[0]
            motif = window == color
            sym_type = _symmetry_type(motif)
            deco_mask = warehouse[sym_type]
            assert deco_mask.shape == (cell_size, cell_size)
            replacement = np.where(deco_mask, color, BLACK)
            out[br0 + 1 + r0 : br0 + 1 + r1, bc0 + 1 + c0 : bc0 + 1 + c1] = replacement

    return out


def _orbit(r: int, c: int, size: int, sym_type: tuple[bool, bool]) -> set[tuple[int, int]]:
    has_h, has_v = sym_type
    orbit = {(r, c)}
    changed = True
    while changed:
        changed = False
        snapshot = list(orbit)
        for rr, cc in snapshot:
            candidates = [(rr, cc)]
            if has_h:
                candidates.append((size - 1 - rr, cc))
            if has_v:
                candidates.append((rr, size - 1 - cc))
            if has_h and has_v:
                candidates.append((size - 1 - rr, size - 1 - cc))
            for pt in candidates:
                if pt not in orbit:
                    orbit.add(pt)
                    changed = True
    return orbit


def _visible_mask(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) < 3 or len(pts) > mask.size - 3:
        return False
    r0, c0, r1, c1 = _bbox(pts)
    return (r1 - r0 + 1) >= 2 and (c1 - c0 + 1) >= 2


def _make_exact_mask(size: int, sym_type: tuple[bool, bool], used: list[np.ndarray]) -> np.ndarray:
    for _ in range(400):
        p = random.uniform(0.28, 0.56)
        mask = np.zeros((size, size), dtype=bool)
        seen: set[tuple[int, int]] = set()
        for r in range(size):
            for c in range(size):
                if (r, c) in seen:
                    continue
                orb = _orbit(r, c, size, sym_type)
                seen.update(orb)
                value = random.random() < p
                for rr, cc in orb:
                    mask[rr, cc] = value
        if not _visible_mask(mask):
            continue
        if _symmetry_type(mask) != sym_type:
            continue
        if any(np.array_equal(mask, other) for other in used):
            continue
        return mask
    raise RuntimeError("failed to build exact symmetry mask")


def _make_symbol_patch(sym_type: tuple[bool, bool]) -> np.ndarray:
    if sym_type == NONE:
        return np.array([[True]], dtype=bool)
    if sym_type == H_ONLY:
        return np.ones((1, 3), dtype=bool)
    if sym_type == V_ONLY:
        return np.ones((3, 1), dtype=bool)
    patch = np.zeros((3, 3), dtype=bool)
    patch[1, :] = True
    patch[:, 1] = True
    return patch


def _window_type_list(count: int) -> list[tuple[bool, bool]]:
    seq: list[tuple[bool, bool]] = []
    while len(seq) < count:
        batch = SYM_TYPES[:]
        random.shuffle(batch)
        seq.extend(batch)
    seq = seq[:count]
    random.shuffle(seq)
    return seq


def _draw_building(
    grid: np.ndarray,
    top: int,
    left: int,
    rows: int,
    cols: int,
    cell_size: int,
    deco_map: dict[tuple[bool, bool], np.ndarray],
) -> tuple[list[tuple[int, int, tuple[bool, bool], int]], tuple[int, int, int, int]]:
    facade_h = 2 + rows * cell_size + (rows - 1)
    facade_w = 2 + cols * cell_size + (cols - 1)
    grid[top : top + facade_h, left : left + facade_w] = GREY

    inner_top = top + 1
    inner_left = left + 1
    inner_h = rows * cell_size + (rows - 1)
    inner_w = cols * cell_size + (cols - 1)

    for ridx in range(1, rows):
        rr = inner_top + ridx * cell_size + (ridx - 1)
        grid[rr, inner_left : inner_left + inner_w] = MAROON
    for cidx in range(1, cols):
        cc = inner_left + cidx * cell_size + (cidx - 1)
        grid[inner_top : inner_top + inner_h, cc] = MAROON

    window_types = _window_type_list(rows * cols)
    windows: list[tuple[int, int, tuple[bool, bool], int]] = []
    used_masks: dict[tuple[bool, bool], list[np.ndarray]] = {sym: [] for sym in SYM_TYPES}

    idx = 0
    for r in range(rows):
        for c in range(cols):
            wr = inner_top + r * (cell_size + 1)
            wc = inner_left + c * (cell_size + 1)
            sym_type = window_types[idx]
            idx += 1
            color = random.choice(COLOR_POOL)
            for _ in range(200):
                mask = _make_exact_mask(cell_size, sym_type, used_masks[sym_type])
                if not np.array_equal(mask, deco_map[sym_type]):
                    break
            else:
                raise RuntimeError("failed to build changing window mask")
            used_masks[sym_type].append(mask)
            grid[wr : wr + cell_size, wc : wc + cell_size] = BLACK
            grid[wr : wr + cell_size, wc : wc + cell_size][mask] = color
            windows.append((wr, wc, sym_type, color))

    return windows, (top, left, top + facade_h - 1, left + facade_w - 1)


def _draw_warehouse(
    grid: np.ndarray,
    top: int,
    left: int,
    cell_size: int,
    row_gap: int,
    col_gap: int,
) -> dict[tuple[bool, bool], np.ndarray]:
    box_outer = cell_size + 2
    slot_h = box_outer + 10
    slot_w = box_outer + 10

    slot_origins = [
        (top, left),
        (top, left + slot_w + col_gap),
        (top + slot_h + row_gap, left),
        (top + slot_h + row_gap, left + slot_w + col_gap),
    ]
    sides = ["up", "down", "left", "right"]

    deco_symmetries = SYM_TYPES[:]
    while True:
        random.shuffle(deco_symmetries)
        if deco_symmetries != SYM_TYPES:
            break

    deco_masks = {}
    used_masks: list[np.ndarray] = []
    deco_colors = random.sample(COLOR_POOL, 4)
    for sym_type, deco_sym in zip(SYM_TYPES, deco_symmetries, strict=False):
        deco_masks[sym_type] = _make_exact_mask(cell_size, deco_sym, used_masks)
        used_masks.append(deco_masks[sym_type])

    for slot_idx, sym_type in enumerate(random.sample(SYM_TYPES, len(SYM_TYPES))):
        slot_r0, slot_c0 = slot_origins[slot_idx]
        frame_r0 = slot_r0 + random.randint(4, 6)
        frame_c0 = slot_c0 + random.randint(4, 6)
        frame_r1 = frame_r0 + box_outer - 1
        frame_c1 = frame_c0 + box_outer - 1

        grid[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = AZURE
        grid[frame_r0, frame_c0 : frame_c1 + 1] = MAROON
        grid[frame_r1, frame_c0 : frame_c1 + 1] = MAROON
        grid[frame_r0 : frame_r1 + 1, frame_c0] = MAROON
        grid[frame_r0 : frame_r1 + 1, frame_c1] = MAROON

        inner = deco_masks[sym_type]
        color = deco_colors[slot_idx]
        grid[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1] = BLACK
        grid[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1][inner] = color

        symbol = _make_symbol_patch(sym_type)
        sh, sw = symbol.shape
        side = random.choice(sides)
        if side == "up":
            sr0 = frame_r0 - 1 - sh
            sc0 = frame_c0 + (box_outer - sw) // 2
        elif side == "down":
            sr0 = frame_r1 + 2
            sc0 = frame_c0 + (box_outer - sw) // 2
        elif side == "left":
            sr0 = frame_r0 + (box_outer - sh) // 2
            sc0 = frame_c0 - 1 - sw
        else:
            sr0 = frame_r0 + (box_outer - sh) // 2
            sc0 = frame_c1 + 2
        grid[sr0 : sr0 + sh, sc0 : sc0 + sw][symbol] = MAROON

    return deco_masks


def _expected_output(
    grid: np.ndarray,
    windows: list[tuple[int, int, tuple[bool, bool], int]],
    deco_map: dict[tuple[bool, bool], np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for wr, wc, sym_type, color in windows:
        mask = deco_map[sym_type]
        out[wr : wr + mask.shape[0], wc : wc + mask.shape[1]] = np.where(mask, color, BLACK)
    return out


def _generate_attempt() -> np.ndarray | None:
    cell_size = random.randint(5, 7)
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    if rows * cols < 4:
        return None

    facade_h = 2 + rows * cell_size + (rows - 1)
    facade_w = 2 + cols * cell_size + (cols - 1)
    box_outer = cell_size + 2
    row_gap = random.randint(2, 5)
    col_gap = random.randint(2, 5)
    warehouse_h = 2 * (box_outer + 10) + row_gap
    warehouse_w = 2 * (box_outer + 10) + col_gap

    gap = random.randint(3, 6)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    height = max(facade_h, warehouse_h) + random.randint(2, 6)
    width = left_margin + facade_w + gap + warehouse_w + right_margin
    grid = np.full((height, width), AZURE, dtype=int)

    build_top = random.randint(1, height - facade_h - 1)
    build_left = left_margin
    warehouse_top = random.randint(1, height - warehouse_h - 1)
    warehouse_left = build_left + facade_w + gap

    deco_map = _draw_warehouse(grid, warehouse_top, warehouse_left, cell_size, row_gap, col_gap)
    windows, _ = _draw_building(grid, build_top, build_left, rows, cols, cell_size, deco_map)
    expected = _expected_output(grid, windows, deco_map)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
