from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
STAMP_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 30)
STAMP_COUNT_RANGE = (2, 4)
INSTR_COUNT_RANGE = (4, 7)
STAMP_SIZE_CHOICES = (3, 5, 7)
MAX_GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _boundary_bg(grid: np.ndarray) -> int:
    boundary = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    return int(Counter(map(int, boundary)).most_common(1)[0][0])


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
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _median_floor(values: np.ndarray) -> int:
    ordered = np.sort(np.asarray(values, dtype=int))
    n = len(ordered)
    return int((int(ordered[(n - 1) // 2]) + int(ordered[n // 2])) // 2)


def _segment_center(coords: np.ndarray) -> tuple[int, int]:
    return _median_floor(coords[:, 0]), _median_floor(coords[:, 1])


def _order_path(coords: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    coord_set = {tuple(map(int, rc)) for rc in coords}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coord_set:
        adj[(r, c)] = [n for n in _neighbors4(r, c, h, w) if n in coord_set]

    assert max(len(nbs) for nbs in adj.values()) <= 2, "path must not branch"
    endpoints = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = min(endpoints, key=lambda rc: (rc[1], rc[0]))

    ordered: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        ordered.append(cur)
        nxt = [cell for cell in adj[cur] if cell != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path ordering must be unique"
        prev, cur = cur, nxt[0]

    assert len(ordered) == len(coord_set), "path must be a single chain"
    return ordered


def _paste_patch(
    out: np.ndarray,
    patch: np.ndarray,
    center: tuple[int, int],
    touched: np.ndarray | None = None,
) -> tuple[bool, bool]:
    h, w = out.shape
    ph, pw = patch.shape
    top = int(center[0]) - ph // 2
    left = int(center[1]) - pw // 2
    bottom = top + ph
    right = left + pw

    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h, bottom)
    c1 = min(w, right)
    assert r0 < r1 and c0 < c1

    src_r0 = r0 - top
    src_c0 = c0 - left
    src_r1 = src_r0 + (r1 - r0)
    src_c1 = src_c0 + (c1 - c0)

    clipped = top < 0 or left < 0 or bottom > h or right > w
    overlapped = False
    if touched is not None:
        overlapped = bool(np.any(touched[r0:r1, c0:c1]))
        touched[r0:r1, c0:c1] = True

    out[r0:r1, c0:c1] = patch[src_r0:src_r1, src_c0:src_c1]
    return clipped, overlapped


def _apply_instructions(
    shape: tuple[int, int],
    instructions: list[tuple[int, np.ndarray]],
    stamps: dict[int, np.ndarray],
    bg: int,
) -> tuple[np.ndarray, bool, bool]:
    out = np.full(shape, int(bg), dtype=int)
    touched = np.zeros(shape, dtype=bool)
    has_clip = False
    has_overlap = False
    for color, coords in instructions:
        clipped, overlapped = _paste_patch(out, stamps[int(color)], _segment_center(coords), touched)
        has_clip |= clipped
        has_overlap |= overlapped
    return out, has_clip, has_overlap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the lower multicolor path as an ordered instruction list and stamp the
    matching upper sprites at each segment center, with later stamps overwriting
    earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _boundary_bg(g)

    infos = []
    for coords in _components(g != bg):
        colors = tuple(sorted(set(map(int, g[coords[:, 0], coords[:, 1]]))))
        infos.append({"coords": coords, "colors": colors, "bbox": _bbox(coords)})

    path_candidates = [info for info in infos if len(info["colors"]) > 1]
    assert len(path_candidates) == 1, "expected exactly one multicolor path component"
    path = path_candidates[0]

    stamps: dict[int, np.ndarray] = {}
    for info in infos:
        if info is path:
            continue
        assert len(info["colors"]) == 1, "stamp components must be monochrome"
        color = int(info["colors"][0])
        assert color not in stamps
        r0, c0, r1, c1 = info["bbox"]
        stamps[color] = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    path_colors = set(map(int, path["colors"]))
    grey_candidates = path_colors - set(stamps)
    assert len(grey_candidates) == 1, "path must contain exactly one separator color"
    grey = int(next(iter(grey_candidates)))

    ordered = _order_path(path["coords"], g.shape)
    ordered_colors = [int(g[r, c]) for r, c in ordered]
    assert ordered_colors[0] != grey and ordered_colors[-1] != grey

    instructions: list[tuple[int, np.ndarray]] = []
    i = 0
    prev_was_grey = False
    while i < len(ordered):
        color = ordered_colors[i]
        j = i + 1
        while j < len(ordered) and ordered_colors[j] == color:
            j += 1
        run = np.asarray(ordered[i:j], dtype=int)
        if color == grey:
            assert not prev_was_grey
            assert 1 <= len(run) <= 2, "grey separators must be length 1 or 2"
            prev_was_grey = True
        else:
            assert color in stamps
            instructions.append((int(color), run))
            prev_was_grey = False
        i = j

    assert instructions, "path must contain at least one instruction segment"
    out, _has_clip, _has_overlap = _apply_instructions(g.shape, instructions, stamps, bg)
    return out


def _make_stamp_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mr = h // 2
    mc = w // 2

    mask[mr, :] = True
    mask[:, mc] = True

    if random.random() < 0.45:
        mask[0, :] = True
    if random.random() < 0.45:
        mask[-1, :] = True
    if random.random() < 0.45:
        mask[:, 0] = True
    if random.random() < 0.45:
        mask[:, -1] = True

    if random.random() < 0.55:
        half_h = random.randint(0, h // 2)
        half_w = random.randint(0, w // 2)
        mask[mr - half_h : mr + half_h + 1, mc - half_w : mc + half_w + 1] = True

    if random.random() < 0.35 and h >= 5:
        side = random.choice([0, h - 1])
        c0 = random.randint(0, mc)
        c1 = random.randint(mc, w - 1)
        mask[side, c0 : c1 + 1] = True

    if random.random() < 0.35 and w >= 5:
        side = random.choice([0, w - 1])
        r0 = random.randint(0, mr)
        r1 = random.randint(mr, h - 1)
        mask[r0 : r1 + 1, side] = True

    return mask


def _make_stamp_patch(color: int, want_large: bool) -> np.ndarray:
    size_pool = [s for s in STAMP_SIZE_CHOICES if (s >= 5 if want_large else True)]
    if not size_pool:
        size_pool = list(STAMP_SIZE_CHOICES)
    h = random.choice(size_pool)
    w = random.choice(size_pool)
    mask = _make_stamp_mask(h, w)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_stamp_patches(grid: np.ndarray, patches: dict[int, np.ndarray]) -> tuple[int, bool]:
    h, w = grid.shape
    order = list(patches)
    random.shuffle(order)
    occupied = np.zeros((h, w), dtype=bool)
    lowest = -1

    for color in order:
        patch = patches[color]
        ph, pw = patch.shape
        placed = False
        row_hi = max(1, h // 2)
        for _ in range(200):
            if ph > row_hi - 1 or pw > w - 2:
                break
            r0 = random.randint(1, row_hi - ph)
            c0 = random.randint(1, w - pw - 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            occupied[rr0:rr1, cc0:cc1] = True
            lowest = max(lowest, r0 + ph - 1)
            placed = True
            break
        if not placed:
            return -1, False

    return lowest + random.randint(2, 3), True


def _choose_instruction_colors(
    patches: dict[int, np.ndarray],
    n_instr: int,
    want_clip: bool,
    want_overlap: bool,
) -> list[int]:
    colors = list(patches)
    by_area = sorted(colors, key=lambda c: int(np.count_nonzero(patches[c])), reverse=True)
    by_width = sorted(colors, key=lambda c: patches[c].shape[1], reverse=True)

    seq: list[int] = []
    if want_clip:
        seq.append(by_width[0])
    if want_overlap and len(colors) > 1:
        for color in by_area:
            if color not in seq:
                seq.append(color)
                break

    while len(seq) < n_instr:
        seq.append(random.choice(colors))

    if len(colors) > 1 and len(set(seq)) < 2:
        alt = next(color for color in colors if color != seq[0])
        seq[-1] = alt

    if random.random() < 0.5:
        tail = seq[2:] if len(seq) > 2 else []
        random.shuffle(tail)
        seq = seq[:2] + tail

    return seq[:n_instr]


def _build_path(
    shape: tuple[int, int],
    path_top: int,
    instr_colors: list[int],
    want_clip: bool,
    want_overlap: bool,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]] | None:
    h, w = shape
    n_instr = len(instr_colors)
    if path_top >= h - 2:
        return None

    first_axis = random.choice([0, 1])
    axes = [first_axis]
    for _ in range(1, n_instr):
        if random.random() < 0.65:
            axes.append(1 - axes[-1])
        else:
            axes.append(axes[-1])
    if n_instr > 1 and len(set(axes)) == 1:
        axes[random.randrange(1, n_instr)] = 1 - axes[0]

    if want_overlap:
        seg_lo, seg_hi = 2, 4
        gap_choices = [1]
    else:
        seg_lo, seg_hi = 3, 6
        gap_choices = [1, 2]
    seg_lens = [random.randint(seg_lo, seg_hi) for _ in range(n_instr)]
    gap_lens = [random.choice(gap_choices) for _ in range(n_instr - 1)]

    start_r = path_top + random.randint(0, min(1, h - path_top - 2))
    start_c = 0 if want_clip else random.randint(0, min(2, w - 2))
    if first_axis == 1 and not want_clip:
        start_r = path_top + random.randint(0, min(2, h - path_top - 2))

    def step(cell: tuple[int, int], axis: int) -> tuple[int, int]:
        r, c = cell
        return (r + 1, c) if axis == 0 else (r, c + 1)

    cells: list[tuple[int, int, int]] = [(start_r, start_c, int(instr_colors[0]))]
    instructions: list[list[tuple[int, int]]] = [[(start_r, start_c)]]
    occupied = {(start_r, start_c)}
    cur = (start_r, start_c)

    for _ in range(seg_lens[0] - 1):
        cur = step(cur, axes[0])
        if cur[0] >= h or cur[1] >= w or cur in occupied:
            return None
        occupied.add(cur)
        cells.append((cur[0], cur[1], int(instr_colors[0])))
        instructions[0].append(cur)

    for i in range(n_instr - 1):
        for _ in range(gap_lens[i]):
            cur = step(cur, axes[i + 1])
            if cur[0] >= h or cur[1] >= w or cur in occupied:
                return None
            occupied.add(cur)
            cells.append((cur[0], cur[1], GREY))

        cur = step(cur, axes[i + 1])
        if cur[0] >= h or cur[1] >= w or cur in occupied:
            return None
        occupied.add(cur)
        instructions.append([cur])
        cells.append((cur[0], cur[1], int(instr_colors[i + 1])))

        for _ in range(seg_lens[i + 1] - 1):
            cur = step(cur, axes[i + 1])
            if cur[0] >= h or cur[1] >= w or cur in occupied:
                return None
            occupied.add(cur)
            instructions[-1].append(cur)
            cells.append((cur[0], cur[1], int(instr_colors[i + 1])))

    grid = np.full(shape, BG, dtype=int)
    for r, c, color in cells:
        grid[r, c] = int(color)
    inst_arrays = [(int(instr_colors[i]), np.asarray(coords, dtype=int)) for i, coords in enumerate(instructions)]
    return grid, inst_arrays


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_stamps = random.randint(*STAMP_COUNT_RANGE)
    n_instr = random.randint(*INSTR_COUNT_RANGE)
    want_clip = random.random() < 0.4
    want_overlap = random.random() < 0.55

    colors = random.sample(STAMP_COLORS, n_stamps)
    patches = {
        color: _make_stamp_patch(
            color,
            want_large=want_clip or want_overlap or random.random() < 0.35,
        )
        for color in colors
    }

    grid = np.full((h, w), BG, dtype=int)
    path_top, placed = _place_stamp_patches(grid, patches)
    if not placed or path_top >= h - 3:
        return None

    instr_colors = _choose_instruction_colors(patches, n_instr, want_clip, want_overlap)
    built = _build_path((h, w), path_top, instr_colors, want_clip, want_overlap)
    if built is None:
        return None
    path_grid, instructions = built
    if np.any((grid != BG) & (path_grid != BG)):
        return None
    grid[path_grid != BG] = path_grid[path_grid != BG]

    expected, has_clip, has_overlap = _apply_instructions(grid.shape, instructions, patches, BG)
    if want_clip and not has_clip:
        return None
    if want_overlap and not has_overlap:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate stamp/path scenes that satisfy the solver assumptions and
    cover both ordering and clipping/overlap behavior.
    """
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
