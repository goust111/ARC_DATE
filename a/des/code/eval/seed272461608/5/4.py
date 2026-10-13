import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

KINDS = ("H", "V", "B", "N")
FRAME_RECOLOR = {"H": RED, "V": BLUE, "B": GREEN, "N": YELLOW}
PATTERN_COLORS = (6, 7, 8, 9)

GRID_H_RANGE = (24, 34)
GRID_W_RANGE = (24, 36)
MARGIN_RANGE = (1, 4)
PATTERN_SIDE_RANGE = (3, 6)
PATTERN_RETRIES = 120
PLACE_RETRIES = 200
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("mask must contain at least one foreground pixel")
    return _bbox_from_coords(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _symmetry_kind(mask: np.ndarray) -> str:
    sym_h = np.array_equal(mask, np.flipud(mask))
    sym_v = np.array_equal(mask, np.fliplr(mask))
    if sym_h and sym_v:
        return "B"
    if sym_h:
        return "H"
    if sym_v:
        return "V"
    return "N"


def _mirror_rows(base: np.ndarray, full_h: int) -> np.ndarray:
    if full_h % 2:
        return np.vstack([base, base[-2::-1]])
    return np.vstack([base, base[::-1]])


def _mirror_cols(base: np.ndarray, full_w: int) -> np.ndarray:
    if full_w % 2:
        return np.hstack([base, base[:, -2::-1]])
    return np.hstack([base, base[:, ::-1]])


def _random_connected_mask(
    height: int,
    width: int,
    *,
    seed: tuple[int, int],
    min_area: int,
    max_area: int,
) -> np.ndarray:
    total = int(height) * int(width)
    lo = max(1, int(min_area))
    hi = min(total, int(max_area))
    target = random.randint(lo, hi)

    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[seed] = True
    frontier = set(_neighbors4(seed[0], seed[1], int(height), int(width)))
    while int(mask.sum()) < target:
        if not frontier:
            break
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c, int(height), int(width)):
            if not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _augment_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    out = mask.copy()
    if kind == "B":
        out = np.rot90(out, random.randrange(4))
        if random.choice([False, True]):
            out = np.fliplr(out)
    elif kind == "N":
        out = np.rot90(out, random.randrange(4))
        if random.choice([False, True]):
            out = np.fliplr(out)
    else:
        if random.choice([False, True]):
            out = np.flipud(out)
        if random.choice([False, True]):
            out = np.fliplr(out)
    return out.copy()


def _make_pattern(kind: str) -> np.ndarray:
    for _ in range(PATTERN_RETRIES):
        full_h = random.randint(*PATTERN_SIDE_RANGE)
        full_w = random.randint(*PATTERN_SIDE_RANGE)

        if kind == "H":
            base_h = (full_h + 1) // 2
            seed = (base_h - 1, random.randrange(full_w))
            base = _random_connected_mask(
                base_h,
                full_w,
                seed=seed,
                min_area=2,
                max_area=max(2, base_h * full_w - 1),
            )
            mask = _mirror_rows(base, full_h)
        elif kind == "V":
            base_w = (full_w + 1) // 2
            seed = (random.randrange(full_h), base_w - 1)
            base = _random_connected_mask(
                full_h,
                base_w,
                seed=seed,
                min_area=2,
                max_area=max(2, full_h * base_w - 1),
            )
            mask = _mirror_cols(base, full_w)
        elif kind == "B":
            base_h = (full_h + 1) // 2
            base_w = (full_w + 1) // 2
            seed = (base_h - 1, base_w - 1)
            base = _random_connected_mask(
                base_h,
                base_w,
                seed=seed,
                min_area=2,
                max_area=max(2, base_h * base_w - 1),
            )
            mask = _mirror_cols(_mirror_rows(base, full_h), full_w)
        else:
            seed = (random.randrange(full_h), random.randrange(full_w))
            mask = _random_connected_mask(
                full_h,
                full_w,
                seed=seed,
                min_area=2,
                max_area=max(2, full_h * full_w - 1),
            )

        cropped = _crop_mask(mask)
        if not _is_connected(cropped):
            continue
        if min(cropped.shape) < 2 or max(cropped.shape) < 3:
            continue
        if kind == "B" and int(cropped.sum()) == cropped.size:
            continue
        if _symmetry_kind(cropped) != kind:
            continue
        return _augment_mask(cropped, kind)
    raise RuntimeError(f"failed to generate {kind} pattern")


def _analyze_frame(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert frame.shape[0] >= 3 and frame.shape[1] >= 3
    assert np.all(frame[0, :] == GREY) and np.all(frame[-1, :] == GREY)
    assert np.all(frame[:, 0] == GREY) and np.all(frame[:, -1] == GREY)
    assert not np.any(frame[1:-1, 1:-1] == GREY)

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = set(map(int, np.unique(interior))) - {BLACK}
    assert len(colors) == 1
    pattern_color = int(next(iter(colors)))
    mask = interior == pattern_color
    pr0, pc0, pr1, pc1 = _mask_bbox(mask)
    tile = interior[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    kind = _symmetry_kind(tile != BLACK)
    return {
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "kind": kind,
        "pattern_color": pattern_color,
        "pattern_top": pr0,
        "pattern_left": pc0,
        "tile": tile,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    infos = [_analyze_frame(grid, coords) for coords in _components_of_color(grid, GREY)]
    infos.sort(key=lambda info: info["bbox"])
    return infos


def _paint_tile_clipped(dest: np.ndarray, top: int, left: int, tile: np.ndarray) -> None:
    ph, pw = tile.shape
    h, w = dest.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(h, int(top) + ph)
    c1 = min(w, int(left) + pw)
    if r0 >= r1 or c0 >= c1:
        return
    sub = tile[r0 - int(top) : r1 - int(top), c0 - int(left) : c1 - int(left)]
    mask = sub != BLACK
    block = dest[r0:r1, c0:c1]
    block[mask] = sub[mask]


def _repeat_steps(start: int, span: int, limit: int) -> range:
    k_min = -((int(start) + int(span) - 1) // int(span))
    k_max = (int(limit) - 1 - int(start)) // int(span)
    return range(k_min, k_max + 1)


def _extend_pattern(
    interior: np.ndarray,
    tile: np.ndarray,
    top: int,
    left: int,
    kind: str,
) -> None:
    if kind == "N":
        return

    ih, iw = interior.shape
    ph, pw = tile.shape
    row_steps = _repeat_steps(top, ph, ih) if kind in {"H", "B"} else range(1)
    col_steps = _repeat_steps(left, pw, iw) if kind in {"V", "B"} else range(1)

    for kr in row_steps:
        for kc in col_steps:
            rr = int(top) + int(kr) * ph
            cc = int(left) + int(kc) * pw
            part = tile
            if kind in {"H", "B"} and abs(int(kr)) % 2 == 1:
                part = np.flipud(part)
            if kind in {"V", "B"} and abs(int(kc)) % 2 == 1:
                part = np.fliplr(part)
            _paint_tile_clipped(interior, rr, cc, part)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey frame by the symmetry class of its inner pattern and
    expand the pattern along the permitted symmetry axis or axes.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for info in _find_frames(g):
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = FRAME_RECOLOR[info["kind"]]
        r0, c0, r1, c1 = info["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        _extend_pattern(
            interior,
            info["tile"],
            info["pattern_top"],
            info["pattern_left"],
            info["kind"],
        )
    return out


def _build_frame_patch(kind: str, color: int) -> np.ndarray:
    mask = _make_pattern(kind)
    ph, pw = mask.shape
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    interior_h = top + ph + bottom
    interior_w = left + pw + right
    patch = np.full((interior_h + 2, interior_w + 2), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    inner = patch[1 + top : 1 + top + ph, 1 + left : 1 + left + pw]
    inner[mask] = int(color)
    return patch


def _sample_grid_shape(patches: list[np.ndarray]) -> tuple[int, int] | None:
    min_h = max(p.shape[0] for p in patches) + 2
    min_w = max(p.shape[1] for p in patches) + 2
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    lo_h = max(GRID_H_RANGE[0], min_h)
    lo_w = max(GRID_W_RANGE[0], min_w)
    if lo_h > GRID_H_RANGE[1] or lo_w > GRID_W_RANGE[1]:
        return None

    for _ in range(40):
        h = random.randint(lo_h, GRID_H_RANGE[1])
        w = random.randint(lo_w, GRID_W_RANGE[1])
        if h * w >= int(total_area * 1.6):
            return h, w
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, int(top) - 1)
    c0 = max(0, int(left) - 1)
    r1 = min(h, int(top) + int(height) + 1)
    c1 = min(w, int(left) + int(width) + 1)
    return not np.any(occupied[r0:r1, c0:c1])


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full(shape, BLACK, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    order = sorted(range(len(patches)), key=lambda i: patches[i].size, reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(0, shape[0] - ph)
            left = random.randint(0, shape[1] - pw)
            if not _can_place(occupied, top, left, ph, pw):
                continue
            grid[top : top + ph, left : left + pw] = patch
            occupied[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _pattern_colors(n: int) -> list[int]:
    colors = random.sample(PATTERN_COLORS, k=len(PATTERN_COLORS))
    while len(colors) < n:
        colors.append(random.choice(PATTERN_COLORS))
    random.shuffle(colors)
    return colors[:n]


def _gen_attempt() -> np.ndarray | None:
    types = list(KINDS)
    if random.choice([False, True]):
        types.append(random.choice(KINDS))
    random.shuffle(types)

    colors = _pattern_colors(len(types))
    patches = [_build_frame_patch(kind, color) for kind, color in zip(types, colors)]
    shape = _sample_grid_shape(patches)
    if shape is None:
        return None

    grid = _place_patches(shape, patches)
    if grid is None:
        return None

    infos = _find_frames(grid)
    if len(infos) != len(types):
        return None
    if Counter(info["kind"] for info in infos) != Counter(types):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
