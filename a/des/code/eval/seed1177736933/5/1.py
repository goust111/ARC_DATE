import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
SEED = 9

SYM_H = "horizontal"
SYM_V = "vertical"
SYM_R = "rot180"
SYM_N = "none"
SYMMETRY_KINDS = (SYM_H, SYM_V, SYM_R, SYM_N)

FILL_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRIES = 400
GEN_K_CHOICES = (3, 4)
GEN_REGION_COUNT_RANGE = (4, 6)
GEN_LEGEND_GAP = 1
GEN_BOX_PAD = 1
GEN_PLACE_TRIES = 250


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
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _symmetry_kind(mask: np.ndarray) -> str:
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or not arr.any():
        raise ValueError("seed mask must be a non-empty square")

    is_h = np.array_equal(arr, np.flipud(arr))
    is_v = np.array_equal(arr, np.fliplr(arr))
    is_r = np.array_equal(arr, np.rot90(arr, 2))

    if is_h and not is_v and not is_r:
        return SYM_H
    if is_v and not is_h and not is_r:
        return SYM_V
    if is_r and not is_h and not is_v:
        return SYM_R
    if not is_h and not is_v and not is_r:
        return SYM_N
    raise ValueError("ambiguous symmetry type")


def _legend_mapping(grid: np.ndarray) -> tuple[dict[str, int], int]:
    blocks = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, WALL, SEED):
            continue
        coords = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        if not coords:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if patch.size != len(coords) or not np.all(patch == color):
            raise ValueError("legend color block must be a solid rectangle")
        if patch.shape[0] != patch.shape[1]:
            raise ValueError("legend color block must be square")
        blocks.append((r0, c0, patch.shape[0], color))

    if len(blocks) != 4:
        raise ValueError("expected exactly four legend color blocks")

    k = blocks[0][2]
    mapping: dict[str, int] = {}
    for r0, c0, size, color in sorted(blocks):
        if size != k or c0 - k < 0:
            raise ValueError("legend entries must share one pattern size")
        demo = grid[r0 : r0 + k, c0 - k : c0] == SEED
        kind = _symmetry_kind(demo)
        mapping[kind] = color

    if set(mapping) != set(SYMMETRY_KINDS):
        raise ValueError("legend must cover all symmetry types exactly once")
    return mapping, k


def _rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == WALL):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(view, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(view == WALL, border):
            rects.append((r0, c0, r1, c1))
    return sorted(rects)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map each seed's symmetry type to a legend color, then fill its enclosing
    rectangle while preserving the maroon seed.
    """
    g = np.asarray(grid, dtype=int)
    mapping, expected_size = _legend_mapping(g)

    out = g.copy()
    for r0, c0, r1, c1 in _rectangles(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = [tuple(map(int, pt)) for pt in np.argwhere(interior == SEED)]
        if not seed_coords:
            continue

        sr0, sc0, sr1, sc1 = _bbox(seed_coords)
        seed_patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1] == SEED
        if seed_patch.shape != (expected_size, expected_size):
            raise ValueError("seed bbox must match legend pattern size")
        fill_color = mapping[_symmetry_kind(seed_patch)]

        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[region != SEED] = fill_color

    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_horizontal_mask(size: int, rng: random.Random) -> np.ndarray:
    top_rows = (size + 1) // 2
    data = np.zeros((size, size), dtype=bool)
    p = rng.uniform(0.25, 0.7)
    for r in range(top_rows):
        for c in range(size):
            data[r, c] = rng.random() < p
    for r in range(size // 2):
        data[size - 1 - r] = data[r]
    return data


def _make_vertical_mask(size: int, rng: random.Random) -> np.ndarray:
    left_cols = (size + 1) // 2
    data = np.zeros((size, size), dtype=bool)
    p = rng.uniform(0.25, 0.7)
    for r in range(size):
        for c in range(left_cols):
            data[r, c] = rng.random() < p
    for c in range(size // 2):
        data[:, size - 1 - c] = data[:, c]
    return data


def _make_rot180_mask(size: int, rng: random.Random) -> np.ndarray:
    data = np.zeros((size, size), dtype=bool)
    p = rng.uniform(0.25, 0.65)
    for r in range(size):
        for c in range(size):
            rr, cc = size - 1 - r, size - 1 - c
            if (r > rr) or (r == rr and c > cc):
                continue
            value = rng.random() < p
            data[r, c] = value
            data[rr, cc] = value
    return data


def _make_none_mask(size: int, rng: random.Random) -> np.ndarray:
    p = rng.uniform(0.2, 0.75)
    return np.array([[rng.random() < p for _ in range(size)] for _ in range(size)], dtype=bool)


def _generate_mask(kind: str, size: int, rng: random.Random, forbid: np.ndarray | None = None) -> np.ndarray:
    maker = {
        SYM_H: _make_horizontal_mask,
        SYM_V: _make_vertical_mask,
        SYM_R: _make_rot180_mask,
        SYM_N: _make_none_mask,
    }[kind]

    for _ in range(600):
        mask = maker(size, rng)
        if not mask.any() or not _touches_all_sides(mask):
            continue
        try:
            if _symmetry_kind(mask) != kind:
                continue
        except ValueError:
            continue
        if forbid is not None and np.array_equal(mask, forbid):
            continue
        return mask
    raise RuntimeError(f"failed to generate a {kind} seed")


def _mark_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _box_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _draw_rectangle(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = WALL
    grid[r0 + h - 1, c0 : c0 + w] = WALL
    grid[r0 : r0 + h, c0] = WALL
    grid[r0 : r0 + h, c0 + w - 1] = WALL


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.choice(GEN_K_CHOICES)
    region_count = rng.randint(*GEN_REGION_COUNT_RANGE)

    legend_h = 4 * size + 3 * GEN_LEGEND_GAP
    min_h = max(legend_h + 2, 20)
    min_w = max(2 * size + 18, 28)
    h = rng.randint(min_h, min_h + 8)
    w = rng.randint(min_w, min_w + 8)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend_top = rng.randint(0, h - legend_h)
    legend_order = list(SYMMETRY_KINDS)
    rng.shuffle(legend_order)
    legend_colors = rng.sample(list(FILL_COLORS), 4)
    color_map = dict(zip(legend_order, legend_colors))
    legend_masks = {kind: _generate_mask(kind, size, rng) for kind in SYMMETRY_KINDS}

    for idx, kind in enumerate(legend_order):
        r0 = legend_top + idx * (size + GEN_LEGEND_GAP)
        grid[r0 : r0 + size, 0:size][legend_masks[kind]] = SEED
        grid[r0 : r0 + size, size : 2 * size] = color_map[kind]
    _mark_box(occupied, legend_top, 0, legend_h, 2 * size, pad=2)

    region_kinds = list(SYMMETRY_KINDS)
    while len(region_kinds) < region_count:
        region_kinds.append(rng.choice(SYMMETRY_KINDS))
    rng.shuffle(region_kinds)

    if region_count >= 6:
        inner_h_max = size + 5
        inner_w_max = size + 6
    else:
        inner_h_max = size + 7
        inner_w_max = size + 8

    specs = []
    mandatory_seen = set()
    for kind in region_kinds:
        inner_h = rng.randint(size + 2, inner_h_max)
        inner_w = rng.randint(size + 2, inner_w_max)
        seed_mask = _generate_mask(
            kind,
            size,
            rng,
            forbid=legend_masks[kind] if kind not in mandatory_seen else None,
        )
        mandatory_seen.add(kind)
        specs.append(
            {
                "kind": kind,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "h": inner_h + 2,
                "w": inner_w + 2,
                "mask": seed_mask,
            }
        )

    specs.sort(key=lambda item: item["h"] * item["w"], reverse=True)
    placements = []
    left_limit = 2 * size + 2

    for spec in specs:
        placed = False
        for _ in range(GEN_PLACE_TRIES):
            r0 = rng.randint(0, h - spec["h"])
            c0 = rng.randint(left_limit, w - spec["w"])
            if not _box_is_free(occupied, r0, c0, spec["h"], spec["w"], GEN_BOX_PAD):
                continue
            placements.append((r0, c0, spec))
            _mark_box(occupied, r0, c0, spec["h"], spec["w"], GEN_BOX_PAD)
            placed = True
            break
        if not placed:
            return None

    rng.shuffle(placements)
    for r0, c0, spec in placements:
        _draw_rectangle(grid, r0, c0, spec["h"], spec["w"])
        seed_r = r0 + 1 + (spec["inner_h"] - size) // 2
        seed_c = c0 + 1 + (spec["inner_w"] - size) // 2
        patch = grid[seed_r : seed_r + size, seed_c : seed_c + size]
        patch[spec["mask"]] = SEED

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
