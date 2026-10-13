import numpy as np
from collections import defaultdict, deque


BG = 0
YELLOW = 4
ORANGE = 7
BORDER_COLORS = (1, 2, 3, 5, 6, 8, 9)

GRID_H_RANGE = (20, 31)  # inclusive
GRID_W_RANGE = (20, 31)  # inclusive
PAIR_COUNT_VALUES = np.array([2, 3, 4], dtype=int)
PAIR_COUNT_WEIGHTS = np.array([3.0, 3.0, 1.0], dtype=float)
REGION_H_RANGE = (5, 8)  # inclusive
REGION_W_RANGE = (5, 8)  # inclusive
MIN_INTERIOR_CELLS = 4
PATTERN_MAX_CELLS = 6
PLACEMENT_GAP = 2
OUTER_MARGIN = 1
MAX_GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 120
PATTERN_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 250

RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8")
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _perimeter(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1, constant_values=False)
    all_nbrs = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~all_nbrs


def _enclosed_interior(border: np.ndarray) -> np.ndarray:
    border = np.asarray(border, dtype=bool)
    h, w = border.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not border[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not border[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not border[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return (~border) & (~outside)


def _rotation_matches(src_border: np.ndarray, src_marker: np.ndarray, dst_border: np.ndarray, dst_marker: np.ndarray) -> list[int]:
    matches: list[int] = []
    for k in range(4):
        if np.array_equal(np.rot90(src_border, k), dst_border) and np.array_equal(np.rot90(src_marker, k), dst_marker):
            matches.append(k)
    return matches


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = _dominant_color(grid)
    mask = (grid != bg) & (grid != YELLOW)
    objs = []
    for coords in _components(mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        border[rel[:, 0], rel[:, 1]] = True

        marker = (sub == ORANGE) & border
        marker_pos = np.argwhere(marker)
        assert len(marker_pos) == 1

        colors = {int(v) for v in sub[border] if int(v) != ORANGE}
        assert len(colors) == 1
        border_color = next(iter(colors))

        interior = _enclosed_interior(border)
        pattern = (sub == YELLOW) & interior
        objs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "border": border,
                "marker": marker,
                "interior": interior,
                "pattern": pattern,
                "border_color": border_color,
            }
        )
    return objs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored outline shapes, infer the relative rotation from the orange border marker,
    and copy the rotated yellow interior pattern from the filled outline into its empty mate.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    objects = _extract_objects(g)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_color[int(obj["border_color"])].append(obj)

    for _, group in by_color.items():
        filled = [obj for obj in group if bool(obj["pattern"].any())]
        empty = [obj for obj in group if not bool(obj["pattern"].any())]
        used_empty: set[int] = set()

        for src in filled:
            matches: list[tuple[int, int]] = []
            for j, dst in enumerate(empty):
                if j in used_empty:
                    continue
                for k in _rotation_matches(src["border"], src["marker"], dst["border"], dst["marker"]):
                    matches.append((j, k))
            assert len(matches) == 1
            j, rot = matches[0]
            used_empty.add(j)

            dst = empty[j]
            rotated_pattern = np.rot90(src["pattern"], rot)
            assert rotated_pattern.shape == dst["interior"].shape
            assert not np.any(rotated_pattern & ~dst["interior"])

            r0, c0, r1, c1 = dst["bbox"]
            sub = out[r0 : r1 + 1, c0 : c1 + 1]
            sub[rotated_pattern] = YELLOW

    return out


def _random_region() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = int(RNG.integers(REGION_H_RANGE[0], REGION_H_RANGE[1] + 1))
        w = int(RNG.integers(REGION_W_RANGE[0], REGION_W_RANGE[1] + 1))
        region = np.ones((h, w), dtype=bool)

        corners = ("tl", "tr", "bl", "br")
        corner_count = int(RNG.integers(0, 4))
        chosen_corners = RNG.choice(np.array(corners), size=corner_count, replace=False)
        for corner in chosen_corners:
            notch_h = int(RNG.integers(1, h - 2))
            notch_w = int(RNG.integers(1, w - 2))
            if corner == "tl":
                region[:notch_h, :notch_w] = False
            elif corner == "tr":
                region[:notch_h, w - notch_w :] = False
            elif corner == "bl":
                region[h - notch_h :, :notch_w] = False
            else:
                region[h - notch_h :, w - notch_w :] = False

        if int(RNG.integers(0, 100)) < 60:
            side = ("top", "bottom", "left", "right")[int(RNG.integers(0, 4))]
            if side in ("top", "bottom"):
                width = int(RNG.integers(1, w - 1))
                start = int(RNG.integers(1, w - width))
                depth = int(RNG.integers(1, h - 2))
                if side == "top":
                    region[:depth, start : start + width] = False
                else:
                    region[h - depth :, start : start + width] = False
            else:
                height = int(RNG.integers(1, h - 1))
                start = int(RNG.integers(1, h - height))
                depth = int(RNG.integers(1, w - 2))
                if side == "left":
                    region[start : start + height, :depth] = False
                else:
                    region[start : start + height, w - depth :] = False

        if not region.any():
            continue
        region = _crop_bool(region)
        if len(_components(region)) != 1:
            continue
        return region
    return None


def _north_exposed(region: np.ndarray) -> np.ndarray:
    exposed = np.zeros_like(region, dtype=bool)
    exposed[0, :] = region[0, :]
    exposed[1:, :] = region[1:, :] & ~region[:-1, :]
    return exposed


def _random_pattern(interior: np.ndarray) -> np.ndarray | None:
    coords = [tuple(map(int, pt)) for pt in np.argwhere(interior)]
    max_cells = min(PATTERN_MAX_CELLS, len(coords) - 1)
    if max_cells < 2:
        return None

    for _ in range(PATTERN_ATTEMPTS):
        count = int(RNG.integers(2, max_cells + 1))
        choose = RNG.choice(len(coords), size=count, replace=False)
        pattern = np.zeros_like(interior, dtype=bool)
        for idx in choose:
            pattern[coords[int(idx)]] = True
        if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            continue
        return pattern
    return None


def _build_base_shape() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_ATTEMPTS):
        region = _random_region()
        if region is None:
            continue
        border = _perimeter(region)
        if len(_components(border)) != 1:
            continue
        interior = _enclosed_interior(border)
        if int(interior.sum()) < MIN_INTERIOR_CELLS:
            continue

        marker_candidates = np.argwhere(border & _north_exposed(region))
        if len(marker_candidates) == 0:
            continue
        mr, mc = map(int, marker_candidates[int(RNG.integers(0, len(marker_candidates)))])
        marker = np.zeros_like(border, dtype=bool)
        marker[mr, mc] = True
        return border, marker, interior
    return None


def _render_patch(border: np.ndarray, marker: np.ndarray, pattern: np.ndarray, border_color: int) -> np.ndarray:
    patch = np.full(border.shape, BG, dtype=int)
    patch[border] = int(border_color)
    patch[pattern] = YELLOW
    patch[marker] = ORANGE
    return patch


def _make_pair(border_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    base = _build_base_shape()
    if base is None:
        return None
    border, marker, interior = base
    pattern = _random_pattern(interior)
    if pattern is None:
        return None

    src_rot = int(RNG.integers(0, 4))
    delta = int(RNG.integers(0, 4))
    dst_rot = (src_rot + delta) % 4

    filled = np.rot90(_render_patch(border, marker, pattern, border_color), src_rot)
    empty = np.rot90(_render_patch(border, marker, np.zeros_like(interior, dtype=bool), border_color), dst_rot)
    return filled, empty


def _place_patches(h: int, w: int, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(patches)), key=lambda i: int(np.count_nonzero(patches[i])), reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        r_min = OUTER_MARGIN
        c_min = OUTER_MARGIN
        r_max = h - ph - OUTER_MARGIN
        c_max = w - pw - OUTER_MARGIN
        if r_max < r_min or c_max < c_min:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(RNG.integers(r_min, r_max + 1))
            c0 = int(RNG.integers(c_min, c_max + 1))
            rr0 = max(0, r0 - PLACEMENT_GAP)
            rr1 = min(h, r0 + ph + PLACEMENT_GAP)
            cc0 = max(0, c0 - PLACEMENT_GAP)
            cc1 = min(w, c0 + pw + PLACEMENT_GAP)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            patch_mask = patch != BG
            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            sub[patch_mask] = patch[patch_mask]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.choice(PAIR_COUNT_VALUES, p=PAIR_COUNT_WEIGHTS / PAIR_COUNT_WEIGHTS.sum()))
    colors = RNG.choice(np.array(BORDER_COLORS, dtype=int), size=pair_count, replace=False)

    patches: list[np.ndarray] = []
    for color in colors:
        pair = _make_pair(int(color))
        if pair is None:
            return None
        a, b = pair
        if int(RNG.integers(0, 2)) == 0:
            patches.extend([a, b])
        else:
            patches.extend([b, a])

    h = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    grid = _place_patches(h, w, patches)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
