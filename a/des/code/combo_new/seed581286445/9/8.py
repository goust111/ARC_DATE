from __future__ import annotations

from functools import lru_cache

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PAIR_COUNT_OPTIONS = [2, 3, 4]
GEN_RETRIES = 500
_RNG = np.random.default_rng()


def _normalize_coords(coords) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    assert pts, "shape must contain at least one cell"
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = tuple(sorted((r - min_r, c - min_c) for r, c in pts))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


@lru_cache(maxsize=None)
def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return (
        max(r for r, _ in shape) + 1,
        max(c for _, c in shape) + 1,
    )


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[int(r), int(c)] = True
    return mask


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(g[r0, c0])
            if color == 0 or seen[r0, c0]:
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            coords.sort()
            shape, sh, sw = _normalize_coords(coords)
            top = min(r for r, _ in coords)
            left = min(c for _, c in coords)
            bottom = max(r for r, _ in coords)
            right = max(c for _, c in coords)
            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "shape": shape,
                    "area": len(coords),
                    "top": int(top),
                    "left": int(left),
                    "bbox": (int(top), int(left), int(bottom), int(right)),
                    "h": int(sh),
                    "w": int(sw),
                }
            )
    return comps


@lru_cache(maxsize=None)
def _pair_can_form_rectangle(
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
) -> bool:
    area = len(shape_a) + len(shape_b)
    ha, wa = _shape_dims(shape_a)
    hb, wb = _shape_dims(shape_b)
    pts_a = np.array(shape_a, dtype=int)

    for rect_h in range(max(ha, hb), area + 1):
        if area % rect_h != 0:
            continue
        rect_w = area // rect_h
        if rect_w < max(wa, wb):
            continue

        for off_r in range(rect_h - ha + 1):
            for off_c in range(rect_w - wa + 1):
                occ = np.zeros((rect_h, rect_w), dtype=bool)
                occ[pts_a[:, 0] + off_r, pts_a[:, 1] + off_c] = True
                comp_pts = np.argwhere(~occ)
                comp_shape, ch, cw = _normalize_coords(comp_pts)
                if ch == hb and cw == wb and comp_shape == shape_b:
                    return True
    return False


def _are_complementary(
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
) -> bool:
    if shape_a <= shape_b:
        return _pair_can_form_rectangle(shape_a, shape_b)
    return _pair_can_form_rectangle(shape_b, shape_a)


def _pair_graph(comps: list[dict]) -> list[tuple[int, ...]]:
    n = len(comps)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _are_complementary(comps[i]["shape"], comps[j]["shape"]):
                adj[i].append(j)
                adj[j].append(i)
    return [tuple(neigh) for neigh in adj]


def _count_matchings(adj: list[tuple[int, ...]], limit: int = 2) -> tuple[int, dict[int, int] | None]:
    n = len(adj)
    used = [False] * n
    cur: dict[int, int] = {}
    first_solution: dict[int, int] | None = None
    count = 0

    def dfs() -> None:
        nonlocal count, first_solution
        if count >= limit:
            return

        pick = None
        pick_options = None
        for i in range(n):
            if used[i]:
                continue
            options = [j for j in adj[i] if not used[j]]
            if not options:
                return
            if pick is None or len(options) < len(pick_options):
                pick = i
                pick_options = options

        if pick is None:
            count += 1
            if first_solution is None:
                first_solution = dict(cur)
            return

        used[pick] = True
        for j in pick_options:
            if used[j]:
                continue
            used[j] = True
            cur[pick] = j
            cur[j] = pick
            dfs()
            del cur[pick]
            del cur[j]
            used[j] = False
            if count >= limit:
                break
        used[pick] = False

    dfs()
    return int(count), first_solution


def _matched_pairs(pairing: dict[int, int]) -> list[tuple[int, int]]:
    return sorted((i, j) for i, j in pairing.items() if i < j)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each scattered fragment pair by the color of the smaller fragment after matching pairs that complete a rectangle.

    Input structure:
    - The background is Black.
    - The foreground contains an even number of monochrome 4-connected fragments, each with its own color.
    - Every true pair of fragments can be translated, without rotation or reflection, so their union becomes one solid axis-aligned rectangle.

    Rule:
    - Extract all foreground fragments and compare their shapes to find which pairs can tile the same filled rectangle by translation alone.
    - For each matched pair, compare the two fragment areas.
    - Recolor only the larger fragment to the smaller fragment's color; if the areas are equal, keep the color of the fragment whose topmost row is higher.
    - Preserve every fragment's position and shape.

    Key insight:
    - The main challenge is recognizing complementary shapes that jointly fill a rectangle; once the pairing is known, the surviving color is chosen by the smaller-area fragment rather than by geometry or location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_components(g)
    assert comps and len(comps) % 2 == 0, "expected an even number of foreground components"

    adj = _pair_graph(comps)
    solution_count, pairing = _count_matchings(adj, limit=1)
    assert solution_count >= 1 and pairing is not None, "no valid complementary pairing found"

    out = g.copy()
    for i, j in _matched_pairs(pairing):
        a = comps[i]
        b = comps[j]
        if a["area"] < b["area"]:
            keep_color = int(a["color"])
            repaint = b
        elif b["area"] < a["area"]:
            keep_color = int(b["color"])
            repaint = a
        elif (a["top"], a["left"]) <= (b["top"], b["left"]):
            keep_color = int(a["color"])
            repaint = b
        else:
            keep_color = int(b["color"])
            repaint = a

        for r, c in repaint["coords"]:
            out[int(r), int(c)] = keep_color
    return out


def _sample_profile(total: int, parts: int, max_part: int) -> list[int] | None:
    for _ in range(120):
        vals = []
        remaining = int(total)
        ok = True
        for i in range(parts):
            slots = parts - i - 1
            lo = max(1, remaining - max_part * slots)
            hi = min(max_part, remaining - slots)
            if lo > hi:
                ok = False
                break
            val = int(_RNG.integers(lo, hi + 1))
            vals.append(val)
            remaining -= val
        if not ok or remaining != 0:
            continue
        vals = [int(v) for v in _RNG.permutation(vals)]
        if len(set(vals)) > 1:
            return vals
    return None


def _mask_to_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(mask)
    shape, _h, _w = _normalize_coords(pts)
    return shape


def _make_pair(equal_area: bool) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] | None:
    for _ in range(200):
        rect_h = int(_RNG.integers(3, 7))
        rect_w = int(_RNG.integers(4, 8))
        total = rect_h * rect_w
        lo = rect_h
        hi = total - rect_h

        if equal_area:
            if total % 2 != 0:
                continue
            target = total // 2
            if not (lo <= target <= hi):
                continue
        else:
            choices = [area for area in range(lo, hi + 1) if not (total % 2 == 0 and area == total // 2)]
            if not choices:
                continue
            target = int(choices[int(_RNG.integers(0, len(choices)))])

        profile = _sample_profile(target, rect_h, rect_w - 1)
        if profile is None:
            continue

        mask_a = np.zeros((rect_h, rect_w), dtype=bool)
        for r, width in enumerate(profile):
            mask_a[int(r), : int(width)] = True
        mask_b = ~mask_a

        if int(_RNG.integers(0, 2)):
            mask_a = mask_a.T
            mask_b = mask_b.T
        if int(_RNG.integers(0, 2)):
            mask_a = np.flipud(mask_a)
            mask_b = np.flipud(mask_b)
        if int(_RNG.integers(0, 2)):
            mask_a = np.fliplr(mask_a)
            mask_b = np.fliplr(mask_b)

        shape_a = _mask_to_shape(mask_a)
        shape_b = _mask_to_shape(mask_b)
        if _are_complementary(shape_a, shape_b):
            if int(_RNG.integers(0, 2)):
                return shape_b, shape_a
            return shape_a, shape_b
    return None


def _branch_flags(n_pairs: int) -> list[bool]:
    if n_pairs == 2:
        flags = [True, False]
    elif n_pairs == 3:
        eq = 1 + int(_RNG.integers(0, 2))
        flags = [True] * eq + [False] * (n_pairs - eq)
    else:
        flags = [True] * (n_pairs // 2) + [False] * (n_pairs - n_pairs // 2)
    flags = np.array(flags, dtype=bool)
    _RNG.shuffle(flags)
    return [bool(x) for x in flags]


def _scatter_shapes(
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> np.ndarray | None:
    masks = [_shape_mask(shape) for shape in shapes]
    coords = [np.argwhere(mask) for mask in masks]
    heights = [mask.shape[0] for mask in masks]
    widths = [mask.shape[1] for mask in masks]
    padded_area = sum((h + 2) * (w + 2) for h, w in zip(heights, widths))

    for _ in range(60):
        min_h = max(max(heights) + 2, int(np.ceil(np.sqrt(padded_area * 0.75))))
        min_h = min(min_h, 30)
        if min_h > 30:
            return None
        grid_h = int(_RNG.integers(min_h, 31))
        min_w = max(max(widths) + 2, int(np.ceil(padded_area / grid_h)))
        if min_w > 30:
            continue
        grid_w = int(_RNG.integers(min_w, 31))

        grid = np.zeros((grid_h, grid_w), dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)

        order = list(_RNG.permutation(len(shapes)))
        order.sort(key=lambda idx: (-len(shapes[idx]), -(heights[idx] * widths[idx])))

        success = True
        for idx in order:
            shape_pts = coords[idx]
            h = heights[idx]
            w = widths[idx]
            spots: list[tuple[int, int]] = []
            for r0 in range(1, grid_h - h):
                for c0 in range(1, grid_w - w):
                    abs_r = shape_pts[:, 0] + r0
                    abs_c = shape_pts[:, 1] + c0
                    if blocked[abs_r, abs_c].any():
                        continue
                    spots.append((int(r0), int(c0)))

            if not spots:
                success = False
                break

            r0, c0 = spots[int(_RNG.integers(0, len(spots)))]
            abs_r = shape_pts[:, 0] + r0
            abs_c = shape_pts[:, 1] + c0
            grid[abs_r, abs_c] = int(colors[idx])
            for rr, cc in zip(abs_r, abs_c):
                r1 = max(0, int(rr) - 1)
                r2 = min(grid_h, int(rr) + 2)
                c1 = max(0, int(cc) - 1)
                c2 = min(grid_w, int(cc) + 2)
                blocked[r1:r2, c1:c2] = True

        if success:
            return grid
    return None


def _valid_generated_sample(grid: np.ndarray) -> bool:
    comps = _find_components(grid)
    if not comps or len(comps) % 2 != 0:
        return False

    adj = _pair_graph(comps)
    if any(len(neigh) != 1 for neigh in adj):
        return False

    count, pairing = _count_matchings(adj, limit=2)
    if count != 1 or pairing is None:
        return False

    pairs = _matched_pairs(pairing)
    equal_pairs = 0
    for i, j in pairs:
        a = comps[i]
        b = comps[j]
        if a["area"] == b["area"]:
            equal_pairs += 1
            if a["top"] == b["top"]:
                return False

    if len(pairs) >= 2 and equal_pairs in (0, len(pairs)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_pairs = int(PAIR_COUNT_OPTIONS[int(_RNG.integers(0, len(PAIR_COUNT_OPTIONS)))])
    shapes: list[tuple[tuple[int, int], ...]] = []

    for equal_area in _branch_flags(n_pairs):
        pair = _make_pair(equal_area)
        if pair is None:
            return None
        shapes.extend(pair)

    colors = [int(c) for c in _RNG.choice(COLOR_POOL, size=len(shapes), replace=False)]
    grid = _scatter_shapes(shapes, colors)
    if grid is None:
        return None
    if not _valid_generated_sample(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment sample")
