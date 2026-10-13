from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
import numpy as np


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in _DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _crop_coords(coords: list[tuple[int, int]]) -> dict:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    return {"coords": rel, "height": int(h), "width": int(w), "area": int(len(rel))}


def _rectangle_candidates(
    coords_a: frozenset[tuple[int, int]],
    coords_b: frozenset[tuple[int, int]],
    ha: int,
    wa: int,
    hb: int,
    wb: int,
) -> list[dict]:
    aset = set(coords_a)
    bset = set(coords_b)
    candidates: list[dict] = []
    for dr in range(-hb, ha + 1):
        for dc in range(-wb, wa + 1):
            shifted = {(int(r + dr), int(c + dc)) for r, c in bset}
            if aset & shifted:
                continue
            union = aset | shifted
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            r0 = min(rows)
            c0 = min(cols)
            h = max(rows) - r0 + 1
            w = max(cols) - c0 + 1
            if int(h * w) != int(len(union)):
                continue
            candidates.append(
                {
                    "dr": int(dr),
                    "dc": int(dc),
                    "height": int(h),
                    "width": int(w),
                    "area": int(len(union)),
                }
            )
    return candidates


def _build_output(sorted_colors: list[int]) -> np.ndarray:
    side = int(2 * len(sorted_colors) - 1)
    out = np.zeros((side, side), dtype=int)
    for layer, color in enumerate(sorted_colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color fragments, reconstruct each hidden rectangle, then stack the colors as concentric rings by area.

    Input structure:
    - A dominant background color fills most of the grid.
    - The foreground consists of monochromatic 4-connected fragments scattered apart on the background.
    - Each foreground color appears in exactly two separate fragments, and those two fragments are complementary parts of one axis-aligned solid rectangle.

    Rule:
    - Group foreground components by color and take each same-color pair.
    - For each pair, find the unique relative translation, without rotation or reflection, that makes the two fragments fit together with no overlap into one filled rectangle.
    - Record the rectangle area for every color pair, sort the pairs from largest reconstructed rectangle to smallest, and draw their colors as nested 1-pixel rings.
    - The output is a square of side 2*n-1 where n is the number of color pairs; the smallest pair fills the center.

    Key insight:
    - The visible objects are only rectangle fragments; the real ordering signal is the area of the hidden full rectangle recovered by complementary translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components(g, bg)
    assert comps, "expected at least one foreground component"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(_crop_coords(comp["coords"]))

    items: list[tuple[int, int]] = []
    for color, pieces in by_color.items():
        assert len(pieces) == 2, "each color must form exactly one fragment pair"
        a, b = pieces
        candidates = _rectangle_candidates(
            a["coords"], b["coords"], a["height"], a["width"], b["height"], b["width"]
        )
        assert len(candidates) == 1, "fragment pair must reconstruct one unique rectangle"
        items.append((int(candidates[0]["area"]), int(color)))

    areas = [area for area, _color in items]
    assert len(areas) == len(set(areas)), "reconstructed rectangle areas must be unique"
    items.sort(key=lambda item: int(item[0]), reverse=True)
    return _build_output([int(color) for _area, color in items])


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    sr, sc = map(int, coords[0])
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in _DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return int(count) == int(mask.sum())


def _random_connected_subset(h: int, w: int, size: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    sr = int(rng.integers(0, h))
    sc = int(rng.integers(0, w))
    mask[sr, sc] = True
    frontier: set[tuple[int, int]] = set()
    for dr, dc in _DIR4:
        nr = int(sr + dr)
        nc = int(sc + dc)
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < int(size):
        frontier = {cell for cell in frontier if not bool(mask[cell[0], cell[1]])}
        if not frontier:
            return None
        pick = list(frontier)[int(rng.integers(0, len(frontier)))]
        pr, pc = map(int, pick)
        mask[pr, pc] = True
        for dr, dc in _DIR4:
            nr = int(pr + dr)
            nc = int(pc + dc)
            if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                frontier.add((nr, nc))
    return mask


def _bbox_area(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]


def _transform_pair(mask_a: np.ndarray, mask_b: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    out_a = np.rot90(mask_a, k)
    out_b = np.rot90(mask_b, k)
    if flip:
        out_a = np.fliplr(out_a)
        out_b = np.fliplr(out_b)
    return out_a, out_b


def _make_pair_masks(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    area = int(h * w)
    lo = max(2, int(math.floor(area * 0.3)))
    hi = min(area - 2, int(math.ceil(area * 0.7)))
    if lo > hi:
        return None

    for _ in range(200):
        size_a = int(rng.integers(lo, hi + 1))
        mask_a = _random_connected_subset(h, w, size_a, rng)
        if mask_a is None:
            continue
        mask_b = ~mask_a
        if not _mask_connected(mask_b):
            continue
        if _bbox_area(mask_a) == int(mask_a.sum()) and _bbox_area(mask_b) == int(mask_b.sum()):
            continue
        ta, tb = _transform_pair(mask_a, mask_b, rng)
        ca = _crop_coords([tuple(map(int, rc)) for rc in np.argwhere(_crop_mask(ta))])
        cb = _crop_coords([tuple(map(int, rc)) for rc in np.argwhere(_crop_mask(tb))])
        candidates = _rectangle_candidates(
            ca["coords"], cb["coords"], ca["height"], ca["width"], cb["height"], cb["width"]
        )
        if len(candidates) == 1:
            return _crop_mask(ta), _crop_mask(tb)
    return None


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(mask: np.ndarray, blocked: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > blocked.shape[0] or left + w > blocked.shape[1]:
        return False
    for r, c in np.argwhere(mask):
        if bool(blocked[int(top + r), int(left + c)]):
            return False
    return True


def _place_patch(
    grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int, top: int, left: int
) -> None:
    for r, c in np.argwhere(mask):
        rr = int(top + r)
        cc = int(left + c)
        grid[rr, cc] = int(color)
        r0 = max(0, rr - 1)
        r1 = min(blocked.shape[0], rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(blocked.shape[1], cc + 2)
        blocked[r0:r1, c0:c1] = True


def _select_rectangles(pair_count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    dims_pool = [(h, w) for h in range(2, 8) for w in range(3, 8) if 8 <= h * w <= 28]
    dims_pool += [(h, 2) for h in range(4, 8)]
    dims_pool = list({(int(h), int(w)) for h, w in dims_pool})
    rng.shuffle(dims_pool)

    chosen: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for h, w in dims_pool:
        area = int(h * w)
        if area in used_areas:
            continue
        chosen.append((int(h), int(w)))
        used_areas.add(area)
        if len(chosen) == int(pair_count):
            break
    return chosen


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    colors = [c for c in range(10) if c != bg]
    pair_count = int(rng.integers(2, min(6, len(colors)) + 1))
    rng.shuffle(colors)
    pair_colors = colors[:pair_count]

    rect_dims = _select_rectangles(pair_count, rng)
    if len(rect_dims) != pair_count:
        return None

    pieces: list[dict] = []
    areas_by_color: dict[int, int] = {}
    for color, (h, w) in zip(pair_colors, rect_dims):
        pair_masks = _make_pair_masks(h, w, rng)
        if pair_masks is None:
            return None
        mask_a, mask_b = pair_masks
        areas_by_color[int(color)] = int(h * w)
        pieces.append({"color": int(color), "mask": mask_a, "fg_area": int(mask_a.sum())})
        pieces.append({"color": int(color), "mask": mask_b, "fg_area": int(mask_b.sum())})

    padded_area = sum((piece["mask"].shape[0] + 2) * (piece["mask"].shape[1] + 2) for piece in pieces)
    base = max(max(piece["mask"].shape[0] for piece in pieces), max(piece["mask"].shape[1] for piece in pieces))
    side = max(base + 2, int(math.ceil(math.sqrt(padded_area * 1.45))))
    lo = max(14, side)
    hi = min(30, side + 5)
    if lo > hi:
        return None
    h = int(rng.integers(lo, hi + 1))
    w = int(rng.integers(lo, hi + 1))

    order = sorted(range(len(pieces)), key=lambda idx: int(pieces[idx]["fg_area"]), reverse=True)
    grid = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    for idx in order:
        piece = pieces[idx]
        mask = piece["mask"]
        ph, pw = mask.shape
        placed = False
        for _ in range(250):
            top = int(rng.integers(0, h - ph + 1))
            left = int(rng.integers(0, w - pw + 1))
            if _can_place(mask, blocked, top, left):
                _place_patch(grid, blocked, mask, int(piece["color"]), top, left)
                placed = True
                break
        if not placed:
            return None

    if int(np.count_nonzero(grid == bg)) <= int(grid.size // 2):
        return None

    comps = _components(grid, bg)
    if len(comps) != 2 * pair_count:
        return None

    by_color = Counter(int(comp["color"]) for comp in comps)
    if any(int(by_color[color]) != 2 for color in pair_colors):
        return None

    expected_colors = [color for color, _area in sorted(areas_by_color.items(), key=lambda item: int(item[1]), reverse=True)]
    expected = _build_output(expected_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
