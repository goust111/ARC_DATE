from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
FENCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
REGION_COUNT_RANGE = (1, 4)  # high is exclusive
REGION_H_RANGE = (8, 13)
REGION_W_RANGE = (8, 13)
MAX_GAPS = 4
TIE_CASE_PROB = 0.35
GEN_RETRIES = 400
REGION_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _find_fences(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    fences: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue

        rel_r = coords[:, 0] - r0
        rel_c = coords[:, 1] - c0
        on_border = (rel_r == 0) | (rel_r == h - 1) | (rel_c == 0) | (rel_c == w - 1)
        if not bool(np.all(on_border)):
            continue

        perimeter = 2 * (h + w) - 4
        coverage = len(coords) / perimeter
        if coverage < 0.65:
            continue

        if not (
            np.any(coords[:, 0] == r0)
            and np.any(coords[:, 0] == r1)
            and np.any(coords[:, 1] == c0)
            and np.any(coords[:, 1] == c1)
        ):
            continue

        fences.append((r0, c0, r1, c1, color))

    fences.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each colored fence, count its missing border pixels, and remove that many
    smallest interior objects.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1, color in _find_fences(g):
        gap_count = sum(1 for r, c in _perimeter_positions(r0, c0, r1, c1) if int(g[r, c]) != color)
        if gap_count <= 0:
            continue

        inner_mask = g[r0 + 1 : r1, c0 + 1 : c1] != BG
        objects = []
        for comp in _components(inner_mask):
            pts = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in comp]
            top = min(r for r, _c in pts)
            left = min(c for _r, c in pts)
            objects.append((len(pts), top, left, pts))

        objects.sort(key=lambda item: (item[0], item[1], item[2]))
        for _area, _top, _left, pts in objects[:gap_count]:
            for r, c in pts:
                out[r, c] = BG

    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _grow_polyomino(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _make_object_patch(rng: np.random.Generator, area: int) -> np.ndarray:
    dims = [(h, w) for h in range(1, 4) for w in range(1, 4) if h * w >= area]
    h, w = dims[int(rng.integers(0, len(dims)))]
    mask = _grow_polyomino(rng, h, w, area)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = GREY
    return patch


def _find_box_positions(
    occ: np.ndarray,
    ph: int,
    pw: int,
    pad: int,
    min_row: int | None = None,
    max_row: int | None = None,
) -> list[tuple[int, int]]:
    h, w = occ.shape
    lo = 0 if min_row is None else max(0, min_row)
    hi = h - ph if max_row is None else min(h - ph, max_row)
    positions: list[tuple[int, int]] = []
    for r in range(lo, hi + 1):
        for c in range(w - pw + 1):
            rr0 = max(0, r - pad)
            rr1 = min(h, r + ph + pad)
            cc0 = max(0, c - pad)
            cc1 = min(w, c + pw + pad)
            if not bool(occ[rr0:rr1, cc0:cc1].any()):
                positions.append((r, c))
    return positions


def _occupy_box(occ: np.ndarray, r: int, c: int, ph: int, pw: int, pad: int) -> None:
    h, w = occ.shape
    rr0 = max(0, r - pad)
    rr1 = min(h, r + ph + pad)
    cc0 = max(0, c - pad)
    cc1 = min(w, c + pw + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _sample_gaps(rng: np.random.Generator, h: int, w: int, gap_count: int) -> set[tuple[int, int]]:
    choices = {
        "top": [(0, c) for c in range(1, w - 1)],
        "bottom": [(h - 1, c) for c in range(1, w - 1)],
        "left": [(r, 0) for r in range(1, h - 1)],
        "right": [(r, w - 1) for r in range(1, h - 1)],
    }
    side_names = ["top", "bottom", "left", "right"]
    selected = [side_names[i] for i in rng.permutation(len(side_names))[:gap_count]]
    return {choices[side][int(rng.integers(0, len(choices[side])))] for side in selected}


def _build_region(
    rng: np.random.Generator,
    color: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(REGION_RETRIES):
        h = int(rng.integers(*REGION_H_RANGE))
        w = int(rng.integers(*REGION_W_RANGE))
        inner_h = h - 4
        inner_w = w - 4
        if inner_h < 1 or inner_w < 1:
            continue

        tie_case = bool(rng.random() < TIE_CASE_PROB and inner_h >= 4)
        if tie_case:
            gap_count = 1
            object_count = int(rng.integers(3, 6))
            smallest = int(rng.integers(1, 3))
            areas = [smallest, smallest]
            for _idx in range(object_count - 2):
                areas.append(int(rng.integers(smallest + 1, 6)))
        else:
            gap_count = int(rng.integers(1, MAX_GAPS + 1))
            extra = int(rng.integers(0, 4))
            object_count = gap_count + extra
            if object_count == 0:
                object_count = gap_count
            if object_count == 0:
                continue
            areas = [int(rng.integers(1, 6)) for _idx in range(object_count)]

        object_patches = [_make_object_patch(rng, area) for area in areas]
        if any(p.shape[0] > inner_h or p.shape[1] > inner_w for p in object_patches):
            continue

        occ = np.zeros((inner_h, inner_w), dtype=bool)
        placements: list[tuple[np.ndarray, int, int]] = []
        failed = False

        start_idx = 0
        if tie_case:
            upper = object_patches[0]
            lower = object_patches[1]
            upper_max = max(0, inner_h // 3)
            upper_pos = _find_box_positions(occ, upper.shape[0], upper.shape[1], pad=1, max_row=upper_max)
            if not upper_pos:
                continue
            ur, uc = upper_pos[int(rng.integers(0, len(upper_pos)))]
            placements.append((upper, ur, uc))
            _occupy_box(occ, ur, uc, upper.shape[0], upper.shape[1], pad=1)

            lower_min = min(inner_h - lower.shape[0], max(1, inner_h // 2))
            lower_pos = _find_box_positions(
                occ,
                lower.shape[0],
                lower.shape[1],
                pad=1,
                min_row=lower_min,
            )
            if not lower_pos:
                continue
            lr, lc = lower_pos[int(rng.integers(0, len(lower_pos)))]
            placements.append((lower, lr, lc))
            _occupy_box(occ, lr, lc, lower.shape[0], lower.shape[1], pad=1)
            start_idx = 2

        for patch in object_patches[start_idx:]:
            pos = _find_box_positions(occ, patch.shape[0], patch.shape[1], pad=1)
            if not pos:
                failed = True
                break
            r, c = pos[int(rng.integers(0, len(pos)))]
            placements.append((patch, r, c))
            _occupy_box(occ, r, c, patch.shape[0], patch.shape[1], pad=1)
        if failed:
            continue

        infos = []
        seen_area_rows = set()
        for patch, r, c in placements:
            local = np.argwhere(patch == GREY)
            pts = [(2 + r + int(rr), 2 + c + int(cc)) for rr, cc in local]
            top = min(rr for rr, _cc in pts)
            left = min(cc for _rr, cc in pts)
            area = len(pts)
            key = (area, top)
            if key in seen_area_rows:
                failed = True
                break
            seen_area_rows.add(key)
            infos.append((area, top, left, pts))
        if failed:
            continue

        patch = np.zeros((h, w), dtype=int)
        gaps = _sample_gaps(rng, h, w, gap_count)
        for r, c in _perimeter_positions(0, 0, h - 1, w - 1):
            if (r, c) not in gaps:
                patch[r, c] = color
        for obj_patch, r, c in placements:
            _paste_nonzero(patch, obj_patch, 2 + r, 2 + c)

        expected = patch.copy()
        infos.sort(key=lambda item: (item[0], item[1], item[2]))
        for _area, _top, _left, pts in infos[:gap_count]:
            for rr, cc in pts:
                expected[rr, cc] = BG

        if np.array_equal(main(patch), expected):
            return patch, expected

    return None


def _build_full_grid(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    region_count = int(rng.integers(*REGION_COUNT_RANGE))
    colors = rng.choice(np.array(FENCE_COLORS), size=region_count, replace=False)
    regions = []
    for color in colors:
        built = _build_region(rng, int(color))
        if built is None:
            return None
        regions.append(built)

    regions.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)

    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    for inp_patch, out_patch in regions:
        ph, pw = inp_patch.shape
        positions = _find_box_positions(occ, ph, pw, pad=2)
        if not positions:
            return None
        r, c = positions[int(rng.integers(0, len(positions)))]
        _paste_nonzero(inp, inp_patch, r, c)
        _paste_nonzero(out, out_patch, r, c)
        _occupy_box(occ, r, c, ph, pw, pad=2)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        built = _build_full_grid(rng)
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
