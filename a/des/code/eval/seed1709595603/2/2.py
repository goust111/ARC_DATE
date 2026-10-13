from __future__ import annotations

from collections import deque
import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _region_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _one_side(fragments: list[tuple[int, int]], bbox: tuple[int, int, int, int], vertical: bool) -> bool:
    if not fragments:
        return False
    r0, c0, r1, c1 = bbox
    axis_sum = (c0 + c1) if vertical else (r0 + r1)
    neg = False
    pos = False
    for r, c in fragments:
        coord2 = 2 * c if vertical else 2 * r
        if coord2 == axis_sum:
            return False
        if coord2 < axis_sum:
            neg = True
        else:
            pos = True
    return neg ^ pos


def _choose_axis(region: list[tuple[int, int]], fragments: list[tuple[int, int]]) -> str | None:
    mask, box = _region_mask(region)
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))

    if vertical and not horizontal:
        return "vertical"
    if horizontal and not vertical:
        return "horizontal"

    vert_side = _one_side(fragments, box, vertical=True)
    horiz_side = _one_side(fragments, box, vertical=False)

    if vertical and vert_side and not (horizontal and horiz_side):
        return "vertical"
    if horizontal and horiz_side and not (vertical and vert_side):
        return "horizontal"
    if vertical and vert_side:
        return "vertical"
    if horizontal and horiz_side:
        return "horizontal"
    return None


def _find_rings(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    rings: list[tuple[int, list[tuple[int, int]]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        wall_mask = grid == int(color)
        wall_components = _components(wall_mask)
        if not wall_components:
            continue

        open_components = _components(~wall_mask)
        for wall in wall_components:
            wall_set = set(wall)
            enclosed: list[list[tuple[int, int]]] = []
            for region in open_components:
                if _touches_boundary(region, h, w):
                    continue
                adjacent = set()
                for r, c in region:
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < h and 0 <= nc < w and bool(wall_mask[nr, nc]):
                            adjacent.add((int(nr), int(nc)))
                if adjacent and adjacent.issubset(wall_set):
                    enclosed.append(region)
            if enclosed:
                rings.append((int(color), max(enclosed, key=len)))
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror fragments inside each colored ring across the ring's own symmetry axis, then recolor
    all fragment pixels to the surrounding ring color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for ring_color, region in _find_rings(g, bg):
        fragments = [(r, c) for r, c in region if int(g[r, c]) != int(bg)]
        if not fragments:
            continue

        axis = _choose_axis(region, fragments)
        if axis is None:
            continue

        region_set = set(region)
        r0, c0, r1, c1 = _bbox(region)
        for r, c in fragments:
            out[r, c] = int(ring_color)
            if axis == "vertical":
                rr, cc = int(r), int(c0 + c1 - c)
            else:
                rr, cc = int(r0 + r1 - r), int(c)
            if (rr, cc) in region_set:
                out[rr, cc] = int(ring_color)

    return out


BG = 0
FRAGMENT_COLORS = (4, 6)
RING_COLORS = (1, 2, 3, 5, 7, 8, 9)
GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (22, 30)
RING_COUNT_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (5, 8)
INNER_WIDTH_RANGE = (5, 9)
PLACEMENT_GAP = 2
GEN_RETRIES = 400
SHAPE_RETRIES = 100
PLACEMENT_RETRIES = 200
FRAGMENT_RETRIES = 120


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr0 = max(0, dr)
            rr1 = min(h, h + dr)
            cc0 = max(0, dc)
            cc1 = min(w, w + dc)
            src_r0 = max(0, -dr)
            src_r1 = src_r0 + (rr1 - rr0)
            src_c0 = max(0, -dc)
            src_c1 = src_c0 + (cc1 - cc0)
            out[rr0:rr1, cc0:cc1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not bool(mask[r, c]):
                continue
            if all(bool(mask[nr, nc]) for nr, nc in _neighbors4(r, c)):
                out[r, c] = True
    return out


def _row_widths(height: int, width: int) -> list[int] | None:
    choices = [v for v in range(max(3, width // 2), width + 1) if v % 2 == width % 2]
    if len(choices) < 2:
        return None
    for _ in range(SHAPE_RETRIES):
        widths = [random.choice(choices) for _ in range(height)]
        if widths == widths[::-1]:
            continue
        if len(set(widths)) < 2:
            continue
        return widths
    return None


def _inner_mask(orientation: str) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(*INNER_HEIGHT_RANGE)
        w = random.randint(*INNER_WIDTH_RANGE)
        widths = _row_widths(h, w)
        if widths is None:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, row_w in enumerate(widths):
            c0 = (w - row_w) // 2
            mask[r, c0:c0 + row_w] = True

        if not bool(np.array_equal(mask, np.fliplr(mask))):
            continue
        if bool(np.array_equal(mask, np.flipud(mask))):
            continue

        if orientation == "vertical":
            return mask
        transposed = mask.T.copy()
        if bool(np.array_equal(transposed, np.flipud(transposed))) and not bool(
            np.array_equal(transposed, np.fliplr(transposed))
        ):
            return transposed
    return None


def _source_candidates(region: np.ndarray, orientation: str, source_side: str) -> list[tuple[int, int]]:
    points = [(int(r), int(c)) for r, c in np.argwhere(region)]
    r0, c0, r1, c1 = _bbox(points)
    core = _erode4(region)
    base = core if int(core.sum()) >= 2 else region
    pts = []
    for r, c in np.argwhere(base):
        rr = int(r + r0)
        cc = int(c + c0)
        if orientation == "vertical":
            if source_side == "negative" and 2 * cc < c0 + c1:
                pts.append((int(rr), int(cc)))
            if source_side == "positive" and 2 * cc > c0 + c1:
                pts.append((int(rr), int(cc)))
        else:
            if source_side == "negative" and 2 * rr < r0 + r1:
                pts.append((int(rr), int(cc)))
            if source_side == "positive" and 2 * rr > r0 + r1:
                pts.append((int(rr), int(cc)))
    return pts


def _place_fragments(region: np.ndarray, orientation: str) -> dict[tuple[int, int], int] | None:
    for _ in range(FRAGMENT_RETRIES):
        source_side = random.choice(("negative", "positive"))
        candidates = _source_candidates(region, orientation, source_side)
        if len(candidates) < 2:
            continue

        target = random.randint(2, min(6, len(candidates)))
        used: set[tuple[int, int]] = {random.choice(candidates)}

        while len(used) < target:
            grow_from = random.choice(tuple(used))
            local = [
                (nr, nc)
                for nr, nc in _neighbors4(*grow_from)
                if (nr, nc) in candidates and (nr, nc) not in used
            ]
            if local and random.random() < 0.7:
                used.add(random.choice(local))
                continue

            far = [
                cell
                for cell in candidates
                if cell not in used
                and all(abs(cell[0] - r) + abs(cell[1] - c) >= 2 for r, c in used)
            ]
            pool = far or [cell for cell in candidates if cell not in used]
            if not pool:
                break
            used.add(random.choice(pool))

        if len(used) < 2:
            continue

        palette = list(FRAGMENT_COLORS if random.random() < 0.55 else (random.choice(FRAGMENT_COLORS),))
        colors: dict[tuple[int, int], int] = {}
        cells = list(used)
        random.shuffle(cells)
        for idx, cell in enumerate(cells):
            if len(palette) == 2 and idx < 2:
                colors[cell] = int(palette[idx])
            else:
                colors[cell] = int(random.choice(palette))
        return colors
    return None


def _mirror_cell(cell: tuple[int, int], region: np.ndarray, orientation: str) -> tuple[int, int]:
    points = [(int(r), int(c)) for r, c in np.argwhere(region)]
    r0, c0, r1, c1 = _bbox(points)
    r, c = cell
    if orientation == "vertical":
        return int(r), int(c0 + c1 - c)
    return int(r0 + r1 - r), int(c)


def _build_ring_patch(ring_color: int, orientation: str) -> tuple[np.ndarray, np.ndarray] | None:
    inner = _inner_mask(orientation)
    if inner is None:
        return None

    region = np.pad(inner, 1, constant_values=False)
    outer = _dilate8(region)
    wall = outer & ~region

    fragments = _place_fragments(region, orientation)
    if fragments is None:
        return None

    inp = np.full(region.shape, BG, dtype=int)
    out = np.full(region.shape, BG, dtype=int)
    inp[wall] = int(ring_color)
    out[wall] = int(ring_color)

    region_set = {tuple(map(int, pt)) for pt in np.argwhere(region)}
    for cell, color in fragments.items():
        inp[cell] = int(color)
        out[cell] = int(ring_color)
        mirrored = _mirror_cell(cell, region, orientation)
        if mirrored not in region_set:
            return None
        out[mirrored] = int(ring_color)

    return inp, out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = dst[top:top + h, left:left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    ring_count = random.randint(*RING_COUNT_RANGE)

    colors = random.sample(RING_COLORS, ring_count)
    orientations = ["vertical", "horizontal"]
    while len(orientations) < ring_count:
        orientations.append(random.choice(("vertical", "horizontal")))
    random.shuffle(orientations)

    patches = []
    for color, orientation in zip(colors, orientations, strict=False):
        patch = _build_ring_patch(int(color), orientation)
        if patch is None:
            return None
        inp_patch, out_patch = patch
        patches.append((inp_patch, out_patch))

    random.shuffle(patches)
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    placed: list[tuple[int, int, int, int]] = []

    if sum(p[0].shape[0] * p[0].shape[1] for p in patches) > (h * w * 3) // 5:
        return None

    for inp_patch, out_patch in patches:
        ph, pw = inp_patch.shape
        if ph >= h or pw >= w:
            return None
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            box = (int(top), int(left), int(top + ph - 1), int(left + pw - 1))
            if any(_boxes_overlap(box, other, PLACEMENT_GAP) for other in placed):
                continue
            _paste(inp, inp_patch, top, left)
            _paste(out, out_patch, top, left)
            placed.append(box)
            break
        else:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _generate_pair()
        if pair is None:
            continue
        inp, expected = pair
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid ring-mirror input")
