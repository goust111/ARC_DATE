from __future__ import annotations

from collections import deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _segment_info(coords: np.ndarray) -> tuple[str, tuple[tuple[int, int], tuple[int, int]]] | None:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1:
        r = int(rows[0])
        c0 = int(cols.min())
        c1 = int(cols.max())
        if int(coords.shape[0]) != c1 - c0 + 1:
            return None
        return "h", ((r, c0), (r, c1))
    if len(cols) == 1:
        c = int(cols[0])
        r0 = int(rows.min())
        r1 = int(rows.max())
        if int(coords.shape[0]) != r1 - r0 + 1:
            return None
        return "v", ((r0, c), (r1, c))
    return None


def _draw_axis_line(mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        a, b = sorted((c0, c1))
        mask[r0, a : b + 1] = True
        return
    assert c0 == c1, "expected an axis-aligned line"
    a, b = sorted((r0, r1))
    mask[a : b + 1, c0] = True


def _boundary_is_single_cycle(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False

    for r, c in pts:
        r = int(r)
        c = int(c)
        deg = 0
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                deg += 1
        if deg != 2:
            return False

    start = tuple(map(int, pts[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (int(nr), int(nc))
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _fill_holes(boundary: np.ndarray) -> np.ndarray:
    open_space = ~boundary
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
    return open_space & ~seen[1:-1, 1:-1]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    return r0, c0, r1, c1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _solve_color(grid: np.ndarray, color: int, bg: int) -> np.ndarray | None:
    mask = grid == int(color)
    components = _connected_components(mask)
    if not (3 <= len(components) <= 5):
        return None

    infos: list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]] = []
    for comp in components:
        info = _segment_info(comp)
        if info is None:
            return None
        infos.append(info)

    orientations = {orient for orient, _ends in infos}
    if len(orientations) != 1:
        return None
    orient = infos[0][0]

    endpoints: list[tuple[int, tuple[int, int]]] = []
    for seg_idx, (_orient, ends) in enumerate(infos):
        endpoints.append((seg_idx, ends[0]))
        endpoints.append((seg_idx, ends[1]))

    used = [False] * len(endpoints)
    pairs: list[tuple[int, int]] = []
    unique: dict[bytes, np.ndarray] = {}

    def validate_current_pairs() -> None:
        boundary = mask.copy()
        for i, j in pairs:
            _seg_i, p0 = endpoints[i]
            _seg_j, p1 = endpoints[j]
            if orient == "h":
                if p0[1] != p1[1]:
                    return
            else:
                if p0[0] != p1[0]:
                    return
            segment_mask = np.zeros_like(boundary)
            _draw_axis_line(segment_mask, p0, p1)
            crossed = grid[segment_mask]
            if np.any((crossed != int(bg)) & (crossed != int(color))):
                return
            boundary |= segment_mask

        if not _boundary_is_single_cycle(boundary):
            return

        holes = _fill_holes(boundary)
        if not bool(holes.any()):
            return
        if np.any(grid[holes] != int(bg)):
            return

        filled = boundary | holes
        unique[filled.tobytes()] = filled

    def search() -> None:
        try:
            i = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            validate_current_pairs()
            return

        used[i] = True
        seg_i, p0 = endpoints[i]
        key = p0[1] if orient == "h" else p0[0]
        for j in range(i + 1, len(endpoints)):
            if used[j]:
                continue
            seg_j, p1 = endpoints[j]
            if seg_i == seg_j:
                continue
            other_key = p1[1] if orient == "h" else p1[0]
            if key != other_key:
                continue
            used[j] = True
            pairs.append((i, j))
            search()
            pairs.pop()
            used[j] = False
        used[i] = False

    search()
    if len(unique) != 1:
        return None
    return next(iter(unique.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group each color's scattered segments, reconnect their endpoints into a closed
    orthogonal polygon, and fill that polygon's interior with the same color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in (int(v) for v in np.unique(g) if int(v) != int(bg)):
        filled = _solve_color(g, color, bg)
        if filled is None:
            continue
        out[filled] = int(color)
    return out


def _sample_local_polygon(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    orientation = rng.choice(("h", "v"))
    segment_count = rng.randint(3, 5)

    for _ in range(400):
        height = rng.randint(segment_count * 2 + 1, segment_count * 3 + 4)
        rows = rng.sample(range(height), segment_count)
        if any(abs(rows[(i + 1) % segment_count] - rows[i]) < 2 for i in range(segment_count)):
            continue

        width = rng.randint(5, 8)
        cols = [rng.randrange(width) for _ in range(segment_count)]
        if any(not (1 <= abs(cols[i] - cols[i - 1]) <= 4) for i in range(segment_count)):
            continue

        boundary = np.zeros((height, width), dtype=bool)
        observed = np.zeros((height, width), dtype=bool)
        for i in range(segment_count):
            r = int(rows[i])
            c0 = int(cols[i - 1])
            c1 = int(cols[i])
            _draw_axis_line(boundary, (r, c0), (r, c1))
            _draw_axis_line(observed, (r, c0), (r, c1))

            r2 = int(rows[(i + 1) % segment_count])
            _draw_axis_line(boundary, (r, c1), (r2, c1))

        if not _boundary_is_single_cycle(boundary):
            continue

        holes = _fill_holes(boundary)
        if not bool(holes.any()):
            continue

        components = _connected_components(observed)
        if len(components) != segment_count:
            continue

        infos = [_segment_info(comp) for comp in components]
        if any(info is None for info in infos):
            continue
        if {info[0] for info in infos if info is not None} != {"h"}:
            continue
        lengths = []
        for info in infos:
            assert info is not None
            (r0, c0), (r1, c1) = info[1]
            lengths.append(abs(r1 - r0) + abs(c1 - c0) + 1)
        if not all(2 <= length <= 5 for length in lengths):
            continue

        inp = observed
        out = boundary | holes
        if orientation == "v":
            inp = inp.T
            out = out.T
        if rng.random() < 0.5:
            inp = np.flipud(inp)
            out = np.flipud(out)
        if rng.random() < 0.5:
            inp = np.fliplr(inp)
            out = np.fliplr(out)

        r0, c0, r1, c1 = _mask_bbox(out)
        out = out[r0 : r1 + 1, c0 : c1 + 1]
        inp = inp[r0 : r1 + 1, c0 : c1 + 1]
        return inp.copy(), out.copy()
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    view = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    view[mask] = int(color)


def _try_place_objects(
    rng: random.Random,
    shape: tuple[int, int],
    objects: list[tuple[np.ndarray, np.ndarray, int]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    padding = 2

    for observed_mask, filled_mask, color in objects:
        ph, pw = observed_mask.shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        rng.shuffle(positions)
        placed = False
        for r, c in positions:
            rr0 = max(0, r - padding)
            rr1 = min(h, r + ph + padding)
            cc0 = max(0, c - padding)
            cc1 = min(w, c + pw + padding)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            _paint_mask(inp, observed_mask, r, c, color)
            _paint_mask(out, filled_mask, r, c, color)
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    object_count = rng.randint(1, 3)
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)

    objects: list[tuple[np.ndarray, np.ndarray, int]] = []
    for idx in range(object_count):
        patch = _sample_local_polygon(rng)
        if patch is None:
            return None
        observed_mask, filled_mask = patch
        objects.append((observed_mask, filled_mask, int(colors[idx])))

    max_h = max(mask.shape[0] for mask, _filled, _color in objects)
    max_w = max(mask.shape[1] for mask, _filled, _color in objects)
    h_min = max(18, max_h + 2)
    w_min = max(18, max_w + 2)
    h_max = max(h_min, 32)
    w_max = max(w_min, 32)

    for _ in range(80):
        h = rng.randint(h_min, h_max)
        w = rng.randint(w_min, w_max)
        placed = _try_place_objects(rng, (h, w), objects, bg)
        if placed is None:
            continue
        inp, out = placed
        if _get_bg(inp) != int(bg):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
