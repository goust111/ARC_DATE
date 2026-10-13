from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
MAGENTA = 6

DARK_TO_BRIGHT = {
    1: 8,  # blue -> azure
    3: 4,  # green -> yellow
    9: 7,  # maroon -> orange
}
BRIGHT_TO_DARK = {bright: dark for dark, bright in DARK_TO_BRIGHT.items()}
DARK_COLORS = tuple(DARK_TO_BRIGHT)
BRIGHT_COLORS = tuple(BRIGHT_TO_DARK)
MARKER_COLORS = (RED, MAGENTA)
SYMMETRIES = ("H", "V", "D")

PACK_GAP = 1
GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (2, 4)
CANVAS_RANGE = (24, 32)
REGION_SIZE_RANGE = (5, 9)
BOTTOM_ATTEMPTS = 80
TOP_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 80


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _points_to_mask(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    return len(_components(mask)) == 1


def _transform(sym: str, h: int, w: int, r: int, c: int) -> tuple[int, int]:
    if sym == "H":
        return h - 1 - r, c
    if sym == "V":
        return r, w - 1 - c
    if sym == "D":
        assert h == w
        return c, r
    raise ValueError(sym)


def _mask_is_symmetric(mask: np.ndarray, sym: str) -> bool:
    h, w = mask.shape
    if sym == "H":
        return bool(np.array_equal(mask, mask[::-1, :]))
    if sym == "V":
        return bool(np.array_equal(mask, mask[:, ::-1]))
    if sym == "D":
        return h == w and bool(np.array_equal(mask, mask.T))
    raise ValueError(sym)


def _valid_candidates(visible_dark: np.ndarray, non_black: np.ndarray) -> list[str]:
    h, w = visible_dark.shape
    coords = np.argwhere(visible_dark)
    candidates: list[str] = []
    for sym in SYMMETRIES:
        if sym == "D" and h != w:
            continue
        ok = True
        for r, c in coords:
            rr, cc = _transform(sym, h, w, int(r), int(c))
            if not non_black[rr, cc]:
                ok = False
                break
        if ok:
            candidates.append(sym)
    return candidates


def _restore_bottom(visible_dark: np.ndarray, sym: str) -> np.ndarray:
    h, w = visible_dark.shape
    restored = visible_dark.copy()
    for r, c in np.argwhere(visible_dark):
        rr, cc = _transform(sym, h, w, int(r), int(c))
        restored[rr, cc] = True
    return restored


def _crop_colored(mask: np.ndarray, color: int) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    out[mask[r0 : r1 + 1, c0 : c1 + 1]] = int(color)
    return out


def _pack_pieces(pieces: list[np.ndarray]) -> np.ndarray:
    assert pieces
    height = max(int(piece.shape[0]) for piece in pieces)
    width = sum(int(piece.shape[1]) for piece in pieces) + PACK_GAP * (len(pieces) - 1)
    out = np.full((height, width), BLACK, dtype=int)
    col = 0
    for piece in pieces:
        ph, pw = map(int, piece.shape)
        out[:ph, col : col + pw] = piece
        col += pw + PACK_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden symmetric bottom layer inside each region, then output the
    layer selected by the external marker.
    """
    g = np.asarray(grid, dtype=int)
    content_mask = (g != BLACK) & (g != RED) & (g != MAGENTA)
    comps = _components(content_mask)
    assert comps, "expected at least one pattern region"

    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere((g == RED) | (g == MAGENTA))]
    used_markers: set[tuple[int, int]] = set()
    items: list[tuple[tuple[int, int], np.ndarray]] = []

    for comp in comps:
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
        region = g[r0 : r1 + 1, c0 : c1 + 1]

        marker_hits = []
        for mr, mc in marker_positions:
            if (mr, mc) in used_markers:
                continue
            touches_side = (
                (mr == r0 - 1 and c0 <= mc <= c1)
                or (mr == r1 + 1 and c0 <= mc <= c1)
                or (mc == c0 - 1 and r0 <= mr <= r1)
                or (mc == c1 + 1 and r0 <= mr <= r1)
            )
            if touches_side:
                marker_hits.append((mr, mc))
        assert len(marker_hits) == 1, "each region must have exactly one adjacent marker"
        marker = int(g[marker_hits[0]])
        used_markers.add(marker_hits[0])

        colors = set(map(int, region[region != BLACK].tolist()))
        darks = [color for color in colors if color in DARK_COLORS]
        brights = [color for color in colors if color in BRIGHT_COLORS]
        assert len(darks) == 1 and len(brights) == 1, "each region must contain one dark/bright pair"
        dark = darks[0]
        bright = brights[0]
        assert DARK_TO_BRIGHT[dark] == bright, "unexpected layer color pairing"

        visible_dark = region == dark
        bright_mask = region == bright
        candidates = _valid_candidates(visible_dark, region != BLACK)
        assert len(candidates) == 1, "bottom symmetry must be uniquely determined"
        restored_bottom = _restore_bottom(visible_dark, candidates[0])

        if marker == RED:
            piece = _crop_colored(restored_bottom, dark)
        else:
            assert marker == MAGENTA
            piece = _crop_colored(bright_mask, bright)
        items.append(((r0, c0), piece))

    assert len(used_markers) == len(marker_positions), "unmatched marker detected"
    items.sort(key=lambda item: item[0])
    return _pack_pieces([piece for _key, piece in items])


def _domain_mask(sym: str, h: int, w: int) -> np.ndarray:
    if sym == "H":
        last = h // 2 if h % 2 else h // 2 - 1
        mask = np.zeros((h, w), dtype=bool)
        mask[: last + 1, :] = True
        return mask
    if sym == "V":
        last = w // 2 if w % 2 else w // 2 - 1
        mask = np.zeros((h, w), dtype=bool)
        mask[:, : last + 1] = True
        return mask
    assert sym == "D" and h == w
    rr, cc = np.indices((h, w))
    return rr <= cc


def _dedupe_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for point in points:
        if point not in seen:
            seen.add(point)
            out.append(point)
    return out


def _required_cells(sym: str, h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    if sym == "H":
        last = h // 2 if h % 2 else h // 2 - 1
        points = [
            (0, rng.randrange(w)),
            (rng.randrange(last + 1), 0),
            (rng.randrange(last + 1), w - 1),
            (last, rng.randrange(w)),
        ]
        return _dedupe_points(points)
    if sym == "V":
        last = w // 2 if w % 2 else w // 2 - 1
        points = [
            (0, rng.randrange(last + 1)),
            (h - 1, rng.randrange(last + 1)),
            (rng.randrange(h), 0),
            (rng.randrange(h), last),
        ]
        return _dedupe_points(points)
    assert sym == "D" and h == w
    d = rng.randrange(h)
    points = [
        (0, rng.randrange(w)),
        (rng.randrange(h), w - 1),
        (d, d),
    ]
    return _dedupe_points(points)


def _neighbors_in_domain(domain: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = domain.shape
    out: list[tuple[int, int]] = []
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w and domain[nr, nc]:
            out.append((nr, nc))
    return out


def _path_to_set(
    domain: np.ndarray,
    start: tuple[int, int],
    targets: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if start in targets:
        return [start]
    q = deque([start])
    parent = {start: None}
    hit: tuple[int, int] | None = None
    while q and hit is None:
        r, c = q.popleft()
        for nxt in _neighbors_in_domain(domain, r, c):
            if nxt in parent:
                continue
            parent[nxt] = (r, c)
            if nxt in targets:
                hit = nxt
                break
            q.append(nxt)
    if hit is None:
        return None

    path = [hit]
    cur = hit
    while cur != start:
        cur = parent[cur]
        assert cur is not None
        path.append(cur)
    path.reverse()
    return path


def _grow_connected_set(
    domain: np.ndarray,
    required: list[tuple[int, int]],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = {required[0]}
    for point in required[1:]:
        if point in cells:
            continue
        path = _path_to_set(domain, point, cells)
        if path is None:
            return None
        cells.update(path)

    target = max(target_size, len(cells))
    available = int(domain.sum())
    target = min(target, available)

    while len(cells) < target:
        frontier = {
            nbr
            for r, c in cells
            for nbr in _neighbors_in_domain(domain, r, c)
            if nbr not in cells
        }
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))
    return cells


def _orbit_for_cell(sym: str, h: int, w: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    rr, cc = _transform(sym, h, w, r, c)
    if (rr, cc) == (r, c):
        return ((r, c),)
    pair = sorted(((r, c), (rr, cc)))
    return tuple(pair)


def _mask_orbits(mask: np.ndarray, sym: str) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        point = (int(r), int(c))
        if point in seen:
            continue
        orbit = _orbit_for_cell(sym, h, w, point[0], point[1])
        assert all(mask[rr, cc] for rr, cc in orbit)
        seen.update(orbit)
        orbits.append(orbit)
    return orbits


def _random_bottom_mask(sym: str, h: int, w: int, rng: random.Random) -> np.ndarray | None:
    domain = _domain_mask(sym, h, w)
    required = _required_cells(sym, h, w, rng)
    domain_area = int(domain.sum())

    for _ in range(BOTTOM_ATTEMPTS):
        extra_cap = max(2, domain_area // 3)
        target = len(required) + rng.randint(1, extra_cap)
        base = _grow_connected_set(domain, required, target, rng)
        if base is None:
            continue
        full = set(base)
        for r, c in tuple(base):
            full.add(_transform(sym, h, w, r, c))
        mask = _points_to_mask(full, (h, w))
        if not _is_connected(mask):
            continue
        if int(mask.sum()) >= h * w - 1:
            continue
        if _bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        valid_syms = [cand for cand in SYMMETRIES if (cand != "D" or h == w) and _mask_is_symmetric(mask, cand)]
        if valid_syms != [sym]:
            continue
        orbits = _mask_orbits(mask, sym)
        if not any(len(orbit) == 2 for orbit in orbits):
            continue
        return mask
    return None


def _random_top_mask(bottom: np.ndarray, sym: str, rng: random.Random) -> np.ndarray | None:
    h, w = bottom.shape
    area = h * w
    domain = np.ones((h, w), dtype=bool)
    orbits = _mask_orbits(bottom, sym)
    two_cell_orbits = [orbit for orbit in orbits if len(orbit) == 2]
    outside_cells = [tuple(map(int, rc)) for rc in np.argwhere(~bottom)]
    if not two_cell_orbits or not outside_cells:
        return None

    for _ in range(TOP_ATTEMPTS):
        overlap_orbit = rng.choice(two_cell_orbits)
        overlap_cell = rng.choice(overlap_orbit)
        outside_cell = rng.choice(outside_cells)
        required = [overlap_cell, outside_cell]
        all_cells = [(r, c) for r in range(h) for c in range(w)]
        extra_required = rng.choice(all_cells)
        required.append(extra_required)
        required = _dedupe_points(required)

        target = rng.randint(max(3, len(required)), max(len(required), min(area - 1, area // 2 + 2)))
        points = _grow_connected_set(domain, required, target, rng)
        if points is None:
            continue
        top = _points_to_mask(points, (h, w))
        if not np.any(top & bottom):
            continue
        if not np.any(top & ~bottom):
            continue
        if any(_mask_is_symmetric(top, cand) for cand in SYMMETRIES if cand != "D" or h == w):
            continue

        visible = bottom & ~top
        if int(visible.sum()) < 4:
            continue
        if any(all(top[r, c] for r, c in orbit) for orbit in orbits):
            continue
        if not any(any(top[r, c] for r, c in orbit) and any(not top[r, c] for r, c in orbit) for orbit in two_cell_orbits):
            continue
        if _valid_candidates(visible, bottom | top) != [sym]:
            continue
        return top
    return None


def _build_region(
    sym: str,
    marker: int,
    dark: int,
    rng: random.Random,
) -> dict[str, object] | None:
    if sym == "D":
        size = rng.randint(*REGION_SIZE_RANGE)
        h = w = size
    else:
        h = rng.randint(*REGION_SIZE_RANGE)
        w = rng.randint(*REGION_SIZE_RANGE)
        if sym == "H":
            while h == w and rng.random() < 0.5:
                w = rng.randint(*REGION_SIZE_RANGE)
        if sym == "V":
            while h == w and rng.random() < 0.5:
                h = rng.randint(*REGION_SIZE_RANGE)

    bottom = _random_bottom_mask(sym, h, w, rng)
    if bottom is None:
        return None
    top = _random_top_mask(bottom, sym, rng)
    if top is None:
        return None

    bright = DARK_TO_BRIGHT[dark]
    region = np.full((h, w), BLACK, dtype=int)
    region[bottom] = dark
    region[top] = bright

    union_mask = region != BLACK
    side = rng.choice(("N", "S", "W", "E"))
    if side == "N":
        border = [(0, c) for c in range(w) if union_mask[0, c]]
        if not border:
            return None
        _r, c = rng.choice(border)
        marker_offset = (-1, c)
    elif side == "S":
        border = [(h - 1, c) for c in range(w) if union_mask[h - 1, c]]
        if not border:
            return None
        _r, c = rng.choice(border)
        marker_offset = (h, c)
    elif side == "W":
        border = [(r, 0) for r in range(h) if union_mask[r, 0]]
        if not border:
            return None
        r, _c = rng.choice(border)
        marker_offset = (r, -1)
    else:
        border = [(r, w - 1) for r in range(h) if union_mask[r, w - 1]]
        if not border:
            return None
        r, _c = rng.choice(border)
        marker_offset = (r, w)

    selected = bottom if marker == RED else top
    piece_color = dark if marker == RED else bright
    piece = _crop_colored(selected, piece_color)

    return {
        "region": region,
        "marker": marker,
        "marker_offset": marker_offset,
        "piece": piece,
        "shape": (h, w),
    }


def _buffer_bbox(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0 - 2, c0 - 2, r0 + h + 1, c0 + w + 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_regions(
    canvas: np.ndarray,
    objects: list[dict[str, object]],
    rng: random.Random,
) -> list[tuple[int, int, np.ndarray]] | None:
    h, w = canvas.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    records: list[tuple[int, int, np.ndarray]] = []

    for obj in objects:
        region = obj["region"]
        rh, rw = map(int, region.shape)
        marker = int(obj["marker"])
        dr, dc = obj["marker_offset"]
        piece = obj["piece"]

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(2, h - rh - 3)
            c0 = rng.randint(2, w - rw - 3)
            mr = r0 + int(dr)
            mc = c0 + int(dc)
            if not (0 <= mr < h and 0 <= mc < w):
                continue

            buf = _buffer_bbox(r0, c0, rh, rw)
            if any(_boxes_overlap(buf, other) for other in placed_boxes):
                continue

            canvas[r0 : r0 + rh, c0 : c0 + rw] = region
            canvas[mr, mc] = marker
            placed_boxes.append(buf)
            records.append((r0, c0, piece))
            placed = True
            break
        if not placed:
            return None
    return records


def _sample_colors(count: int, rng: random.Random) -> list[int]:
    colors = list(DARK_COLORS)
    rng.shuffle(colors)
    out = colors[: min(count, len(colors))]
    while len(out) < count:
        out.append(rng.choice(DARK_COLORS))
    rng.shuffle(out)
    return out


def _sample_symmetries(count: int, rng: random.Random) -> list[str]:
    base = list(SYMMETRIES)
    rng.shuffle(base)
    out = base[: min(count, len(base))]
    while len(out) < count:
        out.append(rng.choice(SYMMETRIES))
    rng.shuffle(out)
    return out


def _sample_markers(count: int, rng: random.Random) -> list[int]:
    half = count // 2
    out = [RED] * half + [MAGENTA] * half
    if count % 2:
        out.append(rng.choice(MARKER_COLORS))
    while len(out) < count:
        out.append(rng.choice(MARKER_COLORS))
    rng.shuffle(out)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*OBJECT_COUNT_RANGE)
    rows = rng.randint(*CANVAS_RANGE)
    cols = rng.randint(*CANVAS_RANGE)
    grid = np.full((rows, cols), BLACK, dtype=int)

    markers = _sample_markers(count, rng)
    syms = _sample_symmetries(count, rng)
    darks = _sample_colors(count, rng)

    objects: list[dict[str, object]] = []
    for sym, marker, dark in zip(syms, markers, darks):
        obj = _build_region(sym, marker, dark, rng)
        if obj is None:
            return None
        objects.append(obj)

    rng.shuffle(objects)
    records = _place_regions(grid, objects, rng)
    if records is None:
        return None

    records.sort(key=lambda item: (item[0], item[1]))
    expected = _pack_pieces([piece for _r, _c, piece in records])

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
