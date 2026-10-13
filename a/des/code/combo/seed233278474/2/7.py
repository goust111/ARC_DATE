from __future__ import annotations

from collections import Counter, deque

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = _NEIGHBORS8 if connectivity == 8 else _NEIGHBORS4
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in neighbors:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_patch(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0)


def _count_holes_in_solid(solid: np.ndarray) -> int:
    solid = np.asarray(solid, dtype=bool)
    h, w = solid.shape
    bg = ~solid
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    for r, c in np.argwhere(bg & ~seen):
        rr = int(r)
        cc = int(c)
        holes += 1
        seen[rr, cc] = True
        q.append((rr, cc))
        while q:
            cr, cc = q.popleft()
            for dr, dc in _NEIGHBORS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return holes


def _count_holes_in_patch(patch: np.ndarray) -> int:
    return _count_holes_in_solid(np.asarray(patch) != 0)


def _transform_by_holes(patch: np.ndarray, holes: int) -> np.ndarray:
    if holes == 0:
        return np.fliplr(patch)
    if holes == 1:
        return np.rot90(patch, -1)
    if holes == 2:
        return patch[::2, ::2].copy()
    raise AssertionError(f"unsupported hole count: {holes}")


def _nearest_corner(anchor: tuple[int, int], h: int, w: int) -> str:
    r, c = anchor
    dists = {
        "tl": r + c,
        "tr": r + (w - 1 - c),
        "bl": (h - 1 - r) + c,
        "br": (h - 1 - r) + (w - 1 - c),
    }
    return min(("tl", "tr", "bl", "br"), key=lambda name: (dists[name], name))


def _corner_origin(anchor: tuple[int, int], corner: str, shape: tuple[int, int]) -> tuple[int, int]:
    r, c = anchor
    hh, ww = shape
    if corner == "tl":
        return r, c
    if corner == "tr":
        return r, c - ww + 1
    if corner == "bl":
        return r - hh + 1, c
    if corner == "br":
        return r - hh + 1, c - ww + 1
    raise AssertionError(f"bad corner: {corner}")


def _paste_nonzero(out: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    hh, ww = patch.shape
    mask = patch != 0
    out[r0 : r0 + hh, c0 : c0 + ww][mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each interior object according to its hole count, then move it to the edge anchor
    whose color matches the object's internal marker pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    comps = _components(g != 0, connectivity=8)
    anchors: dict[int, tuple[int, int]] = {}
    objects: list[list[tuple[int, int]]] = []

    for comp in comps:
        touches_edge = any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp)
        if len(comp) == 1 and touches_edge:
            r, c = comp[0]
            anchors[int(g[r, c])] = (r, c)
        else:
            assert not touches_edge, "foreground objects are expected to stay away from the border"
            objects.append(comp)

    assert anchors, "expected edge anchors"
    assert objects, "expected interior objects"

    out = g.copy()
    for comp in objects:
        for r, c in comp:
            out[r, c] = 0

    for comp in objects:
        patch, _ = _extract_patch(g, comp)
        counts = Counter(int(v) for v in patch[patch != 0].ravel())
        assert len(counts) == 2, "each object should contain one body color and one marker color"
        body_color, _ = max(counts.items(), key=lambda item: item[1])
        marker_colors = [color for color, count in counts.items() if color != body_color and count == 1]
        assert len(marker_colors) == 1, "expected a unique one-pixel marker"
        marker_color = marker_colors[0]

        holes = _count_holes_in_patch(patch)
        transformed = _transform_by_holes(patch, holes)
        anchor = anchors[marker_color]
        corner = _nearest_corner(anchor, h, w)
        r0, c0 = _corner_origin(anchor, corner, transformed.shape)
        th, tw = transformed.shape
        assert 0 <= r0 <= h - th and 0 <= c0 <= w - tw
        _paste_nonzero(out, transformed, r0, c0)

    return out


_RNG = np.random.default_rng()

_GRID_SIDE_RANGE = (24, 31)
_OBJECT_COUNT_CHOICES = (3, 4)
_INPUT_MARGIN = 2
_INPUT_PADDING = 1
_ANCHOR_OFFSET_MAX = 3
_GEN_ATTEMPTS = 400
_HOLE_PATTERNS = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
]
_CORNERS = ("tl", "tr", "bl", "br")


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def _expand_rect(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return max(0, r0 - pad), max(0, c0 - pad), min(h, r1 + pad), min(w, c1 + pad)


def _shuffle_in_place(items: list) -> None:
    for i in range(len(items) - 1, 0, -1):
        j = int(_RNG.integers(0, i + 1))
        items[i], items[j] = items[j], items[i]


def _sample_dims(hole_count: int) -> tuple[int, int]:
    if hole_count == 2:
        options = [(5, 7), (7, 5), (7, 7)]
        return options[int(_RNG.integers(0, len(options)))]
    return int(_RNG.integers(6, 9)), int(_RNG.integers(6, 9))


def _carve_random_notch(mask: np.ndarray) -> None:
    h, w = mask.shape
    edge = int(_RNG.integers(0, 4))
    if edge in (0, 1):
        width = int(_RNG.integers(1, min(4, w - 1)))
        start = int(_RNG.integers(1, w - width))
        depth = int(_RNG.integers(1, min(3, h - 1)))
        if edge == 0:
            mask[:depth, start : start + width] = False
        else:
            mask[h - depth :, start : start + width] = False
    else:
        height = int(_RNG.integers(1, min(4, h - 1)))
        start = int(_RNG.integers(1, h - height))
        depth = int(_RNG.integers(1, min(3, w - 1)))
        if edge == 2:
            mask[start : start + height, :depth] = False
        else:
            mask[start : start + height, w - depth :] = False


def _place_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    pattern_ids = list(range(len(_HOLE_PATTERNS)))
    _shuffle_in_place(pattern_ids)
    for pid in pattern_ids:
        cells = _HOLE_PATTERNS[pid]
        hh = max(dr for dr, _ in cells) + 1
        ww = max(dc for _, dc in cells) + 1
        starts = [(r, c) for r in range(1, h - hh) for c in range(1, w - ww)]
        _shuffle_in_place(starts)
        for r, c in starts:
            r1 = r - 1
            r2 = r + hh + 1
            c1 = c - 1
            c2 = c + ww + 1
            if not bool(mask[r1:r2, c1:c2].all()):
                continue
            for dr, dc in cells:
                mask[r + dr, c + dc] = False
            return True
    return False


def _make_object_mask(h: int, w: int, hole_count: int) -> np.ndarray | None:
    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        notch_count = int(_RNG.integers(2, 5))
        for _ in range(notch_count):
            _carve_random_notch(mask)
        if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
            continue
        ok = True
        for _ in range(hole_count):
            if not _place_hole(mask):
                ok = False
                break
        if not ok:
            continue
        if len(_components(mask, connectivity=8)) != 1:
            continue
        if _count_holes_in_solid(mask) != hole_count:
            continue
        if int(mask.sum()) < (h * w) // 2:
            continue
        return mask
    return None


def _marker_candidates(mask: np.ndarray, hole_count: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    coords = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if 0 < int(r) < h - 1 and 0 < int(c) < w - 1
    ]
    if hole_count == 2:
        coords = [(r, c) for r, c in coords if r % 2 == 0 and c % 2 == 0]
    return coords


def _make_object_patch(hole_count: int, body_color: int, marker_color: int) -> dict | None:
    for _ in range(300):
        h, w = _sample_dims(hole_count)
        mask = _make_object_mask(h, w, hole_count)
        if mask is None:
            continue
        candidates = _marker_candidates(mask, hole_count)
        if not candidates:
            continue
        _shuffle_in_place(candidates)
        for mr, mc in candidates:
            patch = np.zeros((h, w), dtype=int)
            patch[mask] = body_color
            patch[mr, mc] = marker_color
            transformed = _transform_by_holes(patch, hole_count)
            if patch.shape == transformed.shape and np.array_equal(patch, transformed):
                continue
            return {
                "holes": hole_count,
                "patch": patch,
                "transformed": transformed,
                "marker_color": marker_color,
            }
    return None


def _anchor_candidates(corner: str, shape: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    hh, ww = shape
    max_row_offset = min(_ANCHOR_OFFSET_MAX, h - hh)
    max_col_offset = min(_ANCHOR_OFFSET_MAX, w - ww)
    candidates: list[tuple[int, int]] = []

    if corner == "tl":
        candidates.extend((0, c) for c in range(max_col_offset + 1))
        candidates.extend((r, 0) for r in range(max_row_offset + 1))
    elif corner == "tr":
        candidates.extend((0, w - 1 - c) for c in range(max_col_offset + 1))
        candidates.extend((r, w - 1) for r in range(max_row_offset + 1))
    elif corner == "bl":
        candidates.extend((h - 1 - r, 0) for r in range(max_row_offset + 1))
        candidates.extend((h - 1, c) for c in range(max_col_offset + 1))
    elif corner == "br":
        candidates.extend((h - 1 - r, w - 1) for r in range(max_row_offset + 1))
        candidates.extend((h - 1, w - 1 - c) for c in range(max_col_offset + 1))
    else:
        raise AssertionError(f"bad corner: {corner}")

    dedup = list(dict.fromkeys(candidates))
    _shuffle_in_place(dedup)
    return dedup


def _placement_rect(anchor: tuple[int, int], corner: str, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = _corner_origin(anchor, corner, shape)
    hh, ww = shape
    return r0, c0, r0 + hh, c0 + ww


def _assign_anchors(objects: list[dict], h: int, w: int) -> bool:
    order = list(range(len(objects)))
    _shuffle_in_place(order)
    placed_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        obj = objects[idx]
        candidates = _anchor_candidates(obj["corner"], obj["transformed"].shape, h, w)
        chosen = None
        chosen_rect = None
        for anchor in candidates:
            rect = _placement_rect(anchor, obj["corner"], obj["transformed"].shape)
            if any(_rect_intersects(rect, other) for other in placed_rects):
                continue
            chosen = anchor
            chosen_rect = rect
            break
        if chosen is None or chosen_rect is None:
            return False
        obj["anchor"] = chosen
        obj["out_rect"] = chosen_rect
        placed_rects.append(chosen_rect)
    return True


def _assign_input_positions(objects: list[dict], h: int, w: int) -> bool:
    order = list(range(len(objects)))
    order.sort(
        key=lambda idx: (
            -objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1],
            -objects[idx]["patch"].shape[0],
        )
    )
    placed_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = objects[idx]["patch"]
        hh, ww = patch.shape
        candidates = [
            (r, c)
            for r in range(_INPUT_MARGIN, h - hh - _INPUT_MARGIN + 1)
            for c in range(_INPUT_MARGIN, w - ww - _INPUT_MARGIN + 1)
        ]
        _shuffle_in_place(candidates)
        chosen = None
        chosen_rect = None
        for r0, c0 in candidates:
            rect = (r0, c0, r0 + hh, c0 + ww)
            padded = _expand_rect(rect, _INPUT_PADDING, h, w)
            if any(_rect_intersects(padded, _expand_rect(other, _INPUT_PADDING, h, w)) for other in placed_rects):
                continue
            chosen = (r0, c0)
            chosen_rect = rect
            break
        if chosen is None or chosen_rect is None:
            return False
        objects[idx]["input_pos"] = chosen
        placed_rects.append(chosen_rect)
    return True


def _build_input(objects: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        ar, ac = obj["anchor"]
        grid[ar, ac] = obj["marker_color"]
    for obj in objects:
        r0, c0 = obj["input_pos"]
        _paste_nonzero(grid, obj["patch"], r0, c0)
    return grid


def _build_expected(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        r0, c0 = obj["input_pos"]
        hh, ww = obj["patch"].shape
        mask = obj["patch"] != 0
        out[r0 : r0 + hh, c0 : c0 + ww][mask] = 0
    for obj in objects:
        ar, ac = obj["anchor"]
        corner = _nearest_corner((ar, ac), inp.shape[0], inp.shape[1])
        r0, c0 = _corner_origin((ar, ac), corner, obj["transformed"].shape)
        _paste_nonzero(out, obj["transformed"], r0, c0)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = int(_RNG.integers(*_GRID_SIDE_RANGE))
    w = int(_RNG.integers(*_GRID_SIDE_RANGE))
    n_objects = int(_OBJECT_COUNT_CHOICES[int(_RNG.integers(0, len(_OBJECT_COUNT_CHOICES)))])

    pool = list(range(1, 10))
    _shuffle_in_place(pool)
    body_colors = pool[:n_objects]
    marker_colors = pool[n_objects : 2 * n_objects]
    corners = list(_CORNERS)
    _shuffle_in_place(corners)
    corners = corners[:n_objects]

    holes = [0, 1, 2]
    while len(holes) < n_objects:
        holes.append(int(_RNG.integers(0, 3)))
    _shuffle_in_place(holes)

    objects: list[dict] = []
    for i in range(n_objects):
        obj = _make_object_patch(holes[i], body_colors[i], marker_colors[i])
        if obj is None:
            return None
        obj["corner"] = corners[i]
        objects.append(obj)

    if not _assign_anchors(objects, h, w):
        return None
    if not _assign_input_positions(objects, h, w):
        return None

    inp = _build_input(objects, h, w)
    expected = _build_expected(inp, objects)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
