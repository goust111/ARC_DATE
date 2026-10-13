from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[Coord] = deque([(sr, sc)])
        pts: list[Coord] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.array(pts, dtype=int))
    return out


def _rect_view(g: np.ndarray, b: BBox) -> np.ndarray:
    r0, c0, r1, c1 = b
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _infer_wallpaper(g: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    counts = Counter(map(int, g.ravel()))
    assert len(counts) >= 3
    ranked = [color for color, _count in counts.most_common()]
    wallpaper = tuple(map(int, ranked[:2]))
    wallpaper_set = set(wallpaper)
    wallpaper_mask = np.isin(g, list(wallpaper_set))
    total = int(wallpaper_mask.sum())
    assert total > 0

    best: tuple[float, int, int, np.ndarray] | None = None
    h, w = g.shape
    for ph in range(1, 5):
        for pw in range(1, 5):
            tile = np.empty((ph, pw), dtype=int)
            mismatches = 0
            valid = True
            for rr in range(ph):
                for cc in range(pw):
                    cells = g[rr:h:ph, cc:w:pw]
                    mask = wallpaper_mask[rr:h:ph, cc:w:pw]
                    vals = cells[mask]
                    if vals.size == 0:
                        valid = False
                        break
                    local = Counter(map(int, vals))
                    color = max(wallpaper, key=lambda col: (local.get(col, 0), -wallpaper.index(col)))
                    tile[rr, cc] = color
                    mismatches += int(vals.size) - int(local.get(color, 0))
                if not valid:
                    break
            if not valid or set(map(int, tile.ravel())) != wallpaper_set:
                continue
            area_penalty = 0.002 * ph * pw
            score = mismatches / total + area_penalty
            candidate = (float(score), int(ph), int(pw), tile.copy())
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    assert best is not None
    score, ph, pw, tile = best
    assert score < 0.35, "wallpaper phase must dominate the scene"

    expected = np.empty_like(g)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            expected[r, c] = int(tile[r % ph, c % pw])
    return wallpaper, expected, wallpaper_mask


def _solid_rect_components(g: np.ndarray, wallpaper: tuple[int, int]) -> list[dict]:
    wallpaper_set = set(map(int, wallpaper))
    solid = ~np.isin(g, list(wallpaper_set))
    comps = []
    for coords in _components(solid):
        vals = {int(g[r, c]) for r, c in coords}
        assert len(vals) == 1, "each slider must be monochrome"
        color = next(iter(vals))
        b = _bbox_from_coords(coords)
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        assert coords.shape[0] == area, "each slider must be a filled rectangle"
        assert np.all(_rect_view(g, b) == color)
        comps.append({"color": int(color), "bbox": b, "area": area})
    return comps


def _band_bbox(slider_box: BBox, direction: Coord, length: int) -> BBox:
    r0, c0, r1, c1 = slider_box
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return int(r0 - length), int(c0), int(r0 - 1), int(c1)
    if (dr, dc) == (1, 0):
        return int(r1 + 1), int(c0), int(r1 + length), int(c1)
    if (dr, dc) == (0, -1):
        return int(r0), int(c0 - length), int(r1), int(c0 - 1)
    if (dr, dc) == (0, 1):
        return int(r0), int(c1 + 1), int(r1), int(c1 + length)
    raise ValueError("invalid direction")


def _dest_bbox(slider_box: BBox, direction: Coord, length: int) -> BBox:
    r0, c0, r1, c1 = slider_box
    dr, dc = direction
    return (
        int(r0 + dr * length),
        int(c0 + dc * length),
        int(r1 + dr * length),
        int(c1 + dc * length),
    )


def _far_seam_is_normal(
    mismatch: np.ndarray,
    wallpaper_mask: np.ndarray,
    slider_box: BBox,
    direction: Coord,
    length: int,
) -> bool:
    h, w = mismatch.shape
    r0, c0, r1, c1 = slider_box
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        rr = r0 - length - 1
        if rr < 0:
            return False
        seam_mismatch = mismatch[rr, c0 : c1 + 1]
        seam_wall = wallpaper_mask[rr, c0 : c1 + 1]
    elif (dr, dc) == (1, 0):
        rr = r1 + length + 1
        if rr >= h:
            return False
        seam_mismatch = mismatch[rr, c0 : c1 + 1]
        seam_wall = wallpaper_mask[rr, c0 : c1 + 1]
    elif (dr, dc) == (0, -1):
        cc = c0 - length - 1
        if cc < 0:
            return False
        seam_mismatch = mismatch[r0 : r1 + 1, cc]
        seam_wall = wallpaper_mask[r0 : r1 + 1, cc]
    else:
        cc = c1 + length + 1
        if cc >= w:
            return False
        seam_mismatch = mismatch[r0 : r1 + 1, cc]
        seam_wall = wallpaper_mask[r0 : r1 + 1, cc]
    return bool(np.all(seam_wall) and not np.any(seam_mismatch))


def _side_edges_are_normal(
    mismatch: np.ndarray,
    wallpaper_mask: np.ndarray,
    band_box: BBox,
    direction: Coord,
) -> bool:
    h, w = mismatch.shape
    br0, bc0, br1, bc1 = band_box
    dr, dc = direction
    checks = []
    if dr != 0:
        if bc0 > 0:
            checks.append((mismatch[br0 : br1 + 1, bc0 - 1], wallpaper_mask[br0 : br1 + 1, bc0 - 1]))
        if bc1 + 1 < w:
            checks.append((mismatch[br0 : br1 + 1, bc1 + 1], wallpaper_mask[br0 : br1 + 1, bc1 + 1]))
    else:
        if br0 > 0:
            checks.append((mismatch[br0 - 1, bc0 : bc1 + 1], wallpaper_mask[br0 - 1, bc0 : bc1 + 1]))
        if br1 + 1 < h:
            checks.append((mismatch[br1 + 1, bc0 : bc1 + 1], wallpaper_mask[br1 + 1, bc0 : bc1 + 1]))
    for edge_mismatch, edge_wall in checks:
        if np.any(edge_wall & edge_mismatch):
            return False
    return True


def _scan_band(
    mismatch: np.ndarray,
    wallpaper_mask: np.ndarray,
    slider_box: BBox,
    direction: Coord,
) -> tuple[int, BBox, BBox] | None:
    h, w = mismatch.shape
    r0, c0, r1, c1 = slider_box
    dr, dc = direction
    length = 0

    if (dr, dc) == (-1, 0):
        rr = r0 - 1
        while rr >= 0 and np.all(mismatch[rr, c0 : c1 + 1]):
            length += 1
            rr -= 1
    elif (dr, dc) == (1, 0):
        rr = r1 + 1
        while rr < h and np.all(mismatch[rr, c0 : c1 + 1]):
            length += 1
            rr += 1
    elif (dr, dc) == (0, -1):
        cc = c0 - 1
        while cc >= 0 and np.all(mismatch[r0 : r1 + 1, cc]):
            length += 1
            cc -= 1
    else:
        cc = c1 + 1
        while cc < w and np.all(mismatch[r0 : r1 + 1, cc]):
            length += 1
            cc += 1

    slider_h = r1 - r0 + 1
    slider_w = c1 - c0 + 1
    min_length = slider_h if dr != 0 else slider_w
    if length < min_length:
        return None

    dest = _dest_bbox(slider_box, direction, length)
    dr0, dc0, dr1, dc1 = dest
    if not (0 <= dr0 <= dr1 < h and 0 <= dc0 <= dc1 < w):
        return None
    band = _band_bbox(slider_box, direction, length)
    if not _far_seam_is_normal(mismatch, wallpaper_mask, slider_box, direction, length):
        return None
    if not _side_edges_are_normal(mismatch, wallpaper_mask, band, direction):
        return None
    return int(length), band, dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore phase-shifted wallpaper trails and move each slider back to the far seam of its trail.

    Input structure:
    - The grid is mostly a two-color periodic wallpaper with one globally consistent phase.
    - Several non-wallpaper colors form filled rectangular slider blocks.
    - On exactly one side of each slider is a same-width rectangular wallpaper band whose colors are a shifted phase of the wallpaper, not a new paint color.

    Rule:
    - Infer the normal wallpaper period and phase from the majority of wallpaper-colored cells.
    - For each slider, find the unique adjacent phase-shift band with the slider's width or height and a normal-wallpaper seam at its far end.
    - Replace the slider's current cells and the whole band by the normal wallpaper phase, then paste the slider rectangle at the far seam.

    Key insight:
    - The motion trace is only a second-order phase inconsistency in an otherwise intact wallpaper; reversing the push requires reading that phase shift, not filling with a flat background color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    wallpaper, expected, wallpaper_mask = _infer_wallpaper(g)
    mismatch = wallpaper_mask & (g != expected)
    sliders = _solid_rect_components(g, wallpaper)
    assert sliders, "expected at least one slider"

    out = expected.copy()
    occupied = np.zeros(g.shape, dtype=bool)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for slider in sliders:
        box = slider["bbox"]
        candidates = []
        for direction in directions:
            found = _scan_band(mismatch, wallpaper_mask, box, direction)
            if found is not None:
                candidates.append((direction, *found))
        assert len(candidates) == 1, "each slider must have one unambiguous phase-shift band"

        _direction, _length, _band, dest = candidates[0]
        r0, c0, r1, c1 = dest
        assert not np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]), "restored sliders must not overlap"
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        out[r0 : r1 + 1, c0 : c1 + 1] = int(slider["color"])

    return out


# ---------------- generator ----------------

_DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_ATTEMPTS = 500
_PLACE_ATTEMPTS = 200


def _wallpaper(shape: tuple[int, int], colors: tuple[int, int], phase: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    return np.where((rr + cc + int(phase)) % 2 == 0, int(colors[0]), int(colors[1])).astype(int)


def _shifted_wallpaper(normal: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    return np.where(normal == int(colors[0]), int(colors[1]), int(colors[0])).astype(int)


def _draw_rect(g: np.ndarray, b: BBox, color: int) -> None:
    r0, c0, r1, c1 = b
    g[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _mark_rect(mask: np.ndarray, b: BBox, pad: int = 0) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = b
    r0 = max(0, int(r0) - pad)
    c0 = max(0, int(c0) - pad)
    r1 = min(h - 1, int(r1) + pad)
    c1 = min(w - 1, int(c1) + pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _union_bbox(a: BBox, b: BBox) -> BBox:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _random_current_box(
    shape: tuple[int, int],
    slider_h: int,
    slider_w: int,
    length: int,
    direction: Coord,
) -> BBox | None:
    h, w = shape
    margin = 1
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        r_min, r_max = margin + length, h - margin - slider_h
        c_min, c_max = margin, w - margin - slider_w
    elif (dr, dc) == (1, 0):
        r_min, r_max = margin, h - margin - slider_h - length
        c_min, c_max = margin, w - margin - slider_w
    elif (dr, dc) == (0, -1):
        r_min, r_max = margin, h - margin - slider_h
        c_min, c_max = margin + length, w - margin - slider_w
    else:
        r_min, r_max = margin, h - margin - slider_h
        c_min, c_max = margin, w - margin - slider_w - length
    if r_min > r_max or c_min > c_max:
        return None
    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    return int(r0), int(c0), int(r0 + slider_h - 1), int(c0 + slider_w - 1)


def _place_one(
    shape: tuple[int, int],
    reserved: np.ndarray,
    direction: Coord,
    color: int,
) -> dict | None:
    dr, dc = direction
    for _ in range(_PLACE_ATTEMPTS):
        slider_h = random.randint(2, 4)
        slider_w = random.randint(2, 5)
        moving_span = slider_h if dr != 0 else slider_w
        length = random.randint(moving_span + 1, moving_span + 6)
        current = _random_current_box(shape, slider_h, slider_w, length, direction)
        if current is None:
            continue
        band = _band_bbox(current, direction, length)
        origin = _dest_bbox(current, direction, length)
        affected = _union_bbox(current, band)
        ar0, ac0, ar1, ac1 = affected
        h, w = shape
        if ar0 < 1 or ac0 < 1 or ar1 >= h - 1 or ac1 >= w - 1:
            continue
        padded = (
            max(0, ar0 - 1),
            max(0, ac0 - 1),
            min(h - 1, ar1 + 1),
            min(w - 1, ac1 + 1),
        )
        if np.any(_rect_view(reserved, padded)):
            continue
        _mark_rect(reserved, affected, pad=1)
        return {
            "color": int(color),
            "direction": direction,
            "length": int(length),
            "current": current,
            "band": band,
            "origin": origin,
        }
    return None


def _mask_rect(shape: tuple[int, int], b: BBox) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = b
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _semantics_ok(
    inp: np.ndarray,
    out: np.ndarray,
    normal: np.ndarray,
    shifted: np.ndarray,
    colors: tuple[int, int],
    specs: list[dict],
) -> bool:
    if len(specs) < 2 or np.array_equal(inp, out):
        return False
    if len({spec["direction"] for spec in specs}) != len(specs):
        return False

    band_mask = np.zeros(inp.shape, dtype=bool)
    current_mask = np.zeros(inp.shape, dtype=bool)
    origin_mask = np.zeros(inp.shape, dtype=bool)
    for spec in specs:
        color = int(spec["color"])
        band = spec["band"]
        current = spec["current"]
        origin = spec["origin"]
        band_mask |= _mask_rect(inp.shape, band)
        current_mask |= _mask_rect(inp.shape, current)
        origin_mask |= _mask_rect(inp.shape, origin)

        if not np.all(_rect_view(inp, current) == color):
            return False
        if not np.all(_rect_view(out, origin) == color):
            return False
        if not np.all(_rect_view(inp, band) == _rect_view(shifted, band)):
            return False
        if np.any(_rect_view(inp, band) == _rect_view(normal, band)):
            return False
        vals = set(map(int, _rect_view(inp, band).ravel()))
        if vals != set(map(int, colors)):
            return False

    if not np.all(inp[~(band_mask | current_mask)] == normal[~(band_mask | current_mask)]):
        return False
    if not np.all(out[~origin_mask] == normal[~origin_mask]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    count = random.randint(2, 4)
    palette = random.sample(range(10), count + 2)
    wallpaper_colors = (int(palette[0]), int(palette[1]))
    slider_colors = list(map(int, palette[2:]))

    phase = random.randint(0, 1)
    normal = _wallpaper((h, w), wallpaper_colors, phase)
    shifted = _shifted_wallpaper(normal, wallpaper_colors)
    inp = normal.copy()
    out = normal.copy()
    reserved = np.zeros((h, w), dtype=bool)

    dirs = random.sample(list(_DIRS), count)
    specs: list[dict] = []
    for direction, color in zip(dirs, slider_colors):
        spec = _place_one((h, w), reserved, direction, color)
        if spec is None:
            return None
        specs.append(spec)
        band = spec["band"]
        current = spec["current"]
        origin = spec["origin"]
        _rect_view(inp, band)[:, :] = _rect_view(shifted, band)
        _draw_rect(inp, current, color)
        _draw_rect(out, origin, color)

    if not _semantics_ok(inp, out, normal, shifted, wallpaper_colors, specs):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid phase-shift wallpaper input")
