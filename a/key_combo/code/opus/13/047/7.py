import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (13, 19)
GREEN_AREA_RANGE = (3, 7)
GEN_ATTEMPTS = 600


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components8(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in neighbors8(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _extract_features(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    full_rows = np.flatnonzero(np.all(g != BG, axis=1))
    full_cols = np.flatnonzero(np.all(g != BG, axis=0))
    assert len(full_rows) == 1, "expected one full horizontal bar row"
    assert len(full_cols) == 1, "expected one full vertical bar column"
    h_row = int(full_rows[0])
    v_col = int(full_cols[0])

    assert Counter(map(int, g[h_row]))[BLUE] >= w - 3
    assert Counter(map(int, g[:, v_col]))[RED] >= h - 3

    h_markers = sorted(int(c) for c in np.flatnonzero(g[h_row] == ORANGE))
    v_markers = sorted(int(r) for r in np.flatnonzero(g[:, v_col] == ORANGE))
    assert len(h_markers) == 2, "expected two horizontal markers"
    assert len(v_markers) == 2, "expected two vertical markers"
    assert v_col not in h_markers
    assert h_row not in v_markers

    green_mask = g == GREEN
    green_components = components8(green_mask)
    assert len(green_components) == 1, "expected one green 8-connected object"
    green_coords = green_components[0]
    green_area = int(green_coords.shape[0])
    green_bbox = bbox_coords(green_coords)

    c_left, c_right = h_markers
    r_top, r_bottom = v_markers
    assert c_right - c_left + 1 >= 3
    assert r_bottom - r_top + 1 > green_area + 1

    return {
        "grid": g,
        "shape": (h, w),
        "h_row": h_row,
        "v_col": v_col,
        "c_left": c_left,
        "c_right": c_right,
        "r_top": r_top,
        "r_bottom": r_bottom,
        "green_area": green_area,
        "green_bbox": green_bbox,
    }


def _solve_from_features(features: dict, *, truncate_rows: int | None = None, force_cross: bool = True) -> np.ndarray:
    g = features["grid"]
    out = g.copy()
    h_row = features["h_row"]
    v_col = features["v_col"]
    c_left = features["c_left"]
    c_right = features["c_right"]
    r_top = features["r_top"]
    r_bottom = features["r_bottom"]
    n_rows = features["green_area"] if truncate_rows is None else int(truncate_rows)

    kept_bottom = r_bottom - n_rows
    for r in range(r_top, r_bottom + 1):
        for c in range(c_left, c_right + 1):
            if r > kept_bottom:
                out[r, c] = BG
            else:
                parity = (r - r_top + c - c_left) % 2
                out[r, c] = BLUE if parity == 0 else RED

    if force_cross and r_top <= h_row <= kept_bottom and c_left <= v_col <= c_right:
        out[h_row, v_col] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Weave the marker-bounded crossed-bar rectangle, then truncate its bottom by the green pixel count.

    Input structure:
    - A black background contains one full blue horizontal bar and one full red vertical bar crossing at one cell.
    - The blue bar carries two orange marker pixels that define the rectangle's left and right columns.
    - The red bar carries two orange marker pixels that define the rectangle's top and bottom rows.
    - A separate green 8-connected blob sits away from the bars; its area is the hidden truncation parameter.

    Rule:
    - Read the orange pair on the horizontal bar as the inclusive column span and the orange pair on the vertical bar as the inclusive row span.
    - Fill that rectangle with a blue/red checker weave: parity relative to the rectangle's top-left corner gives blue on even cells and red on odd cells.
    - If the bar crossing lies inside the kept part of the rectangle, force that crossing cell to red so the vertical bar visually covers the horizontal bar.
    - Count the green blob's pixels as N, then erase the bottom N rows of the woven rectangle back to black; everything outside the rectangle stays unchanged.

    Key insight:
    - The weaving rectangle is defined only by the crossed bars and their markers, while the apparently unrelated green blob contributes a hidden numeric parameter: its pixel count is exactly how many bottom rows must be removed.
    """
    return _solve_from_features(_extract_features(np.asarray(grid, dtype=int)))


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_blob(area: int, rng: random.Random) -> np.ndarray:
    for _ in range(400):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors8(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(rng.choice(sorted(frontier)))

        norm = _normalize_shape(cells)
        coords = np.array(norm, dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 2 or bw < 2:
            continue
        if area in {bh, bw, bh * bw, max(bh, bw)}:
            continue
        return coords
    raise RuntimeError("failed to build a non-degenerate green blob")


def _sample_marker_pair(
    length: int,
    cross_idx: int,
    min_span: int,
    rng: random.Random,
    prefer: str | None = None,
) -> tuple[int, int]:
    candidates: dict[str, list[tuple[int, int]]] = {"straddle": [], "left": [], "right": []}
    for a in range(length):
        for b in range(a + 1, length):
            if b - a < min_span:
                continue
            if a < cross_idx < b:
                candidates["straddle"].append((a, b))
            elif b < cross_idx:
                candidates["left"].append((a, b))
            elif a > cross_idx:
                candidates["right"].append((a, b))

    order = {
        None: ["straddle", "left", "right"],
        "same": ["left", "right", "straddle"],
        "straddle": ["straddle", "left", "right"],
        "outside_left": ["left", "right", "straddle"],
        "outside_right": ["right", "left", "straddle"],
    }[prefer]
    viable = [kind for kind in order if candidates[kind]]
    assert viable, "no valid marker pair"

    if prefer == "same":
        pick_kind = rng.choice([kind for kind in viable if kind in {"left", "right"}] or viable)
    elif prefer == "outside_left":
        pick_kind = viable[0]
    elif prefer == "outside_right":
        pick_kind = viable[0]
    else:
        weighted = []
        for kind in viable:
            repeats = 3 if kind == order[0] else 1
            weighted.extend([kind] * repeats)
        pick_kind = rng.choice(weighted)
    return rng.choice(candidates[pick_kind])


def _expanded_mask(h: int, w: int, r0: int, c0: int, r1: int, c1: int, pad: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _place_blob(
    grid: np.ndarray,
    blob: np.ndarray,
    forbidden: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    bh = int(blob[:, 0].max()) + 1
    bw = int(blob[:, 1].max()) + 1
    h, w = grid.shape
    slots = []
    for top in range(h - bh + 1):
        for left in range(w - bw + 1):
            cells = blob + np.array([top, left], dtype=int)
            if np.any(forbidden[cells[:, 0], cells[:, 1]]):
                continue
            slots.append(cells)
    if not slots:
        return None
    return rng.choice(slots)


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    area = rng.randint(*GREEN_AREA_RANGE)

    if h < area + 8 or w < 9:
        return None

    grid = np.zeros((h, w), dtype=int)

    h_row = rng.randint(3, h - 4)
    v_col = rng.randint(3, w - 4)

    branch = rng.choice(["override", "override", "cross_outside", "cross_deleted", "cross_deleted", "free"])

    if branch == "cross_outside":
        if rng.random() < 0.5:
            h_markers = _sample_marker_pair(w, v_col, 2, rng, prefer="same")
            v_markers = _sample_marker_pair(h, h_row, area + 1, rng, prefer=None)
        else:
            h_markers = _sample_marker_pair(w, v_col, 2, rng, prefer=None)
            v_markers = _sample_marker_pair(h, h_row, area + 1, rng, prefer="same")
    else:
        for _ in range(80):
            if branch == "override":
                h_markers = _sample_marker_pair(w, v_col, 2, rng, prefer="straddle")
                v_markers = _sample_marker_pair(h, h_row, area + 1, rng, prefer="straddle")
            elif branch == "cross_deleted":
                h_markers = _sample_marker_pair(w, v_col, 2, rng, prefer="straddle")
                v_markers = _sample_marker_pair(h, h_row, area + 1, rng, prefer="straddle")
            else:
                h_markers = _sample_marker_pair(w, v_col, 2, rng, prefer="straddle" if rng.random() < 0.5 else "same")
                v_markers = _sample_marker_pair(h, h_row, area + 1, rng, prefer="straddle" if rng.random() < 0.5 else "same")

            c_left, c_right = h_markers
            r_top, r_bottom = v_markers
            kept_bottom = r_bottom - area
            cross_inside = r_top <= h_row <= r_bottom and c_left <= v_col <= c_right
            cross_kept = cross_inside and h_row <= kept_bottom
            cross_deleted = cross_inside and h_row > kept_bottom
            cross_parity_even = ((h_row - r_top + v_col - c_left) % 2) == 0 if cross_inside else False

            if branch == "override" and cross_kept and cross_parity_even:
                break
            if branch == "cross_deleted" and cross_deleted:
                break
            if branch == "free":
                break
        else:
            return None

    c_left, c_right = h_markers
    r_top, r_bottom = v_markers
    kept_bottom = r_bottom - area

    cross_inside = r_top <= h_row <= r_bottom and c_left <= v_col <= c_right
    if branch == "cross_outside" and cross_inside:
        return None

    grid[h_row, :] = BLUE
    grid[:, v_col] = RED
    grid[h_row, v_col] = RED
    grid[h_row, c_left] = ORANGE
    grid[h_row, c_right] = ORANGE
    grid[r_top, v_col] = ORANGE
    grid[r_bottom, v_col] = ORANGE

    blob = _random_blob(area, rng)
    forbidden = grid != BG
    forbidden[max(0, h_row - 1) : min(h, h_row + 2), :] = True
    forbidden[:, max(0, v_col - 1) : min(w, v_col + 2)] = True
    forbidden |= _expanded_mask(h, w, r_top, c_left, r_bottom, c_right, 1)

    blob_cells = _place_blob(grid, blob, forbidden, rng)
    if blob_cells is None:
        return None
    grid[blob_cells[:, 0], blob_cells[:, 1]] = GREEN

    if not _validate_instance(grid):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        features = _extract_features(grid)
    except AssertionError:
        return False

    h, w = features["shape"]
    h_row = features["h_row"]
    v_col = features["v_col"]
    c_left = features["c_left"]
    c_right = features["c_right"]
    r_top = features["r_top"]
    r_bottom = features["r_bottom"]
    area = features["green_area"]
    g_r0, g_c0, g_r1, g_c1 = features["green_bbox"]

    kept_bottom = r_bottom - area
    if kept_bottom - r_top + 1 < 2:
        return False

    green_mask = grid == GREEN
    if len(components8(green_mask)) != 1:
        return False

    if g_r0 <= h_row + 1 and g_r1 >= h_row - 1:
        return False
    if g_c0 <= v_col + 1 and g_c1 >= v_col - 1:
        return False

    alt_ns = {g_r1 - g_r0 + 1, g_c1 - g_c0 + 1, (g_r1 - g_r0 + 1) * (g_c1 - g_c0 + 1)}
    if area in alt_ns:
        return False

    out = _solve_from_features(features)
    for bad_n in alt_ns:
        bad = _solve_from_features(features, truncate_rows=bad_n)
        if np.array_equal(out, bad):
            return False

    if r_top <= h_row <= kept_bottom and c_left <= v_col <= c_right:
        parity = (h_row - r_top + v_col - c_left) % 2
        if parity == 0:
            no_cross = _solve_from_features(features, force_cross=False)
            if np.array_equal(out, no_cross):
                return False

    raw_weave = _solve_from_features(features, truncate_rows=0)
    if np.array_equal(out, raw_weave):
        return False

    if not np.any(out[r_top : kept_bottom + 1, c_left : c_right + 1] != BG):
        return False

    if h > 30 or w > 30 or out.shape[0] > 30 or out.shape[1] > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
