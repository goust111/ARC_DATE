from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
FG = 1
CTRL = 2

ADJACENT_EDGE_PAIRS = [
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
]
KEEP_AREAS = (4, 9, 16)
VANISH_AREAS = (2, 3, 5, 6)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _row_letter(edge: str) -> str:
    return "T" if edge == "top" else "B"


def _col_letter(edge: str) -> str:
    return "L" if edge == "left" else "R"


def _opposite_row(row_side: str) -> str:
    return "B" if row_side == "T" else "T"


def _opposite_col(col_side: str) -> str:
    return "R" if col_side == "L" else "L"


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _has_hole(mask: np.ndarray) -> bool:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not coords:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    padded = np.ones((hh + 2, ww + 2), dtype=bool)
    padded[1:-1, 1:-1] = ~sub
    seen = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(padded.shape, nr, nc) and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    holes = padded & ~seen
    return bool(np.any(holes))


def _parse_controls(grid: np.ndarray):
    h, w = grid.shape
    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(grid == CTRL)]
    if len(ctrl_pts) not in (1, 2):
        raise AssertionError("expected 1 or 2 control pixels")

    v_edge = None
    h_edge = None
    r_split = None
    c_split = None

    for r, c in ctrl_pts:
        if r == 0:
            if v_edge is not None:
                raise AssertionError("at most one vertical control")
            v_edge = "top"
            c_split = c
        elif r == h - 1:
            if v_edge is not None:
                raise AssertionError("at most one vertical control")
            v_edge = "bottom"
            c_split = c
        elif c == 0:
            if h_edge is not None:
                raise AssertionError("at most one horizontal control")
            h_edge = "left"
            r_split = r
        elif c == w - 1:
            if h_edge is not None:
                raise AssertionError("at most one horizontal control")
            h_edge = "right"
            r_split = r
        else:
            raise AssertionError("control pixel must be on the border")

    if len(ctrl_pts) == 2 and (v_edge is None or h_edge is None):
        raise AssertionError("two controls must sit on different edge orientations")

    return ctrl_pts, v_edge, h_edge, r_split, c_split


def _classify_region(r: int, c: int, v_edge: str | None, h_edge: str | None,
                     r_split: int | None, c_split: int | None) -> str:
    if v_edge is not None and h_edge is not None:
        if r == r_split:
            row_side = _row_letter(v_edge)
        else:
            row_side = "T" if r < r_split else "B"

        if c == c_split:
            col_side = _col_letter(h_edge)
        else:
            col_side = "L" if c < c_split else "R"
        return row_side + col_side

    if v_edge is not None:
        return "L" if c <= c_split else "R"

    if h_edge is not None:
        return "T" if r <= r_split else "B"

    raise AssertionError("missing control edge")


def _kept_labels(v_edge: str | None, h_edge: str | None) -> tuple[list[str], dict[str, str]]:
    if v_edge is not None and h_edge is not None:
        vr = _row_letter(v_edge)
        hc = _col_letter(h_edge)
        labels = ["TL", "TR", "BL", "BR"]
        keep = [lab for lab in labels if lab[0] == vr or lab[1] == hc]
        return keep, {lab: lab for lab in keep}
    if v_edge is not None:
        top_or_bottom = "T" if v_edge == "top" else "B"
        return ["L", "R"], {
            "L": top_or_bottom + "L",
            "R": top_or_bottom + "R",
        }
    right_or_left = "L" if h_edge == "left" else "R"
    return ["T", "B"], {
        "T": "T" + right_or_left,
        "B": "B" + right_or_left,
    }


def _draw_pyramid(out: np.ndarray, corner: str, height: int, color: int = FG) -> None:
    h, w = out.shape
    width = 2 * height - 1
    if corner[0] == "T":
        top = 1
    else:
        top = h - 1 - height - 1
    if corner[1] == "L":
        left = 1
    else:
        left = w - width - 2

    center_c = left + height - 1
    for level in range(height):
        row = top + level
        half = level
        out[row, center_c - half : center_c + half + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the central blue object by edge rays, keep only border-owned slices, and rebuild them as corner pyramids.

    Input structure:
    - A black background contains one large blue 4-connected object near the center.
    - One or two red singleton control pixels sit on grid edges; valid generated samples use one vertical-edge control and one horizontal-edge control.
    - The control row and control column define the ray-cut lines that partition the blue object into border-related regions.

    Rule:
    - Treat the red edge pixels as inward ray emitters along their row or column; the resulting horizontal and vertical split lines partition the blue object.
    - Blue pixels exactly on a split line belong to the side anchored by the corresponding control edge, so every blue pixel is assigned to one border-owned region.
    - Keep only the blue regions that touch at least one control edge, erase the others, and discard the red controls.
    - For each kept region, let k be the square root of its pixel count and redraw those pixels as a blue pyramid of height k with row widths 1,3,...,2k-1.
    - Place each pyramid in the corner determined by its retained region, with a one-cell margin and the pyramid baseline aligned to the bottom of its corner footprint.

    Key insight:
    - The ray partition decides how many blue pixels survive in each border-owned slice, and those surviving square counts uniquely determine the final corner pyramids.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    _ctrl_pts, v_edge, h_edge, r_split, c_split = _parse_controls(g)
    keep_labels, corner_map = _kept_labels(v_edge, h_edge)

    groups: dict[str, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == FG):
        label = _classify_region(int(r), int(c), v_edge, h_edge, r_split, c_split)
        groups.setdefault(label, []).append((int(r), int(c)))

    for label in keep_labels:
        pts = groups.get(label, [])
        if not pts:
            continue
        area = len(pts)
        height = math.isqrt(area)
        if height * height != area:
            raise AssertionError(f"kept region {label} area {area} is not a square")
        _draw_pyramid(out, corner_map[label], height, FG)

    return out


def _seed_for_label(label: str, control_label: str, r_split: int, c_split: int) -> tuple[int, int]:
    row_side, col_side = label
    control_row, control_col = control_label
    if row_side == control_row:
        r = r_split
    else:
        r = r_split - 1 if row_side == "T" else r_split + 1
    if col_side == control_col:
        c = c_split
    else:
        c = c_split - 1 if col_side == "L" else c_split + 1
    return r, c


def _allowed_cells(shape: tuple[int, int], label: str, v_edge: str, h_edge: str,
                   r_split: int, c_split: int, radius: int) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(max(1, r_split - radius), min(h - 1, r_split + radius + 1)):
        for c in range(max(1, c_split - radius), min(w - 1, c_split + radius + 1)):
            if _classify_region(r, c, v_edge, h_edge, r_split, c_split) == label:
                cells.add((r, c))
    return cells


def _grow_region(shape: tuple[int, int], allowed: set[tuple[int, int]], seed: tuple[int, int],
                 count: int, occupied: set[tuple[int, int]], center: tuple[int, int]):
    if seed not in allowed or seed in occupied:
        return None
    q = deque([seed])
    seen = {seed}
    order: list[tuple[int, int]] = []
    while q and len(order) < count:
        cell = q.popleft()
        order.append(cell)
        nbrs = list(_neighbors4(*cell))
        random.shuffle(nbrs)
        nbrs.sort(key=lambda pt: abs(pt[0] - center[0]) + abs(pt[1] - center[1]))
        for nr, nc in nbrs:
            if (nr, nc) in allowed and (nr, nc) not in occupied and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    if len(order) != count:
        return None
    return set(order)


def _sample_region_areas() -> dict[str, int] | None:
    for _ in range(100):
        v_edge, h_edge = random.choice(ADJACENT_EDGE_PAIRS)
        control_label = _row_letter(v_edge) + _col_letter(h_edge)
        vanish_label = _opposite_row(control_label[0]) + _opposite_col(control_label[1])
        counts = {lab: random.choice(KEEP_AREAS) for lab in ("TL", "TR", "BL", "BR") if lab != vanish_label}
        counts[vanish_label] = random.choice(VANISH_AREAS)
        total = sum(counts.values())
        if 20 <= total <= 30:
            kept = [counts[lab] for lab in counts if lab != vanish_label]
            if len(set(kept)) >= 2 or 16 in kept:
                counts["_v_edge"] = v_edge  # type: ignore[index]
                counts["_h_edge"] = h_edge  # type: ignore[index]
                counts["_vanish"] = vanish_label  # type: ignore[index]
                return counts
    return None


def _semantic_summary(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    palette = set(map(int, np.unique(g)))
    if palette - {BG, FG, CTRL}:
        raise AssertionError("unexpected colors")

    ctrl_pts, v_edge, h_edge, r_split, c_split = _parse_controls(g)
    if len(ctrl_pts) != 2 or v_edge is None or h_edge is None:
        raise AssertionError("generator must produce two adjacent controls")

    blue_mask = g == FG
    comps = _component_coords(blue_mask)
    if len(comps) != 1:
        raise AssertionError("blue object must be 4-connected")
    if _has_hole(blue_mask):
        raise AssertionError("blue object must be hole-free")
    if not np.any(blue_mask[r_split, :]) or not np.any(blue_mask[:, c_split]):
        raise AssertionError("both split lines must cut through the blue object")

    groups: dict[str, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(blue_mask):
        label = _classify_region(int(r), int(c), v_edge, h_edge, r_split, c_split)
        groups.setdefault(label, []).append((int(r), int(c)))

    control_label = _row_letter(v_edge) + _col_letter(h_edge)
    vanish_label = _opposite_row(control_label[0]) + _opposite_col(control_label[1])
    keep_labels = [lab for lab in ("TL", "TR", "BL", "BR") if lab != vanish_label]

    if any(len(groups.get(lab, [])) == 0 for lab in ("TL", "TR", "BL", "BR")):
        raise AssertionError("all four quadrants must be represented")
    for lab in keep_labels:
        area = len(groups[lab])
        height = math.isqrt(area)
        if height * height != area:
            raise AssertionError(f"kept label {lab} is not square-sized")
    vanish_area = len(groups[vanish_label])
    if vanish_area in KEEP_AREAS or vanish_area <= 0:
        raise AssertionError("vanishing fragment must be non-square and non-empty")

    out = main(g)
    if out.shape != g.shape or max(out.shape) > 30:
        raise AssertionError("output shape invalid")
    blue_out = out == FG
    out_comps = _component_coords(blue_out)
    if len(out_comps) != len(keep_labels):
        raise AssertionError("output must contain one pyramid per kept region")

    return {
        "shape": (h, w),
        "control_label": control_label,
        "vanish_label": vanish_label,
        "areas": {lab: len(groups[lab]) for lab in ("TL", "TR", "BL", "BR")},
    }


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_region_areas()
    if sampled is None:
        return None

    v_edge = sampled.pop("_v_edge")  # type: ignore[assignment]
    h_edge = sampled.pop("_h_edge")  # type: ignore[assignment]
    control_label = _row_letter(v_edge) + _col_letter(h_edge)

    h = random.randint(18, 24)
    w = random.randint(18, 24)
    r_split = random.randint(7, h - 8)
    c_split = random.randint(7, w - 8)
    radius = random.randint(4, 5)
    center = (r_split, c_split)

    labels = ["TL", "TR", "BL", "BR"]
    order = sorted(labels, key=lambda lab: (-int(sampled[lab]), lab != control_label))
    occupied: set[tuple[int, int]] = set()
    regions: dict[str, set[tuple[int, int]]] = {}

    for lab in order:
        allowed = _allowed_cells((h, w), lab, v_edge, h_edge, r_split, c_split, radius)
        seed = _seed_for_label(lab, control_label, r_split, c_split)
        region = _grow_region((h, w), allowed, seed, int(sampled[lab]), occupied, center)
        if region is None:
            return None
        regions[lab] = region
        occupied |= region

    grid = np.zeros((h, w), dtype=int)
    for lab in labels:
        for r, c in regions[lab]:
            grid[r, c] = FG

    if v_edge == "top":
        grid[0, c_split] = CTRL
    else:
        grid[h - 1, c_split] = CTRL
    if h_edge == "left":
        grid[r_split, 0] = CTRL
    else:
        grid[r_split, w - 1] = CTRL

    try:
        _semantic_summary(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} retries")
