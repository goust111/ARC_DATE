from __future__ import annotations

from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

ALL_COLORS = tuple(range(10))
CORNER_ORDER = ("TL", "TR", "BL", "BR")
CORNER_TO_DIR = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}
SIDE_TO_CORNERS = {
    "top": ("TL", "TR"),
    "bottom": ("BL", "BR"),
    "left": ("TL", "BL"),
    "right": ("TR", "BR"),
}

GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _flood_background(mask: np.ndarray, seeds: list[Coord]) -> np.ndarray:
    region = np.zeros_like(mask, dtype=bool)
    q: deque[Coord] = deque()
    for r, c in seeds:
        if mask[r, c] and not region[r, c]:
            region[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not region[nr, nc]:
                region[nr, nc] = True
                q.append((nr, nc))
    return region


def _frame_patch(side: int) -> np.ndarray:
    patch = np.zeros((side, side), dtype=bool)
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    patch[:, -1] = True
    return patch


def _shape_h(b: BBox) -> int:
    return b[2] - b[0] + 1


def _shape_w(b: BBox) -> int:
    return b[3] - b[1] + 1


def _extract_objects(grid: np.ndarray) -> tuple[int, int, int, BBox, BBox]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 2, "expected exactly two non-background colors"

    solid_color = None
    frame_color = None
    solid_bbox = None
    frame_bbox = None

    for color in colors:
        mask = g == color
        box = bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        patch = mask[r0 : r1 + 1, c0 : c1 + 1]
        if patch.all():
            assert solid_color is None, "expected one solid rectangle"
            solid_color = color
            solid_bbox = box
            continue

        side_h = _shape_h(box)
        side_w = _shape_w(box)
        if side_h == side_w and np.array_equal(patch, _frame_patch(side_h)):
            assert frame_color is None, "expected one hollow square frame"
            frame_color = color
            frame_bbox = box

    assert solid_color is not None and solid_bbox is not None, "solid rectangle not found"
    assert frame_color is not None and frame_bbox is not None, "hollow square frame not found"
    return bg, solid_color, frame_color, solid_bbox, frame_bbox


def _rect_corners(rect_bbox: BBox) -> dict[str, Coord]:
    r0, c0, r1, c1 = rect_bbox
    return {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BL": (r1, c0),
        "BR": (r1, c1),
    }


def _frame_corners(frame_bbox: BBox) -> dict[str, Coord]:
    r0, c0, r1, c1 = frame_bbox
    return {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BL": (r1, c0),
        "BR": (r1, c1),
    }


def _side_seed_cells(side: str, frame_bbox: BBox, shape: tuple[int, int]) -> list[Coord]:
    r0, c0, r1, c1 = frame_bbox
    seeds: list[Coord] = []
    if side == "top" and r0 > 0:
        seeds = [(r0 - 1, c) for c in range(c0 + 1, c1)]
    elif side == "bottom" and r1 + 1 < shape[0]:
        seeds = [(r1 + 1, c) for c in range(c0 + 1, c1)]
    elif side == "left" and c0 > 0:
        seeds = [(r, c0 - 1) for r in range(r0 + 1, r1)]
    elif side == "right" and c1 + 1 < shape[1]:
        seeds = [(r, c1 + 1) for r in range(r0 + 1, r1)]
    return seeds


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, rect_color, frame_color, rect_bbox, frame_bbox = _extract_objects(g)
    out = g.copy()
    ray_mask = np.zeros_like(g, dtype=bool)

    rect_corners = _rect_corners(rect_bbox)
    frame_corners = _frame_corners(frame_bbox)
    active_hits: dict[str, Coord] = {}
    first_hits: dict[str, Coord | None] = {}

    for name in CORNER_ORDER:
        r, c = rect_corners[name]
        dr, dc = CORNER_TO_DIR[name]
        rr, cc = r + dr, c + dc
        first: Coord | None = None
        while in_bounds(g.shape, rr, cc) and g[rr, cc] == bg:
            rr += dr
            cc += dc
        if in_bounds(g.shape, rr, cc):
            first = (rr, cc)
        first_hits[name] = first
        if first is None or first != frame_corners[name]:
            continue

        active_hits[name] = first
        rr, cc = r + dr, c + dc
        while in_bounds(g.shape, rr, cc):
            if g[rr, cc] == bg:
                out[rr, cc] = rect_color
                ray_mask[rr, cc] = True
            rr += dr
            cc += dc

    active_sides = [side for side, corners in SIDE_TO_CORNERS.items() if all(corner in active_hits for corner in corners)]
    fill_regions: dict[str, np.ndarray] = {}
    for side in active_sides:
        seeds = _side_seed_cells(side, frame_bbox, g.shape)
        if not seeds:
            continue
        region = _flood_background(out == bg, seeds)
        if not region.any():
            continue
        touches_boundary = bool(
            region[0, :].any()
            or region[-1, :].any()
            or region[:, 0].any()
            or region[:, -1].any()
        )
        if not touches_boundary:
            continue
        fill_regions[side] = region
        out[region] = frame_color

    return {
        "bg": bg,
        "rect_color": rect_color,
        "frame_color": frame_color,
        "rect_bbox": rect_bbox,
        "frame_bbox": frame_bbox,
        "active_hits": active_hits,
        "first_hits": first_hits,
        "active_sides": active_sides,
        "ray_mask": ray_mask,
        "fill_regions": fill_regions,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Use frame-corner-selected diagonal rays to wall off and fill the outside pocket around a hollow square.

    Input structure:
    - The grid has a dominant background, one solid rectangle, and one one-cell-thick hollow square in a different color.
    - The rectangle and the frame are separated by background cells; the rectangle may sit inside the frame.

    Rule:
    - From each outer corner of the solid rectangle, trace the outward diagonal.
    - If the first non-background cell on that diagonal is the matching corner of the hollow square, activate that ray and paint every background cell on the whole diagonal from the rectangle corner to the canvas edge with the rectangle color.
    - After all active rays are drawn, any boundary-touching background chamber immediately outside a frame side whose two end corners were both hit by active rays is flood-filled with the frame color.

    Key insight:
    - The frame corners do not just select which diagonals are real; pairs of selected diagonals on the same side turn those rays into temporary walls that create a fillable pocket outside the frame.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _adjacent_colors(grid: np.ndarray, region: np.ndarray) -> set[int]:
    out: set[int] = set()
    for r, c in np.argwhere(region):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if in_bounds(grid.shape, nr, nc) and not region[nr, nc]:
                out.add(int(grid[nr, nc]))
    return out


def _ray_targets_rect_side(side: str, grid_shape: tuple[int, int], frame_bbox: BBox) -> bool:
    h, w = grid_shape
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return r0 <= c0 and r0 <= (w - 1 - c1)
    if side == "bottom":
        return (h - 1 - r1) <= c0 and (h - 1 - r1) <= (w - 1 - c1)
    if side == "left":
        return c0 <= r0 and c0 <= (h - 1 - r1)
    assert side == "right"
    return (w - 1 - c1) <= r0 and (w - 1 - c1) <= (h - 1 - r1)


def _semantic_check(grid: np.ndarray, expected_mode: str | None = None, expected_side: str | None = None) -> bool:
    g = np.asarray(grid, dtype=int)
    try:
        analysis = _analyze(g)
    except AssertionError:
        return False

    bg = analysis["bg"]
    rect_color = analysis["rect_color"]
    frame_color = analysis["frame_color"]
    rect_bbox = analysis["rect_bbox"]
    frame_bbox = analysis["frame_bbox"]
    active_hits = analysis["active_hits"]
    active_sides = analysis["active_sides"]
    fill_regions = analysis["fill_regions"]
    out = analysis["output"]

    if np.array_equal(out, g):
        return False

    rect_mask = g == rect_color
    frame_mask = g == frame_color
    if len(_mask_components(rect_mask)) != 1 or len(_mask_components(frame_mask)) != 1:
        return False

    # The objects must be separated by at least one background cell.
    for r, c in np.argwhere(rect_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if in_bounds(g.shape, nr, nc) and frame_mask[nr, nc]:
                return False

    if expected_mode == "all":
        if set(active_hits) != set(CORNER_ORDER):
            return False
        if set(active_sides) != {"top", "bottom", "left", "right"}:
            return False
    elif expected_mode == "single":
        if expected_side is None or active_sides != [expected_side]:
            return False
        if len(active_hits) != 2:
            return False

    if not fill_regions:
        return False
    if expected_side is not None and expected_side not in fill_regions:
        return False

    # Every filled chamber must touch the canvas boundary and be bounded only by frame/ray cells.
    for side, region in fill_regions.items():
        if not (region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any()):
            return False
        if np.any(g[region] != bg):
            return False
        if not np.all(out[region] == frame_color):
            return False
        if rect_color in _adjacent_colors(g, region):
            return False
        if not _ray_targets_rect_side(side, g.shape, frame_bbox):
            return False

    # Rays must add new rectangle-colored pixels, and inactive corners must not add any.
    ray_mask = analysis["ray_mask"]
    if not ray_mask.any():
        return False
    if not np.all(out[ray_mask] == rect_color):
        return False

    # Inactive diagonals in the single-side mode should hit non-corner frame pixels.
    if expected_mode == "single":
        frame_corners = set(_frame_corners(frame_bbox).values())
        inactive = [corner for corner in CORNER_ORDER if corner not in active_hits]
        if any(analysis["first_hits"][corner] in frame_corners for corner in inactive):
            return False

    # Fill regions should stay outside the frame interior.
    fr0, fc0, fr1, fc1 = frame_bbox
    frame_interior = np.zeros_like(g, dtype=bool)
    frame_interior[fr0 + 1 : fr1, fc0 + 1 : fc1] = True
    if any(np.any(region & frame_interior) for region in fill_regions.values()):
        return False

    # Background must remain dominant after generation; otherwise object parsing becomes unstable.
    if np.count_nonzero(g == bg) <= max(np.count_nonzero(rect_mask), np.count_nonzero(frame_mask)):
        return False

    # Output must contain both step colors.
    if not np.any((out == rect_color) & (g == bg)):
        return False
    if not np.any((out == frame_color) & (g == bg)):
        return False
    return True


def _build_case(
    h: int,
    w: int,
    bg: int,
    rect_color: int,
    frame_color: int,
    frame_top: int,
    frame_left: int,
    frame_side: int,
    gaps: tuple[int, int, int, int],
    rect_h: int,
    rect_w: int,
) -> np.ndarray:
    g = np.full((h, w), bg, dtype=int)
    frame_bottom = frame_top + frame_side - 1
    frame_right = frame_left + frame_side - 1

    g[frame_top, frame_left : frame_right + 1] = frame_color
    g[frame_bottom, frame_left : frame_right + 1] = frame_color
    g[frame_top : frame_bottom + 1, frame_left] = frame_color
    g[frame_top : frame_bottom + 1, frame_right] = frame_color

    gap_top, _gap_bottom, gap_left, _gap_right = gaps
    rect_top = frame_top + 1 + gap_top
    rect_left = frame_left + 1 + gap_left
    g[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = rect_color
    return g


def _make_single_side_case(rng: np.random.Generator) -> tuple[np.ndarray, str]:
    side = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    frame_side = int(rng.integers(7, 13))
    interior = frame_side - 2

    d_options = [d for d in range(1, (interior - 2) // 2 + 1)]
    if not d_options:
        raise ValueError("no valid diagonal gap")
    d = int(rng.choice(d_options))

    if side in {"top", "bottom"}:
        rect_w = interior - 2 * d
        t_options = [t for t in range(1, interior - d - 1) if t != d]
        if not t_options:
            raise ValueError("no valid vertical gap")
        t = int(rng.choice(t_options))
        rect_h = interior - d - t
        if rect_h < 2:
            raise ValueError("invalid rectangle height")
        gaps = (d, t, d, d) if side == "top" else (t, d, d, d)

        side_margin = int(rng.integers(2, 6))
        left_margin = int(rng.integers(side_margin, side_margin + 5))
        right_margin = int(rng.integers(side_margin, side_margin + 5))
        other_margin = int(rng.integers(1, 5))
        if side == "top":
            top_margin, bottom_margin = side_margin, other_margin
        else:
            top_margin, bottom_margin = other_margin, side_margin
    else:
        rect_h = interior - 2 * d
        t_options = [t for t in range(1, interior - d - 1) if t != d]
        if not t_options:
            raise ValueError("no valid horizontal gap")
        t = int(rng.choice(t_options))
        rect_w = interior - d - t
        if rect_w < 2:
            raise ValueError("invalid rectangle width")
        gaps = (d, d, d, t) if side == "left" else (d, d, t, d)

        side_margin = int(rng.integers(2, 6))
        top_margin = int(rng.integers(side_margin, side_margin + 5))
        bottom_margin = int(rng.integers(side_margin, side_margin + 5))
        other_margin = int(rng.integers(1, 5))
        if side == "left":
            left_margin, right_margin = side_margin, other_margin
        else:
            left_margin, right_margin = other_margin, side_margin

    h = top_margin + frame_side + bottom_margin
    w = left_margin + frame_side + right_margin
    if max(h, w) > 30:
        raise ValueError("grid too large")

    bg, rect_color, frame_color = map(int, rng.choice(ALL_COLORS, size=3, replace=False))
    grid = _build_case(
        h=h,
        w=w,
        bg=bg,
        rect_color=rect_color,
        frame_color=frame_color,
        frame_top=top_margin,
        frame_left=left_margin,
        frame_side=frame_side,
        gaps=gaps,
        rect_h=rect_h,
        rect_w=rect_w,
    )
    return grid, side


def _make_all_side_case(rng: np.random.Generator) -> np.ndarray:
    frame_side = int(rng.integers(8, 13))
    interior = frame_side - 2
    d_options = [d for d in range(1, (interior - 2) // 2 + 1)]
    if not d_options:
        raise ValueError("no valid equal gap")
    d = int(rng.choice(d_options))
    rect_side = interior - 2 * d
    if rect_side < 2:
        raise ValueError("invalid square rectangle")

    outer_margin = int(rng.integers(2, 6))
    h = frame_side + 2 * outer_margin
    w = frame_side + 2 * outer_margin
    if max(h, w) > 30:
        raise ValueError("grid too large")

    bg, rect_color, frame_color = map(int, rng.choice(ALL_COLORS, size=3, replace=False))
    return _build_case(
        h=h,
        w=w,
        bg=bg,
        rect_color=rect_color,
        frame_color=frame_color,
        frame_top=outer_margin,
        frame_left=outer_margin,
        frame_side=frame_side,
        gaps=(d, d, d, d),
        rect_h=rect_side,
        rect_w=rect_side,
    )


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        try:
            if rng.random() < 0.3:
                grid = _make_all_side_case(rng)
                if _semantic_check(grid, expected_mode="all"):
                    return grid
            else:
                grid, side = _make_single_side_case(rng)
                if _semantic_check(grid, expected_mode="single", expected_side=side):
                    return grid
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid ray-wall input")
