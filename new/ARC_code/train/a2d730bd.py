import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random


Coord = tuple[int, int]


def _claim(g: np.ndarray, claims: dict[Coord, int], coord: Coord, color: int, bg: int) -> None:
    h, w = g.shape
    r, c = coord
    assert 0 <= r < h and 0 <= c < w, (coord, g.shape)
    assert int(g[r, c]) in (bg, color), (coord, int(g[r, c]), bg, color)
    previous = claims.get(coord)
    assert previous is None or previous == color, (coord, previous, color)
    claims[coord] = color


def _claim_range(
    g: np.ndarray,
    claims: dict[Coord, int],
    start: Coord,
    end: Coord,
    step: Coord,
    skip: Coord,
    color: int,
    bg: int,
) -> None:
    r, c = start
    er, ec = end
    dr, dc = step
    while True:
        if (r, c) != skip:
            _claim(g, claims, (r, c), color, bg)
        if (r, c) == (er, ec):
            break
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each solid foreground rectangle toward its same-color isolated pixels with capped one-cell paths, clearing each original marker pixel.

    Input structure
    - The background is the dominant color.
    - Each foreground color has exactly one solid rectangular object and one or more isolated same-color marker pixels.
    - Every marker is horizontally or vertically aligned with the rectangle bbox and lies outside it.
    Rule
    - For each marker, draw a one-cell-wide straight path from the adjacent side of the rectangle through one cell beyond the marker.
    - Add a three-pixel perpendicular cap at the rectangle end and around the marker endpoint.
    - Recolor the original marker pixel to background.
    Key insight
    - The isolated pixels are not copied; they mark the hollow endpoints of capped protrusions grown from their matching rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    claims: dict[Coord, int] = {}
    clears: set[Coord] = set()

    for color, comps in by_color.items():
        rectangles = [comp for comp in comps if int(comp["area"]) > 1]
        markers = [comp for comp in comps if int(comp["area"]) == 1]
        assert len(rectangles) == 1, (color, [(comp["area"], comp["bbox"]) for comp in comps])
        assert markers and len(rectangles) + len(markers) == len(comps)

        rect = rectangles[0]
        r0, c0, r1, c1 = map(int, rect["bbox"])
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        assert int(rect["area"]) == rect_h * rect_w
        assert min(rect_h, rect_w) >= 2

        for marker in markers:
            mr, mc = map(int, marker["coords"][0])
            horizontal = r0 <= mr <= r1 and (mc < c0 or mc > c1)
            vertical = c0 <= mc <= c1 and (mr < r0 or mr > r1)
            assert horizontal ^ vertical, (color, (mr, mc), rect["bbox"])

            if horizontal:
                dc = -1 if mc < c0 else 1
                side_c = c0 - 1 if dc < 0 else c1 + 1
                for rr in (mr - 1, mr, mr + 1):
                    _claim(g, claims, (rr, side_c), color, bg)
                for rr in (mr - 1, mr + 1):
                    _claim(g, claims, (rr, mc), color, bg)
                _claim_range(
                    g,
                    claims,
                    (mr, side_c),
                    (mr, mc + dc),
                    (0, dc),
                    (mr, mc),
                    color,
                    bg,
                )
            else:
                dr = -1 if mr < r0 else 1
                side_r = r0 - 1 if dr < 0 else r1 + 1
                for cc in (mc - 1, mc, mc + 1):
                    _claim(g, claims, (side_r, cc), color, bg)
                for cc in (mc - 1, mc + 1):
                    _claim(g, claims, (mr, cc), color, bg)
                _claim_range(
                    g,
                    claims,
                    (side_r, mc),
                    (mr + dr, mc),
                    (dr, 0),
                    (mr, mc),
                    color,
                    bg,
                )

            clears.add((mr, mc))

    assert not (set(claims) & clears)

    out = g.copy()
    for (r, c), color in claims.items():
        out[r, c] = color
    for r, c in clears:
        out[r, c] = bg
    return out

# --------------- generator (att2) ---------------

GRID_ROWS = (13, 30)
GRID_COLS = (16, 30)
PALETTE = tuple(range(10))
FOREGROUND_COLOR_COUNT = (1, 3)
RECT_SIZE = (2, 11)
MARKER_COUNT = (1, 3)
MARKER_DISTANCE = (3, 10)
SAME_SIDE_MARKER_GAP = 3
SIDES = ("top", "bottom", "left", "right")
PATCH_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 30
GENERATE_ATTEMPTS = 300

def _gen_marker_cells(rect_h: int, rect_w: int, side: str, offset: int, dist: int):
    cells: set[tuple[int, int]] = set()
    if side == "top":
        marker = (-dist, offset)
        for cc in range(offset - 1, offset + 2):
            cells.add((-1, cc))
        for rr in range(-1, -dist - 2, -1):
            if rr != marker[0]:
                cells.add((rr, offset))
        cells.add((marker[0], offset - 1))
        cells.add((marker[0], offset + 1))
    elif side == "bottom":
        marker = (rect_h - 1 + dist, offset)
        for cc in range(offset - 1, offset + 2):
            cells.add((rect_h, cc))
        for rr in range(rect_h, marker[0] + 2):
            if rr != marker[0]:
                cells.add((rr, offset))
        cells.add((marker[0], offset - 1))
        cells.add((marker[0], offset + 1))
    elif side == "left":
        marker = (offset, -dist)
        for rr in range(offset - 1, offset + 2):
            cells.add((rr, -1))
        for cc in range(-1, -dist - 2, -1):
            if cc != marker[1]:
                cells.add((offset, cc))
        cells.add((offset - 1, marker[1]))
        cells.add((offset + 1, marker[1]))
    else:
        marker = (offset, rect_w - 1 + dist)
        for rr in range(offset - 1, offset + 2):
            cells.add((rr, rect_w))
        for cc in range(rect_w, marker[1] + 2):
            if cc != marker[1]:
                cells.add((offset, cc))
        cells.add((offset - 1, marker[1]))
        cells.add((offset + 1, marker[1]))
    return marker, cells


def _updated_margins(
    margins: tuple[int, int, int, int],
    side: str,
    dist: int,
) -> tuple[int, int, int, int]:
    top_extra, bottom_extra, left_extra, right_extra = margins
    if side == "top":
        top_extra = max(top_extra, dist + 1)
    elif side == "bottom":
        bottom_extra = max(bottom_extra, dist + 1)
    elif side == "left":
        left_extra = max(left_extra, dist + 1)
    else:
        right_extra = max(right_extra, dist + 1)
    return top_extra, bottom_extra, left_extra, right_extra


def _gen_marker_options(grid_h, grid_w, rect_h, rect_w, side_offsets, margins, markers, protrusions):
    options = []
    for side in SIDES:
        offsets = range(1, rect_w - 1) if side in ("top", "bottom") else range(1, rect_h - 1)
        for offset in offsets:
            if any(abs(offset - old) < SAME_SIDE_MARKER_GAP for old in side_offsets[side]):
                continue
            for dist in range(MARKER_DISTANCE[0], MARKER_DISTANCE[1] + 1):
                next_margins = _updated_margins(margins, side, dist)
                ntop, nbottom, nleft, nright = next_margins
                if ntop + rect_h + nbottom > grid_h:
                    continue
                if nleft + rect_w + nright > grid_w:
                    continue
                marker, cells = _gen_marker_cells(rect_h, rect_w, side, offset, dist)
                if marker in protrusions:
                    continue
                if any(cell in markers for cell in cells):
                    continue
                if protrusions & cells:
                    continue
                options.append((side, offset, dist, marker, cells, next_margins))
    return options


def _gen_make_patch(grid_h: int, grid_w: int, color: int, bg: int):
    for _ in range(PATCH_ATTEMPTS):
        rect_h = random.randint(RECT_SIZE[0], min(RECT_SIZE[1], grid_h))
        rect_w = random.randint(RECT_SIZE[0], min(RECT_SIZE[1], grid_w))
        if rect_h == 2 and rect_w == 2:
            continue

        marker_n = random.randint(MARKER_COUNT[0], MARKER_COUNT[1])
        side_offsets: dict[str, list[int]] = {side: [] for side in SIDES}
        margins = (0, 0, 0, 0)
        markers: list[tuple[int, int]] = []
        protrusions: set[tuple[int, int]] = set()

        for _marker_i in range(marker_n):
            options = _gen_marker_options(
                grid_h,
                grid_w,
                rect_h,
                rect_w,
                side_offsets,
                margins,
                markers,
                protrusions,
            )

            if not options:
                break
            side, offset, _dist, marker, cells, margins = random.choice(options)
            side_offsets[side].append(offset)
            markers.append(marker)
            protrusions.update(cells)
        else:
            rect_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
            marker_cells = set(markers)
            if protrusions & rect_cells or protrusions & marker_cells:
                continue

            reserved_cells = rect_cells | marker_cells | protrusions
            rows = [r for r, _c in reserved_cells]
            cols = [c for _r, c in reserved_cells]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            patch_h = max_r - min_r + 1
            patch_w = max_c - min_c + 1
            if patch_h > grid_h or patch_w > grid_w:
                continue

            out_patch = np.full((patch_h, patch_w), bg, dtype=int)
            inp_patch = np.full((patch_h, patch_w), bg, dtype=int)
            reserve_patch = np.full((patch_h, patch_w), bg, dtype=int)
            for r, c in rect_cells | protrusions:
                out_patch[r - min_r, c - min_c] = color
            for r, c in rect_cells | marker_cells:
                inp_patch[r - min_r, c - min_c] = color
            for r, c in reserved_cells:
                reserve_patch[r - min_r, c - min_c] = color
            return out_patch, inp_patch, reserve_patch
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_ROWS[0], GRID_ROWS[1])
    w = random.randint(GRID_COLS[0], GRID_COLS[1])
    bg = random.choice(PALETTE)
    colors = random.sample(
        [c for c in PALETTE if c != bg],
        random.randint(FOREGROUND_COLOR_COUNT[0], FOREGROUND_COLOR_COUNT[1]),
    )

    out = np.full((h, w), bg, dtype=int)
    inp = np.full((h, w), bg, dtype=int)
    reserved = np.full((h, w), bg, dtype=int)

    for color in colors:
        for _ in range(PLACEMENT_ATTEMPTS):
            patches = _gen_make_patch(h, w, color, bg)
            if patches is None:
                continue
            out_patch, inp_patch, reserve_patch = patches
            try:
                r0, c0 = random_free_location(reserved, reserve_patch, transparent=bg)
            except ValueError:
                continue
            paste(out, out_patch, r0, c0, transparent=bg)
            paste(inp, inp_patch, r0, c0, transparent=bg)
            paste(reserved, reserve_patch, r0, c0, transparent=bg)
            break
        else:
            return None

    counts = np.bincount(inp.ravel(), minlength=10)
    if counts[bg] <= max(counts[c] for c in range(10) if c != bg):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATE_ATTEMPTS} attempts")
