from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


QUADRANTS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SIZE_OPTIONS: tuple[tuple[int, int], ...] = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
)
PAIR_COUNT_RANGE = (2, 4)
INTERIOR_GAP_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _foreground_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != bg)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < int(shape[0]) and 0 <= c < int(shape[1])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = map(int, grid.shape)

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _bbox_shape(box: BBox) -> tuple[int, int]:
    return int(box[2] - box[0] + 1), int(box[3] - box[1] + 1)


def _bbox_in_bounds(box: BBox, shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < int(shape[0]) and 0 <= c0 <= c1 < int(shape[1])


def _solid_rectangle_bbox(grid: np.ndarray, color: int, coords: np.ndarray) -> BBox:
    box = _bbox_from_coords(coords)
    h, w = _bbox_shape(box)
    assert int(coords.shape[0]) == h * w, "body component must be a solid rectangle"
    r0, c0, r1, c1 = box
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))
    return box


def _body_corner(box: BBox, sr: int, sc: int) -> Coord:
    return (int(box[2] if sr > 0 else box[0]), int(box[3] if sc > 0 else box[1]))


def _partner_box_from_near(near: Coord, h: int, w: int, sr: int, sc: int) -> BBox:
    nr, nc = near
    r0 = nr if sr > 0 else nr - h + 1
    r1 = nr + h - 1 if sr > 0 else nr
    c0 = nc if sc > 0 else nc - w + 1
    c1 = nc + w - 1 if sc > 0 else nc
    return int(r0), int(c0), int(r1), int(c1)


def _far_corner(box: BBox, sr: int, sc: int) -> Coord:
    return (int(box[2] if sr > 0 else box[0]), int(box[3] if sc > 0 else box[1]))


def _body_corners(box: BBox) -> tuple[Coord, Coord, Coord, Coord]:
    r0, c0, r1, c1 = box
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _anchor_avoids_body_diagonals(body: BBox, anchor: Coord) -> bool:
    ar, ac = anchor
    return all(abs(ar - br) != abs(ac - bc) for br, bc in _body_corners(body))


def _line_cells(start: Coord, sr: int, sc: int, steps: int) -> list[Coord]:
    return [(int(start[0] + sr * k), int(start[1] + sc * k)) for k in range(1, steps)]


def _enumerate_candidates(grid: np.ndarray, color: int, bg: int) -> tuple[BBox, Coord, list[dict[str, object]]]:
    comps = _components_of_color(grid, color)
    assert len(comps) == 2, "each color must appear as one rectangle and one isolated pixel"
    comps.sort(key=lambda coords: int(coords.shape[0]), reverse=True)
    body_coords, dot_coords = comps
    assert int(body_coords.shape[0]) > 1 and int(dot_coords.shape[0]) == 1

    body = _solid_rectangle_bbox(grid, color, body_coords)
    h, w = _bbox_shape(body)
    anchor = tuple(map(int, dot_coords[0]))
    assert _anchor_avoids_body_diagonals(body, anchor)

    candidates: list[dict[str, object]] = []
    for sr, sc in QUADRANTS:
        br, bc = _body_corner(body, sr, sc)
        near = (int(anchor[0] - sr * (h - 1)), int(anchor[1] - sc * (w - 1)))
        dr = int(near[0] - br)
        dc = int(near[1] - bc)
        if dr == 0 or dc == 0 or abs(dr) != abs(dc):
            continue
        if (1 if dr > 0 else -1) != sr or (1 if dc > 0 else -1) != sc:
            continue

        partner = _partner_box_from_near(near, h, w, sr, sc)
        if not _bbox_in_bounds(partner, grid.shape):
            continue

        ok = True
        for cell in _bbox_cells(partner):
            if cell == anchor:
                continue
            if int(grid[cell]) != bg:
                ok = False
                break
        if not ok:
            continue

        steps = abs(dr)
        line = _line_cells((br, bc), sr, sc, steps)
        for cell in line:
            if int(grid[cell]) != bg:
                ok = False
                break
        if not ok:
            continue

        candidates.append(
            {
                "body": body,
                "anchor": anchor,
                "partner": partner,
                "near": near,
                "body_corner": (br, bc),
                "line": line,
                "size": (h, w),
                "direction": (sr, sc),
            }
        )

    return body, anchor, candidates


def _paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each missing rectangle from its far-corner dot, then draw the diagonal connector between the two facing near corners.

    Input structure:
    - The background color is dominant.
    - Each foreground color appears exactly twice: one solid rectangular body and one isolated single-pixel anchor.
    - The anchor is the far corner of a same-sized missing partner rectangle, and the anchor itself is not on any diagonal extension of the body's corners.

    Rule:
    - For each color, read the body's height and width from the visible solid rectangle.
    - Treat the isolated pixel as the missing partner rectangle's far corner and expand a same-sized solid rectangle back toward the body.
    - Only one expansion direction makes the reconstructed partner's near corner lie on a 45-degree diagonal with the corresponding body corner; draw that rectangle and fill the background cells strictly between those two near corners along the diagonal.

    Key insight:
    - The lone pixel is not a line endpoint: it is the hidden rectangle's far-corner anchor, so the diagonal can only be found after reconstructing the full partner and identifying the two facing near corners.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in _foreground_colors(g, bg):
        _body, _anchor, candidates = _enumerate_candidates(g, color, bg)
        assert len(candidates) == 1, "anchor direction must be uniquely determined"
        candidate = candidates[0]
        partner = candidate["partner"]
        line = candidate["line"]

        _paint_rect(out, partner, color)
        for r, c in line:
            assert int(out[r, c]) == bg, "connector must fill only background cells"
            out[r, c] = int(color)

    return out


def _dilate_cells(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    height, width = map(int, shape)
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _plan_sizes(n_pairs: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    if n_pairs >= 2:
        sizes.append(random.choice([size for size in SIZE_OPTIONS if size[0] > size[1]]))
        sizes.append(random.choice([size for size in SIZE_OPTIONS if size[0] < size[1]]))
    while len(sizes) < n_pairs:
        sizes.append(random.choice(SIZE_OPTIONS))
    random.shuffle(sizes)
    return sizes


def _plan_quadrants(n_pairs: int) -> list[tuple[int, int]]:
    quadrants = random.sample(list(QUADRANTS), k=min(n_pairs, len(QUADRANTS)))
    while len(quadrants) < n_pairs:
        quadrants.append(random.choice(QUADRANTS))
    random.shuffle(quadrants)
    return quadrants


def _pair_candidate(
    shape: tuple[int, int],
    h: int,
    w: int,
    sr: int,
    sc: int,
    interior_gap: int,
    reserved: set[Coord],
) -> dict[str, object] | None:
    height, width = map(int, shape)
    steps = interior_gap + 1

    for _ in range(PLACEMENT_ATTEMPTS):
        body = (
            random.randint(0, height - h),
            random.randint(0, width - w),
            0,
            0,
        )
        body = (body[0], body[1], body[0] + h - 1, body[1] + w - 1)
        body_corner = _body_corner(body, sr, sc)
        near = (int(body_corner[0] + sr * steps), int(body_corner[1] + sc * steps))
        partner = _partner_box_from_near(near, h, w, sr, sc)
        if not _bbox_in_bounds(partner, shape):
            continue
        anchor = _far_corner(partner, sr, sc)
        if not _anchor_avoids_body_diagonals(body, anchor):
            continue

        line = _line_cells(body_corner, sr, sc, steps)
        footprint = set(_bbox_cells(body)) | set(_bbox_cells(partner)) | set(line)
        if footprint & reserved:
            continue

        return {
            "body": body,
            "partner": partner,
            "anchor": anchor,
            "line": line,
            "body_corner": body_corner,
            "near": near,
            "size": (h, w),
            "direction": (sr, sc),
            "gap": interior_gap,
        }
    return None


def _build_from_pairs(shape: tuple[int, int], bg: int, pairs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)

    for pair in pairs:
        color = int(pair["color"])
        body = pair["body"]
        partner = pair["partner"]
        anchor = pair["anchor"]
        line = pair["line"]

        _paint_rect(inp, body, color)
        inp[anchor] = color

        _paint_rect(out, body, color)
        _paint_rect(out, partner, color)
        for r, c in line:
            out[r, c] = color

    return inp, out


def _validate_generated(inp: np.ndarray, out: np.ndarray, pairs: list[dict[str, object]], bg: int) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.array_equal(_build_from_pairs(inp.shape, bg, pairs)[1], out):
        return False
    if np.array_equal(inp, out):
        return False
    if _get_bg(inp) != bg:
        return False

    for pair in pairs:
        color = int(pair["color"])
        body = pair["body"]
        partner = pair["partner"]
        anchor = pair["anchor"]
        line = pair["line"]

        if not _anchor_avoids_body_diagonals(body, anchor):
            return False
        if len(line) < INTERIOR_GAP_RANGE[0]:
            return False

        for cell in _bbox_cells(partner):
            if cell == anchor:
                if int(inp[cell]) != color:
                    return False
            elif int(inp[cell]) != bg:
                return False
        for cell in line:
            if int(inp[cell]) != bg:
                return False

        try:
            body2, anchor2, candidates = _enumerate_candidates(inp, color, bg)
        except Exception:
            return False
        if body2 != body or anchor2 != anchor or len(candidates) != 1:
            return False
        if candidates[0]["partner"] != partner or candidates[0]["line"] != line:
            return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(range(10), k=n_pairs + 1)
    bg = colors[0]
    fg_colors = colors[1:]
    sizes = _plan_sizes(n_pairs)
    quadrants = _plan_quadrants(n_pairs)
    gaps = [random.randint(*INTERIOR_GAP_RANGE) for _ in range(n_pairs)]

    base = 15 + 3 * n_pairs
    height = random.randint(base, min(30, base + 4))
    width = random.randint(base, min(30, base + 4))

    plans = []
    for color, size, quadrant, gap in zip(fg_colors, sizes, quadrants, gaps):
        h, w = size
        sr, sc = quadrant
        plans.append(
            {
                "color": color,
                "size": (h, w),
                "direction": (sr, sc),
                "gap": gap,
                "priority": 2 * h * w + gap,
            }
        )
    plans.sort(key=lambda plan: int(plan["priority"]), reverse=True)

    reserved: set[Coord] = set()
    pairs: list[dict[str, object]] = []
    for plan in plans:
        h, w = plan["size"]
        sr, sc = plan["direction"]
        gap = int(plan["gap"])
        pair = _pair_candidate((height, width), h, w, sr, sc, gap, reserved)
        if pair is None:
            return None
        pair["color"] = int(plan["color"])
        pairs.append(pair)
        footprint = set(_bbox_cells(pair["body"])) | set(_bbox_cells(pair["partner"])) | set(pair["line"])
        reserved |= _dilate_cells(footprint, (height, width))

    inp, out = _build_from_pairs((height, width), bg, pairs)
    if not _validate_generated(inp, out, pairs, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid far-corner anchored rectangle puzzle")
