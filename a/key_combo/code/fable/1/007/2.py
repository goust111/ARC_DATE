from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, is_contiguous, paste


BLACK = int(Color.BLACK)
YELLOW = int(Color.YELLOW)
GREY = int(Color.GREY)

SPRITE_PALETTE = (
    int(Color.BLUE),
    int(Color.RED),
    int(Color.GREEN),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
EXTRA_COEFFS = ((-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))

GRID_ROWS_RANGE = (15, 24)
GRID_COLS_RANGE = (15, 24)
SPRITE_AREA_RANGE = (4, 8)
MIN_VECTOR_CHEB = 3
MAX_VECTOR_COMPONENT = 8
MIN_STAMP_COUNT = 4
MAX_STAMP_COUNT = 8
VECTOR_TRIES = 320
ANCHOR_TRIES = 16
MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]
Node = tuple[Coord, Coord, Coord]  # (translation, coeffs, node_anchor)


def _det(a: Coord, b: Coord) -> int:
    return int(a[0] * b[1] - a[1] * b[0])


def _patch_rel_coords(patch: np.ndarray, anchor_local: Coord, bg: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(patch != int(bg))
    rel_coords = coords - np.array(anchor_local, dtype=int)
    colors = patch[coords[:, 0], coords[:, 1]].astype(int)
    return rel_coords.astype(int), colors


def _valid_lattice_nodes(
    shape: tuple[int, int],
    rel_coords: np.ndarray,
    anchor: Coord,
    v1: Coord,
    v2: Coord,
) -> list[Node]:
    h, w = map(int, shape)
    anchor_vec = np.array(anchor, dtype=int)
    abs_coords = rel_coords + anchor_vec

    tr_lo = -int(abs_coords[:, 0].min())
    tr_hi = int(h - 1 - abs_coords[:, 0].max())
    tc_lo = -int(abs_coords[:, 1].min())
    tc_hi = int(w - 1 - abs_coords[:, 1].max())

    det_v = _det(v1, v2)
    assert det_v != 0, "generator points must not be collinear with the anchor"

    nodes: list[Node] = []
    for tr in range(tr_lo, tr_hi + 1):
        for tc in range(tc_lo, tc_hi + 1):
            t = (int(tr), int(tc))
            i_num = _det(t, v2)
            j_num = _det(v1, t)
            if i_num % det_v != 0 or j_num % det_v != 0:
                continue
            coeffs = (int(i_num // det_v), int(j_num // det_v))
            node_anchor = (int(anchor[0] + tr), int(anchor[1] + tc))
            nodes.append((t, coeffs, node_anchor))

    nodes.sort(key=lambda item: (item[1][0], item[1][1], item[2][0], item[2][1]))
    return nodes


def _render_lattice(
    shape: tuple[int, int],
    rel_coords: np.ndarray,
    colors: np.ndarray,
    anchor: Coord,
    v1: Coord,
    v2: Coord,
    bg: int,
) -> tuple[np.ndarray, list[Node]]:
    out = np.full(shape, int(bg), dtype=int)
    nodes = _valid_lattice_nodes(shape, rel_coords, anchor, v1, v2)

    rel_list = [tuple(map(int, rc)) for rc in rel_coords]
    color_list = [int(v) for v in colors]
    for _t, _coeffs, node_anchor in nodes:
        nr, nc = node_anchor
        for (dr, dc), color in zip(rel_list, color_list):
            rr = nr + dr
            cc = nc + dc
            cur = int(out[rr, cc])
            assert cur in {int(bg), color}, "overlapping stamps must agree in color"
            out[rr, cc] = color

    return out, nodes


def _extract_instance(grid: np.ndarray) -> tuple[int, Coord, list[Coord], np.ndarray, Coord]:
    bg = int(get_bg(grid))
    assert bg not in {YELLOW, GREY}, "background must differ from the yellow anchor and grey markers"

    yellow = np.argwhere(grid == YELLOW)
    assert yellow.shape == (1, 2), "expected exactly one yellow anchor pixel"
    anchor = tuple(map(int, yellow[0]))

    grey_parts = find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[GREY])
    assert len(grey_parts) == 2, "expected exactly two grey singleton markers"
    assert all(int(comp["area"]) == 1 for comp in grey_parts), "grey markers must be singleton pixels"
    grey_points = [tuple(map(int, comp["coords"][0])) for comp in grey_parts]

    sprite_colors = sorted(set(map(int, np.unique(grid))) - {bg, GREY})
    assert YELLOW in sprite_colors, "the sprite must contain the yellow anchor"
    sprite_parts = find_components(grid, bg=bg, connectivity=4, monochromatic=False, colors=sprite_colors)
    assert len(sprite_parts) == 1, "expected one connected multicolor sprite"
    sprite = sprite_parts[0]

    sprite_coords = [tuple(map(int, rc)) for rc in sprite["coords"]]
    sprite_coord_set = set(sprite_coords)
    assert anchor in sprite_coord_set, "the yellow pixel must belong to the sprite"
    assert all(point not in sprite_coord_set for point in grey_points), "grey markers must be separate from the sprite"
    assert int(np.count_nonzero(grid != bg)) == int(sprite["area"]) + 2, "unexpected extra foreground pixels"
    assert len({int(grid[r, c]) for r, c in sprite_coords}) >= 2, "the sprite must be multicolor"

    r0, c0, r1, c1 = map(int, sprite["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in sprite_coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    anchor_local = (int(anchor[0] - r0), int(anchor[1] - c0))
    return bg, anchor, grey_points, patch, anchor_local


def _touches_sprite(v: Coord, rel_coords: np.ndarray) -> bool:
    vr, vc = map(int, v)
    for rr, cc in rel_coords:
        if max(abs(vr - int(rr)), abs(vc - int(cc))) <= 1:
            return True
    return False


def _stamp_sets_clear(rel_coords: np.ndarray, nodes: list[Node]) -> bool:
    rel_list = [tuple(map(int, rc)) for rc in rel_coords]
    stamps: list[set[Coord]] = []
    for _translation, _coeffs, node_anchor in nodes:
        nr, nc = node_anchor
        stamps.append({(int(nr + dr), int(nc + dc)) for dr, dc in rel_list})

    for i in range(len(stamps)):
        for j in range(i + 1, len(stamps)):
            if stamps[i] & stamps[j]:
                return False
            for r1, c1 in stamps[i]:
                for r2, c2 in stamps[j]:
                    if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                        return False
    return True


def _make_sprite() -> tuple[np.ndarray, Coord] | None:
    for _ in range(100):
        canvas = np.zeros((5, 5), dtype=int)
        start = (random.randrange(5), random.randrange(5))
        cells = {start}
        target_area = random.randint(*SPRITE_AREA_RANGE)

        while len(cells) < target_area:
            frontier: list[Coord] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        coords = np.array(sorted(cells), dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if not (2 <= h <= 5 and 2 <= w <= 5):
            continue

        coords[:, 0] -= r0
        coords[:, 1] -= c0
        if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
            continue

        mask = np.zeros((h, w), dtype=int)
        mask[coords[:, 0], coords[:, 1]] = 1
        if not is_contiguous(mask, background=0, connectivity=4):
            continue

        occupied = [tuple(map(int, rc)) for rc in coords.tolist()]
        anchor_local = random.choice(occupied)
        patch = np.zeros((h, w), dtype=int)
        patch[anchor_local] = YELLOW

        others = [coord for coord in occupied if coord != anchor_local]
        color_count = random.randint(1, min(3, len(others)))
        palette = random.sample(SPRITE_PALETTE, color_count)
        random.shuffle(others)
        for idx, (r, c) in enumerate(others):
            patch[r, c] = palette[idx] if idx < len(palette) else random.choice(palette)
        return patch, anchor_local

    return None


def _sample_vector() -> Coord:
    while True:
        v = (random.randint(-MAX_VECTOR_COMPONENT, MAX_VECTOR_COMPONENT), random.randint(-MAX_VECTOR_COMPONENT, MAX_VECTOR_COMPONENT))
        if v != (0, 0) and max(abs(v[0]), abs(v[1])) >= MIN_VECTOR_CHEB:
            return (int(v[0]), int(v[1]))


def _anchor_range_for_translations(
    shape: tuple[int, int],
    rel_coords: np.ndarray,
    translations: list[Coord],
) -> tuple[int, int, int, int] | None:
    h, w = map(int, shape)
    min_r, min_c = rel_coords.min(axis=0)
    max_r, max_c = rel_coords.max(axis=0)

    r_lo = max(int(-min_r - tr) for tr, _tc in translations)
    r_hi = min(int(h - 1 - max_r - tr) for tr, _tc in translations)
    c_lo = max(int(-min_c - tc) for _tr, tc in translations)
    c_hi = min(int(w - 1 - max_c - tc) for _tr, tc in translations)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (r_lo, r_hi, c_lo, c_hi)


def _sample_anchor(r_lo: int, r_hi: int, c_lo: int, c_hi: int) -> Coord:
    if random.random() < 0.5:
        r = random.choice((r_lo, r_hi)) if r_lo != r_hi else r_lo
        c = random.choice((c_lo, c_hi)) if c_lo != c_hi else c_lo
        return (int(r), int(c))
    return (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))


def _semantic_ok(grid: np.ndarray, required_extra: Coord | None = None) -> bool:
    try:
        bg, anchor, grey_points, patch, anchor_local = _extract_instance(grid)
        rel_coords, colors = _patch_rel_coords(patch, anchor_local, bg)
        v1 = (int(grey_points[0][0] - anchor[0]), int(grey_points[0][1] - anchor[1]))
        v2 = (int(grey_points[1][0] - anchor[0]), int(grey_points[1][1] - anchor[1]))
        assert _det(v1, v2) != 0

        expected, nodes = _render_lattice(grid.shape, rel_coords, colors, anchor, v1, v2, bg)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        coeffs = {coeff for _translation, coeff, _node_anchor in nodes}
        if len(nodes) < MIN_STAMP_COUNT or len(nodes) > MAX_STAMP_COUNT:
            return False
        if not {(0, 0), (1, 0), (0, 1)} <= coeffs:
            return False
        if required_extra is not None and required_extra not in coeffs:
            return False
        if not any(coeff not in {(0, 0), (1, 0), (0, 1)} for coeff in coeffs):
            return False
        if not any((i < 0) or (j < 0) or (i != 0 and j != 0) for i, j in coeffs if (i, j) != (0, 0)):
            return False
        if np.any(actual == GREY):
            return False
        if int(np.count_nonzero(actual == YELLOW)) != len(nodes):
            return False
        if int(np.count_nonzero(actual != bg)) <= int(np.count_nonzero(grid != bg)):
            return False
        if not _stamp_sets_clear(rel_coords, nodes):
            return False

        sprite_cells = rel_coords + np.array(anchor, dtype=int)
        for point in grey_points:
            if any(max(abs(int(point[0] - rr)), abs(int(point[1] - cc))) <= 1 for rr, cc in sprite_cells):
                return False
        if max(abs(grey_points[0][0] - grey_points[1][0]), abs(grey_points[0][1] - grey_points[1][1])) <= 1:
            return False
    except Exception:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    made = _make_sprite()
    if made is None:
        return None
    patch, anchor_local = made
    rel_coords, colors = _patch_rel_coords(patch, anchor_local, BLACK)

    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    extra_coeff = random.choice(EXTRA_COEFFS)

    for _ in range(VECTOR_TRIES):
        v1 = _sample_vector()
        v2 = _sample_vector()
        if _det(v1, v2) == 0:
            continue
        if max(abs(v1[0] - v2[0]), abs(v1[1] - v2[1])) <= 1:
            continue
        if _touches_sprite(v1, rel_coords) or _touches_sprite(v2, rel_coords):
            continue

        extra_translation = (
            int(extra_coeff[0] * v1[0] + extra_coeff[1] * v2[0]),
            int(extra_coeff[0] * v1[1] + extra_coeff[1] * v2[1]),
        )
        anchor_range = _anchor_range_for_translations((h, w), rel_coords, [(0, 0), v1, v2, extra_translation])
        if anchor_range is None:
            continue
        r_lo, r_hi, c_lo, c_hi = anchor_range

        for _ in range(ANCHOR_TRIES):
            anchor = _sample_anchor(r_lo, r_hi, c_lo, c_hi)
            try:
                rendered, nodes = _render_lattice((h, w), rel_coords, colors, anchor, v1, v2, BLACK)
            except AssertionError:
                continue
            coeffs = {coeff for _translation, coeff, _node_anchor in nodes}

            if len(nodes) < MIN_STAMP_COUNT or len(nodes) > MAX_STAMP_COUNT:
                continue
            if not {(0, 0), (1, 0), (0, 1), extra_coeff} <= coeffs:
                continue
            if not any((i < 0) or (j < 0) or (i != 0 and j != 0) for i, j in coeffs if (i, j) != (0, 0)):
                continue
            if not _stamp_sets_clear(rel_coords, nodes):
                continue

            grid = np.full((h, w), BLACK, dtype=int)
            paste(grid, patch, anchor[0] - anchor_local[0], anchor[1] - anchor_local[1], transparent=BLACK)
            grey1 = (int(anchor[0] + v1[0]), int(anchor[1] + v1[1]))
            grey2 = (int(anchor[0] + v2[0]), int(anchor[1] + v2[1]))
            if int(grid[grey1]) != BLACK or int(grid[grey2]) != BLACK:
                continue
            if max(abs(grey1[0] - grey2[0]), abs(grey1[1] - grey2[1])) <= 1:
                continue
            grid[grey1] = GREY
            grid[grey2] = GREY

            if not np.array_equal(main(grid), rendered):
                continue
            if not _semantic_ok(grid, required_extra=extra_coeff):
                continue
            return grid

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the anchored sprite on every in-bounds node of the 2D lattice generated by the two grey points.

    Input structure:
    - A dominant background contains one connected multicolor sprite whose cells
      include exactly one yellow anchor pixel.
    - Exactly two separate grey singleton pixels lie elsewhere on the grid and
      act as markers rather than sprite cells.
    - No other foreground objects are present.

    Rule:
    - Use the yellow anchor as the origin and read the anchor-to-grey offsets as
      the two generator vectors v1 and v2.
    - Consider every lattice node reached by an integer combination i*v1 + j*v2.
    - Whenever shifting the whole sprite so that its yellow anchor lands on such
      a node keeps every sprite pixel inside the grid, stamp a full copy there.
      The grey markers themselves disappear and are replaced only by stamped
      sprite colors.

    Key insight:
    - The grey pixels are not two paste targets; together with the yellow anchor
      they define an entire 2D lattice, so unstated nodes like -v1, -v2, or
      v1+v2 must also receive the sprite whenever the full copy fits.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    bg, anchor, grey_points, patch, anchor_local = _extract_instance(g)
    rel_coords, colors = _patch_rel_coords(patch, anchor_local, bg)

    v1 = (int(grey_points[0][0] - anchor[0]), int(grey_points[0][1] - anchor[1]))
    v2 = (int(grey_points[1][0] - anchor[0]), int(grey_points[1][1] - anchor[1]))
    assert _det(v1, v2) != 0, "the two grey points must not be collinear with the anchor"

    out, _nodes = _render_lattice(g.shape, rel_coords, colors, anchor, v1, v2, bg)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-stamping instance")
