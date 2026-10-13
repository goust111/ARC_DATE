from __future__ import annotations

import importlib.util
import itertools
import random
from dataclasses import dataclass

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"
_ARC_SPEC = importlib.util.spec_from_file_location("_arc_utils_022_4", _ARC_UTILS_PATH)
assert _ARC_SPEC is not None and _ARC_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
CYAN = int(Color.TEAL)

NWSE = "nwse"
NESW = "nesw"
CORNERS = ("NW", "NE", "SW", "SE")
GUIDE_MARGIN = 4
GUIDE_GAP = 6
RETRY_LIMIT = 400


@dataclass(frozen=True)
class Motif:
    r: int
    c: int
    k: int
    diag: str


def _corner_top_left(center_r: int, center_c: int, k: int, corner: str) -> tuple[int, int]:
    if corner == "NW":
        return center_r - k, center_c - k
    if corner == "NE":
        return center_r - k, center_c + 1
    if corner == "SW":
        return center_r + 1, center_c - k
    if corner == "SE":
        return center_r + 1, center_c + 1
    raise ValueError(f"unknown corner {corner}")


def _diag_corners(diag: str) -> set[str]:
    if diag == NWSE:
        return {"NW", "SE"}
    if diag == NESW:
        return {"NE", "SW"}
    raise ValueError(f"unknown diagonal {diag}")


def _paint_block(grid: np.ndarray, r0: int, c0: int, k: int, color: int) -> None:
    paste(grid, np.full((k, k), int(color), dtype=int), int(r0), int(c0))


def _paint_motif(grid: np.ndarray, motif: Motif) -> None:
    for corner in _diag_corners(motif.diag):
        r0, c0 = _corner_top_left(motif.r, motif.c, motif.k, corner)
        _paint_block(grid, r0, c0, motif.k, GREEN)


def _red_guides(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == RED)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == RED)]
    assert len(rows) >= 2 and len(cols) >= 2, "expected both horizontal and vertical full red guide lines"
    return rows, cols


def _detect_motif_at(grid: np.ndarray, center_r: int, center_c: int, k: int) -> Motif | None:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    if center_r - k < 0 or center_r + k >= H or center_c - k < 0 or center_c + k >= W:
        return None

    if not np.all(g[center_r, center_c - k : center_c + k + 1] == BLACK):
        return None
    if not np.all(g[center_r - k : center_r + k + 1, center_c] == BLACK):
        return None

    filled: set[str] = set()
    for corner in CORNERS:
        r0, c0 = _corner_top_left(center_r, center_c, k, corner)
        block = g[r0 : r0 + k, c0 : c0 + k]
        if np.all(block == GREEN):
            filled.add(corner)
        elif not np.all(block == BLACK):
            return None

    if filled == {"NW", "SE"}:
        return Motif(int(center_r), int(center_c), int(k), NWSE)
    if filled == {"NE", "SW"}:
        return Motif(int(center_r), int(center_c), int(k), NESW)
    return None


def _find_motifs(grid: np.ndarray) -> tuple[list[int], list[int], list[Motif]]:
    rows, cols = _red_guides(grid)
    row_set = set(rows)
    col_set = set(cols)
    motifs: list[Motif] = []
    for r in range(grid.shape[0]):
        if r in row_set:
            continue
        for c in range(grid.shape[1]):
            if c in col_set:
                continue
            motif = None
            for k in (3, 2):
                motif = _detect_motif_at(grid, r, c, k)
                if motif is not None:
                    break
            if motif is not None:
                motifs.append(motif)

    assert motifs, "expected at least one diagonal motif"
    motifs.sort(key=lambda m: (m.c, m.r))
    return rows, cols, motifs


def _paint_vertical_ray(out: np.ndarray, col: int) -> None:
    for r in range(out.shape[0]):
        if int(out[r, col]) == BLACK:
            out[r, col] = BLUE


def _paint_horizontal_ray(out: np.ndarray, row: int) -> None:
    for c in range(out.shape[1]):
        if int(out[row, c]) == BLACK:
            out[row, c] = BLUE


def _stamp(out: np.ndarray, center_r: int, center_c: int, motif: Motif, *, use_complement: bool) -> None:
    protected = {RED, GREEN, BLUE}
    cyan_corners = set(CORNERS) - _diag_corners(motif.diag) if use_complement else _diag_corners(motif.diag)
    H, W = out.shape

    for corner in CORNERS:
        target = CYAN if corner in cyan_corners else BLACK
        r0, c0 = _corner_top_left(center_r, center_c, motif.k, corner)
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(H, r0 + motif.k)
        cc1 = min(W, c0 + motif.k)
        for rr in range(rr0, rr1):
            for cc in range(cc0, cc1):
                if int(out[rr, cc]) not in protected:
                    out[rr, cc] = target


def _render_with_motifs(
    grid: np.ndarray,
    guide_rows: list[int],
    guide_cols: list[int],
    motifs: list[Motif],
    *,
    swap_axis: bool = False,
    use_complement: bool = True,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for motif in sorted(motifs, key=lambda m: (m.c, m.r)):
        vertical = motif.diag == NWSE
        if swap_axis:
            vertical = not vertical

        if vertical:
            _paint_vertical_ray(out, motif.c)
            for rr in guide_rows:
                _stamp(out, rr, motif.c, motif, use_complement=use_complement)
        else:
            _paint_horizontal_ray(out, motif.r)
            for cc in guide_cols:
                _stamp(out, motif.r, cc, motif, use_complement=use_complement)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project blue rays from diagonal green block motifs toward a red guide lattice and stamp the cyan complementary diagonal at the crossings.

    Input structure
    - The background is black with several full-span horizontal and vertical red guide lines.
    - Between the guide lines sit square motifs: four equal corner blocks surround a black center row and center column.
    - Exactly one diagonal pair of corner blocks is solid green and the opposite diagonal pair is black.

    Rule
    - Detect each motif's filled diagonal: northwest-southeast or northeast-southwest.
    - Northwest-southeast motifs emit blue rays vertically through their center column; northeast-southwest motifs emit blue rays horizontally through their center row.
    - Rays paint only black cells blue and skip over any non-black cell without stopping.
    - At every other red-guide crossing on that ray, stamp a same-size complementary motif: the source's green-corner blocks become black there, the source's black-corner blocks become cyan there, while red, green, and blue pixels stay unchanged.
    - Process motifs from left to right; a later stamp can overwrite earlier cyan with black or cyan inside unprotected cells.

    Key insight
    - The same diagonal analysis controls two outputs at once: it chooses the ray axis and it chooses which corner blocks the stamped complement fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(get_bg(g)) == BLACK, "expected black to be the dominant background"

    guide_rows, guide_cols, motifs = _find_motifs(g)
    return _render_with_motifs(g, guide_rows, guide_cols, motifs)


def _spaced_positions(limit: int, count: int, margin: int, gap: int) -> list[int]:
    candidates = range(margin, limit - margin)
    options = [
        combo
        for combo in itertools.combinations(candidates, count)
        if all(b - a >= gap for a, b in zip(combo, combo[1:]))
    ]
    if not options:
        return []
    return list(random.choice(options))


def _sample_diagonals(count: int) -> list[str]:
    if count == 1:
        return [random.choice([NWSE, NESW])]
    diagonals = [NWSE, NESW]
    while len(diagonals) < count:
        diagonals.append(random.choice([NWSE, NESW]))
    random.shuffle(diagonals)
    return diagonals


def _sample_sizes(count: int) -> list[int]:
    if count == 1:
        return [random.choice([2, 3])]
    sizes = [2, 3]
    while len(sizes) < count:
        sizes.append(random.choice([2, 3]))
    random.shuffle(sizes)
    return sizes


def _build_input(shape: tuple[int, int], guide_rows: list[int], guide_cols: list[int], motifs: list[Motif]) -> np.ndarray:
    H, W = shape
    grid = np.full((H, W), BLACK, dtype=int)
    grid[guide_rows, :] = RED
    grid[:, guide_cols] = RED
    for motif in motifs:
        _paint_motif(grid, motif)
    return grid


def _motif_tuples(motifs: list[Motif]) -> set[tuple[int, int, int, str]]:
    return {(m.r, m.c, m.k, m.diag) for m in motifs}


def _motif_boxes_do_not_overlap(motifs: list[Motif]) -> bool:
    boxes = []
    for motif in motifs:
        box = (motif.r - motif.k, motif.c - motif.k, motif.r + motif.k, motif.c + motif.k)
        for r0, c0, r1, c1 in boxes:
            if not (box[2] < r0 or r1 < box[0] or box[3] < c0 or c1 < box[1]):
                return False
        boxes.append(box)
    return True


def _visible_stamp_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(grid) == CYAN))


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, guide_rows: list[int], guide_cols: list[int], motifs: list[Motif]) -> bool:
    try:
        parsed_rows, parsed_cols, parsed_motifs = _find_motifs(inp)
    except AssertionError:
        return False

    if parsed_rows != guide_rows or parsed_cols != guide_cols:
        return False
    if _motif_tuples(parsed_motifs) != _motif_tuples(motifs):
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.any(expected == BLUE) or _visible_stamp_count(expected) == 0:
        return False
    if np.array_equal(expected, _render_with_motifs(inp, guide_rows, guide_cols, motifs, swap_axis=True)):
        return False
    if np.array_equal(expected, _render_with_motifs(inp, guide_rows, guide_cols, motifs, use_complement=False)):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _center_candidates(
    shape: tuple[int, int], guide_rows: list[int], guide_cols: list[int], k: int
) -> list[tuple[int, int]]:
    H, W = shape
    candidates = []
    for r in range(k, H - k):
        if any(abs(r - rr) <= k for rr in guide_rows):
            continue
        for c in range(k, W - k):
            if any(abs(c - cc) <= k for cc in guide_cols):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_motifs(
    shape: tuple[int, int], guide_rows: list[int], guide_cols: list[int], sizes: list[int], diagonals: list[str]
) -> list[Motif] | None:
    specs = sorted(list(zip(sizes, diagonals)), reverse=True)

    def fits(candidate: Motif, placed: list[Motif]) -> bool:
        if any(candidate.r == other.r or candidate.c == other.c for other in placed):
            return False
        box = (candidate.r - candidate.k, candidate.c - candidate.k, candidate.r + candidate.k, candidate.c + candidate.k)
        for other in placed:
            other_box = (other.r - other.k, other.c - other.k, other.r + other.k, other.c + other.k)
            if not (box[2] < other_box[0] or other_box[2] < box[0] or box[3] < other_box[1] or other_box[3] < box[1]):
                return False
        return True

    def rec(i: int, placed: list[Motif]) -> list[Motif] | None:
        if i == len(specs):
            return placed
        k, diag = specs[i]
        for r, c in _center_candidates(shape, guide_rows, guide_cols, k):
            motif = Motif(int(r), int(c), int(k), str(diag))
            if fits(motif, placed):
                result = rec(i + 1, placed + [motif])
                if result is not None:
                    return result
        return None

    return rec(0, [])


def _gen_attempt() -> np.ndarray | None:
    guide_row_count = random.randint(2, 4)
    guide_col_count = random.randint(2, 4)
    motif_count = random.randint(2, min(3, guide_row_count, guide_col_count))

    min_h = 2 * GUIDE_MARGIN + 1 + GUIDE_GAP * (guide_row_count - 1)
    min_w = 2 * GUIDE_MARGIN + 1 + GUIDE_GAP * (guide_col_count - 1)
    H = random.randint(min_h, 29)
    W = random.randint(min_w, 29)

    guide_rows = _spaced_positions(H, guide_row_count, GUIDE_MARGIN, GUIDE_GAP)
    guide_cols = _spaced_positions(W, guide_col_count, GUIDE_MARGIN, GUIDE_GAP)
    if len(guide_rows) != guide_row_count or len(guide_cols) != guide_col_count:
        return None

    diagonals = _sample_diagonals(motif_count)
    sizes = _sample_sizes(motif_count)
    motifs = _place_motifs((H, W), guide_rows, guide_cols, sizes, diagonals)
    if motifs is None:
        return None
    motifs.sort(key=lambda m: (m.c, m.r))

    inp = _build_input((H, W), guide_rows, guide_cols, motifs)
    expected = _render_with_motifs(inp, guide_rows, guide_cols, motifs)
    if not _semantic_checks(inp, expected, guide_rows, guide_cols, motifs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray stamp input")
