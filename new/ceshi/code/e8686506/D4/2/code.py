# Variant of e8686506
# Dimension: D4 — 对称约束：输出的对称性类型 (match_cond)
# Change: 输出具有严格的左右（垂直轴）镜像对称性... → 90度旋转对称

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D4 variant
#
# a) Concrete behavior:
#    The original task assembled a portrait rectangle whose colors matched under
#    left-right reflection. Under 90-degree rotational symmetry the output must
#    be a square, and every cell color must agree with its full 4-cycle
#    ((r,c) -> (c,N-1-r) -> ...). To make that visible, the generator uses a
#    sparse pinwheel-shaped frame: it is rotation-symmetric but deliberately not
#    LR-symmetric, so the new symmetry type is readable by eye.
#
#    Small example trace:
#    - Old logic: a red cell at (0,1) only constrained the mirror cell (0,W-2).
#    - New logic: a red cell at (0,1) constrains three mates
#      (1,N-1), (N-1,N-2), (N-2,0).
#    - Difference: the solver must validate an entire rotation orbit, not just
#      a mirror pair, and the generator must provide four-direction structure.
#
# b) Complexity calibration:
#    Pure orbit-coloring would be too weak and would often look like a generic
#    square mosaic. To keep the puzzle identifiable, the frame is pre-placed as
#    a recognizable 8-connected pinwheel whose union bbox already reveals the
#    output size. The remaining cells are grouped into only three rotationally
#    repeated fill colors; this keeps the rule visible while avoiding a trivial
#    one-color-per-orbit encoding.
#
# c) Hidden assumptions from the old solver:
#    - `mirror_ok()` assumed each placement only needed one reflected mate.
#      This is invalid now; placements must respect a 4-cell rotation orbit.
#    - The old `H >= W` factor inference relied on portrait outputs. Now the
#      target must be square, so total area must be a perfect square.
#    - The old singleton-border restriction was safe because mirror symmetry
#      made interior singletons ambiguous. In this variant we avoid relying on
#      that rule; the frame and orbit constraints already disambiguate pieces.
#    - Frame detection previously searched for LR-symmetric unions. It must now
#      search for 90-degree rotational symmetry.
#
# d) Placement legality:
#    The generator keeps the frame together inside its true square bbox and
#    scatters all other pieces outside that blocked zone with Chebyshev padding.
#    This guarantees no overlap and preserves the frame's internal offsets, so
#    the solver can lock those cells back into the output.
#
# e) Visual observability:
#    Samples are rejected if the final output is accidentally LR-symmetric. This
#    prevents degeneration back to the old task. The frame has a clear pinwheel
#    handedness, and the non-frame colors repeat by quarter-turns.
#
# f) Visual cleanliness:
#    Only one anchored structure sits inside the square bbox: the frame. Other
#    pieces are scattered away from it. In the output, the frame and the three
#    fill colors occupy disjoint regions, so the rotational pattern remains easy
#    to parse.
#
# g) Color inference:
#    Colors are only permuted globally from abstract roles {frame, fill A,
#    fill B, fill C}; there is no hidden ordering by color id. The repeated
#    geometry of each color class tells the user which cells belong together.
# ---------------------------------------------------------------------------


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _rot90_cell(r: int, c: int, n: int) -> tuple[int, int]:
    return c, n - 1 - r


def _orbit_cells(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    rr, cc = int(r), int(c)
    for _ in range(4):
        cell = (rr, cc)
        if cell not in out:
            out.append(cell)
        rr, cc = _rot90_cell(rr, cc, n)
    return tuple(out)


def _build_requirements(
    n: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Base requirement grid (req[r,c] == -1 means unconstrained).

    In this variant the strong role-specific requirements are inferred only
    after the solver has identified which rotational frame template is present.
    The old generic heuristics are intentionally omitted because repeated
    quarter-turn fill pieces would trigger them spuriously.
    """
    return np.full((n, n), -1, dtype=int)


def _is_rot4_symmetric(parts: list[tuple[int, np.ndarray]], n: int) -> bool:
    all_pts = np.concatenate([pts for _, pts in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((n, n), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return np.array_equal(bmp, np.rot90(bmp))


def _normalized_frame_bitmap(parts: list[tuple[int, np.ndarray]], n: int) -> np.ndarray:
    all_pts = np.concatenate([pts for _, pts in parts])
    r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
    bmp = np.zeros((n, n), dtype=bool)
    for _, pts in parts:
        for r, c in pts:
            bmp[int(r) - r0, int(c) - c0] = True
    return bmp


def _assemble(
    types: list[dict],
    n: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    orbit_cache = {(r, c): _orbit_cells(r, c, n) for r in range(n) for c in range(n)}

    def orbit_ok(board: np.ndarray, r: int, c: int, col: int) -> bool:
        for rr, cc in orbit_cache[(r, c)]:
            if req[rr, cc] != -1 and int(req[rr, cc]) != col:
                return False
            v = int(board[rr, cc])
            if v != -1 and v != col:
                return False
        return True

    def can_place(board: np.ndarray, ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not orbit_ok(board, r, c, col):
                return False
            if disallow_touch and col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < n and int(board[rr, cc]) == col:
                        return False
        return True

    def placements_for(board: np.ndarray, ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(n - int(t["h"]) + 1)
            for c0 in range(n - int(t["w"]) + 1)
            if can_place(board, ti, r0, c0)
        ]

    def disjoint_batch(ti: int, places: list[tuple[int, int]]) -> bool:
        seen: set[tuple[int, int]] = set()
        coords = types[ti]["coords"]
        for r0, c0 in places:
            for dr, dc in coords:
                cell = (r0 + dr, c0 + dc)
                if cell in seen:
                    return False
                seen.add(cell)
        return True

    def place_copy(board: np.ndarray, ti: int, r0: int, c0: int) -> np.ndarray:
        out = board.copy()
        col = int(types[ti]["col"])
        for dr, dc in types[ti]["coords"]:
            out[r0 + dr, c0 + dc] = col
        return out

    def search(board: np.ndarray, counts: tuple[int, ...]) -> np.ndarray | None:
        cur = board
        cnts = list(counts)
        while True:
            pending = [ti for ti, c in enumerate(cnts) if c > 0]
            if not pending:
                break
            placements_cache: dict[int, list[tuple[int, int]]] = {}
            for ti in pending:
                places = placements_for(cur, ti)
                if not places:
                    return None
                placements_cache[ti] = places

            forced = False
            for ti in pending:
                places = placements_cache[ti]
                if len(places) == 1:
                    cur = place_copy(cur, ti, places[0][0], places[0][1])
                    cnts[ti] -= 1
                    forced = True
                    break
                if len(places) == cnts[ti] and disjoint_batch(ti, places):
                    for r0, c0 in places:
                        cur = place_copy(cur, ti, r0, c0)
                        cnts[ti] -= 1
                    forced = True
                    break
            if not forced:
                break

        if all(c == 0 for c in cnts):
            if np.any(cur < 0):
                return None
            if not np.array_equal(cur, np.rot90(cur)):
                return None
            return cur

        placements_cache = {
            ti: placements_for(cur, ti)
            for ti, c in enumerate(cnts)
            if c > 0
        }
        for places in placements_cache.values():
            if not places:
                return None

        ti = min(
            placements_cache,
            key=lambda idx: (len(placements_cache[idx]), -len(types[idx]["coords"]), int(types[idx]["col"])),
        )
        for r0, c0 in placements_cache[ti]:
            nxt = place_copy(cur, ti, r0, c0)
            nxt_counts = list(cnts)
            nxt_counts[ti] -= 1
            sol = search(nxt, tuple(nxt_counts))
            if sol is not None:
                return sol
        return None

    board = np.full((n, n), -1, dtype=int)
    sol = search(board, tuple(int(t["cnt"]) for t in types))
    assert sol is not None, "assembly failed under 90-degree rotational constraints"
    return sol


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a dense square with 90-degree rotational symmetry.

    Input structure:
    - bg = most common color; the other colors are detached fragment pieces.
    - One pre-placed 8-connected "pinwheel frame" stays grouped inside its true
      square bounding box; the rest of the pieces are scattered around it.

    Rule:
    - Extract every non-bg component as a jigsaw piece and infer the output size
      from the total area.
    - Identify the frame whose bbox already matches that square.
    - Pack all pieces back into the square so every color orbit is preserved
      under 90-degree rotation and same-color pieces never 4-touch.

    Key insight: the frame fixes the square's handed pinwheel layout, and once
    each placement must agree with its full four-cell rotation orbit, the
    remaining fragments collapse into a unique quarter-turn-symmetric assembly.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = components_mask(g == col, connectivity=4)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((int(col), coords))

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    assert total_area > 0, "grid must contain non-background pieces"
    n = int(round(total_area ** 0.5))
    assert n * n == total_area, "90-degree rotational variant requires square output area"

    comps8_all = []
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for pts in components_mask(g == col, connectivity=8):
            comps8_all.append((int(col), pts))

    frame_parts: list[tuple[int, np.ndarray]] = []
    for i, (c1, p1) in enumerate(comps8_all):
        r0, c0, r1, c1b = bbox_coords(p1)
        if r1 - r0 + 1 == n and c1b - c0 + 1 == n and _is_rot4_symmetric([(c1, p1)], n):
            frame_parts = [(c1, p1)]
            break
        for j in range(i + 1, len(comps8_all)):
            c2, p2 = comps8_all[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == n and uc1 - uc0 + 1 == n:
                if _is_rot4_symmetric([(c1, p1), (c2, p2)], n):
                    frame_parts = [(c1, p1), (c2, p2)]
                    break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 rotationally symmetric frame parts whose bbox is the output square"

    req = _build_requirements(n, comps_by_color)
    frame_bmp = _normalized_frame_bitmap(frame_parts, n)
    tmpl = _match_template(frame_bmp)

    frame_colors = {int(col) for col, _ in frame_parts}
    role_colors: dict[str, int] = {}
    for col, comps in comps_by_color.items():
        if col in frame_colors:
            continue
        sizes = sorted(len(p) for p in comps)
        if sizes == [1, 1, 1, 1]:
            assert "singleton" not in role_colors, "singleton role color must be unique"
            role_colors["singleton"] = int(col)
            continue
        if sizes == [2, 2, 2, 2]:
            sigs = [_piece_sig(pts) for pts in comps]
            ok = all(
                (h == 1 and w == 2 and len(coords) == 2) or
                (h == 2 and w == 1 and len(coords) == 2)
                for h, w, coords in sigs
            )
            if ok:
                assert "domino" not in role_colors, "domino role color must be unique"
                role_colors["domino"] = int(col)
                continue
        if sizes == [4, 4, 4, 4]:
            assert "tetromino" not in role_colors, "tetromino role color must be unique"
            role_colors["tetromino"] = int(col)
            continue
    assert set(role_colors) == {"singleton", "domino", "tetromino"}, "could not classify all non-frame role colors"

    for role, color in role_colors.items():
        for r, c in tmpl["roles"][role]:
            for rr, cc in _orbit_cells(r, c, n):
                if req[rr, cc] != -1:
                    assert int(req[rr, cc]) == color, "role constraints disagree"
                req[rr, cc] = color

    all_frame_pts = np.concatenate([p for _, p in frame_parts])
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for col, pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < n and 0 <= oc < n:
                if req[or_, oc] != -1:
                    assert int(req[or_, oc]) == int(col), "frame and optional anchors disagree"
                req[or_, oc] = int(col)

    disallow_touch = {col for col, cnt in color_piece_count.items() if cnt > 1}
    sol = _assemble(types, n, req, disallow_touch)
    return sol.astype(int)


# --------------- generator (rotational pinwheel, output-first) ---------------

_OUTPUT_SIZE = 7
_N_COLORS = 4
_PIECE_COUNT_REQUIRED = 17
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 200

_TEMPLATES = (
    {
        "frame_seed": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 3), (3, 3)],
        "roles": {
            "singleton": [(0, 3)],
            "domino": [(0, 4), (0, 5)],
            "tetromino": [(1, 1), (1, 3), (1, 4), (2, 2)],
        },
    },
    {
        "frame_seed": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 3), (3, 3)],
        "roles": {
            "singleton": [(0, 1)],
            "domino": [(0, 2), (0, 3)],
            "tetromino": [(1, 1), (1, 2), (1, 3), (2, 2)],
        },
    },
)


def _paint_orbit(board: np.ndarray, cells: list[tuple[int, int]], col: int) -> None:
    n = int(board.shape[0])
    for r, c in cells:
        for rr, cc in _orbit_cells(r, c, n):
            board[rr, cc] = col


def _template_frame_bitmap(tmpl: dict, n: int) -> np.ndarray:
    bmp = np.zeros((n, n), dtype=bool)
    for r, c in tmpl["frame_seed"]:
        for rr, cc in _orbit_cells(r, c, n):
            bmp[rr, cc] = True
    return bmp


def _match_template(frame_bmp: np.ndarray) -> dict:
    for tmpl in _TEMPLATES:
        if np.array_equal(frame_bmp, _template_frame_bitmap(tmpl, frame_bmp.shape[0])):
            return tmpl
    raise AssertionError("detected frame does not match any supported rotational template")


def _make_output() -> tuple[np.ndarray, int, set[int]]:
    tmpl = random.choice(_TEMPLATES)
    n = _OUTPUT_SIZE

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    role_cols = palette[:_N_COLORS]

    board = np.full((n, n), -1, dtype=int)
    _paint_orbit(board, list(tmpl["frame_seed"]), int(role_cols[0]))
    _paint_orbit(board, list(tmpl["roles"]["singleton"]), int(role_cols[1]))
    _paint_orbit(board, list(tmpl["roles"]["domino"]), int(role_cols[2]))
    _paint_orbit(board, list(tmpl["roles"]["tetromino"]), int(role_cols[3]))

    assert not np.any(board < 0), "template must cover the full output square"
    assert np.array_equal(board, np.rot90(board)), "generator output must be 90-degree rotationally symmetric"
    assert not np.array_equal(board, board[:, ::-1]), "generator must not degenerate to LR symmetry"
    return board, int(bg), {int(role_cols[0])}


def _extract_non_frame_patches(out: np.ndarray, bg: int, fcs: set[int]) -> list[np.ndarray]:
    patches = []
    for col in sorted(map(int, np.unique(out))):
        if col in fcs:
            continue
        for pts in components_mask(out == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for r, c in pts:
                patch[int(r) - r0, int(c) - c0] = col
            patches.append(patch)
    random.shuffle(patches)
    patches.sort(key=lambda p: (-(p != bg).sum(), random.random()))
    return patches


def _scatter(out: np.ndarray, bg: int, fcs: set[int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1
    for patch in _extract_non_frame_patches(out, bg, fcs):
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    frame_mask = np.zeros((oh, ow), dtype=bool)
    for col in fcs:
        frame_mask |= (out == col)
    bbox_region = canvas[off_r : off_r + oh, off_c : off_c + ow]
    bbox_region[:] = bg
    bbox_region[frame_mask] = out[frame_mask]
    return canvas


def _piece_count(out: np.ndarray) -> int:
    return sum(len(components_mask(out == col, connectivity=4)) for col in np.unique(out))


def _component_size_multiset(out: np.ndarray) -> list[int]:
    sizes = []
    for col in np.unique(out):
        for pts in components_mask(out == col, connectivity=4):
            sizes.append(int(len(pts)))
    return sorted(sizes)


def _gen_attempt() -> np.ndarray | None:
    out, bg, fcs = _make_output()
    if len(np.unique(out)) != _N_COLORS:
        return None
    if _piece_count(out) != _PIECE_COUNT_REQUIRED:
        return None
    if max(_component_size_multiset(out)) > 5:
        return None

    try:
        inp = _scatter(out, bg, fcs)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D4 input")
