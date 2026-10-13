import random

import numpy as np


BG = 0
MOLD_COLOR = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _rot90(norm: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    return _normalize([(c, h - 1 - r) for r, c in norm])


def _rotations4(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    norm, h, w = _normalize(coords)
    out: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    cur = (norm, h, w)
    for _ in range(4):
        out.append(cur)
        cur = _rot90(*cur)
    return out


def _unique_rotations(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    for norm, h, w in _rotations4(coords):
        if norm not in seen:
            seen.add(norm)
            out.append((norm, h, w))
    return out


def _canonical_shape(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(norm for norm, _, _ in _unique_rotations(coords))


def _translate(norm: tuple[tuple[int, int], ...], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in norm]


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _find_mold(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {BG}):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all((sub == BG) | (sub == color)):
                continue
            if np.any(sub[0, :] != color) or np.any(sub[-1, :] != color):
                continue
            if np.any(sub[:, 0] != color) or np.any(sub[:, -1] != color):
                continue
            holes = []
            for local_coords in _components(sub == BG):
                if any(
                    rr in (0, sub.shape[0] - 1) or cc in (0, sub.shape[1] - 1)
                    for rr, cc in local_coords
                ):
                    holes = []
                    break
                norm, _, _ = _normalize(local_coords)
                holes.append(
                    {
                        "coords": [(r0 + rr, c0 + cc) for rr, cc in local_coords],
                        "norm": norm,
                        "area": len(local_coords),
                    }
                )
            if holes:
                candidates.append(
                    {
                        "color": int(color),
                        "bbox": (r0, c0, r1, c1),
                        "bbox_area": sub.shape[0] * sub.shape[1],
                        "area": len(coords),
                        "holes": holes,
                    }
                )
    assert candidates, "expected one rectangle mold with enclosed holes"
    candidates.sort(key=lambda item: (item["bbox_area"], item["area"], len(item["holes"])), reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert best["bbox_area"] > candidates[1]["bbox_area"], "mold must be uniquely largest"
    return best


def _outside_pieces(grid: np.ndarray, mold: dict) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = mold["bbox"]
    pieces: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {BG, mold["color"]}):
        for coords in _components(g == color):
            assert all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in coords), "pieces must be outside the mold"
            norm, _, _ = _normalize(coords)
            pieces.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "norm": norm,
                    "area": len(coords),
                    "rots": _unique_rotations(coords),
                }
            )
    return pieces


def _match_pieces_to_holes(pieces: list[dict], holes: list[dict]) -> dict[int, int]:
    candidates: dict[int, list[int]] = {}
    for pi, piece in enumerate(pieces):
        options = []
        for hi, hole in enumerate(holes):
            if piece["area"] != hole["area"]:
                continue
            if any(rot_norm == hole["norm"] for rot_norm, _, _ in piece["rots"]):
                options.append(hi)
        assert options, "every piece must match some hole"
        candidates[pi] = options

    order = sorted(candidates, key=lambda idx: (len(candidates[idx]), pieces[idx]["area"]))
    used: set[int] = set()
    current: dict[int, int] = {}
    solutions: list[dict[int, int]] = []

    def search(i: int) -> None:
        if len(solutions) > 1:
            return
        if i == len(order):
            solutions.append(current.copy())
            return
        pi = order[i]
        for hi in candidates[pi]:
            if hi in used:
                continue
            used.add(hi)
            current[pi] = hi
            search(i + 1)
            current.pop(pi, None)
            used.remove(hi)

    search(0)
    assert len(solutions) == 1, "piece-hole correspondence must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the uniquely matching black cavity inside the grey mold.
    Fragments may need to be rotated by 0/90/180/270 degrees before matching.
    """
    g = np.asarray(grid, dtype=int)
    mold = _find_mold(g)
    holes = mold["holes"]
    pieces = _outside_pieces(g, mold)
    assert len(pieces) == len(holes) and pieces, "need a one-to-one set of pieces and holes"

    assignment = _match_pieces_to_holes(pieces, holes)

    out = g.copy()
    for piece in pieces:
        _paint(out, piece["coords"], BG)
    for pi, hi in assignment.items():
        _paint(out, holes[hi]["coords"], pieces[pi]["color"])
    return out


# --------------- generator ---------------

PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PIECE_COUNT_RANGE = (2, 4)
PIECE_AREA_RANGE = (4, 7)
SHAPE_SIDE_MAX = 4
INNER_SIDE_RANGE = (9, 13)
OUTER_PAD_RANGE = (6, 10)
GENERATION_RETRIES = 400


def _random_shape(area: int) -> tuple[tuple[int, int], ...] | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        seen = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if cand in cells or cand in seen:
                    continue
                seen.add(cand)
                norm, h, w = _normalize(tuple(cells | {cand}))
                if h <= SHAPE_SIDE_MAX and w <= SHAPE_SIDE_MAX:
                    frontier.append(cand)
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    norm, h, w = _normalize(tuple(cells))
    if h < 2 or w < 2 or len(norm) == h * w:
        return None
    if len(_unique_rotations(norm)) != 4:
        return None
    return norm


def _sample_shapes(count: int) -> list[tuple[tuple[int, int], ...]] | None:
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(100):
        while len(shapes) < count:
            area = random.randint(*PIECE_AREA_RANGE)
            shape = _random_shape(area)
            if shape is None:
                continue
            sig = _canonical_shape(shape)
            if sig in seen:
                continue
            seen.add(sig)
            shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _placement_options(box_h: int, box_w: int, norm: tuple[tuple[int, int], ...], blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    _, h, w = _normalize(norm)
    options: list[list[tuple[int, int]]] = []
    for r0 in range(box_h - h + 1):
        for c0 in range(box_w - w + 1):
            coords = _translate(norm, r0, c0)
            ok = True
            for r, c in coords:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < box_h and 0 <= cc < box_w) or blocked[rr, cc]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                options.append(coords)
    return options


def _block(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked[r + dr, c + dc] = True


def _place_oriented_shapes(
    box_h: int, box_w: int, oriented_shapes: list[tuple[tuple[int, int], ...]]
) -> list[list[tuple[int, int]]] | None:
    blocked = np.zeros((box_h, box_w), dtype=bool)
    placements: list[list[tuple[int, int]] | None] = [None] * len(oriented_shapes)
    order = sorted(range(len(oriented_shapes)), key=lambda idx: -len(oriented_shapes[idx]))
    for idx in order:
        options = _placement_options(box_h, box_w, oriented_shapes[idx], blocked)
        if not options:
            return None
        coords = random.choice(options)
        placements[idx] = coords
        _block(blocked, coords)
    return [coords for coords in placements if coords is not None]


def _expected_output(
    grid: np.ndarray,
    hole_coords: list[list[tuple[int, int]]],
    piece_coords: list[list[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for coords in piece_coords:
        _paint(out, coords, BG)
    for coords, color in zip(hole_coords, colors):
        _paint(out, coords, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.randint(*PIECE_COUNT_RANGE)
    shapes = _sample_shapes(piece_count)
    if shapes is None:
        return None

    inner_h = random.randint(*INNER_SIDE_RANGE)
    inner_w = random.randint(*INNER_SIDE_RANGE)
    hole_oriented = [random.choice(_rotations4(shape))[0] for shape in shapes]
    hole_local = _place_oriented_shapes(inner_h, inner_w, hole_oriented)
    if hole_local is None:
        return None

    rect_h, rect_w = inner_h + 2, inner_w + 2
    grid_h = rect_h + random.randint(*OUTER_PAD_RANGE)
    grid_w = rect_w + random.randint(*OUTER_PAD_RANGE)
    mold_r0 = random.randint(2, grid_h - rect_h - 2)
    mold_c0 = random.randint(2, grid_w - rect_w - 2)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[mold_r0 : mold_r0 + rect_h, mold_c0 : mold_c0 + rect_w] = MOLD_COLOR

    hole_abs = []
    for coords in hole_local:
        abs_coords = [(mold_r0 + 1 + r, mold_c0 + 1 + c) for r, c in coords]
        hole_abs.append(abs_coords)
        _paint(grid, abs_coords, BG)

    piece_oriented = [random.choice(_rotations4(shape))[0] for shape in shapes]
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    blocked[
        max(0, mold_r0 - 1) : min(grid_h, mold_r0 + rect_h + 1),
        max(0, mold_c0 - 1) : min(grid_w, mold_c0 + rect_w + 1),
    ] = True

    piece_abs: list[list[tuple[int, int]]] = []
    order = list(range(piece_count))
    order.sort(key=lambda idx: -len(piece_oriented[idx]))
    tmp_piece_abs: list[list[tuple[int, int]] | None] = [None] * piece_count
    for idx in order:
        options = _placement_options(grid_h, grid_w, piece_oriented[idx], blocked)
        if not options:
            return None
        coords = random.choice(options)
        tmp_piece_abs[idx] = coords
        _block(blocked, coords)
    piece_abs = [coords for coords in tmp_piece_abs if coords is not None]

    colors = random.sample(list(PIECE_COLORS), piece_count)
    for coords, color in zip(piece_abs, colors):
        _paint(grid, coords, color)

    expected = _expected_output(grid, hole_abs, piece_abs, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
