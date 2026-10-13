from __future__ import annotations

import numpy as np


BG = 0
FRAME = 1
MARK = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": sub,
                    "area": int(len(cells)),
                }
            )
    return comps


def _infer_thickness(blue_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    limit = min(h, w) // 2
    thickness = 0
    for k in range(1, limit + 1):
        if (
            blue_mask[r0 : r0 + k, c0 : c0 + k].all()
            and blue_mask[r0 : r0 + k, c1 - k + 1 : c1 + 1].all()
            and blue_mask[r1 - k + 1 : r1 + 1, c0 : c0 + k].all()
            and blue_mask[r1 - k + 1 : r1 + 1, c1 - k + 1 : c1 + 1].all()
        ):
            thickness = k
        else:
            break
    assert thickness > 0, "failed to infer frame thickness"
    return thickness


def _full_frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], thickness: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    full = np.zeros(shape, dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if min(r - r0, r1 - r, c - c0, c1 - c) < thickness:
                full[r, c] = True
    return full


def _gap_outer_side(gap_bbox: tuple[int, int, int, int], board_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = gap_bbox
    br0, bc0, br1, bc1 = board_bbox
    touches: list[str] = []
    if r0 == br0:
        touches.append("top")
    if r1 == br1:
        touches.append("bottom")
    if c0 == bc0:
        touches.append("left")
    if c1 == bc1:
        touches.append("right")
    assert len(touches) == 1, "each gap must touch exactly one outer side"
    return touches[0]


def _extract_board_and_gaps(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict[str, object]]]:
    blue_mask = grid == FRAME
    blue_comps = _components(blue_mask)
    assert blue_comps, "expected a blue frame"
    board = max(blue_comps, key=lambda comp: int(comp["area"]))
    board_bbox = board["bbox"]
    thickness = _infer_thickness(blue_mask, board_bbox)
    full_frame = _full_frame_mask(grid.shape, board_bbox, thickness)
    gaps_mask = full_frame & ~blue_mask
    gaps: list[dict[str, object]] = []
    for comp in _components(gaps_mask):
        outer_side = _gap_outer_side(comp["bbox"], board_bbox)
        gaps.append(
            {
                "bbox": comp["bbox"],
                "mask": comp["mask"],
                "outer_side": outer_side,
                "desired_side": OPPOSITE[outer_side],
            }
        )
    gaps.sort(key=lambda gap: gap["bbox"])
    return board_bbox, gaps


def _extract_pieces(grid: np.ndarray) -> list[dict[str, object]]:
    piece_mask = (grid != BG) & (grid != FRAME)
    pieces: list[dict[str, object]] = []
    for comp in _components(piece_mask):
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        local_mask = comp["mask"]
        patch[~local_mask] = BG
        marker_coords = np.argwhere((patch == MARK) & local_mask)
        colors = {int(v) for v in patch[local_mask]} - {MARK}
        assert len(marker_coords) == 1, "each piece must contain exactly one marker"
        assert len(colors) == 1, "each piece must use one base color"
        pieces.append(
            {
                "bbox": comp["bbox"],
                "mask": local_mask,
                "patch": patch,
                "marker": tuple(map(int, marker_coords[0])),
                "color": colors.pop(),
            }
        )
    return pieces


def _rotate_piece(piece: dict[str, object], turns: int) -> dict[str, object]:
    patch = np.rot90(piece["patch"], turns)
    mask = np.rot90(piece["mask"], turns)
    marker_mask = np.zeros_like(piece["mask"], dtype=bool)
    marker_mask[piece["marker"]] = True
    rotated_marker = np.argwhere(np.rot90(marker_mask, turns))
    assert len(rotated_marker) == 1
    return {
        "patch": patch,
        "mask": mask,
        "marker": tuple(map(int, rotated_marker[0])),
    }


def _marker_side(mask: np.ndarray, marker: tuple[int, int]) -> str:
    coords = np.argwhere(mask)
    center_r, center_c = coords.mean(axis=0)
    mr, mc = marker
    scores = {
        "top": center_r - mr,
        "bottom": mr - center_r,
        "left": center_c - mc,
        "right": mc - center_c,
    }
    best = max(scores.values())
    candidates = [side for side, value in scores.items() if abs(value - best) < 1e-9]
    if len(candidates) == 1:
        return candidates[0]

    edge_dists = {
        "top": mr,
        "bottom": mask.shape[0] - 1 - mr,
        "left": mc,
        "right": mask.shape[1] - 1 - mc,
    }
    return min(candidates, key=lambda side: (edge_dists[side], side))


def _solve_assignment(pieces: list[dict[str, object]], gaps: list[dict[str, object]]) -> dict[int, tuple[int, dict[str, object]]]:
    candidates: list[list[tuple[int, dict[str, object]]]] = [[] for _ in gaps]
    for gap_index, gap in enumerate(gaps):
        desired_side = gap["desired_side"]
        gap_mask = gap["mask"]
        for piece_index, piece in enumerate(pieces):
            for turns in range(4):
                rotated = _rotate_piece(piece, turns)
                if rotated["mask"].shape != gap_mask.shape:
                    continue
                if not np.array_equal(rotated["mask"], gap_mask):
                    continue
                if _marker_side(rotated["mask"], rotated["marker"]) != desired_side:
                    continue
                candidates[gap_index].append((piece_index, rotated))
        assert candidates[gap_index], "every gap must have at least one candidate"

    order = sorted(range(len(gaps)), key=lambda index: (len(candidates[index]), gaps[index]["bbox"]))
    solution: list[dict[int, tuple[int, dict[str, object]]]] = []
    used: set[int] = set()
    current: dict[int, tuple[int, dict[str, object]]] = {}

    def dfs(depth: int) -> None:
        if len(solution) > 1:
            return
        if depth == len(order):
            solution.append(current.copy())
            return
        gap_index = order[depth]
        for piece_index, rotated in candidates[gap_index]:
            if piece_index in used:
                continue
            used.add(piece_index)
            current[gap_index] = (piece_index, rotated)
            dfs(depth + 1)
            del current[gap_index]
            used.remove(piece_index)

    dfs(0)
    assert len(solution) == 1, "piece-to-gap matching must be unique"
    return solution[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the missing frame gaps by rotating the scattered pieces so their marker side faces the board center.
    """
    g = np.asarray(grid, dtype=int)
    _board_bbox, gaps = _extract_board_and_gaps(g)
    pieces = _extract_pieces(g)
    assert 2 <= len(gaps) <= 4
    assert len(pieces) == len(gaps), "piece count must equal gap count"

    assignment = _solve_assignment(pieces, gaps)
    out = g.copy()
    out[(g != BG) & (g != FRAME)] = BG
    for gap_index, gap in enumerate(gaps):
        _piece_index, rotated = assignment[gap_index]
        r0, c0, r1, c1 = gap["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[rotated["mask"]] = rotated["patch"][rotated["mask"]]
    return out


GEN_ATTEMPTS = 400
GEN_COLORS = [2, 3, 4, 5, 6, 7, 8]
GEN_LENGTHS = [3, 4, 5, 6, 7, 8]
GEN_THICKNESS = 3


def _center_index(span: int, rng: np.random.Generator) -> int:
    mid = span // 2
    if span % 2 == 1:
        return mid
    return int(rng.choice([mid - 1, mid]))


def _marker_coord(shape: tuple[int, int], desired_side: str, rng: np.random.Generator) -> tuple[int, int]:
    h, w = shape
    if desired_side == "top":
        return 0, _center_index(w, rng)
    if desired_side == "bottom":
        return h - 1, _center_index(w, rng)
    if desired_side == "left":
        return _center_index(h, rng), 0
    if desired_side == "right":
        return _center_index(h, rng), w - 1
    raise AssertionError("unknown side")


def _canonical_piece(side: str, length: int, color: int, rng: np.random.Generator) -> np.ndarray:
    if side in {"top", "bottom"}:
        shape = (GEN_THICKNESS, length)
    else:
        shape = (length, GEN_THICKNESS)
    patch = np.full(shape, color, dtype=int)
    mr, mc = _marker_coord(shape, OPPOSITE[side], rng)
    patch[mr, mc] = MARK
    return patch


def _draw_full_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    local = _full_frame_mask(grid.shape, bbox, GEN_THICKNESS)
    grid[local] = FRAME


def _gap_abs_bbox(board_bbox: tuple[int, int, int, int], side: str, start: int, length: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = board_bbox
    if side == "top":
        return r0, c0 + start, r0 + GEN_THICKNESS - 1, c0 + start + length - 1
    if side == "bottom":
        return r1 - GEN_THICKNESS + 1, c0 + start, r1, c0 + start + length - 1
    if side == "left":
        return r0 + start, c0, r0 + start + length - 1, c0 + GEN_THICKNESS - 1
    if side == "right":
        return r0 + start, c1 - GEN_THICKNESS + 1, r0 + start + length - 1, c1
    raise AssertionError("unknown side")


def _can_place(forbidden: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    er0 = max(0, r - 1)
    ec0 = max(0, c - 1)
    er1 = min(forbidden.shape[0], r + h + 1)
    ec1 = min(forbidden.shape[1], c + w + 1)
    return not forbidden[er0:er1, ec0:ec1].any()


def _reserve(forbidden: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    er0 = max(0, r - 1)
    ec0 = max(0, c - 1)
    er1 = min(forbidden.shape[0], r + h + 1)
    ec1 = min(forbidden.shape[1], c + w + 1)
    forbidden[er0:er1, ec0:ec1] = True


def _expected_output(inp: np.ndarray, gap_specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    out[(out != BG) & (out != FRAME)] = BG
    for spec in gap_specs:
        r0, c0, r1, c1 = spec["abs_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = spec["canonical_patch"]
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    num_gaps = int(rng.integers(2, 5))
    sides = list(rng.choice(SIDES, size=num_gaps, replace=False))
    lengths = list(map(int, rng.choice(GEN_LENGTHS, size=num_gaps, replace=False)))

    horizontal_need = max([length for side, length in zip(sides, lengths) if side in {"top", "bottom"}] + [8])
    vertical_need = max([length for side, length in zip(sides, lengths) if side in {"left", "right"}] + [8])

    inner_h_low = max(vertical_need, 8)
    inner_w_low = max(horizontal_need, 8)
    inner_h_high = 13
    inner_w_high = 13
    if inner_h_low > inner_h_high or inner_w_low > inner_w_high:
        return None

    inner_h = int(rng.integers(inner_h_low, inner_h_high + 1))
    inner_w = int(rng.integers(inner_w_low, inner_w_high + 1))
    outer_h = inner_h + 2 * GEN_THICKNESS
    outer_w = inner_w + 2 * GEN_THICKNESS

    min_h = max(outer_h + 6, 18)
    min_w = max(outer_w + 6, 18)
    max_h = min(28, outer_h + 10)
    max_w = min(28, outer_w + 10)
    if min_h > max_h or min_w > max_w:
        return None

    grid_h = int(rng.integers(min_h, max_h + 1))
    grid_w = int(rng.integers(min_w, max_w + 1))
    board_r0 = int(rng.integers(2, grid_h - outer_h - 1))
    board_c0 = int(rng.integers(2, grid_w - outer_w - 1))
    board_bbox = (board_r0, board_c0, board_r0 + outer_h - 1, board_c0 + outer_w - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_full_frame(grid, board_bbox)

    colors = list(map(int, rng.choice(GEN_COLORS, size=num_gaps, replace=False)))
    gap_specs: list[dict[str, object]] = []
    for side, length, color in zip(sides, lengths, colors):
        if side in {"top", "bottom"}:
            inner_span = outer_w - 2 * GEN_THICKNESS
        else:
            inner_span = outer_h - 2 * GEN_THICKNESS
        max_start = inner_span - length
        if max_start < 0:
            return None
        start = GEN_THICKNESS + int(rng.integers(0, max_start + 1))
        abs_bbox = _gap_abs_bbox(board_bbox, side, start, length)
        r0, c0, r1, c1 = abs_bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG
        gap_specs.append(
            {
                "side": side,
                "length": length,
                "color": color,
                "abs_bbox": abs_bbox,
                "canonical_patch": _canonical_piece(side, length, color, rng),
            }
        )

    forbidden = np.zeros_like(grid, dtype=bool)
    br0, bc0, br1, bc1 = board_bbox
    _reserve(forbidden, br0, bc0, outer_h, outer_w)

    rng.shuffle(gap_specs)
    for spec in gap_specs:
        patch = np.rot90(spec["canonical_patch"], int(rng.integers(0, 4)))
        ph, pw = patch.shape
        placed = False
        for _ in range(300):
            rr = int(rng.integers(0, grid_h - ph + 1))
            cc = int(rng.integers(0, grid_w - pw + 1))
            if not _can_place(forbidden, rr, cc, ph, pw):
                continue
            grid[rr : rr + ph, cc : cc + pw] = patch
            _reserve(forbidden, rr, cc, ph, pw)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, gap_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    _board_bbox, gaps = _extract_board_and_gaps(grid)
    pieces = _extract_pieces(grid)
    if len(gaps) != num_gaps or len(pieces) != num_gaps:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
