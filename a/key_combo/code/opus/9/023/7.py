from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
HINGE = 7
MAIN_COLORS = [1, 2, 3, 4, 6, 8, 9]

NORTH = (-1, 0)
EAST = (0, 1)
SOUTH = (1, 0)
WEST = (0, -1)
DIRS = (NORTH, EAST, SOUTH, WEST)

TEMPLATE_LIBRARY = [
    {
        "shape": (6, 3),
        "pieces": [
            {"fixed_len": 3, "tail_len": 4, "fixed_dir": EAST, "fold_dir": SOUTH},
            {"fixed_len": 2, "tail_len": 4, "fixed_dir": WEST, "fold_dir": SOUTH},
            {"fixed_len": 2, "tail_len": 3, "fixed_dir": EAST, "fold_dir": SOUTH},
        ],
    },
    {
        "shape": (3, 6),
        "pieces": [
            {"fixed_len": 3, "tail_len": 4, "fixed_dir": SOUTH, "fold_dir": EAST},
            {"fixed_len": 2, "tail_len": 4, "fixed_dir": NORTH, "fold_dir": EAST},
            {"fixed_len": 2, "tail_len": 3, "fixed_dir": SOUTH, "fold_dir": EAST},
        ],
    },
    {
        "shape": (6, 4),
        "pieces": [
            {"fixed_len": 3, "tail_len": 5, "fixed_dir": EAST, "fold_dir": SOUTH},
            {"fixed_len": 3, "tail_len": 5, "fixed_dir": SOUTH, "fold_dir": EAST},
            {"fixed_len": 2, "tail_len": 2, "fixed_dir": NORTH, "fold_dir": EAST},
            {"fixed_len": 2, "tail_len": 2, "fixed_dir": WEST, "fold_dir": SOUTH},
        ],
    },
    {
        "shape": (4, 6),
        "pieces": [
            {"fixed_len": 3, "tail_len": 5, "fixed_dir": SOUTH, "fold_dir": WEST},
            {"fixed_len": 3, "tail_len": 5, "fixed_dir": NORTH, "fold_dir": EAST},
            {"fixed_len": 2, "tail_len": 2, "fixed_dir": SOUTH, "fold_dir": EAST},
            {"fixed_len": 2, "tail_len": 2, "fixed_dir": SOUTH, "fold_dir": WEST},
        ],
    },
    {
        "shape": (6, 5),
        "pieces": [
            {"fixed_len": 3, "tail_len": 5, "fixed_dir": NORTH, "fold_dir": EAST},
            {"fixed_len": 4, "tail_len": 4, "fixed_dir": EAST, "fold_dir": NORTH},
            {"fixed_len": 3, "tail_len": 3, "fixed_dir": EAST, "fold_dir": NORTH},
            {"fixed_len": 2, "tail_len": 2, "fixed_dir": NORTH, "fold_dir": EAST},
            {"fixed_len": 2, "tail_len": 2, "fixed_dir": SOUTH, "fold_dir": EAST},
        ],
    },
]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _scale(direction: tuple[int, int], k: int) -> tuple[int, int]:
    return int(direction[0] * k), int(direction[1] * k)


def _neg(direction: tuple[int, int]) -> tuple[int, int]:
    return -int(direction[0]), -int(direction[1])


def _cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    return dc, -dr


def _ccw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    return -dc, dr


def _transpose_dir(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    return dc, dr


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _normalize_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = np.asarray(sorted(coords), dtype=int)
    pts = pts - pts.min(axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    return tuple((int(r), int(c)) for r, c in pts.tolist())


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    pts = np.asarray(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_from_spec(spec: dict) -> tuple[tuple[int, int], ...]:
    hinge = (0, 0)
    coords = [hinge]
    for step in range(1, int(spec["fixed_len"])):
        coords.append(_add(hinge, _scale(spec["fixed_dir"], step)))
    for step in range(1, int(spec["tail_len"]) + 1):
        coords.append(_add(hinge, _scale(spec["fold_dir"], step)))
    return _normalize_coords(coords)


def _orientations(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    def rot90_clockwise(cells):
        return [(c, -r) for r, c in cells]

    def mirror_horizontal(cells):
        return [(r, -c) for r, c in cells]

    out = []
    current = list(coords)
    for _ in range(4):
        out.append(_normalize_coords(current))
        current = rot90_clockwise(current)

    current = mirror_horizontal(list(coords))
    for _ in range(4):
        out.append(_normalize_coords(current))
        current = rot90_clockwise(current)
    return out


def _distance_to_edge(shape: tuple[int, int], point: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = map(int, shape)
    r, c = map(int, point)
    if direction == NORTH:
        return int(r)
    if direction == SOUTH:
        return int(h - 1 - r)
    if direction == WEST:
        return int(c)
    if direction == EAST:
        return int(w - 1 - c)
    raise AssertionError("invalid direction")


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, int]:
    frames = []
    for comp in _components_from_mask(grid == FRAME):
        if len(comp) < 8:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[0, :] == FRAME):
            continue
        if not np.all(patch[-1, :] == FRAME):
            continue
        if not np.all(patch[:, 0] == FRAME):
            continue
        if not np.all(patch[:, -1] == FRAME):
            continue
        if not np.all(patch[1:-1, 1:-1] == BG):
            continue
        border_mask = np.zeros_like(patch, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if int(border_mask.sum()) != len(comp):
            continue
        frames.append((patch.shape[0] - 2, patch.shape[1] - 2))
    assert len(frames) == 1, "expected exactly one hollow grey frame"
    return int(frames[0][0]), int(frames[0][1])


def _parse_pieces(grid: np.ndarray) -> list[dict]:
    fg = grid != BG
    pieces = []
    for comp in _components_from_mask(fg):
        values = {int(grid[r, c]) for r, c in comp}
        if HINGE not in values:
            continue

        endpoints = [(r, c) for r, c in comp if int(grid[r, c]) == FRAME]
        hinges = [(r, c) for r, c in comp if int(grid[r, c]) == HINGE]
        colors = sorted(values - {FRAME, HINGE})
        assert len(endpoints) == 1 and len(hinges) == 1 and len(colors) == 1, "invalid rod markers"
        endpoint = endpoints[0]
        hinge = hinges[0]
        color = int(colors[0])

        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        horizontal = len(rows) == 1
        vertical = len(cols) == 1
        assert horizontal ^ vertical, "each rod must be straight"

        if horizontal:
            fixed_dir = EAST if endpoint[1] > hinge[1] else WEST
        else:
            fixed_dir = SOUTH if endpoint[0] > hinge[0] else NORTH
        tail_dir = _neg(fixed_dir)

        fixed_len = abs(endpoint[0] - hinge[0]) + abs(endpoint[1] - hinge[1]) + 1
        tail_len = 0
        while True:
            tail_len += 1
            rr, cc = _add(hinge, _scale(tail_dir, tail_len))
            if (rr, cc) not in set(comp):
                tail_len -= 1
                break
            assert int(grid[rr, cc]) == color, "moving tail must use the main color"

        desired_cw = _cw(tail_dir)
        desired_ccw = _ccw(tail_dir)
        cw_space = _distance_to_edge(grid.shape, hinge, desired_cw)
        ccw_space = _distance_to_edge(grid.shape, hinge, desired_ccw)
        fold_dir = desired_cw if cw_space >= ccw_space else desired_ccw

        piece_cells = {hinge}
        for step in range(1, int(fixed_len)):
            piece_cells.add(_add(hinge, _scale(fixed_dir, step)))
        for step in range(1, int(tail_len) + 1):
            piece_cells.add(_add(hinge, _scale(fold_dir, step)))

        pieces.append(
            {
                "anchor": (int(endpoint[0]), int(endpoint[1])),
                "area": int(len(piece_cells)),
                "color": color,
                "coords": _normalize_coords(piece_cells),
            }
        )

    assert 3 <= len(pieces) <= 5, "expected 3-5 rods"
    return pieces


def _pack_pieces(pieces: list[dict], out_shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, out_shape)
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    ordered = sorted(pieces, key=lambda piece: (-piece["area"], piece["anchor"][0], piece["anchor"][1]))
    for piece in ordered:
        placed = False
        for orient in _orientations(piece["coords"]):
            arr = np.asarray(orient, dtype=int)
            ph = int(arr[:, 0].max()) + 1
            pw = int(arr[:, 1].max()) + 1
            if ph > h or pw > w:
                continue
            for r0 in range(h - ph + 1):
                for c0 in range(w - pw + 1):
                    rr = arr[:, 0] + r0
                    cc = arr[:, 1] + c0
                    if bool(occupied[rr, cc].any()):
                        continue
                    occupied[rr, cc] = True
                    out[rr, cc] = int(piece["color"])
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        assert placed, "all pieces must fit the target rectangle"

    assert bool(np.all(occupied)), "packing must fill the rectangle without gaps"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked straight rod into a colored puzzle piece, then greedily pack all pieces into the frame-sized rectangle.

    Input structure:
    - The background is black.
    - There are 3-5 non-touching straight rods, each with one grey endpoint marker, one orange hinge marker, and one main color.
    - A separate hollow grey rectangle marks the target frame; its black inner cavity gives the output height and width.

    Rule:
    - For each rod, keep the segment from the grey endpoint through the orange hinge fixed, and fold the tail on the far side of the hinge by 90 degrees.
    - Compare the two perpendicular hinge-to-edge distances; fold toward the side with more space, and if they are equal choose the clockwise turn.
    - Recolor the grey endpoint and orange hinge to the rod's main color, then sort the resulting pieces by decreasing area, breaking ties by the endpoint marker's row-major position in the input.
    - Pack the pieces into the target rectangle by trying, for each piece in order, the 8 orientations (original, cw90, 180, 270, horizontal mirror, then the same 3 rotations) and scanning placements from top-left to bottom-right; keep the first non-overlapping fit each time.

    Key insight:
    The visible rods are only manufacturing instructions: the hinge position and fold-direction rule determine each L-shaped piece, and only those derived pieces can be packed into the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out_shape = _find_frame(g)
    pieces = _parse_pieces(g)
    return _pack_pieces(pieces, out_shape)
def _choose_template() -> dict:
    template = random.choice(TEMPLATE_LIBRARY)
    return {
        "shape": tuple(template["shape"]),
        "pieces": [dict(piece) for piece in template["pieces"]],
    }


def _buffered(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            out.add((int(nr), int(nc)))
    return out


def _template_pieces(template: dict, colors: list[int]) -> list[dict]:
    pieces = []
    tie_candidates = []
    for idx, (spec, color) in enumerate(zip(template["pieces"], colors)):
        piece = dict(spec)
        piece["color"] = int(color)
        piece["coords"] = _shape_from_spec(spec)
        piece["area"] = int(spec["fixed_len"] + spec["tail_len"])
        tail_dir = _neg(spec["fixed_dir"])
        piece["can_tie"] = spec["fold_dir"] == _cw(tail_dir)
        if piece["can_tie"]:
            tie_candidates.append(idx)
        pieces.append(piece)

    if tie_candidates and random.random() < 0.7:
        forced = random.choice(tie_candidates)
    else:
        forced = None
    for idx, piece in enumerate(pieces):
        piece["branch"] = "tie" if idx == forced else "strict"
    return pieces


def _place_frame(grid: np.ndarray, inner_shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, inner_shape)
    outer_h, outer_w = h + 2, w + 2
    gh, gw = grid.shape
    corner = random.choice(["nw", "ne", "sw", "se"])
    pad_r = random.randint(0, 1)
    pad_c = random.randint(0, 1)
    if corner == "nw":
        r0, c0 = pad_r, pad_c
    elif corner == "ne":
        r0, c0 = pad_r, gw - outer_w - pad_c
    elif corner == "sw":
        r0, c0 = gh - outer_h - pad_r, pad_c
    else:
        r0, c0 = gh - outer_h - pad_r, gw - outer_w - pad_c

    protected = set()
    for rr in range(r0, r0 + outer_h):
        for cc in range(c0, c0 + outer_w):
            protected.add((int(rr), int(cc)))
            if rr in (r0, r0 + outer_h - 1) or cc in (c0, c0 + outer_w - 1):
                grid[rr, cc] = FRAME
    return protected


def _branch_ok(shape: tuple[int, int], hinge: tuple[int, int], piece: dict) -> bool:
    tail_dir = _neg(piece["fixed_dir"])
    other_dir = _ccw(tail_dir) if piece["fold_dir"] == _cw(tail_dir) else _cw(tail_dir)
    desired_space = _distance_to_edge(shape, hinge, piece["fold_dir"])
    other_space = _distance_to_edge(shape, hinge, other_dir)
    if piece["branch"] == "tie":
        return piece["fold_dir"] == _cw(tail_dir) and desired_space == other_space
    return desired_space > other_space


def _rod_cells(hinge: tuple[int, int], piece: dict):
    fixed_len = int(piece["fixed_len"])
    tail_len = int(piece["tail_len"])
    fixed_dir = piece["fixed_dir"]
    tail_dir = _neg(fixed_dir)
    fold_dir = piece["fold_dir"]

    endpoint = _add(hinge, _scale(fixed_dir, fixed_len - 1))
    fixed_mid = {_add(hinge, _scale(fixed_dir, step)) for step in range(1, fixed_len - 1)}
    tail = {_add(hinge, _scale(tail_dir, step)) for step in range(1, tail_len + 1)}
    fold_zone = {_add(hinge, _scale(fold_dir, step)) for step in range(1, tail_len + 1)}
    actual = {endpoint, hinge} | fixed_mid | tail
    envelope = actual | fold_zone
    return endpoint, fixed_mid, tail, actual, envelope


def _candidate_hinges(grid_shape: tuple[int, int], piece: dict) -> list[tuple[int, int]]:
    gh, gw = map(int, grid_shape)
    points = []
    for r in range(gh):
        for c in range(gw):
            endpoint, _fixed_mid, _tail, actual, envelope = _rod_cells((r, c), piece)
            if any(not _in_bounds(grid_shape, rr, cc) for rr, cc in envelope):
                continue
            if not _branch_ok(grid_shape, (r, c), piece):
                continue
            points.append(((int(endpoint[0]), int(endpoint[1])), (int(r), int(c)), actual, envelope))
    points.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))
    return points


def _place_rods(grid: np.ndarray, pieces: list[dict], protected: set[tuple[int, int]]) -> bool:
    blocked: set[tuple[int, int]] = set()
    for r, c in np.argwhere(grid != BG):
        blocked.add((int(r), int(c)))
    blocked = _buffered(blocked)
    blocked |= set(protected)

    last_anchor_by_area: dict[int, tuple[int, int]] = {}
    for piece in pieces:
        placed = False
        for endpoint, hinge, actual, envelope in _candidate_hinges(grid.shape, piece):
            prev = last_anchor_by_area.get(int(piece["area"]))
            if prev is not None and endpoint <= prev:
                continue
            if any(cell in blocked for cell in envelope):
                continue

            _endpoint, fixed_mid, tail, actual_cells, envelope_cells = _rod_cells(hinge, piece)
            for rr, cc in fixed_mid | tail:
                grid[rr, cc] = int(piece["color"])
            grid[hinge] = HINGE
            grid[endpoint] = FRAME
            blocked |= _buffered(envelope_cells)
            last_anchor_by_area[int(piece["area"])] = endpoint
            piece["anchor"] = endpoint
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(shape: tuple[int, int], pieces: list[dict]) -> np.ndarray:
    ordered = []
    for piece in pieces:
        ordered.append(
            {
                "anchor": piece["anchor"],
                "area": piece["area"],
                "color": piece["color"],
                "coords": piece["coords"],
            }
        )
    return _pack_pieces(ordered, shape)


def generate_input() -> np.ndarray:
    for _ in range(400):
        template = _choose_template()
        out_shape = tuple(template["shape"])
        colors = random.sample(MAIN_COLORS, len(template["pieces"]))
        pieces = _template_pieces(template, colors)

        if any(piece["branch"] == "tie" and min(out_shape) == 3 for piece in pieces):
            # A 3-cell side still fits, but odd full-grid dimensions make tie placement easier with larger canvases.
            pass

        grid_h = random.choice([19, 21, 23])
        grid_w = random.choice([19, 21, 23])
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        protected = _place_frame(grid, out_shape)
        ordered = sorted(pieces, key=lambda piece: -piece["area"])
        if not _place_rods(grid, ordered, protected):
            continue

        expected = _expected_output(out_shape, ordered)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        unique_colors = {int(v) for v in grid.flatten() if int(v) not in {BG, FRAME, HINGE}}
        if len(unique_colors) != len(pieces):
            continue
        if len(_parse_pieces(grid)) != len(pieces):
            continue
        return grid

    raise AssertionError("failed to generate a valid puzzle instance")
