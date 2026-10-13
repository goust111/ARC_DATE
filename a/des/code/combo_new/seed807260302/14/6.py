from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PLATE_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 28
PLATE_SIDE_MIN = 8
PLATE_SIDE_MAX = 15
HOLE_COUNT_MIN = 2
HOLE_COUNT_MAX = 4
HOLE_AREA_MIN = 3
HOLE_AREA_MAX = 6
MAX_POLYOMINO_SPAN = 4
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _boundary_background(grid: np.ndarray) -> int:
    border = np.concatenate(
        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
    )
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _rotation_signatures(mask: np.ndarray) -> list[tuple[tuple[int, int], bytes]]:
    out: list[tuple[tuple[int, int], bytes]] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        out.append(((int(rot.shape[0]), int(rot.shape[1])), rot.astype(np.uint8).tobytes()))
    return out


def _distinct_rotation_count(mask: np.ndarray) -> int:
    return len(set(_rotation_signatures(mask)))


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(set(_rotation_signatures(mask)))


def _find_plate(grid: np.ndarray, bg: int) -> dict:
    best: dict | None = None
    colors = [int(c) for c in np.unique(grid) if int(c) != int(bg)]
    for color in colors:
        for coords in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not (
                np.all(sub[0, :] == int(color))
                and np.all(sub[-1, :] == int(color))
                and np.all(sub[:, 0] == int(color))
                and np.all(sub[:, -1] == int(color))
            ):
                continue
            if np.any((sub != int(color)) & (sub != int(bg))):
                continue
            if not bool(np.any(sub[1:-1, 1:-1] == int(bg))):
                continue
            score = (int(sub.shape[0] * sub.shape[1]), int(len(coords)))
            if best is None or score > best["score"]:
                best = {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "score": score,
                }

    if best is not None:
        plate_mask = np.zeros(grid.shape, dtype=bool)
        plate_mask[best["coords"][:, 0], best["coords"][:, 1]] = True
        best["mask"] = plate_mask
        return best

    fallback: dict | None = None
    for color in colors:
        for coords in _components(grid == int(color)):
            score = int(len(coords))
            if fallback is None or score > fallback["score"]:
                fallback = {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "score": score,
                }
    assert fallback is not None, "expected a non-background plate component"
    plate_mask = np.zeros(grid.shape, dtype=bool)
    plate_mask[fallback["coords"][:, 0], fallback["coords"][:, 1]] = True
    fallback["mask"] = plate_mask
    return fallback


def _extract_holes(grid: np.ndarray, plate_bbox: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, c0, r1, c1 = plate_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for coords in _components(sub == int(bg)):
        if any(
            rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1
            for rr, cc in coords
        ):
            continue
        mask, local_bbox = _tight_mask(coords)
        global_coords = coords + np.array([[r0, c0]], dtype=int)
        holes.append(
            {
                "mask": mask,
                "coords": global_coords,
                "bbox": (
                    int(local_bbox[0] + r0),
                    int(local_bbox[1] + c0),
                    int(local_bbox[2] + r0),
                    int(local_bbox[3] + c0),
                ),
                "area": int(mask.sum()),
            }
        )
    holes.sort(key=lambda h: (-int(h["area"]), int(h["bbox"][0]), int(h["bbox"][1])))
    return holes


def _extract_pieces(grid: np.ndarray, plate_mask: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    for coords in _components((grid != int(bg)) & (~plate_mask)):
        mask, box = _tight_mask(coords)
        color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
        pieces.append(
            {
                "mask": mask,
                "bbox": box,
                "color": color,
                "area": int(mask.sum()),
            }
        )
    pieces.sort(key=lambda p: (-int(p["area"]), int(p["bbox"][0]), int(p["bbox"][1])))
    return pieces


def _match_holes_to_pieces(holes: list[dict], pieces: list[dict]) -> dict[int, int]:
    candidates: list[list[int]] = []
    for hole in holes:
        compatible: list[int] = []
        for piece_index, piece in enumerate(pieces):
            if int(piece["area"]) != int(hole["area"]):
                continue
            if any(np.array_equal(np.rot90(piece["mask"], k), hole["mask"]) for k in range(4)):
                compatible.append(int(piece_index))
        assert compatible, "every hole must have at least one matching piece"
        candidates.append(compatible)

    order = sorted(range(len(holes)), key=lambda i: (len(candidates[i]), -int(holes[i]["area"])))
    assignment: dict[int, int] = {}
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        hole_index = int(order[pos])
        for piece_index in candidates[hole_index]:
            if piece_index in used:
                continue
            used.add(piece_index)
            assignment[hole_index] = piece_index
            if backtrack(pos + 1):
                return True
            used.remove(piece_index)
            del assignment[hole_index]
        return False

    assert backtrack(0), "shape matching should yield a one-to-one assignment"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each scattered colored piece back into the matching hole of the grey board.

    Input structure:
    - The background is black.
    - One large grey rectangular board appears in the grid; its interior contains black holes.
    - Outside the board are several small 4-connected colored pieces, one color per piece.

    Rule:
    - Find each interior hole of the grey rectangle and each outside piece.
    - Match a piece to a hole when the piece shape becomes identical after a 0/90/180/270 degree rotation.
    - Fill every hole with the matched piece color, erase the original outside pieces to black, and keep the grey board itself unchanged.

    Key insight:
    - The mapping is determined by rotational shape equivalence, so the solver must detect both the hole geometry and the required orientation of each detached piece.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _boundary_background(g)
    plate = _find_plate(g, bg)
    holes = _extract_holes(g, plate["bbox"], bg)
    pieces = _extract_pieces(g, plate["mask"], bg)

    assert holes, "expected at least one board hole"
    assert len(holes) == len(pieces), "piece count must equal hole count"

    match = _match_holes_to_pieces(holes, pieces)
    out = np.full_like(g, int(bg))
    out[plate["mask"]] = int(plate["color"])

    for hole_index, piece_index in match.items():
        coords = holes[int(hole_index)]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(pieces[int(piece_index)]["color"])

    return out


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = grid[int(top) : int(top) + mask.shape[0], int(left) : int(left) + mask.shape[1]]
    patch[mask] = int(color)


def _mark_dilated(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = occupied.shape
    for dr, dc in np.argwhere(mask):
        r = int(top + dr)
        c = int(left + dc)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                occupied[rr, cc] = True


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if top < 1 or left < 1 or top + mh >= h or left + mw >= w:
        return False
    window = occupied[int(top) : int(top) + mh, int(left) : int(left) + mw]
    return not bool(np.any(window & mask))


def _valid_positions(occupied: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = occupied.shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    for top in range(1, h - mh):
        for left in range(1, w - mw):
            if _can_place_mask(occupied, mask, int(top), int(left)):
                positions.append((int(top), int(left)))
    return positions


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < int(area):
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (int(cell[0] + dr), int(cell[1] + dc))
                if nxt not in cells:
                    frontier.add(nxt)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if max(r1 - r0 + 1, c1 - c0 + 1) > MAX_POLYOMINO_SPAN:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[int(r - r0), int(c - c0)] = True
        if _distinct_rotation_count(mask) < 2:
            continue
        return mask
    return None


def _place_holes_in_plate(plate_h: int, plate_w: int, hole_count: int) -> list[dict] | None:
    occupied = np.zeros((plate_h, plate_w), dtype=bool)
    holes: list[dict] = []
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    saw_four_way = False

    for index in range(hole_count):
        force_four_way = index == hole_count - 1 and not saw_four_way
        placed = False
        for _ in range(300):
            area = random.randint(HOLE_AREA_MIN, HOLE_AREA_MAX)
            mask = _random_polyomino(area)
            if mask is None:
                continue
            rot_count = _distinct_rotation_count(mask)
            if force_four_way and rot_count < 4:
                continue
            shape_key = _canonical_shape_key(mask)
            if shape_key in used_shapes:
                continue
            positions = _valid_positions(occupied, mask)
            if not positions:
                continue
            top, left = random.choice(positions)
            holes.append(
                {
                    "mask": mask,
                    "top": int(top),
                    "left": int(left),
                    "area": int(mask.sum()),
                    "rot_count": int(rot_count),
                }
            )
            used_shapes.add(shape_key)
            saw_four_way = saw_four_way or rot_count == 4
            _mark_dilated(occupied, mask, int(top), int(left))
            placed = True
            break
        if not placed:
            return None

    if not saw_four_way:
        return None
    return holes


def _build_expected_grid(
    shape: tuple[int, int], plate_top: int, plate_left: int, plate_h: int, plate_w: int, holes: list[dict]
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[int(plate_top) : int(plate_top) + plate_h, int(plate_left) : int(plate_left) + plate_w] = PLATE_COLOR
    for hole in holes:
        top = int(plate_top + hole["top"])
        left = int(plate_left + hole["left"])
        _paint_mask(out, hole["mask"], top, left, int(hole["color"]))
    return out


def _build_input_grid(
    shape: tuple[int, int], plate_top: int, plate_left: int, plate_h: int, plate_w: int, holes: list[dict]
) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    grid[int(plate_top) : int(plate_top) + plate_h, int(plate_left) : int(plate_left) + plate_w] = PLATE_COLOR
    for hole in holes:
        top = int(plate_top + hole["top"])
        left = int(plate_left + hole["left"])
        _paint_mask(grid, hole["mask"], top, left, BG)

    occupied = np.zeros(shape, dtype=bool)
    r0 = max(0, int(plate_top) - 1)
    c0 = max(0, int(plate_left) - 1)
    r1 = min(shape[0], int(plate_top) + plate_h + 1)
    c1 = min(shape[1], int(plate_left) + plate_w + 1)
    occupied[r0:r1, c0:c1] = True

    pieces = holes[:]
    pieces.sort(key=lambda item: -int(item["area"]))
    for hole in pieces:
        rotation = random.randint(0, 3)
        piece_mask = np.rot90(hole["mask"], rotation)
        positions = _valid_positions(occupied, piece_mask)
        if not positions:
            return None
        top, left = random.choice(positions)
        _paint_mask(grid, piece_mask, int(top), int(left), int(hole["color"]))
        _mark_dilated(occupied, piece_mask, int(top), int(left))

    return grid


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.randint(HOLE_COUNT_MIN, HOLE_COUNT_MAX)
    plate_h = random.randint(max(PLATE_SIDE_MIN, 2 * hole_count + 4), PLATE_SIDE_MAX)
    plate_w = random.randint(max(PLATE_SIDE_MIN, 2 * hole_count + 4), PLATE_SIDE_MAX)

    holes = _place_holes_in_plate(plate_h, plate_w, hole_count)
    if holes is None:
        return None

    colors = random.sample(PIECE_COLORS, hole_count)
    random.shuffle(colors)
    for hole, color in zip(holes, colors):
        hole["color"] = int(color)

    min_h = max(GRID_SIDE_MIN, plate_h + 6)
    min_w = max(GRID_SIDE_MIN, plate_w + 6)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None
    grid_h = random.randint(min_h, GRID_SIDE_MAX)
    grid_w = random.randint(min_w, GRID_SIDE_MAX)

    plate_top = random.randint(2, grid_h - plate_h - 2)
    plate_left = random.randint(2, grid_w - plate_w - 2)

    expected = _build_expected_grid((grid_h, grid_w), plate_top, plate_left, plate_h, plate_w, holes)
    inp = _build_input_grid((grid_h, grid_w), plate_top, plate_left, plate_h, plate_w, holes)
    if inp is None:
        return None

    if not np.array_equal(main(inp), expected):
        return None

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    if int(Counter(map(int, border)).most_common(1)[0][0]) != BG:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
