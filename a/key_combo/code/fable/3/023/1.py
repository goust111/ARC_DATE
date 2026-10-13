from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    color = int(color)

    for sr, sc in np.argwhere(grid == color):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[Coord] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _find_sockets(grid: np.ndarray, bg: int) -> list[dict]:
    """Find 3x3 background cavities whose immediate 5x5 perimeter is one wall color."""
    H, W = grid.shape
    sockets: list[dict] = []
    for r0 in range(1, H - 3):
        for c0 in range(1, W - 3):
            if not np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == bg):
                continue

            frame = grid[r0 - 1 : r0 + 4, c0 - 1 : c0 + 4]
            border = np.concatenate((frame[0, :], frame[-1, :], frame[1:-1, 0], frame[1:-1, -1]))
            vals = set(map(int, border))
            if len(vals) != 1:
                continue
            wall = vals.pop()
            if wall == int(bg):
                continue
            sockets.append(
                {
                    "bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)),
                    "center": (int(r0 + 1), int(c0 + 1)),
                    "wall": int(wall),
                }
            )

    sockets.sort(key=lambda item: item["center"])
    return sockets


def _find_pieces(grid: np.ndarray, bg: int, wall_colors: set[int]) -> list[dict]:
    pieces: list[dict] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != int(bg) and int(v) not in wall_colors]:
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
                continue
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((region != int(bg)) & (region != int(color))):
                continue

            patch = np.full((3, 3), int(bg), dtype=int)
            patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
            pieces.append(
                {
                    "color": int(color),
                    "coords": coords.copy(),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "center": (int(r0 + 1), int(c0 + 1)),
                    "patch": patch,
                }
            )

    pieces.sort(key=lambda item: item["center"])
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll each external 3x3 piece into its aligned square socket, rotating by travel distance.

    Input structure:
    - The background is the most common color.
    - One wall color forms several large structures, each with a 3x3 background cavity bordered by that wall color.
    - Outside the structures are monochrome asymmetric 3x3-bbox pieces.
    - Each socket center is aligned in exactly one row or column with exactly one external piece center.

    Rule:
    - For each 3x3 cavity, find the uniquely row- or column-aligned 3x3 piece.
    - Let d be the Manhattan distance between their centers along that alignment axis.
    - Erase the piece from its original position, rotate it clockwise by (d mod 4) quarter-turns, and place it into the cavity.
    - Wall cells and every unrelated pixel remain unchanged.

    Key insight:
    - The empty square socket carries no orientation cue; the rolling distance from piece center to socket center is the only rotation code.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)

    sockets = _find_sockets(g, bg)
    assert sockets, "expected at least one 3x3 interior socket"
    wall_colors = {int(s["wall"]) for s in sockets}

    pieces = _find_pieces(g, bg, wall_colors)
    assert len(pieces) == len(sockets), "expected one movable 3x3 piece per socket"

    assignments: list[tuple[dict, dict, int]] = []
    used_piece_ids: set[int] = set()
    for socket in sockets:
        sr, sc = socket["center"]
        aligned = [
            (idx, piece)
            for idx, piece in enumerate(pieces)
            if piece["center"][0] == sr or piece["center"][1] == sc
        ]
        assert len(aligned) == 1, "each socket must have exactly one aligned piece"
        idx, piece = aligned[0]
        assert idx not in used_piece_ids, "a piece cannot serve two sockets"
        used_piece_ids.add(idx)
        pr, pc = piece["center"]
        assert (pr == sr) ^ (pc == sc), "piece must align on exactly one axis"
        assignments.append((socket, piece, abs(int(sr - pr)) + abs(int(sc - pc))))

    assert len(used_piece_ids) == len(pieces), "all pieces must be matched"

    out = g.copy()
    for _socket, piece, _dist in assignments:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for socket, piece, dist in assignments:
        r0, c0, _r1, _c1 = socket["bbox"]
        rotated = np.rot90(piece["patch"], -(int(dist) % 4))
        out[r0 : r0 + 3, c0 : c0 + 3] = int(bg)
        mask = rotated != int(bg)
        out[r0 : r0 + 3, c0 : c0 + 3][mask] = rotated[mask]

    return out


# --------------- generator ---------------

_DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_RETRIES = 800
_PLACE_PADDING = 1
_MAX_DISTANCE = 8

_BASE_PIECES = (
    np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
)


def _mask_is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    q: deque[Coord] = deque([cells[0]])
    cell_set = set(cells)
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _rot_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _random_piece_mask() -> np.ndarray:
    mask = random.choice(_BASE_PIECES).copy()
    if random.randrange(2):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    assert _mask_is_connected(mask) and _rot_asymmetric(mask)
    return mask


def _distance_for_mod(struct_size: int, mod: int) -> int:
    min_dist = struct_size // 2 + 2
    choices = [d for d in range(min_dist, _MAX_DISTANCE + 1) if d % 4 == int(mod)]
    assert choices
    return int(random.choice(choices))


def _paste_visible(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != int(bg)
    dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _make_pair_patch(
    *,
    bg: int,
    wall: int,
    piece_color: int,
    piece_mask: np.ndarray,
    struct_size: int,
    direction: Coord,
    distance: int,
) -> dict:
    half = int(struct_size // 2)
    dr, dc = map(int, direction)
    piece_center_rel = (int(dr * distance), int(dc * distance))

    min_r = min(-half, piece_center_rel[0] - 1)
    min_c = min(-half, piece_center_rel[1] - 1)
    max_r = max(half, piece_center_rel[0] + 1)
    max_c = max(half, piece_center_rel[1] + 1)
    h = int(max_r - min_r + 1)
    w = int(max_c - min_c + 1)

    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)

    struct_r = int(-half - min_r)
    struct_c = int(-half - min_c)
    socket_center = (int(-min_r), int(-min_c))
    socket_top = (int(socket_center[0] - 1), int(socket_center[1] - 1))
    piece_center = (int(piece_center_rel[0] - min_r), int(piece_center_rel[1] - min_c))
    piece_top = (int(piece_center[0] - 1), int(piece_center[1] - 1))

    for rr in range(struct_size):
        for cc in range(struct_size):
            rel_r = rr - half
            rel_c = cc - half
            if -1 <= rel_r <= 1 and -1 <= rel_c <= 1:
                continue
            inp[struct_r + rr, struct_c + cc] = int(wall)
            out[struct_r + rr, struct_c + cc] = int(wall)

    piece_patch = np.full((3, 3), int(bg), dtype=int)
    piece_patch[piece_mask] = int(piece_color)
    inp[piece_top[0] : piece_top[0] + 3, piece_top[1] : piece_top[1] + 3][piece_mask] = int(piece_color)

    rotated = np.rot90(piece_patch, -(int(distance) % 4))
    rot_mask = rotated != int(bg)
    out[socket_top[0] : socket_top[0] + 3, socket_top[1] : socket_top[1] + 3][rot_mask] = rotated[rot_mask]

    return {
        "input": inp,
        "output": out,
        "direction": (int(dr), int(dc)),
        "distance": int(distance),
        "mod": int(distance) % 4,
        "socket_center": socket_center,
        "piece_center": piece_center,
        "shape": (int(h), int(w)),
    }


def _place_zone(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    if h > H or w > W:
        return None
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        r0 = max(0, r - _PLACE_PADDING)
        c0 = max(0, c - _PLACE_PADDING)
        r1 = min(H, r + h + _PLACE_PADDING)
        c1 = min(W, c + w + _PLACE_PADDING)
        if not occupied[r0:r1, c0:c1].any():
            occupied[r0:r1, c0:c1] = True
            return int(r), int(c)
    return None


def _choose_directions(n: int) -> list[Coord]:
    repeated = random.choice(_DIRS)
    if repeated[0] == 0:
        perpendicular = [d for d in _DIRS if d[1] == 0]
    else:
        perpendicular = [d for d in _DIRS if d[0] == 0]
    dirs = [repeated, repeated, random.choice(perpendicular)]
    while len(dirs) < n:
        dirs.append(random.choice(_DIRS))
    return dirs


def _semantic_metadata_ok(items: list[dict]) -> bool:
    socket_centers = [item["global_socket_center"] for item in items]
    piece_centers = [item["global_piece_center"] for item in items]

    for i, socket in enumerate(socket_centers):
        aligned = [j for j, piece in enumerate(piece_centers) if piece[0] == socket[0] or piece[1] == socket[1]]
        if aligned != [i]:
            return False

    for i, piece in enumerate(piece_centers):
        aligned = [j for j, socket in enumerate(socket_centers) if piece[0] == socket[0] or piece[1] == socket[1]]
        if aligned != [i]:
            return False

    mods = {int(item["mod"]) for item in items}
    if 0 not in mods or len(mods) < 3:
        return False
    if not any(item["direction"][0] == 0 for item in items):
        return False
    if not any(item["direction"][1] == 0 for item in items):
        return False

    by_dir: dict[Coord, set[int]] = {}
    for item in items:
        by_dir.setdefault(item["direction"], set()).add(int(item["mod"]))
    if not any(len(mods_for_dir) >= 2 for mods_for_dir in by_dir.values()):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(24, 30)
    W = random.randint(24, 30)
    bg = random.randrange(10)
    wall = random.choice([c for c in range(10) if c != bg])
    piece_color = random.choice([c for c in range(10) if c not in (bg, wall)])

    count = random.choice((3, 3, 4))
    mods = [0] + random.sample([1, 2, 3], count - 1)
    random.shuffle(mods)
    dirs = _choose_directions(count)
    piece_mask = _random_piece_mask()

    patches = []
    for direction, mod in zip(dirs, mods):
        struct_size = random.choice((5, 5, 7))
        distance = _distance_for_mod(struct_size, mod)
        patches.append(
            _make_pair_patch(
                bg=bg,
                wall=wall,
                piece_color=piece_color,
                piece_mask=piece_mask,
                struct_size=struct_size,
                direction=direction,
                distance=distance,
            )
        )

    random.shuffle(patches)
    patches.sort(key=lambda p: p["shape"][0] * p["shape"][1], reverse=True)

    inp = np.full((H, W), int(bg), dtype=int)
    expected = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    placed: list[dict] = []

    for patch in patches:
        ph, pw = patch["shape"]
        loc = _place_zone(occupied, ph, pw)
        if loc is None:
            return None
        r0, c0 = loc
        _paste_visible(inp, patch["input"], r0, c0, bg)
        _paste_visible(expected, patch["output"], r0, c0, bg)

        item = dict(patch)
        item["global_socket_center"] = (int(r0 + patch["socket_center"][0]), int(c0 + patch["socket_center"][1]))
        item["global_piece_center"] = (int(r0 + patch["piece_center"][0]), int(c0 + patch["piece_center"][1]))
        placed.append(item)

    if not _semantic_metadata_ok(placed):
        return None
    if _get_bg(inp) != int(bg):
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rolling-into-socket input")
