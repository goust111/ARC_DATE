import math
import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
GREY = 5
GEN_ATTEMPTS = 400
PIECE_COUNT_OPTIONS = (2, 3, 4)
AREA_OPTIONS = (3, 4, 5, 6, 7)

_RNG = random.Random()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=bool)
    return (tuple(int(x) for x in arr.shape), arr.tobytes())


def _canonical_rot_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = [_mask_key(np.rot90(mask, k)) for k in range(4)]
    return min(keys, key=lambda item: (item[0][0], item[0][1], item[1]))


def _rotation_count(mask: np.ndarray) -> int:
    return len({_mask_key(np.rot90(mask, k)) for k in range(4)})


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _has_hole(mask: np.ndarray) -> bool:
    holes = 0
    for comp in _components(~mask):
        if _touches_edge(comp, mask.shape[0], mask.shape[1]):
            continue
        holes += 1
    return holes > 0


def _extract_molds(grid: np.ndarray, bg: int) -> list[dict]:
    molds: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(sub[0, :] != color) or np.any(sub[-1, :] != color):
                continue
            if np.any(sub[:, 0] != color) or np.any(sub[:, -1] != color):
                continue
            if np.any((sub != color) & (sub != bg)):
                continue

            holes = []
            for local in _components(sub == bg):
                if _touches_edge(local, sub.shape[0], sub.shape[1]):
                    continue
                holes.append(local)
            if len(holes) != 1:
                continue

            hole_local = holes[0]
            hole_global = hole_local + np.array([r0, c0], dtype=int)
            hole_mask = _mask_from_coords(hole_local)
            molds.append(
                {
                    "color": color,
                    "coords": coords,
                    "hole_coords": hole_global,
                    "shape_key": _canonical_rot_key(hole_mask),
                    "area": int(len(hole_local)),
                }
            )
    return molds


def _extract_pieces(grid: np.ndarray, bg: int, mold_mask: np.ndarray) -> list[dict]:
    pieces: list[dict] = []
    loose_mask = (grid != int(bg)) & (~mold_mask)
    for coords in _components(loose_mask):
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "each fragment must be monochrome"
        mask = _mask_from_coords(coords)
        pieces.append(
            {
                "coords": coords,
                "shape_key": _canonical_rot_key(mask),
                "area": int(len(coords)),
                "color": int(colors[0]),
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate-match scattered colored fragments to grey molds and fill the matching cavities.

    Input structure:
    - The background is black.
    - The grid contains several grey rectangular mold blocks; each block keeps a complete grey outer border and has one connected black cavity inside it.
    - Separate single-color 4-connected fragment objects lie outside the molds, one fragment per cavity.

    Rule:
    - Detect every grey rectangular mold and extract the relative shape of its interior black cavity.
    - Detect every colored fragment and compare its shape with cavity shapes under the four rotations 0°, 90°, 180°, and 270°.
    - Match each fragment to the unique cavity with the same rotation-invariant shape, fill that cavity with the fragment color, and erase the fragment's original cells back to black.
    - Keep the grid size fixed and leave all non-cavity mold pixels unchanged.

    Key insight:
    - Pixel count alone is not enough because different fragments can share the same area; the correct pairing comes from rotation-aware shape matching.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    molds = _extract_molds(g, bg)
    assert molds, "expected at least one mold block"

    mold_mask = np.zeros(g.shape, dtype=bool)
    for mold in molds:
        mold_mask[mold["coords"][:, 0], mold["coords"][:, 1]] = True

    pieces = _extract_pieces(g, bg, mold_mask)
    assert len(pieces) == len(molds), "molds and fragments must have the same count"

    mold_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
    for mold in molds:
        key = mold["shape_key"]
        assert key not in mold_by_key, "mold cavity shapes must be unique up to rotation"
        mold_by_key[key] = mold

    piece_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
    for piece in pieces:
        key = piece["shape_key"]
        assert key not in piece_by_key, "fragment shapes must be unique up to rotation"
        piece_by_key[key] = piece

    assert set(mold_by_key) == set(piece_by_key), "every cavity must have a matching fragment"

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = int(bg)

    for key, mold in mold_by_key.items():
        color = int(piece_by_key[key]["color"])
        out[mold["hole_coords"][:, 0], mold["hole_coords"][:, 1]] = color

    return out


def _random_shape_with_area(rng: random.Random, area: int) -> np.ndarray | None:
    for _ in range(120):
        h = rng.randint(2, min(5, area))
        w = rng.randint(2, min(5, area + 1))
        if h * w < area or h * w == area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        frontier = set()
        for dr, dc in DIR4:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while int(mask.sum()) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for dr, dc in DIR4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != area:
            continue
        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) == mask.shape[0] * mask.shape[1]:
            continue
        if _has_hole(mask):
            continue
        if _rotation_count(mask) < 2:
            continue
        return mask
    return None


def _generate_shapes(rng: random.Random, piece_count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen = set()

    shared_area = rng.choice(AREA_OPTIONS[1:])
    while len(shapes) < 2:
        shape = _random_shape_with_area(rng, shared_area)
        if shape is None:
            return None
        key = _canonical_rot_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)

    while len(shapes) < piece_count:
        area = rng.choice(AREA_OPTIONS)
        shape = _random_shape_with_area(rng, area)
        if shape is None:
            return None
        key = _canonical_rot_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)

    rng.shuffle(shapes)
    area_counts = {}
    for shape in shapes:
        area_counts[int(shape.sum())] = area_counts.get(int(shape.sum()), 0) + 1
    if max(area_counts.values()) < 2:
        return None
    return shapes


def _build_mold_patch(rng: random.Random, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sh, sw = shape.shape
    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    patch = np.full((sh + top + bottom, sw + left + right), GREY, dtype=int)
    patch[top : top + sh, left : left + sw][shape] = BG
    hole_local = np.argwhere(shape) + np.array([top, left], dtype=int)
    return patch, hole_local


def _build_piece_patch(shape: np.ndarray, color: int, rotation: int) -> np.ndarray:
    rotated = np.rot90(shape, rotation)
    patch = np.full(rotated.shape, BG, dtype=int)
    patch[rotated] = int(color)
    return patch


def _fits(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h >= H or c0 + w >= W:
        return False
    area = occupied[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)]
    return not bool(np.any(area))


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = occupied.shape
    occupied[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)] = True


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    attempts: int,
) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = patch.shape
    for _ in range(attempts):
        r0 = rng.randint(1, H - h - 1)
        c0 = rng.randint(1, W - w - 1)
        if not _fits(occupied, r0, c0, h, w):
            continue
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        _mark_occupied(occupied, r0, c0, h, w)
        return (r0, c0)
    return None


def _expected_output(
    grid: np.ndarray,
    mold_records: list[dict],
    piece_records: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for piece in piece_records:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for mold, piece in zip(mold_records, piece_records):
        coords = mold["hole_coords"]
        out[coords[:, 0], coords[:, 1]] = int(piece["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.choice(PIECE_COUNT_OPTIONS)
    shapes = _generate_shapes(rng, piece_count)
    if shapes is None:
        return None

    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], piece_count)
    rotations = [rng.randrange(4) for _ in range(piece_count)]

    mold_patches = []
    piece_patches = []
    for shape, color, rotation in zip(shapes, colors, rotations):
        mold_patch, hole_local = _build_mold_patch(rng, shape)
        piece_patch = _build_piece_patch(shape, color, rotation)
        mold_patches.append((mold_patch, hole_local, shape))
        piece_patches.append((piece_patch, color))

    max_h = max(max(patch.shape[0] for patch, _, _ in mold_patches), max(patch.shape[0] for patch, _ in piece_patches))
    max_w = max(max(patch.shape[1] for patch, _, _ in mold_patches), max(patch.shape[1] for patch, _ in piece_patches))
    reserved = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch, _, _ in mold_patches)
    reserved += sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch, _ in piece_patches)
    base = int(math.ceil(math.sqrt(reserved))) + rng.randint(1, 3)
    min_side = max(base, max_h + 3, max_w + 3, 15)
    if min_side > 30:
        return None

    H = rng.randint(min_side, 30)
    W = rng.randint(min_side, 30)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    mold_records = []
    for mold_patch, hole_local, shape in sorted(mold_patches, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        pos = _place_patch(rng, grid, occupied, mold_patch, attempts=140)
        if pos is None:
            return None
        r0, c0 = pos
        mold_records.append(
            {
                "hole_coords": hole_local + np.array([r0, c0], dtype=int),
                "shape_key": _canonical_rot_key(shape),
            }
        )

    piece_records = []
    for piece_patch, color in sorted(piece_patches, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        pos = _place_patch(rng, grid, occupied, piece_patch, attempts=160)
        if pos is None:
            return None
        r0, c0 = pos
        coords = np.argwhere(piece_patch != BG) + np.array([r0, c0], dtype=int)
        piece_records.append(
            {
                "coords": coords,
                "shape_key": _canonical_rot_key(_mask_from_coords(coords)),
                "color": int(color),
            }
        )

    piece_by_key = {piece["shape_key"]: piece for piece in piece_records}
    if len(piece_by_key) != piece_count:
        return None
    for mold in mold_records:
        if mold["shape_key"] not in piece_by_key:
            return None

    mold_records.sort(key=lambda item: (item["shape_key"][0], item["shape_key"][1]))
    ordered_piece_records = [piece_by_key[mold["shape_key"]] for mold in mold_records]
    expected = _expected_output(grid, mold_records, ordered_piece_records)

    if _most_common_color(grid) != BG:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mold-fragment sample")
