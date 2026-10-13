import random
from collections import Counter

import numpy as np


BG_COLOR = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SLOT_COUNT = 2

# --------------- shared helpers ---------------


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _slot_slices(length: int) -> tuple[slice, slice]:
    inner = length - 4
    assert inner >= 4, f"piece side too short for slot extraction: {length}"
    slot_width = inner // SLOT_COUNT
    gap = inner - SLOT_COUNT * slot_width
    start0 = 2
    end0 = start0 + slot_width
    start1 = end0 + gap
    end1 = start1 + slot_width
    return slice(start0, end0), slice(start1, end1)


def _partner_code(code: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    top, right, bottom, left = code
    return (
        tuple(-v for v in reversed(bottom)),
        tuple(-v for v in reversed(left)),
        tuple(-v for v in reversed(top)),
        tuple(-v for v in reversed(right)),
    )


def _enclosed_holes(shape_mask: np.ndarray) -> list[np.ndarray]:
    h, w = shape_mask.shape
    holes: list[np.ndarray] = []
    for coords in _components_mask(~shape_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        holes.append(coords)
    return holes


def _single_hole(shape_mask: np.ndarray) -> np.ndarray:
    holes = _enclosed_holes(shape_mask)
    assert len(holes) == 1, f"expected exactly one enclosed hole, found {len(holes)}"
    hole = holes[0]
    r0, c0, r1, c1 = _bbox_coords(hole)
    assert (r1 - r0 + 1) * (c1 - c0 + 1) == len(hole), "hole must be rectangular"
    return hole


def _profile_states(profile: np.ndarray) -> tuple[int, ...]:
    states = []
    for sl in _slot_slices(len(profile)):
        segment = profile[sl]
        assert segment.size > 0
        val = int(segment[0])
        assert 0 <= val <= 2, f"unexpected edge depth {val}"
        assert np.all(segment == val), "slot profile should be uniform"
        states.append(val - 1)
    return tuple(states)


def _shape_code(shape_mask: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    h, w = shape_mask.shape
    top = np.empty(w, dtype=int)
    bottom = np.empty(w, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(shape_mask[:, c])
        assert rows.size, "every column should contain foreground"
        top[c] = int(rows[0])
        bottom[c] = int(h - 1 - rows[-1])

    left = np.empty(h, dtype=int)
    right = np.empty(h, dtype=int)
    for r in range(h):
        cols = np.flatnonzero(shape_mask[r, :])
        assert cols.size, "every row should contain foreground"
        left[r] = int(cols[0])
        right[r] = int(w - 1 - cols[-1])

    return (
        _profile_states(top),
        _profile_states(right),
        _profile_states(bottom),
        _profile_states(left),
    )


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in _components_mask(grid != bg):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each shape should be monochromatic"
        color = colors.pop()
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(grid, bbox)
        shape_mask = patch == color
        hole_local = _single_hole(shape_mask)
        hole_global = hole_local + np.array([[bbox[0], bbox[1]]], dtype=int)
        objects.append(
            {
                "color": color,
                "bbox": bbox,
                "code": _shape_code(shape_mask),
                "hole": hole_global,
            }
        )
    assert len(objects) % 2 == 0 and objects, "expected an even number of shapes"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair hollow jigsaw pieces by complementary edge signatures and fill each hole
    with its partner's color.

    Each monochrome component contains one enclosed rectangular bg hole. The
    middle span of every side has two slot regions whose boundary depths encode
    convex / flat / concave states. Two pieces are partners iff one piece's
    clockwise side code equals the other's 180-rotated complement.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _extract_objects(g, bg)

    code_to_index: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]], int] = {}
    for i, obj in enumerate(objects):
        assert obj["code"] not in code_to_index, "piece signatures must be unique"
        code_to_index[obj["code"]] = i

    out = g.copy()
    for i, obj in enumerate(objects):
        partner_idx = code_to_index.get(_partner_code(obj["code"]))
        assert partner_idx is not None and partner_idx != i, "missing unique partner"
        partner_color = int(objects[partner_idx]["color"])
        hole = obj["hole"]
        out[hole[:, 0], hole[:, 1]] = partner_color
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
PAIR_COUNTS = (2, 3)
PIECE_SIDE_MIN = 8
PIECE_SIDE_MAX = 10
PLACEMENT_ATTEMPTS = 180
PIECE_BUILD_ATTEMPTS = 120
COLOR_POOL = [c for c in range(10) if c != BG_COLOR]
STATE_POOL = [-1, 0, 1]
STATE_WEIGHTS = [3, 4, 3]


def _random_code() -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    while True:
        values = random.choices(STATE_POOL, weights=STATE_WEIGHTS, k=4 * SLOT_COUNT)
        if values.count(-1) == 0 or values.count(1) == 0:
            continue
        if sum(v != 0 for v in values) < 4:
            continue
        code = (
            tuple(values[0:SLOT_COUNT]),
            tuple(values[SLOT_COUNT : 2 * SLOT_COUNT]),
            tuple(values[2 * SLOT_COUNT : 3 * SLOT_COUNT]),
            tuple(values[3 * SLOT_COUNT : 4 * SLOT_COUNT]),
        )
        if code != _partner_code(code):
            return code


def _random_hole_box(h: int, w: int) -> tuple[int, int, int, int]:
    hole_h = random.randint(1, 2)
    hole_w = random.randint(1, 2)
    r0 = random.randint(3, h - hole_h - 3)
    c0 = random.randint(3, w - hole_w - 3)
    return r0, c0, hole_h, hole_w


def _build_piece_mask(
    h: int,
    w: int,
    code: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    hole_box: tuple[int, int, int, int],
) -> np.ndarray:
    top_code, right_code, bottom_code, left_code = code
    mask = np.ones((h, w), dtype=bool)

    mask[0, 2 : w - 2] = False
    mask[h - 1, 2 : w - 2] = False
    mask[2 : h - 2, 0] = False
    mask[2 : h - 2, w - 1] = False

    for sl, state in zip(_slot_slices(w), top_code):
        if state == -1:
            mask[0, sl] = True
        elif state == 1:
            mask[1, sl] = False

    for sl, state in zip(_slot_slices(w), bottom_code):
        if state == -1:
            mask[h - 1, sl] = True
        elif state == 1:
            mask[h - 2, sl] = False

    for sl, state in zip(_slot_slices(h), left_code):
        if state == -1:
            mask[sl, 0] = True
        elif state == 1:
            mask[sl, 1] = False

    for sl, state in zip(_slot_slices(h), right_code):
        if state == -1:
            mask[sl, w - 1] = True
        elif state == 1:
            mask[sl, w - 2] = False

    r0, c0, hole_h, hole_w = hole_box
    mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False

    comps = _components_mask(mask)
    assert len(comps) == 1, "piece must stay connected"
    _single_hole(mask)
    return mask


def _make_pair_masks() -> tuple[dict, dict] | None:
    for _ in range(PIECE_BUILD_ATTEMPTS):
        h = random.randint(PIECE_SIDE_MIN, PIECE_SIDE_MAX)
        w = random.randint(PIECE_SIDE_MIN, PIECE_SIDE_MAX)
        code_a = _random_code()
        code_b = _partner_code(code_a)

        try:
            mask_a = _build_piece_mask(h, w, code_a, _random_hole_box(h, w))
            mask_b = _build_piece_mask(h, w, code_b, _random_hole_box(h, w))
        except AssertionError:
            continue

        rot = random.randrange(4)
        mask_a = np.rot90(mask_a, rot).copy()
        mask_b = np.rot90(mask_b, rot).copy()

        try:
            sig_a = _shape_code(mask_a)
            sig_b = _shape_code(mask_b)
            hole_a = _single_hole(mask_a)
            hole_b = _single_hole(mask_b)
        except AssertionError:
            continue

        if _partner_code(sig_a) != sig_b:
            continue

        return (
            {"mask": mask_a, "code": sig_a, "hole": hole_a},
            {"mask": mask_b, "code": sig_b, "hole": hole_b},
        )
    return None


def _place_pieces(pieces: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(pieces)
    order = list(range(len(pieces)))
    random.shuffle(order)

    for idx in order:
        mask = pieces[idx]["mask"]
        ph, pw = mask.shape
        if ph + 2 > h or pw + 2 > w:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r = random.randint(1, h - ph - 1)
            c = random.randint(1, w - pw - 1)
            if occupied[r - 1 : r + ph + 1, c - 1 : c + pw + 1].any():
                continue
            occupied[r : r + ph, c : c + pw] = True
            positions[idx] = (r, c)
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _gen_attempt(with_output: bool = False):
    n_pairs = random.choice(PAIR_COUNTS)
    colors = random.sample(COLOR_POOL, 2 * n_pairs)
    pieces: list[dict] = []

    for pair_id in range(n_pairs):
        pair = _make_pair_masks()
        if pair is None:
            return None
        a, b = pair
        a["partner"] = 2 * pair_id + 1
        b["partner"] = 2 * pair_id
        pieces.extend([a, b])

    codes = [piece["code"] for piece in pieces]
    if len(set(codes)) != len(codes):
        return None
    for i, code in enumerate(codes):
        matches = [j for j, other in enumerate(codes) if i != j and other == _partner_code(code)]
        if len(matches) != 1 or matches[0] != int(pieces[i]["partner"]):
            return None

    for piece, color in zip(pieces, colors):
        piece["color"] = int(color)

    max_h = max(piece["mask"].shape[0] for piece in pieces)
    max_w = max(piece["mask"].shape[1] for piece in pieces)
    grid_h = max(max_h + 4, 10 + 6 * n_pairs + random.randint(0, 4))
    grid_w = max(max_w + 4, 10 + 6 * n_pairs + random.randint(0, 4))

    positions = _place_pieces(pieces, grid_h, grid_w)
    if positions is None:
        return None

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    out = grid.copy()

    for piece, (r0, c0) in zip(pieces, positions):
        piece["pos"] = (r0, c0)
        mask = piece["mask"]
        ph, pw = mask.shape
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = int(piece["color"])
        out[r0 : r0 + ph, c0 : c0 + pw][mask] = int(piece["color"])

    for piece in pieces:
        partner_color = int(pieces[int(piece["partner"])]["color"])
        hole = piece["hole"]
        r0, c0 = piece["pos"]
        out[r0 + hole[:, 0], c0 + hole[:, 1]] = partner_color

    if _background_color(grid) != BG_COLOR:
        return None
    if not np.array_equal(main(grid), out):
        return None

    if with_output:
        return grid, out
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(with_output=False)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
