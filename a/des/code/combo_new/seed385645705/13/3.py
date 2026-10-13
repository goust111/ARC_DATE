import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components(grid == int(color))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _matches_under_rotation(src: np.ndarray, dst: np.ndarray) -> bool:
    return any(np.array_equal(np.rot90(src, k), dst) for k in range(4))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate and insert each small key piece into the same-color rectangular lockhole.

    Input structure:
    - The background is the most common color.
    - Each non-background color appears in exactly two 4-connected monochrome components.
    - For each color, the larger component is a rectangle with one connected corner gap, and the smaller component is a detached key with the same shape under rotation.

    Rule:
    - For each color, identify the larger lock component and its full rectangular bbox.
    - Inside that bbox, treat the missing cells as the lockhole and compare its normalized shape with the smaller key under the four 90-degree rotations.
    - Erase the detached key from its original position and fill the lockhole with that color, completing the rectangle.

    Key insight:
    - The two same-color components are complementary parts of one rectangle: the smaller piece is the rotated shape of the larger piece's missing corner.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _bg_color(g)

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        comps = _components_of_color(g, color)
        assert len(comps) == 2
        comps.sort(key=lambda pts: int(pts.shape[0]))
        key_coords, lock_coords = comps[0], comps[1]

        lock_mask, (r0, c0, _r1, _c1) = _mask_from_coords(lock_coords)
        hole_comps = _components(~lock_mask)
        assert len(hole_comps) == 1
        hole_coords_local = hole_comps[0]
        hole_mask, _ = _mask_from_coords(hole_coords_local)

        key_mask, _ = _mask_from_coords(key_coords)
        assert _matches_under_rotation(key_mask, hole_mask)

        out[key_coords[:, 0], key_coords[:, 1]] = bg
        out[r0 + hole_coords_local[:, 0], c0 + hole_coords_local[:, 1]] = color

    return out


# --------------- generator ---------------

GEN_RETRIES = 500
PAIR_COUNT_OPTIONS = (1, 2, 3)
COLOR_POOL = tuple(range(1, 10))
FAMILIES = ("l", "t", "concave")
CORNER_TO_ROT = (0, 3, 2, 1)  # tl, tr, br, bl


def _rotation_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in range(1, 4))


def _make_l_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        arm_h = rng.randint(1, h - 1)
        arm_w = rng.randint(1, w - 1)
        mask = np.zeros((h, w), dtype=bool)
        mask[:arm_h, :] = True
        mask[:, :arm_w] = True
        area = int(mask.sum())
        if 4 <= area <= 12 and _rotation_asymmetric(mask):
            return mask
    raise RuntimeError("failed to create L-shaped key")


def _make_t_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, 5)
        w = rng.randint(4, 6)
        bar_h = rng.randint(1, min(2, h - 1))
        stem_h = rng.randint(bar_h + 1, h)
        stem_w = rng.randint(1, w - 2)
        stem_col = rng.randint(1, w - stem_w - 1)
        mask = np.zeros((h, w), dtype=bool)
        mask[:bar_h, :] = True
        mask[:stem_h, stem_col : stem_col + stem_w] = True
        area = int(mask.sum())
        if 4 <= area <= 12 and _rotation_asymmetric(mask):
            return mask
    raise RuntimeError("failed to create T-shaped key")


def _make_concave_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        cut_h = rng.randint(1, h - 1)
        cut_w = rng.randint(1, w - 1)
        mask = np.ones((h, w), dtype=bool)
        mask[h - cut_h :, w - cut_w :] = False
        area = int(mask.sum())
        if 4 <= area <= 12 and _rotation_asymmetric(mask):
            return mask
    raise RuntimeError("failed to create concave key")


def _base_shape(family: str, rng: random.Random) -> np.ndarray:
    if family == "l":
        return _make_l_shape(rng)
    if family == "t":
        return _make_t_shape(rng)
    if family == "concave":
        return _make_concave_shape(rng)
    raise ValueError(f"unknown family {family}")


def _corner_origin(full_h: int, full_w: int, small_h: int, small_w: int, corner: int) -> tuple[int, int]:
    if corner == 0:
        return 0, 0
    if corner == 1:
        return 0, full_w - small_w
    if corner == 2:
        return full_h - small_h, full_w - small_w
    return full_h - small_h, 0


def _build_pair_spec(color: int, rng: random.Random) -> dict:
    for _ in range(100):
        family = rng.choice(FAMILIES)
        base_hole = _base_shape(family, rng)
        corner = rng.randrange(4)
        hole = np.rot90(base_hole, CORNER_TO_ROT[corner])
        hole_area = int(hole.sum())
        hole_h, hole_w = map(int, hole.shape)

        rect_h = hole_h + rng.randint(2, 4)
        rect_w = hole_w + rng.randint(2, 4)
        while rect_h * rect_w <= 2 * hole_area:
            if rect_h <= rect_w:
                rect_h += 1
            else:
                rect_w += 1
        if rect_h > 10 or rect_w > 10:
            continue

        lock = np.ones((rect_h, rect_w), dtype=bool)
        hr0, hc0 = _corner_origin(rect_h, rect_w, hole_h, hole_w, corner)
        hole_cells = np.argwhere(hole)
        lock[hr0 + hole_cells[:, 0], hc0 + hole_cells[:, 1]] = False

        key = np.rot90(hole, rng.randrange(4))
        return {
            "color": int(color),
            "family": family,
            "corner": int(corner),
            "lock_mask": lock,
            "key_mask": key,
            "rect_shape": (int(rect_h), int(rect_w)),
        }
    raise RuntimeError("failed to create pair spec")


def _place_patch(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    grid[top + coords[:, 0], left + coords[:, 1]] = int(color)


def _reserve(forbidden: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    hh, ww = forbidden.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(hh, top + h + 1)
    c1 = min(ww, left + w + 1)
    forbidden[r0:r1, c0:c1] = True


def _find_spot(forbidden: np.ndarray, h: int, w: int, rng: random.Random) -> tuple[int, int] | None:
    hh, ww = forbidden.shape
    if h > hh or w > ww:
        return None
    for _ in range(250):
        top = rng.randint(0, hh - h)
        left = rng.randint(0, ww - w)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(hh, top + h + 1)
        c1 = min(ww, left + w + 1)
        if not forbidden[r0:r1, c0:c1].any():
            return int(top), int(left)
    return None


def _valid_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if _bg_color(inp) != 0:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    for color in [int(v) for v in np.unique(inp) if int(v) != 0]:
        comps = _components_of_color(inp, color)
        if len(comps) != 2:
            return False
        comps.sort(key=lambda pts: int(pts.shape[0]))
        key_coords, lock_coords = comps[0], comps[1]
        if key_coords.shape[0] >= lock_coords.shape[0]:
            return False
        lock_mask, _ = _mask_from_coords(lock_coords)
        hole_comps = _components(~lock_mask)
        if len(hole_comps) != 1:
            return False
        hole_mask, _ = _mask_from_coords(hole_comps[0])
        key_mask, _ = _mask_from_coords(key_coords)
        if not _matches_under_rotation(key_mask, hole_mask):
            return False

    for color in [int(v) for v in np.unique(expected) if int(v) != 0]:
        comps = _components_of_color(expected, color)
        if len(comps) != 1:
            return False
        rect_mask, _ = _mask_from_coords(comps[0])
        if not rect_mask.all():
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    pair_count = rng.choice(PAIR_COUNT_OPTIONS)
    min_dim = {1: 14, 2: 18, 3: 22}[pair_count]
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    try:
        specs = [_build_pair_spec(color, rng) for color in rng.sample(COLOR_POOL, pair_count)]
    except RuntimeError:
        return None
    specs.sort(
        key=lambda spec: max(
            spec["lock_mask"].shape[0] * spec["lock_mask"].shape[1],
            spec["key_mask"].shape[0] * spec["key_mask"].shape[1],
        ),
        reverse=True,
    )

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    for spec in specs:
        lock_h, lock_w = map(int, spec["lock_mask"].shape)
        key_h, key_w = map(int, spec["key_mask"].shape)

        lock_pos = _find_spot(forbidden, lock_h, lock_w, rng)
        if lock_pos is None:
            return None
        _reserve(forbidden, lock_pos[0], lock_pos[1], lock_h, lock_w)

        key_pos = _find_spot(forbidden, key_h, key_w, rng)
        if key_pos is None:
            return None
        _reserve(forbidden, key_pos[0], key_pos[1], key_h, key_w)

        _place_patch(inp, spec["lock_mask"], lock_pos[0], lock_pos[1], spec["color"])
        _place_patch(inp, spec["key_mask"], key_pos[0], key_pos[1], spec["color"])

        rect_h, rect_w = spec["rect_shape"]
        expected[lock_pos[0] : lock_pos[0] + rect_h, lock_pos[1] : lock_pos[1] + rect_w] = spec["color"]

    if not _valid_instance(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
