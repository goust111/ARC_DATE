import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
KEY_COLORS = [1, 2, 3, 4, 6, 7, 9]

MAX_GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return _coords_to_mask(coords)


def _rotation_count(mask: np.ndarray) -> int:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        seen.add((tuple(map(int, rot.shape)), rot.tobytes()))
    return len(seen)


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(keys)


def _match_under_rotation(src: np.ndarray, dst: np.ndarray) -> bool:
    for k in range(4):
        rot = np.rot90(src, k)
        if rot.shape == dst.shape and np.array_equal(rot, dst):
            return True
    return False


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        border_area = 2 * (h + w) - 4
        if int(comp.shape[0]) != border_area:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not np.any(interior == AZURE):
            continue
        if not np.all(np.isin(interior, [BLACK, AZURE])):
            continue
        holes = _components(interior == BLACK)
        if len(holes) != 1:
            continue
        hole = holes[0]
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "hole_mask": _coords_to_mask(hole),
                "hole_coords": [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in hole],
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _find_keys(grid: np.ndarray) -> list[dict]:
    keys = []
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (BLACK, GREY, AZURE):
            continue
        for comp in _components(grid == color):
            keys.append(
                {
                    "color": int(color),
                    "coords": [(int(r), int(c)) for r, c in comp],
                    "mask": _coords_to_mask(comp),
                }
            )
    keys.sort(key=lambda item: (item["color"], len(item["coords"])))
    return keys


def _solve_assignment(keys: list[dict], holes: list[dict]) -> dict[int, int]:
    candidates = {
        ki: [hi for hi, hole in enumerate(holes) if _match_under_rotation(key["mask"], hole["hole_mask"])]
        for ki, key in enumerate(keys)
    }
    assert all(candidates.values()), "each key must match at least one hole"
    order = sorted(range(len(keys)), key=lambda idx: (len(candidates[idx]), -len(keys[idx]["coords"])))
    solutions: list[dict[int, int]] = []
    used: set[int] = set()
    partial: dict[int, int] = {}

    def dfs(i: int) -> None:
        if len(solutions) > 1:
            return
        if i == len(order):
            solutions.append(partial.copy())
            return
        ki = order[i]
        for hi in candidates[ki]:
            if hi in used:
                continue
            used.add(hi)
            partial[ki] = hi
            dfs(i + 1)
            used.remove(hi)
            partial.pop(ki)

    dfs(0)
    assert solutions, "matching must exist"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered keys to framed negative-space lockholes and paint each hole with the key color.

    Input structure:
    - Background is Black.
    - Several Grey 1-pixel rectangular frames contain Azure interiors with one irregular Black connected hole inside each frame.
    - Several small monochrome key sprites in distinct colors are scattered outside the frames.

    Rule:
    - Detect each Grey rectangle and extract the Black connected region inside its Azure interior as the lockhole shape.
    - Detect each colored key sprite outside the frames and compare it against every lockhole under the four 90-degree rotations.
    - Pair every key with the unique lockhole whose shape matches after rotation, fill that lockhole with the key color, and keep the Grey frame and Azure fill unchanged.
    - Erase the original key sprites back to Black background.

    Key insight:
    - The meaningful shapes are split across positive and negative space: keys are colored foreground components, while lockholes are Black cavities embedded in Azure regions, and only rotation-invariant shape matching links them.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    keys = _find_keys(g)
    assert len(frames) == len(keys) and frames, "input must contain one key per frame"
    assignment = _solve_assignment(keys, frames)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = BLACK
    for ki, hi in assignment.items():
        color = int(keys[ki]["color"])
        for r, c in frames[hi]["hole_coords"]:
            out[r, c] = color
    return out


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        max_area = min(8, h * w - 1)
        if max_area < 3:
            continue
        target = rng.randint(3, max_area)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask = _coords_to_mask(coords)
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if _rotation_count(mask) != 4:
            continue
        return mask
    return None


def _free_positions(
    occ: np.ndarray, h: int, w: int, rect_h: int, rect_w: int, pad: int, border: int
) -> list[tuple[int, int]]:
    out = []
    r_start = border
    r_stop = h - rect_h - border + 1
    c_start = border
    c_stop = w - rect_w - border + 1
    if r_stop <= r_start or c_stop <= c_start:
        return out
    for r0 in range(r_start, r_stop):
        for c0 in range(c_start, c_stop):
            rr0 = max(0, r0 - pad)
            rr1 = min(h, r0 + rect_h + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + rect_w + pad)
            if not occ[rr0:rr1, cc0:cc1].any():
                out.append((r0, c0))
    return out


def _occupy(occ: np.ndarray, r0: int, c0: int, rect_h: int, rect_w: int, pad: int) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r0 + rect_h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c0 + rect_w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords = []
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        grid[r, c] = color
        coords.append((r, c))
    return coords


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.randint(1, 4)
    min_side = max(18, 10 + 4 * n_pairs)
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    expected = grid.copy()
    occ = np.zeros((h, w), dtype=bool)
    colors = rng.sample(KEY_COLORS, n_pairs)
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    specs = []

    for color in colors:
        shape = _random_shape(rng)
        if shape is None:
            return None
        shape_key = _canonical_shape_key(shape)
        if shape_key in used_shapes:
            return None
        used_shapes.add(shape_key)

        hole_mask = np.rot90(shape, rng.randrange(4))
        key_mask = np.rot90(shape, rng.randrange(4))
        hole_h, hole_w = map(int, hole_mask.shape)
        inner_h = rng.randint(hole_h + 2, min(hole_h + 5, 9))
        inner_w = rng.randint(hole_w + 2, min(hole_w + 5, 9))
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        hole_top = rng.randint(1, inner_h - hole_h - 1)
        hole_left = rng.randint(1, inner_w - hole_w - 1)
        specs.append(
            {
                "color": int(color),
                "hole_mask": hole_mask,
                "key_mask": key_mask,
                "frame_h": frame_h,
                "frame_w": frame_w,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "hole_top": hole_top,
                "hole_left": hole_left,
            }
        )

    rng.shuffle(specs)
    for spec in specs:
        candidates = _free_positions(occ, h, w, spec["frame_h"], spec["frame_w"], pad=1, border=1)
        if not candidates:
            return None
        frame_top, frame_left = rng.choice(candidates)
        _occupy(occ, frame_top, frame_left, spec["frame_h"], spec["frame_w"], pad=1)

        patch = np.full((spec["frame_h"], spec["frame_w"]), AZURE, dtype=int)
        patch[0, :] = GREY
        patch[-1, :] = GREY
        patch[:, 0] = GREY
        patch[:, -1] = GREY
        for rr, cc in np.argwhere(spec["hole_mask"]):
            patch[1 + spec["hole_top"] + rr, 1 + spec["hole_left"] + cc] = BLACK
        grid[frame_top : frame_top + spec["frame_h"], frame_left : frame_left + spec["frame_w"]] = patch
        expected[frame_top : frame_top + spec["frame_h"], frame_left : frame_left + spec["frame_w"]] = patch
        for rr, cc in np.argwhere(spec["hole_mask"]):
            expected[frame_top + 1 + spec["hole_top"] + rr, frame_left + 1 + spec["hole_left"] + cc] = spec["color"]

    for spec in specs:
        key_h, key_w = map(int, spec["key_mask"].shape)
        candidates = _free_positions(occ, h, w, key_h, key_w, pad=1, border=1)
        if not candidates:
            return None
        key_top, key_left = rng.choice(candidates)
        _occupy(occ, key_top, key_left, key_h, key_w, pad=1)
        _paint_mask(grid, spec["key_mask"], key_top, key_left, spec["color"])

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
