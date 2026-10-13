import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _centered_start(first: int, last: int, span: int) -> int:
    return (int(first) + int(last) - (int(span) - 1)) // 2


def _distance_to_gap(cell: tuple[int, int], frame: dict) -> int:
    r, c = cell
    side = frame["gap_side"]
    if side == "top":
        return int(r - frame["r0"])
    if side == "bottom":
        return int(frame["r1"] - r)
    if side == "left":
        return int(c - frame["c0"])
    return int(frame["c1"] - c)


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        cells = {tuple(map(int, pt)) for pt in comp}
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in cells):
            continue

        sides = {
            "top": [(r0, c) for c in range(c0, c1 + 1)],
            "bottom": [(r1, c) for c in range(c0, c1 + 1)],
            "left": [(r, c0) for r in range(r0, r1 + 1)],
            "right": [(r, c1) for r in range(r0, r1 + 1)],
        }

        missing_by_side: dict[str, list[int]] = {}
        valid = True
        for side, pts in sides.items():
            missing = [idx for idx, pt in enumerate(pts) if pt not in cells]
            if missing:
                if 0 in missing or len(pts) - 1 in missing:
                    valid = False
                    break
                if any(missing[i] + 1 != missing[i + 1] for i in range(len(missing) - 1)):
                    valid = False
                    break
            missing_by_side[side] = missing
        if not valid:
            continue

        gap_sides = [side for side, missing in missing_by_side.items() if missing]
        if len(gap_sides) != 1:
            continue

        gap_side = gap_sides[0]
        missing = missing_by_side[gap_side]
        if gap_side in ("top", "bottom"):
            gap_start = c0 + missing[0]
            gap_end = c0 + missing[-1]
        else:
            gap_start = r0 + missing[0]
            gap_end = r0 + missing[-1]

        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "gap_side": gap_side,
                "gap_start": int(gap_start),
                "gap_end": int(gap_end),
            }
        )

    frames.sort(key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    return frames


def _extract_shape_and_seed(grid: np.ndarray, frame: dict):
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    sub = grid[r0 + 1 : r1, c0 + 1 : c1]
    colored = (sub != BG) & (sub != FRAME)
    if not colored.any():
        return None

    comps = [comp + np.array([r0 + 1, c0 + 1], dtype=int) for comp in _components(colored)]
    singleton_cells = [tuple(map(int, comp[0])) for comp in comps if len(comp) == 1]
    if not singleton_cells:
        return None

    seed = min(singleton_cells, key=lambda cell: _distance_to_gap(cell, frame))
    seed_color = int(grid[seed])

    all_shape = np.argwhere(colored) + np.array([r0 + 1, c0 + 1], dtype=int)
    keep = (all_shape[:, 0] != seed[0]) | (all_shape[:, 1] != seed[1])
    shape = all_shape[keep]
    if len(shape) == 0:
        return None

    return seed_color, shape


def _paint_copy(out: np.ndarray, frame: dict, shape: np.ndarray, color: int) -> None:
    sr0, sc0, sr1, sc1 = _bbox(shape)
    mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
    mask[shape[:, 0] - sr0, shape[:, 1] - sc0] = True

    side = frame["gap_side"]
    if side == "top":
        mask = np.flipud(mask)
        dst_r0 = frame["r0"] - mask.shape[0]
        dst_c0 = _centered_start(frame["gap_start"], frame["gap_end"], mask.shape[1])
    elif side == "bottom":
        mask = np.flipud(mask)
        dst_r0 = frame["r1"] + 1
        dst_c0 = _centered_start(frame["gap_start"], frame["gap_end"], mask.shape[1])
    elif side == "left":
        mask = np.fliplr(mask)
        dst_r0 = _centered_start(frame["gap_start"], frame["gap_end"], mask.shape[0])
        dst_c0 = frame["c0"] - mask.shape[1]
    else:
        mask = np.fliplr(mask)
        dst_r0 = _centered_start(frame["gap_start"], frame["gap_end"], mask.shape[0])
        dst_c0 = frame["c1"] + 1

    rr, cc = np.nonzero(mask)
    for dr, dc in zip(rr.tolist(), cc.tolist()):
        r = dst_r0 + dr
        c = dst_c0 + dc
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each inner colored shape out through its frame gap using the seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for frame in _find_frames(g):
        extracted = _extract_shape_and_seed(g, frame)
        if extracted is None:
            continue
        seed_color, shape = extracted
        _paint_copy(out, frame, shape, seed_color)
    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    raw_h = rng.randint(2, 5)
    raw_w = rng.randint(2, 5)
    total = raw_h * raw_w
    target_min = max(3, (total + 2) // 3)
    target_max = max(target_min, total - 1)
    target = rng.randint(target_min, target_max)

    start = (rng.randrange(raw_h), rng.randrange(raw_w))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < raw_h and 0 <= nc < raw_w:
            frontier.add((nr, nc))

    while len(cells) < target:
        choice = rng.choice(tuple(frontier))
        frontier.remove(choice)
        cells.add(choice)
        for nr, nc in _neighbors4(*choice):
            if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((raw_h, raw_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _seed_candidates(ih: int, iw: int, side: str, gap_start: int, gap_end: int, shape_cells: set[tuple[int, int]]):
    band = 3
    for r in range(1, ih + 1):
        for c in range(1, iw + 1):
            if (r, c) in shape_cells:
                continue
            if side == "top":
                if r > min(ih, band) or not (gap_start <= c <= gap_end):
                    continue
            elif side == "bottom":
                if r < max(1, ih - band + 1) or not (gap_start <= c <= gap_end):
                    continue
            elif side == "left":
                if c > min(iw, band) or not (gap_start <= r <= gap_end):
                    continue
            else:
                if c < max(1, iw - band + 1) or not (gap_start <= r <= gap_end):
                    continue
            if any((nr, nc) in shape_cells for nr, nc in _neighbors4(r, c)):
                continue
            yield (r, c)


def _build_spec(rng: random.Random):
    side = rng.choice(SIDES)
    shape_mask = _random_connected_mask(rng)
    sh, sw = shape_mask.shape

    ih = rng.randint(sh + 2, min(sh + 5, 9))
    iw = rng.randint(sw + 2, min(sw + 5, 9))
    align_span = sw if side in ("top", "bottom") else sh
    side_cap = iw if side in ("top", "bottom") else ih
    gap_options = [length for length in range(align_span, side_cap + 1) if length % 2 == align_span % 2]
    if not gap_options:
        return None
    gap_len = rng.choice(gap_options[: min(len(gap_options), 4)])
    gap_start = rng.randint(1, side_cap - gap_len + 1)
    gap_end = gap_start + gap_len - 1

    for _ in range(40):
        shape_r = rng.randint(1, ih - sh + 1)
        shape_c = rng.randint(1, iw - sw + 1)
        local_shape = {
            (shape_r + int(r), shape_c + int(c))
            for r, c in np.argwhere(shape_mask)
        }
        seeds = list(_seed_candidates(ih, iw, side, gap_start, gap_end, local_shape))
        if seeds:
            break
    else:
        return None

    seed_r, seed_c = rng.choice(seeds)
    shape_color, seed_color = rng.sample(PALETTE, 2)
    fh, fw = ih + 2, iw + 2

    if side == "top":
        copy_r0 = -sh
        copy_c0 = _centered_start(gap_start, gap_end, sw)
    elif side == "bottom":
        copy_r0 = fh
        copy_c0 = _centered_start(gap_start, gap_end, sw)
    elif side == "left":
        copy_r0 = _centered_start(gap_start, gap_end, sh)
        copy_c0 = -sw
    else:
        copy_r0 = _centered_start(gap_start, gap_end, sh)
        copy_c0 = fw

    copy_r1 = copy_r0 + sh - 1
    copy_c1 = copy_c0 + sw - 1
    reserve_r0 = min(0, copy_r0)
    reserve_c0 = min(0, copy_c0)
    reserve_r1 = max(fh - 1, copy_r1)
    reserve_c1 = max(fw - 1, copy_c1)

    return {
        "side": side,
        "shape_mask": shape_mask,
        "shape_r": int(shape_r),
        "shape_c": int(shape_c),
        "seed_r": int(seed_r),
        "seed_c": int(seed_c),
        "shape_color": int(shape_color),
        "seed_color": int(seed_color),
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
        "ih": int(ih),
        "iw": int(iw),
        "fh": int(fh),
        "fw": int(fw),
        "copy_r0": int(copy_r0),
        "copy_c0": int(copy_c0),
        "reserve_r0": int(reserve_r0),
        "reserve_c0": int(reserve_c0),
        "reserve_r1": int(reserve_r1),
        "reserve_c1": int(reserve_c1),
    }


def _draw_input_object(grid: np.ndarray, spec: dict) -> None:
    fr, fc = spec["frame_top"], spec["frame_left"]
    fh, fw = spec["fh"], spec["fw"]

    grid[fr, fc : fc + fw] = FRAME
    grid[fr + fh - 1, fc : fc + fw] = FRAME
    grid[fr : fr + fh, fc] = FRAME
    grid[fr : fr + fh, fc + fw - 1] = FRAME

    if spec["side"] == "top":
        grid[fr, fc + spec["gap_start"] : fc + spec["gap_end"] + 1] = BG
    elif spec["side"] == "bottom":
        grid[fr + fh - 1, fc + spec["gap_start"] : fc + spec["gap_end"] + 1] = BG
    elif spec["side"] == "left":
        grid[fr + spec["gap_start"] : fr + spec["gap_end"] + 1, fc] = BG
    else:
        grid[fr + spec["gap_start"] : fr + spec["gap_end"] + 1, fc + fw - 1] = BG

    rr, cc = np.nonzero(spec["shape_mask"])
    grid[fr + spec["shape_r"] + rr, fc + spec["shape_c"] + cc] = spec["shape_color"]
    grid[fr + spec["seed_r"], fc + spec["seed_c"]] = spec["seed_color"]


def _draw_expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        mask = spec["shape_mask"]
        if spec["side"] in ("top", "bottom"):
            mask = np.flipud(mask)
        else:
            mask = np.fliplr(mask)
        rr, cc = np.nonzero(mask)
        base_r = spec["frame_top"] + spec["copy_r0"]
        base_c = spec["frame_left"] + spec["copy_c0"]
        out[base_r + rr, base_c + cc] = spec["seed_color"]
    return out


def _place_specs(rng: random.Random, h: int, w: int, specs: list[dict]):
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for raw_spec in specs:
        spec = dict(raw_spec)
        low_r = -spec["reserve_r0"]
        high_r = h - 1 - spec["reserve_r1"]
        low_c = -spec["reserve_c0"]
        high_c = w - 1 - spec["reserve_c1"]
        if low_r > high_r or low_c > high_c:
            return None

        for _ in range(200):
            frame_top = rng.randint(low_r, high_r)
            frame_left = rng.randint(low_c, high_c)
            rr0 = max(0, frame_top + spec["reserve_r0"] - 1)
            rr1 = min(h - 1, frame_top + spec["reserve_r1"] + 1)
            cc0 = max(0, frame_left + spec["reserve_c0"] - 1)
            cc1 = min(w - 1, frame_left + spec["reserve_c1"] + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            spec["frame_top"] = int(frame_top)
            spec["frame_left"] = int(frame_left)
            _draw_input_object(grid, spec)
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed.append(spec)
            break
        else:
            return None

    return grid, placed


def _gen_attempt(rng: random.Random):
    num_objects = rng.randint(1, 3)
    specs = []
    for _ in range(num_objects):
        spec = _build_spec(rng)
        if spec is None:
            return None
        specs.append(spec)

    h = rng.randint(max(18, 7 * num_objects), min(30, 11 * num_objects + 8))
    w = rng.randint(max(18, 7 * num_objects), min(30, 11 * num_objects + 8))
    placed = _place_specs(rng, h, w, specs)
    if placed is None:
        return None

    grid, placed_specs = placed
    expected = _draw_expected_output(grid, placed_specs)
    if len(_find_frames(grid)) != num_objects:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
