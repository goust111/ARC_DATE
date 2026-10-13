from collections import deque

import numpy as np


BG = 0
SEED_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GAP_SIDES = ("top", "right", "bottom", "left")
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _gap_side_from_coord(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str:
    gr, gc = gap
    assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if gr == r0:
        return "top"
    if gr == r1:
        return "bottom"
    if gc == c0:
        return "left"
    if gc == c1:
        return "right"
    raise AssertionError("gap must lie on the border")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    colors = sorted(int(color) for color in np.unique(grid) if int(color) not in {BG, SEED_COLOR})
    for color in colors:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            actual = {tuple(map(int, coord)) for coord in coords}
            expected = _border_coords(r0, c0, r1, c1)
            assert actual <= expected, "frame component must stay on its rectangle border"
            missing = list(expected - actual)
            assert len(missing) == 1, "frame border must have exactly one missing cell"
            frames.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "gap": missing[0],
                    "gap_side": _gap_side_from_coord(missing[0], r0, c0, r1, c1),
                    "seed": None,
                }
            )
    assert frames, "expected at least one frame"
    return frames


def _extract_seeds(grid: np.ndarray) -> list[dict]:
    seeds: list[dict] = []
    for coords in _components(grid == SEED_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == SEED_COLOR), "seed must be a filled grey rectangle"
        seeds.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "mask": np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool),
            }
        )
    return seeds


def _seed_belongs_to_frame(seed: dict, frame: dict) -> bool:
    return (
        frame["r0"] < seed["r0"] <= seed["r1"] < frame["r1"]
        and frame["c0"] < seed["c0"] <= seed["c1"] < frame["c1"]
    )


def _validate_seed_position(seed: dict, frame: dict) -> None:
    inner_h = frame["r1"] - frame["r0"] - 1
    inner_w = frame["c1"] - frame["c0"] - 1
    seed_h = seed["r1"] - seed["r0"] + 1
    seed_w = seed["c1"] - seed["c0"] + 1
    expected_r0 = frame["r0"] + 1 + (inner_h - seed_h) // 2
    expected_c0 = frame["c0"] + 1 + (inner_w - seed_w) // 2

    if frame["gap_side"] == "top":
        assert seed["r1"] == frame["r1"] - 1
        assert seed["c0"] == expected_c0
    elif frame["gap_side"] == "bottom":
        assert seed["r0"] == frame["r0"] + 1
        assert seed["c0"] == expected_c0
    elif frame["gap_side"] == "left":
        assert seed["c1"] == frame["c1"] - 1
        assert seed["r0"] == expected_r0
    else:
        assert frame["gap_side"] == "right"
        assert seed["c0"] == frame["c0"] + 1
        assert seed["r0"] == expected_r0


def _attach_seeds(grid: np.ndarray, frames: list[dict]) -> None:
    seeds = _extract_seeds(grid)
    assert len(seeds) == len(frames), "each frame must contain exactly one seed"
    for seed in seeds:
        matches = [frame for frame in frames if _seed_belongs_to_frame(seed, frame)]
        assert len(matches) == 1, "seed must belong to exactly one frame"
        frame = matches[0]
        assert frame["seed"] is None, "frame must contain only one seed"
        _validate_seed_position(seed, frame)
        frame["seed"] = seed
    assert all(frame["seed"] is not None for frame in frames)


def _stamp_seed(out: np.ndarray, frame: dict) -> None:
    seed = frame["seed"]
    assert seed is not None

    inner_r0 = frame["r0"] + 1
    inner_c0 = frame["c0"] + 1
    inner_r1 = frame["r1"] - 1
    inner_c1 = frame["c1"] - 1
    seed_h, seed_w = seed["mask"].shape

    if frame["gap_side"] == "top":
        step_r, step_c = -seed_h, 0
    elif frame["gap_side"] == "bottom":
        step_r, step_c = seed_h, 0
    elif frame["gap_side"] == "left":
        step_r, step_c = 0, -seed_w
    else:
        step_r, step_c = 0, seed_w

    k = 0
    while True:
        base_r = seed["r0"] + k * step_r
        base_c = seed["c0"] + k * step_c
        wrote = False
        for rr in range(seed_h):
            tr = base_r + rr
            if tr < inner_r0 or tr > inner_r1:
                continue
            for cc in range(seed_w):
                if not seed["mask"][rr, cc]:
                    continue
                tc = base_c + cc
                if inner_c0 <= tc <= inner_c1:
                    out[tr, tc] = frame["color"]
                    wrote = True
        if not wrote:
            break
        k += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey seed to its frame color and stamp it repeatedly through the frame interior toward the frame's one-cell gap.

    Input structure:
    - The background is black.
    - Several one-cell-thick colored rectangular frames appear on the canvas, and each frame border has exactly one one-cell gap on one side away from the corners.
    - Inside each frame there is exactly one filled grey rectangular seed placed centrally against the side opposite the gap.

    Rule:
    - Detect every colored frame and read the gap side as the propagation direction toward that gap.
    - Replace the grey seed with the frame's border color.
    - Repeat the seed pattern along the propagation axis using the seed's own height or width as the stride, clipping the last stamp if it extends past the frame interior.
    - Keep the frame border unchanged, including the original gap cell.

    Key insight:
    - The missing border cell is not an opening to fill through; it only tells you which way the inner grey seed should be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))

    frames = _extract_frames(g)
    _attach_seeds(g, frames)

    out = g.copy()
    for frame in frames:
        _stamp_seed(out, frame)
    return out


def _gap_index(rng: np.random.Generator, side_len: int) -> int:
    start = max(1, side_len // 3)
    end = min(side_len - 2, side_len - 1 - side_len // 3)
    if start > end:
        start, end = 1, side_len - 2
    return int(rng.integers(start, end + 1))


def _sample_object(rng: np.random.Generator, gap_side: str, color: int) -> dict:
    seed_h = int(rng.integers(1, 4))
    seed_w = int(rng.integers(1, 4))

    inner_h_min = max(seed_h, 3)
    inner_w_min = max(seed_w, 3)
    if gap_side in {"top", "bottom"}:
        inner_h_min = max(inner_h_min, seed_h + 2)
    else:
        inner_w_min = max(inner_w_min, seed_w + 2)

    inner_h = int(rng.integers(inner_h_min, 10))
    inner_w = int(rng.integers(inner_w_min, 10))
    full_h = inner_h + 2
    full_w = inner_w + 2
    gap_idx = _gap_index(rng, full_w if gap_side in {"top", "bottom"} else full_h)

    return {
        "color": color,
        "gap_side": gap_side,
        "seed_h": seed_h,
        "seed_w": seed_w,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "full_h": full_h,
        "full_w": full_w,
        "gap_idx": gap_idx,
    }


def _seed_top_left(frame: dict, obj: dict) -> tuple[int, int]:
    inner_h = frame["r1"] - frame["r0"] - 1
    inner_w = frame["c1"] - frame["c0"] - 1
    seed_h = obj["seed_h"]
    seed_w = obj["seed_w"]
    center_r = frame["r0"] + 1 + (inner_h - seed_h) // 2
    center_c = frame["c0"] + 1 + (inner_w - seed_w) // 2

    if obj["gap_side"] == "top":
        return frame["r1"] - seed_h, center_c
    if obj["gap_side"] == "bottom":
        return frame["r0"] + 1, center_c
    if obj["gap_side"] == "left":
        return center_r, frame["c1"] - seed_w
    return center_r, frame["c0"] + 1


def _draw_input_object(grid: np.ndarray, frame: dict, obj: dict) -> dict:
    color = obj["color"]
    grid[frame["r0"], frame["c0"] : frame["c1"] + 1] = color
    grid[frame["r1"], frame["c0"] : frame["c1"] + 1] = color
    grid[frame["r0"] : frame["r1"] + 1, frame["c0"]] = color
    grid[frame["r0"] : frame["r1"] + 1, frame["c1"]] = color

    if obj["gap_side"] == "top":
        gap = (frame["r0"], frame["c0"] + obj["gap_idx"])
    elif obj["gap_side"] == "bottom":
        gap = (frame["r1"], frame["c0"] + obj["gap_idx"])
    elif obj["gap_side"] == "left":
        gap = (frame["r0"] + obj["gap_idx"], frame["c0"])
    else:
        gap = (frame["r0"] + obj["gap_idx"], frame["c1"])
    grid[gap] = BG

    seed_r0, seed_c0 = _seed_top_left(frame, obj)
    seed_r1 = seed_r0 + obj["seed_h"] - 1
    seed_c1 = seed_c0 + obj["seed_w"] - 1
    grid[seed_r0 : seed_r1 + 1, seed_c0 : seed_c1 + 1] = SEED_COLOR

    return {
        **frame,
        "color": color,
        "gap": gap,
        "gap_side": obj["gap_side"],
        "seed": {
            "r0": seed_r0,
            "c0": seed_c0,
            "r1": seed_r1,
            "c1": seed_c1,
            "mask": np.ones((obj["seed_h"], obj["seed_w"]), dtype=bool),
        },
    }


def _region_is_free(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_region(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    object_count = int(rng.integers(2, 5))
    min_dim = 12 + 4 * (object_count - 1)
    height = int(rng.integers(min_dim, 31))
    width = int(rng.integers(min_dim, 31))

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    gap_sides = list(GAP_SIDES)
    rng.shuffle(gap_sides)
    colors = [int(color) for color in rng.choice(FRAME_COLORS, size=object_count, replace=False)]

    placed_frames: list[dict] = []
    for gap_side, color in zip(gap_sides[:object_count], colors):
        obj = _sample_object(rng, gap_side, color)
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = int(rng.integers(0, height - obj["full_h"] + 1))
            left = int(rng.integers(0, width - obj["full_w"] + 1))
            if not _region_is_free(occupied, top, left, obj["full_h"], obj["full_w"], pad=1):
                continue
            frame = {
                "r0": top,
                "c0": left,
                "r1": top + obj["full_h"] - 1,
                "c1": left + obj["full_w"] - 1,
            }
            placed_frames.append(_draw_input_object(grid, frame, obj))
            _mark_region(occupied, top, left, obj["full_h"], obj["full_w"], pad=1)
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for frame in placed_frames:
        _stamp_seed(expected, frame)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
