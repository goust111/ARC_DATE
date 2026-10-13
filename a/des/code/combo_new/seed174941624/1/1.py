import random
from collections import Counter

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_MIN = 2
MAX_FRAME_COUNT_SMALL = 3
MAX_FRAME_COUNT_LARGE = 4
MAX_RETRIES = 500

FRAME_SIZE = 5
INNER_SIZE = 3
MIN_RAY_LEN = 2
BOX_PAD = 1
PATTERN_RETRIES = 200

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GAP_OFFSETS = {
    "up": (0, 2),
    "down": (4, 2),
    "left": (2, 0),
    "right": (2, 4),
}

VERTICAL_ORBITS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]

HORIZONTAL_ORBITS = [
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 2)),
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
]


def _make_frame_template(gap_dir: str) -> np.ndarray:
    template = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    template[0, :] = True
    template[-1, :] = True
    template[:, 0] = True
    template[:, -1] = True
    gr, gc = GAP_OFFSETS[gap_dir]
    template[gr, gc] = False
    return template


FRAME_TEMPLATES = {gap_dir: _make_frame_template(gap_dir) for gap_dir in DIRS}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _match_frame_component(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) != 15:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 != FRAME_SIZE or c1 - c0 + 1 != FRAME_SIZE:
        return None
    local = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    for gap_dir, template in FRAME_TEMPLATES.items():
        if np.array_equal(local, template):
            return gap_dir
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            gap_dir = _match_frame_component(comp)
            if gap_dir is None:
                continue
            r0, c0, _, _ = _bbox(comp)
            frames.append({"r": int(r0), "c": int(c0), "gap_dir": gap_dir, "color": color})
    assert frames, "expected at least one frame"
    frames.sort(key=lambda frame: (frame["r"], frame["c"]))
    return frames


def _mismatch_pairs(pattern: np.ndarray, axis: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if axis == "vertical":
        pairs = [((r, 0), (r, 2)) for r in range(INNER_SIZE)]
    else:
        pairs = [((0, c), (2, c)) for c in range(INNER_SIZE)]
    return [pair for pair in pairs if int(pattern[pair[0]]) != int(pattern[pair[1]])]


def _candidate_axes(pattern: np.ndarray) -> list[str]:
    axes: list[str] = []
    for axis in ("vertical", "horizontal"):
        if len(_mismatch_pairs(pattern, axis)) == 1:
            axes.append(axis)
    return axes


def _infer_ray_color(pattern: np.ndarray, bg: int) -> int:
    counts = Counter(int(value) for value in pattern.flat)
    inferred: list[int] = []

    for axis in ("vertical", "horizontal"):
        mismatches = _mismatch_pairs(pattern, axis)
        if len(mismatches) != 1:
            continue
        a, b = mismatches[0]
        ca = int(pattern[a])
        cb = int(pattern[b])
        if counts[ca] < counts[cb]:
            inferred.append(ca)
        elif counts[cb] < counts[ca]:
            inferred.append(cb)
        elif ca != bg and cb == bg:
            inferred.append(ca)
        elif cb != bg and ca == bg:
            inferred.append(cb)

    if inferred:
        for color in inferred:
            if color != bg:
                return int(color)
        return int(inferred[0])

    ordered_colors: list[int] = []
    for value in pattern.flat:
        color = int(value)
        if color != bg and color not in ordered_colors:
            ordered_colors.append(color)
    assert ordered_colors, "expected at least one non-background interior color"
    min_count = min(counts[color] for color in ordered_colors)
    for color in ordered_colors:
        if counts[color] == min_count:
            return int(color)
    raise AssertionError("failed to infer the ray color")


def _frame_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        template = FRAME_TEMPLATES[frame["gap_dir"]]
        r0 = frame["r"]
        c0 = frame["c"]
        mask[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] |= template
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot colored rays outward from square-frame gaps using the asymmetric pixel inside each frame.

    Input structure:
    - The background is the dominant color.
    - Several 5x5 hollow square frames appear in the grid; each frame border is one color and has exactly one one-cell gap at the midpoint of one side.
    - Inside each frame is a 3x3 pattern made from background plus non-frame colors; it is almost left-right or top-bottom symmetric except for one differently colored pixel.

    Rule:
    - Detect every 5x5 frame and read its gap direction.
    - Inspect the interior 3x3 pattern, find the unique symmetry-breaking pixel, and use that pixel's color as the ray color.
    - Starting at the gap cell, paint a straight ray outward in the gap's direction until the ray would leave the grid or hit a frame border cell; do not overwrite that blocking border cell.

    Key insight:
    - The frame gap determines where the ray goes, but the ray color must be inferred separately from the lone anomaly that breaks the interior pattern's mirror symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    frame_mask = _frame_mask(g.shape, frames)

    out = g.copy()
    for frame in frames:
        r0 = frame["r"]
        c0 = frame["c"]
        pattern = g[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE]
        ray_color = _infer_ray_color(pattern, bg)

        gr, gc = GAP_OFFSETS[frame["gap_dir"]]
        dr, dc = DIRS[frame["gap_dir"]]
        r = r0 + gr
        c = c0 + gc

        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and not frame_mask[r, c]:
            out[r, c] = ray_color
            r += dr
            c += dc

    return out


def _boxes_too_close(a: dict, b: dict, pad: int = BOX_PAD) -> bool:
    ar0, ac0 = a["r"] - pad, a["c"] - pad
    ar1, ac1 = a["r"] + FRAME_SIZE - 1 + pad, a["c"] + FRAME_SIZE - 1 + pad
    br0, bc0 = b["r"] - pad, b["c"] - pad
    br1, bc1 = b["r"] + FRAME_SIZE - 1 + pad, b["c"] + FRAME_SIZE - 1 + pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_frame_count(h: int, w: int) -> int:
    max_count = MAX_FRAME_COUNT_LARGE if min(h, w) >= 22 else MAX_FRAME_COUNT_SMALL
    return random.randint(FRAME_COUNT_MIN, max_count)


def _sample_random_frame(h: int, w: int) -> dict:
    return {
        "r": random.randint(0, h - FRAME_SIZE),
        "c": random.randint(0, w - FRAME_SIZE),
        "gap_dir": random.choice(list(DIRS)),
    }


def _sample_blocker_pair(h: int, w: int) -> list[dict] | None:
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        max_gap = min(8, w - 10)
        if max_gap < 2:
            return None
        gap = random.randint(2, max_gap)
        span = 10 + gap
        r = random.randint(0, h - FRAME_SIZE)
        c_start = random.randint(0, w - span)
        if random.random() < 0.5:
            source = {"r": r, "c": c_start, "gap_dir": "right"}
            target = {
                "r": r,
                "c": c_start + FRAME_SIZE + gap,
                "gap_dir": random.choice(["up", "down", "right"]),
            }
        else:
            target = {"r": r, "c": c_start, "gap_dir": random.choice(["up", "down", "left"])}
            source = {
                "r": r,
                "c": c_start + FRAME_SIZE + gap,
                "gap_dir": "left",
            }
        return [source, target]

    max_gap = min(8, h - 10)
    if max_gap < 2:
        return None
    gap = random.randint(2, max_gap)
    span = 10 + gap
    r_start = random.randint(0, h - span)
    c = random.randint(0, w - FRAME_SIZE)
    if random.random() < 0.5:
        source = {"r": r_start, "c": c, "gap_dir": "down"}
        target = {
            "r": r_start + FRAME_SIZE + gap,
            "c": c,
            "gap_dir": random.choice(["left", "right", "down"]),
        }
    else:
        target = {"r": r_start, "c": c, "gap_dir": random.choice(["left", "right", "up"])}
        source = {
            "r": r_start + FRAME_SIZE + gap,
            "c": c,
            "gap_dir": "up",
        }
    return [source, target]


def _place_extra_frames(h: int, w: int, frames: list[dict], target_count: int) -> list[dict] | None:
    placed = [frame.copy() for frame in frames]
    while len(placed) < target_count:
        found = False
        for _ in range(80):
            candidate = _sample_random_frame(h, w)
            if any(_boxes_too_close(candidate, other) for other in placed):
                continue
            placed.append(candidate)
            found = True
            break
        if not found:
            return None
    return placed


def _ray_info(shape: tuple[int, int], frames: list[dict]) -> list[dict]:
    h, w = shape
    frame_mask = _frame_mask(shape, frames)
    info: list[dict] = []
    for frame in frames:
        gr, gc = GAP_OFFSETS[frame["gap_dir"]]
        dr, dc = DIRS[frame["gap_dir"]]
        r = frame["r"] + gr
        c = frame["c"] + gc
        path: list[tuple[int, int]] = []
        while 0 <= r < h and 0 <= c < w and not frame_mask[r, c]:
            path.append((r, c))
            r += dr
            c += dc
        stop_reason = "boundary" if not (0 <= r < h and 0 <= c < w) else "blocker"
        info.append({"path": path, "stop_reason": stop_reason})
    return info


def _path_enters_other_bbox(path: list[tuple[int, int]], frame_index: int, frames: list[dict]) -> bool:
    for idx, frame in enumerate(frames):
        if idx == frame_index:
            continue
        r0 = frame["r"]
        c0 = frame["c"]
        r1 = r0 + FRAME_SIZE - 1
        c1 = c0 + FRAME_SIZE - 1
        for r, c in path:
            if r0 <= r <= r1 and c0 <= c <= c1:
                return True
    return False


def _validate_layout(shape: tuple[int, int], frames: list[dict], mode: str) -> bool:
    h, w = shape
    for frame in frames:
        if not (0 <= frame["r"] <= h - FRAME_SIZE and 0 <= frame["c"] <= w - FRAME_SIZE):
            return False

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if _boxes_too_close(frames[i], frames[j]):
                return False

    info = _ray_info(shape, frames)
    path_sets = [set(item["path"]) for item in info]

    for idx, item in enumerate(info):
        if len(item["path"]) < MIN_RAY_LEN:
            return False
        if _path_enters_other_bbox(item["path"], idx, frames):
            return False

    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                return False

    if mode == "boundary_only":
        return all(item["stop_reason"] == "boundary" for item in info)
    return any(item["stop_reason"] == "blocker" for item in info)


def _sample_pattern() -> tuple[np.ndarray, str] | None:
    for _ in range(PATTERN_RETRIES):
        axis = random.choice(["vertical", "horizontal"])
        orbits = VERTICAL_ORBITS if axis == "vertical" else HORIZONTAL_ORBITS
        major_color, anomaly_color = random.sample(COLOR_POOL, 2)

        pattern = np.full((INNER_SIZE, INNER_SIZE), BG_COLOR, dtype=int)
        filled_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        filled_cells = 0

        for orbit in orbits:
            if random.random() < 0.58:
                for r, c in orbit:
                    pattern[r, c] = major_color
                    filled_cells += 1
                if len(orbit) == 2:
                    filled_pairs.append(orbit)  # type: ignore[arg-type]

        if not filled_pairs or filled_cells < 4 or filled_cells > 8:
            continue
        if axis == "vertical" and np.array_equal(pattern[0, :], pattern[2, :]):
            continue
        if axis == "horizontal" and np.array_equal(pattern[:, 0], pattern[:, 2]):
            continue

        orbit = random.choice(filled_pairs)
        cell = random.choice(orbit)
        pattern[cell] = anomaly_color

        if _candidate_axes(pattern) != [axis]:
            continue
        if np.count_nonzero(pattern == anomaly_color) != 1:
            continue
        if np.count_nonzero(pattern == major_color) < 3:
            continue
        return pattern, axis
    return None


def _assign_patterns(frames: list[dict]) -> bool:
    for frame in frames:
        sample = _sample_pattern()
        if sample is None:
            return False
        pattern, axis = sample
        frame["pattern"] = pattern
        frame["axis"] = axis
    return True


def _render_grid(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for frame in frames:
        r0 = frame["r"]
        c0 = frame["c"]
        template = FRAME_TEMPLATES[frame["gap_dir"]]
        grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE][template] = FRAME_COLOR
        grid[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE] = frame["pattern"]
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    mode = random.choice(["boundary_only", "blocked_pair"])
    frame_count = _sample_frame_count(h, w)

    if mode == "blocked_pair":
        pair = _sample_blocker_pair(h, w)
        if pair is None:
            return None
        frames = _place_extra_frames(h, w, pair, frame_count)
    else:
        frames = _place_extra_frames(h, w, [], frame_count)

    if frames is None:
        return None
    if not _validate_layout((h, w), frames, mode):
        return None
    if not _assign_patterns(frames):
        return None

    grid = _render_grid((h, w), frames)
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
