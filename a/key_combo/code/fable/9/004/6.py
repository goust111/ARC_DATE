import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append({"coords": arr, "bbox": _bbox(arr), "area": int(len(arr))})
    return components


def _rot_cw(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, k=-1)


def _rotate_phase(patch: np.ndarray, step_sign: int) -> np.ndarray:
    return np.rot90(patch, k=-1 if int(step_sign) == 1 else 1)


def _paste_with_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + h)
    cc1 = min(dst.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    dst[rr0:rr1, cc0:cc1] = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing member of an equally spaced rotating sprite family.

    Input structure:
    - A black-background grid contains repeated copies of one fully filled multicolor square sprite.
    - The copies lie on one horizontal or vertical arithmetic progression of positions.
    - Consecutive slots use the same sprite rotated by 90 degrees each step, and the first or last visible slot may be clipped by the canvas edge.

    Rule:
    - Identify the fully visible square copies; their top-left coordinates form an arithmetic progression with exactly one doubled gap.
    - Use an adjacent visible pair with normal spacing to determine whether phase advances clockwise or counterclockwise as position increases.
    - Fill the empty slot inside the doubled gap with the correctly rotated full sprite, leaving every existing visible copy unchanged.

    Key insight:
    - The missing frame is determined jointly by translation arithmetic and rotation phase; the doubled spacing locates the hole, while a normal neighbor pair fixes which 90-degree phase belongs there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert len(components) >= 3

    max_area = max(comp["area"] for comp in components)
    full = [comp for comp in components if comp["area"] == max_area]
    assert len(full) >= 3

    records = []
    for comp in full:
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert patch.shape[0] == patch.shape[1]
        assert int(np.count_nonzero(patch == bg)) == 0
        records.append({"bbox": comp["bbox"], "patch": patch})

    row_starts = [rec["bbox"][0] for rec in records]
    col_starts = [rec["bbox"][1] for rec in records]
    row_span = max(row_starts) - min(row_starts)
    col_span = max(col_starts) - min(col_starts)
    axis = "horizontal" if col_span >= row_span else "vertical"
    key = 1 if axis == "horizontal" else 0
    records.sort(key=lambda rec: rec["bbox"][key])

    diffs = [records[i + 1]["bbox"][key] - records[i]["bbox"][key] for i in range(len(records) - 1)]
    step = min(diffs)
    assert step > 0

    phase_sign = None
    for i, diff in enumerate(diffs):
        if diff != step:
            continue
        left_patch = records[i]["patch"]
        right_patch = records[i + 1]["patch"]
        if np.array_equal(_rot_cw(left_patch), right_patch):
            phase_sign = 1
            break
        if np.array_equal(_rot_cw(right_patch), left_patch):
            phase_sign = -1
            break
    assert phase_sign in (-1, 1)

    gap_indices = [i for i, diff in enumerate(diffs) if diff == 2 * step]
    assert len(gap_indices) == 1
    gap_left = gap_indices[0]
    left_box = records[gap_left]["bbox"]
    missing_patch = _rotate_phase(records[gap_left]["patch"], phase_sign)
    expected_right = _rotate_phase(missing_patch, phase_sign)
    assert np.array_equal(expected_right, records[gap_left + 1]["patch"])

    mr = left_box[0] + (step if axis == "vertical" else 0)
    mc = left_box[1] + (step if axis == "horizontal" else 0)
    size = missing_patch.shape[0]
    assert 0 <= mr and 0 <= mc
    assert mr + size <= g.shape[0] and mc + size <= g.shape[1]
    assert bool(np.all(g[mr : mr + size, mc : mc + size] == bg))

    out = g.copy()
    out[mr : mr + size, mc : mc + size] = missing_patch
    return out


def _sample_sprite(size: int) -> np.ndarray:
    while True:
        colors = random.sample(COLOR_POOL, 3)
        sprite = np.array([[random.choice(colors) for _ in range(size)] for _ in range(size)], dtype=int)
        if set(map(int, sprite.flat)) != set(colors):
            continue
        rotations = [np.rot90(sprite, k=-k) for k in range(4)]
        distinct = {rot.tobytes() for rot in rotations}
        if len(distinct) == 4:
            return sprite


def _choose_layout() -> tuple[int, int, int, bool, bool]:
    while True:
        size = random.choice((3, 3, 5))
        blank = random.choice((1, 2, 3)) if size == 3 else 1
        clip_low, clip_high = random.choice(
            (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            )
        )
        candidates = []
        for count in range(5, 8):
            if count - int(clip_low) - int(clip_high) < 4:
                continue
            span = size + (count - 1) * (size + blank)
            if span <= 30:
                candidates.append(count)
        if candidates:
            return size, blank, random.choice(candidates), clip_low, clip_high


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    axis = random.choice(("horizontal", "vertical"))
    size, blank, count, clip_low, clip_high = _choose_layout()
    step = size + blank
    sprite = _sample_sprite(size)
    base_phase = random.randrange(4)
    phase_sign = random.choice((-1, 1))

    if clip_low:
        start = -random.randint(1, size - 1)
    else:
        start = random.randint(0, 2)

    last = start + (count - 1) * step
    if clip_high:
        dim_primary = last + random.randint(1, size - 1)
    else:
        dim_primary = last + size + random.randint(0, 1)
    if dim_primary > 30:
        return None

    dim_other = random.randint(size + 1, min(18, 30))
    offset_other = random.randint(0, dim_other - size)

    if axis == "horizontal":
        height, width = dim_other, dim_primary
        row0 = offset_other
        col0 = start
    else:
        height, width = dim_primary, dim_other
        row0 = start
        col0 = offset_other

    full_low = 1 if clip_low else 0
    full_high = count - 2 if clip_high else count - 1
    gap_index = random.randint(full_low + 1, full_high - 1)

    output = np.full((height, width), BACKGROUND, dtype=int)
    for idx in range(count):
        phase = (base_phase + phase_sign * idx) % 4
        patch = np.rot90(sprite, k=-phase)
        rr = row0 + (idx * step if axis == "vertical" else 0)
        cc = col0 + (idx * step if axis == "horizontal" else 0)
        _paste_with_clip(output, patch, rr, cc)

    input_grid = output.copy()
    gap_r = row0 + (gap_index * step if axis == "vertical" else 0)
    gap_c = col0 + (gap_index * step if axis == "horizontal" else 0)
    assert 0 <= gap_r and 0 <= gap_c
    assert gap_r + size <= height and gap_c + size <= width
    input_grid[gap_r : gap_r + size, gap_c : gap_c + size] = BACKGROUND

    try:
        if not np.array_equal(main(input_grid), output):
            return None
    except Exception:
        return None

    components = _find_components(input_grid, BACKGROUND)
    if len(components) < 4:
        return None

    areas = Counter(comp["area"] for comp in components)
    if areas[max(areas)] < 3:
        return None

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid rotating-gap ARC input")
