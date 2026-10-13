from __future__ import annotations

import numpy as np


SIDE_SPECS = (
    {"mid": (0, 1), "step": (-1, 0)},
    {"mid": (2, 1), "step": (1, 0)},
    {"mid": (1, 0), "step": (0, -1)},
    {"mid": (1, 2), "step": (0, 1)},
)

COLOR_POOL = tuple(range(10))
PAIR_COUNT_RANGE = (1, 5)
CABLE_LENGTH_RANGE = (1, 11)
GRID_MIN_SIZE = 12
GRID_MAX_SIZE = 30
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_border(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    return np.array(
        [
            grid[r, c],
            grid[r, c + 1],
            grid[r, c + 2],
            grid[r + 1, c],
            grid[r + 1, c + 2],
            grid[r + 2, c],
            grid[r + 2, c + 1],
            grid[r + 2, c + 2],
        ],
        dtype=int,
    )


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    seen_colors: set[int] = set()
    for r in range(h - 2):
        for c in range(w - 2):
            color = int(g[r, c])
            if color == bg:
                continue
            border = _frame_border(g, r, c)
            if np.all(border == color):
                center = (r + 1, c + 1)
                frame = {
                    "top_left": (r, c),
                    "color": color,
                    "center": center,
                    "center_color": int(g[center]),
                }
                assert color not in seen_colors, "frame colors must be unique"
                seen_colors.add(color)
                frames.append(frame)
    return frames


def _entry_map(frames: list[dict]) -> dict[tuple[int, int, int, int], dict]:
    entry: dict[tuple[int, int, int, int], dict] = {}
    for frame in frames:
        r0, c0 = frame["top_left"]
        for spec in SIDE_SPECS:
            mr = r0 + spec["mid"][0]
            mc = c0 + spec["mid"][1]
            sr, sc = spec["step"]
            key = (mr, mc, -sr, -sc)
            assert key not in entry
            entry[key] = frame
    return entry


def _trace_receiver(
    grid: np.ndarray,
    source: dict,
    mid_r: int,
    mid_c: int,
    step_r: int,
    step_c: int,
    entry: dict[tuple[int, int, int, int], dict],
) -> dict | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    color = int(source["color"])

    r = mid_r + step_r
    c = mid_c + step_c
    if not (0 <= r < h and 0 <= c < w):
        return None
    if int(g[r, c]) != color:
        return None

    while 0 <= r < h and 0 <= c < w and int(g[r, c]) == color:
        r += step_r
        c += step_c

    if not (0 <= r < h and 0 <= c < w):
        return None
    receiver = entry.get((r, c, step_r, step_c))
    if receiver is None or receiver is source:
        return None
    return receiver


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify 3x3 frame sprites and straight cable links, then copy each source
    center color into its connected receiver center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one frame"

    entry = _entry_map(frames)
    out = g.copy()

    for source in frames:
        signal = int(source["center_color"])
        if signal == bg:
            continue
        r0, c0 = source["top_left"]
        for spec in SIDE_SPECS:
            mid_r = r0 + spec["mid"][0]
            mid_c = c0 + spec["mid"][1]
            step_r, step_c = spec["step"]
            receiver = _trace_receiver(g, source, mid_r, mid_c, step_r, step_c, entry)
            if receiver is None:
                continue
            rr, cc = receiver["center"]
            current = int(out[rr, cc])
            assert current in (bg, signal), "receiver center cannot have conflicting signals"
            out[rr, cc] = signal

    return out


def _paint_frame(grid: np.ndarray, r0: int, c0: int, border: int, center: int) -> None:
    grid[r0 : r0 + 3, c0 : c0 + 3] = int(border)
    grid[r0 + 1, c0 + 1] = int(center)


def _build_pair_patch(
    bg: int,
    source_border: int,
    receiver_border: int,
    signal: int,
    vertical: bool,
    source_first: bool,
    cable_len: int,
) -> tuple[np.ndarray, tuple[int, int], int]:
    along = cable_len + 6
    patch = np.full((along, 3), bg, dtype=int) if vertical else np.full((3, along), bg, dtype=int)

    if source_first:
        source_pos = 0
        receiver_pos = cable_len + 3
    else:
        receiver_pos = 0
        source_pos = cable_len + 3

    if vertical:
        _paint_frame(patch, source_pos, 0, source_border, signal)
        _paint_frame(patch, receiver_pos, 0, receiver_border, bg)
        patch[3 : cable_len + 3, 1] = int(source_border)
        receiver_center = (receiver_pos + 1, 1)
    else:
        _paint_frame(patch, 0, source_pos, source_border, signal)
        _paint_frame(patch, 0, receiver_pos, receiver_border, bg)
        patch[1, 3 : cable_len + 3] = int(source_border)
        receiver_center = (1, receiver_pos + 1)

    return patch, receiver_center, int(signal)


def _reserve_ok(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0], r0 + h + padding)
    cc1 = min(blocked.shape[1], c0 + w + padding)
    return not bool(np.any(blocked[rr0:rr1, cc0:cc1]))


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> None:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0], r0 + h + padding)
    cc1 = min(blocked.shape[1], c0 + w + padding)
    blocked[rr0:rr1, cc0:cc1] = True


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _balanced_orientations(rng: np.random.Generator, n_pairs: int) -> list[bool]:
    n_vertical = n_pairs // 2
    if n_pairs % 2:
        n_vertical += int(rng.integers(2))
    orientations = [True] * n_vertical + [False] * (n_pairs - n_vertical)
    rng.shuffle(orientations)
    return orientations


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    bg = int(rng.integers(len(COLOR_POOL)))
    non_bg = [color for color in COLOR_POOL if color != bg]
    n_pairs = int(rng.integers(*PAIR_COUNT_RANGE))
    if 2 * n_pairs > len(non_bg):
        return None

    frame_colors = [int(color) for color in rng.choice(non_bg, size=2 * n_pairs, replace=False)]
    orientations = _balanced_orientations(rng, n_pairs)

    patches: list[tuple[np.ndarray, tuple[int, int], int]] = []
    max_h = 0
    max_w = 0
    for i in range(n_pairs):
        source_border = frame_colors[2 * i]
        receiver_border = frame_colors[2 * i + 1]
        signal_choices = [color for color in non_bg if color not in {source_border, receiver_border}]
        signal = int(rng.choice(signal_choices))
        cable_len = int(rng.integers(*CABLE_LENGTH_RANGE))
        vertical = orientations[i]
        source_first = bool(rng.integers(2))
        patch_info = _build_pair_patch(
            bg=bg,
            source_border=source_border,
            receiver_border=receiver_border,
            signal=signal,
            vertical=vertical,
            source_first=source_first,
            cable_len=cable_len,
        )
        patches.append(patch_info)
        max_h = max(max_h, patch_info[0].shape[0])
        max_w = max(max_w, patch_info[0].shape[1])

    lower = max(GRID_MIN_SIZE, 12 + 2 * (n_pairs - 1), max_h + 2, max_w + 2)
    if lower > GRID_MAX_SIZE:
        return None
    rows = int(rng.integers(lower, GRID_MAX_SIZE + 1))
    cols = int(rng.integers(lower, GRID_MAX_SIZE + 1))

    inp = np.full((rows, cols), bg, dtype=int)
    expected = inp.copy()
    blocked = np.zeros((rows, cols), dtype=bool)

    order = list(range(n_pairs))
    rng.shuffle(order)
    for idx in order:
        patch, receiver_center, signal = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(rows - ph + 1):
            for c0 in range(cols - pw + 1):
                if _reserve_ok(blocked, r0, c0, ph, pw, PLACEMENT_PADDING):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        _reserve(blocked, r0, c0, ph, pw, PLACEMENT_PADDING)
        _paste(inp, patch, r0, c0, bg)
        _paste(expected, patch, r0, c0, bg)
        rr = r0 + receiver_center[0]
        cc = c0 + receiver_center[1]
        expected[rr, cc] = signal

    if _get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _build_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
