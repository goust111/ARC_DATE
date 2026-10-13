import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PARITY_KEYS = ((1, 1), (0, 0), (1, 0), (0, 1))
SIZE_BY_PARITY = {
    1: (5, 7),
    0: (6, 8),
}
PAIR_COUNT_OPTIONS = (2, 3)
FRAME_GAP = 1
MARGIN_CHOICES = (1, 2)
MAX_ATTEMPTS = 400


def _monochrome_components(grid: np.ndarray) -> List[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: List[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return components


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_hollow_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border_count = 2 * (h + w) - 4
    if int(comp["area"]) != border_count:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(h, w)
    color = int(comp["color"])
    return bool(np.all(patch[border] == color) and np.all(patch[~border] != color))


def _extract_frames(grid: np.ndarray) -> List[dict]:
    frames: List[dict] = []
    for comp in _monochrome_components(grid):
        if not _is_hollow_frame(grid, comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        pattern: List[Tuple[int, int, int]] = []
        rows, cols = np.nonzero(inner != BG)
        for rr, cc in zip(rows, cols):
            color = int(inner[rr, cc])
            if color != int(comp["color"]):
                pattern.append((int(rr), int(cc), color))
        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "height": h,
                "width": w,
                "inner_shape": (h - 2, w - 2),
                "pattern": pattern,
            }
        )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each frame's interior dot pattern with the frame that shares its width-height parity signature.

    Input structure:
    - Black background containing several non-overlapping monochrome hollow rectangular frames.
    - Every frame has a distinct border color and encloses a dot pattern made of non-background, non-border colors.
    - The number of frames is even, and exactly two frames share each used combination of width parity and height parity.

    Rule:
    - Detect every hollow rectangular frame and compute its full width and full height.
    - Group frames by the pair (width parity, height parity): odd-odd, even-even, odd-even, or even-odd.
    - Within each parity-matched pair, clear both interiors and paste each frame's interior pattern into the other frame, preserving every pixel's offset from the inner top-left corner.

    Key insight:
    - The destination of each pattern is hidden in the geometry of the colored frame: matching width/height parity, not position or color, determines the exchange partner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    frames = _extract_frames(g)
    assert len(frames) >= 4 and len(frames) % 2 == 0

    groups: Dict[Tuple[int, int], List[dict]] = {}
    for frame in frames:
        key = (int(frame["width"]) % 2, int(frame["height"]) % 2)
        groups.setdefault(key, []).append(frame)

    assert len(groups) * 2 == len(frames)
    assert all(len(group) == 2 for group in groups.values())

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

    for pair in groups.values():
        a, b = pair
        for dst, src in ((a, b), (b, a)):
            r0, c0, _r1, _c1 = dst["bbox"]
            inner_h, inner_w = dst["inner_shape"]
            for dr, dc, color in src["pattern"]:
                assert 0 <= dr < inner_h and 0 <= dc < inner_w
                out[r0 + 1 + dr, c0 + 1 + dc] = int(color)

    return out


def _sample_distinct_colors(n_frames: int) -> Tuple[List[int], List[int]]:
    pool = list(range(1, 10))
    random.shuffle(pool)
    if 2 * n_frames <= len(pool):
        return pool[:n_frames], pool[n_frames : 2 * n_frames]

    frame_colors = random.sample(pool, n_frames)
    pattern_colors = frame_colors[:]
    while True:
        random.shuffle(pattern_colors)
        if all(a != b for a, b in zip(frame_colors, pattern_colors)):
            return frame_colors, pattern_colors


def _sample_pair_dims(key: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    width_parity, height_parity = key
    widths = SIZE_BY_PARITY[width_parity]
    heights = SIZE_BY_PARITY[height_parity]
    options = [(h, w) for h in heights for w in widths]
    first = random.choice(options)
    second = random.choice([option for option in options if option != first])
    return first, second


def _neighbors_in_box(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _sample_pattern(inner_h: int, inner_w: int) -> List[Tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    area = inner_h * inner_w
    target = random.randint(2, min(6, area - 1))
    mode = random.choice(("cluster", "cluster", "scatter"))

    if mode == "scatter":
        picked = set(random.sample(cells, target))
    else:
        picked = {random.choice(cells)}
        while len(picked) < target:
            frontier = []
            for r, c in picked:
                for nbr in _neighbors_in_box(r, c, inner_h, inner_w):
                    if nbr not in picked:
                        frontier.append(nbr)
            if frontier:
                picked.add(random.choice(frontier))
            else:
                remaining = [cell for cell in cells if cell not in picked]
                picked.add(random.choice(remaining))
        if random.random() < 0.35 and len(picked) < target:
            remaining = [cell for cell in cells if cell not in picked]
            picked.add(random.choice(remaining))

    return sorted((int(r), int(c)) for r, c in picked)


def _build_expected(grid: np.ndarray, specs: Sequence[dict]) -> np.ndarray:
    out = grid.copy()
    by_pair: Dict[int, List[dict]] = {}
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        out[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = BG
        by_pair.setdefault(int(spec["pair_id"]), []).append(spec)

    for pair in by_pair.values():
        assert len(pair) == 2
        a, b = pair
        for dst, src in ((a, b), (b, a)):
            for dr, dc in src["pattern"]:
                out[int(dst["r0"]) + 1 + dr, int(dst["c0"]) + 1 + dc] = int(src["pattern_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    used_keys = random.sample(list(PARITY_KEYS), pair_count)
    frame_count = pair_count * 2
    frame_colors, pattern_colors = _sample_distinct_colors(frame_count)

    specs: List[dict] = []
    color_index = 0
    for pair_id, key in enumerate(used_keys):
        (h1, w1), (h2, w2) = _sample_pair_dims(key)
        safe_h = min(h1, h2) - 2
        safe_w = min(w1, w2) - 2
        pattern_a = _sample_pattern(safe_h, safe_w)
        pattern_b = _sample_pattern(safe_h, safe_w)
        retries = 0
        while pattern_b == pattern_a and retries < 20:
            pattern_b = _sample_pattern(safe_h, safe_w)
            retries += 1
        if pattern_a == pattern_b:
            return None

        specs.append(
            {
                "pair_id": pair_id,
                "parity_key": key,
                "h": h1,
                "w": w1,
                "frame_color": frame_colors[color_index],
                "pattern_color": pattern_colors[color_index],
                "pattern": pattern_a,
            }
        )
        color_index += 1
        specs.append(
            {
                "pair_id": pair_id,
                "parity_key": key,
                "h": h2,
                "w": w2,
                "frame_color": frame_colors[color_index],
                "pattern_color": pattern_colors[color_index],
                "pattern": pattern_b,
            }
        )
        color_index += 1

    rows = pair_count
    cols = 2
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(slots)
    random.shuffle(specs)

    for spec, slot in zip(specs, slots):
        spec["slot"] = slot

    row_heights = [0] * rows
    col_widths = [0] * cols
    for spec in specs:
        slot_r, slot_c = spec["slot"]
        row_heights[slot_r] = max(row_heights[slot_r], int(spec["h"]))
        col_widths[slot_c] = max(col_widths[slot_c], int(spec["w"]))

    top = random.choice(MARGIN_CHOICES)
    bottom = random.choice(MARGIN_CHOICES)
    left = random.choice(MARGIN_CHOICES)
    right = random.choice(MARGIN_CHOICES)

    height = top + bottom + sum(row_heights) + FRAME_GAP * (rows - 1)
    width = left + right + sum(col_widths) + FRAME_GAP * (cols - 1)
    if height > 30 or width > 30:
        return None

    row_starts = []
    cursor = top
    for row_h in row_heights:
        row_starts.append(cursor)
        cursor += row_h + FRAME_GAP

    col_starts = []
    cursor = left
    for col_w in col_widths:
        col_starts.append(cursor)
        cursor += col_w + FRAME_GAP

    for spec in specs:
        slot_r, slot_c = spec["slot"]
        extra_r = row_heights[slot_r] - int(spec["h"])
        extra_c = col_widths[slot_c] - int(spec["w"])
        spec["r0"] = row_starts[slot_r] + random.randint(0, extra_r)
        spec["c0"] = col_starts[slot_c] + random.randint(0, extra_c)

    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        frame_color = int(spec["frame_color"])
        pattern_color = int(spec["pattern_color"])

        grid[r0, c0 : c0 + w] = frame_color
        grid[r0 + h - 1, c0 : c0 + w] = frame_color
        grid[r0 : r0 + h, c0] = frame_color
        grid[r0 : r0 + h, c0 + w - 1] = frame_color
        for dr, dc in spec["pattern"]:
            grid[r0 + 1 + dr, c0 + 1 + dc] = pattern_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = _build_expected(grid, specs)
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-frame swap input")
