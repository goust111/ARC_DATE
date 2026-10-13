import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_H_RANGE = (17, 26)
GRID_W_RANGE = (17, 26)
FRAME_THICK_RANGE = (3, 5)
FRAME_SPAN_CHOICES = (3, 5, 7, 9)
TOWER_JITTER = 3
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: Iterable[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(grid: np.ndarray, bg: int) -> List[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRECTIONS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _is_solid_3x3(grid: np.ndarray, comp: dict) -> bool:
    if comp["area"] != 9:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == comp["color"]))


def _is_hollow_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if comp["area"] != 2 * (h + w) - 4:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == comp["color"]):
        return False
    return bool(np.all(sub[~border] == bg))


def _frame_direction(frame_bbox: Tuple[int, int, int, int], tower_bbox: Tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_bbox
    tr0, tc0, tr1, tc1 = tower_bbox
    labels = []
    if fr1 < tr0:
        labels.append("up")
    if fr0 > tr1:
        labels.append("down")
    if fc1 < tc0:
        labels.append("left")
    if fc0 > tc1:
        labels.append("right")
    assert len(labels) == 1, "each receiver frame must lie in exactly one tower direction"
    return labels[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Signal leaves the 3x3 tower along the four orthogonal channels.
    A filter pixel recolors the signal without blocking it.
    The interior of the receiver frame in that direction is filled with the final signal color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    components = _components_by_color(g, bg)
    tower_candidates = [comp for comp in components if _is_solid_3x3(g, comp)]
    assert len(tower_candidates) == 1, "expected exactly one 3x3 signal tower"
    tower = tower_candidates[0]
    tower_bbox = tower["bbox"]
    tower_color = int(tower["color"])

    frames = [comp for comp in components if comp is not tower and _is_hollow_frame(g, comp, bg)]
    assert len(frames) == 4, "expected four hollow receiver frames"

    frames_by_dir: Dict[str, dict] = {}
    for frame in frames:
        direction = _frame_direction(frame["bbox"], tower_bbox)
        assert direction not in frames_by_dir, "receiver direction must be unique"
        frames_by_dir[direction] = frame
    assert set(frames_by_dir) == set(DIRECTIONS), "missing receiver direction"

    tr0, tc0, _, _ = tower_bbox
    center_r = tr0 + 1
    center_c = tc0 + 1
    ray_starts = {
        "up": (center_r - 2, center_c),
        "down": (center_r + 2, center_c),
        "left": (center_r, center_c - 2),
        "right": (center_r, center_c + 2),
    }

    out = g.copy()
    for direction, (dr, dc) in DIRECTIONS.items():
        frame = frames_by_dir[direction]
        fr0, fc0, fr1, fc1 = frame["bbox"]
        signal_color = tower_color

        r, c = ray_starts[direction]
        while 0 <= r < h and 0 <= c < w:
            if fr0 <= r <= fr1 and fc0 <= c <= fc1:
                break
            value = int(g[r, c])
            if value != bg:
                signal_color = value
            r += dr
            c += dc

        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = signal_color

    return out


def _draw_frame(grid: np.ndarray, bbox: Tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _valid_center_values(lo: int, hi: int, nominal: int) -> List[int]:
    tight = [v for v in range(max(lo, nominal - TOWER_JITTER), min(hi, nominal + TOWER_JITTER) + 1)]
    if tight:
        return tight
    return list(range(lo, hi + 1))


def _choose_span(limit: int) -> int:
    options = [span for span in FRAME_SPAN_CHOICES if span <= limit]
    if not options:
        raise ValueError("no valid odd receiver span")
    return random.choice(options)


def _channel_cells(
    direction: str,
    h: int,
    w: int,
    center_r: int,
    center_c: int,
    top_h: int,
    bottom_h: int,
    left_w: int,
    right_w: int,
) -> List[Tuple[int, int]]:
    if direction == "up":
        return [(r, center_c) for r in range(top_h, center_r - 1)]
    if direction == "down":
        return [(r, center_c) for r in range(center_r + 2, h - bottom_h)]
    if direction == "left":
        return [(center_r, c) for c in range(left_w, center_c - 1)]
    return [(center_r, c) for c in range(center_c + 2, w - right_w)]


def _expected_output(
    inp: np.ndarray,
    tower_color: int,
    frame_bboxes: Dict[str, Tuple[int, int, int, int]],
    filter_colors: Dict[str, int],
) -> np.ndarray:
    out = inp.copy()
    for direction, bbox in frame_bboxes.items():
        r0, c0, r1, c1 = bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = filter_colors.get(direction, tower_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    top_h = random.randint(*FRAME_THICK_RANGE)
    bottom_h = random.randint(*FRAME_THICK_RANGE)
    left_w = random.randint(*FRAME_THICK_RANGE)
    right_w = random.randint(*FRAME_THICK_RANGE)

    center_r_lo = top_h + 3
    center_r_hi = h - bottom_h - 4
    center_c_lo = left_w + 3
    center_c_hi = w - right_w - 4
    if center_r_lo > center_r_hi or center_c_lo > center_c_hi:
        return None

    center_r = random.choice(_valid_center_values(center_r_lo, center_r_hi, h // 2))
    center_c = random.choice(_valid_center_values(center_c_lo, center_c_hi, w // 2))

    try:
        top_span = _choose_span(2 * min(center_c, w - 1 - center_c) + 1)
        bottom_span = _choose_span(2 * min(center_c, w - 1 - center_c) + 1)
        left_span = _choose_span(2 * min(center_r, h - 1 - center_r) + 1)
        right_span = _choose_span(2 * min(center_r, h - 1 - center_r) + 1)
    except ValueError:
        return None

    tower_r0 = center_r - 1
    tower_c0 = center_c - 1

    frame_bboxes = {
        "up": (0, center_c - top_span // 2, top_h - 1, center_c + top_span // 2),
        "down": (h - bottom_h, center_c - bottom_span // 2, h - 1, center_c + bottom_span // 2),
        "left": (center_r - left_span // 2, 0, center_r + left_span // 2, left_w - 1),
        "right": (center_r - right_span // 2, w - right_w, center_r + right_span // 2, w - 1),
    }

    occupied = np.zeros((h, w), dtype=bool)
    tower_bbox = (tower_r0, tower_c0, tower_r0 + 2, tower_c0 + 2)
    occupied[tower_r0 : tower_r0 + 3, tower_c0 : tower_c0 + 3] = True

    for bbox in frame_bboxes.values():
        r0, c0, r1, c1 = bbox
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    colors = list(range(10))
    random.shuffle(colors)
    bg, tower_color, frame_color = colors[:3]
    filter_palette = colors[3:]

    grid = np.full((h, w), bg, dtype=int)
    grid[tower_r0 : tower_r0 + 3, tower_c0 : tower_c0 + 3] = tower_color
    for bbox in frame_bboxes.values():
        _draw_frame(grid, bbox, frame_color)

    filtered_dirs = set(random.sample(list(DIRECTIONS), 2))
    filter_colors: Dict[str, int] = {}
    for direction in DIRECTIONS:
        cells = _channel_cells(direction, h, w, center_r, center_c, top_h, bottom_h, left_w, right_w)
        if len(cells) < 2:
            return None
        if direction in filtered_dirs:
            if not filter_palette:
                return None
            color = filter_palette.pop()
            filter_colors[direction] = color
            fr, fc = random.choice(cells)
            grid[fr, fc] = color

    expected = _expected_output(grid, tower_color, frame_bboxes, filter_colors)

    if _most_common_color(grid) != bg:
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
