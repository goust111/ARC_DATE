from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
DIRECTIONS = ("up", "down", "left", "right")
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Generator configuration.
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
FRAME_COUNT_CHOICES = (2, 3, 4)
INNER_SHAPES = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3)]
ARROW_AREA_CHOICES = list(range(6, 13))
PLACE_TRIES = 250
GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals).argmax())


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for coords in _connected_components(grid == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    h, w = r1 - r0 + 1, c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _classify_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    mask = _component_mask(comp)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None
    missing = np.argwhere(border & ~mask)
    if missing.shape[0] != 1:
        return None
    if int(mask.sum()) != int(border.sum()) - 1:
        return None

    gr, gc = map(int, missing[0])
    if gr == 0 and 0 < gc < w - 1:
        direction = "up"
    elif gr == h - 1 and 0 < gc < w - 1:
        direction = "down"
    elif gc == 0 and 0 < gr < h - 1:
        direction = "left"
    elif gc == w - 1 and 0 < gr < h - 1:
        direction = "right"
    else:
        return None

    return {
        **comp,
        "direction": direction,
        "cavity_area": int((h - 2) * (w - 2)),
    }


def _classify_arrow(comp: dict) -> dict | None:
    mask = _component_mask(comp)
    h, w = mask.shape
    area = int(mask.sum())

    if w == 3 and h >= 4 and bool(mask[:, 1].all()):
        left_rows = np.flatnonzero(mask[:, 0])
        right_rows = np.flatnonzero(mask[:, 2])
        if left_rows.size == 1 and right_rows.size == 1 and int(left_rows[0]) == int(right_rows[0]) and area == h + 2:
            wing_row = int(left_rows[0])
            if wing_row == 1:
                direction = "up"
            elif wing_row == h - 2:
                direction = "down"
            else:
                direction = None
            if direction is not None:
                return {**comp, "direction": direction}

    if h == 3 and w >= 4 and bool(mask[1, :].all()):
        top_cols = np.flatnonzero(mask[0, :])
        bottom_cols = np.flatnonzero(mask[2, :])
        if top_cols.size == 1 and bottom_cols.size == 1 and int(top_cols[0]) == int(bottom_cols[0]) and area == w + 2:
            wing_col = int(top_cols[0])
            if wing_col == 1:
                direction = "left"
            elif wing_col == w - 2:
                direction = "right"
            else:
                direction = None
            if direction is not None:
                return {**comp, "direction": direction}

    return None


def _match_by_area(frames: list[dict], arrows: list[dict]) -> list[tuple[dict, dict]]:
    frames_sorted = sorted(frames, key=lambda x: (int(x["cavity_area"]), x["bbox"]))
    arrows_sorted = sorted(arrows, key=lambda x: (int(x["area"]), x["bbox"]))
    m, n = len(frames_sorted), len(arrows_sorted)
    assert n >= m

    inf = 10**9
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    take = [[False] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        dp[0][j] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best = dp[i][j - 1]
            use_arrow = False
            cand = dp[i - 1][j - 1]
            if cand < inf:
                cand += abs(int(frames_sorted[i - 1]["cavity_area"]) - int(arrows_sorted[j - 1]["area"]))
                if cand < best:
                    best = cand
                    use_arrow = True
            dp[i][j] = best
            take[i][j] = use_arrow

    assert dp[m][n] < inf
    pairs: list[tuple[dict, dict]] = []
    i, j = m, n
    while i > 0:
        assert j > 0
        if take[i][j]:
            pairs.append((frames_sorted[i - 1], arrows_sorted[j - 1]))
            i -= 1
            j -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey frame cavity with the color of the outside arrow whose direction
    matches the frame gap and whose area best matches the cavity area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    frames: list[dict] = []
    arrows: list[dict] = []
    for comp in _foreground_components(g, bg):
        frame = _classify_frame(comp)
        if frame is not None:
            frames.append(frame)
            continue
        arrow = _classify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
            continue
        raise AssertionError("all foreground components must be frames or arrow sprites")

    assert frames
    assert arrows

    arrows_by_dir: dict[str, list[dict]] = defaultdict(list)
    for arrow in arrows:
        arrows_by_dir[str(arrow["direction"])].append(arrow)

    out = g.copy()
    for direction in DIRECTIONS:
        dir_frames = [frame for frame in frames if frame["direction"] == direction]
        if not dir_frames:
            continue
        dir_arrows = arrows_by_dir.get(direction, [])
        assert len(dir_arrows) >= len(dir_frames)
        for frame, arrow in _match_by_area(dir_frames, dir_arrows):
            coords = arrow["coords"]
            out[coords[:, 0], coords[:, 1]] = bg
            r0, c0, r1, c1 = frame["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = int(arrow["color"])

    return out


def _make_frame(inner_h: int, inner_w: int, direction: str) -> tuple[np.ndarray, int]:
    h, w = inner_h + 2, inner_w + 2
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    if direction == "up":
        gap_idx = random.randint(1, w - 2)
        patch[0, gap_idx] = BG
    elif direction == "down":
        gap_idx = random.randint(1, w - 2)
        patch[-1, gap_idx] = BG
    elif direction == "left":
        gap_idx = random.randint(1, h - 2)
        patch[gap_idx, 0] = BG
    elif direction == "right":
        gap_idx = random.randint(1, h - 2)
        patch[gap_idx, -1] = BG
    else:
        raise ValueError(direction)
    return patch, inner_h * inner_w


def _make_arrow(direction: str, area: int, color: int) -> np.ndarray:
    stem = area - 2
    assert stem >= 4
    if direction in ("up", "down"):
        patch = np.zeros((stem, 3), dtype=int)
        patch[:, 1] = color
        wing_row = 1 if direction == "up" else stem - 2
        patch[wing_row, 0] = color
        patch[wing_row, 2] = color
        return patch

    patch = np.zeros((3, stem), dtype=int)
    patch[1, :] = color
    wing_col = 1 if direction == "left" else stem - 2
    patch[0, wing_col] = color
    patch[2, wing_col] = color
    return patch


def _bbox_is_free(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = blocked.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - pad)
    r1 = min(hh, top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(ww, left + w + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_bbox(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    blocked[top : top + h, left : left + w] = True


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> np.ndarray:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    sub[mask] = patch[mask]
    coords = np.argwhere(mask)
    coords[:, 0] += top
    coords[:, 1] += left
    return coords


def _random_place(blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    hh, ww = blocked.shape
    if h > hh or w > ww:
        return None
    for _ in range(PLACE_TRIES):
        top = random.randint(0, hh - h)
        left = random.randint(0, ww - w)
        if _bbox_is_free(blocked, top, left, h, w, pad=1):
            return top, left
    return None


def _nearest_area(target: int) -> int:
    best_dist = min(abs(area - target) for area in ARROW_AREA_CHOICES)
    best = [area for area in ARROW_AREA_CHOICES if abs(area - target) == best_dist]
    return random.choice(best)


def _distractor_area(target: int, chosen: int) -> int:
    chosen_dist = abs(chosen - target)
    preferred = [area for area in ARROW_AREA_CHOICES if area != chosen and abs(area - target) >= chosen_dist + 2]
    if preferred:
        return random.choice(preferred)
    fallback = [area for area in ARROW_AREA_CHOICES if area != chosen and abs(area - target) > chosen_dist]
    assert fallback
    return random.choice(fallback)


def _manual_expected(grid: np.ndarray, frame_records: list[dict], chosen_by_dir: dict[str, dict]) -> np.ndarray:
    out = grid.copy()
    for frame in frame_records:
        arrow = chosen_by_dir[frame["direction"]]
        coords = arrow["placed_coords"]
        out[coords[:, 0], coords[:, 1]] = BG
        r0, c0, r1, c1 = frame["placed_bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(arrow["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    frame_dirs = random.sample(DIRECTIONS, k=random.choice(FRAME_COUNT_CHOICES))
    unused_dirs = [direction for direction in DIRECTIONS if direction not in frame_dirs]

    frame_specs: list[dict] = []
    chosen_by_dir: dict[str, dict] = {}
    arrow_specs: list[dict] = []

    for direction in frame_dirs:
        inner_h, inner_w = random.choice(INNER_SHAPES)
        frame_patch, cavity_area = _make_frame(inner_h, inner_w, direction)
        chosen_area = _nearest_area(cavity_area)
        distractor_area = _distractor_area(cavity_area, chosen_area)
        frame_specs.append(
            {
                "direction": direction,
                "patch": frame_patch,
                "cavity_area": cavity_area,
            }
        )
        chosen_by_dir[direction] = {
            "direction": direction,
            "area": chosen_area,
            "selected": True,
        }
        arrow_specs.append(chosen_by_dir[direction])
        arrow_specs.append(
            {
                "direction": direction,
                "area": distractor_area,
                "selected": False,
            }
        )

    for direction in unused_dirs:
        arrow_specs.append(
            {
                "direction": direction,
                "area": random.choice(ARROW_AREA_CHOICES),
                "selected": False,
            }
        )

    random.shuffle(frame_specs)
    random.shuffle(arrow_specs)
    colors = ARROW_COLORS[:]
    random.shuffle(colors)
    if len(arrow_specs) > len(colors):
        return None

    for spec, color in zip(arrow_specs, colors):
        spec["color"] = color
        spec["patch"] = _make_arrow(spec["direction"], int(spec["area"]), color)

    for frame in frame_specs:
        patch = frame["patch"]
        placed = _random_place(blocked, patch)
        if placed is None:
            return None
        top, left = placed
        _paint_patch(grid, patch, top, left)
        _mark_bbox(blocked, top, left, *patch.shape)
        h, w = patch.shape
        frame["placed_bbox"] = (top, left, top + h - 1, left + w - 1)

    for spec in arrow_specs:
        patch = spec["patch"]
        placed = _random_place(blocked, patch)
        if placed is None:
            return None
        top, left = placed
        spec["placed_coords"] = _paint_patch(grid, patch, top, left)
        _mark_bbox(blocked, top, left, *patch.shape)

    expected = _manual_expected(grid, frame_specs, chosen_by_dir)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
