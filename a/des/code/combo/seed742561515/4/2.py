from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400
PLACEMENT_PAD = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _is_frame(points: list[tuple[int, int]], grid: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return False
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return bool(np.all(interior == BG) and np.all(grid[r0 : r1 + 1, c0 : c1 + 1][border] == color))


def _source_direction(
    fragment_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]
) -> str:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = fragment_bbox
    dr2 = (sr0 + sr1) - (fr0 + fr1)
    dc2 = (sc0 + sc1) - (fc0 + fc1)
    if abs(dc2) >= abs(dr2):
        return "left" if dc2 < 0 else "right"
    return "up" if dr2 < 0 else "down"


def _docked_origin(
    frame_bbox: tuple[int, int, int, int], frag_shape: tuple[int, int], source_direction: str
) -> tuple[int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_r0, inner_c0 = fr0 + 1, fc0 + 1
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    frag_h, frag_w = frag_shape
    assert frag_h <= inner_h and frag_w <= inner_w
    if source_direction == "left":
        row = inner_r0 + (inner_h - frag_h) // 2
        col = inner_c0 + inner_w - frag_w
    elif source_direction == "right":
        row = inner_r0 + (inner_h - frag_h) // 2
        col = inner_c0
    elif source_direction == "up":
        row = inner_r0 + inner_h - frag_h
        col = inner_c0 + (inner_w - frag_w) // 2
    else:
        row = inner_r0
        col = inner_c0 + (inner_w - frag_w) // 2
    return int(row), int(col)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each fragment with the same-color hollow frame and move it into the frame interior.

    The fragment keeps its shape. Its relative source direction determines the opposite wall it
    docks against, and the orthogonal axis is centered with a bias toward smaller indices.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors

    pair_data: dict[int, tuple[tuple[int, int, int, int], np.ndarray, list[tuple[int, int]]]] = {}
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, f"color {color} must have exactly one frame and one fragment"
        frame_bbox = None
        fragment_points = None
        for pts in comps:
            if _is_frame(pts, g, color):
                assert frame_bbox is None, f"color {color} has multiple frame candidates"
                frame_bbox = _bbox(pts)
            else:
                assert fragment_points is None, f"color {color} has multiple fragment candidates"
                fragment_points = pts
        assert frame_bbox is not None and fragment_points is not None
        frag_mask, frag_bbox = _tight_mask(fragment_points)
        inner_h = frame_bbox[2] - frame_bbox[0] - 1
        inner_w = frame_bbox[3] - frame_bbox[1] - 1
        assert frag_mask.shape[0] <= inner_h and frag_mask.shape[1] <= inner_w
        pair_data[color] = (frame_bbox, frag_mask, fragment_points)

    out = g.copy()
    for _color, (_frame_bbox, _frag_mask, fragment_points) in pair_data.items():
        for r, c in fragment_points:
            out[r, c] = BG

    for color, (frame_bbox, frag_mask, fragment_points) in pair_data.items():
        frag_bbox = _bbox(fragment_points)
        direction = _source_direction(frag_bbox, frame_bbox)
        top, left = _docked_origin(frame_bbox, frag_mask.shape, direction)
        _paste_mask(out, frag_mask, top, left, color)

    return out


def _random_shape() -> np.ndarray:
    for _ in range(100):
        area = random.randint(2, 5)
        while True:
            canvas_h = random.randint(1, 4)
            canvas_w = random.randint(1, 4)
            if canvas_h * canvas_w >= area:
                break
        seed = (random.randrange(canvas_h), random.randrange(canvas_w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        return mask
    raise RuntimeError("failed to sample a fragment shape")


def _rect_clear(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PLACEMENT_PAD) -> bool:
    H, W = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _occupy(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    blocked[r0 : r0 + h, c0 : c0 + w] = True


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _direction_matches(
    fragment_bbox: tuple[int, int, int, int],
    frame_bbox: tuple[int, int, int, int],
    direction: str,
    tie_case: bool,
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = fragment_bbox
    dr2 = (sr0 + sr1) - (fr0 + fr1)
    dc2 = (sc0 + sc1) - (fc0 + fc1)

    if direction == "left":
        if dc2 >= 0:
            return False
        return abs(dc2) == abs(dr2) if tie_case else abs(dc2) >= abs(dr2)
    if direction == "right":
        if dc2 <= 0:
            return False
        return abs(dc2) == abs(dr2) if tie_case else abs(dc2) >= abs(dr2)
    if direction == "up":
        return dr2 < 0 and abs(dr2) > abs(dc2)
    return dr2 > 0 and abs(dr2) > abs(dc2)


def _fragment_candidates(
    blocked: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    frag_shape: tuple[int, int],
    direction: str,
    tie_case: bool,
) -> list[tuple[int, int]]:
    H, W = blocked.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    frag_h, frag_w = frag_shape
    candidates: list[tuple[int, int]] = []

    if direction == "left":
        row_bias = frag_h - 1 - (fr0 + fr1)
        for c0 in range(0, fc0 - frag_w):
            if c0 + frag_w >= fc0:
                continue
            dc2 = (2 * c0 + frag_w - 1) - (fc0 + fc1)
            D = -dc2
            if tie_case:
                rows = []
                for target in (-D, D):
                    numerator = target - row_bias
                    if numerator % 2 == 0:
                        rows.append(numerator // 2)
                for r0 in sorted(set(rows)):
                    if 0 <= r0 <= H - frag_h and _rect_clear(blocked, r0, c0, frag_h, frag_w):
                        candidates.append((r0, c0))
                continue
            lo = max(0, _ceil_div(-D - row_bias, 2))
            hi = min(H - frag_h, (D - row_bias) // 2)
            for r0 in range(lo, hi + 1):
                if _rect_clear(blocked, r0, c0, frag_h, frag_w):
                    candidates.append((r0, c0))
    elif direction == "right":
        row_bias = frag_h - 1 - (fr0 + fr1)
        for c0 in range(fc1 + 2, W - frag_w + 1):
            dc2 = (2 * c0 + frag_w - 1) - (fc0 + fc1)
            D = dc2
            if tie_case:
                rows = []
                for target in (-D, D):
                    numerator = target - row_bias
                    if numerator % 2 == 0:
                        rows.append(numerator // 2)
                for r0 in sorted(set(rows)):
                    if 0 <= r0 <= H - frag_h and _rect_clear(blocked, r0, c0, frag_h, frag_w):
                        candidates.append((r0, c0))
                continue
            lo = max(0, _ceil_div(-D - row_bias, 2))
            hi = min(H - frag_h, (D - row_bias) // 2)
            for r0 in range(lo, hi + 1):
                if _rect_clear(blocked, r0, c0, frag_h, frag_w):
                    candidates.append((r0, c0))
    elif direction == "up":
        col_bias = frag_w - 1 - (fc0 + fc1)
        for r0 in range(0, fr0 - frag_h):
            if r0 + frag_h >= fr0:
                continue
            dr2 = (2 * r0 + frag_h - 1) - (fr0 + fr1)
            D = -dr2
            lo = max(0, _ceil_div(-(D - 1) - col_bias, 2))
            hi = min(W - frag_w, ((D - 1) - col_bias) // 2)
            for c0 in range(lo, hi + 1):
                if _rect_clear(blocked, r0, c0, frag_h, frag_w):
                    candidates.append((r0, c0))
    else:
        col_bias = frag_w - 1 - (fc0 + fc1)
        for r0 in range(fr1 + 2, H - frag_h + 1):
            dr2 = (2 * r0 + frag_h - 1) - (fr0 + fr1)
            D = dr2
            lo = max(0, _ceil_div(-(D - 1) - col_bias, 2))
            hi = min(W - frag_w, ((D - 1) - col_bias) // 2)
            for c0 in range(lo, hi + 1):
                if _rect_clear(blocked, r0, c0, frag_h, frag_w):
                    candidates.append((r0, c0))
    return candidates


def _sample_pair_specs() -> tuple[int, int, list[dict[str, object]]]:
    n_pairs = random.randint(2, 4)
    height = random.randint(16 + 3 * n_pairs, 20 + 3 * n_pairs)
    width = random.randint(16 + 3 * n_pairs, 22 + 3 * n_pairs)
    colors = random.sample(COLOR_POOL, n_pairs)
    directions = random.sample(list(DIRECTIONS), n_pairs)
    specs: list[dict[str, object]] = []
    for color, direction in zip(colors, directions, strict=False):
        frag = _random_shape()
        frag_h, frag_w = frag.shape
        inner_h = random.randint(max(3, frag_h + 1), min(8, frag_h + 4))
        inner_w = random.randint(max(3, frag_w + 1), min(8, frag_w + 4))
        tie_case = bool(direction in ("left", "right") and random.random() < 0.2)
        specs.append(
            {
                "color": int(color),
                "direction": str(direction),
                "tie_case": tie_case,
                "frag": frag,
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
            }
        )
    specs.sort(key=lambda spec: int(spec["outer_h"]) * int(spec["outer_w"]), reverse=True)
    return height, width, specs


def _gen_attempt() -> np.ndarray | None:
    H, W, specs = _sample_pair_specs()
    inp = np.full((H, W), BG, dtype=int)
    expected = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    for spec in specs:
        color = int(spec["color"])
        direction = str(spec["direction"])
        tie_case = bool(spec["tie_case"])
        frag = np.asarray(spec["frag"], dtype=bool)
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        frag_h, frag_w = frag.shape

        chosen_frame = None
        frag_positions = None
        for _ in range(240):
            fr0 = random.randint(0, H - outer_h)
            fc0 = random.randint(0, W - outer_w)
            if not _rect_clear(blocked, fr0, fc0, outer_h, outer_w):
                continue
            frame_bbox = (fr0, fc0, fr0 + outer_h - 1, fc0 + outer_w - 1)
            positions = _fragment_candidates(blocked, frame_bbox, (frag_h, frag_w), direction, tie_case)
            if not positions:
                continue
            chosen_frame = frame_bbox
            frag_positions = positions
            break
        if chosen_frame is None or frag_positions is None:
            return None

        frame_bbox = chosen_frame
        frag_top, frag_left = random.choice(frag_positions)

        _draw_frame(inp, frame_bbox, color)
        _draw_frame(expected, frame_bbox, color)
        _paste_mask(inp, frag, frag_top, frag_left, color)

        dock_top, dock_left = _docked_origin(frame_bbox, frag.shape, direction)
        _paste_mask(expected, frag, dock_top, dock_left, color)

        _occupy(blocked, frame_bbox[0], frame_bbox[1], outer_h, outer_w)
        _occupy(blocked, frag_top, frag_left, frag_h, frag_w)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
