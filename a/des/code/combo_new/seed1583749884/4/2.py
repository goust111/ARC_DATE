from __future__ import annotations

import random
from collections import deque

import numpy as np


AZURE = 8
GREY = 5
YELLOW = 4
MAROON = 9

PAIR_COLORS = (0, 1, 2, 3, 6, 7)
GRID_MIN = 12
GRID_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (2, 5)
GAP_RANGE = (1, 6)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 80


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "coords": tuple(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _is_filled_rectangle(grid: np.ndarray, comp: dict[str, object], color: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if int(comp["area"]) != height * width:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _gap_bbox(a: dict[str, object], b: dict[str, object]) -> tuple[int, int, int, int] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ar0 == br0 and ar1 == br1:
        left, right = (a, b) if ac0 < bc0 else (b, a)
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        if lc1 + 1 <= rc0 - 1:
            return lr0, lc1 + 1, lr1, rc0 - 1

    if ac0 == bc0 and ac1 == bc1:
        top, bottom = (a, b) if ar0 < br0 else (b, a)
        tr0, tc0, tr1, tc1 = top["bbox"]
        br0, bc0, br1, bc1 = bottom["bbox"]
        if tr1 + 1 <= br0 - 1:
            return tr1 + 1, tc0, br0 - 1, tc1

    return None


def _pair_gap_infos(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    infos: list[tuple[int, int, int, int]] = []
    colors = [int(color) for color in np.unique(grid) if int(color) not in {AZURE, GREY}]
    for color in colors:
        comps = _components_of_color(grid, color)
        if len(comps) != 2:
            continue
        if not all(_is_filled_rectangle(grid, comp, color) for comp in comps):
            continue
        gap = _gap_bbox(comps[0], comps[1])
        if gap is None:
            continue
        r0, c0, r1, c1 = gap
        gap_view = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(np.isin(gap_view, (AZURE, GREY))):
            continue
        infos.append(gap)
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the gap between each aligned same-colored rectangle pair, using the bridge presence to choose the fill color.

    Input structure:
    - The background is Azure.
    - Several non-grey colors each form exactly two separate filled rectangles.
    - In each color pair, the two rectangles share the same row span or the same column span.
    - Some pair gaps contain a tiny Grey bridge object of one or two cells, while other gaps are pure background.

    Rule:
    - For every same-colored rectangle pair aligned in one row band or one column band, identify the rectangular gap between them.
    - If that gap contains any Grey cells, fill the entire gap with Yellow.
    - Otherwise, fill the entire gap with Maroon.
    - Leave the original rectangles and every unrelated cell unchanged.

    Key insight:
    - The task is controlled by a boolean property of each pair gap: the Grey bridge does not stay as an object, it only switches the gap fill from Maroon to Yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    gaps = _pair_gap_infos(g)
    assert gaps, "expected at least one aligned rectangle pair"
    for r0, c0, r1, c1 in gaps:
        gap_view = g[r0 : r1 + 1, c0 : c1 + 1]
        fill = YELLOW if np.any(gap_view == GREY) else MAROON
        out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _bridge_flags(pair_count: int) -> list[bool]:
    bridge_count = random.randint(1, pair_count - 1)
    flags = [True] * bridge_count + [False] * (pair_count - bridge_count)
    random.shuffle(flags)
    return flags


def _bridge_shape(height: int, width: int) -> tuple[int, int]:
    shapes = [(1, 1)]
    if width >= 2:
        shapes.append((1, 2))
    if height >= 2:
        shapes.append((2, 1))
    return random.choice(shapes)


def _make_pair_patch(color: int, bridged: bool) -> tuple[np.ndarray, np.ndarray]:
    orientation = random.choice(("h", "v"))
    fill = YELLOW if bridged else MAROON

    if orientation == "h":
        height = random.randint(*RECT_SIDE_RANGE)
        left_w = random.randint(*RECT_SIDE_RANGE)
        right_w = random.randint(*RECT_SIDE_RANGE)
        gap_w = random.randint(*GAP_RANGE)

        inp = np.full((height, left_w + gap_w + right_w), AZURE, dtype=int)
        inp[:, :left_w] = color
        inp[:, left_w + gap_w :] = color

        out = inp.copy()
        out[:, left_w : left_w + gap_w] = fill

        if bridged:
            bridge_h, bridge_w = _bridge_shape(height, gap_w)
            br = random.randint(0, height - bridge_h)
            bc = random.randint(left_w, left_w + gap_w - bridge_w)
            inp[br : br + bridge_h, bc : bc + bridge_w] = GREY
        return inp, out

    width = random.randint(*RECT_SIDE_RANGE)
    top_h = random.randint(*RECT_SIDE_RANGE)
    bottom_h = random.randint(*RECT_SIDE_RANGE)
    gap_h = random.randint(*GAP_RANGE)

    inp = np.full((top_h + gap_h + bottom_h, width), AZURE, dtype=int)
    inp[:top_h, :] = color
    inp[top_h + gap_h :, :] = color

    out = inp.copy()
    out[top_h : top_h + gap_h, :] = fill

    if bridged:
        bridge_h, bridge_w = _bridge_shape(gap_h, width)
        br = random.randint(top_h, top_h + gap_h - bridge_h)
        bc = random.randint(0, width - bridge_w)
        inp[br : br + bridge_h, bc : bc + bridge_w] = GREY
    return inp, out


def _place_patches(
    patches: list[tuple[np.ndarray, np.ndarray]], height: int, width: int
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((height, width), AZURE, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((height, width), dtype=bool)

    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1],
        reverse=True,
    )

    for idx in order:
        inp_patch, out_patch = patches[idx]
        ph, pw = inp_patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(height - ph + 1):
            rr0 = max(0, r - 1)
            rr1 = min(height, r + ph + 1)
            for c in range(width - pw + 1):
                cc0 = max(0, c - 1)
                cc1 = min(width, c + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None

        r, c = random.choice(candidates)
        grid[r : r + ph, c : c + pw] = inp_patch
        expected[r : r + ph, c : c + pw] = out_patch
        occupied[max(0, r - 1) : min(height, r + ph + 1), max(0, c - 1) : min(width, c + pw + 1)] = True

    return grid, expected


def _sample_canvas(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, int] | None:
    min_h = max(inp.shape[0] + 2 for inp, _ in patches)
    min_w = max(inp.shape[1] + 2 for inp, _ in patches)
    area_need = sum((inp.shape[0] + 2) * (inp.shape[1] + 2) for inp, _ in patches)

    for _ in range(PLACEMENT_ATTEMPTS):
        height = random.randint(max(GRID_MIN, min_h), GRID_MAX)
        width_low = max(GRID_MIN, min_w, (area_need + height - 1) // height)
        if width_low > GRID_MAX:
            continue
        width = random.randint(width_low, GRID_MAX)
        return height, width
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PAIR_COLORS, pair_count)
    flags = _bridge_flags(pair_count)
    patches = [_make_pair_patch(color, bridged) for color, bridged in zip(colors, flags)]

    canvas = _sample_canvas(patches)
    if canvas is None:
        return None

    height, width = canvas
    for _ in range(PLACEMENT_ATTEMPTS):
        placed = _place_patches(patches, height, width)
        if placed is None:
            continue
        grid, expected = placed
        if max(grid.shape) > GRID_MAX:
            continue
        if not np.any(grid == GREY):
            continue
        if np.count_nonzero(grid == AZURE) == 0:
            continue
        actual = main(grid)
        if not (np.any(actual == YELLOW) and np.any(actual == MAROON)):
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
