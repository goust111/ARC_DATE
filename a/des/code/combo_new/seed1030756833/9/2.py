from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == BLACK or seen[sr, sc]:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = int(r) + dr
                    cc = int(c) + dc
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or int(grid[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))

            coords = np.array(pts, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )

    return out


def _is_hollow_rectangle(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    expected_area = 2 * height + 2 * width - 4
    if int(comp["area"]) != int(expected_area):
        return False

    coords = comp["coords"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    on_perimeter = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
    if not bool(np.all(on_perimeter)):
        return False

    if not bool(np.all(grid[r0, c0 : c1 + 1] == color)):
        return False
    if not bool(np.all(grid[r1, c0 : c1 + 1] == color)):
        return False
    if not bool(np.all(grid[r0 : r1 + 1, c0] == color)):
        return False
    if not bool(np.all(grid[r0 : r1 + 1, c1] == color)):
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK))


def _piece_sort_key(comp: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return (-int(comp["area"]), int(r0), int(c0), int(r1), int(c1))


def _paint_stack(out: np.ndarray, frame: dict, pieces: list[dict]) -> None:
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    inner_top = int(r0 + 1)
    inner_left = int(c0 + 1)
    inner_bottom = int(r1 - 1)
    inner_right = int(c1 - 1)
    inner_width = int(inner_right - inner_left + 1)
    cursor_bottom = inner_bottom

    for piece in sorted(pieces, key=_piece_sort_key):
        coords = piece["coords"]
        pr0, pc0, pr1, pc1 = piece["bbox"]
        piece_h = int(pr1 - pr0 + 1)
        piece_w = int(pc1 - pc0 + 1)
        assert piece_h <= int(inner_bottom - inner_top + 1)
        assert piece_w <= inner_width

        top = int(cursor_bottom - piece_h + 1)
        left = int(inner_left + (inner_width - piece_w) // 2)
        local = coords - np.array([pr0, pc0], dtype=int)
        placed = local + np.array([top, left], dtype=int)

        assert bool(
            np.all(
                (inner_top <= placed[:, 0])
                & (placed[:, 0] <= inner_bottom)
                & (inner_left <= placed[:, 1])
                & (placed[:, 1] <= inner_right)
            )
        )
        assert bool(np.all(out[placed[:, 0], placed[:, 1]] == BLACK))
        out[placed[:, 0], placed[:, 1]] = color
        cursor_bottom = int(top - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered colored fragments into their matching hollow containers and stack them by size.

    Input structure:
    - The background is black.
    - Each non-background container is a one-pixel-thick hollow rectangle, and every container has a distinct color.
    - Away from the containers are several 4-connected monochrome fragments whose colors match the container colors.

    Rule:
    - For each color, identify its hollow rectangular container and all same-color fragments outside it.
    - Clear the original fragment locations to black.
    - Sort the fragments of that color by pixel area from largest to smallest, preserving each fragment's shape.
    - Place the sorted fragments inside the matching container from bottom to top, horizontally centered with the left cell chosen when centering is uneven, and with adjacent fragment bounding boxes touching vertically.

    Key insight:
    - Color is both the fragment label and the container address; after matching by color, the only remaining order is visible fragment area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in _components(g):
        by_color[int(comp["color"])].append(comp)

    frames: dict[int, dict] = {}
    fragments: dict[int, list[dict]] = {}
    for color, comps in by_color.items():
        frame_comps = [comp for comp in comps if _is_hollow_rectangle(g, comp)]
        assert len(frame_comps) == 1
        frame = frame_comps[0]
        frames[int(color)] = frame
        frame_id = id(frame)
        pieces = [comp for comp in comps if id(comp) != frame_id]
        assert pieces
        fragments[int(color)] = pieces

    out = g.copy()
    for pieces in fragments.values():
        for piece in pieces:
            coords = piece["coords"]
            out[coords[:, 0], coords[:, 1]] = BLACK

    for color, frame in frames.items():
        _paint_stack(out, frame, fragments[int(color)])

    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
CONTAINER_COUNT_RANGE = (2, 3)
FRAGMENT_COUNT_RANGE = (2, 4)
FRAGMENT_AREA_VALUES = tuple(range(2, 8))
MAX_FRAGMENT_H = 4
MAX_FRAGMENT_W = 4
INNER_HEIGHT_MAX = 13
INNER_WIDTH_MAX = 10
TOP_SLACK_RANGE = (0, 3)
WIDTH_SLACK_VALUES = tuple(range(0, 6))
GRID_SIZE_RANGE = (22, 30)
OBJECT_MARGIN = 1
GENERATION_RETRIES = 500
POSITION_RETRIES = 40


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]


def _is_frame_like_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    return bool(np.array_equal(mask, perimeter))


def _random_fragment_mask(area: int) -> np.ndarray:
    for _ in range(300):
        cells = {(0, 0)}
        while len(cells) < int(area):
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(DIR4)
            cells.add((int(r) + dr, int(c) + dc))

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        if h > MAX_FRAGMENT_H or w > MAX_FRAGMENT_W:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r - min(rows)), int(c - min(cols))] = True
        mask = _trim_mask(mask)
        if _is_frame_like_mask(mask):
            continue
        return mask

    mask = np.zeros((1, int(area)), dtype=bool)
    mask[0, :] = True
    if mask.shape[1] <= MAX_FRAGMENT_W:
        return mask
    mask = np.ones((2, int((area + 1) // 2)), dtype=bool)
    if mask.size > area:
        mask[-1, -1] = False
    return _trim_mask(mask)


def _make_spec(color: int) -> dict | None:
    for _ in range(200):
        count = random.randint(*FRAGMENT_COUNT_RANGE)
        areas = random.sample(FRAGMENT_AREA_VALUES, count)
        masks = [_random_fragment_mask(area) for area in areas]
        if len({int(mask.sum()) for mask in masks}) != len(masks):
            continue

        heights = [int(mask.shape[0]) for mask in masks]
        widths = [int(mask.shape[1]) for mask in masks]
        inner_h = int(sum(heights) + random.randint(*TOP_SLACK_RANGE))
        if inner_h > INNER_HEIGHT_MAX:
            continue

        max_w = max(widths)
        anchor_w = random.choice(widths)
        want_odd_centering = random.choice((0, 1))
        slack_options = [
            slack
            for slack in WIDTH_SLACK_VALUES
            if (max_w + slack - anchor_w) % 2 == want_odd_centering
        ]
        if not slack_options:
            continue
        inner_w = int(max_w + random.choice(slack_options))
        if inner_w > INNER_WIDTH_MAX:
            continue

        return {
            "color": int(color),
            "masks": masks,
            "inner_h": int(inner_h),
            "inner_w": int(inner_w),
            "frame_h": int(inner_h + 2),
            "frame_w": int(inner_w + 2),
        }
    return None


def _frame_patch(spec: dict) -> np.ndarray:
    h = int(spec["frame_h"])
    w = int(spec["frame_w"])
    color = int(spec["color"])
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _fits(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = blocked.shape
    rr0 = int(r0 - OBJECT_MARGIN)
    cc0 = int(c0 - OBJECT_MARGIN)
    rr1 = int(r0 + h + OBJECT_MARGIN)
    cc1 = int(c0 + w + OBJECT_MARGIN)
    if rr0 < 0 or cc0 < 0 or rr1 > H or cc1 > W:
        return False
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = blocked.shape
    rr0 = max(0, int(r0 - OBJECT_MARGIN))
    cc0 = max(0, int(c0 - OBJECT_MARGIN))
    rr1 = min(H, int(r0 + h + OBJECT_MARGIN))
    cc1 = min(W, int(c0 + w + OBJECT_MARGIN))
    blocked[rr0:rr1, cc0:cc1] = True


def _free_position(blocked: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = blocked.shape
    candidates = [
        (r, c)
        for r in range(0, H - int(h) + 1)
        for c in range(0, W - int(w) + 1)
        if _fits(blocked, r, c, int(h), int(w))
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BLACK
    window = grid[int(r0) : int(r0) + patch.shape[0], int(c0) : int(c0) + patch.shape[1]]
    window[mask] = patch[mask]


def _paste_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    rows, cols = np.where(mask)
    grid[int(r0) + rows, int(c0) + cols] = int(color)


def _paint_expected_stack(grid: np.ndarray, spec: dict, frame_r: int, frame_c: int) -> None:
    color = int(spec["color"])
    inner_left = int(frame_c + 1)
    inner_bottom = int(frame_r + spec["frame_h"] - 2)
    inner_w = int(spec["inner_w"])
    cursor_bottom = inner_bottom

    for mask in sorted(spec["masks"], key=lambda m: -int(m.sum())):
        h, w = map(int, mask.shape)
        top = int(cursor_bottom - h + 1)
        left = int(inner_left + (inner_w - w) // 2)
        _paste_mask(grid, mask, color, top, left)
        cursor_bottom = int(top - 1)


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(COLOR_POOL, random.randint(*CONTAINER_COUNT_RANGE))
    specs: list[dict] = []
    for color in colors:
        spec = _make_spec(int(color))
        if spec is None:
            return None
        specs.append(spec)

    min_h = max(GRID_SIZE_RANGE[0], max(int(spec["frame_h"]) for spec in specs) + 4)
    min_w = max(GRID_SIZE_RANGE[0], max(int(spec["frame_w"]) for spec in specs) + 4)
    if min_h > GRID_SIZE_RANGE[1] or min_w > GRID_SIZE_RANGE[1]:
        return None
    H = random.randint(min_h, GRID_SIZE_RANGE[1])
    W = random.randint(min_w, GRID_SIZE_RANGE[1])

    for _ in range(POSITION_RETRIES):
        inp = np.zeros((H, W), dtype=int)
        expected = np.zeros((H, W), dtype=int)
        blocked = np.zeros((H, W), dtype=bool)

        placed_specs: list[tuple[dict, int, int]] = []
        frame_order = specs[:]
        random.shuffle(frame_order)
        ok = True
        for spec in frame_order:
            patch = _frame_patch(spec)
            pos = _free_position(blocked, patch.shape[0], patch.shape[1])
            if pos is None:
                ok = False
                break
            r0, c0 = pos
            _paste_patch(inp, patch, r0, c0)
            _paste_patch(expected, patch, r0, c0)
            _mark(blocked, r0, c0, patch.shape[0], patch.shape[1])
            placed_specs.append((spec, int(r0), int(c0)))

        if not ok:
            continue

        for spec, r0, c0 in placed_specs:
            _paint_expected_stack(expected, spec, r0, c0)

        fragments: list[tuple[np.ndarray, int]] = []
        for spec in specs:
            for mask in spec["masks"]:
                fragments.append((mask, int(spec["color"])))
        random.shuffle(fragments)

        for mask, color in fragments:
            pos = _free_position(blocked, mask.shape[0], mask.shape[1])
            if pos is None:
                ok = False
                break
            r0, c0 = pos
            _paste_mask(inp, mask, color, r0, c0)
            _mark(blocked, r0, c0, mask.shape[0], mask.shape[1])

        if not ok:
            continue

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid container-stacking input")
