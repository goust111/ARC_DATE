from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
MARKER = 2
CANVAS = 9
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_TO_IDX = {side: idx for idx, side in enumerate(SIDE_ORDER)}
FRAME_COLORS = [0, 1, 3, 4, 5, 6, 7]

TEMP_SIZE = 30
GEN_ATTEMPTS = 400
PATTERN_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 80


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_rect_frame(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    coord_set = {tuple(map(int, xy)) for xy in coords}
    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return coord_set == expected


def _midpoint_outside(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mid_r = r0 + (h - 1) // 2
    mid_c = c0 + (w - 1) // 2
    if side == "top":
        return r0 - 1, mid_c
    if side == "right":
        return mid_r, c1 + 1
    if side == "bottom":
        return r1 + 1, mid_c
    if side == "left":
        return mid_r, c0 - 1
    raise ValueError(side)


def _frame_marker_side(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str | None:
    h, w = grid.shape
    found = []
    for side in SIDE_ORDER:
        rr, cc = _midpoint_outside(bbox, side)
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == MARKER:
            found.append(side)
    if len(found) != 1:
        return None
    return found[0]


def _find_frame_objects(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in {BG, MARKER, CANVAS}:
            continue
        for coords in _find_color_components(grid, color):
            bbox = _bbox_from_coords(coords)
            if not _is_rect_frame(coords, bbox):
                continue
            side = _frame_marker_side(grid, bbox)
            if side is None:
                continue
            frames.append({"color": color, "bbox": bbox, "side": side})
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside stamp to the same-colored hole in the maroon canvas and rotate its interior into place.

    Input structure:
    - The background is azure, and one large maroon rectangle acts as a canvas.
    - Inside the canvas are rectangular azure holes surrounded by one-cell-thick colored frames, with one red marker just outside one side of each frame.
    - Outside the canvas are same-sized rectangular stamps with colored frames, asymmetric multi-color interiors, and one red marker just outside one side of the frame.

    Rule:
    - Find the canvas, then detect every rectangular colored frame with its red outside marker.
    - Frames strictly inside the canvas are holes; frames outside the canvas are stamps, and each stamp matches the unique hole with the same frame color.
    - Rotate each stamp interior clockwise until its marker side matches the hole marker side, copy that rotated interior into the hole, and erase the original stamp plus its marker to azure.

    Key insight:
    - Frame color determines which stamp belongs to which hole, while the red marker side determines the required clockwise rotation before filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    canvas_cells = np.argwhere(g == CANVAS)
    assert canvas_cells.size > 0, "expected a maroon canvas"
    canvas_bbox = _bbox_from_coords(canvas_cells)
    cr0, cc0, cr1, cc1 = canvas_bbox

    holes: dict[int, dict] = {}
    stamps: dict[int, dict] = {}
    for frame in _find_frame_objects(g):
        r0, c0, r1, c1 = frame["bbox"]
        if cr0 < r0 and r1 < cr1 and cc0 < c0 and c1 < cc1:
            assert frame["color"] not in holes, "hole colors must be unique"
            holes[frame["color"]] = frame
        else:
            assert frame["color"] not in stamps, "stamp colors must be unique"
            stamps[frame["color"]] = frame

    assert holes and stamps and set(holes) == set(stamps), "frame colors must pair holes and stamps"

    out = g.copy()
    for color, hole in holes.items():
        stamp = stamps[color]
        sr0, sc0, sr1, sc1 = stamp["bbox"]
        hr0, hc0, hr1, hc1 = hole["bbox"]

        pattern = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        turns = (SIDE_TO_IDX[hole["side"]] - SIDE_TO_IDX[stamp["side"]]) % 4
        rotated = np.rot90(pattern, -turns)
        assert rotated.shape == (hr1 - hr0 - 1, hc1 - hc0 - 1)

        out[hr0 + 1 : hr1, hc0 + 1 : hc1] = rotated
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
        mr, mc = _midpoint_outside(stamp["bbox"], stamp["side"])
        if 0 <= mr < out.shape[0] and 0 <= mc < out.shape[1]:
            out[mr, mc] = BG

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, outer: int, color: int, side: str, interior: np.ndarray | None) -> None:
    bottom = top + outer - 1
    right = left + outer - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    if interior is None:
        grid[top + 1 : bottom, left + 1 : right] = BG
    else:
        grid[top + 1 : bottom, left + 1 : right] = interior
    mr, mc = _midpoint_outside((top, left, bottom, right), side)
    grid[mr, mc] = MARKER


def _random_pattern(size: int, colors: list[int]) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        n_colors = random.randint(2, min(3, len(colors)))
        chosen = random.sample(colors, n_colors)
        pattern = np.array(
            [[random.choice(chosen) for _ in range(size)] for _ in range(size)],
            dtype=int,
        )
        if len(set(map(int, pattern.ravel()))) < 2:
            continue
        if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            continue
        return pattern
    raise RuntimeError("failed to generate a non-symmetric pattern")


def _find_free_rect(blocked: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    spots: list[tuple[int, int]] = []
    H, W = blocked.shape
    for r in range(1, H - h):
        for c in range(1, W - w):
            if not blocked[r : r + h, c : c + w].any():
                spots.append((r, c))
    if not spots:
        return None
    return random.choice(spots)


def _mark_blocked(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _build_canvas(n_pairs: int, inner: int, frame_colors: list[int]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    outer = inner + 2
    slot = outer + 2
    if n_pairs == 2:
        rows, cols = random.choice([(1, 2), (2, 1)])
    else:
        rows, cols = 2, 2
    row_gap = random.randint(1, 2)
    col_gap = random.randint(1, 2)
    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)

    canvas_h = margin_top + rows * slot + (rows - 1) * row_gap + margin_bottom
    canvas_w = margin_left + cols * slot + (cols - 1) * col_gap + margin_right
    canvas = np.full((canvas_h, canvas_w), CANVAS, dtype=int)
    solved = canvas.copy()

    slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    chosen_slots = random.sample(slots, n_pairs)
    hole_data: list[dict] = []

    deltas = random.sample(range(4), n_pairs)
    random.shuffle(frame_colors)
    pattern_colors = [color for color in FRAME_COLORS if color not in frame_colors]
    for idx, ((slot_r, slot_c), border_color, turns) in enumerate(zip(chosen_slots, frame_colors, deltas, strict=True)):
        top = margin_top + slot_r * (slot + row_gap) + 1
        left = margin_left + slot_c * (slot + col_gap) + 1
        hole_side = random.choice(SIDE_ORDER)
        stamp_side = SIDE_ORDER[(SIDE_TO_IDX[hole_side] - turns) % 4]
        pattern = _random_pattern(inner, pattern_colors)

        _draw_frame(canvas, top, left, outer, border_color, hole_side, None)
        _draw_frame(solved, top, left, outer, border_color, hole_side, np.rot90(pattern, -turns))
        hole_data.append(
            {
                "color": border_color,
                "hole_side": hole_side,
                "stamp_side": stamp_side,
                "pattern": pattern,
                "turns": turns,
                "outer": outer,
                "index": idx,
            }
        )
    return canvas, solved, hole_data


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    pad_top = random.randint(0, min(2, r0))
    pad_left = random.randint(0, min(2, c0))
    pad_bottom = random.randint(0, min(2, inp.shape[0] - 1 - r1))
    pad_right = random.randint(0, min(2, inp.shape[1] - 1 - c1))
    rs = slice(r0 - pad_top, r1 + pad_bottom + 1)
    cs = slice(c0 - pad_left, c1 + pad_right + 1)
    return inp[rs, cs], out[rs, cs]


def _generate_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        n_pairs = random.randint(2, 4)
        inner = random.randint(2, 4)
        frame_colors = random.sample(FRAME_COLORS, n_pairs)
        canvas, solved_canvas, hole_data = _build_canvas(n_pairs, inner, frame_colors)

        board_in = np.full((TEMP_SIZE, TEMP_SIZE), BG, dtype=int)
        board_out = np.full((TEMP_SIZE, TEMP_SIZE), BG, dtype=int)
        blocked = np.zeros((TEMP_SIZE, TEMP_SIZE), dtype=bool)
        blocked[0, :] = True
        blocked[-1, :] = True
        blocked[:, 0] = True
        blocked[:, -1] = True

        canvas_pos = _find_free_rect(blocked, canvas.shape[0], canvas.shape[1])
        if canvas_pos is None:
            continue
        cr, cc = canvas_pos
        board_in[cr : cr + canvas.shape[0], cc : cc + canvas.shape[1]] = canvas
        board_out[cr : cr + solved_canvas.shape[0], cc : cc + solved_canvas.shape[1]] = solved_canvas
        _mark_blocked(blocked, cr, cc, canvas.shape[0], canvas.shape[1], pad=1)

        random.shuffle(hole_data)
        ok = True
        for item in hole_data:
            reserve = item["outer"] + 2
            placed = None
            for _ in range(PLACEMENT_ATTEMPTS):
                cand = _find_free_rect(blocked, reserve, reserve)
                if cand is None:
                    break
                placed = cand
                break
            if placed is None:
                ok = False
                break
            rr, cc2 = placed
            stamp_top = rr + 1
            stamp_left = cc2 + 1
            _draw_frame(
                board_in,
                stamp_top,
                stamp_left,
                item["outer"],
                item["color"],
                item["stamp_side"],
                item["pattern"],
            )
            _mark_blocked(blocked, rr, cc2, reserve, reserve, pad=1)
        if not ok:
            continue

        inp, expected = _crop_pair(board_in, board_out)
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp, expected

    raise RuntimeError("failed to generate a valid example")


def generate_input() -> np.ndarray:
    return _generate_pair()[0]
