import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    rel.sort()
    return tuple(rel)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _frame_color(grid: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(grid[grid != int(bg)], return_counts=True)
    assert len(vals) > 0, "expected a non-background frame color"
    return int(vals[np.argmax(counts)])


def _mono_components(grid: np.ndarray, skip: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in skip or seen[r, c]:
                continue
            seen[r, c] = True
            dq = deque([(r, c)])
            pts = []
            while dq:
                cr, cc = dq.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords, "shape": _normalize(coords)})
    return comps


def _slot_region(
    grid: np.ndarray,
    bg: int,
    top_line: int,
    bottom_line: int,
    left_line: int,
    right_line: int,
) -> np.ndarray:
    box = grid[top_line : bottom_line + 1, left_line : right_line + 1]
    traversable = box == int(bg)
    seeds = [(r, c) for r in range(1, box.shape[0] - 1) for c in range(1, box.shape[1] - 1) if traversable[r, c]]
    assert seeds, "each slot must contain interior background seeds"

    seen = np.zeros_like(traversable, dtype=bool)
    dq = deque([seeds[0]])
    seen[seeds[0]] = True

    while dq:
        r, c = dq.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < box.shape[0] and 0 <= nc < box.shape[1] and traversable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                dq.append((nr, nc))

    assert all(seen[r, c] for r, c in seeds), "slot cavity must stay 4-connected"

    coords = np.argwhere(seen)
    coords[:, 0] += int(top_line)
    coords[:, 1] += int(left_line)
    return coords


def _extract_slots(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    frame_coords = np.argwhere(grid == int(frame_color))
    assert frame_coords.size > 0, "expected a grey frame"
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)
    frame = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]

    row_counts = np.sum(frame == int(frame_color), axis=1)
    col_counts = np.sum(frame == int(frame_color), axis=0)
    line_rows = [int(fr0 + r) for r, count in enumerate(row_counts) if int(count) > frame.shape[1] // 2]
    line_cols = [int(fc0 + c) for c, count in enumerate(col_counts) if int(count) > frame.shape[0] // 2]
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "failed to detect frame grid lines"

    cell_heights = [line_rows[i + 1] - line_rows[i] - 1 for i in range(len(line_rows) - 1)]
    cell_widths = [line_cols[i + 1] - line_cols[i] - 1 for i in range(len(line_cols) - 1)]
    assert min(cell_heights) >= 2 and len(set(cell_heights)) == 1
    assert min(cell_widths) >= 2 and len(set(cell_widths)) == 1

    slots: list[dict] = []
    for i in range(len(line_rows) - 1):
        for j in range(len(line_cols) - 1):
            coords = _slot_region(grid, bg, line_rows[i], line_rows[i + 1], line_cols[j], line_cols[j + 1])
            slots.append({"slot": (i, j), "coords": coords, "shape": _normalize(coords)})
    return slots


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    height = max(r for r, _c in shape) + 1
    width = max(c for _r, c in shape) + 1
    patch = np.full((height, width), BG, dtype=int)
    for r, c in shape:
        patch[int(r), int(c)] = int(color)
    return patch


def _frame_geometry(top: int, left: int, rows: int, cols: int, cell_h: int, cell_w: int):
    line_rows = [int(top + k * (cell_h + 1)) for k in range(rows + 1)]
    line_cols = [int(left + k * (cell_w + 1)) for k in range(cols + 1)]
    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1
    return line_rows, line_cols, height, width


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    vertical_features: dict[tuple[int, int], tuple[int, int]],
    horizontal_features: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int, int]:
    line_rows, line_cols, frame_h, frame_w = _frame_geometry(top, left, rows, cols, cell_h, cell_w)
    grid[np.array(line_rows), left : left + frame_w] = FRAME
    grid[top : top + frame_h, np.array(line_cols)] = FRAME

    for (slot_r, slot_c), (local_r, owner) in vertical_features.items():
        gap_r = line_rows[slot_r] + 1 + int(local_r)
        gap_c = line_cols[slot_c + 1]
        grid[gap_r, gap_c] = BG
        if int(owner) < 0:
            grid[gap_r, gap_c + 1] = FRAME
        else:
            grid[gap_r, gap_c - 1] = FRAME

    for (slot_r, slot_c), (local_c, owner) in horizontal_features.items():
        gap_r = line_rows[slot_r + 1]
        gap_c = line_cols[slot_c] + 1 + int(local_c)
        grid[gap_r, gap_c] = BG
        if int(owner) < 0:
            grid[gap_r + 1, gap_c] = FRAME
        else:
            grid[gap_r - 1, gap_c] = FRAME

    return int(top), int(left), int(top + frame_h - 1), int(left + frame_w - 1)


def _can_place_patch(
    occupied: np.ndarray,
    forbidden: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 1 or left < 1 or top + ph >= h or left + pw >= w:
        return False

    cells = np.argwhere(patch != BG)
    for rr, cc in cells:
        r = int(top + rr)
        c = int(left + cc)
        if forbidden[r, c]:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw-like pieces into the frame slots whose edge teeth complement them.

    Input structure:
    - The background is black.
    - A single grey rectangular frame is subdivided by grey grid lines into equal rectangular slots.
    - Shared slot boundaries can contain one-pixel gaps in the divider line or one-pixel grey bumps extending into a slot.
    - Outside the frame, there is one colored 4-connected piece per slot; each piece has the exact protrusions and notches needed for one slot.

    Rule:
    - Detect the frame and recover each slot's true fillable cavity, including accessible divider gaps and excluding grey bumps.
    - Extract every external colored piece as a normalized shape.
    - Match each piece to the unique slot cavity with the same translated shape, which is equivalent to complementary tooth matching.
    - Clear the scattered pieces from outside the frame and paint each matched slot cavity with the piece's own color; keep the grey frame unchanged.

    Key insight:
    The tooth patterns can be solved purely geometrically by viewing each slot as a shaped cavity and each external piece as its translated counterpart.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frame_color = _frame_color(g, bg)

    slots = _extract_slots(g, bg, frame_color)
    pieces = _mono_components(g, {int(bg), int(frame_color)})
    assert len(pieces) == len(slots), "expected one colored piece per slot"

    shape_to_slot: dict[tuple[tuple[int, int], ...], dict] = {}
    for slot in slots:
        assert slot["shape"] not in shape_to_slot, "slot shapes must be unique"
        shape_to_slot[slot["shape"]] = slot

    out = g.copy()
    out[(out != int(bg)) & (out != int(frame_color))] = int(bg)

    for piece in pieces:
        assert piece["shape"] in shape_to_slot, "piece shape must match a slot cavity"
        slot = shape_to_slot.pop(piece["shape"])
        coords = slot["coords"]
        out[coords[:, 0], coords[:, 1]] = int(piece["color"])

    assert not shape_to_slot, "every slot must receive a piece"
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(((2, 2), (2, 3), (3, 2)))
    cell_h = random.randint(4, 5)
    cell_w = random.randint(4, 5)

    frame_h = rows * cell_h + rows + 1
    frame_w = cols * cell_w + cols + 1

    min_h = max(frame_h + 8, 22)
    min_w = max(frame_w + 8, 22)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    top = random.randint(2, grid_h - frame_h - 2)
    left = random.randint(2, grid_w - frame_w - 2)

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    vertical_features: dict[tuple[int, int], tuple[int, int]] = {}
    horizontal_features: dict[tuple[int, int], tuple[int, int]] = {}

    for slot_r in range(rows):
        for slot_c in range(cols - 1):
            vertical_features[(slot_r, slot_c)] = (
                random.choice(list(range(1, cell_h - 1))),
                random.choice((-1, 1)),
            )

    for slot_r in range(rows - 1):
        for slot_c in range(cols):
            horizontal_features[(slot_r, slot_c)] = (
                random.choice(list(range(1, cell_w - 1))),
                random.choice((-1, 1)),
            )

    fr0, fc0, fr1, fc1 = _draw_frame(
        grid,
        top,
        left,
        rows,
        cols,
        cell_h,
        cell_w,
        vertical_features,
        horizontal_features,
    )

    try:
        slots = _extract_slots(grid, BG, FRAME)
    except AssertionError:
        return None
    if len({slot["shape"] for slot in slots}) != len(slots):
        return None

    colors = random.sample(PALETTE, len(slots))
    pieces = []
    for slot, color in zip(slots, colors):
        patch = _shape_to_patch(slot["shape"], int(color))
        pieces.append({"color": int(color), "patch": patch, "coords": slot["coords"], "area": int(np.count_nonzero(patch))})

    occupied = grid != BG
    forbidden = np.zeros_like(occupied, dtype=bool)
    forbidden[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    for piece in sorted(pieces, key=lambda item: (-item["area"], item["color"])):
        placed = False
        ph, pw = piece["patch"].shape
        for _ in range(800):
            top0 = random.randint(1, grid_h - ph - 1)
            left0 = random.randint(1, grid_w - pw - 1)
            if not _can_place_patch(occupied, forbidden, piece["patch"], top0, left0):
                continue
            cells = np.argwhere(piece["patch"] != BG)
            for rr, cc in cells:
                grid[top0 + int(rr), left0 + int(cc)] = int(piece["color"])
                occupied[top0 + int(rr), left0 + int(cc)] = True
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    expected[(expected != BG) & (expected != FRAME)] = BG
    for piece in pieces:
        coords = piece["coords"]
        expected[coords[:, 0], coords[:, 1]] = int(piece["color"])

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
    raise RuntimeError("failed to generate a valid jigsaw-slot puzzle after retries")
