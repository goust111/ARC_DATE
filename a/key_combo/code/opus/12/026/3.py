import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
MARKER_PRIORITY = {2: 0, 3: 1, 4: 2, 5: 3}
FRAME_SIZE = 4
INNER_SLICE = slice(1, 3)
OBJECT_GAP = 1
MAX_ATTEMPTS = 600

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (20, 24)
FRAME_COUNT_WEIGHTS = ((3, 1), (4, 3))
SOLID_COUNT_RANGE = (1, 3)
AXES = ("horizontal", "vertical")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLUE:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BLUE:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _classify_object(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    box = _bbox(coords)
    patch = _crop(grid, box)
    assert patch.shape == (FRAME_SIZE, FRAME_SIZE), "every blue object must have a 4x4 bbox"
    if np.all(patch == BLUE):
        return {"kind": "solid", "bbox": box, "patch": patch}

    assert bool(np.all(patch[0, :] == BLUE))
    assert bool(np.all(patch[-1, :] == BLUE))
    assert bool(np.all(patch[:, 0] == BLUE))
    assert bool(np.all(patch[:, -1] == BLUE))
    interior = patch[INNER_SLICE, INNER_SLICE]
    assert not bool(np.any(interior == BLUE)), "frame interior may not contain blue"
    marker_coords = np.argwhere(interior != BG)
    assert marker_coords.shape == (1, 2), "frame must contain exactly one interior marker"
    mr, mc = map(int, marker_coords[0])
    marker_color = int(interior[mr, mc])
    assert marker_color in MARKER_PRIORITY, "marker color must be one of red/green/yellow/grey"
    assert int(np.count_nonzero(interior == BG)) == 3
    return {
        "kind": "frame",
        "bbox": box,
        "patch": patch,
        "marker_color": marker_color,
        "marker_pos": (mr, mc),
    }


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = [_classify_object(g, comp) for comp in _blue_components(g)]
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]
    assert frames, "expected at least one hollow frame"
    marker_colors = [obj["marker_color"] for obj in frames]
    assert len(set(marker_colors)) == len(marker_colors), "frame marker colors must be distinct"

    rows = [obj["bbox"][0] for obj in frames]
    cols = [obj["bbox"][1] for obj in frames]
    row_span = max(rows) - min(rows)
    col_span = max(cols) - min(cols)
    axis = "horizontal" if col_span > row_span else "vertical"
    axis_key = 1 if axis == "horizontal" else 0

    return {
        "grid": g,
        "frames": frames,
        "solids": solids,
        "axis": axis,
        "axis_key": axis_key,
        "row_span": row_span,
        "col_span": col_span,
    }


def _pack_frames(frames: list[dict], axis: str) -> np.ndarray:
    ordered = sorted(frames, key=lambda obj: MARKER_PRIORITY[obj["marker_color"]])
    n = len(ordered)
    if axis == "horizontal":
        out = np.zeros((FRAME_SIZE, FRAME_SIZE * n), dtype=int)
        for i, obj in enumerate(ordered):
            _paste(out, obj["patch"], 0, i * FRAME_SIZE)
        return out

    out = np.zeros((FRAME_SIZE * n, FRAME_SIZE), dtype=int)
    for i, obj in enumerate(ordered):
        _paste(out, obj["patch"], i * FRAME_SIZE, 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the hollow blue frames into a strip ordered by their embedded marker colors.

    Input structure:
    - A black background contains separated blue 4x4 objects.
    - Some objects are solid blue 4x4 distractor blocks.
    - The retained objects are hollow blue 4x4 square frames whose 2x2 interior has exactly one marker pixel in red, green, yellow, or grey and three black cells.

    Rule:
    - Detect every blue 4x4 object and discard the solid blue squares.
    - For each hollow frame, read the unique non-black marker color inside its 2x2 interior.
    - Sort the retained frames by marker color priority: red first, then green, then yellow, then grey.
    - If the hollow-frame column span is larger than their row span, concatenate the sorted frames left-to-right; otherwise concatenate them top-to-bottom.

    Key insight:
    - The packing direction comes from the frames' original layout, but the order itself comes from the contained marker colors rather than from the frames' own positions.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    return _pack_frames(analysis["frames"], analysis["axis"])


def _choice_from_weighted_pairs(pairs: tuple[tuple[int, int], ...]) -> int:
    values = [value for value, _weight in pairs]
    weights = [weight for _value, weight in pairs]
    return random.choices(values, weights=weights, k=1)[0]


def _slot_anchors(length: int) -> list[int]:
    count = (length + OBJECT_GAP) // (FRAME_SIZE + OBJECT_GAP)
    total = count * FRAME_SIZE + (count - 1) * OBJECT_GAP
    slack = length - total
    offset = random.randint(0, slack)
    return [offset + (FRAME_SIZE + OBJECT_GAP) * i for i in range(count)]


def _frame_positions(
    axis: str, row_anchors: list[int], col_anchors: list[int], n_frames: int
) -> list[tuple[int, int]] | None:
    if axis == "horizontal":
        if len(col_anchors) < n_frames:
            return None
        col_slots = random.sample(range(len(col_anchors)), n_frames)
        row_band = random.choice([1, 2]) if len(row_anchors) >= 2 else 1
        if row_band > len(row_anchors):
            return None
        row_start = random.randint(0, len(row_anchors) - row_band)
        row_slots = [random.randint(row_start, row_start + row_band - 1) for _ in range(n_frames)]
        positions = [(row_anchors[rs], col_anchors[cs]) for rs, cs in zip(row_slots, col_slots)]
        cols = [c for _r, c in positions]
        rows = [r for r, _c in positions]
        if max(cols) - min(cols) <= max(rows) - min(rows):
            return None
        return positions

    if len(row_anchors) < n_frames:
        return None
    row_slots = random.sample(range(len(row_anchors)), n_frames)
    col_band = random.choice([1, 2]) if len(col_anchors) >= 2 else 1
    if col_band > len(col_anchors):
        return None
    col_start = random.randint(0, len(col_anchors) - col_band)
    col_slots = [random.randint(col_start, col_start + col_band - 1) for _ in range(n_frames)]
    positions = [(row_anchors[rs], col_anchors[cs]) for rs, cs in zip(row_slots, col_slots)]
    cols = [c for _r, c in positions]
    rows = [r for r, _c in positions]
    if max(cols) - min(cols) > max(rows) - min(rows):
        return None
    return positions


def _free_positions(
    row_anchors: list[int], col_anchors: list[int], occupied: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [(r, c) for r in row_anchors for c in col_anchors if (r, c) not in occupied]


def _frame_patch(marker_color: int, marker_pos: tuple[int, int]) -> np.ndarray:
    patch = np.full((FRAME_SIZE, FRAME_SIZE), BLUE, dtype=int)
    patch[INNER_SLICE, INNER_SLICE] = BG
    patch[1 + marker_pos[0], 1 + marker_pos[1]] = marker_color
    return patch


def _solid_patch() -> np.ndarray:
    return np.full((FRAME_SIZE, FRAME_SIZE), BLUE, dtype=int)


def _expected_from_analysis(analysis: dict) -> np.ndarray:
    return _pack_frames(analysis["frames"], analysis["axis"])


def _order_signature(frames: list[dict], key_fn) -> tuple[int, ...]:
    return tuple(obj["marker_color"] for obj in sorted(frames, key=key_fn))


def _anti_shortcut_ok(analysis: dict) -> bool:
    frames = analysis["frames"]
    color_order = tuple(
        obj["marker_color"] for obj in sorted(frames, key=lambda obj: MARKER_PRIORITY[obj["marker_color"]])
    )
    axis_order = _order_signature(frames, key_fn=lambda obj: obj["bbox"][analysis["axis_key"]])
    if axis_order == color_order:
        return False

    row_major = _order_signature(frames, key_fn=lambda obj: obj["marker_pos"][0] * 2 + obj["marker_pos"][1])
    col_major = _order_signature(frames, key_fn=lambda obj: obj["marker_pos"][1] * 2 + obj["marker_pos"][0])
    return row_major != color_order and col_major != color_order


def _generator_checks(grid: np.ndarray, axis: str, n_frames: int, n_solids: int) -> bool:
    try:
        analysis = _analyze_grid(grid)
        expected = _expected_from_analysis(analysis)
    except AssertionError:
        return False

    if len(analysis["frames"]) != n_frames or len(analysis["solids"]) != n_solids:
        return False
    if analysis["axis"] != axis:
        return False
    if not _anti_shortcut_ok(analysis):
        return False
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(AXES)
    n_frames = _choice_from_weighted_pairs(FRAME_COUNT_WEIGHTS)
    n_solids = random.randint(*SOLID_COUNT_RANGE)
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    row_anchors = _slot_anchors(h)
    col_anchors = _slot_anchors(w)

    positions = _frame_positions(axis, row_anchors, col_anchors, n_frames)
    if positions is None:
        return None

    marker_colors = random.sample(list(MARKER_PRIORITY), n_frames)
    internal_positions = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], n_frames)
    random.shuffle(positions)

    grid = np.zeros((h, w), dtype=int)
    occupied = set(positions)
    for (r, c), marker_color, marker_pos in zip(positions, marker_colors, internal_positions):
        _paste(grid, _frame_patch(marker_color, marker_pos), r, c)

    free = _free_positions(row_anchors, col_anchors, occupied)
    if len(free) < n_solids:
        return None
    for r, c in random.sample(free, n_solids):
        _paste(grid, _solid_patch(), r, c)

    if not _generator_checks(grid, axis, n_frames, n_solids):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
