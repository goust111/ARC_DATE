from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7
CYAN = 8

INPUT_SIZE = 6
TEMPLATE_SIZE = 3
OUTPUT_SIZE = 15
GEN_ATTEMPTS = 400

DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
INPUT_CELLS = tuple((r, c) for r in range(INPUT_SIZE) for c in range(INPUT_SIZE))
MODE_SEQUENCE = (
    "right_long_both_shift",
    "down_short",
    "right_long_one_shift",
    "right_short",
)


def _connected8(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                    nxt = (int(nr), int(nc))
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
    return len(seen) == int(np.count_nonzero(mask))


def _enumerate_glyph_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (TEMPLATE_SIZE * TEMPLATE_SIZE)):
        flat = [(bits >> i) & 1 for i in range(TEMPLATE_SIZE * TEMPLATE_SIZE)]
        mask = np.array(flat, dtype=bool).reshape(TEMPLATE_SIZE, TEMPLATE_SIZE)
        area = int(np.count_nonzero(mask))
        if not 3 <= area <= 5:
            continue
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != TEMPLATE_SIZE - 1:
            continue
        if cols.min() != 0 or cols.max() != TEMPLATE_SIZE - 1:
            continue
        if not _connected8(mask):
            continue
        masks.append(mask)
    return masks


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(grid == CYAN)
    assert 3 <= len(coords) <= 5, "expected a small cyan glyph"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    assert (r1 - r0 + 1, c1 - c0 + 1) == (TEMPLATE_SIZE, TEMPLATE_SIZE), "glyph bbox must be 3x3"
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == CYAN
    assert _connected8(mask), "glyph must be 8-connected"
    return mask, (r0, c0, r1, c1)


def _find_singleton(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) == 1, f"expected one marker of color {color}"
    return tuple(map(int, coords[0]))


def _tile_template(mask: np.ndarray) -> np.ndarray:
    rr, cc = np.indices((OUTPUT_SIZE, OUTPUT_SIZE))
    return np.where(mask[rr % TEMPLATE_SIZE, cc % TEMPLATE_SIZE], CYAN, BLACK).astype(int)


def _remap_marker(raw: tuple[int, int], canvas: np.ndarray) -> tuple[tuple[int, int], int]:
    r = int(raw[0] * 2 + 1)
    c = int(raw[1] * 2 + 1)
    shifts = 0
    while 0 <= r < OUTPUT_SIZE and 0 <= c < OUTPUT_SIZE and int(canvas[r, c]) == CYAN:
        r += 1
        c += 1
        shifts += 1
    assert 0 <= r < OUTPUT_SIZE and 0 <= c < OUTPUT_SIZE, "shifted marker left the canvas"
    assert int(canvas[r, c]) == BLACK, "shifted marker must land on a free cell"
    return (int(r), int(c)), int(shifts)


def _trace_path(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> dict[str, object]:
    direction = 1 if end[1] > start[1] else 2
    pos = start
    path = [start]
    turns = 0
    obstacle_turns = 0
    edge_turns = 0
    seen_states: set[tuple[tuple[int, int], int]] = set()

    while True:
        if pos == end:
            break
        state = (pos, direction)
        if state in seen_states:
            break
        seen_states.add(state)

        moved = False
        for _ in range(4):
            dr, dc = DIRS4[direction]
            nr = pos[0] + dr
            nc = pos[1] + dc
            if 0 <= nr < OUTPUT_SIZE and 0 <= nc < OUTPUT_SIZE and int(canvas[nr, nc]) != CYAN:
                pos = (int(nr), int(nc))
                path.append(pos)
                moved = True
                break
            turns += 1
            if not (0 <= nr < OUTPUT_SIZE and 0 <= nc < OUTPUT_SIZE):
                edge_turns += 1
            else:
                obstacle_turns += 1
            direction = (direction + 1) % 4
        if not moved:
            break

    unique_path = tuple(dict.fromkeys(path))
    return {
        "path": tuple(path),
        "unique_path": unique_path,
        "reached": bool(pos == end),
        "steps": int(len(path) - 1),
        "turns": int(turns),
        "obstacle_turns": int(obstacle_turns),
        "edge_turns": int(edge_turns),
        "initial_right": bool(end[1] > start[1]),
    }


def _solve_with_template(
    mask: np.ndarray,
    raw_start: tuple[int, int],
    raw_end: tuple[int, int],
) -> dict[str, object]:
    canvas = _tile_template(mask)
    start, start_shift = _remap_marker(raw_start, canvas)
    end, end_shift = _remap_marker(raw_end, canvas)
    assert start != end, "mapped start and end must differ"

    trace = _trace_path(canvas, start, end)
    out = canvas.copy()
    for r, c in trace["unique_path"]:
        out[r, c] = GREEN
    out[start] = GREEN
    out[end] = RED

    return {
        "output": out,
        "canvas": canvas,
        "start": start,
        "end": end,
        "start_shift": int(start_shift),
        "end_shift": int(end_shift),
        "trace": trace,
        "raw_start": raw_start,
        "raw_end": raw_end,
    }


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_SIZE, INPUT_SIZE), "expected a 6x6 input grid"
    allowed = {BLACK, RED, GREEN, ORANGE, CYAN}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"
    assert int(np.count_nonzero(g == ORANGE)) >= 27, "orange must remain the dominant input background"

    mask, bbox = _extract_template(g)
    start = _find_singleton(g, GREEN)
    end = _find_singleton(g, RED)
    r0, c0, r1, c1 = bbox
    assert not (r0 <= start[0] <= r1 and c0 <= start[1] <= c1), "start marker must lie outside the glyph bbox"
    assert not (r0 <= end[0] <= r1 and c0 <= end[1] <= c1), "end marker must lie outside the glyph bbox"

    solved = _solve_with_template(mask, start, end)
    solved["mask"] = mask
    solved["bbox"] = bbox
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the cyan 3x3 glyph into a tiled obstacle phase and trace the turn-forced path from the green marker to the red marker.

    Input structure:
    - The input is a 6x6 grid with orange background.
    - One cyan 8-connected glyph occupies exactly one 3x3 bounding box and acts as a binary phase template.
    - One green start marker and one red end marker each appear once, both outside that 3x3 glyph box.

    Rule:
    - Read the cyan cells inside the glyph bbox as 1 and the orange cells there as 0 to obtain a 3x3 binary template.
    - Build a 15x15 black canvas and tile that template every 3 rows and 3 columns, painting the 1-cells cyan as obstacles.
    - Map the green and red markers from (r, c) to (2r + 1, 2c + 1); if a mapped cell is cyan, repeatedly shift it one cell down-right until it lands on black.
    - Starting from green, move one step forward each turn; when the next step would hit cyan or leave the canvas, rotate clockwise in place and try again, using right as the initial direction when the end is to the right and down otherwise.
    - Paint every traversed free cell green, keep obstacles cyan, and restore the final end cell to red.

    Key insight:
    - The small cyan glyph is not a picture to copy: it is a phase code whose tiled residues determine the obstacle lattice, and that lattice in turn forces the local turning path.
    """
    return _analyze_input(grid)["output"]


def _precompute_mask_pairs(mask: np.ndarray) -> list[dict[str, object]]:
    solved_blank = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    canvas = _tile_template(mask)
    mapped: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {}
    for raw in INPUT_CELLS:
        try:
            mapped[raw] = _remap_marker(raw, canvas)
        except AssertionError:
            mapped[raw] = None

    pairs: list[dict[str, object]] = []
    for raw_start in INPUT_CELLS:
        src = mapped[raw_start]
        if src is None:
            continue
        for raw_end in INPUT_CELLS:
            if raw_end == raw_start:
                continue
            dst = mapped[raw_end]
            if dst is None or src[0] == dst[0]:
                continue

            trace = _trace_path(canvas, src[0], dst[0])
            if not bool(trace["reached"]):
                continue

            blank_trace = _trace_path(solved_blank, src[0], dst[0])
            pairs.append(
                {
                    "raw_start": raw_start,
                    "raw_end": raw_end,
                    "start_shift": int(src[1]),
                    "end_shift": int(dst[1]),
                    "initial_right": bool(trace["initial_right"]),
                    "steps": int(trace["steps"]),
                    "turns": int(trace["turns"]),
                    "obstacle_turns": int(trace["obstacle_turns"]),
                    "edge_turns": int(trace["edge_turns"]),
                    "path": trace["path"],
                    "differs_from_blank": bool(trace["path"] != blank_trace["path"]),
                    "raw_distance": int(abs(raw_start[0] - raw_end[0]) + abs(raw_start[1] - raw_end[1])),
                }
            )
    return pairs


def _build_instance_pools() -> tuple[list[np.ndarray], dict[str, list[dict[str, object]]]]:
    masks: list[np.ndarray] = []
    pools = {name: [] for name in MODE_SEQUENCE}
    for mask in _enumerate_glyph_masks():
        pairs = _precompute_mask_pairs(mask)
        if not pairs:
            continue

        mask_idx = len(masks)
        masks.append(mask)
        for top in range(INPUT_SIZE - TEMPLATE_SIZE + 1):
            for left in range(INPUT_SIZE - TEMPLATE_SIZE + 1):
                bbox_cells = {(top + dr, left + dc) for dr in range(TEMPLATE_SIZE) for dc in range(TEMPLATE_SIZE)}
                for pair in pairs:
                    if pair["raw_start"] in bbox_cells or pair["raw_end"] in bbox_cells:
                        continue
                    if not bool(pair["differs_from_blank"]):
                        continue

                    both_shift = bool(pair["start_shift"] > 0 and pair["end_shift"] > 0)
                    record = {
                        "mask_idx": int(mask_idx),
                        "bbox_top": int(top),
                        "bbox_left": int(left),
                        "raw_start": pair["raw_start"],
                        "raw_end": pair["raw_end"],
                        "start_shift": int(pair["start_shift"]),
                        "end_shift": int(pair["end_shift"]),
                        "steps": int(pair["steps"]),
                        "initial_right": bool(pair["initial_right"]),
                        "both_shift": both_shift,
                        "raw_distance": int(pair["raw_distance"]),
                        "key": (
                            int(mask_idx),
                            int(top),
                            int(left),
                            pair["raw_start"],
                            pair["raw_end"],
                        ),
                    }

                    if record["initial_right"] and record["steps"] == 4 and record["both_shift"]:
                        pools["right_long_both_shift"].append(record)
                    elif record["initial_right"] and record["steps"] == 4 and not record["both_shift"]:
                        pools["right_long_one_shift"].append(record)
                    elif not record["initial_right"] and record["steps"] == 2:
                        pools["down_short"].append(record)
                    elif record["initial_right"] and record["steps"] == 2:
                        pools["right_short"].append(record)
    for name, records in pools.items():
        assert records, f"instance pool {name} should not be empty"
    return masks, pools


_MASK_LIBRARY, _INSTANCE_POOLS = _build_instance_pools()
_RECENT_KEYS: deque[tuple[object, ...]] = deque(maxlen=24)
_GENERATE_CALLS = 0


def _render_input(record: dict[str, object]) -> np.ndarray:
    grid = np.full((INPUT_SIZE, INPUT_SIZE), ORANGE, dtype=int)
    top = int(record["bbox_top"])
    left = int(record["bbox_left"])
    mask = _MASK_LIBRARY[int(record["mask_idx"])]
    patch = grid[top : top + TEMPLATE_SIZE, left : left + TEMPLATE_SIZE]
    patch[mask] = CYAN
    grid[record["raw_start"]] = GREEN
    grid[record["raw_end"]] = RED
    return grid


def _pick_record(pool_name: str) -> dict[str, object]:
    pool = _INSTANCE_POOLS[pool_name]
    preferred = pool
    if pool_name == "down_short":
        far = [record for record in pool if int(record["raw_distance"]) >= 2]
        if far:
            preferred = far

    for _ in range(20):
        record = random.choice(preferred)
        if record["key"] not in _RECENT_KEYS:
            return record
    return random.choice(preferred)


def _semantic_checks(grid: np.ndarray, record: dict[str, object]) -> None:
    analysis = _analyze_input(grid)
    trace = analysis["trace"]
    assert bool(trace["reached"]), "generator must only emit reachable instances"
    assert int(analysis["start_shift"]) + int(analysis["end_shift"]) >= 1, "at least one marker should trigger the shift rule"
    assert int(trace["obstacle_turns"]) >= 1, "the path must turn because of an obstacle"
    assert bool(trace["path"] != _trace_path(np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int), analysis["start"], analysis["end"])["path"])
    assert int(trace["steps"]) == int(record["steps"])
    assert bool(trace["initial_right"]) == bool(record["initial_right"])
    assert (int(analysis["start_shift"]) > 0 and int(analysis["end_shift"]) > 0) == bool(record["both_shift"])


def generate_input() -> np.ndarray:
    global _GENERATE_CALLS

    for _ in range(GEN_ATTEMPTS):
        mode = MODE_SEQUENCE[_GENERATE_CALLS % len(MODE_SEQUENCE)]
        _GENERATE_CALLS += 1
        record = _pick_record(mode)
        grid = _render_input(record)
        _semantic_checks(grid, record)
        _RECENT_KEYS.append(record["key"])
        return grid
    raise RuntimeError("failed to generate a valid input")
