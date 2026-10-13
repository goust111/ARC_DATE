from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
BORDER = 5
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == BORDER):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h not in (3, 5):
            continue
        inner_size = h - 2
        if len(coords) != h * h - inner_size * inner_size:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        ring = np.ones((h, w), dtype=bool)
        ring[1:-1, 1:-1] = False
        if not np.all(patch[ring] == BORDER):
            continue
        inner = patch[1:-1, 1:-1]
        inner_colors = np.unique(inner)
        if len(inner_colors) != 1:
            continue
        color = int(inner_colors[0])
        if color in (BACKGROUND, BORDER):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "color": color,
                "outer_size": h,
                "inner_size": inner_size,
            }
        )
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))
    return frames


def _find_control_rows(grid: np.ndarray, bottom_of_frames: int) -> list[dict]:
    rows = [r for r in range(bottom_of_frames + 1, grid.shape[0]) if np.any(grid[r] != BACKGROUND)]
    if not rows:
        return []
    return [{"row": int(r), "seq": [int(v) for v in grid[r] if int(v) != BACKGROUND]} for r in rows]


def _bridge_between(frame_a: dict, frame_b: dict) -> dict | None:
    ar0, ac0, ar1, ac1 = frame_a["bbox"]
    br0, bc0, br1, bc1 = frame_b["bbox"]
    iar0, iac0, iar1, iac1 = frame_a["inner_bbox"]
    ibr0, ibc0, ibr1, ibc1 = frame_b["inner_bbox"]

    candidates: list[dict] = []

    row0 = max(iar0, ibr0)
    row1 = min(iar1, ibr1)
    if row0 <= row1:
        if ac1 < bc0 - 1:
            candidates.append(
                {
                    "orientation": "h",
                    "r0": row0,
                    "r1": row1,
                    "c0": ac1 + 1,
                    "c1": bc0 - 1,
                    "gap": bc0 - ac1 - 1,
                }
            )
        elif bc1 < ac0 - 1:
            candidates.append(
                {
                    "orientation": "h",
                    "r0": row0,
                    "r1": row1,
                    "c0": bc1 + 1,
                    "c1": ac0 - 1,
                    "gap": ac0 - bc1 - 1,
                }
            )

    col0 = max(iac0, ibc0)
    col1 = min(iac1, ibc1)
    if col0 <= col1:
        if ar1 < br0 - 1:
            candidates.append(
                {
                    "orientation": "v",
                    "r0": ar1 + 1,
                    "r1": br0 - 1,
                    "c0": col0,
                    "c1": col1,
                    "gap": br0 - ar1 - 1,
                }
            )
        elif br1 < ar0 - 1:
            candidates.append(
                {
                    "orientation": "v",
                    "r0": br1 + 1,
                    "r1": ar0 - 1,
                    "c0": col0,
                    "c1": col1,
                    "gap": ar0 - br1 - 1,
                }
            )

    if not candidates:
        return None
    return min(candidates, key=lambda b: (b["gap"], b["orientation"]))


def _apply_bridge(out: np.ndarray, bridge: dict, color: int) -> None:
    out[bridge["r0"] : bridge["r1"] + 1, bridge["c0"] : bridge["c1"] + 1] = int(color)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _find_frames(g)
    assert frames, "expected framed squares"
    color_counts = Counter(int(frame["color"]) for frame in frames)
    active_by_color = {int(frame["color"]): frame for frame in frames if color_counts[int(frame["color"])] == 1}
    inactive_frames = [frame for frame in frames if color_counts[int(frame["color"])] > 1]

    bottom = max(int(frame["bbox"][2]) for frame in frames)
    control_rows = _find_control_rows(g, bottom)
    assert control_rows, "expected candidate control rows below the frames"

    valid_rows: list[int] = []
    bridges_by_row: dict[int, list[dict]] = {}
    for idx, row in enumerate(control_rows):
        seq = row["seq"]
        if len(seq) < 2 or any(color not in active_by_color for color in seq):
            continue
        bridges: list[dict] = []
        ok = True
        for a, b in zip(seq, seq[1:]):
            if a == b:
                ok = False
                break
            bridge = _bridge_between(active_by_color[a], active_by_color[b])
            if bridge is None:
                ok = False
                break
            bridges.append(bridge)
        if ok:
            valid_rows.append(idx)
            bridges_by_row[idx] = bridges

    return {
        "grid": g,
        "frames": frames,
        "active_by_color": active_by_color,
        "inactive_frames": inactive_frames,
        "control_rows": control_rows,
        "valid_rows": valid_rows,
        "bridges_by_row": bridges_by_row,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only uniquely colored framed squares, then use the sole surviving control sequence to draw bridges.

    Input structure:
    - The background is black, and 6-8 objects are grey 3x3 or 5x5 framed squares with a uniform non-black inner fill.
    - Some fill colors appear once across all squares while others repeat; repeated-color squares are distractors.
    - Two or three adjacent control rows sit below the squares, each encoding a left-to-right color sequence with isolated colored pixels.

    Rule:
    - Count how often each square's inner fill color appears among all framed squares.
    - Erase every square whose fill color is not unique, and erase all control rows.
    - Find the unique control row whose whole sequence refers only to surviving squares; for each consecutive color pair, draw a straight bridge between the corresponding squares along their shared row band or column band, coloring that bridge with the earlier color and filling only the gap between the two frames.

    Key insight:
    - Frequency filtering determines which squares survive, and that surviving set in turn determines which control row is valid and therefore which bridge topology is drawn.
    """
    state = _analyze(grid)
    assert len(state["valid_rows"]) == 1, "expected exactly one valid control row"
    valid_idx = state["valid_rows"][0]
    seq = state["control_rows"][valid_idx]["seq"]

    out = np.zeros_like(state["grid"])
    for frame in state["active_by_color"].values():
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = state["grid"][r0 : r1 + 1, c0 : c1 + 1]

    for color, bridge in zip(seq, state["bridges_by_row"][valid_idx]):
        _apply_bridge(out, bridge, int(color))
    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, outer_size: int, color: int) -> dict:
    bottom = top + outer_size - 1
    right = left + outer_size - 1
    grid[top : bottom + 1, left : right + 1] = BORDER
    grid[top + 1 : bottom, left + 1 : right] = int(color)
    return {
        "bbox": (top, left, bottom, right),
        "inner_bbox": (top + 1, left + 1, bottom - 1, right - 1),
        "color": int(color),
        "outer_size": outer_size,
        "inner_size": outer_size - 2,
    }


def _choose_spaced_positions(max_start: int, count: int, min_gap: int) -> list[int] | None:
    choices = list(range(1, max_start + 1))
    random.shuffle(choices)

    def search(start_idx: int, picked: list[int]) -> list[int] | None:
        if len(picked) == count:
            return sorted(picked)
        for idx in range(start_idx, len(choices)):
            pos = choices[idx]
            if any(abs(pos - prev) < min_gap for prev in picked):
                continue
            picked.append(pos)
            found = search(idx + 1, picked)
            if found is not None:
                return found
            picked.pop()
        return None

    return search(0, [])


def _adjacent_nodes(rows: int, cols: int, node: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    random.shuffle(out)
    return out


def _path_has_turn(path: list[tuple[int, int]]) -> bool:
    kinds = set()
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        kinds.add("h" if r0 == r1 else "v")
    return len(kinds) >= 2


def _random_lattice_path(rows: int, cols: int, length: int, require_turn: bool) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(starts)

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            if not require_turn or _path_has_turn(path):
                return path[:]
            return None
        for nxt in _adjacent_nodes(rows, cols, path[-1]):
            if nxt in used:
                continue
            used.add(nxt)
            path.append(nxt)
            found = dfs(path, used)
            if found is not None:
                return found
            path.pop()
            used.remove(nxt)
        return None

    for start in starts:
        found = dfs([start], {start})
        if found is not None:
            return found
    return None


def _straight_lattice_path(rows: int, cols: int, length: int, orientation: str) -> list[tuple[int, int]] | None:
    if orientation == "h" and cols >= length:
        r = random.randrange(rows)
        start_c = random.randint(0, cols - length)
        return [(r, start_c + i) for i in range(length)]
    if orientation == "v" and rows >= length:
        c = random.randrange(cols)
        start_r = random.randint(0, rows - length)
        return [(start_r + i, c) for i in range(length)]
    return None


def _bridge_cells(frame_a: dict, frame_b: dict) -> set[tuple[int, int]]:
    bridge = _bridge_between(frame_a, frame_b)
    if bridge is None:
        return set()
    return {
        (r, c)
        for r in range(bridge["r0"], bridge["r1"] + 1)
        for c in range(bridge["c0"], bridge["c1"] + 1)
    }


def _bbox_clear(bbox: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _expanded_cells(bbox: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = set()
    for r in range(max(0, r0 - pad), min(h, r1 + pad + 1)):
        for c in range(max(0, c0 - pad), min(w, c1 + pad + 1)):
            out.add((r, c))
    return out


def _partition_repeated_count(total: int) -> list[int] | None:
    if total < 2:
        return None
    if total == 2:
        return [2]
    if total == 3:
        return [3]
    options = ([2, 2], [4])
    if total == 5:
        options = ([2, 3], [5])
    if total == 6:
        options = ([3, 3], [2, 4], [6])
    valid = [list(opt) for opt in options if sum(opt) == total]
    return random.choice(valid) if valid else None


def _make_invalid_sequence(valid_seq: list[int], inert_colors: list[int]) -> list[int]:
    seq = valid_seq[:]
    replace_count = random.choice((1, 1, 2))
    idxs = random.sample(range(len(seq)), replace_count)
    for idx in idxs:
        seq[idx] = random.choice(inert_colors)
    if seq == valid_seq:
        seq[0] = random.choice(inert_colors)
    return seq


def _compose_input() -> np.ndarray | None:
    h = random.randint(19, 25)
    w = random.randint(21, 29)
    control_count = random.choice((2, 3))
    control_start = h - control_count

    active_outer = random.choice((3, 5))
    total_frames = random.randint(6, 8)

    max_top = control_start - active_outer - 1
    max_left = w - active_outer - 1
    if max_top < 3 or max_left < 5:
        return None

    active_len = random.randint(3, min(5, total_frames - 2))
    row_count_options = [count for count in (2, 3) if count * 2 >= active_len or count * 3 >= active_len]
    if active_outer == 5:
        row_count_options = [count for count in row_count_options if count <= 2]
    if not row_count_options:
        return None

    for _ in range(40):
        row_count = random.choice(row_count_options)
        col_count = random.choice((3, 4)) if active_outer == 3 else 3
        if row_count * col_count < active_len:
            continue
        row_lanes = _choose_spaced_positions(max_top, row_count, active_outer + 2)
        col_lanes = _choose_spaced_positions(max_left, col_count, active_outer + 2)
        if row_lanes is None or col_lanes is None:
            continue
        path_mode = random.choices(("walk", "h", "v"), weights=(6, 2, 2), k=1)[0]
        if path_mode == "walk":
            require_turn = active_len >= 4 and random.random() < 0.7
            lattice_path = _random_lattice_path(row_count, col_count, active_len, require_turn)
        else:
            lattice_path = _straight_lattice_path(row_count, col_count, active_len, path_mode)
        if lattice_path is None:
            continue
        active_positions = [(row_lanes[r], col_lanes[c]) for r, c in lattice_path]
        break
    else:
        return None

    groups = _partition_repeated_count(total_frames - active_len)
    if groups is None:
        return None

    needed_colors = active_len + len(groups)
    if needed_colors > len(FILL_COLORS):
        return None
    palette = random.sample(FILL_COLORS, needed_colors)
    active_colors = palette[:active_len]
    repeated_colors = palette[active_len:]

    inert_specs: list[tuple[int, int]] = []
    for color, count in zip(repeated_colors, groups):
        for _ in range(count):
            inert_specs.append((random.choice((3, 5)), color))
    random.shuffle(inert_specs)

    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    active_frames: list[dict] = []
    for (top, left), color in zip(active_positions, active_colors):
        frame = _draw_frame(grid, top, left, active_outer, color)
        active_frames.append(frame)
        placed_boxes.append(frame["bbox"])

    reserved_bridge_cells: set[tuple[int, int]] = set()
    for a, b in zip(active_frames, active_frames[1:]):
        cells = _bridge_cells(a, b)
        if not cells:
            return None
        reserved_bridge_cells |= cells

    for outer_size, color in inert_specs:
        max_inert_top = control_start - outer_size - 1
        max_inert_left = w - outer_size - 1
        if max_inert_top < 1 or max_inert_left < 1:
            return None
        placed = False
        for _ in range(200):
            top = random.randint(1, max_inert_top)
            left = random.randint(1, max_inert_left)
            bbox = (top, left, top + outer_size - 1, left + outer_size - 1)
            if not _bbox_clear(bbox, placed_boxes, gap=1):
                continue
            if _expanded_cells(bbox, h, w, pad=0) & reserved_bridge_cells:
                continue
            frame = _draw_frame(grid, top, left, outer_size, color)
            placed_boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    valid_row_index = random.randrange(control_count)
    sequences: list[list[int]] = []
    repeated_color_list = [color for _outer, color in inert_specs]
    for idx in range(control_count):
        if idx == valid_row_index:
            sequences.append(active_colors[:])
        else:
            sequences.append(_make_invalid_sequence(active_colors, repeated_color_list))

    gaps = [random.randint(2, 4) for _ in range(active_len - 1)]
    width_needed = 1 + sum(gaps)
    if width_needed >= w - 2:
        return None
    start_col = random.randint(1, w - width_needed - 1)
    columns = [start_col]
    for gap in gaps:
        columns.append(columns[-1] + gap)

    for idx, seq in enumerate(sequences):
        row = control_start + idx
        for col, color in zip(columns, seq):
            grid[row, col] = int(color)

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        state = _analyze(grid)
        if not (6 <= len(state["frames"]) <= 8):
            return False
        if not (2 <= len(state["control_rows"]) <= 3):
            return False
        rows = [row["row"] for row in state["control_rows"]]
        if rows != list(range(rows[0], rows[0] + len(rows))):
            return False
        active_count = len(state["active_by_color"])
        if not (3 <= active_count <= 5):
            return False
        if len(state["inactive_frames"]) < 2:
            return False
        repeated = Counter(frame["color"] for frame in state["inactive_frames"])
        if not repeated or min(repeated.values()) < 2:
            return False
        if len(state["valid_rows"]) != 1:
            return False
        valid_idx = state["valid_rows"][0]
        seq = state["control_rows"][valid_idx]["seq"]
        if not (3 <= len(seq) <= 5):
            return False
        bridge_cells = set()
        for bridge in state["bridges_by_row"][valid_idx]:
            for r in range(bridge["r0"], bridge["r1"] + 1):
                for c in range(bridge["c0"], bridge["c1"] + 1):
                    bridge_cells.add((r, c))
        if not bridge_cells:
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if np.any(out[rows[0] : rows[-1] + 1] != BACKGROUND):
            return False
        for frame in state["inactive_frames"]:
            r0, c0, r1, c1 = frame["bbox"]
            if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BACKGROUND):
                return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _compose_input()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
