import numpy as np


BLACK = 0
RED = 2
ORANGE = 7
TEAL = 8

EDGE_SIDES = ("top", "bottom", "left", "right")
EDGE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
TOUCH_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}

GRID_MIN = 17
GRID_MAX = 22
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 7
FRAME_COUNT_CHOICES = (2, 3)
LANE_GAP = 2
GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_edge_direction(grid: np.ndarray) -> tuple[str, int, int]:
    candidates: list[tuple[str, int, int]] = []
    if np.all(grid[0, :] == TEAL):
        candidates.append(("top", -1, 0))
    if np.all(grid[-1, :] == TEAL):
        candidates.append(("bottom", 1, 0))
    if np.all(grid[:, 0] == TEAL):
        candidates.append(("left", 0, -1))
    if np.all(grid[:, -1] == TEAL):
        candidates.append(("right", 0, 1))
    assert len(candidates) == 1, "expected exactly one full teal boundary line"
    return candidates[0]


def _frame_border_cells(frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _noncorner_border_cells(frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    cells: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _side_cells(frame: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise AssertionError(f"unknown side: {side}")


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int], color: int) -> None:
    for r, c in _frame_border_cells(frame):
        grid[r, c] = color


def _fill_frame_interior(grid: np.ndarray, frame: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = frame
    grid[r0 + 1 : r1, c0 + 1 : c1] = color


def _classify_red_markers(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    embedded: list[tuple[int, int]] = []
    isolated: list[tuple[int, int]] = []
    red_cells = np.argwhere(grid == RED)
    for r0, c0 in red_cells:
        r = int(r0)
        c = int(c0)
        orange_neighbors = 0
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == ORANGE:
                orange_neighbors += 1
        if orange_neighbors == 0:
            isolated.append((r, c))
        else:
            # In a 1-pixel hollow rectangle, a side-gap marker exposes only 2 orange
            # neighbors, so all red cells adjacent to orange belong to the repair branch.
            embedded.append((r, c))
    return embedded, isolated


def _extract_frames(repaired: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components_of_color(repaired, ORANGE):
        r0, c0, r1, c1 = _bbox(comp)
        border = set(_frame_border_cells((r0, c0, r1, c1)))
        comp_set = set(comp)
        assert comp_set == border, "orange components must be complete rectangular frames"
        frames.append({"bbox": (r0, c0, r1, c1), "cells": comp_set})
    frames.sort(key=lambda item: item["bbox"])
    assert 2 <= len(frames) <= 3
    return frames


def _analyze_input(grid: np.ndarray) -> dict:
    side, dr, dc = _find_edge_direction(grid)
    embedded, isolated = _classify_red_markers(grid)

    repaired = np.array(grid, copy=True)
    for r, c in embedded:
        repaired[r, c] = ORANGE

    frames = _extract_frames(repaired)
    border_to_frame: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame["cells"]:
            border_to_frame[cell] = idx

    touched: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for r, c in isolated:
        temp = (r + dr, c + dc)
        assert in_bounds(grid.shape, temp[0], temp[1]), "temporary marker must stay in bounds"
        assert int(grid[temp]) == BLACK, "temporary marker cell must be black in the input"
        frame_ids = set()
        for nr, nc in neighbors4(*temp):
            if (nr, nc) in border_to_frame:
                frame_ids.add(border_to_frame[(nr, nc)])
        assert len(frame_ids) <= 1, "temporary marker must not touch multiple frames"
        if frame_ids:
            touched.append(((r, c), next(iter(frame_ids)), temp))

    return {
        "edge_side": side,
        "direction": (dr, dc),
        "embedded": embedded,
        "isolated": isolated,
        "repaired": repaired,
        "frames": frames,
        "touched": touched,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair frame-embedded red markers, then use isolated red markers to select which rectangular frames get teal-filled.

    Input structure:
    - A black grid contains 2-3 orange hollow rectangular frames and one full teal line on a single outer border.
    - Some red markers replace non-corner frame-border cells, while other red markers sit isolated in the black background.
    - Each isolated marker is positioned so that moving it one cell toward the teal edge line may place it next to exactly one repaired frame.

    Rule:
    - Use the full teal boundary line to determine a single cardinal direction.
    - Recolor every red marker that touches orange to orange, repairing the affected frame border cells.
    - For every isolated red marker, consider the one-step temporary marker obtained by moving it toward the teal edge line; if that temporary cell is 4-adjacent to an orange frame border, fill that repaired frame's interior with teal.
    - Finally erase all original red markers and leave the teal edge line and orange frame borders unchanged.

    Key insight:
    - The same red color splits by spatial context: red cells embedded in orange are border defects to repair, while red cells isolated on black are direction-driven triggers that choose which repaired frame to flood-fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _analyze_input(g)
    out = np.array(info["repaired"], copy=True)

    for _marker, frame_id, _temp in info["touched"]:
        _fill_frame_interior(out, info["frames"][frame_id]["bbox"], TEAL)

    out[out == RED] = BLACK
    return out


def _random_partition(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.integers(0, total + 1, size=parts - 1).tolist())
    values: list[int] = []
    prev = 0
    for cut in cuts:
        values.append(int(cut - prev))
        prev = int(cut)
    values.append(int(total - prev))
    return values


def _sample_frame_sizes(rng: np.random.Generator, count: int) -> list[int]:
    return [int(rng.integers(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1)) for _ in range(count)]


def _place_frames(rng: np.random.Generator, edge_side: str, count: int, h: int, w: int):
    if edge_side in ("left", "right"):
        heights = _sample_frame_sizes(rng, count)
        widths = _sample_frame_sizes(rng, count)
        usable_rows = h - 2
        need = sum(heights) + LANE_GAP * (count - 1)
        if need > usable_rows:
            return None
        slack = usable_rows - need
        row_extras = _random_partition(rng, slack, count + 1)

        frames = []
        cur = 1 + row_extras[0]
        for i in range(count):
            fh = heights[i]
            fw = widths[i]
            r0 = cur
            r1 = r0 + fh - 1
            if edge_side == "right":
                c_min = 3
                c_max = w - 3 - fw
            else:
                c_min = 1
                c_max = w - 5 - fw
            if c_min > c_max:
                return None
            c0 = int(rng.integers(c_min, c_max + 1))
            c1 = c0 + fw - 1
            frames.append((r0, c0, r1, c1))
            cur = r1 + 1 + LANE_GAP + row_extras[i + 1]
        return frames

    heights = _sample_frame_sizes(rng, count)
    widths = _sample_frame_sizes(rng, count)
    usable_cols = w - 2
    need = sum(widths) + LANE_GAP * (count - 1)
    if need > usable_cols:
        return None
    slack = usable_cols - need
    col_extras = _random_partition(rng, slack, count + 1)

    frames = []
    cur = 1 + col_extras[0]
    for i in range(count):
        fh = heights[i]
        fw = widths[i]
        c0 = cur
        c1 = c0 + fw - 1
        if edge_side == "bottom":
            r_min = 3
            r_max = h - 3 - fh
        else:
            r_min = 1
            r_max = h - 5 - fh
        if r_min > r_max:
            return None
        r0 = int(rng.integers(r_min, r_max + 1))
        r1 = r0 + fh - 1
        frames.append((r0, c0, r1, c1))
        cur = c1 + 1 + LANE_GAP + col_extras[i + 1]
    return frames


def _touch_border_and_marker(
    frame: tuple[int, int, int, int], edge_side: str, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    side = TOUCH_SIDE[edge_side]
    border = tuple(_side_cells(frame, side)[int(rng.integers(0, len(_side_cells(frame, side))))])
    dr, dc = EDGE_TO_DIR[edge_side]
    temp = (border[0] - dr, border[1] - dc)
    origin = (border[0] - 2 * dr, border[1] - 2 * dc)
    return border, temp, origin


def _choose_filled_ids(rng: np.random.Generator, frame_count: int) -> list[int]:
    if frame_count == 2:
        filled_count = 1 if rng.random() < 0.7 else 2
    else:
        filled_count = int(rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25]))
    return sorted(rng.choice(frame_count, size=filled_count, replace=False).tolist())


def _choose_gap_cells(
    rng: np.random.Generator,
    frames: list[tuple[int, int, int, int]],
    filled_ids: list[int],
    trigger_borders: dict[int, tuple[int, int]],
) -> list[tuple[int, int]]:
    all_gaps: list[tuple[int, int]] = []
    force_coupled = set()
    if filled_ids:
        force_coupled.add(filled_ids[int(rng.integers(0, len(filled_ids)))])

    for idx, frame in enumerate(frames):
        count = int(rng.choice([0, 1, 2], p=[0.20, 0.55, 0.25]))
        required: list[tuple[int, int]] = []
        if idx in force_coupled:
            required = [trigger_borders[idx]]
            count = max(count, 1)
        if idx not in force_coupled and idx in filled_ids and rng.random() < 0.35:
            required = [trigger_borders[idx]]
            count = max(count, 1)

        candidates = _noncorner_border_cells(frame)
        chosen = list(required)
        for cell in candidates:
            if len(chosen) >= count:
                break
            if cell in chosen:
                continue
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) <= 1 for rr, cc in chosen):
                continue
            if rng.random() < 0.4:
                chosen.append(cell)
        for cell in candidates:
            if len(chosen) >= count:
                break
            if cell in chosen:
                continue
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) <= 1 for rr, cc in chosen):
                continue
            chosen.append(cell)
        all_gaps.extend(chosen)

    if not all_gaps:
        idx = filled_ids[0] if filled_ids else 0
        all_gaps.append(trigger_borders.get(idx, _noncorner_border_cells(frames[idx])[0]))
    return all_gaps


def _build_expected_output(
    shape: tuple[int, int],
    edge_side: str,
    frames: list[tuple[int, int, int, int]],
    filled_ids: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    if edge_side == "top":
        out[0, :] = TEAL
    elif edge_side == "bottom":
        out[-1, :] = TEAL
    elif edge_side == "left":
        out[:, 0] = TEAL
    else:
        out[:, -1] = TEAL

    for idx, frame in enumerate(frames):
        _draw_frame(out, frame, ORANGE)
        if idx in filled_ids:
            _fill_frame_interior(out, frame, TEAL)
    return out


def _build_input(
    shape: tuple[int, int],
    edge_side: str,
    frames: list[tuple[int, int, int, int]],
    gap_cells: list[tuple[int, int]],
    isolated_origins: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    if edge_side == "top":
        grid[0, :] = TEAL
    elif edge_side == "bottom":
        grid[-1, :] = TEAL
    elif edge_side == "left":
        grid[:, 0] = TEAL
    else:
        grid[:, -1] = TEAL

    for frame in frames:
        _draw_frame(grid, frame, ORANGE)
    for r, c in gap_cells:
        grid[r, c] = RED
    for r, c in isolated_origins:
        grid[r, c] = RED
    return grid


def _validate_generated_case(
    grid: np.ndarray,
    expected: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    filled_ids: list[int],
    gap_cells: list[tuple[int, int]],
    isolated_origins: list[tuple[int, int]],
    edge_side: str,
) -> bool:
    if grid.shape != expected.shape:
        return False
    if max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) - {BLACK, RED, ORANGE, TEAL}:
        return False

    info = _analyze_input(grid)
    if info["edge_side"] != edge_side:
        return False
    if len(info["frames"]) != len(frames):
        return False
    if sorted(info["embedded"]) != sorted(gap_cells):
        return False
    if sorted(info["isolated"]) != sorted(isolated_origins):
        return False

    touched_ids = sorted(frame_id for _marker, frame_id, _temp in info["touched"])
    if touched_ids != sorted(filled_ids):
        return False
    if len(set(touched_ids)) != len(touched_ids):
        return False

    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame
        interior = expected[r0 + 1 : r1, c0 + 1 : c1]
        if idx in filled_ids:
            if not np.all(interior == TEAL):
                return False
        else:
            if not np.all(interior == BLACK):
                return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if np.any(solved == RED):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.choice(FRAME_COUNT_CHOICES))
    edge_side = str(rng.choice(EDGE_SIDES))
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))

    frames = _place_frames(rng, edge_side, frame_count, h, w)
    if frames is None:
        return None

    filled_ids = _choose_filled_ids(rng, frame_count)

    trigger_borders: dict[int, tuple[int, int]] = {}
    isolated_origins: list[tuple[int, int]] = []
    reserved = set()
    for idx in filled_ids:
        border, temp, origin = _touch_border_and_marker(frames[idx], edge_side, rng)
        if temp in reserved or origin in reserved:
            return None
        if not in_bounds((h, w), *temp) or not in_bounds((h, w), *origin):
            return None
        outer = (origin[0] - EDGE_TO_DIR[edge_side][0], origin[1] - EDGE_TO_DIR[edge_side][1])
        if not in_bounds((h, w), *outer):
            return None
        trigger_borders[idx] = border
        isolated_origins.append(origin)
        reserved.add(temp)
        reserved.add(origin)

    gap_cells = _choose_gap_cells(rng, frames, filled_ids, trigger_borders)
    if len(set(gap_cells)) != len(gap_cells):
        return None

    expected = _build_expected_output((h, w), edge_side, frames, filled_ids)
    grid = _build_input((h, w), edge_side, frames, gap_cells, isolated_origins)

    if not _validate_generated_case(
        grid=grid,
        expected=expected,
        frames=frames,
        filled_ids=filled_ids,
        gap_cells=gap_cells,
        isolated_origins=isolated_origins,
        edge_side=edge_side,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
