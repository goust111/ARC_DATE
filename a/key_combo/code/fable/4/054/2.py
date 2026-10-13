import numpy as np


Color = type(
    "Color",
    (),
    {
        "BLACK": 0,
        "ALL": tuple(range(10)),
        "NON_BLACK": tuple(range(1, 10)),
    },
)

_RNG = np.random.default_rng()

_MIN_HEIGHT = 8
_MAX_HEIGHT = 18
_MIN_WIDTH = 7
_MAX_WIDTH = 16
_MIN_OBJECTS = 4
_MAX_OBJECTS = 6
_MAX_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_bar(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row_color = int(grid[r, 0])
        if row_color != Color.BLACK and np.all(grid[r] == row_color):
            candidates.append((r, row_color))
    assert len(candidates) == 1, "expected exactly one full-width horizontal bar"
    return candidates[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _parse_objects(grid: np.ndarray, bar_row: int) -> list[dict[str, int]]:
    mask = grid != Color.BLACK
    mask[bar_row, :] = False
    objects = []
    for comp in _components(mask):
        assert len(comp) == 2, "expected every non-bar object to be a two-pixel component"
        comp = sorted(comp)
        (r0, c0), (r1, c1) = comp
        assert c0 == c1 and r1 == r0 + 1, "expected vertical engine-cargo pairs"
        assert r1 < bar_row or r0 > bar_row, "pairs must lie entirely on one side of the bar"
        if r1 < bar_row:
            cargo = (r1, c1)
            engine = (r0, c0)
            dr = 1
        else:
            cargo = (r0, c0)
            engine = (r1, c1)
            dr = -1
        cargo_color = int(grid[cargo])
        engine_color = int(grid[engine])
        assert cargo_color != engine_color, "engine and cargo colors must differ"
        objects.append(
            {
                "row": int(cargo[0]),
                "col": int(cargo[1]),
                "cargo_color": cargo_color,
                "engine_color": engine_color,
                "dr": int(dr),
            }
        )
    assert objects, "expected at least one engine-cargo pair"
    return objects


def _simulate(
    shape: tuple[int, int],
    bar_row: int,
    bar_color: int,
    objects: list[dict[str, int]],
) -> list[dict[str, int]]:
    h, _w = map(int, shape)
    active = [obj.copy() for obj in objects]
    stopped: list[dict[str, int]] = []
    stopped_cells: set[tuple[int, int]] = set()

    while active:
        next_active: list[dict[str, int]] = []
        stop_now: list[dict[str, int]] = []
        move_targets: list[tuple[int, int]] = []

        for obj in active:
            row = int(obj["row"])
            col = int(obj["col"])
            dr = int(obj["dr"])
            next_row = row + dr

            if not (0 <= next_row < h):
                stop_now.append(obj)
                continue
            if (next_row, col) in stopped_cells:
                stop_now.append(obj)
                continue

            if next_row == bar_row:
                if int(obj["engine_color"]) != int(bar_color):
                    stop_now.append(obj)
                    continue
                next_row += dr
                if not (0 <= next_row < h):
                    stop_now.append(obj)
                    continue
                if (next_row, col) in stopped_cells:
                    stop_now.append(obj)
                    continue

            moved = obj.copy()
            moved["row"] = int(next_row)
            move_targets.append((int(next_row), col))
            next_active.append(moved)

        stop_cells_this_step = {(int(obj["row"]), int(obj["col"])) for obj in stop_now}
        assert len(set(move_targets)) == len(move_targets), "ambiguous simultaneous move collision"
        assert stop_cells_this_step.isdisjoint(move_targets), "move into a cell that stops this step"

        for obj in stop_now:
            cell = (int(obj["row"]), int(obj["col"]))
            stopped.append(obj.copy())
            stopped_cells.add(cell)
        active = next_active

    return stopped


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each cargo straight toward the horizontal bar, letting only same-color engines pass through it.

    Input structure:
    - The background is black.
    - Exactly one non-black row is a full-width horizontal bar of a single color.
    - Every other foreground object is a vertical two-pixel pair on one side of the bar.
    - In each pair, the pixel nearer the bar is the cargo and the farther pixel is the engine.

    Rule:
    - Remove every engine pixel and clear every cargo start position.
    - All cargos move simultaneously one step at a time toward the bar.
    - A cargo stops if its next step would leave the grid or hit a cargo that has already stopped.
    - When the next step reaches the bar, cargos whose engine color matches the bar skip across the bar and keep moving; all others stop adjacent to the bar.

    Key insight:
    - The engine contributes two signals at once: its position fixes the cargo's motion direction, and its color decides whether the cargo treats the bar as a blocker or a transparent gate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    bg = _get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    bar_row, bar_color = _find_bar(g)
    objects = _parse_objects(g, bar_row)
    final_objects = _simulate(g.shape, bar_row, bar_color, objects)

    out = np.full_like(g, Color.BLACK)
    out[bar_row, :] = int(bar_color)
    for obj in final_objects:
        out[int(obj["row"]), int(obj["col"])] = int(obj["cargo_color"])
    return out


def _pick_non_bar_colors(bar_color: int) -> list[int]:
    return [color for color in Color.NON_BLACK if color != bar_color]


def _make_positions(bar_row: int, height: int, side: str, count: int) -> list[int]:
    if side == "top":
        choices = list(range(1, bar_row))
    else:
        choices = list(range(bar_row + 1, height - 1))
    assert choices, f"no valid cargo rows on {side} side"
    return [int(_RNG.choice(choices)) for _ in range(count)]


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(_MIN_HEIGHT, _MAX_HEIGHT + 1))
    width = int(_RNG.integers(_MIN_WIDTH, _MAX_WIDTH + 1))
    if width < _MIN_OBJECTS:
        return None

    bar_row_lo = 3
    bar_row_hi = height - 4
    if bar_row_lo > bar_row_hi:
        return None
    bar_row = int(_RNG.integers(bar_row_lo, bar_row_hi + 1))
    bar_color = int(_RNG.choice(np.array(Color.NON_BLACK, dtype=int)))

    object_count = int(_RNG.integers(_MIN_OBJECTS, min(_MAX_OBJECTS, width) + 1))
    if object_count < 2:
        return None

    columns = [int(c) for c in _RNG.choice(np.arange(width), size=object_count, replace=False)]
    _RNG.shuffle(columns)

    top_count = int(_RNG.integers(1, object_count))
    bottom_count = object_count - top_count
    if bottom_count == 0:
        bottom_count = 1
        top_count -= 1
    if top_count == 0 or bottom_count == 0:
        return None

    pass_flags = np.zeros(object_count, dtype=bool)
    pass_flags[0] = True
    pass_flags[1] = False
    if object_count > 2:
        pass_flags[2:] = _RNG.random(object_count - 2) < 0.5
    _RNG.shuffle(pass_flags)

    cargo_palette = _pick_non_bar_colors(bar_color)
    if object_count > len(cargo_palette):
        return None
    cargo_colors = [int(c) for c in _RNG.choice(np.array(cargo_palette, dtype=int), size=object_count, replace=False)]

    grid = np.zeros((height, width), dtype=int)
    grid[bar_row, :] = bar_color

    top_rows = _make_positions(bar_row, height, "top", top_count)
    bottom_rows = _make_positions(bar_row, height, "bottom", bottom_count)

    for idx in range(object_count):
        col = int(columns[idx])
        cargo_color = int(cargo_colors[idx])
        if bool(pass_flags[idx]):
            engine_color = int(bar_color)
        else:
            blocker_choices = [color for color in cargo_palette if color != cargo_color]
            engine_color = int(_RNG.choice(np.array(blocker_choices, dtype=int)))

        if idx < top_count:
            cargo_row = int(top_rows[idx])
            engine_row = cargo_row - 1
        else:
            cargo_row = int(bottom_rows[idx - top_count])
            engine_row = cargo_row + 1

        if not (0 <= engine_row < height):
            return None
        if grid[cargo_row, col] != Color.BLACK or grid[engine_row, col] != Color.BLACK:
            return None

        grid[cargo_row, col] = cargo_color
        grid[engine_row, col] = engine_color

    non_bar_mask = grid != Color.BLACK
    non_bar_mask[bar_row, :] = False
    comps = _components(non_bar_mask)
    if len(comps) != object_count:
        return None
    if not any(pass_flags) or np.all(pass_flags):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
