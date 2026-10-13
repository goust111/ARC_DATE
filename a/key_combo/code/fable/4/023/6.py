from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
AXIS_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

_SCENE_CYCLE = ("lines_only", "mixed", "pyramids_only", "mixed")
_GEN_CALL_COUNT = 0
_GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    out.sort(key=lambda coords: (_bbox(coords)[0], _bbox(coords)[1]))
    return out


def _try_parse_line(grid: np.ndarray, coords: np.ndarray) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if c0 != c1:
        return None
    color_values = {int(grid[int(r), int(c)]) for r, c in coords}
    if len(color_values) != 1:
        return None
    color = next(iter(color_values))
    if color in (BLACK, GREY):
        return None
    height = r1 - r0 + 1
    if coords.shape[0] != height:
        return None
    if set(map(int, coords[:, 0])) != set(range(r0, r1 + 1)):
        return None
    return {
        "state": "line",
        "color": color,
        "height": height,
        "bottom_row": r1,
        "center_col": c0,
    }


def _try_parse_pyramid(grid: np.ndarray, coords: np.ndarray) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width != 2 * height - 1 or (width % 2) != 1:
        return None
    if coords.shape[0] != height * height:
        return None

    non_grey = {int(grid[int(r), int(c)]) for r, c in coords if int(grid[int(r), int(c)]) != GREY}
    if len(non_grey) != 1:
        return None
    color = next(iter(non_grey))
    if color == BLACK:
        return None

    center_col = (c0 + c1) // 2
    for r in range(r0, r1 + 1):
        span = r - r0
        for c in range(c0, c1 + 1):
            inside = abs(c - center_col) <= span
            expected = BLACK
            if inside:
                expected = color if abs(c - center_col) % 2 == 0 else GREY
            if int(grid[r, c]) != expected:
                return None

    return {
        "state": "pyramid",
        "color": color,
        "height": height,
        "bottom_row": r1,
        "center_col": center_col,
    }


def _parse_objects(grid: np.ndarray) -> list[dict[str, int | str]]:
    specs: list[dict[str, int | str]] = []
    for coords in _components(grid != BLACK):
        spec = _try_parse_line(grid, coords)
        if spec is None:
            spec = _try_parse_pyramid(grid, coords)
        if spec is None:
            raise AssertionError("every foreground object must be either a vertical line or an alternating pyramid")
        specs.append(spec)
    return specs


def _footprint_bbox(spec: dict[str, int | str]) -> tuple[int, int, int, int]:
    height = int(spec["height"])
    bottom = int(spec["bottom_row"])
    center = int(spec["center_col"])
    return bottom - height + 1, center - height + 1, bottom, center + height - 1


def _paint_line(out: np.ndarray, spec: dict[str, int | str]) -> None:
    height = int(spec["height"])
    bottom = int(spec["bottom_row"])
    center = int(spec["center_col"])
    color = int(spec["color"])
    top = bottom - height + 1
    assert np.all(out[top : bottom + 1, center] == BLACK)
    out[top : bottom + 1, center] = color


def _paint_pyramid(out: np.ndarray, spec: dict[str, int | str]) -> None:
    height = int(spec["height"])
    bottom = int(spec["bottom_row"])
    center = int(spec["center_col"])
    color = int(spec["color"])
    top = bottom - height + 1
    for r in range(top, bottom + 1):
        span = r - top
        for c in range(center - span, center + span + 1):
            assert out[r, c] == BLACK
            out[r, c] = color if abs(c - center) % 2 == 0 else GREY


def _render_scene(
    shape: tuple[int, int],
    specs: list[dict[str, int | str]],
    *,
    flip_state: bool,
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        state = str(spec["state"])
        if flip_state:
            state = "pyramid" if state == "line" else "line"
        if state == "line":
            _paint_line(out, spec)
        else:
            _paint_pyramid(out, spec)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Toggle each foreground object between its vertical-line state and its alternating grey pyramid state.

    Input structure:
    - The background is black.
    - Each foreground object is isolated from the others.
    - Every object is either a single-color vertical line segment, or a centered pyramid whose odd horizontal offsets are grey and whose even offsets use one non-grey axis color.

    Rule:
    - For a vertical line of height L with bottom cell at (r, c), replace it by the height-L pyramid centered on column c and ending on row r.
    - For a pyramid, recover its unique inverse image: the axis-color vertical line on the center column, from the bottom row upward for the pyramid height.
    - Apply the swap independently to every object and keep the grid size unchanged.

    Key insight:
    - Lines and pyramids are two adjacent states of the same expand/collapse operator, so the task is to recognize the current state of each object before moving one step in the opposite direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    if not np.any(g != BLACK):
        return g.copy()
    specs = _parse_objects(g)
    return _render_scene(g.shape, specs, flip_state=True)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _inflated_bbox(spec: dict[str, int | str]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _footprint_bbox(spec)
    return r0 - 1, c0 - 1, r1 + 1, c1 + 1


def _choose_scene_type() -> str:
    global _GEN_CALL_COUNT
    scene = _SCENE_CYCLE[_GEN_CALL_COUNT % len(_SCENE_CYCLE)]
    _GEN_CALL_COUNT += 1
    return scene


def _choose_states(count: int, scene_type: str, rng: np.random.Generator) -> list[str]:
    if scene_type == "lines_only":
        return ["line"] * count
    if scene_type == "pyramids_only":
        return ["pyramid"] * count

    n_lines = int(rng.integers(1, count))
    states = ["line"] * n_lines + ["pyramid"] * (count - n_lines)
    rng.shuffle(states)
    return states


def _sample_canvas(heights: list[int], count: int, rng: np.random.Generator) -> tuple[int, int]:
    max_h = max(heights)
    min_rows = max(10, max_h + 4, 5 * count)
    min_cols = max(12, 2 * max_h + 5, 6 * count)
    rows = int(rng.integers(min_rows, min(30, min_rows + 8) + 1))
    cols = int(rng.integers(min_cols, min(30, min_cols + 10) + 1))
    return rows, cols


def _place_specs(
    rows: int,
    cols: int,
    heights: list[int],
    states: list[str],
    colors: list[int],
    rng: np.random.Generator,
) -> list[dict[str, int | str]] | None:
    specs: list[dict[str, int | str] | None] = [None] * len(heights)
    inflated_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(heights)), key=lambda i: (-heights[i], i))

    for idx in order:
        height = int(heights[idx])
        if rows < height + 3 or cols < 2 * height + 3:
            return None
        placed = False
        for _ in range(250):
            bottom = int(rng.integers(height, rows - 1))
            center = int(rng.integers(height, cols - height))
            spec: dict[str, int | str] = {
                "state": states[idx],
                "color": colors[idx],
                "height": height,
                "bottom_row": bottom,
                "center_col": center,
            }
            box = _inflated_bbox(spec)
            if any(_boxes_overlap(box, other) for other in inflated_boxes):
                continue
            specs[idx] = spec
            inflated_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return [spec for spec in specs if spec is not None]


def _gen_attempt(rng: np.random.Generator, scene_type: str) -> np.ndarray | None:
    count = int(rng.integers(2, 5))
    states = _choose_states(count, scene_type, rng)
    heights = [int(rng.integers(2, 7)) for _ in range(count)]
    colors = list(map(int, rng.choice(np.asarray(AXIS_COLORS, dtype=int), size=count, replace=False)))
    rows, cols = _sample_canvas(heights, count, rng)
    specs = _place_specs(rows, cols, heights, states, colors, rng)
    if specs is None:
        return None

    inp = _render_scene((rows, cols), specs, flip_state=False)
    out = _render_scene((rows, cols), specs, flip_state=True)

    if len(_components(inp != BLACK)) != count:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if not np.array_equal(main(out), inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    scene_type = _choose_scene_type()
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng, scene_type)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
