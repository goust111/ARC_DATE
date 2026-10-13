from __future__ import annotations

import numpy as np


BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == bg:
                continue
            color = int(grid[sr, sc])
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": _bbox(arr),
                }
            )
    return comps


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _is_hollow_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return False

    expected_area = 2 * height + 2 * width - 4
    if int(comp["area"]) != expected_area:
        return False

    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == bg))


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    candidates = [comp for comp in _components_by_color(grid, bg) if _is_hollow_frame(grid, comp, bg)]
    assert len(candidates) == 1, "expected exactly one hollow rectangular frame"
    frame = candidates[0]
    return frame["bbox"], int(frame["color"])


def _classify_pixel(
    r: int, c: int, box: tuple[int, int, int, int]
) -> tuple[str, str, int, tuple[int, int]]:
    r0, c0, r1, c1 = box
    row_aligned = r0 < r < r1 and (c < c0 or c > c1)
    col_aligned = c0 < c < c1 and (r < r0 or r > r1)
    assert row_aligned ^ col_aligned, "each outer pixel must align with exactly one frame interior row or column"

    if row_aligned:
        if c < c0:
            return "H", "L", r, (r, c0)
        return "H", "R", r, (r, c1)

    if r < r0:
        return "V", "T", c, (r0, c)
    return "V", "B", c, (r1, c)


def _opposite_side(side: str) -> str:
    return {"L": "R", "R": "L", "T": "B", "B": "T"}[side]


def _paint_segment(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
    bg: int,
    frame_color: int,
) -> None:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1, "rays must be horizontal or vertical"

    if r0 == r1:
        cols = range(min(c0, c1), max(c0, c1) + 1)
        cells = [(r0, c) for c in cols]
    else:
        rows = range(min(r0, r1), max(r0, r1) + 1)
        cells = [(r, c0) for r in rows]

    for r, c in cells:
        current = int(out[r, c])
        assert current in (bg, int(color), int(frame_color)), "rays must not cross a different colored object"
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Connect only opposing same-color rays through a hollow rectangular frame.

    Input structure:
    - A black-background grid contains one large single-color hollow rectangular
      frame whose interior is empty background.
    - Outside the frame are isolated colored pixels. Each such pixel lies on a
      row crossing the frame interior, or on a column crossing the frame interior.
    - For any aligned row or column there is at most one pixel on each side of
      the frame, so the possible opposite partner is unique.

    Rule:
    - Each outside pixel fires a same-colored straight ray toward the frame.
    - If the opposite side of the same row or column contains a same-color pixel,
      the two rays penetrate both frame walls and fill the whole straight segment
      between the two pixels, recoloring the crossed wall cells.
    - If the opposite side has no pixel or has a different-colored pixel, the ray
      stops at the near wall and recolors only that wall cell; no cell inside the
      frame is painted by that ray.

    Key insight:
    - Row or column alignment only identifies the wall hit. Penetration requires
      the stricter bilateral condition: the unique opposite-side pixel must have
      the same color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _most_common_color(g)
    box, frame_color = _find_frame(g, bg)
    mask = _frame_mask(g.shape, box)

    pixels = []
    groups: dict[tuple[str, int], dict[str, int]] = {}
    for comp in _components_by_color(g, bg):
        coords = comp["coords"]
        in_frame = mask[coords[:, 0], coords[:, 1]]
        if bool(np.all(in_frame)):
            assert int(comp["color"]) == frame_color, "only the frame may occupy frame cells"
            continue

        assert int(comp["area"]) == 1, "non-frame foreground objects must be isolated pixels"
        r, c = map(int, coords[0])
        assert not mask[r, c], "outside pixels must not sit on the frame"
        axis, side, line_index, wall = _classify_pixel(r, c, box)
        record = {
            "coord": (r, c),
            "color": int(comp["color"]),
            "axis": axis,
            "side": side,
            "line_index": int(line_index),
            "wall": wall,
        }
        idx = len(pixels)
        key = (axis, int(line_index))
        groups.setdefault(key, {})
        assert side not in groups[key], "each side of an aligned row or column may contain only one pixel"
        groups[key][side] = idx
        pixels.append(record)

    out = g.copy()
    for idx, pixel in enumerate(pixels):
        key = (pixel["axis"], pixel["line_index"])
        opposite = groups[key].get(_opposite_side(pixel["side"]))
        if opposite is not None and pixels[opposite]["color"] == pixel["color"]:
            _paint_segment(out, pixel["coord"], pixels[opposite]["coord"], pixel["color"], bg, frame_color)
        else:
            _paint_segment(out, pixel["coord"], pixel["wall"], pixel["color"], bg, frame_color)
    return out


def _choose_spaced(values: range, count: int) -> list[int] | None:
    vals = list(values)
    for _ in range(200):
        order = np.random.permutation(vals)
        chosen: list[int] = []
        for value in order:
            v = int(value)
            if all(abs(v - old) > 1 for old in chosen):
                chosen.append(v)
                if len(chosen) == count:
                    return chosen
    return None


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _is_available_singleton(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BACKGROUND:
        return False
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if int(grid[nr, nc]) != BACKGROUND:
                return False
    return True


def _place_side_pixel(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    axis: str,
    side: str,
    line_index: int,
    color: int,
) -> tuple[int, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    if axis == "H":
        r = int(line_index)
        choices = list(range(0, c0 - 1)) if side == "L" else list(range(c1 + 2, w))
        np.random.shuffle(choices)
        for c in choices:
            if _is_available_singleton(grid, r, int(c)):
                grid[r, int(c)] = int(color)
                return r, int(c)
    else:
        c = int(line_index)
        choices = list(range(0, r0 - 1)) if side == "T" else list(range(r1 + 2, h))
        np.random.shuffle(choices)
        for r in choices:
            if _is_available_singleton(grid, int(r), c):
                grid[int(r), c] = int(color)
                return int(r), c
    raise ValueError("failed to place an isolated outer pixel")


def _paint_expected_segment(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        out[r0, min(c0, c1) : max(c0, c1) + 1] = int(color)
    else:
        out[min(r0, r1) : max(r0, r1) + 1, c0] = int(color)


def _wall_for(coord: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[int, int]:
    axis, _side, _line_index, wall = _classify_pixel(coord[0], coord[1], box)
    assert axis in ("H", "V")
    return wall


def _expected_from_events(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    events: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for event in events:
        if event["kind"] == "same_pair":
            a, b = event["pixels"]
            _paint_expected_segment(out, a, b, event["color"])
        else:
            for coord, color in event["pixels"]:
                _paint_expected_segment(out, coord, _wall_for(coord, box), color)
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    box: tuple[int, int, int, int],
    events: list[dict],
) -> bool:
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False

    kinds = [event["kind"] for event in events]
    if kinds.count("same_pair") < 2 or kinds.count("mismatch_pair") < 2 or kinds.count("unpaired") < 2:
        return False

    r0, c0, r1, c1 = box
    for event in events:
        if event["kind"] == "same_pair":
            color = int(event["color"])
            a, b = event["pixels"]
            if a[0] == b[0]:
                row = a[0]
                if not np.all(expected[row, min(a[1], b[1]) : max(a[1], b[1]) + 1] == color):
                    return False
            else:
                col = a[1]
                if not np.all(expected[min(a[0], b[0]) : max(a[0], b[0]) + 1, col] == color):
                    return False
        else:
            for coord, color in event["pixels"]:
                axis, _side, line_index, wall = _classify_pixel(coord[0], coord[1], box)
                if int(expected[wall]) != int(color):
                    return False
                if axis == "H":
                    row = int(line_index)
                    if np.any(expected[row, c0 + 1 : c1] == int(color)):
                        return False
                else:
                    col = int(line_index)
                    if np.any(expected[r0 + 1 : r1, col] == int(color)):
                        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(20, 31))
    width = int(np.random.randint(20, 31))
    frame_h = int(np.random.randint(8, min(15, height - 5)))
    frame_w = int(np.random.randint(8, min(15, width - 5)))
    r0 = int(np.random.randint(3, height - frame_h - 2))
    c0 = int(np.random.randint(3, width - frame_w - 2))
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    box = (r0, c0, r1, c1)

    h_rows = _choose_spaced(range(r0 + 1, r1), 3)
    v_cols = _choose_spaced(range(c0 + 1, c1), 3)
    if h_rows is None or v_cols is None:
        return None

    colors = list(map(int, np.random.permutation(np.arange(1, 10))))
    frame_color = colors[0]
    penetrate_color = colors[1]
    h_mismatch_colors = (colors[2], colors[3])
    v_mismatch_colors = (colors[4], colors[5])
    h_unpaired_color = colors[6]
    v_unpaired_color = colors[7]

    grid = np.full((height, width), BACKGROUND, dtype=int)
    _draw_frame(grid, box, frame_color)

    events: list[dict] = []
    try:
        h_same = (
            _place_side_pixel(grid, box, "H", "L", h_rows[0], penetrate_color),
            _place_side_pixel(grid, box, "H", "R", h_rows[0], penetrate_color),
        )
        events.append({"kind": "same_pair", "axis": "H", "color": penetrate_color, "pixels": h_same})

        v_same = (
            _place_side_pixel(grid, box, "V", "T", v_cols[0], penetrate_color),
            _place_side_pixel(grid, box, "V", "B", v_cols[0], penetrate_color),
        )
        events.append({"kind": "same_pair", "axis": "V", "color": penetrate_color, "pixels": v_same})

        h_mismatch = (
            (_place_side_pixel(grid, box, "H", "L", h_rows[1], h_mismatch_colors[0]), h_mismatch_colors[0]),
            (_place_side_pixel(grid, box, "H", "R", h_rows[1], h_mismatch_colors[1]), h_mismatch_colors[1]),
        )
        events.append({"kind": "mismatch_pair", "axis": "H", "pixels": h_mismatch})

        v_mismatch = (
            (_place_side_pixel(grid, box, "V", "T", v_cols[1], v_mismatch_colors[0]), v_mismatch_colors[0]),
            (_place_side_pixel(grid, box, "V", "B", v_cols[1], v_mismatch_colors[1]), v_mismatch_colors[1]),
        )
        events.append({"kind": "mismatch_pair", "axis": "V", "pixels": v_mismatch})

        h_side = "L" if bool(np.random.randint(0, 2)) else "R"
        h_unpaired = ((_place_side_pixel(grid, box, "H", h_side, h_rows[2], h_unpaired_color), h_unpaired_color),)
        events.append({"kind": "unpaired", "axis": "H", "pixels": h_unpaired})

        v_side = "T" if bool(np.random.randint(0, 2)) else "B"
        v_unpaired = ((_place_side_pixel(grid, box, "V", v_side, v_cols[2], v_unpaired_color), v_unpaired_color),)
        events.append({"kind": "unpaired", "axis": "V", "pixels": v_unpaired})
    except ValueError:
        return None

    expected = _expected_from_events(grid, box, events)
    if not _semantic_checks(grid, expected, box, events):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-penetration input")
