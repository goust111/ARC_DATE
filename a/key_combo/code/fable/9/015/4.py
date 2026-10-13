from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GRID_HEIGHT_RANGE = (15, 24)
GRID_WIDTH_RANGE = (15, 24)
TOP_COUNT_OPTIONS = (2, 3)
LEFT_COUNT_OPTIONS = (2, 3)
TOP_HEIGHT_RANGE = (2, 5)
TOP_WIDTH_RANGE = (3, 6)
LEFT_HEIGHT_RANGE = (3, 6)
LEFT_WIDTH_RANGE = (2, 5)
GENERATION_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _solid_patch(color: int, h: int, w: int) -> np.ndarray:
    return np.full((int(h), int(w)), int(color), dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _control_is_isolated(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) != bg:
            return False
    return True


def _extract_items(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    top_items: list[dict] = []
    left_items: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        coords = np.argwhere(g == color)
        edge_mask = (coords[:, 0] == 0) | (coords[:, 1] == 0)
        edge_coords = coords[edge_mask]
        body_coords = coords[~edge_mask]

        assert len(edge_coords) == 1, "each color must have exactly one edge control pixel"
        assert len(body_coords) >= 1, "each color must have a rectangle body"
        control_r, control_c = map(int, edge_coords[0])
        assert (control_r == 0) ^ (control_c == 0), "control pixels cannot occupy the top-left corner"
        assert _control_is_isolated(g, control_r, control_c, bg), "control pixels must be isolated"

        r0, c0, r1, c1 = _bbox(body_coords)
        assert r0 > 0 and c0 > 0, "rectangle bodies must stay off the header edges"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "each body must be a solid same-color rectangle"
        assert int(body_coords.shape[0]) == patch.size, "body cells must exactly fill their bounding box"

        item = {
            "phase": "top" if control_r == 0 else "left",
            "control": int(control_c if control_r == 0 else control_r),
            "color": int(color),
            "patch": _solid_patch(int(color), *patch.shape),
            "h": int(patch.shape[0]),
            "w": int(patch.shape[1]),
        }
        if item["phase"] == "top":
            assert 0 < item["control"] < w
            top_items.append(item)
        else:
            assert 0 < item["control"] < h
            left_items.append(item)

    assert top_items and left_items, "expected both top-controlled and left-controlled rectangles"
    top_items.sort(key=lambda item: int(item["control"]))
    left_items.sort(key=lambda item: int(item["control"]))
    return bg, top_items, left_items


def _fits(out: np.ndarray, patch: np.ndarray, row: int, col: int, bg: int) -> bool:
    h, w = map(int, patch.shape)
    return bool(np.all(out[row : row + h, col : col + w] == bg))


def _drop_from_top(out: np.ndarray, item: dict, bg: int) -> tuple[int, int]:
    h, w = map(int, out.shape)
    ph, pw = map(int, item["patch"].shape)
    col = int(item["control"])
    assert 0 <= col <= w - pw, "top control column must fit the rectangle width"
    assert _fits(out, item["patch"], 0, col, bg), "rectangle must be able to enter from the top edge"

    row = 0
    while row + 1 <= h - ph and _fits(out, item["patch"], row + 1, col, bg):
        row += 1
    out[row : row + ph, col : col + pw] = item["patch"]
    return int(row), int(col)


def _slide_from_left(out: np.ndarray, item: dict, bg: int) -> tuple[int, int]:
    h, w = map(int, out.shape)
    ph, pw = map(int, item["patch"].shape)
    row = int(item["control"])
    assert 0 <= row <= h - ph, "left control row must fit the rectangle height"
    assert _fits(out, item["patch"], row, 0, bg), "rectangle must be able to enter from the left edge"

    col = 0
    while col + 1 <= w - pw and _fits(out, item["patch"], row, col + 1, bg):
        col += 1
    out[row : row + ph, col : col + pw] = item["patch"]
    return int(row), int(col)


def _assemble(
    shape: tuple[int, int],
    bg: int,
    top_items: list[dict],
    left_items: list[dict],
    *,
    phase_order: tuple[str, str] = ("top", "left"),
) -> tuple[np.ndarray, list[dict]]:
    out = np.full(shape, int(bg), dtype=int)
    placed: list[dict] = []
    groups = {"top": top_items, "left": left_items}

    for phase in phase_order:
        for item in groups[phase]:
            if phase == "top":
                row, col = _drop_from_top(out, item, bg)
            else:
                row, col = _slide_from_left(out, item, bg)
            placed.append(
                {
                    "phase": phase,
                    "control": int(item["control"]),
                    "color": int(item["color"]),
                    "row": int(row),
                    "col": int(col),
                    "h": int(item["h"]),
                    "w": int(item["w"]),
                }
            )
    return out, placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use top-edge controls to drop rectangles downward, then left-edge controls to slide rectangles rightward into the assembled stack.

    Input structure:
    - The background is black, and every non-background color appears exactly twice as one color class:
      one isolated control pixel on the top row or left column, plus one same-colored solid rectangle elsewhere.
    - Top-row controls specify vertically falling rectangles; left-column controls specify horizontally sliding rectangles.
    - Rectangle colors are unique, so each control pixel matches exactly one rectangle body of the same color.

    Rule:
    - Clear the whole canvas to the background.
    - Read top-row control pixels from left to right. For each one, align the matched rectangle's left edge to that control column and let the rigid rectangle fall from the top until the bottom border or an already assembled rectangle blocks it.
    - Then read left-column control pixels from top to bottom. For each one, align the matched rectangle's top edge to that control row and let the rigid rectangle slide in from the left until the right border or an already assembled rectangle blocks it.
    - Do not keep the control pixels in the output.

    Key insight:
    - The edge location of each singleton control chooses the gravity axis, and the global assembly order matters because earlier settled rectangles become supports or blockers for later ones.
    """
    g = np.asarray(grid, dtype=int)
    bg, top_items, left_items = _extract_items(g)
    out, _placed = _assemble(g.shape, bg, top_items, left_items)
    return out


def _sample_dims(count: int, hrange: tuple[int, int], wrange: tuple[int, int]) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for _ in range(count):
        dims.append(
            (
                random.randint(*hrange),
                random.randint(*wrange),
            )
        )
    return dims


def _sample_overlapping_positions(limit: int, spans: list[int]) -> list[int] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        positions = sorted(random.sample(range(1, limit - 1), len(spans)))
        if any(positions[i + 1] - positions[i] < 2 for i in range(len(spans) - 1)):
            continue
        if any(positions[i] + spans[i] > limit for i in range(len(spans))):
            continue
        if not all(positions[i + 1] < positions[i] + spans[i] for i in range(len(spans) - 1)):
            continue
        return [int(p) for p in positions]
    return None


def _paint_rect(grid: np.ndarray, row: int, col: int, h: int, w: int, color: int) -> None:
    grid[row : row + h, col : col + w] = int(color)


def _scatter_rectangles(
    shape: tuple[int, int],
    items: list[dict],
    control_marks: list[tuple[int, int, int]],
) -> np.ndarray | None:
    h, w = map(int, shape)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[0, :] = BACKGROUND
    grid[:, 0] = BACKGROUND

    for r, c, color in control_marks:
        grid[int(r), int(c)] = int(color)

    reserved = np.zeros((h, w), dtype=bool)
    reserved[0:2, :] = True
    reserved[:, 0:2] = True
    for r, c, _color in control_marks:
        reserved[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True

    order = items[:]
    random.shuffle(order)
    for item in order:
        ph = int(item["h"])
        pw = int(item["w"])
        if ph + 3 > h or pw + 3 > w:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS * 3):
            row = random.randint(2, h - ph)
            col = random.randint(2, w - pw)
            halo = reserved[row - 1 : row + ph + 1, col - 1 : col + pw + 1]
            if np.any(halo):
                continue
            _paint_rect(grid, row, col, ph, pw, int(item["color"]))
            reserved[row - 1 : row + ph + 1, col - 1 : col + pw + 1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _order_signature(placed: list[dict]) -> tuple[tuple[str, int, int], ...]:
    return tuple((str(p["phase"]), int(p["row"]), int(p["col"])) for p in placed)


def _try_assemble(
    shape: tuple[int, int],
    bg: int,
    top_items: list[dict],
    left_items: list[dict],
    *,
    phase_order: tuple[str, str] = ("top", "left"),
) -> tuple[np.ndarray, list[dict]] | None:
    try:
        return _assemble(shape, bg, top_items, left_items, phase_order=phase_order)
    except AssertionError:
        return None


def _semantics_ok(
    shape: tuple[int, int],
    bg: int,
    top_items: list[dict],
    left_items: list[dict],
    actual_out: np.ndarray,
    actual_placed: list[dict],
) -> bool:
    h, w = map(int, shape)
    top_blocked = any(p["phase"] == "top" and int(p["row"]) < h - int(p["h"]) for p in actual_placed)
    left_blocked = any(p["phase"] == "left" and int(p["col"]) < w - int(p["w"]) for p in actual_placed)
    if not (top_blocked and left_blocked):
        return False

    swapped = _try_assemble(shape, bg, top_items, left_items, phase_order=("left", "top"))
    if swapped is not None:
        swapped_out, swapped_placed = swapped
        if np.array_equal(actual_out, swapped_out):
            return False
        if _order_signature(actual_placed) == _order_signature(swapped_placed):
            return False

    reversed_top = _try_assemble(shape, bg, list(reversed(top_items)), left_items)
    if reversed_top is not None:
        reversed_top_out, reversed_top_placed = reversed_top
        if np.array_equal(actual_out, reversed_top_out):
            return False
        if _order_signature(actual_placed) == _order_signature(reversed_top_placed):
            return False

    reversed_left = _try_assemble(shape, bg, top_items, list(reversed(left_items)))
    if reversed_left is not None:
        reversed_left_out, reversed_left_placed = reversed_left
        if np.array_equal(actual_out, reversed_left_out):
            return False
        if _order_signature(actual_placed) == _order_signature(reversed_left_placed):
            return False

    if swapped is None and reversed_top is None and reversed_left is None:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    top_count = random.choice(TOP_COUNT_OPTIONS)
    left_count = random.choice(LEFT_COUNT_OPTIONS)
    total = top_count + left_count

    top_dims = _sample_dims(top_count, TOP_HEIGHT_RANGE, TOP_WIDTH_RANGE)
    left_dims = _sample_dims(left_count, LEFT_HEIGHT_RANGE, LEFT_WIDTH_RANGE)
    top_controls = _sample_overlapping_positions(w, [dim[1] for dim in top_dims])
    left_controls = _sample_overlapping_positions(h, [dim[0] for dim in left_dims])
    if top_controls is None or left_controls is None:
        return None

    colors = random.sample(PALETTE, total)
    top_items: list[dict] = []
    left_items: list[dict] = []
    for idx, (dims, control) in enumerate(zip(top_dims, top_controls)):
        hh, ww = map(int, dims)
        color = int(colors[idx])
        top_items.append(
            {
                "phase": "top",
                "control": int(control),
                "color": color,
                "patch": _solid_patch(color, hh, ww),
                "h": hh,
                "w": ww,
            }
        )
    for idx, (dims, control) in enumerate(zip(left_dims, left_controls), start=top_count):
        hh, ww = map(int, dims)
        color = int(colors[idx])
        left_items.append(
            {
                "phase": "left",
                "control": int(control),
                "color": color,
                "patch": _solid_patch(color, hh, ww),
                "h": hh,
                "w": ww,
            }
        )

    try:
        actual_out, actual_placed = _assemble((h, w), BACKGROUND, top_items, left_items)
    except AssertionError:
        return None

    if not _semantics_ok((h, w), BACKGROUND, top_items, left_items, actual_out, actual_placed):
        return None

    control_marks = [(0, int(item["control"]), int(item["color"])) for item in top_items]
    control_marks += [(int(item["control"]), 0, int(item["color"])) for item in left_items]
    inp = _scatter_rectangles((h, w), top_items + left_items, control_marks)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None

    if np.array_equal(inp, solved):
        return None
    if not np.array_equal(solved, actual_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-gravity assembly input")
