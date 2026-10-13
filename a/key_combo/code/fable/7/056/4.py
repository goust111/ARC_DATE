from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_path = str(_ARC_UTILS_DIR)
    if arc_utils_path not in sys.path:
        sys.path.append(arc_utils_path)

try:
    from arc_utils import find_components, get_bg
except Exception:
    from collections import Counter, deque

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        if not monochromatic or return_format != "dict":
            raise ValueError("fallback find_components only supports monochromatic dict mode")
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        bg = int(bg)
        if colors is None:
            palette = sorted(set(map(int, g.ravel())) - {bg})
        else:
            palette = [int(c) for c in colors if int(c) != bg]
        if connectivity != 4:
            raise ValueError("fallback find_components only supports 4-connectivity")

        H, W = g.shape
        seen = np.zeros((H, W), dtype=bool)
        out: list[dict] = []
        for color in palette:
            coords = np.argwhere(g == color)
            for r, c in coords:
                rr = int(r)
                cc = int(c)
                if seen[rr, cc]:
                    continue
                q = deque([(rr, cc)])
                seen[rr, cc] = True
                comp: list[tuple[int, int]] = []
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                if len(comp) < int(min_area):
                    continue
                arr = np.asarray(comp, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": int(color),
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


GREY = 5
ALL_COLORS = list(range(10))
ODD_SIDES = (1, 3, 5, 7)
MAX_GEN_ATTEMPTS = 400
PLACEMENT_GAP = 1


def _bbox_from_center(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    r, c = center
    d = side // 2
    return int(r - d), int(c - d), int(r + d), int(c + d)


def _bbox_side(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _clear_with_gap(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 0) -> bool:
    return all(_boxes_separated(box, other, gap=gap) for other in others)


def _draw_square(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _is_solid_odd_square(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    side = r1 - r0 + 1
    if side != c1 - c0 + 1 or side % 2 == 0:
        return False
    if int(comp["area"]) != side * side:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))


def _square_info(comp: dict) -> dict:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    side = int(r1 - r0 + 1)
    return {
        "color": int(comp["color"]),
        "coords": np.asarray(comp["coords"], dtype=int),
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "center": ((r0 + r1) // 2, (c0 + c1) // 2),
    }


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_task(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    squares = [_square_info(comp) for comp in comps if _is_solid_odd_square(g, comp)]
    assert len(squares) >= 3, "expected at least the example pair and one extra source square"

    coord_to_square: dict[tuple[int, int], int] = {}
    for idx, sq in enumerate(squares):
        for r, c in sq["coords"]:
            coord_to_square[(int(r), int(c))] = idx

    connector_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for comp in comps:
        if int(comp["color"]) != GREY or int(comp["area"]) != 1:
            continue
        gr, gc = map(int, comp["coords"][0])
        adjacent = {
            coord_to_square[(nr, nc)]
            for nr, nc in _neighbors4(gr, gc)
            if (nr, nc) in coord_to_square
        }
        if len(adjacent) == 2:
            connector_candidates.append(((gr, gc), tuple(sorted(adjacent))))
    assert len(connector_candidates) == 1, "expected exactly one gray example connector"

    gray_coord, example_pair = connector_candidates[0]
    example_ids = set(example_pair)
    example_colors = {squares[idx]["color"] for idx in example_pair}
    assert len(example_colors) == 2, "example must connect two differently colored squares"

    other_ids = [idx for idx in range(len(squares)) if idx not in example_ids]
    other_colors = [squares[idx]["color"] for idx in other_ids]
    source_candidates = [color for color in example_colors if color in other_colors]
    assert len(source_candidates) == 1, "exactly one example color must reappear elsewhere"
    source_color = int(source_candidates[0])
    target_color = int((example_colors - {source_color}).pop())
    assert target_color not in other_colors, "target color must be unique to the example"

    example_source = next(idx for idx in example_pair if squares[idx]["color"] == source_color)
    example_target = next(idx for idx in example_pair if squares[idx]["color"] == target_color)
    source_side = int(squares[example_source]["side"])
    target_side = int(squares[example_target]["side"])
    assert source_side != target_side, "example must change the side length"

    source_ids = [idx for idx in other_ids if squares[idx]["color"] == source_color]
    assert source_ids, "need at least one non-example source square"
    return {
        "bg": bg,
        "squares": squares,
        "gray_coord": gray_coord,
        "example_ids": example_ids,
        "source_ids": source_ids,
        "source_color": source_color,
        "target_color": target_color,
        "source_side": source_side,
        "target_side": target_side,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the gray-linked square example and apply its recolor-plus-resize transform to the other source-colored squares.

    Input structure:
    - A dominant background contains several separated monochrome solid odd-sided squares.
    - Exactly one gray singleton pixel touches two squares and serves as the example pair.
    - Of the two example colors, only one appears again elsewhere in the grid.

    Rule:
    - Use the repeated example color as the source color and the other example color as the target color.
    - Read the source and target side lengths from the two example squares.
    - Erase the gray connector and both example squares, then replace every other source-colored square by a centered target-colored square of the target side length.
    - Leave all non-source distractor squares unchanged.

    Key insight:
    - The gray-touched pair is a legend that defines both a color substitution and a size substitution, and the repeated color outside the legend gives the transformation direction.
    """
    task = _extract_task(np.asarray(grid, dtype=int))
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(task["bg"])

    erase_ids = set(task["example_ids"]) | set(task["source_ids"])
    for idx in erase_ids:
        coords = task["squares"][idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    gr, gc = task["gray_coord"]
    out[gr, gc] = bg

    new_side = int(task["target_side"])
    new_color = int(task["target_color"])
    H, W = out.shape
    for idx in task["source_ids"]:
        cr, cc = task["squares"][idx]["center"]
        r0, c0, r1, c1 = _bbox_from_center((cr, cc), new_side)
        assert 0 <= r0 <= r1 < H and 0 <= c0 <= c1 < W, "transformed square must stay in bounds"
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(region == bg), "transformed squares must not overlap preserved objects"
        region[:, :] = new_color
    return out


def _example_boxes(
    rows: int,
    cols: int,
    source_side: int,
    target_side: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int]] | None:
    source_first = random.choice([False, True])
    orientation = random.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        left_side = source_side if source_first else target_side
        right_side = target_side if source_first else source_side
        row_margin = max(left_side, right_side) // 2
        row_choices = list(range(row_margin, rows - row_margin))
        col_choices = list(range(left_side, cols - right_side))
        if not row_choices or not col_choices:
            return None
        rg = random.choice(row_choices)
        cg = random.choice(col_choices)
        left_box = (rg - left_side // 2, cg - left_side, rg + left_side // 2, cg - 1)
        right_box = (rg - right_side // 2, cg + 1, rg + right_side // 2, cg + right_side)
        if source_first:
            return left_box, right_box, (rg, cg)
        return right_box, left_box, (rg, cg)

    top_side = source_side if source_first else target_side
    bottom_side = target_side if source_first else source_side
    col_margin = max(top_side, bottom_side) // 2
    row_choices = list(range(top_side, rows - bottom_side))
    col_choices = list(range(col_margin, cols - col_margin))
    if not row_choices or not col_choices:
        return None
    rg = random.choice(row_choices)
    cg = random.choice(col_choices)
    top_box = (rg - top_side, cg - top_side // 2, rg - 1, cg + top_side // 2)
    bottom_box = (rg + 1, cg - bottom_side // 2, rg + bottom_side, cg + bottom_side // 2)
    if source_first:
        return top_box, bottom_box, (rg, cg)
    return bottom_box, top_box, (rg, cg)


def _union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pixel: tuple[int, int]) -> tuple[int, int, int, int]:
    pr, pc = pixel
    return (
        min(a[0], b[0], pr),
        min(a[1], b[1], pc),
        max(a[2], b[2], pr),
        max(a[3], b[3], pc),
    )


def _place_centered_box(
    rows: int,
    cols: int,
    draw_side: int,
    reserve_side: int,
    reserved: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    margin = reserve_side // 2
    centers = [(r, c) for r in range(margin, rows - margin) for c in range(margin, cols - margin)]
    random.shuffle(centers)
    for center in centers:
        reserve_box = _bbox_from_center(center, reserve_side)
        if not _clear_with_gap(reserve_box, reserved, gap=PLACEMENT_GAP):
            continue
        return _bbox_from_center(center, draw_side), center
    return None


def _expected_output(
    rows: int,
    cols: int,
    bg: int,
    target_color: int,
    target_side: int,
    source_centers: list[tuple[int, int]],
    distractors: list[tuple[int, tuple[int, int, int, int]]],
) -> np.ndarray:
    out = np.full((rows, cols), int(bg), dtype=int)
    for color, box in distractors:
        _draw_square(out, color, box)
    for center in source_centers:
        _draw_square(out, target_color, _bbox_from_center(center, target_side))
    return out


def _validate_semantics(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        task = _extract_task(grid)
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if int(task["source_color"]) != int(meta["source_color"]):
        return False
    if int(task["target_color"]) != int(meta["target_color"]):
        return False
    if int(task["source_side"]) != int(meta["source_side"]):
        return False
    if int(task["target_side"]) != int(meta["target_side"]):
        return False
    if len(task["source_ids"]) != len(meta["source_centers"]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(18, 30)
    cols = random.randint(18, 30)
    bg = random.choice([color for color in ALL_COLORS if color != GREY])
    palette = [color for color in ALL_COLORS if color not in {bg, GREY}]
    source_color, target_color = random.sample(palette, 2)
    distractor_palette = [color for color in palette if color not in {source_color, target_color}]

    branch = random.choice(["grow", "shrink"])
    pairs = [(a, b) for a in ODD_SIDES for b in ODD_SIDES if a != b and ((b > a) == (branch == "grow"))]
    source_side, target_side = random.choice(pairs)
    max_side = max(source_side, target_side)

    source_count = random.randint(1, 4)
    extra_distractors = random.randint(0, 2)
    distractor_sides = [source_side, target_side]
    distractor_sides.extend(random.choice(ODD_SIDES) for _ in range(extra_distractors))
    random.shuffle(distractor_sides)

    example = _example_boxes(rows, cols, source_side, target_side)
    if example is None:
        return None
    example_source_box, example_target_box, gray_coord = example
    reserved = [_union_box(example_source_box, example_target_box, gray_coord)]

    source_boxes: list[tuple[int, int, int, int]] = []
    source_centers: list[tuple[int, int]] = []
    for _ in range(source_count):
        placed = _place_centered_box(rows, cols, source_side, max_side, reserved)
        if placed is None:
            return None
        box, center = placed
        source_boxes.append(box)
        source_centers.append(center)
        reserved.append(_bbox_from_center(center, max_side))

    distractors: list[tuple[int, tuple[int, int, int, int]]] = []
    size_n_count = 0
    size_m_count = 0
    for side in distractor_sides:
        placed = _place_centered_box(rows, cols, side, side, reserved)
        if placed is None:
            return None
        box, _center = placed
        color = random.choice(distractor_palette)
        distractors.append((color, box))
        reserved.append(box)
        if side == source_side:
            size_n_count += 1
        if side == target_side:
            size_m_count += 1

    if size_n_count == 0 or size_m_count == 0:
        return None

    inp = np.full((rows, cols), int(bg), dtype=int)
    _draw_square(inp, source_color, example_source_box)
    _draw_square(inp, target_color, example_target_box)
    gr, gc = gray_coord
    inp[gr, gc] = GREY
    for box in source_boxes:
        _draw_square(inp, source_color, box)
    for color, box in distractors:
        _draw_square(inp, color, box)

    expected = _expected_output(rows, cols, bg, target_color, target_side, source_centers, distractors)
    meta = {
        "source_color": source_color,
        "target_color": target_color,
        "source_side": source_side,
        "target_side": target_side,
        "source_centers": source_centers,
    }
    return inp if _validate_semantics(inp, expected, meta) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
