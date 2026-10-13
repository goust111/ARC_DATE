from __future__ import annotations

from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_for_opus_2_035_5",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


ARM_INPUT_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARM_EXPANDED_OFFSETS = [(-2, 0), (-1, 0), (0, -2), (0, -1), (0, 1), (0, 2), (1, 0), (2, 0)]
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_RNG = np.random.default_rng()
GEN_ATTEMPTS = 400


def _shuffle(values) -> list[int]:
    items = list(values)
    GEN_RNG.shuffle(items)
    return [int(v) for v in items]


def _bbox_conflicts(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)

    overlaps = not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)
    if overlaps:
        return True

    rows_overlap = max(ar0, br0) <= min(ar1, br1)
    cols_overlap = max(ac0, bc0) <= min(ac1, bc1)
    if rows_overlap and (ac1 + 1 == bc0 or bc1 + 1 == ac0):
        return True
    if cols_overlap and (ar1 + 1 == br0 or br1 + 1 == ar0):
        return True
    return False


def _cross_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = map(int, center)
    return r - 1, c - 1, r + 1, c + 1


def _domino_bbox(orientation: str, axis: int, start: int) -> tuple[int, int, int, int]:
    axis = int(axis)
    start = int(start)
    if orientation == "h":
        return axis, start, axis, start + 1
    return start, axis, start + 1, axis


def _cross_intersects_row(cross: dict, row: int) -> bool:
    return abs(int(row) - int(cross["center"][0])) <= 2


def _cross_intersects_col(cross: dict, col: int) -> bool:
    return abs(int(col) - int(cross["center"][1])) <= 2


def _cross_covers(cross: dict, r: int, c: int) -> bool:
    cr, cc = map(int, cross["center"])
    return abs(int(r) - cr) <= 2 and abs(int(c) - cc) <= 2


def _paint_expanded_cross(out: np.ndarray, cross: dict) -> None:
    r, c = map(int, cross["center"])
    arm_color = int(cross["arm_color"])
    center_color = int(cross["center_color"])

    out[r, c] = center_color
    for dr, dc in DIAGONAL_OFFSETS:
        out[r + dr, c + dc] = center_color
    for dr, dc in ARM_EXPANDED_OFFSETS:
        out[r + dr, c + dc] = arm_color


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    objects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="objects",
    )
    assert objects, "expected foreground objects"

    crosses: list[dict] = []
    horizontals: list[dict] = []
    verticals: list[dict] = []

    for obj in objects:
        mask = obj != bg
        comp_bbox = bbox(mask)
        assert comp_bbox is not None
        r0, c0, r1, c1 = map(int, comp_bbox)
        coords = np.argwhere(mask)
        area = int(coords.shape[0])
        colors = sorted({int(v) for v in obj[mask]})

        if area == 2:
            assert len(colors) == 1, "domino marker must be monochrome"
            color = int(colors[0])
            if r0 == r1 and c1 == c0 + 1:
                horizontals.append({"row": r0, "color": color})
                continue
            if c0 == c1 and r1 == r0 + 1:
                verticals.append({"col": c0, "color": color})
                continue
            raise AssertionError("two-pixel object must be horizontal or vertical")

        if area == 5 and r1 == r0 + 2 and c1 == c0 + 2:
            cr, cc = r0 + 1, c0 + 1
            center_color = int(obj[cr, cc])
            assert center_color != bg, "cross center must be non-background"

            arm_colors = [int(obj[cr + dr, cc + dc]) for dr, dc in ARM_INPUT_OFFSETS]
            assert all(color != bg for color in arm_colors), "cross arms must be present"
            assert len(set(arm_colors)) == 1, "cross arms must share one color"
            arm_color = int(arm_colors[0])
            assert arm_color != center_color, "cross center and arms must differ"
            assert all(int(obj[r0 + dr, c0 + dc]) == bg for dr, dc in ((0, 0), (0, 2), (2, 0), (2, 2)))

            crosses.append(
                {
                    "center": (cr, cc),
                    "arm_color": arm_color,
                    "center_color": center_color,
                }
            )
            continue

        raise AssertionError("unexpected foreground object")

    assert crosses, "expected at least one cross"
    assert horizontals, "expected at least one horizontal domino"
    assert verticals, "expected at least one vertical domino"
    assert len({item["row"] for item in horizontals}) == len(horizontals), "horizontal rows must be unique"
    assert len({item["col"] for item in verticals}) == len(verticals), "vertical columns must be unique"
    return crosses, horizontals, verticals, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each two-color cross and extend each domino marker into a full bar with direction-dependent overlap rules.

    Input structure:
    - The background is black.
    - Each foreground object is an isolated 4-connected component.
    - Objects are either 3x3 two-color crosses or same-color two-pixel dominoes that are horizontal or vertical.

    Rule:
    - Turn every vertical domino into a full column bar of its color.
    - Expand every cross into a 5x5 footprint: the arm color fills the orthogonal cross out to distance 2, while the center color stays at the center and fills the four inner diagonals.
    - Turn every horizontal domino into a full row bar of its color.
    - These three layers are applied in that order, so columns lose to expanded crosses, and expanded crosses lose to rows.

    Key insight:
    - The conflict rule depends on ray orientation rather than a single global priority: horizontal bars cut through crosses, while vertical bars are blocked by them.
    """
    g = np.asarray(grid, dtype=int)
    crosses, horizontals, verticals, bg = _parse_scene(g)

    out = np.full_like(g, int(bg))
    for item in verticals:
        out[:, int(item["col"])] = int(item["color"])
    for cross in crosses:
        _paint_expanded_cross(out, cross)
    for item in horizontals:
        out[int(item["row"]), :] = int(item["color"])
    return out


def _pick_num_dominoes(num_crosses: int) -> int:
    max_dominoes = min(4, 9 - 2 * int(num_crosses))
    if max_dominoes == 2:
        return 2
    if max_dominoes == 3:
        return int(GEN_RNG.choice(np.array([2, 3])))
    return int(GEN_RNG.choice(np.array([2, 3, 4]), p=np.array([0.4, 0.4, 0.2])))


def _sample_crosses(height: int, width: int, num_crosses: int, colors: list[int]) -> list[dict]:
    slots = [
        {"rows": [3, 4], "cols": [3, 4]},
        {"rows": [height - 5, height - 4], "cols": [width - 5, width - 4]},
        {"rows": [height - 5, height - 4], "cols": [3, 4]},
    ]

    crosses = []
    for idx in range(int(num_crosses)):
        slot = slots[idx]
        center = (int(GEN_RNG.choice(np.array(slot["rows"]))), int(GEN_RNG.choice(np.array(slot["cols"]))))
        crosses.append(
            {
                "center": center,
                "arm_color": int(colors[2 * idx]),
                "center_color": int(colors[2 * idx + 1]),
                "bbox": _cross_bbox(center),
            }
        )
    return crosses


def _place_domino(
    orientation: str,
    axis: int,
    color: int,
    *,
    height: int,
    width: int,
    existing_bboxes: list[tuple[int, int, int, int]],
    used_rows: set[int],
    used_cols: set[int],
    side_preference: str | None = None,
) -> dict | None:
    axis = int(axis)
    color = int(color)

    if orientation == "h":
        if axis in used_rows:
            return None
        starts = list(range(width - 1))
        if side_preference == "right":
            starts = _shuffle([c for c in starts if c >= width // 2]) + _shuffle([c for c in starts if c < width // 2])
        elif side_preference == "left":
            starts = _shuffle([c for c in starts if c < width // 2]) + _shuffle([c for c in starts if c >= width // 2])
        else:
            starts = _shuffle(starts)
        for start in starts:
            comp_bbox = _domino_bbox("h", axis, start)
            if any(_bbox_conflicts(comp_bbox, other) for other in existing_bboxes):
                continue
            used_rows.add(axis)
            existing_bboxes.append(comp_bbox)
            return {"orientation": "h", "row": axis, "col": int(start), "color": color}
        return None

    if axis in used_cols:
        return None
    starts = list(range(height - 1))
    if side_preference == "top":
        starts = _shuffle([r for r in starts if r < height // 2]) + _shuffle([r for r in starts if r >= height // 2])
    elif side_preference == "bottom":
        starts = _shuffle([r for r in starts if r >= height // 2]) + _shuffle([r for r in starts if r < height // 2])
    else:
        starts = _shuffle(starts)
    for start in starts:
        comp_bbox = _domino_bbox("v", axis, start)
        if any(_bbox_conflicts(comp_bbox, other) for other in existing_bboxes):
            continue
        used_cols.add(axis)
        existing_bboxes.append(comp_bbox)
        return {"orientation": "v", "row": int(start), "col": axis, "color": color}
    return None


def _render_input(height: int, width: int, crosses: list[dict], dominoes: list[dict]) -> np.ndarray:
    grid = np.full((int(height), int(width)), Color.BLACK, dtype=int)

    for cross in crosses:
        r, c = map(int, cross["center"])
        grid[r, c] = int(cross["center_color"])
        for dr, dc in ARM_INPUT_OFFSETS:
            grid[r + dr, c + dc] = int(cross["arm_color"])

    for domino in dominoes:
        color = int(domino["color"])
        if domino["orientation"] == "h":
            r = int(domino["row"])
            c = int(domino["col"])
            grid[r, c : c + 2] = color
        else:
            r = int(domino["row"])
            c = int(domino["col"])
            grid[r : r + 2, c] = color

    return grid


def _validate_layout(crosses: list[dict], dominoes: list[dict], height: int, width: int) -> bool:
    if not (2 <= len(crosses) <= 3):
        return False
    if not (2 <= len(dominoes) <= 4):
        return False

    has_horizontal = any(item["orientation"] == "h" for item in dominoes)
    has_vertical = any(item["orientation"] == "v" for item in dominoes)
    if not (has_horizontal and has_vertical):
        return False

    for idx, first in enumerate(crosses):
        for second in crosses[idx + 1 :]:
            if _bbox_conflicts(first["bbox"], second["bbox"]):
                return False

    object_bboxes = [tuple(map(int, cross["bbox"])) for cross in crosses]
    for domino in dominoes:
        if domino["orientation"] == "h":
            comp_bbox = _domino_bbox("h", domino["row"], domino["col"])
        else:
            comp_bbox = _domino_bbox("v", domino["col"], domino["row"])
        for other in object_bboxes:
            if _bbox_conflicts(comp_bbox, other):
                return False
        object_bboxes.append(comp_bbox)

    if len({int(item["row"]) for item in dominoes if item["orientation"] == "h"}) != sum(item["orientation"] == "h" for item in dominoes):
        return False
    if len({int(item["col"]) for item in dominoes if item["orientation"] == "v"}) != sum(item["orientation"] == "v" for item in dominoes):
        return False

    if not any(_cross_intersects_row(crosses[0], item["row"]) for item in dominoes if item["orientation"] == "h"):
        return False
    if not any(_cross_intersects_col(crosses[1], item["col"]) for item in dominoes if item["orientation"] == "v"):
        return False

    if not any(
        item["orientation"] == "h" and any(_cross_intersects_row(cross, item["row"]) for cross in crosses)
        for item in dominoes
    ):
        return False
    if not any(
        item["orientation"] == "v" and any(_cross_intersects_col(cross, item["col"]) for cross in crosses)
        for item in dominoes
    ):
        return False

    horizontal_rows = [int(item["row"]) for item in dominoes if item["orientation"] == "h"]
    vertical_cols = [int(item["col"]) for item in dominoes if item["orientation"] == "v"]
    if not any(not any(_cross_covers(cross, row, col) for cross in crosses) for row in horizontal_rows for col in vertical_cols):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(GEN_RNG.integers(14, 19))
    width = int(GEN_RNG.integers(14, 19))
    num_crosses = int(GEN_RNG.choice(np.array([2, 3]), p=np.array([0.65, 0.35])))
    num_dominoes = _pick_num_dominoes(num_crosses)
    total_colors = 2 * num_crosses + num_dominoes
    palette = [int(v) for v in GEN_RNG.choice(np.arange(1, 10), size=total_colors, replace=False)]

    crosses = _sample_crosses(height, width, num_crosses, palette[: 2 * num_crosses])
    existing_bboxes = [tuple(map(int, cross["bbox"])) for cross in crosses]
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    domino_colors = palette[2 * num_crosses :]
    dominoes: list[dict] = []

    cross_a = crosses[0]
    cross_b = crosses[1]

    h_axis = int(GEN_RNG.choice(np.arange(cross_a["center"][0] - 2, cross_a["center"][0] + 3)))
    h_domino = _place_domino(
        "h",
        h_axis,
        domino_colors[0],
        height=height,
        width=width,
        existing_bboxes=existing_bboxes,
        used_rows=used_rows,
        used_cols=used_cols,
        side_preference="right",
    )
    if h_domino is None:
        return None
    dominoes.append(h_domino)

    v_axis = int(GEN_RNG.choice(np.arange(cross_b["center"][1] - 2, cross_b["center"][1] + 3)))
    v_domino = _place_domino(
        "v",
        v_axis,
        domino_colors[1],
        height=height,
        width=width,
        existing_bboxes=existing_bboxes,
        used_rows=used_rows,
        used_cols=used_cols,
        side_preference="top",
    )
    if v_domino is None:
        return None
    dominoes.append(v_domino)

    safe_rows = [r for r in range(height) if all(not _cross_intersects_row(cross, r) for cross in crosses)]
    safe_cols = [c for c in range(width) if all(not _cross_intersects_col(cross, c) for cross in crosses)]

    for color in domino_colors[2:]:
        placed = False
        for _ in range(30):
            orientation = str(GEN_RNG.choice(np.array(["h", "v"])))
            wants_interaction = bool(GEN_RNG.random() < 0.3)
            target_cross = crosses[int(GEN_RNG.integers(len(crosses)))]

            if orientation == "h":
                if wants_interaction:
                    axes = [r for r in range(target_cross["center"][0] - 2, target_cross["center"][0] + 3) if r not in used_rows]
                    side_preference = "right" if int(target_cross["center"][1]) < width // 2 else "left"
                else:
                    axes = [r for r in safe_rows if r not in used_rows]
                    side_preference = None
                if not axes:
                    continue
                axis = int(GEN_RNG.choice(np.array(axes)))
            else:
                if wants_interaction:
                    axes = [c for c in range(target_cross["center"][1] - 2, target_cross["center"][1] + 3) if c not in used_cols]
                    side_preference = "bottom" if int(target_cross["center"][0]) < height // 2 else "top"
                else:
                    axes = [c for c in safe_cols if c not in used_cols]
                    side_preference = None
                if not axes:
                    continue
                axis = int(GEN_RNG.choice(np.array(axes)))

            domino = _place_domino(
                orientation,
                axis,
                color,
                height=height,
                width=width,
                existing_bboxes=existing_bboxes,
                used_rows=used_rows,
                used_cols=used_cols,
                side_preference=side_preference,
            )
            if domino is not None:
                dominoes.append(domino)
                placed = True
                break

        if not placed:
            return None

    if not _validate_layout(crosses, dominoes, height, width):
        return None

    grid = _render_input(height, width, crosses, dominoes)

    try:
        parsed_crosses, parsed_h, parsed_v, bg = _parse_scene(grid)
        assert bg == Color.BLACK
        assert len(parsed_crosses) == len(crosses)
        assert len(parsed_h) == sum(item["orientation"] == "h" for item in dominoes)
        assert len(parsed_v) == sum(item["orientation"] == "v" for item in dominoes)
        out = main(grid)
        assert out.shape == grid.shape
        assert max(grid.shape) <= 30 and max(out.shape) <= 30
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for ARC task 5")
