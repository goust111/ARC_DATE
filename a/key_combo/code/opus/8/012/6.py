from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, find_components


BACKGROUND_COLORS = (Color.BLUE, Color.GREEN, Color.GREY)
FOREGROUND_COLORS = (Color.RED, Color.ORANGE)
PERIOD_HEIGHTS = (2, 3, 4)
PERIOD_WIDTHS = (2, 3)


SHAPES = (
    np.array([[1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
)
STACK_SUPPORT_SHAPES = SHAPES[:3] + (SHAPES[5],)

GEN_ATTEMPTS = 400


def _inside(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _shape_coords(shape: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(shape > 0)
    return [(top + int(r), left + int(c)) for r, c in coords]


def _shape_footprint(shape: np.ndarray, which: str) -> list[int]:
    rows = np.argwhere(shape > 0)
    target_row = int(rows[:, 0].min() if which == "top" else rows[:, 0].max())
    cols = rows[rows[:, 0] == target_row][:, 1]
    return sorted(map(int, cols))


def _sort_objects(objects: list[dict], reverse: bool = True) -> list[dict]:
    if reverse:
        return sorted(objects, key=lambda obj: (-obj["bbox"][0], obj["bbox"][1]))
    return sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))


def _infer_period(
    grid: np.ndarray,
    field_box: tuple[int, int, int, int],
) -> tuple[int, int, dict[tuple[int, int], int]]:
    r0, c0, r1, c1 = field_box
    visible = [
        (r, c, int(grid[r, c]))
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if int(grid[r, c]) in BACKGROUND_COLORS
    ]
    assert visible

    best = None
    for ph in PERIOD_HEIGHTS:
        for pw in PERIOD_WIDTHS:
            classes: dict[tuple[int, int], int] = {}
            ok = True
            seen = set()
            for r, c, color in visible:
                key = (r % ph, c % pw)
                seen.add(key)
                prev = classes.get(key)
                if prev is None:
                    classes[key] = color
                elif prev != color:
                    ok = False
                    break
            if ok and len(seen) == ph * pw:
                candidate = (ph * pw, ph, pw, classes)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate

    assert best is not None, "failed to infer a complete minimal period"
    _area, ph, pw, classes = best
    return ph, pw, classes


def _repair_grid(
    grid: np.ndarray,
    field_box: tuple[int, int, int, int],
    classes: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = field_box
    ph = 1 + max(key[0] for key in classes)
    pw = 1 + max(key[1] for key in classes)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(out[r, c]) == Color.BLACK:
                out[r, c] = classes[(r % ph, c % pw)]
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for comp in find_components(
        np.asarray(grid, dtype=int),
        bg=Color.BLACK,
        colors=list(FOREGROUND_COLORS),
        connectivity=4,
        monochromatic=True,
    ):
        r0, c0, r1, c1 = comp["bbox"]
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        objects.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"]),
            }
        )
    return objects


def _under_color(
    r: int,
    c: int,
    field_box: tuple[int, int, int, int],
    classes: dict[tuple[int, int], int],
) -> int:
    if not _inside(field_box, r, c):
        return Color.BLACK
    ph = 1 + max(key[0] for key in classes)
    pw = 1 + max(key[1] for key in classes)
    return classes[(r % ph, c % pw)]


def _simulate(
    grid: np.ndarray,
    *,
    repair: bool,
    bottom_first: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    field_mask = np.isin(g, BACKGROUND_COLORS)
    field_box = bbox(field_mask)
    assert field_box is not None
    _ph, _pw, classes = _infer_period(g, field_box)

    base = g.copy()
    if repair:
        base = _repair_grid(base, field_box, classes)

    objects = _extract_objects(g)
    for obj in objects:
        for r, c in obj["coords"]:
            base[r, c] = _under_color(r, c, field_box, classes)

    scene = base.copy()
    ordered = _sort_objects(objects, reverse=bottom_first)
    for obj in ordered:
        coords = np.array(obj["coords"], dtype=int)
        while True:
            shifted = coords + np.array([1, 0], dtype=int)
            if int(shifted[:, 0].max()) >= scene.shape[0]:
                break
            if any(int(scene[r, c]) != Color.BLACK for r, c in shifted):
                break
            coords = shifted
        scene[coords[:, 0], coords[:, 1]] = obj["color"]
    return scene


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the lower periodic landing field, then drop each floating object onto the restored surface.

    Input structure:
    - The lower part of the grid contains one rectangular field made from a small blue/green/grey periodic tile, with black missing cells punched out of it.
    - The area outside that rectangle is black air.
    - Two or three small red/orange 4-connected objects float above the field.

    Rule:
    - Infer the field's minimal period from the visible non-black field cells and fill every black cell inside the field rectangle with its periodic color.
    - Treat the repaired field as solid ground while black cells outside it remain empty air.
    - Process the foreground objects from lower to higher source position; erase each source object, then drop it straight down until the next step would hit repaired field cells, an already landed object, or the bottom edge.

    Key insight:
    - The black gaps inside the periodic field are fake empty space: they must be repaired before gravity is applied, and lower objects can become landing platforms for higher ones.
    """
    return _simulate(np.asarray(grid, dtype=int), repair=True, bottom_first=True)


def _has_smaller_period(tile: np.ndarray, ph: int, pw: int) -> bool:
    for sh in range(1, ph + 1):
        for sw in range(1, pw + 1):
            if sh == ph and sw == pw:
                continue
            if ph % sh != 0 or pw % sw != 0:
                continue
            ok = True
            for r in range(ph):
                for c in range(pw):
                    if int(tile[r, c]) != int(tile[r % sh, c % sw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _random_tile(ph: int, pw: int) -> np.ndarray:
    for _ in range(200):
        palette = random.sample(BACKGROUND_COLORS, random.randint(2, 3))
        tile = np.array(
            [[random.choice(palette) for _ in range(pw)] for _ in range(ph)],
            dtype=int,
        )
        if len(set(map(int, tile.flatten()))) < 2:
            continue
        if _has_smaller_period(tile, ph, pw):
            continue
        return tile
    raise RuntimeError("failed to build an exact-period tile")


def _render_field(
    shape: tuple[int, int],
    field_box: tuple[int, int, int, int],
    tile: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = field_box
    ph, pw = tile.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = int(tile[r % ph, c % pw])
    return out


def _coords_disjoint(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    return not (set(a) & set(b))


def _too_close(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    aset = set(a)
    for r, c in b:
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in aset:
                return True
    return False


def _paint_object(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in _shape_coords(shape, top, left):
        grid[r, c] = color


def _choose_lifts(final_specs: list[dict], field_r0: int, height: int) -> list[int] | None:
    lifts = []
    for idx, spec in enumerate(final_specs):
        final_top = spec["final_top"]
        min_lift = 1
        max_lift = min(6, final_top)
        if max_lift < min_lift:
            return None
        choices = list(range(min_lift, max_lift + 1))
        random.shuffle(choices)
        chosen = None
        for lift in choices:
            input_top = final_top - lift
            if input_top >= height // 2:
                continue
            if spec["support"] == "object":
                lower = final_specs[spec["support_idx"]]
                if lift < lifts[spec["support_idx"]]:
                    continue
                lower_input_top = lower["final_top"] - lifts[spec["support_idx"]]
                if input_top + spec["shape"].shape[0] > lower_input_top:
                    continue
            chosen = lift
            break
        if chosen is None:
            return None
        lifts.append(chosen)
    return lifts


def _build_specs(
    field_box: tuple[int, int, int, int],
    n_objects: int,
) -> list[dict] | None:
    r0, c0, _r1, c1 = field_box
    field_w = c1 - c0 + 1
    specs: list[dict] = []

    stack_mode = n_objects >= 2 and random.random() < 0.6
    if stack_mode:
        lower_shape = random.choice(STACK_SUPPORT_SHAPES)
        top_width = len(_shape_footprint(lower_shape, "top"))
        upper_candidates = [shape for shape in SHAPES if shape.shape[1] <= top_width]
        upper_shape = random.choice(upper_candidates)
        lower_h, lower_w = lower_shape.shape
        upper_h, upper_w = upper_shape.shape
        lower_top = r0 - lower_h
        upper_top = lower_top - upper_h
        if upper_top < 0 or field_w < lower_w + 2:
            return None
        lower_left = random.randint(c0 + 1, c1 - lower_w)
        upper_left = random.randint(lower_left, lower_left + top_width - upper_w)
        lower_coords = _shape_coords(lower_shape, lower_top, lower_left)
        upper_coords = _shape_coords(upper_shape, upper_top, upper_left)
        if not _coords_disjoint(lower_coords, upper_coords):
            return None
        specs.append(
            {
                "shape": lower_shape,
                "color": random.choice(FOREGROUND_COLORS),
                "final_top": lower_top,
                "final_left": lower_left,
                "final_coords": lower_coords,
                "bbox": (lower_top, lower_left, lower_top + lower_h - 1, lower_left + lower_w - 1),
                "support": "field",
            }
        )
        specs.append(
            {
                "shape": upper_shape,
                "color": random.choice(FOREGROUND_COLORS),
                "final_top": upper_top,
                "final_left": upper_left,
                "final_coords": upper_coords,
                "bbox": (upper_top, upper_left, upper_top + upper_h - 1, upper_left + upper_w - 1),
                "support": "object",
                "support_idx": 0,
            }
        )

    attempts = 0
    while len(specs) < n_objects:
        attempts += 1
        if attempts > 80:
            return None
        shape = random.choice(SHAPES)
        h, w = shape.shape
        top = r0 - h
        if top < 0 or field_w < w + 2:
            return None
        left = random.randint(c0 + 1, c1 - w)
        coords = _shape_coords(shape, top, left)
        if any(_too_close(coords, spec["final_coords"]) for spec in specs):
            continue
        specs.append(
            {
                "shape": shape,
                "color": random.choice(FOREGROUND_COLORS),
                "final_top": top,
                "final_left": left,
                "final_coords": coords,
                "bbox": (top, left, top + h - 1, left + w - 1),
                "support": "field",
            }
        )

    if len({tuple(spec["bbox"]) for spec in specs}) != len(specs):
        return None
    return specs


def _support_holes(specs: list[dict], field_box: tuple[int, int, int, int]) -> tuple[set[tuple[int, int]], bool]:
    r0, _c0, r1, _c1 = field_box
    holes: set[tuple[int, int]] = set()
    stack_mode = any(spec["support"] == "object" for spec in specs)
    for spec in specs:
        if spec["support"] != "field":
            continue
        gap_len = random.randint(1, min(3, r1 - r0 + 1))
        bottom_cols = [spec["final_left"] + c for c in _shape_footprint(spec["shape"], "bottom")]
        for c in bottom_cols:
            for dr in range(gap_len):
                holes.add((r0 + dr, c))
    return holes, stack_mode


def _add_random_holes(
    field: np.ndarray,
    field_box: tuple[int, int, int, int],
    holes: set[tuple[int, int]],
    tile: np.ndarray,
) -> bool:
    r0, c0, r1, c1 = field_box
    ph, pw = tile.shape
    field_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    min_holes = (field_area * 15 + 99) // 100
    max_holes = (field_area * 30) // 100
    if not (min_holes <= len(holes) <= max_holes):
        target = random.randint(max(min_holes, len(holes)), max_holes)
    else:
        target = random.randint(len(holes), max_holes)

    residue_counts: dict[tuple[int, int], int] = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            key = (r % ph, c % pw)
            residue_counts[key] = residue_counts.get(key, 0) + 1
    for r, c in holes:
        key = (r % ph, c % pw)
        residue_counts[key] -= 1
        if residue_counts[key] == 0:
            return False

    candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in holes
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if len(holes) >= target:
            break
        key = (r % ph, c % pw)
        if residue_counts[key] <= 1:
            continue
        residue_counts[key] -= 1
        holes.add((r, c))

    if len(holes) < min_holes:
        return False
    for r, c in holes:
        field[r, c] = Color.BLACK
    return True


def _manual_output(
    repaired_field: np.ndarray,
    specs: list[dict],
) -> np.ndarray:
    out = repaired_field.copy()
    ordered = _sort_objects(
        [
            {
                "bbox": spec["input_bbox"],
                "coords": spec["input_coords"],
                "color": spec["color"],
            }
            for spec in specs
        ],
        reverse=True,
    )
    base = repaired_field.copy()
    for spec in specs:
        for r, c in spec["input_coords"]:
            base[r, c] = repaired_field[r, c]
    scene = base.copy()
    by_bbox = {tuple(spec["input_bbox"]): spec for spec in specs}
    for obj in ordered:
        spec = by_bbox[tuple(obj["bbox"])]
        coords = np.array(spec["input_coords"], dtype=int)
        while True:
            shifted = coords + np.array([1, 0], dtype=int)
            if int(shifted[:, 0].max()) >= scene.shape[0]:
                break
            if any(int(scene[r, c]) != Color.BLACK for r, c in shifted):
                break
            coords = shifted
        scene[coords[:, 0], coords[:, 1]] = spec["color"]
    return scene


def _attempt_case() -> np.ndarray | None:
    height = random.randint(14, 19)
    width = random.randint(11, 15)

    ph = random.choice(PERIOD_HEIGHTS)
    pw = random.choice(PERIOD_WIDTHS)
    tile = _random_tile(ph, pw)

    field_h = random.randint(max(ph + 3, 6), min(8, height - 4))
    field_w = random.randint(max(pw + 5, 8), width)
    field_r0 = random.randint(height - field_h - 2, height - field_h)
    field_c0 = random.randint(0, width - field_w)
    field_box = (field_r0, field_c0, field_r0 + field_h - 1, field_c0 + field_w - 1)

    repaired_field = _render_field((height, width), field_box, tile)
    n_objects = random.randint(2, 3)
    specs = _build_specs(field_box, n_objects)
    if specs is None:
        return None

    lifts = _choose_lifts(specs, field_r0, height)
    if lifts is None:
        return None
    for spec, lift in zip(specs, lifts):
        spec["lift"] = lift
        spec["input_top"] = spec["final_top"] - lift
        spec["input_left"] = spec["final_left"]
        spec["input_coords"] = _shape_coords(spec["shape"], spec["input_top"], spec["input_left"])
        h, w = spec["shape"].shape
        spec["input_bbox"] = (
            spec["input_top"],
            spec["input_left"],
            spec["input_top"] + h - 1,
            spec["input_left"] + w - 1,
        )
        if spec["input_top"] < 0:
            return None
        if spec["input_top"] + h > field_r0:
            return None

    for i, spec in enumerate(specs):
        for other in specs[i + 1 :]:
            if not _coords_disjoint(spec["input_coords"], other["input_coords"]):
                return None
            if _too_close(spec["input_coords"], other["input_coords"]):
                return None

    inp = repaired_field.copy()
    holes, stack_mode = _support_holes(specs, field_box)
    if not _add_random_holes(inp, field_box, holes, tile):
        return None

    for spec in specs:
        _paint_object(inp, spec["shape"], spec["input_top"], spec["input_left"], spec["color"])

    expected = _manual_output(repaired_field, specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    unrepaired = _simulate(inp, repair=False, bottom_first=True)
    if np.array_equal(unrepaired, actual):
        return None

    if stack_mode:
        wrong_order = _simulate(inp, repair=True, bottom_first=False)
        if np.array_equal(wrong_order, actual):
            return None

    field_mask = np.isin(inp, BACKGROUND_COLORS)
    inferred_box = bbox(field_mask)
    if inferred_box != field_box:
        return None
    iph, ipw, _classes = _infer_period(inp, field_box)
    if (iph, ipw) != (ph, pw):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-repair gravity input")
