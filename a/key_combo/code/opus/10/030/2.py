import random
from typing import Iterable

import numpy as np


BG = 0
CONTROL = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
MAX_ATTEMPTS = 400


def _mask(rows: Iterable[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SYMMETRIC_MASKS = (
    _mask(("111", "111")),
    _mask(("1111", "1111")),
    _mask(("111", "010", "111")),
    _mask(("1111", "0110", "0110")),
    _mask(("11111", "00100", "00100")),
    _mask(("1001", "0110", "0110", "1001")),
)

ASYMMETRIC_MASKS = (
    _mask(("111", "110")),
    _mask(("111", "010", "110")),
    _mask(("1111", "0010", "0110")),
    _mask(("1110", "0100", "0110", "0011")),
    _mask(("11111", "00110", "00010")),
)


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _occupied_corners(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return [(r, c) for r, c in corners if mask[r, c]]


def _corner_shift(mask: np.ndarray, corner: tuple[int, int]) -> tuple[int, int]:
    h, w = mask.shape
    mapping = {
        (0, 0): (-1, -1),
        (0, w - 1): (-1, 1),
        (h - 1, 0): (1, -1),
        (h - 1, w - 1): (1, 1),
    }
    return mapping[corner]


def _transform_mask(mask: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = mask
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _global_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((top + int(r), left + int(c)))
    return cells


def _bbox_area(mask: np.ndarray) -> int:
    return int(mask.shape[0] * mask.shape[1])


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    objects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != BG
        colors = {int(patch[r - r0, c - c0]) for r, c in comp}
        assert CONTROL in colors, "each sprite must contain one orange control pixel"
        base_colors = sorted(colors - {CONTROL})
        assert len(base_colors) == 1, "each sprite must be otherwise monochrome"
        control_cells = np.argwhere(patch == CONTROL)
        assert len(control_cells) == 1, "expected one control pixel per sprite"
        control_local = tuple(int(v) for v in control_cells[0])
        h, w = mask.shape
        assert control_local in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}, "control must sit on a bbox corner"
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top": r0,
                "left": c0,
                "mask": mask,
                "color": base_colors[0],
                "control_local": control_local,
                "control_global": (r0 + control_local[0], c0 + control_local[1]),
                "bbox_area": _bbox_area(mask),
            }
        )
    return objects


def _is_lr_symmetric(mask: np.ndarray) -> bool:
    return np.array_equal(mask, np.fliplr(mask))


def _render_asymmetric(out: np.ndarray, obj: dict) -> None:
    body = _global_cells(obj["mask"], obj["top"], obj["left"])
    body.remove(obj["control_global"])
    for r, c in body:
        out[r, c] = obj["color"]
    halo: set[tuple[int, int]] = set()
    for r, c in body:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(out.shape, rr, cc) and (rr, cc) not in body:
                    halo.add((rr, cc))
    for r, c in halo:
        if out[r, c] == BG:
            out[r, c] = CONTROL


def _render_symmetric(out: np.ndarray, obj: dict) -> None:
    dr, dc = _corner_shift(obj["mask"], obj["control_local"])
    for r, c in _global_cells(obj["mask"], obj["top"], obj["left"]):
        rr = r + dr
        cc = c + dc
        if out[rr, cc] == BG:
            out[rr, cc] = obj["color"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Branch on each sprite's left-right symmetry: symmetric sprites shift diagonally while the lone asymmetric sprite gains an orange halo.

    Input structure:
    - The grid uses black (0) as background and contains 4-5 well-separated 8-connected sprites.
    - Each sprite is otherwise monochrome but has exactly one orange (7) control pixel replacing an occupied bbox corner.
    - All but one sprite have a tight-bbox foreground mask that is mirror-symmetric about the vertical axis when the control pixel is treated as ordinary foreground.

    Rule:
    - Split the non-black pixels into 8-connected sprites and test each sprite's tight-bbox foreground mask for left-right symmetry.
    - For each symmetric sprite, restore the control corner to the sprite color and translate the whole completed sprite by one cell along the diagonal indicated by that corner.
    - The unique asymmetric sprite does not move: its orange control pixel is deleted to black, and every black cell in the 8-neighborhood of any remaining sprite pixel becomes orange.
    - Place symmetric translated sprites in ascending bbox-area order, keeping earlier pixels whenever a later translated pixel would overlap them.

    Key insight:
    - Vertical symmetry is not the target object selector here; it is the branch condition that decides whether a sprite follows the motion path or the halo-marking path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _extract_objects(g)
    symmetric = [obj for obj in objects if _is_lr_symmetric(obj["mask"])]
    asymmetric = [obj for obj in objects if not _is_lr_symmetric(obj["mask"])]
    assert len(asymmetric) == 1, "expected exactly one asymmetric sprite"

    out = np.zeros_like(g)
    _render_asymmetric(out, asymmetric[0])
    for obj in sorted(symmetric, key=lambda item: (item["bbox_area"], item["bbox"][0], item["bbox"][1], item["color"])):
        _render_symmetric(out, obj)
    return out


def _make_object(mask: np.ndarray, color: int, kind: str) -> dict:
    corners = _occupied_corners(mask)
    if kind == "asym":
        valid_corners = []
        for corner in corners:
            trimmed = mask.copy()
            trimmed[corner] = False
            if trimmed.any() and _is_connected(trimmed):
                valid_corners.append(corner)
        corners = valid_corners
    assert corners, "template must allow a valid control corner"
    control = random.choice(corners)
    return {
        "kind": kind,
        "mask": mask,
        "color": color,
        "control_local": control,
        "bbox_area": _bbox_area(mask),
    }


def _input_bbox(obj: dict, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = obj["mask"].shape
    return top, left, top + h - 1, left + w - 1


def _bboxes_with_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + 2 < b[0] or b[2] + 2 < a[0] or a[3] + 2 < b[1] or b[3] + 2 < a[1]


def _can_place_bbox(bbox: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    return all(_bboxes_with_gap(bbox, other) for other in placed)


def _sym_output_cells(obj: dict, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = _corner_shift(obj["mask"], obj["control_local"])
    cells = set()
    for r, c in _global_cells(obj["mask"], top, left):
        rr = r + dr
        cc = c + dc
        if not _in_bounds(shape, rr, cc):
            return set()
        cells.add((rr, cc))
    return cells


def _asym_output_footprint(obj: dict, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    body = _global_cells(obj["mask"], top, left)
    control = (top + obj["control_local"][0], left + obj["control_local"][1])
    body.remove(control)
    footprint = set(body)
    for r, c in body:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(shape, rr, cc) and (rr, cc) not in body:
                    footprint.add((rr, cc))
    return footprint


def _paint_input(grid: np.ndarray, obj: dict, top: int, left: int) -> None:
    for r, c in np.argwhere(obj["mask"]):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = obj["color"]
    cr = top + obj["control_local"][0]
    cc = left + obj["control_local"][1]
    grid[cr, cc] = CONTROL


def _expected_from_blueprint(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    asym = [obj for obj in objects if obj["kind"] == "asym"]
    assert len(asym) == 1
    asym_obj = asym[0]
    body = _global_cells(asym_obj["mask"], asym_obj["top"], asym_obj["left"])
    control = (asym_obj["top"] + asym_obj["control_local"][0], asym_obj["left"] + asym_obj["control_local"][1])
    body.remove(control)
    for r, c in body:
        out[r, c] = asym_obj["color"]
    for r, c in body:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(shape, rr, cc) and (rr, cc) not in body and out[rr, cc] == BG:
                    out[rr, cc] = CONTROL

    symmetric = sorted(
        [obj for obj in objects if obj["kind"] == "sym"],
        key=lambda item: (item["bbox_area"], item["top"], item["left"], item["color"]),
    )
    for obj in symmetric:
        for r, c in _sym_output_cells(obj, obj["top"], obj["left"], shape):
            if out[r, c] == BG:
                out[r, c] = obj["color"]
    return out


def _choose_symmetric_objects(count: int, colors: list[int]) -> list[dict]:
    area_to_masks: dict[int, list[np.ndarray]] = {}
    for mask in SYMMETRIC_MASKS:
        area_to_masks.setdefault(_bbox_area(mask), []).append(mask)
    chosen_areas = random.sample(sorted(area_to_masks), count)
    objects = []
    for color, area in zip(colors, chosen_areas):
        mask = random.choice(area_to_masks[area])
        flip_ud = random.random() < 0.5
        transformed = _transform_mask(mask, flip_ud=flip_ud, flip_lr=False)
        objects.append(_make_object(transformed, color, "sym"))
    return objects


def _choose_asymmetric_object(color: int) -> dict:
    for _ in range(50):
        mask = random.choice(ASYMMETRIC_MASKS)
        transformed = _transform_mask(
            mask,
            flip_ud=random.random() < 0.5,
            flip_lr=random.random() < 0.5,
        )
        if not _is_lr_symmetric(transformed):
            return _make_object(transformed, color, "asym")
    raise RuntimeError("failed to choose an asymmetric template")


def _place_objects(shape: tuple[int, int], objects: list[dict]) -> list[dict] | None:
    h, w = shape
    placed_bboxes: list[tuple[int, int, int, int]] = []
    output_footprints: list[set[tuple[int, int]]] = []
    placed: list[dict] = []

    ordered = sorted(objects, key=lambda obj: (-len(_global_cells(obj["mask"], 0, 0)), -obj["bbox_area"]))
    for obj in ordered:
        mh, mw = obj["mask"].shape
        if obj["kind"] == "asym":
            row_range = range(1, h - mh)
            col_range = range(1, w - mw)
        else:
            dr, dc = _corner_shift(obj["mask"], obj["control_local"])
            row_min = 1 if dr == -1 else 0
            row_max = h - mh - (1 if dr == 1 else 0)
            col_min = 1 if dc == -1 else 0
            col_max = w - mw - (1 if dc == 1 else 0)
            row_range = range(row_min, row_max + 1)
            col_range = range(col_min, col_max + 1)

        candidates = [(r, c) for r in row_range for c in col_range]
        random.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            bbox = _input_bbox(obj, top, left)
            if not _can_place_bbox(bbox, placed_bboxes):
                continue
            if obj["kind"] == "asym":
                footprint = _asym_output_footprint(obj, top, left, shape)
            else:
                footprint = _sym_output_cells(obj, top, left, shape)
            if not footprint:
                continue
            if any(footprint & other for other in output_footprints):
                continue
            chosen = (top, left, bbox, footprint)
            break
        if chosen is None:
            return None

        top, left, bbox, footprint = chosen
        placed_obj = dict(obj)
        placed_obj["top"] = top
        placed_obj["left"] = left
        placed_obj["bbox"] = bbox
        placed_obj["control_global"] = (top + obj["control_local"][0], left + obj["control_local"][1])
        placed.append(placed_obj)
        placed_bboxes.append(bbox)
        output_footprints.append(footprint)
    return placed


def _validate_blueprint(grid: np.ndarray, objects: list[dict]) -> None:
    parsed = _extract_objects(grid)
    assert len(parsed) == len(objects)
    parsed_asym = sum(not _is_lr_symmetric(obj["mask"]) for obj in parsed)
    assert parsed_asym == 1, "generator must create exactly one asymmetric sprite"

    bboxes = [obj["bbox"] for obj in parsed]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            assert _bboxes_with_gap(bboxes[i], bboxes[j]), "input bboxes must be separated by at least two cells"

    sym_areas = [obj["bbox_area"] for obj in parsed if _is_lr_symmetric(obj["mask"])]
    assert len(sym_areas) == len(set(sym_areas)), "symmetric bbox areas must be unique"

    output_footprints: list[set[tuple[int, int]]] = []
    for obj in objects:
        if obj["kind"] == "asym":
            footprint = _asym_output_footprint(obj, obj["top"], obj["left"], grid.shape)
            body = _global_cells(obj["mask"], obj["top"], obj["left"])
            body.remove(obj["control_global"])
            assert footprint - body, "asymmetric sprite should produce a visible halo"
        else:
            footprint = _sym_output_cells(obj, obj["top"], obj["left"], grid.shape)
            assert footprint, "symmetric shift must stay in bounds"
        assert all(_in_bounds(grid.shape, r, c) for r, c in footprint)
        assert not any(footprint & other for other in output_footprints), "generated outputs must not overlap"
        output_footprints.append(footprint)

    expected = _expected_from_blueprint(grid.shape, objects)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output disagrees with generator semantics"


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        height = random.randint(17, 20)
        width = random.randint(17, 20)
        sym_count = random.randint(3, 4)
        colors = random.sample(PALETTE, sym_count + 1)

        objects = _choose_symmetric_objects(sym_count, colors[:sym_count])
        objects.append(_choose_asymmetric_object(colors[-1]))
        placed = _place_objects((height, width), objects)
        if placed is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        for obj in placed:
            _paint_input(grid, obj, obj["top"], obj["left"])

        _validate_blueprint(grid, placed)
        return grid
    raise RuntimeError("failed to generate a valid input")
