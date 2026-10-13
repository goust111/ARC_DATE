from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SHADOW = 2
OBJECT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _bbox_dims(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _shift_box(
    box: tuple[int, int, int, int], dr: int, dc: int, distance: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        r0 + dr * distance,
        c0 + dc * distance,
        r1 + dr * distance,
        c1 + dc * distance,
    )


def _in_bounds(shape: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _fill_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            comps.append(
                {
                    "color": color,
                    "coords": np.asarray(cells, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(cells),
                }
            )
    return comps


def _solid_rect(grid: np.ndarray, comp: dict) -> dict | None:
    box = comp["bbox"]
    h, w = _bbox_dims(box)
    if int(comp["area"]) != h * w:
        return None
    if not np.all(_crop(grid, box) == int(comp["color"])):
        return None
    return {
        "color": int(comp["color"]),
        "bbox": box,
        "h": h,
        "w": w,
    }


def _distance_for(box: tuple[int, int, int, int], dr: int, dc: int) -> int:
    h, w = _bbox_dims(box)
    return h if dr != 0 else w


def _analysis_for_direction(grid: np.ndarray, dr: int, dc: int) -> dict | None:
    comps = _find_components(grid)
    red_rects: list[dict] = []
    object_rects: list[dict] = []
    singletons: list[dict] = []

    for comp in comps:
        color = int(comp["color"])
        if color == SHADOW:
            rect = _solid_rect(grid, comp)
            if rect is None:
                return None
            red_rects.append(rect)
            continue
        if int(comp["area"]) == 1:
            pos = tuple(int(x) for x in comp["coords"][0])
            singletons.append({"color": color, "pos": pos})
            continue
        rect = _solid_rect(grid, comp)
        if rect is None:
            return None
        object_rects.append(rect)

    if not red_rects or not object_rects:
        return None

    red_by_box = {rect["bbox"]: rect for rect in red_rects}
    singleton_by_pos = {item["pos"]: item for item in singletons}
    paired_red_boxes: set[tuple[int, int, int, int]] = set()
    complete_pairs: list[tuple[dict, dict]] = []
    object_only: list[dict] = []

    for obj in object_rects:
        shadow_box = _shift_box(obj["bbox"], dr, dc, _distance_for(obj["bbox"], dr, dc))
        if not _in_bounds(grid.shape, shadow_box):
            return None
        if shadow_box in red_by_box:
            complete_pairs.append((obj, red_by_box[shadow_box]))
            paired_red_boxes.add(shadow_box)
        else:
            if not np.all(_crop(grid, shadow_box) == BG):
                return None
            object_only.append(obj)

    lone_shadows: list[dict] = []
    used_residues: set[tuple[int, int]] = set()
    for shadow in red_rects:
        if shadow["bbox"] in paired_red_boxes:
            continue
        object_box = _shift_box(
            shadow["bbox"], -dr, -dc, _distance_for(shadow["bbox"], dr, dc)
        )
        if not _in_bounds(grid.shape, object_box):
            return None
        region = _crop(grid, object_box)
        non_bg = np.argwhere(region != BG)
        if non_bg.shape[0] != 1:
            return None
        rel_r, rel_c = (int(x) for x in non_bg[0])
        residue = (object_box[0] + rel_r, object_box[1] + rel_c)
        if residue in used_residues or residue not in singleton_by_pos:
            return None
        info = singleton_by_pos[residue]
        used_residues.add(residue)
        lone_shadows.append(
            {
                "shadow": shadow,
                "object_box": object_box,
                "color": int(info["color"]),
                "residue": residue,
            }
        )

    if len(used_residues) != len(singletons):
        return None
    if not complete_pairs or not object_only or not lone_shadows:
        return None

    return {
        "direction": (dr, dc),
        "complete_pairs": complete_pairs,
        "object_only": object_only,
        "lone_shadows": lone_shadows,
    }


def _pick_unique_analysis(grid: np.ndarray) -> dict:
    analyses = []
    for dr, dc in DIRS.values():
        parsed = _analysis_for_direction(grid, dr, dc)
        if parsed is not None:
            analyses.append(parsed)
    assert len(analyses) == 1, "scene must imply exactly one global shadow direction"
    return analyses[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the unseen shadow generator and use it to finish missing shadows and missing rectangles.

    Input structure:
    - A black-background grid contains several solid monochromatic rectangles in non-red colors.
    - Some rectangles already have a same-sized red shadow touching them after a uniform global translation direction.
    - Other rectangles have no shadow yet, and some red shadows have lost their rectangle, leaving only one original-color residue pixel inside the erased rectangle site.

    Rule:
    - Infer the single global shadow direction from the intact object-plus-shadow pairs.
    - For every rectangle without a shadow, draw a red shadow in that direction, offset by the rectangle's own size along that axis.
    - For every lone red shadow, reconstruct the same-sized original rectangle at the reverse offset; its color is given by the unique residue pixel inside that reconstructed area.
    - Keep intact pairs unchanged.

    Key insight:
    - The hidden generator is the common direction, but each rectangle travels by its own height or width, so intact pairs reveal a rule that must be applied both forward and backward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    analysis = _pick_unique_analysis(g)

    out = g.copy()
    for obj in analysis["object_only"]:
        shadow_box = _shift_box(
            obj["bbox"],
            analysis["direction"][0],
            analysis["direction"][1],
            _distance_for(obj["bbox"], *analysis["direction"]),
        )
        _fill_box(out, shadow_box, SHADOW)

    for item in analysis["lone_shadows"]:
        _fill_box(out, item["object_box"], item["color"])

    return out


def _size_along_direction(h: int, w: int, dr: int, dc: int) -> int:
    return h if dr != 0 else w


def _sample_categories() -> list[str]:
    total = random.randint(4, 6)
    kinds = ["complete", "complete", "object_only", "lone_shadow"]
    while len(kinds) < total:
        kinds.append(random.choice(("complete", "object_only", "lone_shadow")))
    random.shuffle(kinds)
    return kinds


def _sample_shapes(kinds: list[str], dr: int, dc: int) -> list[dict] | None:
    for _ in range(200):
        records = []
        all_spans = []
        complete_spans = []
        for kind in kinds:
            h = random.randint(2, 5)
            w = random.randint(2, 5)
            records.append({"kind": kind, "h": h, "w": w})
            span = _size_along_direction(h, w, dr, dc)
            all_spans.append(span)
            if kind == "complete":
                complete_spans.append(span)
        if len(set(all_spans)) < 2 or len(set(complete_spans)) < 2:
            continue
        return records
    return None


def _expanded_box(
    box: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _union_box(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _paint_expected_pair(grid: np.ndarray, record: dict) -> None:
    _fill_box(grid, record["object_box"], record["color"])
    _fill_box(grid, record["shadow_box"], SHADOW)


def _place_records(shape: tuple[int, int], records: list[dict], dr: int, dc: int) -> bool:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)

    random.shuffle(records)
    for record in records:
        obj_h = int(record["h"])
        obj_w = int(record["w"])
        span = _size_along_direction(obj_h, obj_w, dr, dc)
        placed = False

        for _ in range(400):
            r0 = random.randint(0, h - obj_h)
            c0 = random.randint(0, w - obj_w)
            object_box = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
            shadow_box = _shift_box(object_box, dr, dc, span)
            if not _in_bounds(shape, shadow_box):
                continue
            whole_box = _union_box(object_box, shadow_box)
            claim = _expanded_box(whole_box, shape, pad=1)
            if np.any(_crop(occupied.astype(int), claim)):
                continue
            _fill_box(occupied, claim, True)
            record["object_box"] = object_box
            record["shadow_box"] = shadow_box
            placed = True
            break

        if not placed:
            return False
    return True


def _build_input_and_output(records: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    expected = np.full(shape, BG, dtype=int)
    for record in records:
        _paint_expected_pair(expected, record)

    inp = expected.copy()
    for record in records:
        if record["kind"] == "complete":
            continue
        if record["kind"] == "object_only":
            _fill_box(inp, record["shadow_box"], BG)
            continue
        _fill_box(inp, record["object_box"], BG)
        r0, c0, r1, c1 = record["object_box"]
        rr = random.randint(r0, r1)
        cc = random.randint(c0, c1)
        inp[rr, cc] = int(record["color"])
        record["residue"] = (rr, cc)
    return inp, expected


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    chosen_dir: tuple[int, int],
    records: list[dict],
) -> bool:
    try:
        analysis = _pick_unique_analysis(inp)
        solved = main(inp)
    except AssertionError:
        return False

    if analysis["direction"] != chosen_dir:
        return False
    if not np.array_equal(solved, expected):
        return False

    expected_counts = {
        "complete": sum(record["kind"] == "complete" for record in records),
        "object_only": sum(record["kind"] == "object_only" for record in records),
        "lone_shadow": sum(record["kind"] == "lone_shadow" for record in records),
    }
    actual_counts = {
        "complete": len(analysis["complete_pairs"]),
        "object_only": len(analysis["object_only"]),
        "lone_shadow": len(analysis["lone_shadows"]),
    }
    if expected_counts != actual_counts:
        return False

    for item in analysis["lone_shadows"]:
        r, c = item["residue"]
        r0, c0, r1, c1 = item["object_box"]
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 26), random.randint(18, 26))
    dir_name = random.choice(tuple(DIRS))
    dr, dc = DIRS[dir_name]
    kinds = _sample_categories()
    records = _sample_shapes(kinds, dr, dc)
    if records is None:
        return None

    colors = random.sample(OBJECT_COLORS, len(records))
    for record, color in zip(records, colors):
        record["color"] = int(color)

    if not _place_records(shape, records, dr, dc):
        return None
    inp, expected = _build_input_and_output(records, shape)
    if not _validate_generated(inp, expected, (dr, dc), records):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-inversion input")
