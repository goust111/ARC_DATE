from __future__ import annotations

import numpy as np


BG = 0
STEP_BY_DIRECTION = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

# --------------- solver ---------------


def _find_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    labels = np.full((h, w), -1, dtype=int)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            comp_id = len(components)
            queue = [(r, c)]
            seen[r, c] = True
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []

            head = 0
            while head < len(queue):
                rr, cc = queue[head]
                head += 1
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "id": comp_id,
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )

    return components, labels


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return h >= 3 and w >= 3 and int(component["area"]) == h * w


def _arrow_hypotheses(component: dict) -> list[dict]:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = int(component["area"])
    local = component["coords"] - np.array([r0, c0], dtype=int)
    out: list[dict] = []

    if h == 2 and w >= 2 and area == w + 1:
        row_counts = [int(np.sum(local[:, 0] == row)) for row in range(2)]
        if sorted(row_counts) == [1, w]:
            full_row = 0 if row_counts[0] == w else 1
            other_row = 1 - full_row
            singleton_cols = local[local[:, 0] == other_row, 1]
            if singleton_cols.size == 1:
                sc = int(singleton_cols[0])
                tip_row = r0 + full_row
                if sc == 0:
                    out.append({"direction": "left", "length": w, "tip": (tip_row, c0)})
                if sc == w - 1:
                    out.append({"direction": "right", "length": w, "tip": (tip_row, c1)})

    if w == 2 and h >= 2 and area == h + 1:
        col_counts = [int(np.sum(local[:, 1] == col)) for col in range(2)]
        if sorted(col_counts) == [1, h]:
            full_col = 0 if col_counts[0] == h else 1
            other_col = 1 - full_col
            singleton_rows = local[local[:, 1] == other_col, 0]
            if singleton_rows.size == 1:
                sr = int(singleton_rows[0])
                tip_col = c0 + full_col
                if sr == 0:
                    out.append({"direction": "up", "length": h, "tip": (r0, tip_col)})
                if sr == h - 1:
                    out.append({"direction": "down", "length": h, "tip": (r1, tip_col)})

    dedup = {}
    for item in out:
        key = (item["direction"], int(item["length"]), tuple(item["tip"]))
        dedup[key] = item
    return [dedup[key] for key in sorted(dedup)]


def _trace_target(
    grid: np.ndarray,
    labels: np.ndarray,
    rectangles_by_id: dict[int, dict],
    tip: tuple[int, int],
    direction: str,
) -> int | None:
    h, w = map(int, grid.shape)
    dr, dc = STEP_BY_DIRECTION[direction]
    r = int(tip[0] + dr)
    c = int(tip[1] + dc)

    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG:
        r += dr
        c += dc

    if not (0 <= r < h and 0 <= c < w):
        return None

    comp_id = int(labels[r, c])
    rect = rectangles_by_id.get(comp_id)
    if rect is None:
        return None

    r0, c0, r1, c1 = rect["bbox"]
    if direction == "right" and c != c0:
        return None
    if direction == "left" and c != c1:
        return None
    if direction == "down" and r != r0:
        return None
    if direction == "up" and r != r1:
        return None
    return comp_id


def _assign_arrows(rectangles: list[dict], arrow_components: list[dict], choices_by_arrow: dict[int, list[dict]]) -> dict[int, dict]:
    rect_ids = {int(rect["id"]) for rect in rectangles}
    assert len(arrow_components) == len(rect_ids), "expected one arrow component per rectangle"

    ordered = sorted(arrow_components, key=lambda comp: (len(choices_by_arrow[comp["id"]]), comp["id"]))
    chosen: dict[int, dict] = {}
    used_rectangles: set[int] = set()

    def dfs(index: int) -> bool:
        if index == len(ordered):
            return used_rectangles == rect_ids

        arrow = ordered[index]
        for choice in choices_by_arrow[arrow["id"]]:
            target = int(choice["target"])
            if target in used_rectangles:
                continue
            chosen[arrow["id"]] = choice
            used_rectangles.add(target)
            if dfs(index + 1):
                return True
            used_rectangles.remove(target)
            del chosen[arrow["id"]]
        return False

    assert dfs(0), "arrow interpretations must resolve uniquely"
    return chosen


def _apply_crop(out: np.ndarray, bbox: tuple[int, int, int, int], direction: str, length: int) -> None:
    r0, c0, r1, c1 = bbox
    if direction == "right":
        out[r0 : r1 + 1, c0 : min(c1 + 1, c0 + length)] = BG
    elif direction == "left":
        out[r0 : r1 + 1, max(c0, c1 - length + 1) : c1 + 1] = BG
    elif direction == "down":
        out[r0 : min(r1 + 1, r0 + length), c0 : c1 + 1] = BG
    else:
        out[max(r0, r1 - length + 1) : r1 + 1, c0 : c1 + 1] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each arrow and crop the solid rectangle it points into by the arrow's shaft length.

    Input structure:
    - The background is black.
    - Foreground components are monochrome 4-connected objects.
    - Some components are solid axis-aligned rectangles of size at least 3x3.
    - Every other component is an arrow made from a horizontal or vertical shaft plus one perpendicular tip pixel at the rectangle-facing end.

    Rule:
    - Detect every solid rectangle and every arrow component.
    - For each arrow, infer which end of the shaft is the tip, use that to get the pointing direction, and read the shaft length L.
    - Follow the ray starting at the tip toward the pointed side to find the unique rectangle on that side, erase the arrow, and delete the first L rows or columns from the touched side of the rectangle.

    Key insight:
    - The extra perpendicular pixel marks which end of the line segment is the arrow tip, while the segment length itself is the crop depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components, labels = _find_components(g)
    rectangles = [comp for comp in components if _is_solid_rectangle(comp)]
    arrow_components = [comp for comp in components if not _is_solid_rectangle(comp)]
    assert rectangles, "expected at least one rectangle"
    assert arrow_components, "expected at least one arrow"

    rectangles_by_id = {int(rect["id"]): rect for rect in rectangles}
    choices_by_arrow: dict[int, list[dict]] = {}

    for component in arrow_components:
        candidates = []
        for hypothesis in _arrow_hypotheses(component):
            target = _trace_target(g, labels, rectangles_by_id, hypothesis["tip"], hypothesis["direction"])
            if target is None:
                continue
            candidates.append(
                {
                    "target": int(target),
                    "direction": hypothesis["direction"],
                    "length": int(hypothesis["length"]),
                }
            )
        dedup = {}
        for item in candidates:
            key = (item["target"], item["direction"], item["length"])
            dedup[key] = item
        choices = [dedup[key] for key in sorted(dedup)]
        assert choices, "every non-rectangle component must be a valid arrow"
        choices_by_arrow[int(component["id"])] = choices

    selected = _assign_arrows(rectangles, arrow_components, choices_by_arrow)
    out = g.copy()

    for component in arrow_components:
        rows = component["coords"][:, 0]
        cols = component["coords"][:, 1]
        out[rows, cols] = BG

    for arrow_id in sorted(selected):
        choice = selected[arrow_id]
        rect = rectangles_by_id[int(choice["target"])]
        _apply_crop(out, rect["bbox"], str(choice["direction"]), int(choice["length"]))

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
GEN_PLACEMENT_ATTEMPTS = 250
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_OBJECT_MIN = 2
GEN_OBJECT_MAX = 4
GEN_RECT_MIN = 3
GEN_RECT_MAX = 8
GEN_ARROW_CAP = 8
GEN_FULL_EXTRA = 2
GEN_GAP_VALUES = (1, 2, 3)
GEN_DIRECTIONS = ("left", "right", "up", "down")


def _bundle_geometry(height: int, width: int, top: int, left: int, spec: dict) -> dict | None:
    rect_h = int(spec["rect_h"])
    rect_w = int(spec["rect_w"])
    direction = str(spec["direction"])
    length = int(spec["length"])
    gap = int(spec["gap"])
    axis = int(spec["axis"])
    head_sign = int(spec["head_sign"])

    rect = {(r, c) for r in range(top, top + rect_h) for c in range(left, left + rect_w)}
    arrow: set[tuple[int, int]] = set()
    gap_cells: set[tuple[int, int]] = set()

    if direction == "right":
        shaft_r = top + axis
        tip_c = left - gap - 1
        arrow |= {(shaft_r, c) for c in range(tip_c - length + 1, tip_c + 1)}
        arrow.add((shaft_r + head_sign, tip_c))
        gap_cells |= {(shaft_r, c) for c in range(tip_c + 1, left)}
    elif direction == "left":
        shaft_r = top + axis
        tip_c = left + rect_w + gap
        arrow |= {(shaft_r, c) for c in range(tip_c, tip_c + length)}
        arrow.add((shaft_r + head_sign, tip_c))
        gap_cells |= {(shaft_r, c) for c in range(left + rect_w, tip_c)}
    elif direction == "down":
        shaft_c = left + axis
        tip_r = top - gap - 1
        arrow |= {(r, shaft_c) for r in range(tip_r - length + 1, tip_r + 1)}
        arrow.add((tip_r, shaft_c + head_sign))
        gap_cells |= {(r, shaft_c) for r in range(tip_r + 1, top)}
    else:
        shaft_c = left + axis
        tip_r = top + rect_h + gap
        arrow |= {(r, shaft_c) for r in range(tip_r, tip_r + length)}
        arrow.add((tip_r, shaft_c + head_sign))
        gap_cells |= {(r, shaft_c) for r in range(top + rect_h, tip_r)}

    occupied = rect | arrow
    footprint = occupied | gap_cells
    if not footprint:
        return None
    if any(not (0 <= r < height and 0 <= c < width) for r, c in footprint):
        return None
    if rect & arrow:
        return None

    padded: set[tuple[int, int]] = set()
    for r, c in footprint:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    padded.add((nr, nc))

    return {
        "rect_cells": rect,
        "arrow_cells": arrow,
        "gap_cells": gap_cells,
        "padded_cells": padded,
        "bbox": (top, left, top + rect_h - 1, left + rect_w - 1),
    }


def _sample_specs(rng: np.random.Generator, n_objects: int) -> list[dict]:
    directions = list(rng.permutation(np.array(GEN_DIRECTIONS, dtype=object))[:n_objects])
    full_flags = rng.integers(0, 2, size=n_objects).astype(bool)
    if n_objects > 1 and bool(np.all(full_flags == full_flags[0])):
        flip_index = int(rng.integers(0, n_objects))
        full_flags[flip_index] = ~full_flags[flip_index]

    palette = list(map(int, rng.permutation(np.arange(1, 10, dtype=int))))
    rect_colors = palette[:n_objects]
    arrow_colors = palette[n_objects : 2 * n_objects]

    specs: list[dict] = []
    for i in range(n_objects):
        rect_h = int(rng.integers(GEN_RECT_MIN, GEN_RECT_MAX + 1))
        rect_w = int(rng.integers(GEN_RECT_MIN, GEN_RECT_MAX + 1))
        direction = str(directions[i])
        size_along_ray = rect_w if direction in ("left", "right") else rect_h
        if full_flags[i]:
            lo = size_along_ray
            hi = min(GEN_ARROW_CAP, size_along_ray + GEN_FULL_EXTRA)
        else:
            lo = 2
            hi = size_along_ray - 1
        assert lo <= hi
        specs.append(
            {
                "rect_h": rect_h,
                "rect_w": rect_w,
                "direction": direction,
                "length": int(rng.integers(lo, hi + 1)),
                "gap": int(rng.choice(np.array(GEN_GAP_VALUES, dtype=int))),
                "axis": int(rng.integers(0, rect_h if direction in ("left", "right") else rect_w)),
                "head_sign": -1 if int(rng.integers(0, 2)) == 0 else 1,
                "rect_color": int(rect_colors[i]),
                "arrow_color": int(arrow_colors[i]),
            }
        )
    return specs


def _place_specs(height: int, width: int, specs: list[dict], rng: np.random.Generator) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []

    order = sorted(
        specs,
        key=lambda spec: -(int(spec["rect_h"]) * int(spec["rect_w"]) + int(spec["length"]) + int(spec["gap"])),
    )

    for spec in order:
        success = False
        for _ in range(GEN_PLACEMENT_ATTEMPTS):
            top = int(rng.integers(0, height - int(spec["rect_h"]) + 1))
            left = int(rng.integers(0, width - int(spec["rect_w"]) + 1))
            geom = _bundle_geometry(height, width, top, left, spec)
            if geom is None:
                continue
            if any(reserved[r, c] for r, c in geom["padded_cells"]):
                continue

            for r, c in geom["rect_cells"]:
                grid[r, c] = int(spec["rect_color"])
            for r, c in geom["arrow_cells"]:
                grid[r, c] = int(spec["arrow_color"])
            for r, c in geom["padded_cells"]:
                reserved[r, c] = True

            placed.append({**spec, **geom})
            success = True
            break

        if not success:
            return None

    return grid, placed


def _expected_output(inp: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in placed:
        for r, c in item["arrow_cells"]:
            out[r, c] = BG
    for item in placed:
        _apply_crop(out, tuple(item["bbox"]), str(item["direction"]), int(item["length"]))
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    width = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    n_objects = int(rng.integers(GEN_OBJECT_MIN, GEN_OBJECT_MAX + 1))
    specs = _sample_specs(rng, n_objects)
    placed = _place_specs(height, width, specs, rng)
    if placed is None:
        return None
    grid, layout = placed
    expected = _expected_output(grid, layout)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-cropping ARC input")
