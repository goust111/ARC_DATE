from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components


BG = 0
GRAY = 5
ORANGE = 7
OBJECT_COLORS = [1, 2, 3, 4, 6, 8, 9]
MAX_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter colored objects by top-row markers, then emit aspect-ratio-coded rays through a periodic frame mask.

    Input structure:
    - Black (0) background with a top-row indicator bar made of 2 or 3 isolated non-black marker pixels.
    - A single gray (5) one-pixel rectangular frame sits in the lower-middle region; inside it, orange (7) stripes repeat periodically across some interior rows and sometimes also some interior columns.
    - Outside the frame are several small 4-connected monochrome objects whose colors mix marker colors and non-marker colors; every object's bounding box is strictly wider than tall or strictly taller than wide.

    Rule:
    - Read the non-black colors on the top row as the legal color set.
    - Erase every outside object whose color is not legal, while leaving the gray frame and orange stripes unchanged.
    - For each surviving object, use its bounding-box aspect ratio to choose a ray direction: width > height emits a horizontal ray from the bbox center row, height > width emits a vertical ray from the bbox center column.
    - Infer the row period from the orange horizontal stripes inside the gray frame and extend that congruence class to the whole grid; infer a column period analogously from orange vertical stripes if they exist.
    - Draw each ray only where its corresponding periodic mask is active, never overwriting any existing non-background pixel.

    Key insight:
    - Three independent cues must be combined: top-row markers decide which objects survive, each survivor's bbox aspect ratio decides whether it emits horizontally or vertically, and the frame's internal stripe periodicity decides whether that ray is visible at all.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    legal_colors = {int(v) for v in g[0, :] if int(v) != BG}
    frame_bbox = _find_frame_bbox(g)
    active_rows, active_cols = _infer_periodic_masks(g, frame_bbox)

    object_scene = g.copy()
    object_scene[0, :] = BG
    object_scene[object_scene == GRAY] = BG
    object_scene[object_scene == ORANGE] = BG

    components = find_components(
        object_scene,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    for comp in components:
        color = int(comp["color"])
        if color in legal_colors:
            continue
        coords = np.asarray(comp["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BG

    filtered = out.copy()
    filtered[0, :] = BG
    filtered[filtered == GRAY] = BG
    filtered[filtered == ORANGE] = BG
    survivors = find_components(
        filtered,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    survivors = sorted(survivors, key=lambda obj: tuple(map(int, obj["bbox"])))

    for comp in survivors:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height != width, "emitter bbox must be non-square"

        if width > height:
            row = r0 + (height - 1) // 2
            if row not in active_rows:
                continue
            for c in range(w):
                if out[row, c] == BG:
                    out[row, c] = color
        else:
            col = c0 + (width - 1) // 2
            if active_cols is not None and col not in active_cols:
                continue
            for r in range(h):
                if out[r, col] == BG:
                    out[r, col] = color

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid periodic-mask ray puzzle")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 24)
    w = random.randint(20, 24)
    grid = np.full((h, w), BG, dtype=int)

    frame_h = random.randint(8, 10)
    frame_w = random.randint(8, 10)

    row_low = 7
    row_high = h - frame_h - 5
    col_low = 4
    col_high = w - frame_w - 4
    if row_low > row_high or col_low > col_high:
        return None

    center_r = h // 2 + random.randint(0, 2)
    center_c = w // 2 + random.randint(-1, 1)
    fr0 = min(max(center_r - frame_h // 2, row_low), row_high)
    fc0 = min(max(center_c - frame_w // 2, col_low), col_high)
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    _draw_frame(grid, fr0, fc0, fr1, fc1)

    interior_h = frame_h - 2
    interior_w = frame_w - 2

    row_period, row_phase = _choose_period_and_phase(interior_h)
    active_row_locals = [r for r in range(interior_h) if r % row_period == row_phase]
    if len(active_row_locals) < 2:
        return None
    for rr in active_row_locals:
        grid[fr0 + 1 + rr, fc0 + 1 : fc1] = ORANGE

    has_vertical_mask = random.random() < 0.55
    active_col_locals: list[int] = []
    col_period = None
    col_phase = None
    if has_vertical_mask:
        col_period, col_phase = _choose_period_and_phase(interior_w)
        active_col_locals = [c for c in range(interior_w) if c % col_period == col_phase]
        if len(active_col_locals) < 2:
            return None
        for cc in active_col_locals:
            grid[fr0 + 1 : fr1, fc0 + 1 + cc] = ORANGE

    marker_count = 3 if random.random() < 0.65 else 2
    marker_colors = random.sample(OBJECT_COLORS, marker_count)
    repeated_color = marker_colors[0]
    vertical_color_a = marker_colors[1]
    vertical_color_b = marker_colors[2] if marker_count == 3 else marker_colors[1]

    invalid_choices = [c for c in OBJECT_COLORS if c not in marker_colors]
    if not invalid_choices:
        return None
    invalid_color = random.choice(invalid_choices)

    _place_markers(grid, marker_colors)

    active_rows = [fr0 + 1 + rr for rr in active_row_locals]
    inactive_rows = [r for r in range(fr0 + 1, fr1) if r not in active_rows]
    if not inactive_rows:
        return None

    if has_vertical_mask:
        active_cols = [fc0 + 1 + cc for cc in active_col_locals]
        inactive_cols = [c for c in range(fc0 + 1, fc1) if c not in active_cols]
        if not inactive_cols:
            return None
    else:
        active_cols = list(range(fc0 + 1, fc1))
        inactive_cols = []

    specs = [
        {
            "name": "survivor_h_on",
            "orientation": "H",
            "color": repeated_color,
            "valid": True,
            "visible": True,
            "allowed_rows": [active_rows[0]],
        },
        {
            "name": "survivor_h_off",
            "orientation": "H",
            "color": repeated_color,
            "valid": True,
            "visible": False,
            "allowed_rows": inactive_rows,
        },
        {
            "name": "survivor_v_main",
            "orientation": "V",
            "color": vertical_color_a,
            "valid": True,
            "visible": True,
            "allowed_cols": active_cols,
        },
        {
            "name": "invalid_h_on",
            "orientation": "H",
            "color": invalid_color,
            "valid": False,
            "visible": True,
            "allowed_rows": active_rows[1:],
        },
    ]

    if has_vertical_mask:
        specs.append(
            {
                "name": "survivor_v_off",
                "orientation": "V",
                "color": vertical_color_b,
                "valid": True,
                "visible": False,
                "allowed_cols": inactive_cols,
            }
        )
    else:
        specs.append(
            {
                "name": "survivor_v_extra",
                "orientation": "V",
                "color": vertical_color_b,
                "valid": True,
                "visible": True,
                "allowed_cols": active_cols[1:] if len(active_cols) > 1 else active_cols,
            }
        )

    placed_objects = []
    for spec in specs:
        mask = _make_object_mask(spec["orientation"])
        placed = _place_object(grid, mask, spec, (fr0, fc0, fr1, fc1))
        if placed is None:
            return None
        placed_objects.append(placed)

    if random.random() < 0.45:
        extra_invalid = {
            "name": "invalid_extra",
            "orientation": random.choice(["H", "V"]),
            "color": random.choice(invalid_choices),
            "valid": False,
            "visible": bool(random.getrandbits(1)),
        }
        extra_invalid["allowed_rows"] = active_rows if extra_invalid["orientation"] == "H" and extra_invalid["visible"] else inactive_rows
        extra_invalid["allowed_cols"] = active_cols if extra_invalid["orientation"] == "V" and extra_invalid["visible"] else (inactive_cols if inactive_cols else active_cols)
        mask = _make_object_mask(extra_invalid["orientation"])
        placed = _place_object(grid, mask, extra_invalid, (fr0, fc0, fr1, fc1))
        if placed is not None:
            placed_objects.append(placed)

    meta = {
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "row_period": row_period,
        "row_phase_global": active_rows[0] % row_period,
        "col_period": col_period,
        "col_phase_global": active_cols[0] % col_period if has_vertical_mask else None,
        "has_vertical_mask": has_vertical_mask,
        "objects": placed_objects,
        "marker_colors": marker_colors,
    }

    if not _validate_generated_case(grid, meta):
        return None
    return grid


def _validate_generated_case(grid: np.ndarray, meta: dict) -> bool:
    marker_colors = set(meta["marker_colors"])
    valid_objects = [obj for obj in meta["objects"] if obj["valid"]]
    invalid_objects = [obj for obj in meta["objects"] if not obj["valid"]]

    if len(valid_objects) < 4 or not invalid_objects:
        return False
    if len({obj["orientation"] for obj in valid_objects}) < 2:
        return False

    valid_color_counts: dict[int, int] = {}
    for obj in valid_objects:
        valid_color_counts[obj["color"]] = valid_color_counts.get(obj["color"], 0) + 1
    if max(valid_color_counts.values()) < 2:
        return False

    if any(obj["color"] not in marker_colors for obj in valid_objects):
        return False
    if all(obj["color"] in marker_colors for obj in meta["objects"]):
        return False

    h_visible = [obj for obj in valid_objects if obj["orientation"] == "H" and obj["visible"]]
    h_hidden = [obj for obj in valid_objects if obj["orientation"] == "H" and not obj["visible"]]
    v_visible = [obj for obj in valid_objects if obj["orientation"] == "V" and obj["visible"]]
    v_hidden = [obj for obj in valid_objects if obj["orientation"] == "V" and not obj["visible"]]
    if not h_visible or not h_hidden or not v_visible:
        return False
    if meta["has_vertical_mask"] and not v_hidden:
        return False

    expected = _render_from_meta(grid, meta)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(actual, grid):
        return False

    erased = grid.copy()
    for obj in invalid_objects:
        erased[obj["coords"][:, 0], obj["coords"][:, 1]] = BG

    painted_sets = []
    for obj in valid_objects:
        cells = _drawable_cells(erased, obj)
        if obj["visible"] and len(cells) < 3:
            return False
        if (not obj["visible"]) and cells:
            return False
        if obj["visible"]:
            painted_sets.append(cells)
    for i in range(len(painted_sets)):
        for j in range(i + 1, len(painted_sets)):
            if painted_sets[i] & painted_sets[j]:
                return False

    if not any(_hypothetical_visible_cells(grid, obj) for obj in invalid_objects):
        return False

    changes = int(np.count_nonzero(actual != grid))
    return changes >= 8


def _render_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    valid_objects = []
    for obj in meta["objects"]:
        if not obj["valid"]:
            out[obj["coords"][:, 0], obj["coords"][:, 1]] = BG
        else:
            valid_objects.append(obj)

    for obj in sorted(valid_objects, key=lambda item: tuple(map(int, item["bbox"]))):
        for r, c in _drawable_cells(out, obj):
            out[r, c] = obj["color"]
    return out


def _drawable_cells(scene: np.ndarray, obj: dict) -> set[tuple[int, int]]:
    if not obj["visible"]:
        return set()
    h, w = scene.shape
    r0, c0, r1, c1 = map(int, obj["bbox"])
    cells: set[tuple[int, int]] = set()
    if obj["orientation"] == "H":
        row = r0 + (r1 - r0) // 2
        for c in range(w):
            if int(scene[row, c]) == BG:
                cells.add((row, c))
    else:
        col = c0 + (c1 - c0) // 2
        for r in range(h):
            if int(scene[r, col]) == BG:
                cells.add((r, col))
    return cells


def _hypothetical_visible_cells(grid: np.ndarray, obj: dict) -> set[tuple[int, int]]:
    if not obj["visible"]:
        return set()
    r0, c0, r1, c1 = map(int, obj["bbox"])
    cells: set[tuple[int, int]] = set()
    if obj["orientation"] == "H":
        row = r0 + (r1 - r0) // 2
        for c in range(grid.shape[1]):
            if int(grid[row, c]) == BG:
                cells.add((row, c))
    else:
        col = c0 + (c1 - c0) // 2
        for r in range(grid.shape[0]):
            if int(grid[r, col]) == BG:
                cells.add((r, col))
    return cells


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY


def _place_markers(grid: np.ndarray, marker_colors: list[int]) -> None:
    col = 0
    for color in marker_colors:
        grid[0, col] = int(color)
        col += random.randint(2, 3)


def _choose_period_and_phase(length: int) -> tuple[int, int]:
    options = []
    for period in (2, 3):
        for phase in range(period):
            idx = [i for i in range(length) if i % period == phase]
            if 2 <= len(idx) < length:
                options.append((period, phase))
    return random.choice(options)


def _make_object_mask(orientation: str) -> np.ndarray:
    if orientation == "H":
        options = [(1, 3), (1, 4), (2, 3), (2, 4)]
    else:
        options = [(3, 1), (4, 1), (3, 2), (4, 2)]
    h, w = random.choice(options)
    mask = np.ones((h, w), dtype=bool)
    if h > 1 and w > 1 and random.random() < 0.7:
        rr = random.choice([0, h - 1])
        cc = random.choice([0, w - 1])
        mask[rr, cc] = False
    return mask


def _place_object(grid: np.ndarray, mask: np.ndarray, spec: dict, frame_bbox: tuple[int, int, int, int]) -> dict | None:
    fr0, fc0, fr1, fc1 = frame_bbox
    h, w = mask.shape
    row_center_offset = (h - 1) // 2
    col_center_offset = (w - 1) // 2

    candidates: list[tuple[int, int]] = []
    if spec["orientation"] == "H":
        allowed_rows = [int(r) for r in spec.get("allowed_rows", [])]
        if not allowed_rows:
            return None
        side_order = ["left", "right"]
        random.shuffle(side_order)
        for center_row in allowed_rows:
            for side in side_order:
                r0 = center_row - row_center_offset
                if r0 < 1 or r0 + h > grid.shape[0]:
                    continue
                if side == "left":
                    col_range = list(range(0, max(0, fc0 - w)))
                else:
                    col_range = list(range(fc1 + 2, grid.shape[1] - w + 1))
                random.shuffle(col_range)
                for c0 in col_range:
                    if _can_place(grid, mask, r0, c0):
                        candidates.append((r0, c0))
    else:
        allowed_cols = [int(c) for c in spec.get("allowed_cols", [])]
        if not allowed_cols:
            return None
        zone_order = ["top", "bottom"]
        random.shuffle(zone_order)
        for center_col in allowed_cols:
            for zone in zone_order:
                c0 = center_col - col_center_offset
                if c0 < 0 or c0 + w > grid.shape[1]:
                    continue
                if zone == "top":
                    row_range = list(range(1, max(1, fr0 - h)))
                else:
                    row_range = list(range(fr1 + 2, grid.shape[0] - h + 1))
                random.shuffle(row_range)
                for r0 in row_range:
                    if _can_place(grid, mask, r0, c0):
                        candidates.append((r0, c0))

    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    color = int(spec["color"])
    coords = []
    for dr in range(h):
        for dc in range(w):
            if not bool(mask[dr, dc]):
                continue
            rr = r0 + dr
            cc = c0 + dc
            grid[rr, cc] = color
            coords.append((rr, cc))
    coords_arr = np.asarray(coords, dtype=int)
    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
    out = dict(spec)
    out["coords"] = coords_arr
    out["bbox"] = bbox
    return out


def _can_place(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    H, W = grid.shape
    for dr in range(h):
        for dc in range(w):
            if not bool(mask[dr, dc]):
                continue
            rr = r0 + dr
            cc = c0 + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                return False
            if int(grid[rr, cc]) != BG:
                return False
            for nr in range(max(0, rr - 1), min(H, rr + 2)):
                for nc in range(max(0, cc - 1), min(W, cc + 2)):
                    if int(grid[nr, nc]) != BG:
                        return False
    return True


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    gray_components = find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GRAY],
        return_format="dict",
    )
    assert gray_components, "expected one gray frame"
    frame = max(gray_components, key=lambda obj: int(obj["area"]))
    return tuple(map(int, frame["bbox"]))


def _infer_periodic_masks(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[set[int], set[int] | None]:
    r0, c0, r1, c1 = frame_bbox
    sub = crop_bbox(np.asarray(grid, dtype=int), frame_bbox)
    interior = sub[1:-1, 1:-1]
    assert interior.size > 0

    row_reps_local = []
    for rr in range(interior.shape[0]):
        if _max_run_length(interior[rr, :] == ORANGE) > 1:
            row_reps_local.append(rr)
    assert len(row_reps_local) >= 2, "horizontal stripe period must be inferable"

    row_reps_global = [r0 + 1 + rr for rr in row_reps_local]
    row_period = _infer_period(row_reps_global)
    active_rows = {r for r in range(grid.shape[0]) if (r - row_reps_global[0]) % row_period == 0}

    col_reps_local = []
    for cc in range(interior.shape[1]):
        if _max_run_length(interior[:, cc] == ORANGE) > 1:
            col_reps_local.append(cc)
    if not col_reps_local:
        return active_rows, None

    col_reps_global = [c0 + 1 + cc for cc in col_reps_local]
    col_period = _infer_period(col_reps_global)
    active_cols = {c for c in range(grid.shape[1]) if (c - col_reps_global[0]) % col_period == 0}
    return active_rows, active_cols


def _infer_period(indices: list[int]) -> int:
    diffs = [b - a for a, b in zip(indices, indices[1:])]
    assert diffs and all(d > 0 for d in diffs)
    period = diffs[0]
    assert period > 1 and all(d == period for d in diffs)
    return int(period)


def _max_run_length(mask_1d: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask_1d.tolist():
        if bool(value):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
