from __future__ import annotations

from collections import defaultdict

import numpy as np


BG = 0
FRAME_COLORS = (2, 3, 4)
MARKER = 6
PAINT = 7
MAX_GENERATION_ATTEMPTS = 300
RNG = np.random.default_rng()


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    if r1 != r0:
        for c in range(c0, c1 + 1):
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
    if c1 != c0:
        for r in range(r0 + 1, r1):
            coords.append((r, c1))
    return coords


def _detect_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(value) for value in np.unique(grid) if int(value) not in (BG, MARKER, PAINT)):
        for coords in _components_of_color(grid, color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 4 or width < 4:
                continue
            if not np.all(grid[r0, c0 : c1 + 1] == color):
                continue
            if not np.all(grid[r1, c0 : c1 + 1] == color):
                continue
            if not np.all(grid[r0 : r1 + 1, c0] == color):
                continue
            if not np.all(grid[r0 : r1 + 1, c1] == color):
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or not np.any(interior != color):
                continue
            if not np.all((interior == BG) | (interior == color)):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # type: ignore[arg-type]
    return frames


def _fill_shared_patterns(grid: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)

    for color, group in by_color.items():
        union: set[tuple[int, int]] = set()
        for frame in group:
            r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
            interior_hits = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == color)
            for rr, cc in interior_hits:
                union.add((int(rr) + 1, int(cc) + 1))
        for frame in group:
            r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            for dr, dc in union:
                if not (1 <= dr < height - 1 and 1 <= dc < width - 1):
                    continue
                rr = r0 + dr
                cc = c0 + dc
                if out[rr, cc] == BG:
                    out[rr, cc] = color
    return out


def _draw_inner_frame(grid: np.ndarray) -> np.ndarray:
    markers = np.argwhere(grid == MARKER)
    assert markers.shape[0] == 4, "expected exactly four marker pixels"
    rmin = int(markers[:, 0].min())
    rmax = int(markers[:, 0].max())
    cmin = int(markers[:, 1].min())
    cmax = int(markers[:, 1].max())
    r0 = rmin + 1
    r1 = rmax - 1
    c0 = cmin + 1
    c1 = cmax - 1
    assert r0 <= r1 and c0 <= c1, "inner frame must exist"

    out = grid.copy()
    for r, c in _perimeter_coords(r0, c0, r1, c1):
        if out[r, c] == BG:
            out[r, c] = PAINT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each same-color framed pattern, then draw an inner orange frame that is blocked by the completed pixels.

    Input structure:
    - The background is black.
    - A central 2x3 or 3x2 array of separated hollow frames uses two or three colors from red, green, and yellow.
    - Frames of the same color contain partial same-color interior pixels, and four magenta marker pixels sit on the outer perimeter gaps of the array.

    Rule:
    - Detect every hollow monochrome frame and group frames by border color.
    - For each color group, collect the relative interior locations of same-color pixels across all frames in that group and write that union back into every same-color frame, filling only black cells.
    - From the four magenta markers, take the extreme rows and columns and shrink that marker rectangle inward by one cell.
    - Draw an orange rectangle border on that inner perimeter, but only where the current cell is black; keep any non-black pixel, including newly completed pattern pixels, unchanged.

    Key insight:
    - Pattern completion must happen before the orange border is drawn, because the newly added interior pixels can lie exactly on the later border path and therefore create visible gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    frames = _detect_frames(g)
    assert frames, "expected at least one framed object"
    completed = _fill_shared_patterns(g, frames)
    return _draw_inner_frame(completed)


def _interior_positions(size: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(1, size - 1) for dc in range(1, size - 1)]


def _path_positions(rows: int, cols: int, size: int, box_row: int, box_col: int) -> list[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    if box_row == 0:
        for dc in range(1, size - 1):
            positions.add((1, dc))
    if box_row == rows - 1:
        for dc in range(1, size - 1):
            positions.add((size - 2, dc))
    if box_col == 0:
        for dr in range(1, size - 1):
            positions.add((dr, 1))
    if box_col == cols - 1:
        for dr in range(1, size - 1):
            positions.add((dr, size - 2))
    return sorted(positions)


def _assign_fragments(
    rng: np.random.Generator,
    box_count: int,
    full_positions: list[tuple[int, int]],
    forbidden_boxes: dict[tuple[int, int], set[int]],
) -> list[list[tuple[int, int]]] | None:
    subsets: list[set[tuple[int, int]]] = [set() for _ in range(box_count)]
    positions = full_positions[:]
    rng.shuffle(positions)

    for pos in positions:
        forbidden = forbidden_boxes.get(pos, set())
        allowed = [box for box in range(box_count) if box not in forbidden]
        if not allowed:
            return None
        min_load = min(len(subsets[box]) for box in allowed)
        primary_choices = [box for box in allowed if len(subsets[box]) == min_load]
        primary = int(primary_choices[int(rng.integers(len(primary_choices)))])
        subsets[primary].add(pos)
        secondary_choices = [box for box in allowed if box != primary]
        if secondary_choices and rng.random() < 0.45:
            secondary = int(secondary_choices[int(rng.integers(len(secondary_choices)))])
            subsets[secondary].add(pos)

    required_nonempty = min(2, box_count)
    if sum(bool(subset) for subset in subsets) < required_nonempty:
        empties = [i for i, subset in enumerate(subsets) if not subset]
        donors = sorted(range(box_count), key=lambda idx: len(subsets[idx]), reverse=True)
        for empty in empties:
            moved = False
            for donor in donors:
                donor_positions = [
                    pos for pos in subsets[donor] if empty not in forbidden_boxes.get(pos, set())
                ]
                if donor_positions:
                    pos = donor_positions[int(rng.integers(len(donor_positions)))]
                    subsets[empty].add(pos)
                    moved = True
                    break
            if not moved:
                return None

    full_set = set(full_positions)
    for box in range(box_count):
        if subsets[box] != full_set:
            continue
        removable = []
        for pos in full_positions:
            if any(pos in subsets[other] for other in range(box_count) if other != box):
                removable.append(pos)
                continue
            receivers = [
                other
                for other in range(box_count)
                if other != box and other not in forbidden_boxes.get(pos, set())
            ]
            if receivers:
                removable.append(pos)
        if not removable:
            return None
        pos = removable[int(rng.integers(len(removable)))]
        if not any(pos in subsets[other] for other in range(box_count) if other != box):
            receivers = [
                other
                for other in range(box_count)
                if other != box and other not in forbidden_boxes.get(pos, set())
            ]
            if not receivers:
                return None
            receiver = int(receivers[int(rng.integers(len(receivers)))])
            subsets[receiver].add(pos)
        subsets[box].remove(pos)
        if not subsets[box]:
            refill_choices = [
                candidate
                for candidate in full_positions
                if candidate != pos and box not in forbidden_boxes.get(candidate, set())
            ]
            if not refill_choices:
                return None
            refill = refill_choices[int(rng.integers(len(refill_choices)))]
            subsets[box].add(refill)

    if any(len(subset) > 3 for subset in subsets):
        return None
    if set().union(*subsets) != full_set:
        return None
    return [sorted(subset) for subset in subsets]


def _fill_from_spec(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    out = grid.copy()
    boxes = spec["boxes"]  # type: ignore[assignment]
    groups = spec["groups"]  # type: ignore[assignment]
    for group in groups.values():
        full_positions = group["full_positions"]  # type: ignore[index]
        for box_index in group["box_indices"]:  # type: ignore[index]
            box = boxes[box_index]  # type: ignore[index]
            r0, c0 = box["top_left"]
            color = box["color"]
            for dr, dc in full_positions:
                rr = r0 + dr
                cc = c0 + dc
                if out[rr, cc] == BG:
                    out[rr, cc] = color
    return out


def _expected_from_spec(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    return _draw_inner_frame(_fill_from_spec(grid, spec))


def _wrong_order_from_spec(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    return _fill_from_spec(_draw_inner_frame(grid), spec)


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]] | None:
    rows, cols = ((2, 3), (3, 2))[int(rng.integers(2))]
    size = int(rng.integers(4, 6))
    array_height = rows * size + (rows - 1)
    array_width = cols * size + (cols - 1)

    top = int(rng.integers(1, 5))
    bottom = int(rng.integers(1, 5))
    left = int(rng.integers(1, 5))
    right = int(rng.integers(1, 5))
    height = array_height + top + bottom
    width = array_width + left + right
    if max(height, width) > 30:
        return None

    if rng.random() < 0.5:
        chosen = list(rng.choice(FRAME_COLORS, size=2, replace=False))
        color_assignment = [int(chosen[0])] * 3 + [int(chosen[1])] * 3
    else:
        chosen = list(rng.choice(FRAME_COLORS, size=3, replace=False))
        color_assignment = [int(chosen[0]), int(chosen[0]), int(chosen[1]), int(chosen[1]), int(chosen[2]), int(chosen[2])]
    rng.shuffle(color_assignment)

    boxes: list[dict[str, object]] = []
    for box_row in range(rows):
        for box_col in range(cols):
            r0 = top + box_row * (size + 1)
            c0 = left + box_col * (size + 1)
            boxes.append(
                {
                    "grid_pos": (box_row, box_col),
                    "top_left": (r0, c0),
                    "color": int(color_assignment[len(boxes)]),
                    "path_positions": _path_positions(rows, cols, size, box_row, box_col),
                }
            )

    groups: dict[int, list[int]] = defaultdict(list)
    for index, box in enumerate(boxes):
        groups[int(box["color"])].append(index)

    target_color = int(sorted(groups)[int(rng.integers(len(groups)))])
    target_box = int(groups[target_color][int(rng.integers(len(groups[target_color])))] )
    target_local = groups[target_color].index(target_box)
    target_path_positions = list(boxes[target_box]["path_positions"])  # type: ignore[arg-type]
    if not target_path_positions:
        return None

    spec_groups: dict[int, dict[str, object]] = {}
    all_interior = _interior_positions(size)
    for color, indices in groups.items():
        full_size = int(rng.integers(2, min(3, len(all_interior)) + 1))
        forbidden: dict[tuple[int, int], set[int]] = {}
        if color == target_color:
            overlap_count = min(full_size, len(target_path_positions), 2 if rng.random() < 0.65 else 1)
            shuffled_path = target_path_positions[:]
            rng.shuffle(shuffled_path)
            overlap_positions = shuffled_path[:overlap_count]
            extras = [pos for pos in all_interior if pos not in overlap_positions]
            rng.shuffle(extras)
            full_positions = overlap_positions + extras[: full_size - overlap_count]
            forbidden = {pos: {target_local} for pos in overlap_positions}
        else:
            shuffled = all_interior[:]
            rng.shuffle(shuffled)
            full_positions = shuffled[:full_size]
            overlap_positions = []

        fragments = _assign_fragments(rng, len(indices), full_positions, forbidden)
        if fragments is None:
            return None

        for local_index, global_index in enumerate(indices):
            boxes[global_index]["partial_positions"] = fragments[local_index]

        spec_groups[color] = {
            "box_indices": indices,
            "full_positions": sorted(full_positions),
            "overlap_positions": sorted(overlap_positions),
        }

    gap_rows = [top + row * (size + 1) + size for row in range(rows - 1)]
    gap_cols = [left + col * (size + 1) + size for col in range(cols - 1)]
    marker_row = int(gap_rows[int(rng.integers(len(gap_rows)))])
    marker_col = int(gap_cols[int(rng.integers(len(gap_cols)))])
    array_r0 = top
    array_r1 = top + array_height - 1
    array_c0 = left
    array_c1 = left + array_width - 1
    markers = [
        (array_r0, marker_col),
        (array_r1, marker_col),
        (marker_row, array_c0),
        (marker_row, array_c1),
    ]

    grid = np.zeros((height, width), dtype=int)
    for box in boxes:
        r0, c0 = box["top_left"]
        color = int(box["color"])
        grid[r0, c0 : c0 + size] = color
        grid[r0 + size - 1, c0 : c0 + size] = color
        grid[r0 : r0 + size, c0] = color
        grid[r0 : r0 + size, c0 + size - 1] = color
        for dr, dc in box["partial_positions"]:  # type: ignore[index]
            grid[r0 + dr, c0 + dc] = color

    for r, c in markers:
        if grid[r, c] != BG:
            return None
        grid[r, c] = MARKER

    spec = {
        "rows": rows,
        "cols": cols,
        "size": size,
        "boxes": boxes,
        "groups": spec_groups,
        "markers": markers,
        "target_box": target_box,
        "target_color": target_color,
    }
    return grid, spec


def _validate_case(grid: np.ndarray, spec: dict[str, object]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    frames = _detect_frames(grid)
    if len(frames) != len(spec["boxes"]):  # type: ignore[arg-type]
        return False
    detected_counts = sorted(int(frame["color"]) for frame in frames)
    expected_counts = sorted(int(box["color"]) for box in spec["boxes"])  # type: ignore[index]
    if detected_counts != expected_counts:
        return False

    expected = _expected_from_spec(grid, spec)
    wrong = _wrong_order_from_spec(grid, spec)
    solved = main(grid)
    if not np.array_equal(expected, solved):
        return False
    if np.array_equal(expected, wrong):
        return False

    target_box = spec["boxes"][spec["target_box"]]  # type: ignore[index]
    target_group = spec["groups"][spec["target_color"]]  # type: ignore[index]
    r0, c0 = target_box["top_left"]
    path_positions = set(target_box["path_positions"])
    partial_positions = set(target_box["partial_positions"])
    overlap_positions = set(target_group["overlap_positions"])
    added_overlap_coords = []
    for dr, dc in overlap_positions:
        if (dr, dc) in path_positions and (dr, dc) not in partial_positions:
            added_overlap_coords.append((r0 + dr, c0 + dc))
    if not added_overlap_coords:
        return False
    if not all(expected[r, c] == int(target_box["color"]) for r, c in added_overlap_coords):
        return False
    if not all(wrong[r, c] == PAINT for r, c in added_overlap_coords):
        return False

    orange_added = (expected == PAINT) & (grid == BG)
    if not np.any(orange_added):
        return False

    completed_pixels = (expected != grid) & (expected != PAINT)
    if not np.any(completed_pixels):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case(RNG)
        if case is None:
            continue
        grid, spec = case
        if _validate_case(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
