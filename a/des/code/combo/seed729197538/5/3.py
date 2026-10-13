from __future__ import annotations

import numpy as np


BG = 0
MARKER = 5
OBJECT_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

QUADRANT_TO_STEP = {
    "TL": (1, 1),
    "TR": (1, -1),
    "BL": (-1, 1),
    "BR": (-1, -1),
}
STEP_TO_BIAS = {step: quad for quad, step in QUADRANT_TO_STEP.items()}
ALL_STEPS = tuple(QUADRANT_TO_STEP.values())

RNG = np.random.default_rng()

# Generator controls.
GEN_RETRIES = 300
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
OBJECT_COUNT_RANGE = (2, 5)
MIN_COPIES = 1
MAX_COPIES = 4
MIN_CLIP_COPIES = 2
SHAPE_SIZE_RANGE = (3, 6)
SHAPE_ATTEMPTS = 200
OBJECT_ATTEMPTS = 80
ANCHOR_ATTEMPTS = 60
CLIP_SCENE_PROB = 0.45


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _half_index(pos: int, size: int) -> int:
    scaled = 2 * pos + 1
    if scaled < size:
        return 0
    if scaled > size:
        return 1
    return -1


def _cell_quadrant(r: int, c: int, h: int, w: int) -> str | None:
    vr = _half_index(r, h)
    hc = _half_index(c, w)
    if vr < 0 or hc < 0:
        return None
    return ("T" if vr == 0 else "B") + ("L" if hc == 0 else "R")


def _quadrant_counts(mask: np.ndarray) -> dict[str, int]:
    h, w = mask.shape
    counts = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
    for r, c in np.argwhere(mask):
        quad = _cell_quadrant(int(r), int(c), h, w)
        if quad is not None:
            counts[quad] += 1
    return counts


def _copy_direction_from_mask(mask: np.ndarray) -> tuple[int, int] | None:
    counts = _quadrant_counts(mask)
    best = max(counts.values())
    if best <= 0:
        return None
    winners = [quad for quad, value in counts.items() if value == best]
    if len(winners) != 1:
        return None
    return QUADRANT_TO_STEP[winners[0]]


def _object_descriptor(color: int, coords: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local_coords = []
    for r, c in coords:
        lr = r - r0
        lc = c - c0
        mask[lr, lc] = True
        local_coords.append((lr, lc))
    return {
        "color": int(color),
        "coords": [(int(r), int(c)) for r, c in coords],
        "coord_set": {(int(r), int(c)) for r, c in coords},
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "local_coords": local_coords,
        "direction": _copy_direction_from_mask(mask),
    }


def _find_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color in (BG, MARKER):
            continue
        for comp in _components(grid == color):
            objects.append(_object_descriptor(color, comp))
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))  # type: ignore[index]
    return objects


def _adjacent_markers(grid: np.ndarray, coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = grid.shape
    found: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == MARKER:
                found.add((nr, nc))
    return found


def _touches_component(cell: tuple[int, int], coord_set: set[tuple[int, int]]) -> bool:
    r, c = cell
    for dr, dc in NEIGHBORS8:
        if (r + dr, c + dc) in coord_set:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    out[out == MARKER] = BG

    for obj in _find_objects(g):
        direction = obj["direction"]
        assert direction is not None, "object must have a unique quadrant bias"
        coords = obj["coords"]
        markers = _adjacent_markers(g, coords)  # type: ignore[arg-type]
        copies = len(markers)
        assert copies > 0, "each object must have at least one marker pixel"

        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[misc]
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        step_r, step_c = direction
        color = int(obj["color"])

        for k in range(1, copies + 1):
            dr = step_r * box_h * k
            dc = step_c * box_w * k
            for r, c in coords:  # type: ignore[assignment]
                nr = r + dr
                nc = c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    out[nr, nc] = color
    return out


def _growth_weight(r: int, c: int, bias: str, h: int, w: int) -> float:
    quad = _cell_quadrant(r, c, h, w)
    if quad is None:
        return 2.0
    if quad == bias:
        return 6.0
    score = 1.0
    if quad[0] == bias[0]:
        score += 1.5
    if quad[1] == bias[1]:
        score += 1.5
    return score


def _random_connected_shape(direction: tuple[int, int]) -> np.ndarray:
    bias = STEP_TO_BIAS[direction]
    for _ in range(SHAPE_ATTEMPTS):
        h = int(RNG.integers(SHAPE_SIZE_RANGE[0], SHAPE_SIZE_RANGE[1]))
        w = int(RNG.integers(SHAPE_SIZE_RANGE[0], SHAPE_SIZE_RANGE[1]))
        bias_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if _cell_quadrant(r, c, h, w) == bias
        ]
        if not bias_cells:
            continue

        target = int(RNG.integers(max(4, (h * w) // 3), h * w))
        start = bias_cells[int(RNG.integers(len(bias_cells)))]
        mask = np.zeros((h, w), dtype=bool)
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: dict[tuple[int, int], float] = {}
            for r, c in map(tuple, np.argwhere(mask)):
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier[(nr, nc)] = frontier.get((nr, nc), 0.0) + _growth_weight(
                            nr, nc, bias, h, w
                        )
            if not frontier:
                break
            cells = list(frontier)
            probs = np.array([frontier[cell] for cell in cells], dtype=float)
            probs /= probs.sum()
            pick = cells[int(RNG.choice(len(cells), p=probs))]
            mask[pick] = True

        cropped = _crop_mask(mask)
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if int(cropped.sum()) < 4:
            continue
        if np.all(cropped):
            continue
        coords = np.argwhere(cropped)
        if len(set(map(int, coords[:, 0]))) < 2 or len(set(map(int, coords[:, 1]))) < 2:
            continue
        if _copy_direction_from_mask(cropped) == direction:
            return cropped
    raise RuntimeError("failed to generate a directional shape")


def _object_footprint(
    local_coords: list[tuple[int, int]],
    top: int,
    left: int,
    step: tuple[int, int],
    copies: int,
    canvas_h: int,
    canvas_w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], bool]:
    box_h = max(r for r, _ in local_coords) + 1
    box_w = max(c for _, c in local_coords) + 1
    original = {(top + r, left + c) for r, c in local_coords}
    visible: set[tuple[int, int]] = set()
    clipped = False

    for k in range(copies + 1):
        dr = step[0] * box_h * k
        dc = step[1] * box_w * k
        for r, c in local_coords:
            nr = top + r + dr
            nc = left + c + dc
            if 0 <= nr < canvas_h and 0 <= nc < canvas_w:
                visible.add((nr, nc))
            elif k > 0:
                clipped = True
    return original, visible, clipped


def _is_far_from_scene(cell: tuple[int, int], scene_cells: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in scene_cells:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _adjacent_positions(cells: set[tuple[int, int]], canvas_h: int, canvas_w: int) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                out.add((nr, nc))
    return list(out)


def _choose_markers(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if len(candidates) < count:
        return None
    candidates = list(candidates)
    for _ in range(40):
        RNG.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _sample_inbounds_anchor(
    canvas_h: int,
    canvas_w: int,
    box_h: int,
    box_w: int,
    step: tuple[int, int],
    copies: int,
) -> tuple[int, int] | None:
    row_lo = 0 if step[0] > 0 else copies * box_h
    row_hi = canvas_h - (copies + 1) * box_h if step[0] > 0 else canvas_h - box_h
    col_lo = 0 if step[1] > 0 else copies * box_w
    col_hi = canvas_w - (copies + 1) * box_w if step[1] > 0 else canvas_w - box_w
    if row_lo > row_hi or col_lo > col_hi:
        return None
    top = int(RNG.integers(row_lo, row_hi + 1))
    left = int(RNG.integers(col_lo, col_hi + 1))
    return top, left


def _try_place_object(
    grid: np.ndarray,
    color: int,
    direction: tuple[int, int],
    must_clip: bool,
    scene_cells: set[tuple[int, int]],
    output_reserved: set[tuple[int, int]],
) -> bool:
    canvas_h, canvas_w = grid.shape
    for _ in range(OBJECT_ATTEMPTS):
        shape = _random_connected_shape(direction)
        local_coords = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
        box_h, box_w = shape.shape
        max_full_copies = min(canvas_h // box_h, canvas_w // box_w) - 1

        if must_clip:
            copies = int(RNG.integers(MIN_CLIP_COPIES, MAX_COPIES + 1))
        else:
            if max_full_copies < MIN_COPIES:
                continue
            copies = int(RNG.integers(MIN_COPIES, min(MAX_COPIES, max_full_copies) + 1))

        for _ in range(ANCHOR_ATTEMPTS):
            if must_clip:
                top = int(RNG.integers(0, canvas_h - box_h + 1))
                left = int(RNG.integers(0, canvas_w - box_w + 1))
            else:
                anchor = _sample_inbounds_anchor(canvas_h, canvas_w, box_h, box_w, direction, copies)
                if anchor is None:
                    break
                top, left = anchor

            original, footprint, clipped = _object_footprint(
                local_coords, top, left, direction, copies, canvas_h, canvas_w
            )
            future = footprint - original
            if not future:
                continue
            if must_clip and not clipped:
                continue
            if not must_clip and clipped:
                continue
            if footprint & output_reserved:
                continue
            if not all(_is_far_from_scene(cell, scene_cells) for cell in original):
                continue

            marker_candidates = [
                cell
                for cell in _adjacent_positions(original, canvas_h, canvas_w)
                if cell not in future and cell not in output_reserved and _is_far_from_scene(cell, scene_cells)
            ]
            markers = _choose_markers(marker_candidates, copies)
            if markers is None:
                continue

            for r, c in original:
                grid[r, c] = color
            for r, c in markers:
                grid[r, c] = MARKER
            scene_cells.update(original)
            scene_cells.update(markers)
            output_reserved.update(footprint)
            return True
    return False


def _validate_scene(grid: np.ndarray) -> bool:
    objects = _find_objects(grid)
    assert OBJECT_COUNT_RANGE[0] <= len(objects) < OBJECT_COUNT_RANGE[1]

    colors = [int(obj["color"]) for obj in objects]
    assert len(colors) == len(set(colors))

    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert markers
    for cell in markers:
        owners = sum(_touches_component(cell, obj["coord_set"]) for obj in objects)  # type: ignore[arg-type]
        assert owners == 1

    for obj in objects:
        assert obj["direction"] in ALL_STEPS
        marker_count = len(_adjacent_markers(grid, obj["coords"]))  # type: ignore[arg-type]
        assert MIN_COPIES <= marker_count <= MAX_COPIES

    base = np.array(grid, copy=True)
    base[base == MARKER] = BG
    out = main(grid)
    assert out.shape == grid.shape
    assert not np.any(out == MARKER)
    assert np.any(out != base)
    return True


def _gen_attempt() -> np.ndarray | None:
    canvas_h = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    canvas_w = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    object_count = int(RNG.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1]))
    colors = [int(c) for c in RNG.choice(OBJECT_COLORS, size=object_count, replace=False)]
    directions = [ALL_STEPS[i] for i in RNG.permutation(len(ALL_STEPS))[:object_count]]

    clip_index = int(RNG.integers(object_count)) if RNG.random() < CLIP_SCENE_PROB else -1
    specs = [
        {"color": colors[i], "direction": directions[i], "must_clip": i == clip_index}
        for i in range(object_count)
    ]
    specs.sort(key=lambda spec: bool(spec["must_clip"]))

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    scene_cells: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()

    for spec in specs:
        ok = _try_place_object(
            grid,
            int(spec["color"]),
            spec["direction"],  # type: ignore[arg-type]
            bool(spec["must_clip"]),
            scene_cells,
            output_reserved,
        )
        if not ok:
            return None

    if _validate_scene(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamping scene")
