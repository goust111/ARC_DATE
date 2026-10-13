from __future__ import annotations

import random
from collections import deque

import numpy as np


ROTATION_BY_FRAME_COLOR = {
    2: 0,  # red
    1: 1,  # blue
    3: 2,  # green
    4: 3,  # yellow
}

FRAME_COLORS = [2, 1, 3, 4]
PIECE_COLORS = [5, 6, 7, 8, 9]
BG_COLOR = 0

FRAME_COUNT_CHOICES = [4, 5]
GRID_SIDE_RANGE = (20, 30)
FRAME_SIDE_RANGE = (4, 8)
GAP_LENGTH_CHOICES = [2, 3, 4]

GENERATION_RETRIES = 500
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _crop_true(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros((0, 0), dtype=bool)
    cropped, _ = _crop_true(mask)
    return cropped


def _rotate_cw(mask: np.ndarray, quarter_turns: int) -> np.ndarray:
    turns = int(quarter_turns) % 4
    rotated = np.rot90(np.asarray(mask, dtype=bool), -turns)
    return _normalize_mask(rotated)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    m = _normalize_mask(mask)
    coords = tuple(map(tuple, np.argwhere(m)))
    return (tuple(map(int, m.shape)), coords)


def _canonical_rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    variants = [_mask_signature(np.rot90(mask, k)) for k in range(4)]
    return min(variants)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    queue: deque[tuple[int, int]] = deque([tuple(map(int, coords[0]))])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return components


def _frame_candidates(grid: np.ndarray, bg: int, component: dict) -> list[dict]:
    rows, cols = grid.shape
    base_r0, base_c0, base_r1, base_c1 = component["bbox"]
    base_mask = np.asarray(component["mask"], dtype=bool)
    base_h, base_w = base_mask.shape
    candidates: list[dict] = []
    seen: set[tuple[tuple[int, int, int, int], tuple[tuple[int, int], tuple[tuple[int, int], ...]]]] = set()

    for expand_top in (0, 1):
        for expand_bottom in (0, 1):
            for expand_left in (0, 1):
                for expand_right in (0, 1):
                    r0 = base_r0 - expand_top
                    c0 = base_c0 - expand_left
                    r1 = base_r1 + expand_bottom
                    c1 = base_c1 + expand_right
                    if r0 < 0 or c0 < 0 or r1 >= rows or c1 >= cols:
                        continue

                    height = r1 - r0 + 1
                    width = c1 - c0 + 1
                    if height < 3 or width < 3:
                        continue

                    local = np.zeros((height, width), dtype=bool)
                    local[expand_top : expand_top + base_h, expand_left : expand_left + base_w] = base_mask

                    border = np.zeros((height, width), dtype=bool)
                    border[0, :] = True
                    border[-1, :] = True
                    border[:, 0] = True
                    border[:, -1] = True

                    if np.any(local & ~border):
                        continue
                    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                        continue

                    gap = border & ~local
                    gap_area = int(gap.sum())
                    if gap_area < 2 or gap_area > 4 or not _is_connected(gap):
                        continue

                    gap_mask, (gr0, gc0) = _crop_true(gap)
                    key = ((int(r0), int(c0), int(r1), int(c1)), _mask_signature(gap_mask))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "color": int(component["color"]),
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "gap_mask": gap_mask,
                            "gap_top_left": (int(r0 + gr0), int(c0 + gc0)),
                        }
                    )

    return candidates


def _match_frames(frame_options: list[list[dict]], pieces: list[dict]) -> list[tuple[dict, int]]:
    choices: list[list[tuple[dict, list[int]]]] = []
    for options in frame_options:
        per_frame: list[tuple[dict, list[int]]] = []
        for frame in options:
            turns = ROTATION_BY_FRAME_COLOR.get(int(frame["color"]))
            assert turns is not None, "unknown frame color"
            matches: list[int] = []
            for pi, piece in enumerate(pieces):
                rotated = _rotate_cw(piece["mask"], turns)
                if rotated.shape == frame["gap_mask"].shape and np.array_equal(rotated, frame["gap_mask"]):
                    matches.append(pi)
            if matches:
                per_frame.append((frame, matches))
        assert per_frame, "every frame component must have a viable interpretation"
        choices.append(per_frame)

    order = sorted(range(len(choices)), key=lambda idx: min(len(matches) for _, matches in choices[idx]))
    assignment: list[tuple[dict, int] | None] = [None] * len(choices)
    used: set[int] = set()

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        fi = order[pos]
        for frame, matches in sorted(choices[fi], key=lambda item: (len(item[1]), int(item[0]["gap_mask"].sum()))):
            for pi in matches:
                if pi in used:
                    continue
                used.add(pi)
                assignment[fi] = (frame, pi)
                if search(pos + 1):
                    return True
                assignment[fi] = None
                used.remove(pi)
        return False

    assert search(0), "frame-piece matching must be solvable"
    return [item for item in assignment if item is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each scattered fragment into the matching frame gap after rotating it by the frame's border color.

    Input structure:
    - The background is the dominant color.
    - Several monochrome rectangular frames are drawn with a 1-cell border; each frame is missing one connected 2-4-cell segment on its perimeter.
    - Separate monochrome fragments lie elsewhere in the grid, and each fragment matches one frame gap up to rotation.
    - Frame borders use red, blue, green, and yellow to encode 0, 90, 180, and 270 degrees clockwise rotation.

    Rule:
    - Detect every frame and extract the exact missing gap shape from its perimeter.
    - For each frame, rotate each candidate fragment by the angle implied by the frame color and find the unique shape match.
    - Erase the fragment from its original location and paint the rotated fragment into the frame gap using the fragment's own color.

    Key insight:
    - The missing border cells define the destination shape, while the frame color tells you how the matching loose fragment must be rotated before insertion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)

    frame_options: list[list[dict]] = []
    pieces: list[dict] = []
    for comp in components:
        options = []
        if int(comp["color"]) in ROTATION_BY_FRAME_COLOR:
            options = _frame_candidates(g, bg, comp)
        if options:
            frame_options.append(options)
        else:
            pieces.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "mask": _normalize_mask(comp["mask"]),
                }
            )

    assert frame_options, "expected at least one frame"
    assert len(pieces) >= len(frame_options), "expected at least as many pieces as frames"

    assignment = _match_frames(frame_options, pieces)
    out = g.copy()

    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for frame, piece_index in assignment:
        piece = pieces[piece_index]
        turns = ROTATION_BY_FRAME_COLOR[int(frame["color"])]
        rotated = _rotate_cw(piece["mask"], turns)
        top, left = frame["gap_top_left"]
        h, w = rotated.shape
        patch = out[top : top + h, left : left + w]
        patch[rotated] = int(piece["color"])

    return out


def _perimeter_cycle(height: int, width: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(width):
        cells.append((0, c))
    for r in range(1, height):
        cells.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        cells.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        cells.append((r, 0))
    return cells


def _random_gap(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    perimeter = _perimeter_cycle(height, width)
    gap_len = random.choice(GAP_LENGTH_CHOICES)
    start = random.randrange(len(perimeter))
    gap = np.zeros((height, width), dtype=bool)
    for step in range(gap_len):
        r, c = perimeter[(start + step) % len(perimeter)]
        gap[r, c] = True
    gap_crop, _ = _crop_true(gap)
    return gap, gap_crop


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    rows, cols = occupied.shape
    if top < 0 or left < 0 or top + height > rows or left + width > cols:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(rows, top + height + pad)
    c1 = min(cols, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_box(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _make_specs(frame_count: int) -> list[dict] | None:
    frame_colors = FRAME_COLORS.copy()
    if frame_count > len(frame_colors):
        frame_colors.extend(random.choices(FRAME_COLORS, k=frame_count - len(frame_colors)))
    random.shuffle(frame_colors)
    piece_colors = random.sample(PIECE_COLORS, frame_count)

    specs: list[dict] = []
    seen_shapes: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for index in range(frame_count):
        frame_color = int(frame_colors[index])
        turns = ROTATION_BY_FRAME_COLOR[frame_color]
        for _ in range(SHAPE_RETRIES):
            height = random.randint(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1])
            width = random.randint(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1])
            gap_full, gap_crop = _random_gap(height, width)
            signature = _canonical_rotation_signature(gap_crop)
            if signature in seen_shapes:
                continue
            seen_shapes.add(signature)
            specs.append(
                {
                    "frame_color": frame_color,
                    "piece_color": int(piece_colors[index]),
                    "height": height,
                    "width": width,
                    "gap_full": gap_full,
                    "gap_crop": gap_crop,
                    "piece_mask": _rotate_cw(gap_crop, (-turns) % 4),
                }
            )
            break
        else:
            return None
    return specs


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    specs = _make_specs(frame_count)
    if specs is None:
        return None

    total_box_area = sum(spec["height"] * spec["width"] for spec in specs)
    total_box_area += sum(spec["piece_mask"].shape[0] * spec["piece_mask"].shape[1] for spec in specs)

    for _ in range(20):
        rows = random.randint(*GRID_SIDE_RANGE)
        cols = random.randint(*GRID_SIDE_RANGE)
        if rows * cols >= total_box_area * 2:
            break
    else:
        return None

    occupied = np.zeros((rows, cols), dtype=bool)

    frame_positions: dict[int, tuple[int, int]] = {}
    frame_order = sorted(range(frame_count), key=lambda i: specs[i]["height"] * specs[i]["width"], reverse=True)
    for index in frame_order:
        height = specs[index]["height"]
        width = specs[index]["width"]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, rows - height)
            left = random.randint(0, cols - width)
            if _can_place(occupied, top, left, height, width, pad=1):
                frame_positions[index] = (top, left)
                _mark_box(occupied, top, left, height, width)
                placed = True
                break
        if not placed:
            return None

    piece_positions: dict[int, tuple[int, int]] = {}
    piece_order = sorted(range(frame_count), key=lambda i: specs[i]["piece_mask"].size, reverse=True)
    for index in piece_order:
        piece_mask = specs[index]["piece_mask"]
        height, width = piece_mask.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, rows - height)
            left = random.randint(0, cols - width)
            if _can_place(occupied, top, left, height, width, pad=1):
                piece_positions[index] = (top, left)
                _mark_box(occupied, top, left, height, width)
                placed = True
                break
        if not placed:
            return None

    inp = np.full((rows, cols), BG_COLOR, dtype=int)
    out = np.full((rows, cols), BG_COLOR, dtype=int)

    for index, spec in enumerate(specs):
        top, left = frame_positions[index]
        height = spec["height"]
        width = spec["width"]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        frame_mask = border & ~spec["gap_full"]
        _draw_mask(inp, top, left, frame_mask, spec["frame_color"])
        _draw_mask(out, top, left, frame_mask, spec["frame_color"])

        gap_crop, (gr0, gc0) = _crop_true(spec["gap_full"])
        _draw_mask(out, top + gr0, left + gc0, gap_crop, spec["piece_color"])

    for index, spec in enumerate(specs):
        top, left = piece_positions[index]
        _draw_mask(inp, top, left, spec["piece_mask"], spec["piece_color"])

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
