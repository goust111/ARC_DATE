from __future__ import annotations

import numpy as np


BG = 0
STAMP_COLOR = 4
MARKER_COLOR = 9
FRAME_COLOR_POOL = (1, 2, 3, 5, 6, 7, 8)
CARDINALS = ("up", "down", "left", "right")
NONEMPTY_DIRECTION_SETS = [
    tuple(direction for bit, direction in enumerate(CARDINALS) if mask & (1 << bit))
    for mask in range(1, 1 << len(CARDINALS))
]

GENERATION_ATTEMPTS = 400
STAMP_ATTEMPTS = 120
CANVAS_SIZE = 30
PLACEMENT_PADDING = 1


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        neighbors = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in neighbors:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros_like(local)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _stamp_copy_shift(
    direction: str,
    *,
    frame_bbox: tuple[int, int, int, int],
    stamp_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = stamp_bbox
    if direction == "up":
        return int(fr0 - 1 - sr1), 0
    if direction == "down":
        return int(fr1 + 1 - sr0), 0
    if direction == "left":
        return 0, int(fc0 - 1 - sc1)
    if direction == "right":
        return 0, int(fc1 + 1 - sc0)
    raise AssertionError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each yellow stamp outside its colored frame in the directions encoded by maroon interior markers.

    Input structure:
    - Black background contains several single-color hollow rectangular frames, each with a distinct frame color.
    - Inside every frame there is exactly one connected yellow stamp shape and one or more maroon singleton markers.
    - Each maroon marker lies strictly above, below, left, or right of the stamp while sharing a stamp row or column.

    Rule:
    - Detect every hollow rectangular frame and inspect only its interior.
    - Use the yellow component as the stamp, and map each maroon marker to a cardinal direction relative to that stamp.
    - For every marked direction, place one translated copy of the stamp immediately outside the matching side of the frame, using the frame color instead of yellow.
    - Keep the original frame, interior stamp, and maroon markers unchanged.

    Key insight:
    - The yellow shape is the reusable template, the maroon dots encode which side(s) should receive a copy, and the copy color comes from the surrounding frame rather than from the stamp itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_found = False
    for color0 in np.unique(g):
        color = int(color0)
        if color in (BG, STAMP_COLOR, MARKER_COLOR):
            continue

        for frame_coords in _components(g == color, connectivity=4):
            if not _is_hollow_rectangle(frame_coords):
                continue
            frame_found = True
            fr0, fc0, fr1, fc1 = _bbox(frame_coords)

            interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
            stamp_components = _components(interior == STAMP_COLOR, connectivity=4)
            assert len(stamp_components) == 1, "each frame must contain one connected yellow stamp"
            stamp_coords = stamp_components[0] + np.array([fr0 + 1, fc0 + 1], dtype=int)
            stamp_bbox = _bbox(stamp_coords)

            stamp_rows = {int(r) for r in stamp_coords[:, 0]}
            stamp_cols = {int(c) for c in stamp_coords[:, 1]}
            sr0, sc0, sr1, sc1 = stamp_bbox

            marker_coords = np.argwhere(interior == MARKER_COLOR) + np.array([fr0 + 1, fc0 + 1], dtype=int)
            assert len(marker_coords) >= 1, "each frame must contain at least one maroon marker"

            directions: set[str] = set()
            for r0, c0 in marker_coords:
                r = int(r0)
                c = int(c0)
                if r < sr0 and c in stamp_cols:
                    directions.add("up")
                elif r > sr1 and c in stamp_cols:
                    directions.add("down")
                elif c < sc0 and r in stamp_rows:
                    directions.add("left")
                elif c > sc1 and r in stamp_rows:
                    directions.add("right")
                else:
                    raise AssertionError("marker is not aligned to a cardinal side of the stamp")

            for direction in directions:
                dr, dc = _stamp_copy_shift(direction, frame_bbox=(fr0, fc0, fr1, fc1), stamp_bbox=stamp_bbox)
                for r0, c0 in stamp_coords:
                    rr = int(r0 + dr)
                    cc = int(c0 + dc)
                    assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "stamp copy leaves the grid"
                    assert out[rr, cc] in (BG, color), "stamp copies must not conflict with other objects"
                    out[rr, cc] = color

    assert frame_found, "expected at least one colored frame"
    return out


def _grow_connected_shape(rng: np.random.Generator, height: int, width: int, area: int) -> np.ndarray:
    seed = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        unique_frontier = list(dict.fromkeys(frontier))
        cells.add(unique_frontier[int(rng.integers(len(unique_frontier)))])

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    patch = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=int)
    for r, c in cells:
        patch[int(r - r0), int(c - c0)] = 1
    return patch


def _make_stamp(rng: np.random.Generator) -> np.ndarray:
    choices = [
        (height, width, area)
        for height in range(2, 5)
        for width in range(2, 5)
        for area in range(3, min(7, height * width) + 1)
        if area < height * width
    ]
    for _ in range(STAMP_ATTEMPTS):
        height, width, area = choices[int(rng.integers(len(choices)))]
        patch = _grow_connected_shape(rng, int(height), int(width), int(area))
        stamp_coords = np.argwhere(patch == 1)
        if len({int(r) for r in stamp_coords[:, 0]}) < 2:
            continue
        if len({int(c) for c in stamp_coords[:, 1]}) < 2:
            continue
        if int(patch.sum()) == patch.size:
            continue
        return patch
    raise RuntimeError("failed to generate a valid stamp")


def _build_frame_block(rng: np.random.Generator, frame_color: int) -> np.ndarray:
    stamp = _make_stamp(rng)
    stamp_coords = np.argwhere(stamp == 1)
    stamp_height, stamp_width = map(int, stamp.shape)

    directions = NONEMPTY_DIRECTION_SETS[int(rng.integers(len(NONEMPTY_DIRECTION_SETS)))]
    direction_set = set(directions)

    top_gap = int(rng.integers(2, 4)) if "up" in direction_set else int(rng.integers(1, 3))
    bottom_gap = int(rng.integers(2, 4)) if "down" in direction_set else int(rng.integers(1, 3))
    left_gap = int(rng.integers(2, 4)) if "left" in direction_set else int(rng.integers(1, 3))
    right_gap = int(rng.integers(2, 4)) if "right" in direction_set else int(rng.integers(1, 3))

    frame_height = int(stamp_height + top_gap + bottom_gap + 2)
    frame_width = int(stamp_width + left_gap + right_gap + 2)
    outer_top = int(stamp_height if "up" in direction_set else 0)
    outer_bottom = int(stamp_height if "down" in direction_set else 0)
    outer_left = int(stamp_width if "left" in direction_set else 0)
    outer_right = int(stamp_width if "right" in direction_set else 0)

    block = np.zeros((outer_top + frame_height + outer_bottom, outer_left + frame_width + outer_right), dtype=int)
    fr0 = int(outer_top)
    fc0 = int(outer_left)
    fr1 = int(fr0 + frame_height - 1)
    fc1 = int(fc0 + frame_width - 1)

    block[fr0, fc0 : fc1 + 1] = frame_color
    block[fr1, fc0 : fc1 + 1] = frame_color
    block[fr0 : fr1 + 1, fc0] = frame_color
    block[fr0 : fr1 + 1, fc1] = frame_color

    sr0 = int(fr0 + 1 + top_gap)
    sc0 = int(fc0 + 1 + left_gap)
    for r0, c0 in stamp_coords:
        block[int(sr0 + r0), int(sc0 + c0)] = STAMP_COLOR

    stamp_rows = sorted({int(r) for r, _ in stamp_coords})
    stamp_cols = sorted({int(c) for _, c in stamp_coords})

    if "up" in direction_set:
        marker_row = int(rng.integers(fr0 + 1, sr0))
        marker_col = int(sc0 + stamp_cols[int(rng.integers(len(stamp_cols)))])
        block[marker_row, marker_col] = MARKER_COLOR
    if "down" in direction_set:
        marker_row = int(rng.integers(sr0 + stamp_height, fr1))
        marker_col = int(sc0 + stamp_cols[int(rng.integers(len(stamp_cols)))])
        block[marker_row, marker_col] = MARKER_COLOR
    if "left" in direction_set:
        marker_row = int(sr0 + stamp_rows[int(rng.integers(len(stamp_rows)))])
        marker_col = int(rng.integers(fc0 + 1, sc0))
        block[marker_row, marker_col] = MARKER_COLOR
    if "right" in direction_set:
        marker_row = int(sr0 + stamp_rows[int(rng.integers(len(stamp_rows)))])
        marker_col = int(rng.integers(sc0 + stamp_width, fc1))
        block[marker_row, marker_col] = MARKER_COLOR

    return block


def _find_block_positions(occupied: np.ndarray, block_shape: tuple[int, int], padding: int) -> list[tuple[int, int]]:
    height, width = map(int, occupied.shape)
    block_height, block_width = map(int, block_shape)
    positions: list[tuple[int, int]] = []
    for r in range(height - block_height + 1):
        check_r0 = max(0, r - padding)
        check_r1 = min(height, r + block_height + padding)
        for c in range(width - block_width + 1):
            check_c0 = max(0, c - padding)
            check_c1 = min(width, c + block_width + padding)
            if not occupied[check_r0:check_r1, check_c0:check_c1].any():
                positions.append((int(r), int(c)))
    return positions


def _place_blocks(rng: np.random.Generator, blocks: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)
    placements: list[tuple[np.ndarray, int, int]] = []

    ordered_blocks = sorted(blocks, key=lambda block: (block.shape[0] * block.shape[1]), reverse=True)
    for block in ordered_blocks:
        positions = _find_block_positions(occupied, block.shape, PLACEMENT_PADDING)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(len(positions)))]
        bh, bw = map(int, block.shape)
        canvas[r0 : r0 + bh, c0 : c0 + bw] = block
        occupied[r0 : r0 + bh, c0 : c0 + bw] = True
        placements.append((block, int(r0), int(c0)))

    min_r = min(r0 for _block, r0, _c0 in placements)
    min_c = min(c0 for _block, _r0, c0 in placements)
    max_r = max(r0 + block.shape[0] - 1 for block, r0, _c0 in placements)
    max_c = max(c0 + block.shape[1] - 1 for block, _r0, c0 in placements)

    crop_top = int(rng.integers(0, min(3, min_r + 1)))
    crop_left = int(rng.integers(0, min(3, min_c + 1)))
    crop_bottom = int(rng.integers(0, min(3, CANVAS_SIZE - max_r)))
    crop_right = int(rng.integers(0, min(3, CANVAS_SIZE - max_c)))

    r0 = int(min_r - crop_top)
    c0 = int(min_c - crop_left)
    r1 = int(max_r + crop_bottom)
    c1 = int(max_c + crop_right)
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.choice(np.array([2, 2, 3], dtype=int)))
    frame_colors = [int(color) for color in rng.choice(np.array(FRAME_COLOR_POOL, dtype=int), size=frame_count, replace=False)]
    blocks = [_build_frame_block(rng, frame_color) for frame_color in frame_colors]

    grid = _place_blocks(rng, blocks)
    if grid is None:
        return None

    if grid.ndim != 2 or max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
