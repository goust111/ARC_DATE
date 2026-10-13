from __future__ import annotations

from collections import Counter, deque

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 5
FILL_COLOR = 8
BODY_COLORS = (1, 2, 3, 4, 6, 7, 9)
ROW_OPTIONS = (3, 4)
COL_COUNT = 3
MAX_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[...] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, k=(-int(quarter_turns)) % 4)


def marker_coord(r0: int, c0: int, shape: tuple[int, int], rotation: int) -> tuple[int, int]:
    h, w = map(int, shape)
    corners = (
        (r0 - 1, c0 - 1),
        (r0 - 1, c0 + w),
        (r0 + h, c0 + w),
        (r0 + h, c0 - 1),
    )
    return corners[int(rotation) % 4]


def _sprite_records(grid: np.ndarray, body_color: int) -> list[dict]:
    records: list[dict] = []
    for coords in components_mask(grid == body_color):
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = crop_bbox(grid == body_color, (r0, c0, r1, c1))
        markers = []
        for rotation in range(4):
            mr, mc = marker_coord(r0, c0, mask.shape, rotation)
            if 0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1] and grid[mr, mc] == MARKER_COLOR:
                markers.append(rotation)
        assert len(markers) == 1, "each sprite must have exactly one corner marker"
        records.append({"r0": r0, "c0": c0, "mask": mask, "rot": int(markers[0])})
    assert records, "expected at least one sprite"
    return records


def _step_from_full_lines(values: list[list[int]]) -> int:
    diffs = set()
    for seq in values:
        if len(seq) < 2:
            continue
        seq_diffs = {(seq[i + 1] - seq[i]) % 4 for i in range(len(seq) - 1)}
        assert len(seq_diffs) == 1, "full line must imply a single rotation increment"
        diffs.add(next(iter(seq_diffs)))
    assert len(diffs) == 1, "full lines must agree on the same increment"
    return int(next(iter(diffs)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a sprite lattice by decoding rotation from each corner marker and stamping teal rotated copies into the missing slots.

    Input structure:
    - A black background contains one repeated foreground sprite color plus grey singleton markers.
    - Each sprite is a 4-connected non-rectangular polyomino, and each sprite bbox has exactly one grey marker one cell diagonally outside one of its four corners.
    - The sprites occupy a rectangular row-column lattice with some empty slots, while at least one full row and one full column are present.

    Rule:
    - Read each marker corner as a rotation code: top-left=0, top-right=90 clockwise, bottom-right=180, bottom-left=90 counterclockwise.
    - Invert one observed sprite rotation to recover the canonical sprite mask, and use the observed sprite top-left positions as the lattice row and column coordinates.
    - Infer the constant row-wise and column-wise rotation increments from the complete row and complete column.
    - For every missing lattice slot, compute its rotation from those increments, stamp the corresponding rotated sprite in teal, and add the matching grey corner marker.

    Key insight:
    - The grey corner marker is not decoration: its corner position encodes the sprite's rotation, and the lattice arranges those rotations as an additive row-step plus column-step pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    body_colors = sorted(set(map(int, g.ravel())) - {bg, MARKER_COLOR})
    assert len(body_colors) == 1, "expected exactly one sprite color besides the marker"
    body_color = int(body_colors[0])

    records = _sprite_records(g, body_color)
    reference = records[0]
    canonical_mask = rotate_cw(reference["mask"], -reference["rot"])

    for record in records:
        expected = rotate_cw(canonical_mask, record["rot"])
        assert np.array_equal(expected, record["mask"]), "all sprites must be rotated copies of one mask"

    row_positions = sorted({int(record["r0"]) for record in records})
    col_positions = sorted({int(record["c0"]) for record in records})
    assert len(row_positions) >= 2 and len(col_positions) >= 2, "expected a sprite lattice"

    row_index = {r: i for i, r in enumerate(row_positions)}
    col_index = {c: j for j, c in enumerate(col_positions)}
    rotation_at: dict[tuple[int, int], int] = {}
    for record in records:
        key = (row_index[int(record["r0"])], col_index[int(record["c0"])])
        assert key not in rotation_at, "lattice slots must be unique"
        rotation_at[key] = int(record["rot"])

    full_rows = []
    for i in range(len(row_positions)):
        if all((i, j) in rotation_at for j in range(len(col_positions))):
            full_rows.append([rotation_at[(i, j)] for j in range(len(col_positions))])
    full_cols = []
    for j in range(len(col_positions)):
        if all((i, j) in rotation_at for i in range(len(row_positions))):
            full_cols.append([rotation_at[(i, j)] for i in range(len(row_positions))])
    assert full_rows and full_cols, "expected at least one complete row and one complete column"

    col_step = _step_from_full_lines(full_rows)
    row_step = _step_from_full_lines(full_cols)

    anchor_key, anchor_rotation = next(iter(rotation_at.items()))
    anchor_row, anchor_col = anchor_key
    for (i, j), rotation in rotation_at.items():
        expected = (anchor_rotation + (i - anchor_row) * row_step + (j - anchor_col) * col_step) % 4
        assert expected == rotation, "observed rotations must fit one row-step and one column-step pattern"

    out = g.copy()
    for i, r0 in enumerate(row_positions):
        for j, c0 in enumerate(col_positions):
            if (i, j) in rotation_at:
                continue
            rotation = (anchor_rotation + (i - anchor_row) * row_step + (j - anchor_col) * col_step) % 4
            rotated_mask = rotate_cw(canonical_mask, rotation)
            patch = np.full(rotated_mask.shape, bg, dtype=int)
            patch[rotated_mask] = FILL_COLOR
            paste(out, patch, r0, c0, transparent=bg)
            mr, mc = marker_coord(r0, c0, rotated_mask.shape, rotation)
            out[mr, mc] = MARKER_COLOR
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _random_template(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        canvas_h = int(rng.integers(2, 5))
        canvas_w = int(rng.integers(2, 5))
        cell_count = int(rng.integers(3, min(6, canvas_h * canvas_w - 1) + 1))

        start = (int(rng.integers(0, canvas_h)), int(rng.integers(0, canvas_w)))
        coords = {start}
        while len(coords) < cell_count:
            frontier = set()
            for r, c in coords:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(tuple(frontier.pop()) if len(frontier) == 1 else list(frontier)[int(rng.integers(0, len(frontier)))])

        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if int(mask.sum()) < 3 or int(mask.sum()) > 6:
            continue
        if int(mask.sum()) == mask.size:
            continue
        rotations = [rotate_cw(mask, k) for k in range(4)]
        if len({_mask_signature(rot) for rot in rotations}) != 4:
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric template")


def _choose_missing(rng: np.random.Generator, rows: int, cols: int) -> set[tuple[int, int]] | None:
    all_slots = [(i, j) for i in range(rows) for j in range(cols)]
    for _ in range(200):
        missing_count = int(rng.integers(2, min(4, len(all_slots) - 1) + 1))
        chosen = rng.choice(len(all_slots), size=missing_count, replace=False)
        missing = {all_slots[int(idx)] for idx in chosen}
        present = set(all_slots) - missing
        if not any(all((i, j) in present for j in range(cols)) for i in range(rows)):
            continue
        if not any(all((i, j) in present for i in range(rows)) for j in range(cols)):
            continue
        if not all(any((i, j) in present for j in range(cols)) for i in range(rows)):
            continue
        if not all(any((i, j) in present for i in range(rows)) for j in range(cols)):
            continue
        return missing
    return None


def _expected_output(
    grid: np.ndarray,
    template: np.ndarray,
    row_positions: list[int],
    col_positions: list[int],
    missing: set[tuple[int, int]],
    base_rotation: int,
    row_step: int,
    col_step: int,
) -> np.ndarray:
    out = grid.copy()
    for i, r0 in enumerate(row_positions):
        for j, c0 in enumerate(col_positions):
            if (i, j) not in missing:
                continue
            rotation = (base_rotation + i * row_step + j * col_step) % 4
            mask = rotate_cw(template, rotation)
            patch = np.full(mask.shape, BACKGROUND, dtype=int)
            patch[mask] = FILL_COLOR
            paste(out, patch, r0, c0, transparent=BACKGROUND)
            mr, mc = marker_coord(r0, c0, mask.shape, rotation)
            out[mr, mc] = MARKER_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    template = _random_template(rng)
    rotations = [rotate_cw(template, k) for k in range(4)]
    max_h = max(int(mask.shape[0]) for mask in rotations)
    max_w = max(int(mask.shape[1]) for mask in rotations)

    rows = int(rng.choice(ROW_OPTIONS))
    cols = COL_COUNT
    missing = _choose_missing(rng, rows, cols)
    if missing is None:
        return None

    row_pitch = int(rng.integers(max_h + 2, max_h + 5))
    col_pitch = int(rng.integers(max_w + 2, max_w + 5))
    top_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))

    row_positions = [top_margin + i * row_pitch for i in range(rows)]
    col_positions = [left_margin + j * col_pitch for j in range(cols)]
    height = row_positions[-1] + max_h + 1 + bottom_margin
    width = col_positions[-1] + max_w + 1 + right_margin
    if height > 30 or width > 30:
        return None

    body_color = int(rng.choice(BODY_COLORS))
    base_rotation = int(rng.integers(0, 4))
    row_step = int(rng.integers(0, 4))
    col_step = int(rng.integers(0, 4))

    grid = np.full((height, width), BACKGROUND, dtype=int)
    present_count = 0
    for i, r0 in enumerate(row_positions):
        for j, c0 in enumerate(col_positions):
            if (i, j) in missing:
                continue
            present_count += 1
            rotation = (base_rotation + i * row_step + j * col_step) % 4
            mask = rotations[rotation]
            patch = np.full(mask.shape, BACKGROUND, dtype=int)
            patch[mask] = body_color
            target = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
            if np.any((patch != BACKGROUND) & (target != BACKGROUND)):
                return None
            paste(grid, patch, r0, c0, transparent=BACKGROUND)
            mr, mc = marker_coord(r0, c0, mask.shape, rotation)
            if not (0 <= mr < height and 0 <= mc < width):
                return None
            if grid[mr, mc] != BACKGROUND:
                return None
            grid[mr, mc] = MARKER_COLOR

    if present_count < 5:
        return None
    if len(components_mask(grid == body_color)) != present_count:
        return None
    if int(np.count_nonzero(grid == MARKER_COLOR)) != present_count:
        return None

    expected = _expected_output(
        grid=grid,
        template=template,
        row_positions=row_positions,
        col_positions=col_positions,
        missing=missing,
        base_rotation=base_rotation,
        row_step=row_step,
        col_step=col_step,
    )
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
