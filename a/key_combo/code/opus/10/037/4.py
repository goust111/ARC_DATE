from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
TEAL = 8

ROTATION_K = {
    "up": 0,
    "right": -1,  # 90 degrees clockwise
    "down": 2,
    "left": 1,    # 270 degrees clockwise
}

DIR_OFFSETS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

_GEN_CALL_COUNT = 0
_DIR_COVERAGE_CYCLE = (
    ("up", "right"),
    ("down", "left"),
    ("up", "down", "left"),
    ("right", "up", "down"),
)


def _line_indices(grid: np.ndarray, axis: int) -> list[int]:
    valid = {TEAL, YELLOW}
    if axis == 0:
        return [r for r in range(grid.shape[0]) if all(int(v) in valid for v in grid[r, :])]
    return [c for c in range(grid.shape[1]) if all(int(v) in valid for v in grid[:, c])]


def _locate_cavities(grid: np.ndarray) -> list[dict[str, object]]:
    row_lines = _line_indices(grid, axis=0)
    col_lines = _line_indices(grid, axis=1)

    assert len(row_lines) == 4 and len(col_lines) == 4, "expected a 3x3 lattice with 4 row/col rails"
    assert all(row_lines[i + 1] - row_lines[i] == 4 for i in range(3)), "row rails must be equally spaced"
    assert all(col_lines[i + 1] - col_lines[i] == 4 for i in range(3)), "col rails must be equally spaced"

    cavities: list[dict[str, object]] = []
    for ri in range(3):
        for ci in range(3):
            r0 = row_lines[ri] + 1
            c0 = col_lines[ci] + 1
            r1 = row_lines[ri + 1] - 1
            c1 = col_lines[ci + 1] - 1
            assert r1 - r0 == 2 and c1 - c0 == 2, "each cavity must be 3x3"
            cavities.append(
                {
                    "index": (ri, ci),
                    "bbox": (r0, c0, r1, c1),
                    "nw_cross": (row_lines[ri], col_lines[ci]),
                }
            )
    return cavities


def _marker_direction(grid: np.ndarray, cross: tuple[int, int]) -> tuple[str | None, tuple[int, int] | None]:
    cr, cc = cross
    found: list[tuple[str, tuple[int, int]]] = []
    for name, (dr, dc) in DIR_OFFSETS.items():
        rr, cc2 = cr + dr, cc + dc
        assert 0 <= rr < grid.shape[0] and 0 <= cc2 < grid.shape[1], "marker probe must stay in bounds"
        if int(grid[rr, cc2]) == YELLOW:
            found.append((name, (rr, cc2)))
    assert len(found) <= 1, "a cavity corner may have at most one marker"
    if not found:
        return None, None
    return found[0]


def _extract_input_state(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    cavities = _locate_cavities(grid)
    red_sources: list[dict[str, object]] = []
    marker_cells: list[tuple[int, int]] = []

    for cavity in cavities:
        r0, c0, r1, c1 = cavity["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        has_red = bool(np.any(patch == RED))
        if has_red:
            red_sources.append(cavity)
        marker_dir, marker_pos = _marker_direction(grid, cavity["nw_cross"])
        cavity["marker_dir"] = marker_dir
        cavity["marker_pos"] = marker_pos
        if marker_pos is not None:
            marker_cells.append(marker_pos)

    assert len(red_sources) == 1, "expected exactly one source cavity containing the sprite"
    source = red_sources[0]
    assert source["marker_dir"] is None, "source cavity must not also carry a marker"

    sr0, sc0, sr1, sc1 = source["bbox"]
    sprite = grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()

    red_count = int(np.count_nonzero(sprite == RED))
    assert 3 <= red_count <= 5, "sprite must contain 3-5 red cells"
    assert not np.any((sprite != BLACK) & (sprite != RED)), "source cavity may only contain black and red"

    for cavity in cavities:
        if cavity is source:
            continue
        r0, c0, r1, c1 = cavity["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(patch == RED), "only the source cavity may contain red in the input"
        assert np.all(patch == BLACK), "non-source cavity interiors must start empty"

    assert int(np.count_nonzero(grid == YELLOW)) == len(marker_cells), "all yellow cells must be valid markers"
    return cavities, source, sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode corner-marker orientations and fill each marked lattice cavity with a rotated copy of the source sprite.

    Input structure:
    - Teal rails form a 3x3 lattice of nine closed square cavities, each with a 3x3 black interior.
    - Exactly one cavity contains a red sprite; some other cavities have a single yellow marker near their northwest lattice cross.
    - A marker sits one step up, right, down, or left of that cavity's northwest cross and uses position, not color, to encode an angle.

    Rule:
    - Identify the unique source cavity and take its full 3x3 contents as the sprite template.
    - For every marked cavity, decode marker direction relative to that cavity's northwest cross: up -> 0 degrees, right -> 90 degrees clockwise, down -> 180 degrees, left -> 270 degrees clockwise.
    - Clear every yellow marker back to teal and write the corresponding rotated 3x3 sprite into that cavity; leave the source sprite and unmarked cavities unchanged.

    Key insight:
    - The yellow control pixel does not indicate a destination by color; its orientation within the local lattice corner is the only clue for how the source sprite must be rotated before filling the cavity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cavities, _source, sprite = _extract_input_state(g)
    out = g.copy()

    for cavity in cavities:
        marker_dir = cavity["marker_dir"]
        marker_pos = cavity["marker_pos"]
        if marker_dir is None:
            continue

        rr, cc = marker_pos
        out[rr, cc] = TEAL

        r0, c0, r1, c1 = cavity["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(sprite, ROTATION_K[marker_dir])

    return out


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    cell_set = set(cells)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cell_set)


def _generate_sprite() -> np.ndarray:
    for _ in range(400):
        count = random.randint(3, 5)
        mask = np.zeros((3, 3), dtype=bool)
        seed = (random.randrange(3), random.randrange(3))
        mask[seed] = True

        while int(mask.sum()) < count:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < 3 and 0 <= nc < 3 and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != count or not _connected(mask):
            continue
        rows, cols = np.where(mask)
        if len(set(rows)) < 2 or len(set(cols)) < 2:
            continue

        rotations = [np.rot90(mask, k) for k in range(4)]
        if any(np.array_equal(rotations[0], rotations[k]) for k in (1, 2, 3)):
            continue

        sprite = np.zeros((3, 3), dtype=int)
        sprite[mask] = RED
        return sprite
    raise RuntimeError("failed to generate a non-symmetric 3x3 sprite")


def _build_lattice(margin: int) -> tuple[np.ndarray, list[int]]:
    size = 2 * margin + 13
    lines = [margin + 4 * i for i in range(4)]
    grid = np.zeros((size, size), dtype=int)
    grid[lines, :] = TEAL
    grid[:, lines] = TEAL
    return grid, lines


def _cavity_bbox(lines: list[int], rc_index: tuple[int, int]) -> tuple[int, int, int, int]:
    ri, ci = rc_index
    return lines[ri] + 1, lines[ci] + 1, lines[ri + 1] - 1, lines[ci + 1] - 1


def _place_marker(grid: np.ndarray, lines: list[int], rc_index: tuple[int, int], direction: str) -> None:
    ri, ci = rc_index
    cross = (lines[ri], lines[ci])
    dr, dc = DIR_OFFSETS[direction]
    grid[cross[0] + dr, cross[1] + dc] = YELLOW


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        cavities, source, sprite = _extract_input_state(inp)
    except Exception:
        return False

    if out.shape != inp.shape or np.any(out == YELLOW):
        return False

    source_bbox = source["bbox"]
    if not np.array_equal(
        out[source_bbox[0] : source_bbox[2] + 1, source_bbox[1] : source_bbox[3] + 1],
        sprite,
    ):
        return False

    for cavity in cavities:
        r0, c0, r1, c1 = cavity["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        marker_dir = cavity["marker_dir"]
        marker_pos = cavity["marker_pos"]

        if marker_dir is None:
            if cavity is source:
                continue
            if not np.all(patch == BLACK):
                return False
        else:
            expected = np.rot90(sprite, ROTATION_K[marker_dir])
            if not np.array_equal(patch, expected):
                return False
            rr, cc = marker_pos
            if int(out[rr, cc]) != TEAL:
                return False

    row_lines = _line_indices(inp, axis=0)
    col_lines = _line_indices(inp, axis=1)
    if not all(np.all(out[r, :] == TEAL) for r in row_lines):
        return False
    if not all(np.all(out[:, c] == TEAL) for c in col_lines):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    global _GEN_CALL_COUNT

    margin = [1, 2, 3][_GEN_CALL_COUNT % 3]
    marker_count = 3 if (_GEN_CALL_COUNT % 2 == 0) else 4
    required_dirs = _DIR_COVERAGE_CYCLE[_GEN_CALL_COUNT % len(_DIR_COVERAGE_CYCLE)]
    _GEN_CALL_COUNT += 1

    grid, lines = _build_lattice(margin)
    cavities = [(r, c) for r in range(3) for c in range(3)]
    source_idx = cavities[(_GEN_CALL_COUNT + random.randrange(9)) % 9]

    sprite = _generate_sprite()
    r0, c0, r1, c1 = _cavity_bbox(lines, source_idx)
    grid[r0 : r1 + 1, c0 : c1 + 1] = sprite

    available = [idx for idx in cavities if idx != source_idx]
    marked = random.sample(available, marker_count)

    dirs = list(required_dirs[:marker_count])
    while len(dirs) < marker_count:
        dirs.append(random.choice(tuple(DIR_OFFSETS)))
    random.shuffle(dirs)
    if len(set(dirs)) < 2:
        return None

    for idx, direction in zip(marked, dirs):
        _place_marker(grid, lines, idx, direction)

    try:
        parsed_cavities, source, parsed_sprite = _extract_input_state(grid)
    except Exception:
        return None

    found_dirs = [cavity["marker_dir"] for cavity in parsed_cavities if cavity["marker_dir"] is not None]
    if len(found_dirs) != marker_count or len(set(found_dirs)) < 2:
        return None
    if not np.array_equal(parsed_sprite, sprite) or tuple(source["index"]) != source_idx:
        return None

    out = main(grid)
    if not _semantic_check(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-rotation input")
