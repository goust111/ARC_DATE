from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


FRAME_COLOR = Color.TEAL
SPRITE_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.MAROON,
]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_nonzero(arr: np.ndarray) -> np.ndarray:
    box = _bbox(arr != Color.BLACK)
    if box is None:
        return arr[:0, :0].copy()
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return tuple(sorted((int(r), int(c)) for r, c in arr.tolist()))


def _unique_transforms(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    pts = [(int(r), int(c)) for r, c in coords]
    variants: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for transformed in (
        [(r, c) for r, c in pts],
        [(r, -c) for r, c in pts],
        [(-r, c) for r, c in pts],
        [(-r, -c) for r, c in pts],
        [(c, r) for r, c in pts],
        [(c, -r) for r, c in pts],
        [(-c, r) for r, c in pts],
        [(-c, -r) for r, c in pts],
    ):
        norm = _normalize_coords(transformed)
        if norm not in seen:
            seen.add(norm)
            variants.append(norm)
    return variants


BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
]

SHAPE_VARIANTS: list[tuple[tuple[int, int], ...]] = []
_shape_seen = set()
for _shape in BASE_SHAPES:
    for _variant in _unique_transforms(_shape):
        if _variant not in _shape_seen:
            _shape_seen.add(_variant)
            SHAPE_VARIANTS.append(_variant)


def _content_key(patch: np.ndarray) -> bytes:
    return np.ascontiguousarray(patch).tobytes()


def _is_empty_patch(patch: np.ndarray) -> bool:
    return not bool(np.any(patch))


def _shape_signature(patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_nonzero((patch != Color.BLACK).astype(int))
    return cropped.shape, np.ascontiguousarray(cropped).tobytes()


def _cropped_content_signature(patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_nonzero(patch)
    return cropped.shape, np.ascontiguousarray(cropped).tobytes()


def _constant_step(positions: np.ndarray) -> int | None:
    if positions.size < 2:
        return None
    diffs = np.diff(positions)
    if np.any(diffs <= 1):
        return None
    if not np.all(diffs == diffs[0]):
        return None
    return int(diffs[0])


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[int], list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []
    for color in map(int, np.unique(g)):
        row_lines = np.flatnonzero(np.all(g == color, axis=1))
        col_lines = np.flatnonzero(np.all(g == color, axis=0))
        if row_lines.size < 2 or col_lines.size < 2:
            continue
        if row_lines[0] != 0 or row_lines[-1] != h - 1 or col_lines[0] != 0 or col_lines[-1] != w - 1:
            continue
        row_step = _constant_step(row_lines)
        col_step = _constant_step(col_lines)
        if row_step is None or col_step is None:
            continue
        candidates.append((color, row_lines.tolist(), col_lines.tolist(), row_step, col_step))

    assert candidates, "failed to detect a periodic lattice frame"
    frame_color, row_lines, col_lines, row_step, col_step = max(
        candidates,
        key=lambda item: (len(item[1]) + len(item[2]), item[3] + item[4]),
    )
    chamber_h = row_step - 1
    chamber_w = col_step - 1
    assert chamber_h >= 1 and chamber_w >= 1

    rows = len(row_lines) - 1
    cols = len(col_lines) - 1
    contents = np.zeros((rows, cols, chamber_h, chamber_w), dtype=int)
    for r in range(rows):
        r0 = row_lines[r] + 1
        r1 = row_lines[r + 1]
        for c in range(cols):
            c0 = col_lines[c] + 1
            c1 = col_lines[c + 1]
            patch = g[r0:r1, c0:c1]
            assert patch.shape == (chamber_h, chamber_w)
            contents[r, c] = patch
    return int(frame_color), row_lines, col_lines, contents


def _render_contents(contents: np.ndarray, frame_color: int = FRAME_COLOR) -> np.ndarray:
    rows, cols, chamber_h, chamber_w = contents.shape
    out = np.full((rows * (chamber_h + 1) + 1, cols * (chamber_w + 1) + 1), int(frame_color), dtype=int)
    for r in range(rows):
        r0 = 1 + r * (chamber_h + 1)
        for c in range(cols):
            c0 = 1 + c * (chamber_w + 1)
            out[r0 : r0 + chamber_h, c0 : c0 + chamber_w] = contents[r, c]
    return out


def _solve_contents(contents: np.ndarray) -> np.ndarray:
    rows, cols, _ch, _cw = contents.shape
    empty = np.all(contents == Color.BLACK, axis=(2, 3))
    out = contents.copy()

    groups: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if not empty[r, c]:
                groups[_content_key(contents[r, c])].append((r, c))

    for positions in groups.values():
        if len(positions) < 2:
            continue
        patch = contents[positions[0]].copy()
        by_row: dict[int, list[int]] = defaultdict(list)
        by_col: dict[int, list[int]] = defaultdict(list)
        for r, c in positions:
            by_row[r].append(c)
            by_col[c].append(r)

        for r, cols_here in by_row.items():
            ordered = sorted(cols_here)
            for c0, c1 in zip(ordered, ordered[1:]):
                if c1 - c0 <= 1:
                    continue
                if np.all(empty[r, c0 + 1 : c1]):
                    for cc in range(c0 + 1, c1):
                        if _is_empty_patch(out[r, cc]):
                            out[r, cc] = patch
                        else:
                            assert np.array_equal(out[r, cc], patch)

        for c, rows_here in by_col.items():
            ordered = sorted(rows_here)
            for r0, r1 in zip(ordered, ordered[1:]):
                if r1 - r0 <= 1:
                    continue
                if np.all(empty[r0 + 1 : r1, c]):
                    for rr in range(r0 + 1, r1):
                        if _is_empty_patch(out[rr, c]):
                            out[rr, c] = patch
                        else:
                            assert np.array_equal(out[rr, c], patch)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each chamber sprite through the empty chambers between aligned identical anchors on the lattice.

    Input structure:
    - The whole grid is a periodic teal lattice whose 1-cell frame lines partition the canvas into equal black rectangular chambers.
    - A minority of chambers contain small sprites; empty chambers are completely black.
    - Chamber content equality is exact: shape, color, and in-chamber position must all match.

    Rule:
    - Treat every chamber interior as one macro cell whose value is its full interior patch.
    - For each pair of identical non-empty chamber patches in the same macro row or macro column, if every chamber strictly between them is empty, copy that exact patch into each intermediate chamber.
    - Keep the lattice and the original sprites unchanged; same-shape-different-color or same-shape-different-position pairs do not connect.

    Key insight:
    - The pixel grid is a disguise: the real reasoning happens on the chamber lattice, and content matching must use the entire chamber patch rather than a cropped shape alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_color, row_lines, col_lines, contents = _parse_lattice(g)
    solved = _solve_contents(contents)

    out = np.full_like(g, frame_color)
    chamber_h, chamber_w = contents.shape[2], contents.shape[3]
    for r in range(len(row_lines) - 1):
        r0 = row_lines[r] + 1
        for c in range(len(col_lines) - 1):
            c0 = col_lines[c] + 1
            out[r0 : r0 + chamber_h, c0 : c0 + chamber_w] = solved[r, c]
    return out


GEN_RETRIES = 400
GEN_EXTRA_EVENT_PROB = 0.5
GEN_SINGLETON_RANGE = (1, 3)


def _segment_cells(orientation: str, fixed: int, start: int, end: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(fixed, c) for c in range(start, end + 1)]
    return [(r, fixed) for r in range(start, end + 1)]


def _free_cell(cell: tuple[int, int], occupied: set[tuple[int, int]], reserved_fill: set[tuple[int, int]]) -> bool:
    return cell not in occupied and cell not in reserved_fill


def _segment_candidates(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
    orientation: str | None,
) -> list[tuple[str, int, int, int, list[tuple[int, int]]]]:
    options: list[tuple[str, int, int, int, list[tuple[int, int]]]] = []
    orientations = [orientation] if orientation is not None else ["h", "v"]
    for orient in orientations:
        if orient == "h":
            for r in range(rows):
                for c0 in range(cols - 2):
                    for c1 in range(c0 + 2, cols):
                        cells = _segment_cells("h", r, c0, c1)
                        if all(_free_cell(cell, occupied, reserved_fill) for cell in cells):
                            options.append(("h", r, c0, c1, cells))
        else:
            for c in range(cols):
                for r0 in range(rows - 2):
                    for r1 in range(r0 + 2, rows):
                        cells = _segment_cells("v", c, r0, r1)
                        if all(_free_cell(cell, occupied, reserved_fill) for cell in cells):
                            options.append(("v", c, r0, r1, cells))
    return options


def _available_offsets(chamber_h: int, chamber_w: int, shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    return [
        (dr, dc)
        for dr in range(chamber_h - max_r)
        for dc in range(chamber_w - max_c)
    ]


def _make_patch(
    chamber_h: int,
    chamber_w: int,
    shape: tuple[tuple[int, int], ...],
    color: int,
    offset: tuple[int, int],
) -> np.ndarray:
    patch = np.zeros((chamber_h, chamber_w), dtype=int)
    dr, dc = offset
    for r, c in shape:
        patch[dr + r, dc + c] = int(color)
    return patch


def _random_shape_and_offsets(
    chamber_h: int,
    chamber_w: int,
    *,
    require_multiple_offsets: bool,
) -> tuple[tuple[tuple[int, int], ...], list[tuple[int, int]]] | None:
    candidates = []
    minimum = 2 if require_multiple_offsets else 1
    for shape in SHAPE_VARIANTS:
        offsets = _available_offsets(chamber_h, chamber_w, shape)
        if len(offsets) >= minimum:
            candidates.append((shape, offsets))
    if not candidates:
        return None
    return random.choice(candidates)


def _random_unique_patch(
    chamber_h: int,
    chamber_w: int,
    used_keys: set[bytes],
    *,
    require_multiple_offsets: bool = False,
    color: int | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[int, int], int] | None:
    for _ in range(200):
        picked = _random_shape_and_offsets(chamber_h, chamber_w, require_multiple_offsets=require_multiple_offsets)
        if picked is None:
            return None
        shape, offsets = picked
        offset = random.choice(offsets)
        chosen_color = int(color if color is not None else random.choice(SPRITE_COLORS))
        patch = _make_patch(chamber_h, chamber_w, shape, chosen_color, offset)
        key = _content_key(patch)
        if key not in used_keys:
            return patch, shape, offset, chosen_color
    return None


def _place_actual_event(
    contents: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
    used_keys: set[bytes],
    orientation: str,
) -> dict | None:
    rows, cols, chamber_h, chamber_w = contents.shape
    candidates = _segment_candidates(rows, cols, occupied, reserved_fill, orientation)
    if not candidates:
        return None
    for _ in range(200):
        orient, fixed, start, end, cells = random.choice(candidates)
        built = _random_unique_patch(chamber_h, chamber_w, used_keys)
        if built is None:
            return None
        patch, _shape, _offset, _color = built
        first, last = cells[0], cells[-1]
        contents[first] = patch
        contents[last] = patch
        occupied.update((first, last))
        targets = set(cells[1:-1])
        reserved_fill |= targets
        used_keys.add(_content_key(patch))
        return {
            "type": "actual",
            "orientation": orient,
            "endpoints": (first, last),
            "targets": targets,
            "patch": patch.copy(),
        }
    return None


def _place_blocked_pair(
    contents: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
    used_keys: set[bytes],
) -> dict | None:
    rows, cols, chamber_h, chamber_w = contents.shape
    candidates = _segment_candidates(rows, cols, occupied, reserved_fill, None)
    if not candidates:
        return None
    for _ in range(200):
        orient, fixed, start, end, cells = random.choice(candidates)
        endpoint_built = _random_unique_patch(chamber_h, chamber_w, used_keys)
        blocker_built = _random_unique_patch(chamber_h, chamber_w, used_keys | {_content_key(endpoint_built[0])} if endpoint_built else used_keys)
        if endpoint_built is None or blocker_built is None:
            continue
        endpoint_patch = endpoint_built[0]
        blocker_patch = blocker_built[0]
        blocker_cell = random.choice(cells[1:-1])
        first, last = cells[0], cells[-1]
        contents[first] = endpoint_patch
        contents[last] = endpoint_patch
        contents[blocker_cell] = blocker_patch
        occupied.update((first, last, blocker_cell))
        used_keys.add(_content_key(endpoint_patch))
        used_keys.add(_content_key(blocker_patch))
        return {
            "type": "blocked",
            "orientation": orient,
            "endpoints": (first, last),
            "blocker": blocker_cell,
            "gap_cells": set(cells[1:-1]),
            "patch_a": endpoint_patch.copy(),
            "patch_b": blocker_patch.copy(),
        }
    return None


def _place_diff_color_pair(
    contents: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
    used_keys: set[bytes],
) -> dict | None:
    rows, cols, chamber_h, chamber_w = contents.shape
    candidates = _segment_candidates(rows, cols, occupied, reserved_fill, None)
    if not candidates:
        return None
    for _ in range(200):
        orient, fixed, start, end, cells = random.choice(candidates)
        picked = _random_shape_and_offsets(chamber_h, chamber_w, require_multiple_offsets=False)
        if picked is None:
            return None
        shape, offsets = picked
        offset = random.choice(offsets)
        color_a, color_b = random.sample(SPRITE_COLORS, 2)
        patch_a = _make_patch(chamber_h, chamber_w, shape, color_a, offset)
        patch_b = _make_patch(chamber_h, chamber_w, shape, color_b, offset)
        key_a = _content_key(patch_a)
        key_b = _content_key(patch_b)
        if key_a == key_b or key_a in used_keys or key_b in used_keys:
            continue
        first, last = cells[0], cells[-1]
        contents[first] = patch_a
        contents[last] = patch_b
        occupied.update((first, last))
        used_keys.update((key_a, key_b))
        return {
            "type": "diff_color",
            "orientation": orient,
            "endpoints": (first, last),
            "gap_cells": set(cells[1:-1]),
            "patch_a": patch_a.copy(),
            "patch_b": patch_b.copy(),
        }
    return None


def _place_shift_pair(
    contents: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
    used_keys: set[bytes],
) -> dict | None:
    rows, cols, chamber_h, chamber_w = contents.shape
    candidates = _segment_candidates(rows, cols, occupied, reserved_fill, None)
    if not candidates:
        return None
    for _ in range(200):
        orient, fixed, start, end, cells = random.choice(candidates)
        picked = _random_shape_and_offsets(chamber_h, chamber_w, require_multiple_offsets=True)
        if picked is None:
            return None
        shape, offsets = picked
        offset_a, offset_b = random.sample(offsets, 2)
        color = random.choice(SPRITE_COLORS)
        patch_a = _make_patch(chamber_h, chamber_w, shape, color, offset_a)
        patch_b = _make_patch(chamber_h, chamber_w, shape, color, offset_b)
        key_a = _content_key(patch_a)
        key_b = _content_key(patch_b)
        if key_a == key_b or key_a in used_keys or key_b in used_keys:
            continue
        first, last = cells[0], cells[-1]
        contents[first] = patch_a
        contents[last] = patch_b
        occupied.update((first, last))
        used_keys.update((key_a, key_b))
        return {
            "type": "shift",
            "orientation": orient,
            "endpoints": (first, last),
            "gap_cells": set(cells[1:-1]),
            "patch_a": patch_a.copy(),
            "patch_b": patch_b.copy(),
        }
    return None


def _place_singletons(
    contents: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_fill: set[tuple[int, int]],
    used_keys: set[bytes],
    count: int,
) -> bool:
    rows, cols, chamber_h, chamber_w = contents.shape
    free_cells = [(r, c) for r in range(rows) for c in range(cols) if _free_cell((r, c), occupied, reserved_fill)]
    random.shuffle(free_cells)
    if len(free_cells) < count:
        return False
    placed = 0
    for cell in free_cells:
        built = _random_unique_patch(chamber_h, chamber_w, used_keys)
        if built is None:
            return False
        patch = built[0]
        contents[cell] = patch
        occupied.add(cell)
        used_keys.add(_content_key(patch))
        placed += 1
        if placed == count:
            return True
    return False


def _new_nonempty_cells(before: np.ndarray, after: np.ndarray) -> set[tuple[int, int]]:
    before_empty = np.all(before == Color.BLACK, axis=(2, 3))
    after_nonempty = np.any(after != Color.BLACK, axis=(2, 3))
    return {
        (r, c)
        for r in range(before.shape[0])
        for c in range(before.shape[1])
        if before_empty[r, c] and after_nonempty[r, c]
    }


def _validate_case(contents: np.ndarray, metadata: list[dict]) -> bool:
    solved = _solve_contents(contents)
    actual_targets = set()
    actual_orientations = set()
    for item in metadata:
        if item["type"] == "actual":
            actual_targets |= item["targets"]
            actual_orientations.add(item["orientation"])
            for cell in item["targets"]:
                if not np.array_equal(solved[cell], item["patch"]):
                    return False
        elif item["type"] == "blocked":
            if _content_key(item["patch_a"]) == _content_key(item["patch_b"]):
                return False
            for cell in item["gap_cells"] - {item["blocker"]}:
                if not _is_empty_patch(solved[cell]):
                    return False
        elif item["type"] == "diff_color":
            if _shape_signature(item["patch_a"]) != _shape_signature(item["patch_b"]):
                return False
            if _content_key(item["patch_a"]) == _content_key(item["patch_b"]):
                return False
            for cell in item["gap_cells"]:
                if not _is_empty_patch(solved[cell]):
                    return False
        elif item["type"] == "shift":
            if _cropped_content_signature(item["patch_a"]) != _cropped_content_signature(item["patch_b"]):
                return False
            if _content_key(item["patch_a"]) == _content_key(item["patch_b"]):
                return False
            for cell in item["gap_cells"]:
                if not _is_empty_patch(solved[cell]):
                    return False

    if actual_orientations != {"h", "v"}:
        return False
    if _new_nonempty_cells(contents, solved) != actual_targets:
        return False
    return bool(actual_targets)


def _attempt_generate() -> np.ndarray | None:
    chamber_h = random.randint(3, 5)
    chamber_w = random.randint(3, 5)
    max_rows = min(5, (30 - 1) // (chamber_h + 1))
    max_cols = min(5, (30 - 1) // (chamber_w + 1))
    if max_rows < 4 or max_cols < 4:
        return None
    rows = random.randint(4, max_rows)
    cols = random.randint(4, max_cols)

    contents = np.zeros((rows, cols, chamber_h, chamber_w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved_fill: set[tuple[int, int]] = set()
    used_keys: set[bytes] = set()
    metadata: list[dict] = []

    horizontal = _place_actual_event(contents, occupied, reserved_fill, used_keys, "h")
    vertical = _place_actual_event(contents, occupied, reserved_fill, used_keys, "v")
    if horizontal is None or vertical is None:
        return None
    metadata.extend((horizontal, vertical))

    if random.random() < GEN_EXTRA_EVENT_PROB:
        extra = _place_actual_event(contents, occupied, reserved_fill, used_keys, random.choice(["h", "v"]))
        if extra is None:
            return None
        metadata.append(extra)

    blocked = _place_blocked_pair(contents, occupied, reserved_fill, used_keys)
    diff_color = _place_diff_color_pair(contents, occupied, reserved_fill, used_keys)
    shifted = _place_shift_pair(contents, occupied, reserved_fill, used_keys)
    if blocked is None or diff_color is None or shifted is None:
        return None
    metadata.extend((blocked, diff_color, shifted))

    singleton_count = random.randint(*GEN_SINGLETON_RANGE)
    if not _place_singletons(contents, occupied, reserved_fill, used_keys, singleton_count):
        return None

    if not _validate_case(contents, metadata):
        return None

    inp = _render_contents(contents, FRAME_COLOR)
    expected = _render_contents(_solve_contents(contents), FRAME_COLOR)
    if not np.array_equal(main(inp), expected):
        return None
    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-stamping puzzle")
