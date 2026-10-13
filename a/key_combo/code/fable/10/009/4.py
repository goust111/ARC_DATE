import random
from collections import defaultdict

import numpy as np


BLACK = 0
GRAY = 5
TEAL = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]
MAX_ATTEMPTS = 600


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    marker_cells = np.argwhere(grid == TEAL)
    assert marker_cells.shape == (1, 2), "expected exactly one teal marker"
    r, c = map(int, marker_cells[0])
    h, w = grid.shape
    on_edge = (r in {0, h - 1}) or (c in {0, w - 1})
    assert on_edge and not (r in {0, h - 1} and c in {0, w - 1}), "marker must be on a non-corner edge"
    return r, c


def _direction_from_marker(shape: tuple[int, int], marker: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = marker
    if c == 0:
        return 0, -1
    if c == w - 1:
        return 0, 1
    if r == 0:
        return -1, 0
    return 1, 0


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in {BLACK, TEAL} or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return objects


def _sort_key(obj: dict, direction: tuple[int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    dr, dc = direction
    if dc == -1:
        return (c0, r0, c0)
    if dc == 1:
        return (-c1, r0, c0)
    if dr == -1:
        return (r0, c0, r0)
    return (-r1, c0, r0)


def _max_shift(coords: np.ndarray, direction: tuple[int, int], occupied: np.ndarray) -> int:
    h, w = occupied.shape
    dr, dc = direction
    shift = 0
    while True:
        trial = coords + np.array([[dr * (shift + 1), dc * (shift + 1)]], dtype=int)
        ok = True
        for r, c in trial:
            rr, cc = int(r), int(c)
            if rr < 0 or rr >= h or cc < 0 or cc >= w or occupied[rr, cc]:
                ok = False
                break
        if not ok:
            return shift
        shift += 1


def _apply_translation(coords: np.ndarray, direction: tuple[int, int], shift: int) -> np.ndarray:
    dr, dc = direction
    return coords + np.array([[dr * shift, dc * shift]], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide every object toward the teal-marked edge and recolor odd-distance movers gray.

    Input structure:
    - The background is black, and exactly one teal pixel sits on a non-corner edge cell.
    - Several disconnected foreground objects are solid, monochromatic components.
    - Objects may share rows or columns along the slide direction, so they can block one another.

    Rule:
    - The marker edge sets the global slide direction: left/right markers compress rows, top/bottom markers compress columns.
    - Treat each foreground component as a rigid body. Components nearer the target edge stop first, and every later component slides until the next step would hit the border, the teal marker, or an already stopped object.
    - Keep the teal marker fixed. If a component's actual displacement is odd, recolor the whole component gray; if the displacement is even, including zero, keep its original color.

    Key insight:
    - The visible output color of each object depends on the hidden parity of its traveled distance, and that distance is only knowable after performing the collision-based compression in the correct edge-centered reference frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker = _find_marker(g)
    direction = _direction_from_marker(g.shape, marker)

    out = np.full_like(g, BLACK)
    occupied = np.zeros(g.shape, dtype=bool)
    out[marker] = TEAL
    occupied[marker] = True

    objects = sorted(_find_components(g), key=lambda obj: _sort_key(obj, direction))
    for obj in objects:
        shift = _max_shift(obj["coords"], direction, occupied)
        shifted = _apply_translation(obj["coords"], direction, shift)
        color = GRAY if shift % 2 == 1 else obj["color"]
        for r, c in shifted:
            rr, cc = int(r), int(c)
            out[rr, cc] = int(color)
            occupied[rr, cc] = True
    return out


def _row_starts(rows: list[int], marker_row: int) -> dict[int, int]:
    return {row: (1 if row == marker_row else 0) for row in rows}


def _sample_counts(num_rows: int, total_objects: int, marker_index: int) -> list[int]:
    counts = [1] * num_rows
    extra = total_objects - num_rows
    eligible = list(range(num_rows))
    random.shuffle(eligible)
    while extra > 0:
        placed = False
        for idx in eligible:
            if counts[idx] < 2:
                counts[idx] += 1
                extra -= 1
                placed = True
                if extra == 0:
                    break
        if not placed:
            break
    if all(count == 1 for count in counts):
        counts[marker_index] = 2
    return counts


def _sample_lengths(width: int, start: int, count: int) -> list[int] | None:
    for _ in range(120):
        lengths = [random.randint(1, 3) for _ in range(count)]
        total_len = sum(lengths)
        slack = width - start - total_len
        if slack >= count - 1 and slack >= 2:
            return lengths
    return None


def _sample_deltas(slack: int, count: int) -> list[int] | None:
    values = list(range(slack + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        deltas = sorted(random.sample(values, count))
        if any(a % 2 == 1 and b % 2 == 1 for a, b in zip(deltas, deltas[1:])):
            continue
        return deltas
    return None


def _rotate_rows(h: int, w: int, data: dict) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    marker_row = data["marker_row"]
    inp[marker_row, 0] = TEAL
    out[marker_row, 0] = TEAL

    for row in data["rows"]:
        start = 1 if row == marker_row else 0
        cursor = start
        for obj in data["objects_by_row"][row]:
            length = obj["length"]
            color = obj["color"]
            delta = obj["delta"]
            inp[row, cursor + delta : cursor + delta + length] = color
            out_color = GRAY if delta % 2 == 1 else color
            out[row, cursor : cursor + length] = out_color
            cursor += length
    return inp, out


def _rotate_case(inp: np.ndarray, out: np.ndarray, turns: int) -> tuple[np.ndarray, np.ndarray]:
    return np.rot90(inp, turns), np.rot90(out, turns)


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    h = random.randint(8, 14)
    w = random.randint(11, 18)
    marker_row = random.randint(1, h - 2)

    num_rows = random.randint(3, min(4, h - 2))
    other_rows = [r for r in range(1, h - 1) if r != marker_row]
    rows = sorted([marker_row] + random.sample(other_rows, num_rows - 1))
    marker_index = rows.index(marker_row)

    total_objects = random.randint(max(4, num_rows + 1), min(6, len(PALETTE)))
    counts = _sample_counts(num_rows, total_objects, marker_index)
    if sum(counts) != total_objects or not any(count >= 2 for count in counts):
        return None

    starts = _row_starts(rows, marker_row)
    colors = random.sample(PALETTE, total_objects)
    color_iter = iter(colors)

    objects_by_row: dict[int, list[dict]] = defaultdict(list)
    all_deltas = []
    marker_row_has_object = False
    blocked_object_exists = False

    for row, count in zip(rows, counts):
        lengths = _sample_lengths(w, starts[row], count)
        if lengths is None:
            return None
        slack = w - starts[row] - sum(lengths)
        deltas = _sample_deltas(slack, count)
        if deltas is None:
            return None
        for index_in_row, (length, delta) in enumerate(zip(lengths, deltas)):
            objects_by_row[row].append(
                {
                    "length": int(length),
                    "delta": int(delta),
                    "color": int(next(color_iter)),
                    "index_in_row": int(index_in_row),
                }
            )
            all_deltas.append(int(delta))
            if row == marker_row:
                marker_row_has_object = True
            if index_in_row > 0 and delta < sum(lengths[:index_in_row]) + delta:
                blocked_object_exists = True

    if not marker_row_has_object or not blocked_object_exists:
        return None

    if not any(delta == 0 for delta in all_deltas):
        return None
    if not any(delta % 2 == 1 for delta in all_deltas):
        return None
    if not any(delta > 0 and delta % 2 == 0 for delta in all_deltas):
        return None

    data = {"marker_row": marker_row, "rows": rows, "objects_by_row": objects_by_row}
    inp, out = _rotate_rows(h, w, data)
    turns = random.randint(0, 3)
    inp, out = _rotate_case(inp, out, turns)
    return inp, out, data


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    marker = _find_marker(inp)
    solved = main(inp)
    if not np.array_equal(solved, out):
        return False

    objects = _find_components(inp)
    if len(objects) < 4:
        return False

    marker_row, marker_col = marker
    if marker_col in {0, inp.shape[1] - 1}:
        line_objects = [obj for obj in objects if obj["bbox"][0] <= marker_row <= obj["bbox"][2]]
    else:
        line_objects = [obj for obj in objects if obj["bbox"][1] <= marker_col <= obj["bbox"][3]]
    if not line_objects:
        return False

    gray_count = int(np.count_nonzero(out == GRAY))
    if gray_count == 0:
        return False
    preserved_color_exists = any(color in out for color in PALETTE)
    if not preserved_color_exists:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, out, _data = built
        if _semantic_checks(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
