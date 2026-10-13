import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(counts.argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    if r1 != r0:
        coords.extend((r1, c) for c in range(c0, c1 + 1))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _find_legends(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    for r in range(h - 2):
        for c in range(w - 2):
            if not (r == 0 or c == 0 or r + 2 == h - 1 or c + 2 == w - 1):
                continue
            patch = grid[r : r + 3, c : c + 3]
            border = np.array(
                [
                    patch[0, 0],
                    patch[0, 1],
                    patch[0, 2],
                    patch[1, 0],
                    patch[1, 2],
                    patch[2, 0],
                    patch[2, 1],
                    patch[2, 2],
                ],
                dtype=int,
            )
            border_color = int(border[0])
            fill_color = int(patch[1, 1])
            if border_color in (bg, GREY) or fill_color in (bg, GREY, border_color):
                continue
            if not np.all(border == border_color):
                continue
            if border_color in mapping:
                assert mapping[border_color] == fill_color
            else:
                mapping[border_color] = fill_color
    return mapping


def _find_rectangles(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    rects: list[dict[str, object]] = []
    for comp in _components(grid == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        ir0, ir1 = min(rows), max(rows)
        ic0, ic1 = min(cols), max(cols)
        if len(comp) != (ir1 - ir0 + 1) * (ic1 - ic0 + 1):
            continue

        fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
        if fr0 < 0 or fc0 < 0 or fr1 >= h or fc1 >= w:
            continue

        marker_color = None
        markers: list[tuple[int, int]] = []
        valid = True
        for r, c in _perimeter_coords(fr0, fc0, fr1, fc1):
            value = int(grid[r, c])
            if value == GREY:
                continue
            if value == bg:
                valid = False
                break
            if marker_color is None:
                marker_color = value
            elif value != marker_color:
                valid = False
                break
            if (r in (fr0, fr1)) and (c in (fc0, fc1)):
                valid = False
                break
            markers.append((r, c))
        if not valid or marker_color is None or len(markers) not in (2, 3):
            continue

        rects.append(
            {
                "frame": (fr0, fc0, fr1, fc1),
                "marker_color": marker_color,
                "markers": markers,
            }
        )

    rects.sort(key=lambda rect: rect["frame"])
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use border markers to split hollow grey rectangles and fill the resulting subregions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legends = _find_legends(g, bg)
    assert legends, "expected at least one edge legend"

    rects = _find_rectangles(g, bg)
    assert rects, "expected at least one marked rectangle"

    out = g.copy()
    for rect in rects:
        fr0, fc0, fr1, fc1 = rect["frame"]
        marker_color = int(rect["marker_color"])
        assert marker_color in legends, "every marker color must appear in the legend"
        fill_color = legends[marker_color]

        ir0, ic0 = fr0 + 1, fc0 + 1
        ir1, ic1 = fr1 - 1, fc1 - 1
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = fill_color

        for mr, mc in rect["markers"]:
            if mr == fr0 or mr == fr1:
                out[ir0 : ir1 + 1, mc] = GREY
            elif mc == fc0 or mc == fc1:
                out[mr, ic0 : ic1 + 1] = GREY
            else:
                raise AssertionError("marker must lie on the frame border")

        for mr, mc in rect["markers"]:
            out[mr, mc] = marker_color

    return out


def _fits_with_buffer(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    buffer_size: int = 1,
) -> bool:
    h, w = occupied.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - buffer_size)
    cc0 = max(0, c0 - buffer_size)
    rr1 = min(h - 1, r1 + buffer_size)
    cc1 = min(w - 1, c1 + buffer_size)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _reserve(
    occupied: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    buffer_size: int = 1,
) -> None:
    h, w = occupied.shape
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    rr0 = max(0, r0 - buffer_size)
    cc0 = max(0, c0 - buffer_size)
    rr1 = min(h - 1, r1 + buffer_size)
    cc1 = min(w - 1, c1 + buffer_size)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _legend_candidates(height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for c in range(width - 2):
        out.append((0, c))
        if height > 3:
            out.append((height - 3, c))
    for r in range(1, height - 3):
        out.append((r, 0))
        if width > 3:
            out.append((r, width - 3))
    return out


def _place_legends(
    grid: np.ndarray,
    occupied: np.ndarray,
    mapping: dict[int, int],
    rng: random.Random,
) -> bool:
    candidates = _legend_candidates(*grid.shape)
    for border_color, fill_color in mapping.items():
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if not _fits_with_buffer(occupied, r0, c0, 3, 3, buffer_size=1):
                continue
            grid[r0 : r0 + 3, c0 : c0 + 3] = border_color
            grid[r0 + 1, c0 + 1] = fill_color
            _reserve(occupied, r0, c0, 3, 3, buffer_size=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _choose_markers(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _ in range(count):
        sides: list[str] = []
        if len(used_cols) < c1 - c0 - 1:
            sides.extend(["top", "bottom"])
        if len(used_rows) < r1 - r0 - 1:
            sides.extend(["left", "right"])
        side = rng.choice(sides)
        if side in ("top", "bottom"):
            choices = [c for c in range(c0 + 1, c1) if c not in used_cols]
            col = rng.choice(choices)
            used_cols.add(col)
            row = r0 if side == "top" else r1
        else:
            choices = [r for r in range(r0 + 1, r1) if r not in used_rows]
            row = rng.choice(choices)
            used_rows.add(row)
            col = c0 if side == "left" else c1
        markers.append((row, col))
    return markers


def _draw_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    marker_color: int,
    markers: list[tuple[int, int]],
) -> None:
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    for r, c in markers:
        grid[r, c] = marker_color


def _render_expected(
    grid: np.ndarray,
    rects: list[dict[str, object]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["frame"]
        marker_color = int(rect["marker_color"])
        fill_color = mapping[marker_color]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
        for mr, mc in rect["markers"]:
            if mr == r0 or mr == r1:
                out[r0 + 1 : r1, mc] = GREY
            else:
                out[mr, c0 + 1 : c1] = GREY
        for mr, mc in rect["markers"]:
            out[mr, mc] = marker_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(18, 28)
    width = rng.randint(18, 28)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    rect_count = rng.choice((1, 2, 3))
    color_count = rng.randint(1, min(3, rect_count))

    palette = list(COLOR_POOL)
    rng.shuffle(palette)
    marker_colors = palette[:color_count]
    fill_colors = palette[color_count : 2 * color_count]
    mapping = {marker_colors[i]: fill_colors[i] for i in range(color_count)}

    if not _place_legends(grid, occupied, mapping, rng):
        return None

    rect_color_assignment = marker_colors[:]
    while len(rect_color_assignment) < rect_count:
        rect_color_assignment.append(rng.choice(marker_colors))
    rng.shuffle(rect_color_assignment)

    rects: list[dict[str, object]] = []
    for marker_color in rect_color_assignment:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            max_frame_h = min(10, height - 4)
            max_frame_w = min(10, width - 4)
            if rect_count == 3:
                max_frame_h = min(max_frame_h, 8)
                max_frame_w = min(max_frame_w, 8)
            if max_frame_h < 5 or max_frame_w < 5:
                return None

            frame_h = rng.randint(5, max_frame_h)
            frame_w = rng.randint(5, max_frame_w)
            r0 = rng.randint(1, height - frame_h - 1)
            c0 = rng.randint(1, width - frame_w - 1)
            if not _fits_with_buffer(occupied, r0, c0, frame_h, frame_w, buffer_size=1):
                continue

            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1
            marker_count = 2 if rng.random() < 0.5 else 3
            markers = _choose_markers(r0, c0, r1, c1, marker_count, rng)

            _draw_rectangle(grid, r0, c0, r1, c1, marker_color, markers)
            _reserve(occupied, r0, c0, frame_h, frame_w, buffer_size=1)
            rects.append(
                {
                    "frame": (r0, c0, r1, c1),
                    "marker_color": marker_color,
                    "markers": markers,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, rects, mapping)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
