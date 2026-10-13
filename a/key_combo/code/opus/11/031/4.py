from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FRAME_OUTER_SIZES = range(6, 9)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return np.where((rows + cols) % 2 == 0, BLACK, GREY).astype(int)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _unique_majority(values: list[int]) -> int | None:
    if not values:
        return None
    ranked = Counter(values).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0])


def _frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    base = _checkerboard(h, w)
    candidates: list[tuple[int, int, int, int]] = []

    for box_h in FRAME_OUTER_SIZES:
        for box_w in FRAME_OUTER_SIZES:
            if box_h > h or box_w > w:
                continue
            for r0 in range(h - box_h + 1):
                r1 = r0 + box_h - 1
                for c0 in range(w - box_w + 1):
                    c1 = c0 + box_w - 1
                    box = (r0, c0, r1, c1)
                    perimeter = _rect_perimeter(box)
                    if any(int(grid[r, c]) != GREY for r, c in perimeter):
                        continue
                    if not any(int(base[r, c]) == BLACK for r, c in perimeter):
                        continue
                    candidates.append(box)

    candidates.sort()
    accepted: list[tuple[int, int, int, int]] = []
    for box in candidates:
        if all(
            box[2] < other[0]
            or other[2] < box[0]
            or box[3] < other[1]
            or other[3] < box[1]
            for other in accepted
        ):
            accepted.append(box)
    return accepted


def _enclosed_regions(wall_coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(wall_coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    wall = np.zeros((h, w), dtype=bool)
    local = wall_coords - np.array([r0, c0], dtype=int)
    wall[local[:, 0], local[:, 1]] = True
    passable = ~wall

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for dr, dc in N4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & ~outside
    return [coords + np.array([r0, c0], dtype=int) for coords in _components(enclosed)]


def _region_infos(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for box in _frame_boxes(grid):
        r0, c0, r1, c1 = box
        region = np.asarray(
            [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)],
            dtype=int,
        )
        markers: list[tuple[int, int, int]] = []
        dark_values: list[int] = []
        all_values: list[int] = []
        for r_raw, c_raw in region:
            r = int(r_raw)
            c = int(c_raw)
            value = int(grid[r, c])
            if value not in MARKER_COLORS:
                continue
            markers.append((r, c, value))
            all_values.append(value)
            if (r + c) % 2 == 0:
                dark_values.append(value)
        infos.append(
            {
                "box": box,
                "region": region,
                "bbox": _bbox(region),
                "markers": markers,
                "dark_majority": _unique_majority(dark_values),
                "all_majority": _unique_majority(all_values),
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed grey outline by voting only with markers on dark checkerboard squares.

    Input structure:
    - The background is a black/grey checkerboard with top-left cell black, so dark squares are exactly the cells where (r+c) is even.
    - Extra grey pixels form 4-connected outline loops or open grey paths layered onto that checkerboard.
    - Colored one-cell markers (colors 1-4) appear inside some closed cavities and may also appear outside every closed cavity as stray distractors.

    Rule:
    - Use checkerboard parity to classify cells: dark squares are even-parity cells and bright squares are odd-parity cells.
    - Identify the closed grey outline components and the regions they enclose.
    - For each enclosed region, collect its marker pixels but count only those on dark squares; ignore bright-square markers.
    - Fill the entire enclosed region with the unique majority color among those dark-square markers, keep the grey outline unchanged, and erase every marker outside closed regions back to the underlying checkerboard color.

    Key insight:
    - The checkerboard is a hidden filter: parity decides which markers are real votes before the majority color is computed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    base = _checkerboard(h, w)
    out = g.copy()
    filled = np.zeros((h, w), dtype=bool)

    for info in _region_infos(g):
        fill_color = info["dark_majority"]
        if fill_color is None:
            continue
        region = info["region"]
        out[region[:, 0], region[:, 1]] = fill_color
        filled[region[:, 0], region[:, 1]] = True

    marker_mask = np.isin(g, MARKER_COLORS)
    erase_mask = marker_mask & ~filled
    out[erase_mask] = base[erase_mask]
    return out


def _rect_perimeter(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _open_rect_perimeter(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    cells = set(_rect_perimeter(box))
    r0, c0, r1, c1 = box
    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        row = r0 if side == "top" else r1
        gap_len = random.choice([1, 2])
        start = random.randint(c0 + 1, c1 - gap_len)
        for c in range(start, start + gap_len):
            cells.discard((row, c))
    else:
        col = c0 if side == "left" else c1
        gap_len = random.choice([1, 2])
        start = random.randint(r0 + 1, r1 - gap_len)
        for r in range(start, start + gap_len):
            cells.discard((r, col))
    return sorted(cells)


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sample_box(
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    existing: list[tuple[int, int, int, int]],
    margin: int,
    attempts: int = 80,
) -> tuple[int, int, int, int] | None:
    if h < box_h + 2 * margin or w < box_w + 2 * margin:
        return None
    for _ in range(attempts):
        r0 = random.randint(margin, h - box_h - margin)
        c0 = random.randint(margin, w - box_w - margin)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if all(_boxes_separated(box, other, gap=2) for other in existing):
            return box
    return None


def _marker_position_search(
    dark_cells: list[tuple[int, int]],
    light_cells: list[tuple[int, int]],
    need_dark: int,
    need_light: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    requests = ["dark"] * need_dark + ["light"] * need_light
    requests.sort(key=lambda kind: len(dark_cells) if kind == "dark" else len(light_cells))
    chosen: list[tuple[int, int]] = []

    def ok(cell: tuple[int, int]) -> bool:
        return all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen)

    def dfs(idx: int) -> bool:
        if idx == len(requests):
            return True
        pool = dark_cells if requests[idx] == "dark" else light_cells
        options = pool[:]
        random.shuffle(options)
        for cell in options:
            if cell in chosen or not ok(cell):
                continue
            chosen.append(cell)
            if dfs(idx + 1):
                return True
            chosen.pop()
        return False

    if not dfs(0):
        return None

    dark_positions = [cell for cell in chosen if (cell[0] + cell[1]) % 2 == 0]
    light_positions = [cell for cell in chosen if (cell[0] + cell[1]) % 2 == 1]
    if len(dark_positions) != need_dark or len(light_positions) != need_light:
        return None
    random.shuffle(dark_positions)
    random.shuffle(light_positions)
    return dark_positions, light_positions


def _frame_pattern(fill_color: int) -> tuple[list[int], list[int]]:
    other_colors = [c for c in MARKER_COLORS if c != fill_color]
    light_majority = random.choice(other_colors)
    remaining = [c for c in other_colors if c != light_majority]
    pattern = random.choice(["2d3l", "1d3l", "1d4l"])
    if pattern == "2d3l":
        return [fill_color, fill_color], [light_majority, light_majority, light_majority]
    if pattern == "1d3l":
        return [fill_color], [light_majority, light_majority, light_majority]
    light_minor = random.choice(remaining)
    return [fill_color], [light_majority, light_majority, light_majority, light_minor]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _inside_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _place_stray_markers(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    existing_markers: set[tuple[int, int]],
) -> int:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden or int(grid[r, c]) != int(_checkerboard(h, w)[r, c]):
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in existing_markers):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)

    if random.random() < 0.25:
        target = 0
    else:
        target = random.randint(1, min(6, len(candidates)))

    count = 0
    for r, c in candidates:
        if count >= target:
            break
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in existing_markers):
            continue
        grid[r, c] = random.choice(MARKER_COLORS)
        existing_markers.add((r, c))
        count += 1
    return count


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(16, 22)
        w = random.randint(16, 22)
        base = _checkerboard(h, w)
        inp = base.copy()
        out = base.copy()

        frame_count = random.choice([2, 3])
        boxes: list[tuple[int, int, int, int]] = []
        frames: list[dict] = []

        ok = True
        for _frame_idx in range(frame_count):
            box_h = random.randint(6, 8)
            box_w = random.randint(6, 8)
            box = _sample_box(h, w, box_h, box_w, boxes, margin=1)
            if box is None:
                ok = False
                break
            boxes.append(box)

            fill_color = random.choice(MARKER_COLORS)
            dark_colors, light_colors = _frame_pattern(fill_color)
            inside = _inside_cells(box)
            dark_cells = [cell for cell in inside if (cell[0] + cell[1]) % 2 == 0]
            light_cells = [cell for cell in inside if (cell[0] + cell[1]) % 2 == 1]
            pos = _marker_position_search(dark_cells, light_cells, len(dark_colors), len(light_colors))
            if pos is None:
                ok = False
                break
            dark_positions, light_positions = pos
            frames.append(
                {
                    "box": box,
                    "perimeter": _rect_perimeter(box),
                    "fill": fill_color,
                    "dark_markers": list(zip(dark_positions, dark_colors)),
                    "light_markers": list(zip(light_positions, light_colors)),
                }
            )

        if not ok:
            continue

        open_shapes: list[list[tuple[int, int]]] = []
        if random.random() < 0.55:
            open_box = _sample_box(
                h,
                w,
                random.randint(6, 8),
                random.randint(6, 8),
                boxes,
                margin=1,
                attempts=60,
            )
            if open_box is not None:
                boxes.append(open_box)
                open_shapes.append(_open_rect_perimeter(open_box))

        for frame in frames:
            _paint_cells(inp, frame["perimeter"], GREY)
            _paint_cells(out, frame["perimeter"], GREY)
            box = frame["box"]
            r0, c0, r1, c1 = box
            out[r0 + 1 : r1, c0 + 1 : c1] = int(frame["fill"])
            for (r, c), color in frame["dark_markers"] + frame["light_markers"]:
                inp[r, c] = int(color)

        for cells in open_shapes:
            _paint_cells(inp, cells, GREY)
            _paint_cells(out, cells, GREY)

        forbidden: set[tuple[int, int]] = set()
        existing_markers: set[tuple[int, int]] = set()
        for frame in frames:
            box = frame["box"]
            r0, c0, r1, c1 = box
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    forbidden.add((r, c))
            for pos, _ in frame["dark_markers"] + frame["light_markers"]:
                existing_markers.add(pos)
        for cells in open_shapes:
            for cell in cells:
                forbidden.add(cell)

        _place_stray_markers(inp, forbidden, existing_markers)

        if not np.array_equal(main(inp), out):
            continue

        infos = [info for info in _region_infos(inp) if info["markers"]]
        if len(infos) != frame_count:
            continue

        valid = True
        for info in infos:
            if info["dark_majority"] is None or info["all_majority"] is None:
                valid = False
                break
            if info["dark_majority"] == info["all_majority"]:
                valid = False
                break
            if not any((r + c) % 2 == 1 for r, c, _ in info["markers"]):
                valid = False
                break
            if not any((r + c) % 2 == 0 for r, c, _ in info["markers"]):
                valid = False
                break
        if not valid:
            continue

        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
