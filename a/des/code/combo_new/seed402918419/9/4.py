from collections import Counter, deque
import random

import numpy as np


AZURE = 8
GREY = 5
MAROON = 9
ARC_MAX = 30
ROOM_SIZE_MIN = 5
ROOM_SIZE_MAX = 8
PATTERN_SIZE_MIN = 2
PATTERN_SIZE_MAX = 4
MAX_GENERATION_ATTEMPTS = 300
LAYOUT_OPTIONS = ((1, 3), (3, 1), (2, 2))
DIRECTIONS = ("up", "down", "left", "right")
PATTERN_COLORS = (0, 1, 2, 3, 4, 6, 7)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _room_spans(lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:], strict=False):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _majority_color(values: list[int]) -> int:
    counts = Counter(values)
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "projection colors must have a unique majority"
    return int(ranked[0][0])


def _direction_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    if mr < r0 and c0 <= mc <= c1:
        return "up"
    if mr > r1 and c0 <= mc <= c1:
        return "down"
    if mc < c0 and r0 <= mr <= r1:
        return "left"
    if mc > c1 and r0 <= mr <= r1:
        return "right"

    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    dr = mr - cr
    dc = mc - cc
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each room's multicolor patch onto the marker-indicated interior wall as a 1D majority-color strip.

    Input structure:
    - An Azure background is enclosed and subdivided by one-pixel Grey fence rows and columns, creating rectangular rooms.
    - Each room contains one connected multicolor patch made of non-Azure, non-Grey, non-Maroon cells.
    - Each room also contains one isolated Maroon marker pixel placed above, below, left, or right of the patch.

    Rule:
    - For every room, identify the patch and read the Maroon marker's relative position to determine the projection direction.
    - If the direction is up or down, compress the patch column by column: each occupied column becomes one pixel colored by that column's majority color.
    - If the direction is left or right, compress the patch row by row in the same way.
    - Erase the original patch and marker, then place the centered 1D strip flush against the corresponding interior Grey wall of that room.

    Key insight:
    - The Maroon marker encodes direction only through relative placement, and the output strip is formed by per-column or per-row majority colors rather than by copying an outer edge of the patch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _dominant_color(g)
    fence = int(g[0, 0])
    assert fence != bg
    assert np.all(g[0, :] == fence) and np.all(g[-1, :] == fence)
    assert np.all(g[:, 0] == fence) and np.all(g[:, -1] == fence)

    fence_rows = [r for r in range(h) if np.all(g[r, :] == fence)]
    fence_cols = [c for c in range(w) if np.all(g[:, c] == fence)]
    row_spans = _room_spans(fence_rows)
    col_spans = _room_spans(fence_cols)
    assert row_spans and col_spans

    out = g.copy()
    for rr0, rr1 in row_spans:
        for cc0, cc1 in col_spans:
            room = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            room_h, room_w = room.shape

            marker_positions = np.argwhere(room == MAROON)
            assert len(marker_positions) == 1, "each room must contain exactly one marker"
            marker = tuple(map(int, marker_positions[0]))

            pattern_mask = (room != bg) & (room != fence) & (room != MAROON)
            comps = _components(pattern_mask)
            assert len(comps) == 1, "each room must contain exactly one patch"
            coords = comps[0]
            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            direction = _direction_from_marker(marker, (r0, c0, r1, c1))

            room_out = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
            room_out[pattern_mask | (room == MAROON)] = bg

            if direction in ("up", "down"):
                strip = []
                for c in range(c0, c1 + 1):
                    values = [int(room[r, c]) for r in range(r0, r1 + 1) if pattern_mask[r, c]]
                    assert values
                    strip.append(_majority_color(values))
                start = (room_w - len(strip)) // 2
                target_row = 0 if direction == "up" else room_h - 1
                for offset, color in enumerate(strip):
                    room_out[target_row, start + offset] = color
            else:
                strip = []
                for r in range(r0, r1 + 1):
                    values = [int(room[r, c]) for c in range(c0, c1 + 1) if pattern_mask[r, c]]
                    assert values
                    strip.append(_majority_color(values))
                start = (room_h - len(strip)) // 2
                target_col = 0 if direction == "left" else room_w - 1
                for offset, color in enumerate(strip):
                    room_out[start + offset, target_col] = color

    return out


def _make_projectable_patch(height: int, width: int, direction: str, colors: list[int]) -> np.ndarray:
    patch = np.empty((height, width), dtype=int)
    if direction in ("up", "down"):
        majorities = colors[:] + [random.choice(colors) for _ in range(width - len(colors))]
        random.shuffle(majorities)
        for c, majority in enumerate(majorities):
            if height == 2:
                patch[:, c] = majority
                continue
            majority_count = random.randint(height // 2 + 1, height)
            values = [majority] * majority_count
            for _ in range(height - majority_count):
                values.append(random.choice([color for color in colors if color != majority]))
            random.shuffle(values)
            patch[:, c] = values
    else:
        majorities = colors[:] + [random.choice(colors) for _ in range(height - len(colors))]
        random.shuffle(majorities)
        for r, majority in enumerate(majorities):
            if width == 2:
                patch[r, :] = majority
                continue
            majority_count = random.randint(width // 2 + 1, width)
            values = [majority] * majority_count
            for _ in range(width - majority_count):
                values.append(random.choice([color for color in colors if color != majority]))
            random.shuffle(values)
            patch[r, :] = values

    assert set(map(int, np.unique(patch))) == set(colors)
    return patch


def _sample_room_payload(room_h: int, room_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    direction = random.choice(DIRECTIONS)
    gap = random.randint(1, 2)
    required_side = gap + 2

    if direction in ("up", "down"):
        max_patch_h = min(PATTERN_SIZE_MAX, room_h - required_side)
        max_patch_w = min(PATTERN_SIZE_MAX, room_w)
        if max_patch_h < PATTERN_SIZE_MIN or max_patch_w < PATTERN_SIZE_MIN:
            return None
        patch_h = random.randint(PATTERN_SIZE_MIN, max_patch_h)
        patch_w = random.randint(PATTERN_SIZE_MIN, max_patch_w)
        left = random.randint(0, room_w - patch_w)
        if direction == "up":
            top = random.randint(required_side, room_h - patch_h)
            marker_r = top - gap - 1
            marker_c = random.randint(left, left + patch_w - 1)
        else:
            top_max = room_h - patch_h - required_side
            if top_max < 0:
                return None
            top = random.randint(0, top_max)
            marker_r = top + patch_h + gap
            marker_c = random.randint(left, left + patch_w - 1)
        max_colors = min(3, patch_w)
    else:
        max_patch_h = min(PATTERN_SIZE_MAX, room_h)
        max_patch_w = min(PATTERN_SIZE_MAX, room_w - required_side)
        if max_patch_h < PATTERN_SIZE_MIN or max_patch_w < PATTERN_SIZE_MIN:
            return None
        patch_h = random.randint(PATTERN_SIZE_MIN, max_patch_h)
        patch_w = random.randint(PATTERN_SIZE_MIN, max_patch_w)
        top = random.randint(0, room_h - patch_h)
        if direction == "left":
            left = random.randint(required_side, room_w - patch_w)
            marker_r = random.randint(top, top + patch_h - 1)
            marker_c = left - gap - 1
        else:
            left_max = room_w - patch_w - required_side
            if left_max < 0:
                return None
            left = random.randint(0, left_max)
            marker_r = random.randint(top, top + patch_h - 1)
            marker_c = left + patch_w + gap
        max_colors = min(3, patch_h)

    color_count = 2 if max_colors == 2 else random.choice((2, 3))
    colors = random.sample(PATTERN_COLORS, color_count)
    patch = _make_projectable_patch(patch_h, patch_w, direction, colors)

    inp_room = np.full((room_h, room_w), AZURE, dtype=int)
    out_room = np.full((room_h, room_w), AZURE, dtype=int)
    inp_room[top : top + patch_h, left : left + patch_w] = patch
    inp_room[marker_r, marker_c] = MAROON

    if direction in ("up", "down"):
        strip = []
        for c in range(patch_w):
            strip.append(_majority_color(list(map(int, patch[:, c]))))
        start = (room_w - patch_w) // 2
        target_row = 0 if direction == "up" else room_h - 1
        out_room[target_row, start : start + patch_w] = strip
    else:
        strip = []
        for r in range(patch_h):
            strip.append(_majority_color(list(map(int, patch[r, :]))))
        start = (room_h - patch_h) // 2
        target_col = 0 if direction == "left" else room_w - 1
        out_room[start : start + patch_h, target_col] = strip

    return inp_room, out_room


def _sample_layout() -> tuple[list[int], list[int]]:
    room_rows, room_cols = random.choice(LAYOUT_OPTIONS)
    heights = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_rows)]
    widths = [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(room_cols)]
    total_h = sum(heights) + room_rows + 1
    total_w = sum(widths) + room_cols + 1
    assert total_h <= ARC_MAX and total_w <= ARC_MAX
    return heights, widths


def _line_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    pos = 0
    for size in sizes:
        pos += size + 1
        positions.append(pos)
    return positions


def _gen_attempt() -> np.ndarray | None:
    heights, widths = _sample_layout()
    h = sum(heights) + len(heights) + 1
    w = sum(widths) + len(widths) + 1

    inp = np.full((h, w), AZURE, dtype=int)
    out = np.full((h, w), AZURE, dtype=int)
    inp[0, :] = inp[-1, :] = inp[:, 0] = inp[:, -1] = GREY
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = GREY

    row_lines = _line_positions(heights)
    col_lines = _line_positions(widths)
    for r in row_lines[1:-1]:
        inp[r, :] = GREY
        out[r, :] = GREY
    for c in col_lines[1:-1]:
        inp[:, c] = GREY
        out[:, c] = GREY

    row_starts = [row_lines[i] + 1 for i in range(len(heights))]
    col_starts = [col_lines[i] + 1 for i in range(len(widths))]
    for r0, room_h in zip(row_starts, heights, strict=False):
        for c0, room_w in zip(col_starts, widths, strict=False):
            payload = _sample_room_payload(room_h, room_w)
            if payload is None:
                return None
            inp_room, out_room = payload
            inp[r0 : r0 + room_h, c0 : c0 + room_w] = inp_room
            out[r0 : r0 + room_h, c0 : c0 + room_w] = out_room

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-room ARC input")
