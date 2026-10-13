import random
from collections import Counter

import numpy as np


DOT_COLOR = 9
LEGEND_COUNTS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    pieces: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pieces.append(comp)
    return pieces


def _frame_from_component(coords: list[tuple[int, int]], color: int) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    side = r1 - r0 + 1
    if side < 5 or c1 - c0 + 1 != side:
        return None

    boundary = set()
    for c in range(c0, c1 + 1):
        boundary.add((r0, c))
        boundary.add((r1, c))
    for r in range(r0 + 1, r1):
        boundary.add((r, c0))
        boundary.add((r, c1))

    coord_set = set(coords)
    if len(coord_set) != len(boundary) - 1:
        return None
    if not coord_set.issubset(boundary):
        return None

    missing = list(boundary - coord_set)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    if gap_r == r0:
        gap_side = "top"
    elif gap_r == r1:
        gap_side = "bottom"
    elif gap_c == c0:
        gap_side = "left"
    else:
        gap_side = "right"

    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "gap_side": gap_side,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for comp in _components(grid == color):
            frame = _frame_from_component(comp, color)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "expected at least one square frame with a single border gap"
    return frames


def _legend_mapping(grid: np.ndarray, bg: int, frame_region: np.ndarray) -> dict[int, int]:
    legend_mask = (grid != bg) & ~frame_region
    coords = np.argwhere(legend_mask)
    assert len(coords) > 0, "expected a legend outside the frames"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]

    colors = [int(v) for v in legend[legend != bg]]
    dot_color = Counter(colors).most_common(1)[0][0]

    mapping: dict[int, int] = {}
    for r in range(legend.shape[0]):
        row = legend[r]
        nz = row[row != bg]
        if nz.size == 0:
            continue
        dot_count = int(np.count_nonzero(nz == dot_color))
        swatches = [int(v) for v in nz if int(v) != dot_color]
        if 1 <= dot_count <= 3 and len(swatches) == 1:
            mapping[dot_count] = swatches[0]

    assert mapping, "failed to parse legend mapping"
    return mapping


def _reflect_mask(mask: np.ndarray, gap_side: str) -> np.ndarray:
    if gap_side in ("left", "right"):
        return np.fliplr(mask)
    return np.flipud(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each frame's interior across the axis indicated by its border gap, then
    recolor the combined pattern using the legend entry indexed by overlap size.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _find_frames(g, bg)

    frame_region = np.zeros_like(g, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_region[r0 : r1 + 1, c0 : c1 + 1] = True

    mapping = _legend_mapping(g, bg, frame_region)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern = (interior != bg) & (interior != frame["color"])
        mirrored = _reflect_mask(pattern, frame["gap_side"])
        overlap_count = int(np.logical_and(pattern, mirrored).sum())
        assert overlap_count in mapping, "legend does not cover this overlap count"
        fill_color = mapping[overlap_count]
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[np.logical_or(pattern, mirrored)] = fill_color
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _make_vertical_pattern(side: int, overlap_count: int) -> np.ndarray:
    center = side // 2
    for _ in range(500):
        dominant = random.choice(("left", "right"))
        cells: set[tuple[int, int]] = set()

        if overlap_count % 2 == 1:
            cells.add((random.randrange(side), center))
        if overlap_count >= 2:
            row = random.randrange(side)
            offset = random.randint(1, center)
            cells.add((row, center - offset))
            cells.add((row, center + offset))

        target_size = len(cells) + random.randint(2, min(5, side + 1))

        def in_dominant(col: int) -> bool:
            return col < center if dominant == "left" else col > center

        def mirror(cell: tuple[int, int]) -> tuple[int, int]:
            row, col = cell
            return row, side - 1 - col

        tries = 0
        while len(cells) < target_size and tries < 200:
            tries += 1
            candidates: set[tuple[int, int]] = set()
            for cell in cells:
                for rr, cc in _neighbors4(*cell, side, side):
                    if not in_dominant(cc):
                        continue
                    if (rr, cc) in cells or mirror((rr, cc)) in cells:
                        continue
                    candidates.add((rr, cc))
            if not candidates:
                for rr in range(side):
                    for cc in range(side):
                        if not in_dominant(cc):
                            continue
                        if (rr, cc) in cells or mirror((rr, cc)) in cells:
                            continue
                        candidates.add((rr, cc))
            if not candidates:
                break
            cells.add(random.choice(tuple(candidates)))

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        actual_overlap = int(np.logical_and(mask, np.fliplr(mask)).sum())
        dominant_count = sum(1 for _r, c in cells if in_dominant(c))
        other_count = sum(1 for _r, c in cells if (c > center if dominant == "left" else c < center))
        if actual_overlap == overlap_count and dominant_count >= 2 and dominant_count > other_count:
            if random.random() < 0.5:
                mask = np.flipud(mask)
            return mask

    raise RuntimeError("failed to create a valid asymmetric pattern")


def _make_pattern(interior_side: int, gap_side: str, overlap_count: int) -> np.ndarray:
    mask = _make_vertical_pattern(interior_side, overlap_count)
    if gap_side in ("top", "bottom"):
        mask = mask.T
        if random.random() < 0.5:
            mask = np.fliplr(mask)
    else:
        if random.random() < 0.5:
            mask = np.flipud(mask)
    return mask


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _paint_legend(grid: np.ndarray, entries: list[tuple[int, int]], corner: str) -> None:
    legend_w = 7

    if "t" in corner:
        row0 = 0
    else:
        row0 = grid.shape[0] - 5
    if "l" in corner:
        col0 = 0
    else:
        col0 = grid.shape[1] - legend_w

    for idx, (count, color) in enumerate(entries):
        r = row0 + 2 * idx
        if "l" in corner:
            grid[r, col0] = color
            for j in range(count):
                grid[r, col0 + 2 + 2 * j] = DOT_COLOR
        else:
            grid[r, col0 + legend_w - 1] = color
            for j in range(count):
                grid[r, col0 + legend_w - 3 - 2 * j] = DOT_COLOR


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int, gap_side: str, gap_pos: int, bg: int) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if gap_side == "top":
        grid[r0, c0 + gap_pos] = bg
    elif gap_side == "bottom":
        grid[r1, c0 + gap_pos] = bg
    elif gap_side == "left":
        grid[r0 + gap_pos, c0] = bg
    else:
        grid[r0 + gap_pos, c1] = bg


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    colors = random.sample([c for c in range(10) if c != DOT_COLOR], 6)
    bg, border_color, source_color, c1, c2, c3 = colors
    legend_mapping = {1: c1, 2: c2, 3: c3}

    interior_side = random.choice((5, 7, 9))
    frame_side = interior_side + 2
    frame_count = random.randint(2, 4)

    base_size = frame_side * max(2, (frame_count + 1) // 2) + 10
    min_size = max(22, base_size)
    max_size = max(min_size + 4, frame_side * (frame_count + 1) + 10)

    for _ in range(80):
        height = random.randint(min_size, max_size)
        width = random.randint(min_size, max_size)
        grid = np.full((height, width), bg, dtype=int)
        expected = grid.copy()

        corner = random.choice(("tl", "tr", "bl", "br"))
        legend_entries = list(legend_mapping.items())
        random.shuffle(legend_entries)
        _paint_legend(grid, legend_entries, corner)
        _paint_legend(expected, legend_entries, corner)

        reserved = np.zeros((height, width), dtype=bool)
        legend_r0 = 0 if "t" in corner else height - 5
        legend_c0 = 0 if "l" in corner else width - 7
        _reserve(reserved, legend_r0, legend_c0, legend_r0 + 4, legend_c0 + 6, pad=2)

        failed = False
        for _ in range(frame_count):
            placed = False
            for _ in range(300):
                r0 = random.randint(1, height - frame_side - 1)
                c0 = random.randint(1, width - frame_side - 1)
                r1 = r0 + frame_side - 1
                c1 = c0 + frame_side - 1
                if not _can_place(reserved, r0, c0, r1, c1, pad=1):
                    continue

                gap_side = random.choice(("top", "bottom", "left", "right"))
                gap_pos = random.randint(1, frame_side - 2)
                overlap_count = random.choice(LEGEND_COUNTS)
                pattern = _make_pattern(interior_side, gap_side, overlap_count)

                _draw_frame(grid, r0, c0, frame_side, border_color, gap_side, gap_pos, bg)
                _draw_frame(expected, r0, c0, frame_side, border_color, gap_side, gap_pos, bg)

                input_patch = grid[r0 + 1 : r1, c0 + 1 : c1]
                input_patch[pattern] = source_color

                out_patch = expected[r0 + 1 : r1, c0 + 1 : c1]
                out_patch[pattern] = source_color
                mirrored = _reflect_mask(pattern, gap_side)
                out_patch[np.logical_or(pattern, mirrored)] = legend_mapping[overlap_count]

                _reserve(reserved, r0, c0, r1, c1, pad=2)
                placed = True
                break
            if not placed:
                failed = True
                break

        if not failed:
            return grid, expected

    raise RuntimeError("failed to place all frames")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _generate_instance()
        except RuntimeError:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input for this ARC task")
