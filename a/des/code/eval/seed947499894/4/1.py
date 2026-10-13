import random
from collections import deque

import numpy as np


ALL_COLORS = tuple(range(10))
MAX_GENERATION_RETRIES = 400

# Connected pattern masks used inside frames.
BASE_PATTERNS = (
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    while c < mask.size:
        if not bool(mask[c]):
            c += 1
            continue
        start = c
        while c < mask.size and bool(mask[c]):
            c += 1
        runs.append((start, c - 1))
    return runs


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, _w = grid.shape
    candidates: list[tuple[int, int, dict[int, int], np.ndarray]] = []
    for r in range(h - 1):
        if not np.array_equal(grid[r], grid[r + 1]):
            continue
        row = grid[r]
        non_bg = row != bg
        runs = _true_runs(non_bg)
        if not runs or not all(end - start + 1 == 2 for start, end in runs):
            continue

        mapping: dict[int, int] = {}
        ok = True
        for start, _end in runs:
            left = int(row[start])
            right = int(row[start + 1])
            if left == bg or right == bg or left == right:
                ok = False
                break
            if left in mapping and mapping[left] != right:
                ok = False
                break
            if right in mapping and mapping[right] != left:
                ok = False
                break
            mapping[left] = right
            mapping[right] = left
        if not ok or len(mapping) < 2:
            continue

        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[r : r + 2] = np.tile(non_bg, (2, 1))
        candidates.append((r, len(runs), mapping, legend_mask))

    assert candidates, "legend not found"
    r, _pairs, mapping, legend_mask = min(candidates, key=lambda item: (item[0], -item[1]))
    assert r < h - 1
    return mapping, legend_mask


def _frame_bbox(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    if int(coords.shape[0]) != 2 * (height + width) - 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == color):
        return None
    if not np.all(sub[-1, :] == color):
        return None
    if not np.all(sub[:, 0] == color):
        return None
    if not np.all(sub[:, -1] == color):
        return None
    if height > 2 and width > 2 and np.any(sub[1:-1, 1:-1] == color):
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in (int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            bbox = _frame_bbox(grid, comp, color)
            if bbox is not None:
                frames.append((*bbox, color))
    frames.sort()
    return frames


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    mapping, legend_mask = _find_legend(g, bg)

    work = g.copy()
    work[legend_mask] = bg
    out = work.copy()

    for r0, c0, r1, c1, border_color in _find_frames(work, bg):
        assert border_color in mapping, "frame border color must appear in the legend"
        mapped_border = mapping[border_color]

        out[r0, c0 : c1 + 1] = mapped_border
        out[r1, c0 : c1 + 1] = mapped_border
        out[r0 : r1 + 1, c0] = mapped_border
        out[r0 : r1 + 1, c1] = mapped_border

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(work[r, c])
                if value == bg:
                    out[r, c] = border_color
                else:
                    out[r, c] = mapping.get(value, value)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _apply_rule(grid)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    rots: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(rot.shape), rot.tobytes())
        if key not in seen:
            seen.add(key)
            rots.append(rot.copy())
    return rots


PATTERN_VARIANTS = tuple(
    variant for base in BASE_PATTERNS for variant in _unique_rotations(base)
)


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_patterns(
    bbox: tuple[int, int, int, int],
    border_color: int,
    usable_colors: list[int],
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    if ih < 2 or iw < 2:
        return []

    variants = [mask for mask in PATTERN_VARIANTS if mask.shape[0] <= ih and mask.shape[1] <= iw and int(mask.sum()) < ih * iw]
    if not variants:
        return []

    occ = np.zeros((ih, iw), dtype=bool)
    placed: list[tuple[int, int, int]] = []
    target = 1
    if ih * iw >= 20 and rng.random() < 0.45:
        target = 2

    for _ in range(target):
        success = False
        for _attempt in range(60):
            shape = rng.choice(variants)
            sh, sw = shape.shape

            row_lo = 1 if ih - sh >= 2 else 0
            row_hi = ih - sh - 1 if ih - sh >= 2 else ih - sh
            col_lo = 1 if iw - sw >= 2 else 0
            col_hi = iw - sw - 1 if iw - sw >= 2 else iw - sw
            if row_hi < row_lo or col_hi < col_lo:
                row_lo, row_hi = 0, ih - sh
                col_lo, col_hi = 0, iw - sw
            top = rng.randint(row_lo, row_hi)
            left = rng.randint(col_lo, col_hi)

            cells = [(top + rr, left + cc) for rr, cc in np.argwhere(shape)]
            if any(bool(occ[rr, cc]) for rr, cc in cells):
                continue
            if any(bool(occ[max(0, rr - 1) : min(ih, rr + 2), max(0, cc - 1) : min(iw, cc + 2)].any()) for rr, cc in cells):
                continue

            color = rng.choice([c for c in usable_colors if c != border_color])
            for rr, cc in cells:
                occ[rr, cc] = True
                placed.append((r0 + 1 + rr, c0 + 1 + cc, color))
            success = True
            break
        if not success:
            break
    return placed


def _expected_output(
    grid: np.ndarray,
    bg: int,
    legend_cells: list[tuple[int, int]],
    mapping: dict[int, int],
    frames: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in legend_cells:
        out[r, c] = bg

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        border_color = int(frame["border_color"])  # type: ignore[index]
        mapped = mapping[border_color]
        out[r0, c0 : c1 + 1] = mapped
        out[r1, c0 : c1 + 1] = mapped
        out[r0 : r1 + 1, c0] = mapped
        out[r0 : r1 + 1, c1] = mapped
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(grid[r, c])
                out[r, c] = border_color if value == bg else mapping[value]
    return out


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(14, 22)
    width = rng.randint(16, 26)
    area = height * width
    max_frames = 3 if area < 280 else 4 if area < 380 else 5
    pair_count = rng.randint(2, min(4, max_frames))
    frame_count = rng.randint(pair_count, max_frames)

    bg = rng.choice(ALL_COLORS)
    usable = [c for c in ALL_COLORS if c != bg]
    rng.shuffle(usable)
    pair_colors = usable[: 2 * pair_count]
    pairs = [(pair_colors[2 * i], pair_colors[2 * i + 1]) for i in range(pair_count)]
    mapping: dict[int, int] = {}
    for left, right in pairs:
        mapping[left] = right
        mapping[right] = left

    sep_widths = [rng.randint(1, 2) for _ in range(pair_count - 1)]
    legend_width = 2 * pair_count + sum(sep_widths)
    if legend_width > width:
        return None
    legend_start = rng.randint(0, width - legend_width)

    grid = np.full((height, width), bg, dtype=int)
    legend_cells: list[tuple[int, int]] = []
    col = legend_start
    for idx, (left, right) in enumerate(pairs):
        grid[0:2, col] = left
        grid[0:2, col + 1] = right
        legend_cells.extend([(0, col), (1, col), (0, col + 1), (1, col + 1)])
        col += 2
        if idx < len(sep_widths):
            col += sep_widths[idx]

    row_start = 4
    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(frame_count):
        placed = False
        for _attempt in range(200):
            fh = rng.randint(4, 7 if frame_count <= 4 else 6)
            fw = rng.randint(4, 8 if frame_count <= 4 else 7)
            if row_start + fh >= height or fw + 2 > width:
                return None
            r0 = rng.randint(row_start, height - fh - 1)
            c0 = rng.randint(1, width - fw - 1)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if all(not _rect_overlaps(bbox, other, pad=1) for other in bboxes):
                bboxes.append(bbox)
                placed = True
                break
        if not placed:
            return None

    border_colors = [rng.choice(pair) for pair in pairs]
    legend_palette = [color for pair in pairs for color in pair]
    while len(border_colors) < frame_count:
        border_colors.append(rng.choice(legend_palette))
    rng.shuffle(border_colors)

    frames: list[dict[str, object]] = []
    capable = [idx for idx, bbox in enumerate(bboxes) if (bbox[2] - bbox[0] - 1) >= 2 and (bbox[3] - bbox[1] - 1) >= 2]
    with_pattern = [idx in capable and rng.random() < 0.65 for idx in range(frame_count)]
    if capable and not any(with_pattern):
        with_pattern[rng.choice(capable)] = True
    if frame_count > 1 and all(with_pattern):
        with_pattern[rng.choice(capable)] = False
    if frame_count > 1 and not any(not flag for flag in with_pattern):
        with_pattern[rng.randrange(frame_count)] = False

    for bbox, border_color, has_pattern in zip(bboxes, border_colors, with_pattern):
        _draw_border(grid, bbox, border_color)
        pattern_cells: list[tuple[int, int, int]] = []
        if has_pattern:
            pattern_cells = _place_patterns(bbox, border_color, legend_palette, rng)
            if not pattern_cells:
                has_pattern = False
        for r, c, color in pattern_cells:
            grid[r, c] = color
        frames.append(
            {
                "bbox": bbox,
                "border_color": border_color,
                "has_pattern": has_pattern,
                "pattern_cells": pattern_cells,
            }
        )

    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return None

    expected = _expected_output(grid, bg, legend_cells, mapping, frames)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        built = _build_case(rng)
        if built is None:
            continue
        grid, expected = built
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
