import random
from collections import deque

import numpy as np


GREY = 5
GEN_ATTEMPTS = 300
INNER_ROWS = 4
INNER_WIDTH_CHOICES = (2, 4, 6)
GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (16, 30)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)

    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _segment_color(segment: np.ndarray, bg: int) -> int | None:
    if segment.size == 0:
        return None
    values = np.unique(segment)
    if len(values) != 1:
        raise AssertionError("expected a uniform stripe or background segment")
    color = int(values[0])
    if color in {bg, GREY}:
        return None
    return color


def _find_palette(
    grid: np.ndarray,
) -> tuple[int, tuple[int, int, int, int], dict[int, int], dict[int, int]]:
    bg = _most_common_color(grid)
    grey_components = _components(grid == GREY)
    candidates = []

    for comp in grey_components:
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 4:
            continue

        border_area = 2 * (height + width) - 4
        if len(comp) != border_area:
            continue

        if not (
            np.all(grid[r0, c0 : c1 + 1] == GREY)
            and np.all(grid[r1, c0 : c1 + 1] == GREY)
            and np.all(grid[r0 : r1 + 1, c0] == GREY)
            and np.all(grid[r0 : r1 + 1, c1] == GREY)
        ):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[0] == 0 or inner.shape[1] < 2 or inner.shape[1] % 2 != 0:
            continue

        half = inner.shape[1] // 2
        forward: dict[int, int] = {}
        backward: dict[int, int] = {}
        ok = True
        for row in inner:
            left = int(row[0])
            right = int(row[half])
            if left in {bg, GREY} or right in {bg, GREY} or left == right:
                ok = False
                break
            if not (np.all(row[:half] == left) and np.all(row[half:] == right)):
                ok = False
                break
            if left in forward or right in backward:
                ok = False
                break
            forward[left] = right
            backward[right] = left

        if not ok:
            continue

        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        target_r = (grid.shape[0] - 1) / 2.0
        target_c = (grid.shape[1] - 1) / 2.0
        center_distance = abs(center_r - target_r) + abs(center_c - target_c)
        candidates.append((center_distance, -border_area, (r0, c0, r1, c1), forward, backward))

    assert candidates, "main: expected one grey-framed palette"
    _, _, bbox, forward, backward = min(candidates)
    return bg, bbox, forward, backward


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor horizontal edge stripes using the central palette's bidirectional mapping.

    Input structure:
    - bg is the most common color.
    - A small central rectangle is outlined by a grey one-cell frame.
    - Inside the frame, each row is split into a uniform left half color A and right half color B, encoding A->B.
    - One-cell-high horizontal stripes may run from the left edge up to the palette or from the right edge up to the palette.

    Rule:
    - Read each palette row as a forward color mapping A->B and also use its reverse B->A.
    - For a left-entering stripe, if its color matches some A, continue it to the right edge with color B.
    - For a right-entering stripe, if its color matches some B, continue it to the left edge with color A.
    - Stripes whose colors are absent from the relevant side of the palette stay unchanged.
    - The grey frame and palette interior remain exactly as they are; the crossing part is hidden by the palette.

    Key insight:
    - The central grey rectangle is a lookup table, and the same table must be read in opposite directions depending on which side a stripe approaches from.
    """
    g = np.asarray(grid, dtype=int)
    bg, (r0, c0, r1, c1), forward, backward = _find_palette(g)

    out = g.copy()
    for r in range(r0 + 1, r1):
        left_color = _segment_color(g[r, :c0], bg)
        right_color = _segment_color(g[r, c1 + 1 :], bg)

        if left_color is not None and left_color in forward:
            out[r, c1 + 1 :] = forward[left_color]
        if right_color is not None and right_color in backward:
            out[r, :c0] = backward[right_color]

    return out


def _centered_start(limit: int, span: int) -> int:
    low = 1
    high = limit - span - 1
    assert low <= high
    centered = (limit - span) // 2
    jitter_low = max(low, centered - 2)
    jitter_high = min(high, centered + 2)
    if jitter_low <= jitter_high:
        return random.randint(jitter_low, jitter_high)
    return random.randint(low, high)


def _gen_attempt() -> np.ndarray:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    inner_w = random.choice(INNER_WIDTH_CHOICES)
    frame_h = INNER_ROWS + 2
    frame_w = inner_w + 2
    if h < frame_h + 2 or w < frame_w + 2:
        return None

    bg = random.choice([c for c in range(10) if c != GREY])
    grid = np.full((h, w), bg, dtype=int)

    r0 = _centered_start(h, frame_h)
    c0 = _centered_start(w, frame_w)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    palette_colors = [c for c in range(10) if c not in {bg, GREY}]
    random.shuffle(palette_colors)
    left_palette = palette_colors[:INNER_ROWS]
    right_palette = palette_colors[INNER_ROWS:]
    random.shuffle(left_palette)
    random.shuffle(right_palette)

    half = inner_w // 2
    for idx, (left_color, right_color) in enumerate(zip(left_palette, right_palette)):
        row = r0 + 1 + idx
        grid[row, c0 + 1 : c0 + 1 + half] = left_color
        grid[row, c0 + 1 + half : c1] = right_color

    row_to_kind = dict(
        zip(
            random.sample(range(INNER_ROWS), INNER_ROWS),
            ("left_mapped", "left_unmapped", "right_mapped", "right_unmapped"),
        )
    )

    left_mapped_color = random.choice(left_palette)
    left_unmapped_color = random.choice(right_palette)
    right_mapped_color = random.choice(right_palette)
    right_unmapped_color = random.choice(left_palette)

    meta = {}
    for idx in range(INNER_ROWS):
        row = r0 + 1 + idx
        kind = row_to_kind[idx]
        meta[kind] = row
        if kind == "left_mapped":
            grid[row, :c0] = left_mapped_color
        elif kind == "left_unmapped":
            grid[row, :c0] = left_unmapped_color
        elif kind == "right_mapped":
            grid[row, c1 + 1 :] = right_mapped_color
        else:
            grid[row, c1 + 1 :] = right_unmapped_color

    out = main(grid)

    lm_row = meta["left_mapped"]
    assert np.all(out[lm_row, c1 + 1 :] == right_palette[left_palette.index(left_mapped_color)])

    lu_row = meta["left_unmapped"]
    assert np.all(out[lu_row, c1 + 1 :] == bg)

    rm_row = meta["right_mapped"]
    assert np.all(out[rm_row, :c0] == left_palette[right_palette.index(right_mapped_color)])

    ru_row = meta["right_unmapped"]
    assert np.all(out[ru_row, :c0] == bg)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
