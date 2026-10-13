from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _frame_from_component(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = np.zeros((h, w), dtype=bool)
    for r, c in comp:
        sub[r - r0, c - c0] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub, expected):
        return None

    border = [(int(r), int(c)) for r, c in comp]
    return {"bbox": (r0, c0, r1, c1), "border": border}


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames = []
    for comp in _components(grid == GREY):
        frame = _frame_from_component(comp)
        if frame is not None:
            frames.append(frame)
    assert frames, "expected at least one grey rectangular frame"
    return frames


def _legend_mapping(grid: np.ndarray, frames: list[dict[str, object]]) -> dict[int, int]:
    covered = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    mapping: dict[int, int] = {}
    legend_mask = (grid != BLACK) & (~covered)
    for comp in _components(legend_mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert BLUE in colors, "each legend row must include blue pattern cells"
        marker_colors = colors - {BLUE}
        assert len(marker_colors) == 1, "each legend row must expose exactly one marker color"
        marker_color = marker_colors.pop()
        marker_count = sum(int(grid[r, c]) == marker_color for r, c in comp)
        blue_count = sum(int(grid[r, c]) == BLUE for r, c in comp)
        assert marker_count == 1, "legend marker must be a single pixel"
        assert blue_count not in mapping, "legend counts must be unique"
        mapping[blue_count] = marker_color

    assert mapping, "expected legend rows outside all frames"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each frame's blue interior pixels, look up the matching legend row by blue-pixel count,
    and recolor the frame border with the legend's marker color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    mapping = _legend_mapping(g, frames)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        blue_count = int(np.sum(interior == BLUE))
        assert blue_count in mapping, "every frame must match a legend entry"
        fill_color = mapping[blue_count]
        for r, c in frame["border"]:
            out[r, c] = fill_color

    return out


@lru_cache(None)
def _legend_shape_options(count: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    options: list[tuple[tuple[int, ...], ...]] = []
    seen: set[tuple[int, ...]] = set()
    for bits in range(1 << 9):
        if bits.bit_count() != count:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        if not _is_connected(mask):
            continue
        trimmed = _trim(mask)
        if not trimmed[:, -1].any():
            continue
        key = tuple(trimmed.shape) + tuple(trimmed.astype(int).ravel().tolist())
        if key in seen:
            continue
        seen.add(key)
        options.append(tuple(tuple(int(v) for v in row) for row in trimmed.astype(int)))
    assert options, f"expected at least one legend shape for count {count}"
    return tuple(options)


def _make_legend_row(count: int, color: int) -> np.ndarray:
    shape = np.array(random.choice(_legend_shape_options(count)), dtype=bool)
    h, w = shape.shape
    patch = np.zeros((h, w + 1), dtype=int)
    patch[:, :w][shape] = BLUE
    attach_rows = [r for r in range(h) if shape[r, -1]]
    patch[random.choice(attach_rows), w] = int(color)
    return patch


def _draw_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    assert not np.any(target[mask] != BLACK), "patch placement must not overlap existing content"
    target[mask] = patch[mask]


def _expand_box(
    box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _box_is_free(
    occupied: np.ndarray, box: tuple[int, int, int, int], pad: int = 0
) -> bool:
    r0, c0, r1, c1 = _expand_box(box, pad, occupied.shape)
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _occupy(occupied: np.ndarray, box: tuple[int, int, int, int], pad: int = 0) -> None:
    r0, c0, r1, c1 = _expand_box(box, pad, occupied.shape)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _frame_patch(count: int) -> np.ndarray:
    options = [
        (ih, iw)
        for ih in range(2, 7)
        for iw in range(2, 7)
        if ih * iw > count
    ]
    ih, iw = random.choice(options)
    patch = np.full((ih + 2, iw + 2), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    interior_cells = [(r, c) for r in range(1, ih + 1) for c in range(1, iw + 1)]
    for r, c in random.sample(interior_cells, count):
        patch[r, c] = BLUE
    return patch


def _render_legend(
    grid: np.ndarray, rows: list[np.ndarray], corner: str
) -> tuple[int, int, int, int]:
    gap = 1
    total_h = sum(int(row.shape[0]) for row in rows) + gap * (len(rows) - 1)
    max_w = max(int(row.shape[1]) for row in rows)
    h, w = grid.shape

    top = 0 if "top" in corner else h - total_h
    left = 0 if "left" in corner else w - max_w
    cursor_r = top
    for row in rows:
        if "left" in corner:
            c0 = left
        else:
            c0 = w - int(row.shape[1])
        _draw_patch(grid, row, cursor_r, c0)
        cursor_r += int(row.shape[0]) + gap
    return top, left, top + total_h - 1, left + max_w - 1


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    entry_count = random.randint(2, 5)
    legend_counts = random.sample(range(1, 10), entry_count)
    marker_colors = random.sample(MARKER_COLORS, entry_count)
    rows = [_make_legend_row(count, color) for count, color in zip(legend_counts, marker_colors, strict=False)]

    min_h = max(18, sum(int(row.shape[0]) for row in rows) + len(rows) + 6)
    min_w = 18
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.full((h, w), BLACK, dtype=int)

    corner = random.choice(("top-left", "top-right", "bottom-left", "bottom-right"))
    legend_box = _render_legend(grid, rows, corner)

    frame_count = random.randint(entry_count, min(6, entry_count + 2))
    frame_counts = legend_counts[:]
    while len(frame_counts) < frame_count:
        frame_counts.append(random.choice(legend_counts))
    random.shuffle(frame_counts)

    frame_patches = [(count, _frame_patch(count)) for count in frame_counts]
    frame_patches.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    _occupy(occupied, legend_box, pad=1)

    placed_frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for count, patch in frame_patches:
        ph, pw = patch.shape
        if ph + 2 > h or pw + 2 > w:
            return None

        placed = False
        for _ in range(200):
            r0 = random.randint(1, h - ph - 1)
            c0 = random.randint(1, w - pw - 1)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if not _box_is_free(occupied, box, pad=1):
                continue
            _draw_patch(grid, patch, r0, c0)
            _occupy(occupied, box, pad=1)
            placed_frames.append((count, box))
            placed = True
            break
        if not placed:
            return None

    output = grid.copy()
    count_to_color = {count: color for count, color in zip(legend_counts, marker_colors, strict=False)}
    for count, (r0, c0, r1, c1) in placed_frames:
        color = count_to_color[count]
        output[r0, c0 : c1 + 1] = color
        output[r1, c0 : c1 + 1] = color
        output[r0 : r1 + 1, c0] = color
        output[r0 : r1 + 1, c1] = color

    return grid, output


def generate_input() -> np.ndarray:
    for _ in range(500):
        case = _build_case()
        if case is None:
            continue
        grid, expected = case
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
