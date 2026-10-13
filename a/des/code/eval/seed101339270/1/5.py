from __future__ import annotations

from collections import deque
import random

import numpy as np


RECOLOR_BY_EDGE = {
    "top": 1,
    "bottom": 2,
    "left": 3,
    "right": 4,
}

SHAPE_COLOR = 6
INPUT_COLOR_POOL = (0, 5, 7, 8, 9)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
SHAPE_RETRIES = 120


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _analyze_frame(coords: np.ndarray) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    mask = _mask_from_coords(coords)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(mask & ~border):
        return None

    missing = np.argwhere(border & ~mask)
    if missing.shape[0] != 1:
        return None
    mr, mc = map(int, missing[0])
    if (mr, mc) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
        return None

    expected = border.copy()
    expected[mr, mc] = False
    if not np.array_equal(mask, expected):
        return None

    if mr == 0:
        gap_side = "top"
    elif mr == h - 1:
        gap_side = "bottom"
    elif mc == 0:
        gap_side = "left"
    elif mc == w - 1:
        gap_side = "right"
    else:
        return None

    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "gap_side": gap_side,
    }


def _is_consecutive(values: list[int]) -> bool:
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


def _edge_segment_side(coords: np.ndarray, shape: tuple[int, int]) -> str | None:
    h, w = shape
    rows = sorted(set(int(v) for v in coords[:, 0]))
    cols = sorted(set(int(v) for v in coords[:, 1]))

    if len(rows) == 1 and 2 <= len(cols) <= 4 and _is_consecutive(cols):
        if rows[0] == 0:
            return "top"
        if rows[0] == h - 1:
            return "bottom"

    if len(cols) == 1 and 2 <= len(rows) <= 4 and _is_consecutive(rows):
        if cols[0] == 0:
            return "left"
        if cols[0] == w - 1:
            return "right"

    return None


def _find_fences(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    items: list[dict[str, int | str]] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        frame_infos = []
        edge_sides = []
        for comp in _components(grid == color):
            frame = _analyze_frame(comp)
            if frame is not None:
                frame_infos.append(frame)
            side = _edge_segment_side(comp, grid.shape)
            if side is not None:
                edge_sides.append(side)
        if not frame_infos:
            continue
        assert len(frame_infos) == 1, "each fence color must have exactly one frame"
        assert len(edge_sides) == 1, "each fence color must have exactly one edge indicator"
        item = dict(frame_infos[0])
        item["color"] = int(color)
        item["edge_side"] = edge_sides[0]
        items.append(item)

    assert items, "expected at least one fenced object"
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each fenced interior shape using the fence gap direction, then recolor it
    from the grid-edge indicator with the same fence color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    for fence in _find_fences(g, bg):
        color = int(fence["color"])
        r0 = int(fence["r0"])
        c0 = int(fence["c0"])
        r1 = int(fence["r1"])
        c1 = int(fence["c1"])
        gap_side = str(fence["gap_side"])
        edge_side = str(fence["edge_side"])

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        shape_mask = (interior != bg) & (interior != color)
        coords = np.argwhere(shape_mask)
        assert coords.size > 0, "each fence must contain an interior shape"

        flipped = coords.copy()
        if gap_side in {"top", "bottom"}:
            flipped[:, 0] = interior.shape[0] - 1 - flipped[:, 0]
        else:
            flipped[:, 1] = interior.shape[1] - 1 - flipped[:, 1]

        view = out[r0 + 1 : r1, c0 + 1 : c1]
        view[shape_mask] = bg
        view[flipped[:, 0], flipped[:, 1]] = int(RECOLOR_BY_EDGE[edge_side])

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_gap_sides(count: int) -> list[str]:
    sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    remaining = [side for side in ("top", "bottom", "left", "right") if side not in sides]
    random.shuffle(remaining)
    sides.extend(remaining[: max(0, count - 2)])
    random.shuffle(sides)
    return sides


def _sample_indicator_sides(count: int) -> list[str]:
    sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    remaining = [side for side in ("top", "bottom", "left", "right") if side not in sides]
    random.shuffle(remaining)
    sides.extend(remaining[: max(0, count - 2)])
    random.shuffle(sides)
    return sides


def _random_shape_mask(inner_size: int) -> np.ndarray | None:
    target_max = min(inner_size * inner_size - 1, 6)
    if target_max < 3:
        return None

    for _ in range(SHAPE_RETRIES):
        target = random.randint(3, target_max)
        start = (random.randrange(inner_size), random.randrange(inner_size))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < inner_size and 0 <= nc < inner_size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue

        mask = np.zeros((inner_size, inner_size), dtype=bool)
        for cell in cells:
            mask[cell] = True
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    return None


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    size: int,
    color: int,
    bg: int,
    gap_side: str,
) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    gap_offset = random.randint(1, size - 2)
    if gap_side == "top":
        grid[top, left + gap_offset] = bg
    elif gap_side == "bottom":
        grid[bottom, left + gap_offset] = bg
    elif gap_side == "left":
        grid[top + gap_offset, left] = bg
    else:
        grid[top + gap_offset, right] = bg


def _place_frames(h: int, w: int, sizes: list[int]) -> list[tuple[int, int, int]] | None:
    placed: list[tuple[int, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    for size in sizes:
        min_r = 2
        max_r = h - size - 3
        min_c = 2
        max_c = w - size - 3
        if min_r > max_r or min_c > max_c:
            return None

        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(min_r, max_r)
            left = random.randint(min_c, max_c)
            box = (top, left, top + size - 1, left + size - 1)
            padded = (top - 1, left - 1, top + size, left + size)
            if any(_boxes_overlap(padded, other) for other in reserved):
                continue
            placed.append((top, left, size))
            reserved.append(padded)
            break
        else:
            return None

    return placed


def _draw_indicator(grid: np.ndarray, side: str, color: int) -> None:
    h, w = grid.shape
    length = random.randint(2, 4)

    if side == "top":
        start = random.randint(1, w - length - 1)
        grid[0, start : start + length] = color
    elif side == "bottom":
        start = random.randint(1, w - length - 1)
        grid[h - 1, start : start + length] = color
    elif side == "left":
        start = random.randint(1, h - length - 1)
        grid[start : start + length, 0] = color
    else:
        start = random.randint(1, h - length - 1)
        grid[start : start + length, w - 1] = color


def _expected_from_spec(
    inp: np.ndarray,
    frame_specs: list[dict[str, int | str | np.ndarray]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for spec in frame_specs:
        top = int(spec["top"])
        left = int(spec["left"])
        size = int(spec["size"])
        gap_side = str(spec["gap_side"])
        edge_side = str(spec["edge_side"])
        shape = np.asarray(spec["shape"], dtype=bool)
        flipped = np.flipud(shape) if gap_side in {"top", "bottom"} else np.fliplr(shape)
        view = out[top + 1 : top + size - 1, left + 1 : left + size - 1]
        view[shape] = bg
        view[flipped] = int(RECOLOR_BY_EDGE[edge_side])
    return out


def _generate_once() -> np.ndarray | None:
    fence_count = random.randint(2, 4)
    base = 12 + 3 * fence_count
    h = random.randint(max(18, base), max(22, base + 4))
    w = random.randint(max(18, base), max(22, base + 4))

    bg = random.choice(INPUT_COLOR_POOL)
    fence_colors = random.sample([c for c in INPUT_COLOR_POOL if c != bg], fence_count)
    sizes = [random.randint(5, 7) for _ in range(fence_count)]
    placements = _place_frames(h, w, sizes)
    if placements is None:
        return None

    gap_sides = _sample_gap_sides(fence_count)
    indicator_sides = _sample_indicator_sides(fence_count)
    grid = np.full((h, w), int(bg), dtype=int)
    frame_specs: list[dict[str, int | str | np.ndarray]] = []

    for color, gap_side, edge_side, (top, left, size) in zip(
        fence_colors, gap_sides, indicator_sides, placements
    ):
        inner_size = size - 2
        shape = _random_shape_mask(inner_size)
        if shape is None:
            return None

        _draw_frame(grid, top, left, size, color, bg, gap_side)
        view = grid[top + 1 : top + size - 1, left + 1 : left + size - 1]
        view[shape] = SHAPE_COLOR
        _draw_indicator(grid, edge_side, color)

        frame_specs.append(
            {
                "top": int(top),
                "left": int(left),
                "size": int(size),
                "gap_side": gap_side,
                "edge_side": edge_side,
                "shape": shape,
            }
        )

    expected = _expected_from_spec(grid, frame_specs, bg)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    """
    Randomly generate ring-fence scenes with all critical branches varied and validated.
    """
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
