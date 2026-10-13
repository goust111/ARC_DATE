from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator bounds.
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_H_RANGE = (7, 11)
FRAME_W_RANGE = (7, 11)
GENERATION_ATTEMPTS = 400
FRAME_PLACE_ATTEMPTS = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_components(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if coords.shape[0] != int(border.sum()):
                continue
            if not np.all(patch[border] == color):
                continue
            if np.any(patch[~border] == color):
                continue
            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})

    frames.sort(key=lambda item: item["bbox"])
    assert frames, "expected at least one hollow frame"
    return frames


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _marker_color_for_frame(grid: np.ndarray, frame: dict, bg: int) -> int:
    r0, c0, r1, c1 = frame["bbox"]
    frame_color = int(frame["color"])
    candidates: list[int] = []

    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue
        for coords in _components(grid == color):
            rr0, cc0, rr1, cc1 = _bbox(coords)
            h = rr1 - rr0 + 1
            w = cc1 - cc0 + 1
            if h != w or h not in (1, 2) or coords.shape[0] != h * w:
                continue
            top = rr1 == r0 - 1 and _overlap_len(cc0, cc1, c0, c1) > 0
            bottom = rr0 == r1 + 1 and _overlap_len(cc0, cc1, c0, c1) > 0
            left = cc1 == c0 - 1 and _overlap_len(rr0, rr1, r0, r1) > 0
            right = cc0 == c1 + 1 and _overlap_len(rr0, rr1, r0, r1) > 0
            if int(top) + int(bottom) + int(left) + int(right) == 1:
                candidates.append(color)

    assert len(candidates) == 1, "each frame needs exactly one adjacent marker"
    return int(candidates[0])


def _fragment_mask(grid: np.ndarray, frame: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != bg) & (interior != int(frame["color"]))
    assert mask.any(), "frame interior must contain a fragment"
    return mask


def _axis_score(mask: np.ndarray, axis: str) -> int:
    mirrored = np.fliplr(mask) if axis == "vertical" else np.flipud(mask)
    return int(np.count_nonzero(mask & mirrored))


def _choose_axis(mask: np.ndarray) -> str:
    vertical = _axis_score(mask, "vertical")
    horizontal = _axis_score(mask, "horizontal")
    assert vertical != horizontal, "fragment axis must be unambiguous"
    return "vertical" if vertical > horizontal else "horizontal"


def _mirror_union(mask: np.ndarray, axis: str) -> np.ndarray:
    mirrored = np.fliplr(mask) if axis == "vertical" else np.flipud(mask)
    return mask | mirrored


def main(grid: np.ndarray) -> np.ndarray:
    """
    根据框架内部碎片的朝向选择镜像轴，再用框架外部色块的颜色对镜像结果进行重着色。

    Input structure:
    - Black background with several single-color hollow rectangular frames.
    - Each frame contains one irregular grey fragment inside.
    - Each frame has one 1x1 or 2x2 solid marker touching one outer side.

    Rule:
    - For each frame, compare overlap with its vertical and horizontal mirror.
    - Mirror the fragment across the better axis and keep the union so the
      result becomes fully symmetric on that axis.
    - Recolor the symmetric fragment with the color of the touching outside
      marker, while keeping frame borders and markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _frame_components(g, bg)

    out = g.copy()
    for frame in frames:
        marker_color = _marker_color_for_frame(g, frame, bg)
        frag = _fragment_mask(g, frame, bg)
        axis = _choose_axis(frag)
        sym = _mirror_union(frag, axis)

        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[(interior != bg) & (interior != int(frame["color"]))] = bg
        interior[sym] = marker_color

    return out


def _connected(mask: np.ndarray) -> bool:
    parts = _components(mask)
    return len(parts) == 1


def _generate_symmetric_base(height: int, width: int, axis: str) -> np.ndarray:
    if axis == "vertical":
        half_w = (width + 1) // 2
        boundary_col = half_w - 1
        half = np.zeros((height, half_w), dtype=bool)
        seed = (random.randrange(height), boundary_col)
        target = random.randint(max(2, (height * width) // 10), max(4, (height * width) // 4))
        frontier = {seed}
        half[seed] = True
        count = 1
        while count < target and frontier:
            cell = random.choice(tuple(frontier))
            options = []
            for dr, dc in DIR4:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < height and 0 <= nc < half_w and not half[nr, nc]:
                    options.append((nr, nc))
            if not options:
                frontier.remove(cell)
                continue
            new_cell = random.choice(options)
            half[new_cell] = True
            frontier.add(new_cell)
            frontier.add(cell)
            count += 1

        full = np.zeros((height, width), dtype=bool)
        full[:, :half_w] |= half
        for r, c in zip(*np.where(half)):
            full[r, width - 1 - c] = True
        return full

    half_h = (height + 1) // 2
    boundary_row = half_h - 1
    half = np.zeros((half_h, width), dtype=bool)
    seed = (boundary_row, random.randrange(width))
    target = random.randint(max(2, (height * width) // 10), max(4, (height * width) // 4))
    frontier = {seed}
    half[seed] = True
    count = 1
    while count < target and frontier:
        cell = random.choice(tuple(frontier))
        options = []
        for dr, dc in DIR4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < half_h and 0 <= nc < width and not half[nr, nc]:
                options.append((nr, nc))
        if not options:
            frontier.remove(cell)
            continue
        new_cell = random.choice(options)
        half[new_cell] = True
        frontier.add(new_cell)
        frontier.add(cell)
        count += 1

    full = np.zeros((height, width), dtype=bool)
    full[:half_h, :] |= half
    for r, c in zip(*np.where(half)):
        full[height - 1 - r, c] = True
    return full


def _extra_cells(base: np.ndarray, axis: str, side: str) -> list[tuple[int, int]]:
    h, w = base.shape
    candidates: list[tuple[int, int]] = []
    for r, c in zip(*np.where(base)):
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or base[nr, nc]:
                continue
            if axis == "vertical":
                if side == "left" and nc < w - 1 - nc:
                    candidates.append((nr, nc))
                if side == "right" and nc > w - 1 - nc:
                    candidates.append((nr, nc))
            else:
                if side == "top" and nr < h - 1 - nr:
                    candidates.append((nr, nc))
                if side == "bottom" and nr > h - 1 - nr:
                    candidates.append((nr, nc))
    random.shuffle(candidates)
    out: list[tuple[int, int]] = []
    seen = set()
    for cell in candidates:
        if cell in seen:
            continue
        out.append(cell)
        seen.add(cell)
    return out


def _generate_fragment(height: int, width: int, axis: str) -> np.ndarray:
    for _ in range(300):
        base = _generate_symmetric_base(height, width, axis)
        side = random.choice(["left", "right"] if axis == "vertical" else ["top", "bottom"])
        extras = _extra_cells(base, axis, side)
        if len(extras) < 1:
            continue

        mask = base.copy()
        for cell in extras[: random.randint(1, min(2, len(extras)))]:
            mask[cell] = True

        if not _connected(mask):
            continue
        vertical = _axis_score(mask, "vertical")
        horizontal = _axis_score(mask, "horizontal")
        if axis == "vertical" and vertical <= horizontal:
            continue
        if axis == "horizontal" and horizontal <= vertical:
            continue
        if abs(vertical - horizontal) < 2:
            continue
        return mask

    raise AssertionError("failed to generate a valid fragment")


def _frame_count(height: int, width: int) -> int:
    if min(height, width) <= 24:
        return random.randint(2, 3)
    return random.randint(2, 4)


def _marker_bbox(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    size: int,
) -> tuple[int, int, int, int]:
    if side == "top":
        mc0 = random.randint(c0, c1 - size + 1)
        return r0 - size, mc0, r0 - 1, mc0 + size - 1
    if side == "bottom":
        mc0 = random.randint(c0, c1 - size + 1)
        return r1 + 1, mc0, r1 + size, mc0 + size - 1
    if side == "left":
        mr0 = random.randint(r0, r1 - size + 1)
        return mr0, c0 - size, mr0 + size - 1, c0 - 1
    mr0 = random.randint(r0, r1 - size + 1)
    return mr0, c1 + 1, mr0 + size - 1, c1 + size


def _side_ranges(height: int, width: int, frame_h: int, frame_w: int, side: str, marker_size: int) -> tuple[range, range]:
    if side == "top":
        return range(marker_size, height - frame_h + 1), range(0, width - frame_w + 1)
    if side == "bottom":
        return range(0, height - frame_h - marker_size + 1), range(0, width - frame_w + 1)
    if side == "left":
        return range(0, height - frame_h + 1), range(marker_size, width - frame_w + 1)
    return range(0, height - frame_h + 1), range(0, width - frame_w - marker_size + 1)


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _reserve_ok(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return not reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _reserve(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _candidate_bbox(frame_bbox: tuple[int, int, int, int], marker_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    mr0, mc0, mr1, mc1 = marker_bbox
    return min(fr0, mr0), min(fc0, mc0), max(fr1, mr1), max(fc1, mc1)


def _generate_attempt() -> np.ndarray:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    frame_count = _frame_count(height, width)
    colors = PALETTE[:]
    random.shuffle(colors)
    assert 2 * frame_count <= len(colors)

    frames: list[dict] = []
    for index in range(frame_count):
        frame_color = colors[2 * index]
        marker_color = colors[2 * index + 1]
        placed = False

        for _ in range(FRAME_PLACE_ATTEMPTS):
            frame_h = random.randint(*FRAME_H_RANGE)
            frame_w = random.randint(*FRAME_W_RANGE)
            side = random.choice(["top", "bottom", "left", "right"])
            marker_size = random.choice([1, 2])
            row_range, col_range = _side_ranges(height, width, frame_h, frame_w, side, marker_size)
            if len(row_range) == 0 or len(col_range) == 0:
                continue

            r0 = random.choice(tuple(row_range))
            c0 = random.choice(tuple(col_range))
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1
            marker_bbox = _marker_bbox(r0, c0, r1, c1, side, marker_size)
            bbox = _candidate_bbox((r0, c0, r1, c1), marker_bbox)
            if not _reserve_ok(reserved, bbox):
                continue

            inner_h = frame_h - 2
            inner_w = frame_w - 2
            axis = random.choice(["vertical", "horizontal"])
            fragment = _generate_fragment(inner_h, inner_w, axis)

            _paint_frame(grid, r0, c0, r1, c1, frame_color)
            fr = grid[r0 + 1 : r1, c0 + 1 : c1]
            fr[fragment] = GREY

            mr0, mc0, mr1, mc1 = marker_bbox
            grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = marker_color
            _reserve(reserved, bbox)

            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "frame_color": frame_color,
                    "marker_color": marker_color,
                    "axis": axis,
                }
            )
            placed = True
            break

        if not placed:
            raise AssertionError("placement failed")

    _ = main(grid)

    axis_counts = {"vertical": 0, "horizontal": 0}
    for frame in _frame_components(grid, BLACK):
        frag = _fragment_mask(grid, frame, BLACK)
        axis_counts[_choose_axis(frag)] += 1
        marker_color = _marker_color_for_frame(grid, frame, BLACK)
        assert marker_color != int(frame["color"])

    assert axis_counts["vertical"] + axis_counts["horizontal"] == frame_count
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            return _generate_attempt()
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid input")
