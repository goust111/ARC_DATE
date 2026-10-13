import random
from math import ceil

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _strictly_inside(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]
) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _frame_components(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == BLACK:
            continue
        for cells in _components(grid == color):
            r0, c0, r1, c1 = _bbox(cells)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            if len(cells) != 2 * (height + width) - 4:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch[0, :] == color):
                continue
            if not np.all(patch[-1, :] == color):
                continue
            if not np.all(patch[:, 0] == color):
                continue
            if not np.all(patch[:, -1] == color):
                continue
            if np.any(patch[1:-1, 1:-1] == color):
                continue
            frames.append((r0, c0, r1, c1, color))
    frames.sort()
    return frames


def _legend_mapping(
    grid: np.ndarray, frames: list[tuple[int, int, int, int, int]]
) -> dict[int, int]:
    frame_boxes = [(r0, c0, r1, c1) for r0, c0, r1, c1, _ in frames]
    mapping: dict[int, int] = {}
    for cells in _components(grid != BLACK):
        if len(cells) != 4:
            continue
        box = _bbox(cells)
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        if any(_strictly_inside(box, frame_box) for frame_box in frame_boxes):
            continue
        values = np.array([int(grid[r, c]) for r, c in cells], dtype=int)
        colors, counts = np.unique(values, return_counts=True)
        if len(colors) != 2 or sorted(map(int, counts)) != [1, 3]:
            continue
        key = int(colors[np.argmax(counts)])
        value = int(colors[np.argmin(counts)])
        if key in mapping:
            assert mapping[key] == value
        mapping[key] = value
    return mapping


def _majority_marker_color(
    grid: np.ndarray, frame: tuple[int, int, int, int, int]
) -> int:
    r0, c0, r1, c1, _ = frame
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    markers = np.argwhere(inner != BLACK)
    assert len(markers) > 0
    counts: dict[int, int] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for rr, cc in markers:
        color = int(inner[rr, cc])
        pos = (int(r0 + 1 + rr), int(c0 + 1 + cc))
        counts[color] = counts.get(color, 0) + 1
        if color not in first_seen:
            first_seen[color] = pos
    return min(counts, key=lambda color: (-counts[color], first_seen[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular border through a two-step lookup:
    majority interior marker color -> legend key -> new border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _frame_components(g)
    assert frames
    legend = _legend_mapping(g, frames)
    assert legend

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1, _ = frame
        key = _majority_marker_color(g, frame)
        assert key in legend
        new_color = legend[key]
        out[r0, c0 : c1 + 1] = new_color
        out[r1, c0 : c1 + 1] = new_color
        out[r0 : r1 + 1, c0] = new_color
        out[r0 : r1 + 1, c1] = new_color
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _legend_inner_offset(corner: str) -> tuple[int, int]:
    return {
        "tl": (1, 1),
        "tr": (1, 0),
        "bl": (0, 1),
        "br": (0, 0),
    }[corner]


def _render_legend(
    grid: np.ndarray, keys: list[int], mapping: dict[int, int], corner: str
) -> tuple[int, int, int, int]:
    count = len(keys)
    max_cols = min(3, count)
    cols = random.randint(1, max_cols)
    rows = int(ceil(count / cols))
    gap = 1
    height = rows * 2 + (rows - 1) * gap
    width = cols * 2 + (cols - 1) * gap
    margin = random.randint(0, 1)
    h, w = grid.shape

    if corner == "tl":
        top, left = margin, margin
    elif corner == "tr":
        top, left = margin, w - margin - width
    elif corner == "bl":
        top, left = h - margin - height, margin
    else:
        top, left = h - margin - height, w - margin - width

    inner_dr, inner_dc = _legend_inner_offset(corner)
    order = keys[:]
    random.shuffle(order)
    for idx, key in enumerate(order):
        rr = idx // cols
        cc = idx % cols
        r0 = top + rr * (2 + gap)
        c0 = left + cc * (2 + gap)
        grid[r0 : r0 + 2, c0 : c0 + 2] = key
        grid[r0 + inner_dr, c0 + inner_dc] = mapping[key]
    return top, left, top + height - 1, left + width - 1


def _contains_full_2x2(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    for r, c in pts:
        block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
        if block <= pts:
            return True
    return False


def _sample_marker_positions(
    box: tuple[int, int, int, int], count: int
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for _ in range(80):
        chosen = random.sample(cells, count)
        if not _contains_full_2x2(chosen):
            return sorted(chosen)
    return None


def _tie_sequence(keys: list[int]) -> tuple[list[int], int]:
    if len(keys) >= 2 and random.random() < 0.5:
        a, b = random.sample(keys, 2)
        winner = random.choice([a, b])
        loser = b if winner == a else a
        seq = random.choice(
            [
                [winner, loser, winner, loser],
                [winner, loser, loser, winner],
            ]
        )
        return seq, winner

    tie_size = random.randint(2, min(4, len(keys)))
    colors = random.sample(keys, tie_size)
    winner = random.choice(colors)
    tail = [c for c in colors if c != winner]
    random.shuffle(tail)
    return [winner] + tail, winner


def _majority_sequence(keys: list[int]) -> tuple[list[int], int]:
    winner = random.choice(keys)
    others = [c for c in keys if c != winner]
    patterns = [[winner], [winner, winner], [winner, winner, winner], [winner] * 4]
    if others:
        loser1 = random.choice(others)
        patterns.extend([[winner, winner, loser1], [winner, winner, winner, loser1]])
    if len(others) >= 2:
        loser1, loser2 = random.sample(others, 2)
        patterns.append([winner, winner, loser1, loser2])
    seq = list(random.choice(patterns))
    random.shuffle(seq)
    return seq, winner


def _sample_marker_sequence(keys: list[int]) -> tuple[list[int], int]:
    if len(keys) >= 2 and random.random() < 0.5:
        return _tie_sequence(keys)
    return _majority_sequence(keys)


def _place_frames(
    h: int, w: int, count: int, reserved: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _idx in range(count):
        placed = False
        for _ in range(160):
            height = random.randint(5, min(10, h - 2))
            width = random.randint(5, min(10, w - 2))
            r0 = random.randint(1, h - height - 1)
            c0 = random.randint(1, w - width - 1)
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(_boxes_too_close(box, other, pad=1) for other in boxes):
                continue
            if any(_boxes_too_close(box, other, pad=1) for other in reserved):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    frame_count = random.randint(1, 4)
    key_count = random.randint(2, min(5, len(COLORS) - frame_count))

    key_colors = random.sample(list(COLORS), key_count)
    frame_pool = [c for c in COLORS if c not in key_colors]
    frame_colors = random.sample(frame_pool, frame_count)
    mapping = {
        key: random.choice([c for c in COLORS if c != key]) for key in key_colors
    }

    grid = np.zeros((h, w), dtype=int)
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_box = _render_legend(grid, key_colors, mapping, corner)

    boxes = _place_frames(h, w, frame_count, [legend_box])
    if boxes is None:
        return None

    out = grid.copy()
    for box, frame_color in zip(boxes, frame_colors):
        _draw_frame(grid, box, frame_color)
        colors, winner = _sample_marker_sequence(key_colors)
        marker_positions = _sample_marker_positions(box, len(colors))
        if marker_positions is None:
            return None
        for (r, c), color in zip(marker_positions, colors):
            grid[r, c] = color
        _draw_frame(out, box, mapping[winner])
        for (r, c), color in zip(marker_positions, colors):
            out[r, c] = color

    if np.array_equal(main(grid), out):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
