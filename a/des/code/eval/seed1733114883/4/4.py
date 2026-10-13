from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


MARKER_COLOR = 4
GEN_RETRIES = 400
PALETTE = tuple(range(10))
FRAME_DIM_CHOICES = ((7, 7), (7, 9), (9, 7), (9, 9))
GAP_CHOICES = (2, 3, 4)
MARGIN_CHOICES = (1, 2, 3)
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
TOUCH_TO_DIR = {
    "top": "down",
    "bottom": "up",
    "left": "right",
    "right": "left",
}
MARKER_VARIANTS = {
    "right": (
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
    ),
    "left": (
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1)),
    ),
    "down": (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ),
    "up": (
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR_TO_DELTA.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_from_center(center_r: int, center_c: int, height: int, width: int) -> tuple[int, int, int, int]:
    hh = height // 2
    hw = width // 2
    return int(center_r - hh), int(center_c - hw), int(center_r + hh), int(center_c + hw)


def _border_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((int(r0), int(c)))
        coords.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        coords.append((int(r), int(c0)))
        coords.append((int(r), int(c1)))
    return coords


def _paint_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frame_box_from_component(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected = set(_border_coords((r0, c0, r1, c1)))
    actual = {(int(r), int(c)) for r, c in coords}
    if actual != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _marker_direction(marker_coords: np.ndarray, inner_shape: tuple[int, int]) -> str:
    if marker_coords.shape != (3, 2):
        raise AssertionError("expected exactly one 3-cell marker inside each frame")
    r0, c0, r1, c1 = _bbox(marker_coords)
    if int(r1 - r0) != 1 or int(c1 - c0) != 1:
        raise AssertionError("marker must be a 2x2 L shape")
    actual = {(int(r), int(c)) for r, c in marker_coords}
    block = {(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)}
    if len(block - actual) != 1:
        raise AssertionError("marker must occupy three cells of a 2x2 block")

    inner_h, inner_w = inner_shape
    touches: list[str] = []
    if any(int(r) == 0 for r, _c in marker_coords):
        touches.append("top")
    if any(int(r) == inner_h - 1 for r, _c in marker_coords):
        touches.append("bottom")
    if any(int(c) == 0 for _r, c in marker_coords):
        touches.append("left")
    if any(int(c) == inner_w - 1 for _r, c in marker_coords):
        touches.append("right")
    if len(touches) != 1:
        raise AssertionError("marker must touch exactly one inner wall")
    return TOUCH_TO_DIR[touches[0]]


def _detect_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    frames: list[dict] = []
    marker_total = int(np.count_nonzero(g == MARKER_COLOR))

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, MARKER_COLOR)):
        for comp in _components(g == color):
            box = _frame_box_from_component(comp)
            if box is None:
                raise AssertionError("every non-background component must be a rectangular frame")
            r0, c0, r1, c1 = box
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            allowed = (inner == bg) | (inner == MARKER_COLOR)
            if not bool(np.all(allowed)):
                raise AssertionError("frame interior must contain only background and marker")
            marker_coords = np.argwhere(inner == MARKER_COLOR)
            direction = _marker_direction(marker_coords, inner.shape)
            frames.append(
                {
                    "box": box,
                    "color": int(color),
                    "direction": direction,
                }
            )

    frames.sort(key=lambda frame: frame["box"])
    marker_in_frames = len(frames) * 3
    if marker_total != marker_in_frames:
        raise AssertionError("all marker pixels must belong to frames")
    if not frames:
        raise AssertionError("expected at least one frame")
    return bg, frames


def _first_hit(source_index: int, frames: list[dict]) -> tuple[int, int] | None:
    source = frames[source_index]
    r0, c0, r1, c1 = source["box"]
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    direction = source["direction"]
    best_target: int | None = None
    best_gap: int | None = None

    for target_index, target in enumerate(frames):
        if target_index == source_index:
            continue
        tr0, tc0, tr1, tc1 = target["box"]
        gap: int | None = None
        if direction == "right" and tc0 > c1 and tr0 <= center_r <= tr1:
            gap = int(tc0 - c1 - 1)
        elif direction == "left" and tc1 < c0 and tr0 <= center_r <= tr1:
            gap = int(c0 - tc1 - 1)
        elif direction == "down" and tr0 > r1 and tc0 <= center_c <= tc1:
            gap = int(tr0 - r1 - 1)
        elif direction == "up" and tr1 < r0 and tc0 <= center_c <= tc1:
            gap = int(r0 - tr1 - 1)
        if gap is None:
            continue
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_target = int(target_index)

    if best_target is None or best_gap is None:
        return None
    return int(best_target), int(best_gap)


def _apply_infections(input_grid: np.ndarray, bg: int, frames: list[dict]) -> np.ndarray:
    out = np.array(input_grid, dtype=int, copy=True)
    out[out == MARKER_COLOR] = bg

    proposals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for source_index, frame in enumerate(frames):
        hit = _first_hit(source_index, frames)
        if hit is None:
            continue
        target_index, gap = hit
        proposals[target_index].append((int(gap), int(frame["color"])))

    for target_index, options in proposals.items():
        best_gap = min(gap for gap, _color in options)
        best_colors = {color for gap, color in options if gap == best_gap}
        if len(best_colors) != 1:
            continue
        color = next(iter(best_colors))
        _paint_border(out, frames[target_index]["box"], int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor rectangular frames based on one-shot ray hits from their internal L markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, frames = _detect_frames(g)
    return _apply_infections(g, bg, frames)


def _sample_dims() -> tuple[int, int]:
    return random.choice(FRAME_DIM_CHOICES)


def _distinct_gaps() -> tuple[int, int]:
    g0, g1 = random.sample(GAP_CHOICES, 2)
    return int(g0), int(g1)


def _boxes_non_adjacent(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, a in enumerate(boxes):
        ar0, ac0, ar1, ac1 = a
        for b in boxes[i + 1 :]:
            br0, bc0, br1, bc1 = b
            separated = ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1
            if not separated:
                return False
    return True


def _template_opposed_conflict() -> list[dict]:
    lh, lw = _sample_dims()
    th, tw = _sample_dims()
    rh, rw = _sample_dims()
    include_extra = random.random() < 0.75
    g_left, g_right = _distinct_gaps()

    frames = [
        {"box": _box_from_center(0, -(tw // 2 + g_left + 1 + lw // 2), lh, lw), "direction": "right"},
        {"box": _box_from_center(0, 0, th, tw), "direction": "down" if include_extra else random.choice(("up", "down"))},
        {"box": _box_from_center(0, tw // 2 + g_right + 1 + rw // 2, rh, rw), "direction": "left"},
    ]
    if include_extra:
        eh, ew = _sample_dims()
        g_down = int(random.choice(GAP_CHOICES))
        frames.append(
            {
                "box": _box_from_center(th // 2 + g_down + 1 + eh // 2, 0, eh, ew),
                "direction": "down",
            }
        )
    return frames


def _template_cross_conflict() -> list[dict]:
    lh, lw = _sample_dims()
    uh, uw = _sample_dims()
    th, tw = _sample_dims()
    rh, rw = _sample_dims()
    g_left, g_up = _distinct_gaps()
    g_right = int(random.choice(GAP_CHOICES))

    return [
        {"box": _box_from_center(0, -(tw // 2 + g_left + 1 + lw // 2), lh, lw), "direction": "right"},
        {"box": _box_from_center(-(th // 2 + g_up + 1 + uh // 2), 0, uh, uw), "direction": "down"},
        {"box": _box_from_center(0, 0, th, tw), "direction": "right"},
        {"box": _box_from_center(0, tw // 2 + g_right + 1 + rw // 2, rh, rw), "direction": "right"},
    ]


def _template_chain() -> list[dict]:
    ah, aw = _sample_dims()
    bh, bw = _sample_dims()
    ch, cw = _sample_dims()
    g_ab = int(random.choice(GAP_CHOICES))
    g_bc = int(random.choice(GAP_CHOICES))

    frames = [
        {"box": _box_from_center(0, -(bw // 2 + g_ab + 1 + aw // 2), ah, aw), "direction": "right"},
        {"box": _box_from_center(0, 0, bh, bw), "direction": "right"},
        {"box": _box_from_center(0, bw // 2 + g_bc + 1 + cw // 2, ch, cw), "direction": "right"},
    ]
    if random.random() < 0.65:
        dh, dw = _sample_dims()
        g_up = int(random.choice(GAP_CHOICES))
        frames.append(
            {
                "box": _box_from_center(-(bh // 2 + g_up + 1 + dh // 2), 0, dh, dw),
                "direction": "up",
            }
        )
    return frames


def _sample_marker_cells(box: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    variant = random.choice(MARKER_VARIANTS[direction])

    if direction in ("right", "left"):
        offset = random.randint(1, inner_h - 3)
        base_r = r0 + 1 + offset
        base_c = c0 + 1 if direction == "right" else c1 - 2
    else:
        offset = random.randint(1, inner_w - 3)
        base_r = r0 + 1 if direction == "down" else r1 - 2
        base_c = c0 + 1 + offset

    return [(int(base_r + dr), int(base_c + dc)) for dr, dc in variant]


def _instantiate_scene(relative_frames: list[dict]) -> tuple[np.ndarray, int, list[dict]] | None:
    boxes = [frame["box"] for frame in relative_frames]
    if not _boxes_non_adjacent(boxes):
        return None

    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    max_r = max(box[2] for box in boxes)
    max_c = max(box[3] for box in boxes)

    top = int(random.choice(MARGIN_CHOICES))
    left = int(random.choice(MARGIN_CHOICES))
    bottom = int(random.choice(MARGIN_CHOICES))
    right = int(random.choice(MARGIN_CHOICES))
    row_shift = top - min_r
    col_shift = left - min_c
    height = int(max_r + row_shift + bottom + 1)
    width = int(max_c + col_shift + right + 1)

    colors = random.sample([c for c in PALETTE if c != MARKER_COLOR], len(relative_frames) + 1)
    bg = int(colors[0])
    frame_colors = [int(color) for color in colors[1:]]

    grid = np.full((height, width), bg, dtype=int)
    frames: list[dict] = []
    for template_frame, color in zip(relative_frames, frame_colors):
        r0, c0, r1, c1 = template_frame["box"]
        box = (int(r0 + row_shift), int(c0 + col_shift), int(r1 + row_shift), int(c1 + col_shift))
        marker_cells = _sample_marker_cells(box, str(template_frame["direction"]))
        frame = {
            "box": box,
            "color": int(color),
            "direction": str(template_frame["direction"]),
            "marker_cells": marker_cells,
        }
        frames.append(frame)
        _paint_border(grid, box, int(color))
        for mr, mc in marker_cells:
            grid[mr, mc] = MARKER_COLOR

    if int(Counter(map(int, grid.flat)).most_common(1)[0][0]) != bg:
        return None
    return grid, bg, frames


def _count_targets_with_multiple_sources(frames: list[dict]) -> int:
    counts: dict[int, int] = defaultdict(int)
    for source_index in range(len(frames)):
        hit = _first_hit(source_index, frames)
        if hit is None:
            continue
        target_index, _gap = hit
        counts[target_index] += 1
    return sum(1 for count in counts.values() if count >= 2)


def _count_no_hit_sources(frames: list[dict]) -> int:
    return sum(1 for source_index in range(len(frames)) if _first_hit(source_index, frames) is None)


def _expected_from_scene(grid: np.ndarray, bg: int, frames: list[dict]) -> np.ndarray:
    minimal_frames = [
        {"box": frame["box"], "color": frame["color"], "direction": frame["direction"]}
        for frame in frames
    ]
    return _apply_infections(grid, bg, minimal_frames)


def _gen_attempt() -> np.ndarray | None:
    template_name = random.choice(("opposed_conflict", "cross_conflict", "chain"))
    if template_name == "opposed_conflict":
        relative_frames = _template_opposed_conflict()
    elif template_name == "cross_conflict":
        relative_frames = _template_cross_conflict()
    else:
        relative_frames = _template_chain()

    built = _instantiate_scene(relative_frames)
    if built is None:
        return None
    grid, bg, frames = built

    conflict_count = _count_targets_with_multiple_sources(frames)
    no_hit_count = _count_no_hit_sources(frames)
    if template_name == "chain":
        if conflict_count != 0 or no_hit_count < 1:
            return None
    else:
        if conflict_count < 1 or no_hit_count < 1:
            return None

    expected = _expected_from_scene(grid, bg, frames)
    if np.array_equal(grid, expected):
        return None

    k = random.randrange(4)
    inp = np.rot90(grid, k).copy()
    out = np.rot90(expected, k).copy()

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
