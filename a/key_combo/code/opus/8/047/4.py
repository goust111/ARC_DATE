from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 800
GRID_H_MIN = 15
GRID_H_MAX = 26
LEFT_W_MIN = 10
LEFT_W_MAX = 16
RIGHT_W_MIN = 8
RIGHT_W_MAX = 14

LAYOUTS = (
    "chain2",
    "chain3",
    "branch3",
    "branch4",
    "triple",
)


class Frame:
    def __init__(
        self,
        color: int,
        bbox: tuple[int, int, int, int],
        anchor: tuple[int, int],
        depth: int,
    ) -> None:
        self.color = color
        self.bbox = bbox
        self.anchor = anchor
        self.depth = depth

    @property
    def center_row(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2


def _frame_boundary_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_col(grid: np.ndarray) -> int | None:
    g = np.asarray(grid, dtype=int)
    candidates = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    if len(candidates) == 1:
        return candidates[0]
    for c in range(g.shape[1]):
        vals = g[:, c]
        if np.all((vals == BG) | (vals == SEP)) and np.count_nonzero(vals == SEP) == g.shape[0]:
            return c
    return None


def _as_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5 or height % 2 == 0 or width % 2 == 0:
        return None
    if set(coords) != _frame_boundary_cells(bbox):
        return None
    return bbox


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _parse_puzzle(grid: np.ndarray) -> tuple[int, list[Frame]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    h, w = g.shape
    if max(h, w) > 30:
        raise ValueError("grid exceeds size limit")

    sep_col = _find_separator_col(g)
    if sep_col is None or sep_col == 0 or sep_col == w - 1:
        raise ValueError("missing separator")
    if np.any(g[:, sep_col + 1 :] != BG):
        raise ValueError("input right side must be blank")

    colors = [int(v) for v in np.unique(g) if int(v) not in (BG, SEP)]
    frame_boxes: dict[int, tuple[int, int, int, int]] = {}
    anchors: dict[int, tuple[int, int]] = {}

    for color in colors:
        comps = _components(g == color)
        for comp in comps:
            if len(comp) == 1:
                if color in anchors:
                    raise ValueError("duplicate anchor")
                anchors[color] = comp[0]
                continue
            bbox = _as_frame_bbox(comp)
            if bbox is None:
                raise ValueError("non-frame foreground component")
            if color in frame_boxes:
                raise ValueError("duplicate frame color")
            frame_boxes[color] = bbox

    if not (2 <= len(frame_boxes) <= 4):
        raise ValueError("wrong frame count")
    if set(frame_boxes) != set(anchors):
        raise ValueError("frame-anchor mismatch")

    frames: list[Frame] = []
    for color, bbox in frame_boxes.items():
        anchor = anchors[color]
        if anchor[1] != sep_col - 1:
            raise ValueError("anchor must touch separator")
        center_row = (bbox[0] + bbox[2]) // 2
        if anchor[0] != center_row:
            raise ValueError("anchor row must equal frame center")
        if bbox[3] >= sep_col - 1:
            raise ValueError("frame overlaps anchor lane")
        depth = 1 + sum(
            _contains(other_bbox, bbox)
            for other_color, other_bbox in frame_boxes.items()
            if other_color != color
        )
        frames.append(Frame(color=color, bbox=bbox, anchor=anchor, depth=depth))

    if max(frame.depth for frame in frames) > 3:
        raise ValueError("nesting too deep")
    if len({frame.center_row for frame in frames}) != len(frames):
        raise ValueError("center rows must be unique")

    return sep_col, sorted(frames, key=lambda frame: (frame.depth, frame.center_row, frame.color))


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_col, frames = _parse_puzzle(g)
    out = g.copy()
    for frame in frames:
        for c in range(sep_col + 1, g.shape[1], frame.depth):
            out[frame.center_row, c] = frame.color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend separator-adjacent frame markers with a period equal to rectangle nesting depth.

    Input structure:
    - The grid has black background and one full-height gray separator column.
    - The left side contains 2-4 differently colored one-pixel rectangular frames.
    - Frames may contain other frames, with at least one black cell between nested borders.
    - Each frame color also appears as a single marker pixel immediately left of the separator on the
      frame's center row.
    - The entire right side of the separator is blank in the input.

    Rule:
    - Detect every colored rectangular frame on the left and compute its nesting depth from containment.
    - Use depth 1 for outermost frames, depth 2 for frames inside one frame, and so on.
    - For each frame, start on its center row at the first column right of the separator and paint that
      frame color every D columns, where D is the frame's depth.
    - If several frames target the same output pixel, the deeper frame wins.
    - Preserve the separator and the whole left side exactly.

    Key insight:
    - The nested geometry is not normalized or redrawn; it is converted into a numeric period, so
      containment depth directly controls the stride of the periodic extension.
    """
    g = np.asarray(grid, dtype=int)
    try:
        sep_col, frames = _parse_puzzle(g)
    except ValueError:
        return g.copy()

    out = g.copy()
    for frame in frames:
        for c in range(sep_col + 1, g.shape[1], frame.depth):
            out[frame.center_row, c] = frame.color
    return out


def _odd_values(lo: int, hi: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == 1]


def _sample_odd(rng: random.Random, lo: int, hi: int) -> int | None:
    values = _odd_values(lo, hi)
    if not values:
        return None
    return rng.choice(values)


def _random_partition(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1))
    result: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        result.append(cut - prev - 1)
        prev = cut
    return result


def _grow_odd_sizes(rng: random.Random, sizes: list[int], max_total: int) -> list[int]:
    grown = list(sizes)
    remaining = max_total - sum(grown)
    while remaining >= 2 and rng.random() < 0.85:
        idx = rng.randrange(len(grown))
        grown[idx] += 2
        remaining -= 2
    return grown


def _sample_root(
    rng: random.Random,
    h: int,
    left_w: int,
    min_h: int,
    min_w: int,
) -> tuple[int, int, int, int] | None:
    height = _sample_odd(rng, min_h, min(h - 2, 23))
    width = _sample_odd(rng, min_w, left_w - 3)
    if height is None or width is None:
        return None
    r0 = rng.randint(1, h - height - 1)
    c0 = rng.randint(1, left_w - width - 2)
    return (r0, c0, r0 + height - 1, c0 + width - 1)


def _sample_nested_child(
    rng: random.Random,
    parent: tuple[int, int, int, int],
    min_h: int,
    min_w: int,
    forbidden_rows: set[int],
) -> tuple[int, int, int, int] | None:
    pr0, pc0, pr1, pc1 = parent
    max_h = pr1 - pr0 - 3
    max_w = pc1 - pc0 - 3
    if max_h < min_h or max_w < min_w:
        return None
    for _ in range(120):
        height = _sample_odd(rng, min_h, max_h)
        width = _sample_odd(rng, min_w, max_w)
        if height is None or width is None:
            return None
        r0 = rng.randint(pr0 + 2, pr1 - height - 1)
        c0 = rng.randint(pc0 + 2, pc1 - width - 1)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        center_row = (bbox[0] + bbox[2]) // 2
        if center_row not in forbidden_rows:
            return bbox
    return None


def _sample_stacked_children(
    rng: random.Random,
    parent: tuple[int, int, int, int],
    min_heights: list[int],
    min_width: int,
) -> list[tuple[int, int, int, int]] | None:
    pr0, pc0, pr1, pc1 = parent
    available_rows = pr1 - pr0 - 3
    available_width = pc1 - pc0 - 3
    if available_width < min_width:
        return None
    base_total = sum(min_heights) + (len(min_heights) - 1)
    if base_total > available_rows:
        return None

    heights = _grow_odd_sizes(rng, min_heights, available_rows - (len(min_heights) - 1))
    used = sum(heights) + (len(heights) - 1)
    row_slack = available_rows - used
    gaps = _random_partition(rng, row_slack, len(heights) + 1)
    boxes: list[tuple[int, int, int, int]] = []
    cur = pr0 + 2 + gaps[0]

    for idx, height in enumerate(heights):
        width = _sample_odd(rng, min_width, available_width)
        if width is None:
            return None
        c0 = rng.randint(pc0 + 2, pc1 - width - 1)
        r0 = cur
        boxes.append((r0, c0, r0 + height - 1, c0 + width - 1))
        cur = r0 + height + 1 + gaps[idx + 1]

    return boxes


def _layout_bboxes(
    rng: random.Random,
    h: int,
    left_w: int,
    layout: str,
) -> list[tuple[int, int, int, int]] | None:
    if layout == "chain2":
        root = _sample_root(rng, h, left_w, 9, 9)
        if root is None:
            return None
        child = _sample_nested_child(rng, root, 5, 5, {(root[0] + root[2]) // 2})
        if child is None:
            return None
        return [root, child]

    if layout == "chain3":
        root = _sample_root(rng, h, left_w, 13, 13)
        if root is None:
            return None
        middle = _sample_nested_child(rng, root, 9, 9, {(root[0] + root[2]) // 2})
        if middle is None:
            return None
        forbidden = {(root[0] + root[2]) // 2, (middle[0] + middle[2]) // 2}
        inner = _sample_nested_child(rng, middle, 5, 5, forbidden)
        if inner is None:
            return None
        return [root, middle, inner]

    if layout == "branch3":
        root = _sample_root(rng, h, left_w, 15, 9)
        if root is None:
            return None
        children = _sample_stacked_children(rng, root, [5, 5], 5)
        if children is None:
            return None
        if (root[0] + root[2]) // 2 in {(box[0] + box[2]) // 2 for box in children}:
            return None
        return [root] + children

    if layout == "branch4":
        root = _sample_root(rng, h, left_w, 19, 9)
        if root is None:
            return None
        children = _sample_stacked_children(rng, root, [5, 9], 5)
        if children is None:
            return None
        host = children[-1]
        forbidden = {(root[0] + root[2]) // 2}
        forbidden.update((box[0] + box[2]) // 2 for box in children)
        inner = _sample_nested_child(rng, host, 5, 5, forbidden)
        if inner is None:
            return None
        return [root, children[0], host, inner]

    if layout == "triple":
        root = _sample_root(rng, h, left_w, 21, 9)
        if root is None:
            return None
        children = _sample_stacked_children(rng, root, [5, 5, 5], 5)
        if children is None:
            return None
        centers = {(box[0] + box[2]) // 2 for box in children}
        if len(centers) != 3 or (root[0] + root[2]) // 2 in centers:
            return None
        return [root] + children

    return None


def _render_instance(
    h: int,
    w: int,
    sep_col: int,
    bboxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:, sep_col] = SEP
    for color, bbox in zip(colors, bboxes):
        for r, c in _frame_boundary_cells(bbox):
            grid[r, c] = color
        center_row = (bbox[0] + bbox[2]) // 2
        grid[center_row, sep_col - 1] = color
    return grid


def _variant_signature(grid: np.ndarray) -> tuple:
    sep_col, frames = _parse_puzzle(grid)
    summary = tuple(
        (
            frame.depth,
            frame.center_row,
            frame.bbox[2] - frame.bbox[0] + 1,
            frame.bbox[3] - frame.bbox[1] + 1,
            frame.bbox[1],
        )
        for frame in frames
    )
    return (tuple(grid.shape), sep_col, summary)


def _is_valid_instance(grid: np.ndarray) -> bool:
    try:
        sep_col, frames = _parse_puzzle(grid)
    except ValueError:
        return False

    g = np.asarray(grid, dtype=int)
    if sep_col < 7 or g.shape[1] - sep_col - 1 < RIGHT_W_MIN:
        return False

    out = main(g)
    if not np.array_equal(out, _expected_output(g)):
        return False
    if np.array_equal(out, g):
        return False

    for frame in frames:
        row = frame.center_row
        period_hits = {c for c in range(sep_col + 1, g.shape[1], frame.depth)}
        actual_hits = {c for c in range(sep_col + 1, g.shape[1]) if int(out[row, c]) == frame.color}
        if not period_hits.issubset(actual_hits):
            return False

    return True


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    layout = rng.choice(LAYOUTS)
    h = rng.randint(GRID_H_MIN, GRID_H_MAX)
    left_w = rng.randint(LEFT_W_MIN, LEFT_W_MAX)
    right_w = rng.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    w = left_w + 1 + right_w
    if w > 29:
        return None

    bboxes = _layout_bboxes(rng, h, left_w, layout)
    if bboxes is None:
        return None

    colors = rng.sample(FRAME_COLORS, len(bboxes))
    grid = _render_instance(h, w, left_w, bboxes, colors)
    if not _is_valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth periodic puzzle")
