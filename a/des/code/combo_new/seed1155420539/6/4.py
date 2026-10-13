import random
from collections import deque

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
MAX_GROUP_BUILD_ATTEMPTS = 100
GROUP_COUNT_RANGE = (1, 4)
INNER_SIZE_RANGE = (3, 9)
GROUP_OUTER_MAX = {1: 23, 2: 18, 3: 15, 4: 13}
SIDE_GAP_RANGE = (1, 2)
GROUP_SPACING_RANGE = (2, 4)
MARGIN_RANGE = (0, 3)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    frames: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or visited[r, c]:
                continue

            queue: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            coords_list: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords_list.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.array(coords_list, dtype=int)
            bbox = _bbox(coords)
            r0, c0, r1, c1 = bbox
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            assert box_h >= 3 and box_w >= 3, "each frame must enclose black interior"

            mask = _component_mask(coords, bbox)
            border = np.zeros_like(mask, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            assert np.array_equal(mask, border), "each colored component must be a 1px rectangular ring"

            frames.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "area": int(coords.shape[0]),
                    "box_area": int(box_h * box_w),
                }
            )

    assert frames, "expected at least one frame"
    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _frame_chains(grid: np.ndarray) -> list[list[dict[str, object]]]:
    frames = _find_frames(grid)
    count = len(frames)
    parents: list[int | None] = [None] * count
    children: list[list[int]] = [[] for _ in range(count)]

    for child_idx, child in enumerate(frames):
        best_parent = None
        best_area = None
        for parent_idx, parent in enumerate(frames):
            if child_idx == parent_idx:
                continue
            if not _contains(parent["bbox"], child["bbox"]):
                continue
            area = int(parent["box_area"])
            if best_area is None or area < best_area:
                best_area = area
                best_parent = parent_idx
        parents[child_idx] = best_parent
        if best_parent is not None:
            children[best_parent].append(child_idx)

    for child_list in children:
        assert len(child_list) <= 1, "each frame should have at most one direct nested child"

    roots = [idx for idx, parent in enumerate(parents) if parent is None]
    chains: list[list[dict[str, object]]] = []
    seen = 0
    for root in roots:
        chain_indices = []
        cur = root
        while True:
            chain_indices.append(cur)
            seen += 1
            if not children[cur]:
                break
            cur = children[cur][0]
        assert 2 <= len(chain_indices) <= 4, "each group must contain 2-4 nested frames"
        chains.append([frames[idx] for idx in chain_indices])

    assert seen == count, "every frame must belong to exactly one nested chain"
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each nested group's innermost frame color outward onto the top edges of its ancestor frames.

    Input structure:
    - The background is black.
    - The grid contains one or more disjoint groups of 2-4 strictly nested
      one-pixel-thick rectangular rings, with a different non-black color on
      each ring of a group.
    - Rings in the same group are separated by black gaps, and the innermost
      ring encloses only black cells.

    Rule:
    - Identify which rings belong to the same nested chain by strict rectangle
      containment, ordered from outermost to innermost.
    - Read the innermost ring color C for each chain.
    - Recolor the entire top edge of every ring in that chain to C; the
      innermost ring is unchanged because its top edge is already color C.
    - Leave all non-top-edge pixels unchanged.

    Key insight:
    - The transformation depends on hierarchy, not adjacency alone: the deepest
      contained frame determines the color that is copied only onto the top
      border of every enclosing frame in the same chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for chain in _frame_chains(g):
        inner_color = int(chain[-1]["color"])
        for frame in chain:
            r0, c0, _r1, c1 = frame["bbox"]
            out[r0, c0 : c1 + 1] = inner_color
    return out


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_group_patch(group_count: int) -> np.ndarray:
    allowed_layers = (2, 3) if group_count == 4 else (2, 3, 4)
    max_outer = GROUP_OUTER_MAX[group_count]

    for _ in range(MAX_GROUP_BUILD_ATTEMPTS):
        layer_count = random.choice(allowed_layers)
        gaps = [random.randint(*SIDE_GAP_RANGE) for _ in range(layer_count - 1)]
        side_offsets = [gap + 1 for gap in gaps]
        total_offset = sum(side_offsets)
        max_inner = min(INNER_SIZE_RANGE[1], max_outer - 2 * total_offset)
        if max_inner < INNER_SIZE_RANGE[0]:
            continue

        inner_h = random.randint(INNER_SIZE_RANGE[0], max_inner)
        inner_w = random.randint(INNER_SIZE_RANGE[0], max_inner)
        outer_h = inner_h + 2 * total_offset
        outer_w = inner_w + 2 * total_offset

        colors = random.sample(COLOR_POOL, layer_count)
        patch = np.zeros((outer_h, outer_w), dtype=int)
        inset = 0
        for idx, color in enumerate(colors):
            bbox = (inset, inset, outer_h - 1 - inset, outer_w - 1 - inset)
            _draw_ring(patch, bbox, color)
            if idx < len(side_offsets):
                inset += side_offsets[idx]

        return patch

    raise RuntimeError("failed to build a valid nested-frame group")


def _positive_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    edges = [0] + cuts + [total]
    return [edges[i + 1] - edges[i] for i in range(parts)]


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    window = grid[top : top + h, left : left + w]
    assert not np.any((window != BLACK) & (patch != BLACK))
    window[patch != BLACK] = patch[patch != BLACK]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        group_count = random.randint(*GROUP_COUNT_RANGE)
        patches = [_build_group_patch(group_count) for _ in range(group_count)]
        random.shuffle(patches)

        row_count = random.randint(1, min(3, group_count))
        row_sizes = _positive_partition(group_count, row_count)

        rows: list[list[np.ndarray]] = []
        cursor = 0
        for size in row_sizes:
            rows.append(patches[cursor : cursor + size])
            cursor += size

        horizontal_gaps = [
            [random.randint(*GROUP_SPACING_RANGE) for _ in range(max(0, len(row) - 1))]
            for row in rows
        ]
        vertical_gaps = [random.randint(*GROUP_SPACING_RANGE) for _ in range(max(0, row_count - 1))]
        top_margin = random.randint(*MARGIN_RANGE)
        bottom_margin = random.randint(*MARGIN_RANGE)
        left_margin = random.randint(*MARGIN_RANGE)
        right_margin = random.randint(*MARGIN_RANGE)

        row_heights = [max(int(patch.shape[0]) for patch in row) for row in rows]
        row_widths = []
        for row, gaps in zip(rows, horizontal_gaps):
            width = sum(int(patch.shape[1]) for patch in row) + sum(gaps)
            row_widths.append(width)

        grid_h = top_margin + sum(row_heights) + sum(vertical_gaps) + bottom_margin
        grid_w = left_margin + max(row_widths) + right_margin
        if grid_h > 30 or grid_w > 30:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        band_top = top_margin
        for row_idx, (row, row_height, row_width, gaps) in enumerate(zip(rows, row_heights, row_widths, horizontal_gaps)):
            band_left = left_margin + random.randint(0, grid_w - left_margin - right_margin - row_width)
            cur_left = band_left
            for patch_idx, patch in enumerate(row):
                max_down_shift = row_height - int(patch.shape[0])
                top = band_top + random.randint(0, max_down_shift)
                _stamp_patch(grid, patch, top, cur_left)
                cur_left += int(patch.shape[1])
                if patch_idx < len(gaps):
                    cur_left += gaps[patch_idx]
            band_top += row_height
            if row_idx < len(vertical_gaps):
                band_top += vertical_gaps[row_idx]

        chains = _frame_chains(grid)
        if len(chains) != group_count:
            continue
        if any(len(chain) < 2 or len(chain) > 4 for chain in chains):
            continue

        main(grid)
        return grid

    raise RuntimeError("failed to generate a valid nested-frame ARC input")
