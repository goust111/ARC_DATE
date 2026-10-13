from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300
MAX_CLUSTER_ATTEMPTS = 200
GRID_MIN = 18
GRID_MAX = 42


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4 or int(coords.shape[0]) <= 8:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return np.array_equal(local, _border_mask(h, w))


def _component_patch(coords: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return patch, (r0, c0, r1, c1)


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _classify_side(
    fragment_box: tuple[int, int, int, int],
    frame_box: tuple[int, int, int, int],
) -> str:
    fr, fc = _bbox_center(fragment_box)
    cr, cc = _bbox_center(frame_box)
    dr = fr - cr
    dc = fc - cc
    if abs(dr) > abs(dc):
        return "top" if dr < 0 else "bottom"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored fragment into the same-colored rectangular frame.

    Fragments are assigned by color, then routed to the frame side implied by
    their bbox-center direction relative to the frame center. On one side,
    larger fragments are packed first against the inner wall and smaller ones
    follow toward the center while staying centered on the perpendicular axis.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    containers: dict[int, dict[str, object]] = {}

    for color in colors:
        comps = _components(g == color)
        frame_box = None
        fragments = []
        for coords in comps:
            if _is_frame_component(coords):
                assert frame_box is None
                frame_box = _bbox(coords)
                continue
            patch, frag_box = _component_patch(coords, color)
            fragments.append(
                {
                    "coords": coords,
                    "bbox": frag_box,
                    "patch": patch,
                    "area": int(coords.shape[0]),
                }
            )
        assert frame_box is not None
        containers[color] = {"frame": frame_box, "fragments": fragments}
        for frag in fragments:
            coords = frag["coords"]
            out[coords[:, 0], coords[:, 1]] = BG

    for color, data in containers.items():
        frame_box = data["frame"]
        fragments = data["fragments"]
        fr0, fc0, fr1, fc1 = frame_box
        inner_h = fr1 - fr0 - 1
        inner_w = fc1 - fc0 - 1
        occupied = np.zeros((inner_h, inner_w), dtype=bool)
        by_side = {side: [] for side in SIDES}
        for frag in fragments:
            by_side[_classify_side(frag["bbox"], frame_box)].append(frag)

        for side in SIDES:
            items = sorted(
                by_side[side],
                key=lambda frag: (-int(frag["area"]), int(frag["bbox"][0]), int(frag["bbox"][1])),
            )
            offset = 0
            for frag in items:
                patch = frag["patch"]
                mask = patch == color
                h, w = patch.shape
                if side == "top":
                    rr = offset
                    cc = (inner_w - w) // 2
                    offset += h
                elif side == "bottom":
                    rr = inner_h - offset - h
                    cc = (inner_w - w) // 2
                    offset += h
                elif side == "left":
                    rr = (inner_h - h) // 2
                    cc = offset
                    offset += w
                else:
                    rr = (inner_h - h) // 2
                    cc = inner_w - offset - w
                    offset += w

                assert 0 <= rr <= inner_h - h
                assert 0 <= cc <= inner_w - w
                assert not np.any(occupied[rr : rr + h, cc : cc + w] & mask)
                occupied[rr : rr + h, cc : cc + w] |= mask
                target = out[fr0 + 1 + rr : fr0 + 1 + rr + h, fc0 + 1 + cc : fc0 + 1 + cc + w]
                assert np.all(target[mask] == BG)
                target[mask] = color

    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_fragment_mask(area: int) -> np.ndarray:
    dims = [(h, w) for h in range(1, 5) for w in range(1, 5) if h * w >= area and h * w > 1]
    for _ in range(200):
        h, w = random.choice(dims)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_crop(mask)
        mh, mw = mask.shape
        if mh > 4 or mw > 4:
            continue
        if mh >= 3 and mw >= 3 and np.array_equal(mask, _border_mask(mh, mw)):
            continue
        if mask.all() and mh > 1 and mw > 1 and random.random() < 0.75:
            continue
        return mask
    raise RuntimeError("failed to build fragment mask")


def _fragment_patch(area: int, color: int) -> np.ndarray:
    mask = _random_fragment_mask(area)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _side_stats(patches: list[np.ndarray], axis: int) -> tuple[int, int]:
    if not patches:
        return 0, 0
    spans = [patch.shape[axis] for patch in patches]
    return sum(int(span) for span in spans), max(int(span) for span in spans)


def _clear_with_gap(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(hh, r0 + h + 1)
    cc1 = min(ww, c0 + w + 1)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_fragment_near_side(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    side: str,
    patch: np.ndarray,
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    fr0, fc0, fr1, fc1 = frame_box
    if side == "top":
        r_lo, r_hi = 0, fr0 - h - 1
        c_lo, c_hi = 0, gw - w
    elif side == "bottom":
        r_lo, r_hi = fr1 + 2, gh - h
        c_lo, c_hi = 0, gw - w
    elif side == "left":
        r_lo, r_hi = 0, gh - h
        c_lo, c_hi = 0, fc0 - w - 1
    else:
        r_lo, r_hi = 0, gh - h
        c_lo, c_hi = fc1 + 2, gw - w
    if r_lo > r_hi or c_lo > c_hi:
        return False

    for _ in range(250):
        rr = random.randint(r_lo, r_hi)
        cc = random.randint(c_lo, c_hi)
        box = (rr, cc, rr + h - 1, cc + w - 1)
        if _classify_side(box, frame_box) != side:
            continue
        if not _clear_with_gap(grid, rr, cc, h, w):
            continue
        mask = patch != BG
        grid[rr : rr + h, cc : cc + w][mask] = patch[mask]
        return True
    return False


def _build_cluster(color: int, side_list: list[str]) -> np.ndarray | None:
    for _ in range(MAX_CLUSTER_ATTEMPTS):
        side_to_patches = {side: [] for side in SIDES}
        valid = True
        for side in SIDES:
            count = int(side_list.count(side))
            if count == 0:
                continue
            areas = random.sample(range(2, 9), count)
            for area in areas:
                try:
                    side_to_patches[side].append(_fragment_patch(area, color))
                except RuntimeError:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        top_stack, _ = _side_stats(side_to_patches["top"], axis=0)
        bottom_stack, _ = _side_stats(side_to_patches["bottom"], axis=0)
        left_stack, _ = _side_stats(side_to_patches["left"], axis=1)
        right_stack, _ = _side_stats(side_to_patches["right"], axis=1)
        _, mid_h = _side_stats(side_to_patches["left"] + side_to_patches["right"], axis=0)
        _, mid_w = _side_stats(side_to_patches["top"] + side_to_patches["bottom"], axis=1)

        all_heights = [patch.shape[0] for side in SIDES for patch in side_to_patches[side]]
        all_widths = [patch.shape[1] for side in SIDES for patch in side_to_patches[side]]
        inner_h = max(6, top_stack + bottom_stack, mid_h + 2 * max(top_stack, bottom_stack), max(all_heights))
        inner_w = max(6, left_stack + right_stack, mid_w + 2 * max(left_stack, right_stack), max(all_widths))
        inner_h += random.randint(0, 2)
        inner_w += random.randint(0, 2)
        if inner_h > 12 or inner_w > 12:
            continue

        top_space = 1 + (max((patch.shape[0] for patch in side_to_patches["top"]), default=0)) + random.randint(0, 1)
        bottom_space = 1 + (max((patch.shape[0] for patch in side_to_patches["bottom"]), default=0)) + random.randint(0, 1)
        left_space = 1 + (max((patch.shape[1] for patch in side_to_patches["left"]), default=0)) + random.randint(0, 1)
        right_space = 1 + (max((patch.shape[1] for patch in side_to_patches["right"]), default=0)) + random.randint(0, 1)

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        patch_h = top_space + outer_h + bottom_space
        patch_w = left_space + outer_w + right_space
        if patch_h > 18 or patch_w > 18:
            continue

        grid = np.zeros((patch_h, patch_w), dtype=int)
        frame_box = (top_space, left_space, top_space + outer_h - 1, left_space + outer_w - 1)
        _draw_frame(grid, frame_box, color)

        fragments = []
        for side in SIDES:
            for patch in side_to_patches[side]:
                fragments.append((side, patch, int(np.count_nonzero(patch))))
        random.shuffle(fragments)
        fragments.sort(key=lambda item: -item[2])

        placed_all = True
        for side, patch, _area in fragments:
            if not _place_fragment_near_side(grid, frame_box, side, patch):
                placed_all = False
                break
        if not placed_all:
            continue
        return grid
    return None


def _choose_side_lists(n_containers: int) -> list[list[str]]:
    counts = [random.randint(2, 3) for _ in range(n_containers)]
    stack_idx = random.randrange(n_containers)
    counts[stack_idx] = 3

    side_lists = [[] for _ in range(n_containers)]
    duplicate_side = random.choice(SIDES)
    side_lists[stack_idx].extend([duplicate_side, duplicate_side])

    slots: list[int] = []
    for idx, count in enumerate(counts):
        slots.extend([idx] * (count - len(side_lists[idx])))
    random.shuffle(slots)

    required = [side for side in SIDES if side != duplicate_side]
    for idx, side in zip(slots, required):
        side_lists[idx].append(side)
    for idx in slots[len(required) :]:
        current = side_lists[idx]
        options = [side for side in SIDES if current.count(side) < 2]
        side_lists[idx].append(random.choice(options or list(SIDES)))

    for side_list in side_lists:
        random.shuffle(side_list)
    return side_lists


def _compose_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    layouts = {2: [(1, 2), (2, 1)], 3: [(1, 3), (3, 1), (2, 2)]}[len(patches)]
    random.shuffle(layouts)

    for rows, cols in layouts:
        order = list(patches)
        random.shuffle(order)
        cells = [None] * (rows * cols)
        positions = list(range(rows * cols))
        random.shuffle(positions)
        for idx, patch in zip(positions, order):
            cells[idx] = patch

        row_heights = []
        col_widths = []
        for r in range(rows):
            items = [cells[r * cols + c] for c in range(cols) if cells[r * cols + c] is not None]
            row_heights.append((max((item.shape[0] for item in items), default=0)) + random.randint(0, 1))
        for c in range(cols):
            items = [cells[r * cols + c] for r in range(rows) if cells[r * cols + c] is not None]
            col_widths.append((max((item.shape[1] for item in items), default=0)) + random.randint(0, 1))

        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        left_margin = random.randint(1, 2)
        right_margin = random.randint(1, 2)
        row_gaps = [random.randint(1, 2) for _ in range(rows - 1)]
        col_gaps = [random.randint(1, 2) for _ in range(cols - 1)]

        height = top_margin + bottom_margin + sum(row_heights) + sum(row_gaps)
        width = left_margin + right_margin + sum(col_widths) + sum(col_gaps)
        if not (GRID_MIN <= height <= GRID_MAX and GRID_MIN <= width <= GRID_MAX):
            continue

        grid = np.zeros((height, width), dtype=int)
        row_offsets = [top_margin]
        for gap, row_h in zip(row_gaps, row_heights):
            row_offsets.append(row_offsets[-1] + row_h + gap)
        col_offsets = [left_margin]
        for gap, col_w in zip(col_gaps, col_widths):
            col_offsets.append(col_offsets[-1] + col_w + gap)

        for r in range(rows):
            for c in range(cols):
                patch = cells[r * cols + c]
                if patch is None:
                    continue
                cell_r = row_offsets[r]
                cell_c = col_offsets[c]
                max_dr = row_heights[r] - patch.shape[0]
                max_dc = col_widths[c] - patch.shape[1]
                dr = random.randint(0, max_dr)
                dc = random.randint(0, max_dc)
                rr = cell_r + dr
                cc = cell_c + dc
                mask = patch != BG
                grid[rr : rr + patch.shape[0], cc : cc + patch.shape[1]][mask] = patch[mask]
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_containers = random.choice((2, 3))
        colors = random.sample(range(1, 10), n_containers)
        side_lists = _choose_side_lists(n_containers)

        patches = []
        ok = True
        for color, side_list in zip(colors, side_lists):
            patch = _build_cluster(color, side_list)
            if patch is None:
                ok = False
                break
            patches.append(patch)
        if not ok:
            continue

        grid = _compose_patches(patches)
        if grid is None:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate valid ARC input")
