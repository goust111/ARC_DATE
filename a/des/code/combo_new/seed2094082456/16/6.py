from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIRS8 = DIRS4 + DIAGS
ROTATION_BY_GROUP_SIZE = {2: 1, 3: 2, 4: 3}
PALETTE = tuple(range(10))
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_fragments(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    fragments: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            fragments.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": _bbox_from_coords(coords),
                }
            )

    return fragments


def _fragments_touch_8(a: dict, b: dict) -> bool:
    b_coords = b["coord_set"]
    for r, c in a["coords"]:
        for dr, dc in DIRS8:
            if (r + dr, c + dc) in b_coords:
                return True
    return False


def _group_fragments(fragments: list[dict]) -> list[list[dict]]:
    n = len(fragments)
    groups: list[list[dict]] = []
    used = [False] * n
    adjacency = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _fragments_touch_8(fragments[i], fragments[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)

    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        queue = deque([i])
        group_indices: list[int] = []
        while queue:
            cur = queue.popleft()
            group_indices.append(cur)
            for nxt in adjacency[cur]:
                if used[nxt]:
                    continue
                used[nxt] = True
                queue.append(nxt)
        groups.append([fragments[idx] for idx in group_indices])

    groups.sort(key=lambda group: min(f["bbox"][0] for f in group))
    return groups


def _group_patch(
    grid: np.ndarray, group: list[dict], bg: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = [coord for fragment in group for coord in fragment["coords"]]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _paste_with_clip(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    for rr in range(h):
        for cc in range(w):
            value = int(patch[rr, cc])
            if value == bg:
                continue
            tr, tc = r0 + rr, c0 + cc
            if _in_bounds(canvas.shape, tr, tc):
                canvas[tr, tc] = value


def _rotation_steps(group_size: int) -> int:
    return int(ROTATION_BY_GROUP_SIZE.get(group_size, 0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge diagonally touching monochrome fragments into groups and redraw each group after a size-coded rotation.

    Input structure:
    - The background is the dominant color.
    - Foreground pixels form several small monochrome 4-connected fragments of sizes 2-5.
    - Fragments with different colors may touch only through 8-neighbor contact, which defines a group.
    - Separate groups do not touch each other, even diagonally.

    Rule:
    - Extract every monochrome 4-connected fragment, then merge fragments into one group whenever their pixels are 8-neighbors.
    - For each group, crop its tight bounding box and treat all fragment colors and relative offsets as one composite pattern.
    - Rotate that composite clockwise by 90 degrees for 2 fragments, 180 degrees for 3 fragments, and 270 degrees for 4 fragments.
    - Clear the original fragments and paste the rotated composite so that the old and new bounding-box centers coincide, clipping at the grid boundary if needed.

    Key insight:
    - The task has two hidden layers: 8-neighbor contact determines which fragments belong together, and the number of fragments inside each group encodes the rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fragments = _extract_fragments(g, bg)
    groups = _group_fragments(fragments)

    out = np.full_like(g, bg)
    for group in groups:
        patch, (r0, c0, r1, c1) = _group_patch(g, group, bg)
        steps = _rotation_steps(len(group))
        rotated = np.rot90(patch, -steps)

        old_h = r1 - r0 + 1
        old_w = c1 - c0 + 1
        new_h, new_w = rotated.shape
        new_r0 = (2 * r0 + old_h - new_h) // 2
        new_c0 = (2 * c0 + old_w - new_w) // 2
        _paste_with_clip(out, rotated, int(new_r0), int(new_c0), bg)

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_fragment_mask(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                pos = (r + dr, c + dc)
                if pos not in cells:
                    frontier.add(pos)
        cells.add(random.choice(tuple(frontier)))

    mask = _cells_to_mask(cells)
    mask = np.rot90(mask, random.randrange(4))
    if random.randrange(2):
        mask = np.fliplr(mask)
    return mask


def _crop_to_bbox(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(arr != bg)
    if coords.size == 0:
        return arr.copy()
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _fragment_position_is_valid(
    occupied: np.ndarray, frag_mask: np.ndarray, r0: int, c0: int
) -> tuple[bool, bool]:
    has_diagonal_contact = False
    for rr, cc in np.argwhere(frag_mask):
        ar, ac = int(r0 + rr), int(c0 + cc)
        if occupied[ar, ac]:
            return False, False
        for dr, dc in DIRS4:
            nr, nc = ar + dr, ac + dc
            if _in_bounds(occupied.shape, nr, nc) and occupied[nr, nc]:
                return False, False
        for dr, dc in DIAGS:
            nr, nc = ar + dr, ac + dc
            if _in_bounds(occupied.shape, nr, nc) and occupied[nr, nc]:
                has_diagonal_contact = True
    return True, has_diagonal_contact


def _random_dihedral(arr: np.ndarray) -> np.ndarray:
    out = np.rot90(arr, random.randrange(4))
    if random.randrange(2):
        out = np.fliplr(out)
    return out


def _make_group_patch(group_size: int, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(200):
        work_h = random.randint(8, 10)
        work_w = random.randint(8, 10)
        canvas = np.full((work_h, work_w), bg, dtype=int)
        occupied = np.zeros((work_h, work_w), dtype=bool)
        colors = random.sample([c for c in PALETTE if c != bg], group_size)

        ok = True
        for idx in range(group_size):
            frag_mask = _random_fragment_mask(random.randint(2, 5))
            h, w = frag_mask.shape
            candidates: list[tuple[int, int]] = []

            if idx == 0:
                for r0 in range(work_h - h + 1):
                    for c0 in range(work_w - w + 1):
                        border_touch = r0 == 0 or c0 == 0 or r0 + h == work_h or c0 + w == work_w
                        if not border_touch:
                            candidates.append((r0, c0))
                if not candidates:
                    candidates = [(r0, c0) for r0 in range(work_h - h + 1) for c0 in range(work_w - w + 1)]
            else:
                for r0 in range(work_h - h + 1):
                    for c0 in range(work_w - w + 1):
                        valid, diagonal = _fragment_position_is_valid(occupied, frag_mask, r0, c0)
                        if valid and diagonal:
                            candidates.append((r0, c0))

            if not candidates:
                ok = False
                break

            r0, c0 = random.choice(candidates)
            canvas[r0 : r0 + h, c0 : c0 + w][frag_mask] = colors[idx]
            occupied[r0 : r0 + h, c0 : c0 + w] |= frag_mask

        if not ok:
            continue

        patch = _crop_to_bbox(canvas, bg)
        patch = _random_dihedral(patch)
        out_patch = np.rot90(patch, -_rotation_steps(group_size))

        if np.array_equal(patch, out_patch):
            continue
        if (patch.shape[0] - out_patch.shape[0]) % 2 != 0:
            continue
        if (patch.shape[1] - out_patch.shape[1]) % 2 != 0:
            continue

        fragments = _extract_fragments(patch, bg)
        groups = _group_fragments(fragments)
        if len(groups) != 1 or len(groups[0]) != group_size:
            continue
        if any(not (2 <= len(fragment["coords"]) <= 5) for fragment in fragments):
            continue
        if len({fragment["color"] for fragment in fragments}) != group_size:
            continue
        return patch, out_patch

    return None


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    for rr, cc in np.argwhere(patch != bg):
        ar, ac = int(r0 + rr), int(c0 + cc)
        r1 = max(0, ar - 1)
        r2 = min(occupied.shape[0], ar + 2)
        c1 = max(0, ac - 1)
        c2 = min(occupied.shape[1], ac + 2)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def _paste_patch(canvas: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    region[mask] = patch[mask]
    occupied[r0 : r0 + h, c0 : c0 + w] |= mask


def _choose_group_sizes() -> list[int]:
    if random.random() < 0.5:
        return random.sample([2, 3, 4], 2)
    sizes = [2, 3, 4]
    random.shuffle(sizes)
    return sizes


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(PALETTE)
    group_sizes = _choose_group_sizes()
    group_specs = []
    for size in group_sizes:
        patches = _make_group_patch(size, bg)
        if patches is None:
            return None
        in_patch, out_patch = patches
        group_specs.append({"size": size, "in_patch": in_patch, "out_patch": out_patch})

    patch_span = max(
        max(max(spec["in_patch"].shape), max(spec["out_patch"].shape))
        for spec in group_specs
    )
    min_side = max(14, patch_span + 4, 8 + 4 * len(group_specs))
    if min_side > 26:
        return None

    height = random.randint(min_side, 26)
    width = random.randint(min_side, 26)
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    inp_occupied = np.zeros((height, width), dtype=bool)
    out_occupied = np.zeros((height, width), dtype=bool)

    order = list(range(len(group_specs)))
    random.shuffle(order)
    order.sort(
        key=lambda idx: max(
            group_specs[idx]["in_patch"].shape[0] * group_specs[idx]["in_patch"].shape[1],
            group_specs[idx]["out_patch"].shape[0] * group_specs[idx]["out_patch"].shape[1],
        ),
        reverse=True,
    )

    for idx in order:
        spec = group_specs[idx]
        in_patch = spec["in_patch"]
        out_patch = spec["out_patch"]
        in_h, in_w = in_patch.shape
        out_h, out_w = out_patch.shape
        candidates: list[tuple[int, int, int, int]] = []

        for in_r0 in range(height - in_h + 1):
            out_r0 = in_r0 + (in_h - out_h) // 2
            if out_r0 < 0 or out_r0 + out_h > height:
                continue
            for in_c0 in range(width - in_w + 1):
                out_c0 = in_c0 + (in_w - out_w) // 2
                if out_c0 < 0 or out_c0 + out_w > width:
                    continue
                if not _can_place_patch(inp_occupied, in_patch, in_r0, in_c0, bg):
                    continue
                if not _can_place_patch(out_occupied, out_patch, out_r0, out_c0, bg):
                    continue
                candidates.append((in_r0, in_c0, out_r0, out_c0))

        if not candidates:
            return None

        in_r0, in_c0, out_r0, out_c0 = random.choice(candidates)
        _paste_patch(inp, inp_occupied, in_patch, in_r0, in_c0, bg)
        _paste_patch(out, out_occupied, out_patch, out_r0, out_c0, bg)

    fragments = _extract_fragments(inp, bg)
    groups = _group_fragments(fragments)
    if sorted(len(group) for group in groups) != sorted(group_sizes):
        return None
    if any(len({fragment["color"] for fragment in group}) != len(group) for group in groups):
        return None
    if any(not (2 <= len(fragment["coords"]) <= 5) for fragment in fragments):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid input")
