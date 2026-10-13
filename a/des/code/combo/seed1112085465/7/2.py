from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 7
BLUE = 1
RED = 2
OBJECT_COLORS = (0, 3, 4, 5, 6, 8, 9)
GROUP_GAP = 2
MAX_GEN_RETRIES = 300
MAX_PLACE_TRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or labels[r, c] != -1:
                continue

            q = deque([(r, c)])
            labels[r, c] = len(comps)
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1 and int(grid[nr, nc]) == color:
                        labels[nr, nc] = len(comps)
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
            patch[arr[:, 0] - r0, arr[:, 1] - c0] = color
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "neighbors": set(),
                }
            )

    for r in range(h):
        for c in range(w):
            cid = int(labels[r, c])
            if cid == -1:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    nid = int(labels[nr, nc])
                    if nid != -1 and nid != cid:
                        comps[cid]["neighbors"].add(nid)

    return comps, labels


def _is_instruction_component(comp: dict, comps: list[dict]) -> bool:
    color = int(comp["color"])
    if color not in (BLUE, RED):
        return False

    neighbor_ids = list(comp["neighbors"])
    if len(neighbor_ids) < 2:
        return False

    r0, c0, r1, c1 = comp["bbox"]
    if color == BLUE:
        has_above = any(comps[nid]["bbox"][2] < r0 for nid in neighbor_ids)
        has_below = any(comps[nid]["bbox"][0] > r1 for nid in neighbor_ids)
        return bool(has_above and has_below)

    has_left = any(comps[nid]["bbox"][3] < c0 for nid in neighbor_ids)
    has_right = any(comps[nid]["bbox"][1] > c1 for nid in neighbor_ids)
    return bool(has_left and has_right)


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> list[set[int]]:
        buckets: dict[int, set[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            buckets.setdefault(root, set()).add(i)
        return list(buckets.values())


def _group_key(member_ids: set[int], objects: list[dict], orient: str) -> tuple[int, int]:
    if orient == "v":
        return (
            min(objects[idx]["bbox"][1] for idx in member_ids),
            min(objects[idx]["bbox"][0] for idx in member_ids),
        )
    return (
        min(objects[idx]["bbox"][0] for idx in member_ids),
        min(objects[idx]["bbox"][1] for idx in member_ids),
    )


def _render_group(member_ids: set[int], objects: list[dict], orient: str) -> np.ndarray:
    ordered = sorted(
        member_ids,
        key=(
            (lambda idx: (objects[idx]["bbox"][0], objects[idx]["bbox"][1]))
            if orient == "v"
            else (lambda idx: (objects[idx]["bbox"][1], objects[idx]["bbox"][0]))
        ),
    )
    patches = [objects[idx]["patch"] for idx in ordered]

    if orient == "v":
        out_h = sum(p.shape[0] for p in patches)
        out_w = max(p.shape[1] for p in patches)
        out = np.full((out_h, out_w), BG, dtype=int)
        row = 0
        for patch in patches:
            sub = out[row : row + patch.shape[0], : patch.shape[1]]
            mask = patch != BG
            sub[mask] = patch[mask]
            row += patch.shape[0]
        return out

    out_h = max(p.shape[0] for p in patches)
    out_w = sum(p.shape[1] for p in patches)
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for patch in patches:
        sub = out[: patch.shape[0], col : col + patch.shape[1]]
        mask = patch != BG
        sub[mask] = patch[mask]
        col += patch.shape[1]
    return out


def _pack_side_by_side(patches: list[np.ndarray]) -> np.ndarray:
    if not patches:
        return np.full((0, 0), BG, dtype=int)

    out_h = max(p.shape[0] for p in patches)
    out_w = sum(p.shape[1] for p in patches) + GROUP_GAP * (len(patches) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for patch in patches:
        sub = out[: patch.shape[0], col : col + patch.shape[1]]
        mask = patch != BG
        sub[mask] = patch[mask]
        col += patch.shape[1] + GROUP_GAP
    return out


def _pack_top_to_bottom(patches: list[np.ndarray]) -> np.ndarray:
    if not patches:
        return np.full((0, 0), BG, dtype=int)

    out_h = sum(p.shape[0] for p in patches) + GROUP_GAP * (len(patches) - 1)
    out_w = max(p.shape[1] for p in patches)
    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    for patch in patches:
        sub = out[row : row + patch.shape[0], : patch.shape[1]]
        mask = patch != BG
        sub[mask] = patch[mask]
        row += patch.shape[0] + GROUP_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by colored inter-object gap instructions and repack them.

    Blue gap components connect a vertical group: member objects are stacked
    top-to-bottom with no gaps and left-aligned. Red gap components connect a
    horizontal group: member objects are packed left-to-right with no gaps and
    top-aligned. Vertical groups are then placed side-by-side, horizontal groups
    are placed top-to-bottom, and the two sections are separated by two
    background rows when both are present.
    """
    g = np.asarray(grid, dtype=int)
    if g.size == 0:
        return g.copy()

    comps, _ = _find_components(g)
    if not comps:
        return np.full((1, 1), BG, dtype=int)

    instruction_ids = {cid for cid, comp in enumerate(comps) if _is_instruction_component(comp, comps)}

    objects: list[dict] = []
    comp_to_object: dict[int, int] = {}
    for cid, comp in enumerate(comps):
        if cid in instruction_ids:
            continue
        comp_to_object[cid] = len(objects)
        objects.append(comp)

    if not objects:
        return np.full((1, 1), BG, dtype=int)

    blue_dsu = _DSU(len(objects))
    red_dsu = _DSU(len(objects))

    for cid in instruction_ids:
        touched = sorted({comp_to_object[nid] for nid in comps[cid]["neighbors"] if nid in comp_to_object})
        if len(touched) < 2:
            continue
        dsu = blue_dsu if int(comps[cid]["color"]) == BLUE else red_dsu
        anchor = touched[0]
        for other in touched[1:]:
            dsu.union(anchor, other)

    blue_groups = [members for members in blue_dsu.groups() if len(members) > 1]
    red_groups = [members for members in red_dsu.groups() if len(members) > 1]

    candidates = [("v", members) for members in blue_groups] + [("h", members) for members in red_groups]
    candidates.sort(key=lambda item: (-len(item[1]), 0 if item[0] == "v" else 1, _group_key(item[1], objects, item[0])))

    chosen_vertical: list[set[int]] = []
    chosen_horizontal: list[set[int]] = []
    used: set[int] = set()

    for orient, members in candidates:
        if members & used:
            continue
        if orient == "v":
            chosen_vertical.append(members)
        else:
            chosen_horizontal.append(members)
        used |= members

    for idx in range(len(objects)):
        if idx not in used:
            chosen_vertical.append({idx})

    chosen_vertical.sort(key=lambda members: _group_key(members, objects, "v"))
    chosen_horizontal.sort(key=lambda members: _group_key(members, objects, "h"))

    vertical_patches = [_render_group(members, objects, "v") for members in chosen_vertical]
    horizontal_patches = [_render_group(members, objects, "h") for members in chosen_horizontal]

    top = _pack_side_by_side(vertical_patches) if vertical_patches else None
    bottom = _pack_top_to_bottom(horizontal_patches) if horizontal_patches else None

    if top is None and bottom is None:
        return np.full((1, 1), BG, dtype=int)
    if top is None:
        return bottom
    if bottom is None:
        return top

    out_h = top.shape[0] + GROUP_GAP + bottom.shape[0]
    out_w = max(top.shape[1], bottom.shape[1])
    out = np.full((out_h, out_w), BG, dtype=int)
    out[: top.shape[0], : top.shape[1]] = top
    out[top.shape[0] + GROUP_GAP :, : bottom.shape[1]] = bottom
    return out


def _make_patch(height: int, width: int, color: int, orient: str) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    mask = np.ones((height, width), dtype=bool)

    if height >= 2 and width >= 2 and random.random() < 0.55:
        corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
        random.shuffle(corners)
        for rr, cc in corners:
            trial = mask.copy()
            trial[rr, cc] = False
            if orient == "v" and trial[0].any() and trial[-1].any():
                mask = trial
                break
            if orient == "h" and trial[:, 0].any() and trial[:, -1].any():
                mask = trial
                break
            if orient == "s" and trial.any():
                mask = trial
                break

    patch[mask] = color
    return patch


def _top_columns(patch: np.ndarray) -> set[int]:
    return {int(c) for c in np.where(patch[0] != BG)[0]}


def _bottom_columns(patch: np.ndarray) -> set[int]:
    return {int(c) for c in np.where(patch[-1] != BG)[0]}


def _left_rows(patch: np.ndarray) -> set[int]:
    return {int(r) for r in np.where(patch[:, 0] != BG)[0]}


def _right_rows(patch: np.ndarray) -> set[int]:
    return {int(r) for r in np.where(patch[:, -1] != BG)[0]}


def _build_vertical_group(colors: list[int]) -> np.ndarray | None:
    patches: list[np.ndarray] = []
    lefts: list[int] = []
    rows: list[int] = []
    gaps: list[int] = []

    for i, color in enumerate(colors):
        height = random.randint(1, 4)
        width = random.randint(2, 5)
        patch = _make_patch(height, width, color, "v")

        if i == 0:
            left = random.randint(0, 2)
        else:
            candidates = []
            for cand in range(4):
                upper = {lefts[-1] + c for c in _bottom_columns(patches[-1])}
                lower = {cand + c for c in _top_columns(patch)}
                if upper & lower:
                    candidates.append(cand)
            if not candidates:
                return None
            left = random.choice(candidates)
            gaps.append(random.randint(1, 3))

        patches.append(patch)
        lefts.append(left)

    row = 0
    for i, patch in enumerate(patches):
        rows.append(row)
        row += patch.shape[0]
        if i < len(gaps):
            row += gaps[i]

    out_h = row
    out_w = max(left + patch.shape[1] for left, patch in zip(lefts, patches))
    grid = np.full((out_h, out_w), BG, dtype=int)

    for patch, top, left in zip(patches, rows, lefts):
        sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        mask = patch != BG
        sub[mask] = patch[mask]

    for i in range(len(gaps)):
        upper_cols = {lefts[i] + c for c in _bottom_columns(patches[i])}
        lower_cols = {lefts[i + 1] + c for c in _top_columns(patches[i + 1])}
        common = sorted(upper_cols & lower_cols)
        if not common:
            return None
        col = random.choice(common)
        start = rows[i] + patches[i].shape[0]
        end = rows[i + 1]
        grid[start:end, col] = BLUE

    return grid


def _build_horizontal_group(colors: list[int]) -> np.ndarray | None:
    patches: list[np.ndarray] = []
    tops: list[int] = []
    cols: list[int] = []
    gaps: list[int] = []

    for i, color in enumerate(colors):
        height = random.randint(2, 5)
        width = random.randint(1, 4)
        patch = _make_patch(height, width, color, "h")

        if i == 0:
            top = random.randint(0, 2)
        else:
            candidates = []
            for cand in range(4):
                left_rows = {tops[-1] + r for r in _right_rows(patches[-1])}
                right_rows = {cand + r for r in _left_rows(patch)}
                if left_rows & right_rows:
                    candidates.append(cand)
            if not candidates:
                return None
            top = random.choice(candidates)
            gaps.append(random.randint(1, 3))

        patches.append(patch)
        tops.append(top)

    col = 0
    for i, patch in enumerate(patches):
        cols.append(col)
        col += patch.shape[1]
        if i < len(gaps):
            col += gaps[i]

    out_h = max(top + patch.shape[0] for top, patch in zip(tops, patches))
    out_w = col
    grid = np.full((out_h, out_w), BG, dtype=int)

    for patch, top, left in zip(patches, tops, cols):
        sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        mask = patch != BG
        sub[mask] = patch[mask]

    for i in range(len(gaps)):
        left_rows = {tops[i] + r for r in _right_rows(patches[i])}
        right_rows = {tops[i + 1] + r for r in _left_rows(patches[i + 1])}
        common = sorted(left_rows & right_rows)
        if not common:
            return None
        row = random.choice(common)
        start = cols[i] + patches[i].shape[1]
        end = cols[i + 1]
        grid[row, start:end] = RED

    return grid


def _build_singleton(color: int) -> np.ndarray:
    height = random.randint(1, 4)
    width = random.randint(1, 4)
    return _make_patch(height, width, color, "s")


def _sample_group_plan() -> list[tuple[str, int]]:
    while True:
        mode = random.choice(("v_only", "h_only", "mixed"))
        plan: list[tuple[str, int]] = []

        if mode == "v_only":
            for _ in range(random.randint(2, 3)):
                plan.append(("v", random.randint(2, 3)))
        elif mode == "h_only":
            for _ in range(random.randint(2, 3)):
                plan.append(("h", random.randint(2, 3)))
        else:
            plan.append(("v", random.randint(2, 3)))
            plan.append(("h", random.randint(2, 3)))
            if random.random() < 0.45:
                plan.append((random.choice(("v", "h")), 2))

        if random.random() < 0.45:
            plan.append(("s", 1))
        if random.random() < 0.20:
            plan.append(("s", 1))

        total = sum(size for _, size in plan)
        if 4 <= total <= len(OBJECT_COLORS):
            return plan


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    pts = np.argwhere(mask)
    for r, c in pts:
        r0 = max(0, int(r) - radius)
        r1 = min(h, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(w, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _place_groups(group_grids: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(grid.shape[0] for grid in group_grids)
    max_w = max(grid.shape[1] for grid in group_grids)
    min_h = max(18, max_h + 4)
    min_w = max(18, max_w + 4)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    canvas = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(range(len(group_grids)))
    random.shuffle(order)
    for idx in order:
        local = group_grids[idx]
        h, w = local.shape
        cells = np.argwhere(local != BG)
        placed = False
        for _ in range(MAX_PLACE_TRIES):
            top = random.randint(0, grid_h - h)
            left = random.randint(0, grid_w - w)
            shifted = np.zeros_like(canvas, dtype=bool)
            shifted[top + cells[:, 0], left + cells[:, 1]] = True
            if np.any(shifted & blocked):
                continue

            patch = canvas[top : top + h, left : left + w]
            mask = local != BG
            patch[mask] = local[mask]
            blocked |= _dilate(shifted, radius=2)
            placed = True
            break

        if not placed:
            return None

    return canvas


def _gen_attempt() -> np.ndarray | None:
    plan = _sample_group_plan()
    colors = random.sample(OBJECT_COLORS, sum(size for _, size in plan))
    cursor = 0
    group_grids: list[np.ndarray] = []

    for orient, size in plan:
        group_colors = colors[cursor : cursor + size]
        cursor += size

        if orient == "v":
            group = _build_vertical_group(group_colors)
        elif orient == "h":
            group = _build_horizontal_group(group_colors)
        else:
            group = _build_singleton(group_colors[0])

        if group is None:
            return None
        group_grids.append(group)

    grid = _place_groups(group_grids)
    if grid is None:
        return None

    out = main(grid)
    if out.size == 0 or np.array_equal(grid, out):
        return None
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_RETRIES} attempts")
