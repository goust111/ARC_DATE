from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BG = 0
OBJECT = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = tuple(DIRS.values())

GEN_ATTEMPTS = 400
GRID_H_RANGE = (14, 23)
GRID_W_RANGE = (14, 23)
OBJECT_COUNT_CHOICES = (1, 2, 3)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 7)
MAX_GROUPS_PER_SIDE = 2
MAX_MARKERS_PER_GROUP = 4
PLACEMENT_ATTEMPTS = 120


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            labels[r, c] = comp_id
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIR_ORDER:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _extract_groups(grid: np.ndarray) -> dict[tuple[int, int, int, int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    labels, n_comp = _label_components(g == OBJECT)
    assert n_comp > 0, "expected at least one grey object"

    groups: dict[tuple[int, int, int, int], set[tuple[int, int]]] = defaultdict(set)
    for r, c in np.argwhere((g != BG) & (g != OBJECT)):
        color = int(g[r, c])
        attachments: set[tuple[int, int, int]] = set()
        for dr, dc in DIR_ORDER:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                comp_id = int(labels[nr, nc])
                if comp_id > 0:
                    attachments.add((comp_id, -int(dr), -int(dc)))
        assert len(attachments) == 1, "each marker must belong to exactly one object side"
        comp_id, arm_dr, arm_dc = next(iter(attachments))
        groups[(comp_id, color, arm_dr, arm_dc)].add((int(r), int(c)))
    assert groups, "expected at least one colored marker group"
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert boundary-adjacent marker counts into straight colored arms.

    Input structure
    - Background is black.
    - One or more 4-connected grey objects appear on the grid.
    - Colored singleton markers sit exactly one cell outside object sides.

    Rule
    - For each grey object, group its markers by color and side.
    - A group with k markers becomes an arm of length k in that direction.
    - The arm keeps the same side-adjacent footprint as the markers and extends
      farther outward for the remaining k - 1 layers.
    - Original marker pixels are erased before painting the arms.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    groups = _extract_groups(g)

    for bases in groups.values():
        for r, c in bases:
            out[r, c] = BG

    for (comp_id, color, dr, dc), bases in groups.items():
        arm_len = len(bases)
        for r, c in bases:
            for step in range(arm_len):
                rr = int(r + dr * step)
                cc = int(c + dc * step)
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] != OBJECT:
                    out[rr, cc] = color

    return out


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _split_slack(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(total + parts - 1), size=parts - 1, replace=False).tolist())
    cuts = [-1] + cuts + [total + parts - 1]
    return [cuts[i + 1] - cuts[i] - 1 for i in range(parts)]


def _sample_side_groups(
    rng: np.random.Generator,
    side_len: int,
    available_colors: list[int],
) -> list[dict[str, int]]:
    if side_len <= 0 or not available_colors:
        return []

    max_groups = min(MAX_GROUPS_PER_SIDE, side_len)
    want_two = max_groups >= 2 and rng.random() < 0.45
    n_groups = 2 if want_two else 1

    if n_groups == 1:
        size = _randint(rng, 1, min(MAX_MARKERS_PER_GROUP, side_len))
        start = _randint(rng, 0, side_len - size)
        return [{"start": start, "size": size, "color": int(rng.choice(available_colors))}]

    valid_pairs: list[tuple[int, int]] = []
    upper = min(MAX_MARKERS_PER_GROUP, side_len)
    for a in range(1, upper + 1):
        for b in range(1, upper + 1):
            if a + b + 1 <= side_len:
                valid_pairs.append((a, b))

    if not valid_pairs:
        size = _randint(rng, 1, min(MAX_MARKERS_PER_GROUP, side_len))
        start = _randint(rng, 0, side_len - size)
        return [{"start": start, "size": size, "color": int(rng.choice(available_colors))}]

    size_a, size_b = valid_pairs[_randint(rng, 0, len(valid_pairs) - 1)]
    slack = side_len - (size_a + size_b + 1)
    before, mid_extra, after = _split_slack(rng, slack, 3)
    start_a = before
    start_b = start_a + size_a + 1 + mid_extra
    colors = rng.choice(np.array(available_colors, dtype=int), size=2, replace=False).tolist()
    return [
        {"start": int(start_a), "size": int(size_a), "color": int(colors[0])},
        {"start": int(start_b), "size": int(size_b), "color": int(colors[1])},
    ]


def _build_object_spec(rng: np.random.Generator) -> dict | None:
    h = _randint(rng, *RECT_H_RANGE)
    w = _randint(rng, *RECT_W_RANGE)

    all_colors = list(rng.permutation(PALETTE))
    side_groups: dict[str, list[dict[str, int]]] = {}
    used_any = False

    for side in DIRS:
        if rng.random() < 0.55:
            side_len = w if side in ("up", "down") else h
            groups = _sample_side_groups(rng, side_len, all_colors)
            if groups:
                side_groups[side] = groups
                used_any = True

    if not used_any:
        forced_side = str(rng.choice(np.array(list(DIRS.keys()), dtype=object)))
        side_len = w if forced_side in ("up", "down") else h
        side_groups[forced_side] = _sample_side_groups(rng, side_len, all_colors)

    sizes = [group["size"] for groups in side_groups.values() for group in groups]
    if not sizes or max(sizes) <= 1:
        candidate_sides = [side for side in DIRS if (w if side in ("up", "down") else h) >= 2]
        if not candidate_sides:
            return None
        side = str(rng.choice(np.array(candidate_sides, dtype=object)))
        limit = w if side in ("up", "down") else h
        size = _randint(rng, 2, min(MAX_MARKERS_PER_GROUP, limit))
        start = _randint(rng, 0, limit - size)
        side_groups[side] = [{"start": start, "size": size, "color": int(rng.choice(PALETTE))}]

    pads = {side: max((group["size"] for group in groups), default=0) for side, groups in side_groups.items()}
    for side in DIRS:
        pads.setdefault(side, 0)

    return {"h": h, "w": w, "groups": side_groups, "pads": pads}


def _object_reserved_bbox(spec: dict, top: int, left: int) -> tuple[int, int, int, int]:
    h = int(spec["h"])
    w = int(spec["w"])
    pads = spec["pads"]
    return (
        top - int(pads["up"]),
        left - int(pads["left"]),
        top + h - 1 + int(pads["down"]),
        left + w - 1 + int(pads["right"]),
    )


def _draw_object(grid: np.ndarray, spec: dict, top: int, left: int) -> None:
    h = int(spec["h"])
    w = int(spec["w"])
    grid[top : top + h, left : left + w] = OBJECT

    for side, groups in spec["groups"].items():
        for group in groups:
            start = int(group["start"])
            size = int(group["size"])
            color = int(group["color"])
            if side == "up":
                r = top - 1
                grid[r, left + start : left + start + size] = color
            elif side == "down":
                r = top + h
                grid[r, left + start : left + start + size] = color
            elif side == "left":
                c = left - 1
                grid[top + start : top + start + size, c] = color
            else:
                c = left + w
                grid[top + start : top + start + size, c] = color


def _can_place(blocked: np.ndarray, spec: dict, top: int, left: int) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = _object_reserved_bbox(spec, top, left)
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_blocked(blocked: np.ndarray, spec: dict, top: int, left: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = _object_reserved_bbox(spec, top, left)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _scene_has_direction_variety(specs: list[dict]) -> bool:
    directions = {side for spec in specs for side in spec["groups"]}
    return len(directions) >= 2


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, *GRID_H_RANGE)
    w = _randint(rng, *GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    n_objects = int(rng.choice(np.array(OBJECT_COUNT_CHOICES, dtype=int)))
    specs: list[dict] = []
    placements: list[tuple[int, int]] = []

    for _ in range(n_objects):
        placed = False
        for _attempt in range(PLACEMENT_ATTEMPTS):
            spec = _build_object_spec(rng)
            if spec is None:
                continue

            top_min = 1 + int(spec["pads"]["up"])
            left_min = 1 + int(spec["pads"]["left"])
            top_max = h - 2 - int(spec["h"]) - int(spec["pads"]["down"]) + 1
            left_max = w - 2 - int(spec["w"]) - int(spec["pads"]["right"]) + 1
            if top_min > top_max or left_min > left_max:
                continue

            top = _randint(rng, top_min, top_max)
            left = _randint(rng, left_min, left_max)
            if not _can_place(blocked, spec, top, left):
                continue

            _draw_object(grid, spec, top, left)
            _mark_blocked(blocked, spec, top, left)
            specs.append(spec)
            placements.append((top, left))
            placed = True
            break
        if not placed:
            return None

    if not _scene_has_direction_variety(specs) and rng.random() < 0.7:
        return None

    groups = _extract_groups(grid)
    if max(len(bases) for bases in groups.values()) <= 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_ATTEMPTS} attempts")
