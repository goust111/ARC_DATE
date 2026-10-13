from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random

import numpy as np


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
MARK = 9

DIR_BY_SIDE = {
    "left": (0, -1),
    "right": (0, 1),
    "top": (-1, 0),
    "bottom": (1, 0),
}

OPPOSITE = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
}


@dataclass
class Container:
    idx: int
    box: list[int]  # [r0, c0, r1, c1]
    interior: np.ndarray
    head_side: str
    head_cells: list[tuple[int, int]]
    dest_cells: list[tuple[int, int]]
    initial_distance: int
    collisions_in: int = 0
    collisions_out: int = 0
    energy_in: int = 0
    energy_out: int = 0

    @property
    def direction(self) -> tuple[int, int]:
        return DIR_BY_SIDE[self.head_side]

    @property
    def grey_count(self) -> int:
        return int(np.count_nonzero(self.interior == GREY))

    @property
    def head_anchor(self) -> tuple[int, int]:
        rows = [r for r, _ in self.head_cells]
        cols = [c for _, c in self.head_cells]
        return min(rows), min(cols)

    @property
    def dest_anchor(self) -> tuple[int, int]:
        rows = [r for r, _ in self.dest_cells]
        cols = [c for _, c in self.dest_cells]
        return min(rows), min(cols)

    @property
    def remaining_distance(self) -> int:
        hr, hc = self.head_anchor
        dr, dc = self.direction
        tr, tc = self.dest_anchor
        if dr == 1:
            return max(0, tr - hr)
        if dr == -1:
            return max(0, hr - tr)
        if dc == 1:
            return max(0, tc - hc)
        return max(0, hc - tc)


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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_coords(box: tuple[int, int, int, int] | list[int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _intersects(a: tuple[int, int, int, int] | list[int], b: tuple[int, int, int, int] | list[int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _normalized_shape(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _component_side(
    box: tuple[int, int, int, int] | list[int], comp: list[tuple[int, int]]
) -> str | None:
    r0, c0, r1, c1 = map(int, box)
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    if all(c == c0 - 1 for c in cols) and min(rows) >= r0 and max(rows) <= r1:
        return "left"
    if all(c == c1 + 1 for c in cols) and min(rows) >= r0 and max(rows) <= r1:
        return "right"
    if all(r == r0 - 1 for r in rows) and min(cols) >= c0 and max(cols) <= c1:
        return "top"
    if all(r == r1 + 1 for r in rows) and min(cols) >= c0 and max(cols) <= c1:
        return "bottom"
    return None


def _head_order_coords(container: Container) -> list[tuple[int, int]]:
    r0, c0, _, _ = map(int, container.box)
    h, w = container.interior.shape
    coords = [(r0 + 1 + rr, c0 + 1 + cc) for rr in range(h) for cc in range(w)]
    if container.head_side == "left":
        coords.sort(key=lambda p: (p[1], p[0]))
    elif container.head_side == "right":
        coords.sort(key=lambda p: (-p[1], p[0]))
    elif container.head_side == "top":
        coords.sort(key=lambda p: (p[0], p[1]))
    else:
        coords.sort(key=lambda p: (-p[0], p[1]))
    return coords


def _consume_one_grey(container: Container) -> bool:
    r0, c0, _, _ = map(int, container.box)
    for ar, ac in _head_order_coords(container):
        rr = ar - r0 - 1
        cc = ac - c0 - 1
        if int(container.interior[rr, cc]) == GREY:
            container.interior[rr, cc] = ORANGE
            return True
    return False


def _exhaust_grey(container: Container) -> int:
    amt = container.grey_count
    if amt:
        container.interior[container.interior == GREY] = ORANGE
    return int(amt)


def _shift_container(container: Container, dr: int, dc: int) -> None:
    container.box = [
        int(container.box[0] + dr),
        int(container.box[1] + dc),
        int(container.box[2] + dr),
        int(container.box[3] + dc),
    ]
    container.head_cells = [(r + dr, c + dc) for r, c in container.head_cells]


def _add_energy(container: Container, side: str, amount: int) -> None:
    if amount <= 0:
        return
    old = container.interior
    h, w = old.shape
    if side in ("left", "right"):
        span = h
        stripes = int(math.ceil(amount / span))
        new = np.full((h, w + stripes), ORANGE, dtype=int)
        if side == "left":
            new[:, stripes:] = old
            container.box[1] -= stripes
            fill_positions = [(rr, cc) for cc in range(stripes) for rr in range(h)]
            if container.head_side == "left":
                container.head_cells = [(r, c - stripes) for r, c in container.head_cells]
        else:
            new[:, :w] = old
            container.box[3] += stripes
            fill_positions = [(rr, cc) for cc in range(w + stripes - 1, w - 1, -1) for rr in range(h)]
            if container.head_side == "right":
                container.head_cells = [(r, c + stripes) for r, c in container.head_cells]
    else:
        span = w
        stripes = int(math.ceil(amount / span))
        new = np.full((h + stripes, w), ORANGE, dtype=int)
        if side == "top":
            new[stripes:, :] = old
            container.box[0] -= stripes
            fill_positions = [(rr, cc) for rr in range(stripes) for cc in range(w)]
            if container.head_side == "top":
                container.head_cells = [(r - stripes, c) for r, c in container.head_cells]
        else:
            new[:h, :] = old
            container.box[2] += stripes
            fill_positions = [(rr, cc) for rr in range(h + stripes - 1, h - 1, -1) for cc in range(w)]
            if container.head_side == "bottom":
                container.head_cells = [(r + stripes, c) for r, c in container.head_cells]
    for rr, cc in fill_positions[:amount]:
        new[rr, cc] = GREY
    container.interior = new
    container.energy_in += int(amount)


def _render(shape: tuple[int, int], containers: list[Container], dests: list[list[tuple[int, int]]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for comp in dests:
        for r, c in comp:
            out[r, c] = MARK
    for container in containers:
        r0, c0, r1, c1 = map(int, container.box)
        for r, c in _border_coords(container.box):
            out[r, c] = BORDER
        out[r0 + 1 : r1, c0 + 1 : c1] = container.interior
        for r, c in container.head_cells:
            out[r, c] = MARK
    return out


def _parse_scene(grid: np.ndarray) -> tuple[list[Container], list[list[tuple[int, int]]], int]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    containers: list[Container] = []
    black_comps = _components(g == BORDER)
    for comp in black_comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(comp) != _border_coords((r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not np.isin(inner, [GREY, ORANGE]).all():
            continue
        containers.append(
            Container(
                idx=len(containers),
                box=[r0, c0, r1, c1],
                interior=inner.copy(),
                head_side="",
                head_cells=[],
                dest_cells=[],
                initial_distance=0,
            )
        )
    if not containers:
        raise ValueError("no containers found")

    mark_comps = _components(g == MARK)
    used_marks: set[int] = set()
    for container in containers:
        matches: list[tuple[int, str]] = []
        for i, comp in enumerate(mark_comps):
            side = _component_side(container.box, comp)
            if side is not None:
                matches.append((i, side))
        if len(matches) != 1:
            raise ValueError("head detection ambiguous")
        i, side = matches[0]
        used_marks.add(i)
        container.head_side = side
        container.head_cells = sorted(mark_comps[i])

    dests: list[list[tuple[int, int]]] = [mark_comps[i] for i in range(len(mark_comps)) if i not in used_marks]
    if len(dests) != len(containers):
        raise ValueError("destination count mismatch")

    for container in containers:
        head_shape = _normalized_shape(container.head_cells)
        hr0, hc0, _, _ = _bbox(container.head_cells)
        dr, dc = container.direction
        best: tuple[int, list[tuple[int, int]]] | None = None
        for comp in dests:
            if _normalized_shape(comp) != head_shape:
                continue
            tr0, tc0, _, _ = _bbox(comp)
            drow = tr0 - hr0
            dcol = tc0 - hc0
            if dr != 0:
                if dcol != 0 or drow * dr <= 0:
                    continue
                dist = abs(drow)
            else:
                if drow != 0 or dcol * dc <= 0:
                    continue
                dist = abs(dcol)
            if best is None or dist < best[0]:
                best = (dist, comp)
        if best is None:
            raise ValueError("destination pairing failed")
        container.dest_cells = sorted(best[1])
        container.initial_distance = int(best[0])

    return containers, dests, bg


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    containers, dests, bg = _parse_scene(grid)
    if bg != BG:
        raise ValueError("unexpected background")
    by_id = {c.idx: c for c in containers}
    stopped: set[int] = set()
    active_stack: set[int] = set()
    collision_count = 0

    def move_container(idx: int) -> None:
        nonlocal collision_count
        container = by_id[idx]
        if idx in active_stack:
            return
        active_stack.add(idx)
        stopped.discard(idx)
        while container.grey_count > 0 and container.remaining_distance > 0:
            dr, dc = container.direction
            shifted = [
                int(container.box[0] + dr),
                int(container.box[1] + dc),
                int(container.box[2] + dr),
                int(container.box[3] + dc),
            ]
            blocker: Container | None = None
            for other_idx in sorted(stopped):
                other = by_id[other_idx]
                if _intersects(shifted, other.box):
                    blocker = other
                    break
            if blocker is not None:
                collision_count += 1
                container.collisions_out += 1
                blocker.collisions_in += 1
                amount = _exhaust_grey(container)
                container.energy_out += int(amount)
                if amount > 0:
                    transfer_side = "bottom" if dr == 1 else "top" if dr == -1 else "right" if dc == 1 else "left"
                    _add_energy(blocker, transfer_side, amount)
                stopped.add(idx)
                active_stack.remove(idx)
                if blocker.grey_count > 0 and blocker.remaining_distance > 0:
                    move_container(blocker.idx)
                return
            _shift_container(container, dr, dc)
            if not _consume_one_grey(container):
                raise ValueError("grey accounting failed")
        stopped.add(idx)
        active_stack.remove(idx)

    order = sorted(
        containers,
        key=lambda c: (
            int(c.initial_distance),
            int(c.head_anchor[0]),
            int(c.head_anchor[1]),
        ),
    )
    for container in order:
        move_container(container.idx)

    out = _render(grid.shape, containers, dests)
    stats = {
        "collision_count": int(collision_count),
        "container_count": int(len(containers)),
        "remaining_greys": [c.grey_count for c in containers],
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward their maroon targets, with collisions transferring remaining charge.

    Input structure:
    - The background is magenta (6).
    - Three to four black-bordered rectangular containers contain only grey (5) and orange (7) interior cells.
    - Each container has a small maroon (9) head sprite attached to one side and a matching maroon destination further ahead on the same axis.

    Rule:
    - Process containers by increasing head-to-destination distance, breaking ties by the head row.
    - A container moves one cell per step along the head-to-destination direction; each step turns the grey interior cell nearest the head into orange.
    - Its initial movement budget is the smaller of its remaining grey count and its remaining distance to destination.
    - If the next step would overlap an already stopped container, the mover stops immediately, all of its remaining grey cells turn orange, and that remaining grey count is transferred to the blocker.
    - The blocker receives that many new grey cells on the side opposite the collision face, expanding its border as needed; if it still has distance left, it resumes moving under the same rules.

    Key insight:
    Grey cells are both fuel and transferable resources, so collisions do not just halt motion: they can enlarge a previously stopped container and let it continue farther toward its own destination.
    """
    out, _stats = _simulate(np.asarray(grid, dtype=int))
    return out


def _interior_template(inner_h: int, inner_w: int, side: str, grey_count: int) -> np.ndarray:
    arr = np.full((inner_h, inner_w), ORANGE, dtype=int)
    coords = [(rr, cc) for rr in range(inner_h) for cc in range(inner_w)]
    if side == "left":
        coords.sort(key=lambda p: (p[1], p[0]))
    elif side == "right":
        coords.sort(key=lambda p: (-p[1], p[0]))
    elif side == "top":
        coords.sort(key=lambda p: (p[0], p[1]))
    else:
        coords.sort(key=lambda p: (-p[0], p[1]))
    for rr, cc in coords[:grey_count]:
        arr[rr, cc] = GREY
    return arr


def _head_cells_for_box(box: tuple[int, int, int, int], side: str, length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side in ("left", "right"):
        start = r0 + (h - length) // 2
        col = c0 - 1 if side == "left" else c1 + 1
        return [(start + i, col) for i in range(length)]
    start = c0 + (w - length) // 2
    row = r0 - 1 if side == "top" else r1 + 1
    return [(row, start + i) for i in range(length)]


def _translate_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in cells]


def _stamp_scene(
    shape: tuple[int, int],
    specs: list[dict],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        grid[r0:r1 + 1, c0:c1 + 1] = BG
        for r, c in _border_coords(spec["box"]):
            grid[r, c] = BORDER
        grid[r0 + 1 : r1, c0 + 1 : c1] = _interior_template(
            r1 - r0 - 1,
            c1 - c0 - 1,
            spec["side"],
            spec["grey"],
        )
        head = _head_cells_for_box(spec["box"], spec["side"], spec["head_len"])
        dr, dc = DIR_BY_SIDE[spec["side"]]
        dest = _translate_cells(head, dr * spec["dist"], dc * spec["dist"])
        for r, c in head + dest:
            grid[r, c] = MARK
    return grid


def _all_object_cells(specs: list[dict]) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                pts.add((r, c))
        head = _head_cells_for_box(spec["box"], spec["side"], spec["head_len"])
        dr, dc = DIR_BY_SIDE[spec["side"]]
        dest = _translate_cells(head, dr * spec["dist"], dc * spec["dist"])
        pts.update(head)
        pts.update(dest)
    return pts


def _build_canonical_scene() -> np.ndarray | None:
    h_a = random.randint(4, 5)
    w_a = random.randint(5, 7)
    h_b = random.randint(5, 7)
    w_b = random.randint(4, 5)
    head_a = random.randint(1, min(2, h_a))
    head_b = random.randint(1, min(2, w_b))
    grey_a = random.randint(2, min(4, (h_a - 2) * (w_a - 2) - 1))
    gap_to_cross = random.randint(1, 4)
    dist_a = grey_a + gap_to_cross
    steps_before_hit = random.randint(1, 3)
    max_r = min(4, 6 - steps_before_hit)
    if max_r < 1:
        return None
    reach_after_transfer = random.random() < 0.5
    if reach_after_transfer:
        if gap_to_cross > max_r:
            return None
        r_transfer = random.randint(gap_to_cross, max_r)
    else:
        if gap_to_cross == 1:
            return None
        r_transfer = random.randint(1, min(gap_to_cross - 1, max_r))
    grey_b = steps_before_hit + r_transfer
    if grey_b > 6 or grey_b >= (h_b - 2) * (w_b - 2):
        return None
    dist_b = max(dist_a + random.randint(1, 3), grey_b + random.randint(0, 2))

    top_margin = random.randint(2, 4)
    left_margin = random.randint(2, 4)
    box_a = (
        top_margin + h_b + steps_before_hit,
        left_margin,
        top_margin + h_b + steps_before_hit + h_a - 1,
        left_margin + w_a - 1,
    )
    final_a_c0 = box_a[1] + grey_a
    overlap_col = final_a_c0 + random.randint(1, max(1, w_a - 2))
    box_b_c0 = max(1, overlap_col - w_b // 2)
    box_b = (
        top_margin,
        box_b_c0,
        top_margin + h_b - 1,
        box_b_c0 + w_b - 1,
    )
    if box_b[2] + steps_before_hit != box_a[0] - 1:
        return None
    if box_b[1] > box_a[3] + grey_a or box_b[3] < box_a[1] + grey_a:
        return None

    specs = [
        {"box": box_a, "side": "right", "head_len": head_a, "grey": grey_a, "dist": dist_a},
        {"box": box_b, "side": "bottom", "head_len": head_b, "grey": grey_b, "dist": dist_b},
    ]

    h_c = random.randint(4, 5)
    w_c = random.randint(5, 7)
    head_c = random.randint(1, min(2, h_c))
    grey_c = random.randint(2, min(6, (h_c - 2) * (w_c - 2)))
    dist_c = random.randint(2, min(7, grey_c + 2))
    gap_c = random.randint(4, 6)
    c_c0 = random.randint(1, 4)
    box_c = (
        box_a[2] + gap_c,
        c_c0,
        box_a[2] + gap_c + h_c - 1,
        c_c0 + w_c - 1,
    )
    specs.append({"box": box_c, "side": "right", "head_len": head_c, "grey": grey_c, "dist": dist_c})

    if random.random() < 0.55:
        h_d = random.randint(5, 7)
        w_d = random.randint(4, 5)
        head_d = random.randint(1, min(2, w_d))
        grey_d = random.randint(2, min(6, (h_d - 2) * (w_d - 2)))
        dist_d = random.randint(2, min(7, grey_d + 2))
        d_col = max(box_a[3], box_b[3]) + random.randint(5, 8)
        d_row = random.randint(1, 4)
        box_d = (d_row, d_col, d_row + h_d - 1, d_col + w_d - 1)
        specs.append({"box": box_d, "side": "bottom", "head_len": head_d, "grey": grey_d, "dist": dist_d})

    cells = _all_object_cells(specs)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_r_obj = max(r for r, _ in cells)
    max_c_obj = max(c for _, c in cells)
    pad_top = random.randint(0, 2)
    pad_left = random.randint(0, 2)
    dr = pad_top - min_r
    dc = pad_left - min_c
    shifted_specs: list[dict] = []
    for spec in specs:
        r0, c0, r1, c1 = spec["box"]
        shifted_specs.append(
            {
                **spec,
                "box": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
            }
        )
    local_h = max_r_obj + dr + random.randint(3, 5)
    local_w = max_c_obj + dc + random.randint(3, 5)
    if local_h > 30 or local_w > 30:
        return None
    grid = _stamp_scene((local_h, local_w), shifted_specs)

    if random.random() < 0.5:
        grid = np.flipud(grid)
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    if random.random() < 0.5:
        grid = grid.T.copy()

    extra_h = random.randint(0, max(0, 30 - grid.shape[0]))
    extra_w = random.randint(0, max(0, 30 - grid.shape[1]))
    final_h = grid.shape[0] + min(extra_h, random.randint(0, 4))
    final_w = grid.shape[1] + min(extra_w, random.randint(0, 4))
    canvas = np.full((final_h, final_w), BG, dtype=int)
    off_r = random.randint(0, final_h - grid.shape[0])
    off_c = random.randint(0, final_w - grid.shape[1])
    canvas[off_r : off_r + grid.shape[0], off_c : off_c + grid.shape[1]] = grid
    return canvas


def _gen_attempt() -> np.ndarray | None:
    grid = _build_canonical_scene()
    if grid is None:
        return None
    try:
        out, stats = _simulate(grid)
    except Exception:
        return None
    if stats["container_count"] not in {3, 4}:
        return None
    if stats["collision_count"] < 1:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
