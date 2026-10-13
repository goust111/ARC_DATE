from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 2
FILL = 3
TIE = 4
MARKER = 5
CONFLICT = 6
OBSTACLE = 7

RNG = random.Random()


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_containers(grid: np.ndarray) -> list[dict[str, int]]:
    containers: list[dict[str, int]] = []
    for comp in _color_components(grid, WALL):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        if right - left < 2 or bottom - top < 1:
            continue

        expected: set[tuple[int, int]] = set()
        for r in range(top, bottom + 1):
            expected.add((r, left))
            expected.add((r, right))
        for c in range(left, right + 1):
            expected.add((bottom, c))

        if set(comp) != expected:
            continue
        if np.any(grid[top:bottom, left + 1 : right] != BG):
            continue

        containers.append(
            {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "area": (bottom - top) * (right - left - 1),
            }
        )
    containers.sort(key=lambda x: (x["top"], x["left"]))
    return containers


def _fill_container(out: np.ndarray, container: dict[str, int]) -> None:
    out[container["top"] : container["bottom"], container["left"] + 1 : container["right"]] = FILL


def _interval_fill(out: np.ndarray, row: int, start: int, end: int, color: int) -> None:
    if start <= end:
        out[row, start : end + 1] = color


def _apply_rays(out: np.ndarray, activated: list[dict[str, int]]) -> np.ndarray:
    static = out.copy()
    h, w = out.shape
    right_emitters: dict[tuple[int, int], int] = {}
    left_emitters: dict[tuple[int, int], int] = {}
    for container in activated:
        row = container["top"]
        right_emitters[(row, container["right"])] = container["area"]
        left_emitters[(row, container["left"])] = container["area"]

    for r in range(h):
        c = 0
        while c < w:
            if static[r, c] != BG:
                c += 1
                continue
            start = c
            while c + 1 < w and static[r, c + 1] == BG:
                c += 1
            end = c
            left_area = right_emitters.get((r, start - 1))
            right_area = left_emitters.get((r, end + 1))
            if left_area is not None or right_area is not None:
                _interval_fill(out, r, start, end, FILL)
                if left_area is not None and right_area is not None and (end - start + 1) % 2 == 1:
                    mid = (start + end) // 2
                    if left_area == right_area:
                        out[r, mid] = TIE
                    else:
                        out[r, mid] = FILL
                        if r - 1 >= 0 and out[r - 1, mid] == BG:
                            out[r - 1, mid] = CONFLICT
            c += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Activate top-hit U containers, then emit horizontal rays from each activated opening.

    Input structure:
    - The grid is mostly black background with red open-up U-shaped containers,
      grey marker pixels on the top row, and orange horizontal bars as obstacles.
    - Each valid container is a one-pixel-thick red U: two vertical walls and one
      solid bottom edge, with a black rectangular interior.

    Rule:
    - Remove every top-row grey marker. If its column drops cleanly into exactly
      one container footprint, fill that container interior green; otherwise color
      the bottom cell of that column green.
    - Every activated container emits one horizontal ray leftward from its left
      opening side and one rightward from its right opening side along the
      container's top row. Rays extend through background only and stop before
      the next non-background cell.
    - When opposite rays face each other across the same empty gap, both sides
      fill the gap. If the gap length is odd, the center cell is the conflict:
      it becomes yellow when the two container interior areas tie, otherwise it
      stays green and the cell directly above it becomes magenta.

    Key insight:
    - The marker drop decides which containers become active, and the later ray
      behavior can be solved row-by-row by looking at maximal background gaps
      between static obstacles and activated container mouths.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers = _detect_containers(g)
    activated_ids: set[int] = set()

    for c in np.where(g[0] == MARKER)[0]:
        out[0, c] = BG
        candidates = [
            idx
            for idx, container in enumerate(containers)
            if container["left"] <= c <= container["right"]
            and np.all(g[1 : container["top"], c] == BG)
        ]
        if candidates:
            idx = min(candidates, key=lambda k: containers[k]["top"])
            activated_ids.add(idx)
            _fill_container(out, containers[idx])
        else:
            out[-1, c] = FILL

    activated = [containers[idx] for idx in sorted(activated_ids)]
    return _apply_rays(out, activated)


def _draw_container(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> dict[str, int]:
    bottom = top + inner_h
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[bottom, left : right + 1] = WALL
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "area": inner_h * inner_w,
    }


def _box_clear(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = grid.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == BG)


def _footprints_overlap(containers: list[dict[str, int]], left: int, right: int) -> bool:
    return any(not (right < c["left"] or left > c["right"]) for c in containers)


def _place_random_container(
    grid: np.ndarray,
    containers: list[dict[str, int]],
    *,
    preferred_top: int | None = None,
    inner_h: int | None = None,
    inner_w: int | None = None,
) -> dict[str, int] | None:
    h, w = grid.shape
    tops = [preferred_top] if preferred_top is not None else list(range(4, h - 2))
    RNG.shuffle(tops)
    for _ in range(80):
        ih = inner_h if inner_h is not None else RNG.randint(1, 4)
        iw = inner_w if inner_w is not None else RNG.randint(2, 5)
        total_w = iw + 2
        total_h = ih + 1
        for top in tops:
            if top + total_h >= h - 1:
                continue
            cols = list(range(1, w - total_w))
            RNG.shuffle(cols)
            for left in cols:
                right = left + total_w - 1
                bottom = top + total_h - 1
                if _footprints_overlap(containers, left, right):
                    continue
                if not _box_clear(grid, top, left, bottom, right, pad=1):
                    continue
                container = _draw_container(grid, top, left, ih, iw)
                containers.append(container)
                return container
    return None


def _place_bar(
    grid: np.ndarray,
    marker_cols: set[int],
    *,
    row: int | None = None,
    c0: int | None = None,
    length: int | None = None,
    forbid: set[tuple[int, int]] | None = None,
) -> bool:
    h, w = grid.shape
    rows = [row] if row is not None else list(range(1, h - 1))
    RNG.shuffle(rows)
    lengths = [length] if length is not None else [2, 3, 4]
    if length is None:
        RNG.shuffle(lengths)
    for rr in rows:
        for ll in lengths:
            starts = [c0] if c0 is not None else list(range(1, w - ll))
            RNG.shuffle(starts)
            for cc in starts:
                cells = [(rr, x) for x in range(cc, cc + ll)]
                if any(x in marker_cols for _, x in cells):
                    continue
                if forbid is not None and any(cell in forbid for cell in cells):
                    continue
                if not _box_clear(grid, rr, cc, rr, cc + ll - 1, pad=1):
                    continue
                grid[rr, cc : cc + ll] = OBSTACLE
                return True
    return False


def _available_marker_col(containers: list[dict[str, int]], target: dict[str, int]) -> int | None:
    cols = list(range(target["left"], target["right"] + 1))
    RNG.shuffle(cols)
    for c in cols:
        hits = [container for container in containers if container["left"] <= c <= container["right"]]
        if hits == [target]:
            return c
    return None


def _miss_marker_col(containers: list[dict[str, int]], width: int) -> int | None:
    cols = list(range(width))
    RNG.shuffle(cols)
    for c in cols:
        if all(not (container["left"] <= c <= container["right"]) for container in containers):
            return c
    return None


def _choose_scenario() -> str:
    roll = RNG.random()
    if roll < 0.35:
        return "no_conflict"
    if roll < 0.7:
        return "conflict_unequal"
    return "conflict_tie"


def _build_conflict_input(tie: bool) -> np.ndarray | None:
    h = RNG.randint(18, 26)
    w = RNG.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    containers: list[dict[str, int]] = []

    top = RNG.randint(5, min(10, h - 7))
    if tie:
        ih1 = ih2 = RNG.randint(1, 4)
        iw1 = iw2 = RNG.randint(2, 5)
    else:
        ih1, iw1 = RNG.randint(1, 4), RNG.randint(2, 5)
        ih2, iw2 = RNG.randint(1, 4), RNG.randint(2, 5)
        if ih1 * iw1 == ih2 * iw2:
            iw2 = 2 + ((iw2 - 1) % 4)
            if ih1 * iw1 == ih2 * iw2:
                ih2 = 1 + (ih2 % 4)
        if ih1 * iw1 == ih2 * iw2:
            return None

    gap = RNG.choice([3, 5, 7])
    width1 = iw1 + 2
    width2 = iw2 + 2
    needed = width1 + width2 + gap + 4
    if needed >= w:
        return None
    left1 = RNG.randint(1, w - needed)
    left2 = left1 + width1 + gap
    c1 = _draw_container(grid, top, left1, ih1, iw1)
    c2 = _draw_container(grid, top, left2, ih2, iw2)
    containers.extend([c1, c2])

    total = RNG.randint(3, 4)
    while len(containers) < total:
        extra = _place_random_container(grid, containers)
        if extra is None:
            return None

    activated = [c1, c2]
    marker_cols: set[int] = set()
    for container in activated:
        col = _available_marker_col(containers, container)
        if col is None:
            return None
        grid[0, col] = MARKER
        marker_cols.add(col)

    if RNG.random() < 0.35:
        miss = _miss_marker_col(containers, w)
        if miss is not None:
            grid[0, miss] = MARKER
            marker_cols.add(miss)

    forbid = {(top, c) for c in range(c1["right"] + 1, c2["left"])}
    outer_target = RNG.choice([(c1, "left"), (c2, "right")])
    target, side = outer_target
    length = RNG.choice([2, 3, 4])
    if side == "left":
        max_start = target["left"] - length - 1
        min_start = max(1, target["left"] - length - 4)
        if min_start <= max_start:
            start = RNG.randint(min_start, max_start)
            _place_bar(grid, marker_cols, row=top, c0=start, length=length, forbid=forbid)
    else:
        min_start = target["right"] + 2
        max_start = min(w - length - 1, target["right"] + 5)
        if min_start <= max_start:
            start = RNG.randint(min_start, max_start)
            _place_bar(grid, marker_cols, row=top, c0=start, length=length, forbid=forbid)

    extra_bars = RNG.randint(0, 2)
    placed = 0
    while placed < extra_bars:
        if not _place_bar(grid, marker_cols, forbid=forbid):
            break
        placed += 1
    if not np.any(grid == OBSTACLE):
        _place_bar(grid, marker_cols, forbid=forbid)
    return grid


def _build_no_conflict_input() -> np.ndarray | None:
    h = RNG.randint(18, 26)
    w = RNG.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    containers: list[dict[str, int]] = []
    total = RNG.randint(2, 4)

    preferred_tops = list(range(4, h - 4, 2))
    RNG.shuffle(preferred_tops)
    while len(containers) < total:
        top = preferred_tops[len(containers)] if len(containers) < len(preferred_tops) else None
        container = _place_random_container(grid, containers, preferred_top=top)
        if container is None:
            return None

    activated_count = RNG.randint(1, total - 1)
    containers_by_top = sorted(containers, key=lambda c: c["top"])
    activated = containers_by_top[:activated_count]
    marker_cols: set[int] = set()
    for container in activated:
        col = _available_marker_col(containers, container)
        if col is None:
            return None
        grid[0, col] = MARKER
        marker_cols.add(col)

    if RNG.random() < 0.45:
        miss = _miss_marker_col(containers, w)
        if miss is not None:
            grid[0, miss] = MARKER
            marker_cols.add(miss)

    anchor = RNG.choice(activated)
    side = RNG.choice(["left", "right"])
    length = RNG.choice([2, 3, 4])
    if side == "left":
        max_start = anchor["left"] - length - 1
        min_start = max(1, anchor["left"] - length - 4)
        if min_start <= max_start:
            start = RNG.randint(min_start, max_start)
            _place_bar(grid, marker_cols, row=anchor["top"], c0=start, length=length)
    else:
        min_start = anchor["right"] + 2
        max_start = min(w - length - 1, anchor["right"] + 5)
        if min_start <= max_start:
            start = RNG.randint(min_start, max_start)
            _place_bar(grid, marker_cols, row=anchor["top"], c0=start, length=length)

    extra_bars = RNG.randint(0, 2)
    for _ in range(extra_bars):
        _place_bar(grid, marker_cols)
    if not np.any(grid == OBSTACLE):
        _place_bar(grid, marker_cols)
    return grid


def _build_input_for_scenario(scenario: str) -> np.ndarray | None:
    if scenario == "conflict_tie":
        return _build_conflict_input(tie=True)
    if scenario == "conflict_unequal":
        return _build_conflict_input(tie=False)
    return _build_no_conflict_input()


def generate_input() -> np.ndarray:
    for _ in range(400):
        scenario = _choose_scenario()
        grid = _build_input_for_scenario(scenario)
        if grid is None:
            continue
        out = main(grid)
        if len(_detect_containers(grid)) < 2:
            continue
        if not np.any(grid[0] == MARKER):
            continue
        if scenario == "conflict_tie" and not np.any(out == TIE):
            continue
        if scenario == "conflict_unequal" and not np.any(out == CONFLICT):
            continue
        if scenario == "no_conflict" and (np.any(out == TIE) or np.any(out == CONFLICT)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
