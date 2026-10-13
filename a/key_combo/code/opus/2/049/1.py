from __future__ import annotations

from collections import deque
import random

import numpy as np

BG = 0
RED = 2
GREEN = 3
BLUE = 1
YELLOW = 4
GREY = 5
ORANGE = 7


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r0 in range(h):
        for c0 in range(w):
            if not bool(mask[r0, c0]) or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _component_to_set(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in coords}


def _choose_fold_dir(hinge: tuple[int, int], free_dir: tuple[int, int], shape: tuple[int, int], free_len: int) -> tuple[int, int] | None:
    h, w = shape
    hr, hc = hinge
    # For clockwise mapping from each free direction.
    cw = {(0, 1): (1, 0), (0, -1): (-1, 0), (1, 0): (0, -1), (-1, 0): (0, 1)}
    if free_dir[0] == 0:
        # free segment is horizontal, fold is vertical.
        up = int(hr)
        down = int(h - 1 - hr)
        if up == 0 and down == 0:
            return None
        if up > down:
            chosen = (-1, 0)
            if up < free_len:
                return None
        elif down > up:
            chosen = (1, 0)
            if down < free_len:
                return None
        else:
            chosen = cw[free_dir]
    else:
        # free segment is vertical, fold is horizontal.
        left = int(hc)
        right = int(w - 1 - hc)
        if left == 0 and right == 0:
            return None
        if left > right:
            chosen = (0, -1)
            if left < free_len:
                return None
        elif right > left:
            chosen = (0, 1)
            if right < free_len:
                return None
        else:
            chosen = cw[free_dir]
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each rod at its hinge and recolor orange regions by folded-segment adjacency.

    Input structure:
    - Black background.
    - 2~3 straight 4-connected rods.
    - Each rod has one gray(5) fixed-end marker, one yellow(4) hinge marker, and body pixels of one unique color (red/green/blue).
    - Orange(7) regions are separate from rods.

    Rule:
    - For each rod, keep the gray-to-yellow segment fixed and fold the opposite tail 90 degrees around the yellow marker.
    - Folding side is chosen from the two perpendicular directions with more space from the yellow marker to the border; if tied, use clockwise direction.
    - Remove each original free segment and draw its folded version.
    - An orange component recolors to the touching folded-body color: one touching rod wins that rod color; multiple touching rods choose the longer folded tail.

    Key insight:
    - The fold is geometry-dependent and can create new adjacencies, so recoloring must be computed after folding geometry is applied.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    owner = np.zeros((h, w), dtype=int)
    rod_infos: list[dict[str, int]] = []

    rods = _components(g != BG)
    for comp in rods:
        comp_set = _component_to_set(comp)
        colors = {int(g[r, c]) for r, c in comp_set}
        if ORANGE in colors:
            continue
        if len(comp_set) < 5 or len(comp_set) > 7:
            continue
        if GRAY := GREY not in colors or YELLOW not in colors:
            continue

        gray_cells = [p for p in comp_set if int(g[p]) == GREY]
        yellow_cells = [p for p in comp_set if int(g[p]) == YELLOW]
        if len(gray_cells) != 1 or len(yellow_cells) != 1:
            continue
        gray = gray_cells[0]
        yellow = yellow_cells[0]

        body_cells = comp_set - {gray, yellow}
        if not body_cells:
            continue
        body_colors = {int(g[r, c]) for r, c in body_cells}
        if len(body_colors) != 1:
            continue
        body_color = int(body_colors.pop())
        if body_color == ORANGE:
            continue

        rows = [r for r, _ in comp_set]
        cols = [c for _, c in comp_set]
        horizontal = len(set(rows)) == 1
        vertical = len(set(cols)) == 1
        if not (horizontal or vertical):
            continue

        ordered = sorted(comp_set, key=lambda p: p[1] if horizontal else p[0])
        try:
            gray_idx = ordered.index(gray)
            yellow_idx = ordered.index(yellow)
        except ValueError:
            continue
        if gray_idx == yellow_idx:
            continue

        min_idx = min(gray_idx, yellow_idx)
        max_idx = max(gray_idx, yellow_idx)
        if min_idx + 1 > max_idx:
            continue

        fixed_len = max_idx - min_idx + 1
        if fixed_len >= len(ordered):
            continue

        # Free segment is the body outside the gray-yelllow interval.
        if gray_idx < yellow_idx:
            free_idx = list(range(yellow_idx + 1, len(ordered)))
            free_dir = (0, 1) if horizontal else (1, 0)
        else:
            free_idx = list(range(0, yellow_idx))
            free_dir = (0, -1) if horizontal else (-1, 0)
        if not free_idx:
            continue
        free_len = len(free_idx)
        if free_len > 4:
            continue

        free_cells = [ordered[i] for i in free_idx]
        fold_dir = _choose_fold_dir(yellow, free_dir, (h, w), free_len)
        if fold_dir is None:
            continue

        folded_cells: list[tuple[int, int]] = []
        valid = True
        hr, hc = yellow
        for step in range(1, free_len + 1):
            nr, nc = hr + fold_dir[0] * step, hc + fold_dir[1] * step
            if not _in_bounds((h, w), nr, nc):
                valid = False
                break
            if out[nr, nc] != BG:
                valid = False
                break
            folded_cells.append((nr, nc))
        if not valid:
            continue

        # Clear original free segment.
        for rr, cc in free_cells:
            out[rr, cc] = BG

        # Draw folded cells.
        for rr, cc in folded_cells:
            out[rr, cc] = body_color

        for rr, cc in folded_cells:
            owner[rr, cc] = len(rod_infos) + 1
        rod_infos.append({
            "body_color": int(body_color),
            "free_len": int(free_len),
        })

    # Recolor orange components according to folded ownership.
    orange_components = _components(g == ORANGE)
    for comp in orange_components:
        touching: set[int] = set()
        comp_set = _component_to_set(comp)
        for r, c in comp_set:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds((h, w), nr, nc):
                    continue
                rid = int(owner[nr, nc])
                if rid > 0:
                    touching.add(rid)

        if not touching:
            continue
        if len(touching) == 1:
            chosen = rod_infos[next(iter(touching)) - 1]
            color = int(chosen["body_color"])
        else:
            chosen = None
            best_len = -1
            for rid in touching:
                info = rod_infos[rid - 1]
                if info["free_len"] > best_len:
                    best_len = int(info["free_len"])
                    chosen = info
            color = int(chosen["body_color"]) if chosen is not None else None
            if color is None:
                continue
        for rr, cc in comp_set:
            out[rr, cc] = color

    return out


def _can_place(coords: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
    return all((r, c) not in used for r, c in coords)


def _add_margin(used: set[tuple[int, int]], coords: list[tuple[int, int]], shape: tuple[int, int]) -> None:
    h, w = shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    used.add((rr, cc))


def _pick_segment(h: int, w: int, used_colors: set[int], used_cells: set[tuple[int, int]], colors: list[int]) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    body_color = random.choice(colors)
    if body_color in used_colors:
        return None

    for _ in range(400):
        length = random.randint(5, 7)
        y_pos = random.randint(1, length - 2)
        horiz = random.choice([True, False])
        if horiz:
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - length - 1)
            coords = [(r, c0 + i) for i in range(length)]
        else:
            c = random.randint(1, w - 2)
            r0 = random.randint(1, h - length - 1)
            coords = [(r0 + i, c) for i in range(length)]

        if not _can_place(coords, used_cells):
            continue

        gray = coords[0]
        yellow = coords[y_pos]
        free_cells = (
            coords[y_pos + 1 :]
            if y_pos == coords.index(gray)
            else coords[:y_pos]
        )
        # ensure yellow is interior in non-empty free segment
        if len(free_cells) < 1 or len(free_cells) > 4:
            continue

        free_dir = (0, 1) if (horiz and y_pos < length - 1) else (0, -1)
        if not horiz:
            free_dir = (1, 0) if y_pos < length - 1 else (-1, 0)

        fold_dir = _choose_fold_dir(yellow, free_dir, (h, w), len(free_cells))
        if fold_dir is None:
            continue

        hr, hc = yellow
        ok = True
        for step in range(1, len(free_cells) + 1):
            fr, fc = hr + fold_dir[0] * step, hc + fold_dir[1] * step
            if not _in_bounds((h, w), fr, fc) or (fr, fc) in used_cells:
                ok = False
                break
        if not ok:
            continue

        return coords, gray, yellow

    return None


def _random_blob(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    min_size: int,
    max_size: int,
) -> list[tuple[int, int]] | None:
    for _ in range(240):
        cells = []
        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if (r, c) not in occupied
        ]
        if not candidates:
            return None
        start = random.choice(candidates)
        cells.append(start)
        frontier = {start}

        target = random.randint(min_size, max_size)
        while len(cells) < target:
            adj = []
            for rr, cc in list(frontier):
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 < nr < h - 1
                        and 0 < nc < w - 1
                        and (nr, nc) not in occupied
                        and (nr, nc) not in cells
                    ):
                        adj.append((nr, nc))
            if not adj:
                break
            nxt = random.choice(adj)
            cells.append(nxt)
            frontier.add(nxt)

        if len(cells) >= min_size:
            return cells
    return None


def _simulate_touch_counts(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)
    owner = np.zeros((h, w), dtype=int)
    rod_infos: list[dict[str, int]] = []

    for comp in _components(grid != BG):
        comp_set = _component_to_set(comp)
        colors = {int(grid[r, c]) for r, c in comp_set}
        if ORANGE in colors or not {GREY, YELLOW} <= colors:
            continue
        if len(comp_set) < 5 or len(comp_set) > 7:
            continue

        gray_cells = [p for p in comp_set if int(grid[p]) == GREY]
        yellow_cells = [p for p in comp_set if int(grid[p]) == YELLOW]
        if len(gray_cells) != 1 or len(yellow_cells) != 1:
            continue
        gray = gray_cells[0]
        yellow = yellow_cells[0]
        rows = [r for r, _ in comp_set]
        cols = [c for _, c in comp_set]
        horizontal = len(set(rows)) == 1
        if not (horizontal or len(set(cols)) == 1):
            continue
        ordered = sorted(comp_set, key=lambda p: p[1] if horizontal else p[0])
        try:
            gi = ordered.index(gray)
            yi = ordered.index(yellow)
        except ValueError:
            continue
        if gi < yi:
            free = ordered[yi + 1 :]
            free_dir = (0, 1) if horizontal else (1, 0)
        else:
            free = ordered[:yi]
            free_dir = (0, -1) if horizontal else (-1, 0)
        if len(free) == 0:
            continue
        fd = _choose_fold_dir(yellow, free_dir, (h, w), len(free))
        if fd is None:
            continue
        hr, hc = yellow
        folded = []
        for step in range(1, len(free) + 1):
            rr, cc = hr + fd[0] * step, hc + fd[1] * step
            if not _in_bounds((h, w), rr, cc):
                break
            if owner[rr, cc] != 0 or out[rr, cc] != BG:
                break
            folded.append((rr, cc))
        if len(folded) != len(free):
            continue
        body_color = 0
        for r2, c2 in free:
            out[r2, c2] = BG
        for rr, cc in folded:
            out[rr, cc] = 1
            owner[rr, cc] = len(rod_infos) + 1
        rod_infos.append({"free_len": len(free)})

    counts = []
    for comp in _components(grid == ORANGE):
        comp_set = _component_to_set(comp)
        touching = set()
        for r, c in comp_set:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds((h, w), nr, nc):
                    continue
                if owner[nr, nc]:
                    touching.add(int(owner[nr, nc]))
        counts.append(len(touching))
    return counts


def generate_input() -> np.ndarray:
    for _ in range(1200):
        h = random.randint(12, 24)
        w = random.randint(12, 24)
        n_rods = random.randint(2, 3)

        grid = np.full((h, w), BG, dtype=int)
        rod_colors = random.sample([RED, GREEN, BLUE], n_rods)

        used_colors: set[int] = set()
        used_cells: set[tuple[int, int]] = set()
        placed_rods = 0
        for color in rod_colors:
            info = _pick_segment(h, w, used_colors, used_cells, [color])
            if info is None:
                break
            coords, gray, yellow = info
            if not coords:
                break
            for idx, (r, c) in enumerate(coords):
                if (r, c) == gray:
                    grid[r, c] = GREY
                elif (r, c) == yellow:
                    grid[r, c] = YELLOW
                else:
                    grid[r, c] = color
            used_colors.add(color)
            _add_margin(used_cells, coords, (h, w))
            placed_rods += 1

        if placed_rods != n_rods:
            continue

        orange_count = random.randint(3, 4)
        for _ in range(orange_count):
            blob = _random_blob(h, w, used_cells, 3, 5)
            if blob is None:
                continue
            for rr, cc in blob:
                grid[rr, cc] = ORANGE
            _add_margin(used_cells, blob, (h, w))

        # Ensure meaningful examples and avoid ambiguous recolor branches.
        touch_counts = _simulate_touch_counts(grid)
        if not touch_counts:
            continue
        if max(touch_counts) > 2:
            continue
        if not any(c == 0 for c in touch_counts):
            # keep some unchanged orange areas visible.
            continue
        if not any(c == 1 or c == 2 for c in touch_counts):
            continue
        try:
            _ = main(grid)
        except Exception:
            continue
        return grid

    raise RuntimeError("failed to generate valid puzzle 1 input")
