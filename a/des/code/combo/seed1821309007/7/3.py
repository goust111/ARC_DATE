from __future__ import annotations

import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_emitter(coords: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    cells = set(coords)
    for dr in (0, 1):
        for dc in (0, 1):
            if (r0 + dr, c0 + dc) not in cells:
                return (-1 if dr == 0 else 1, -1 if dc == 0 else 1)
    return None


def _classify_target(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = len(coords)
    cells = set(coords)
    if area == h * w:
        return ("solid", (r0, c0, r1, c1))
    if h >= 3 and w >= 3:
        border = set()
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        if cells == border:
            return ("frame", (r0, c0, r1, c1))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit diagonal color rays from 2x2 L-shape launchers.

    Input structure:
    - The background is black/dominant.
    - Launchers are monochrome 3-cell L shapes inside a 2x2 box.
    - Targets are monochrome solid rectangles or hollow rectangular frames.

    Rule:
    - The missing launcher corner defines the diagonal ray direction.
    - The ray starts from the missing corner cell and paints background cells until
      it hits a target, another launcher, or the grid edge.
    - Solid targets are fully repainted and stop the ray.
    - Hollow-frame targets have only their border repainted; the ray tunnels
      through and continues beyond the frame without painting the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    objects: list[dict[str, object]] = []
    emitters: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []

    for color in [int(c) for c in np.unique(g) if int(c) != bg]:
        for coords in _components(g == color):
            emitter_dir = _classify_emitter(coords)
            if emitter_dir is not None:
                r0, c0, _r1, _c1 = _bbox(coords)
                cells = set(coords)
                missing = None
                for dr in (0, 1):
                    for dc in (0, 1):
                        p = (r0 + dr, c0 + dc)
                        if p not in cells:
                            missing = p
                            break
                    if missing is not None:
                        break
                assert missing is not None
                emitters.append(
                    {
                        "color": color,
                        "coords": cells,
                        "dir": emitter_dir,
                        "start": missing,
                    }
                )
                continue

            target_info = _classify_target(coords)
            if target_info is not None:
                kind, bbox = target_info
                targets.append(
                    {
                        "color": color,
                        "coords": set(coords),
                        "kind": kind,
                        "bbox": bbox,
                    }
                )
                continue

            objects.append({"color": color, "coords": set(coords)})

    assert emitters, "expected at least one launcher"
    assert targets, "expected at least one target"
    assert not objects, "all foreground objects must be launchers or targets"

    target_at: dict[tuple[int, int], int] = {}
    for i, target in enumerate(targets):
        for p in target["coords"]:
            assert p not in target_at, "targets must not overlap"
            target_at[p] = i

    emitter_cells = set()
    for emitter in emitters:
        for p in emitter["coords"]:
            emitter_cells.add(p)

    out = g.copy()
    for emitter in sorted(emitters, key=lambda item: item["start"]):
        color = int(emitter["color"])
        dr, dc = emitter["dir"]
        r, c = emitter["start"]
        while 0 <= r < h and 0 <= c < w:
            if (r, c) in emitter_cells:
                break
            target_i = target_at.get((r, c))
            if target_i is not None:
                target = targets[target_i]
                for tr, tc in target["coords"]:
                    out[tr, tc] = color
                if target["kind"] == "solid":
                    break
                r += dr
                c += dc
                r0, c0, r1, c1 = target["bbox"]
                while r0 <= r <= r1 and c0 <= c <= c1:
                    r += dr
                    c += dc
                continue

            if int(g[r, c]) == bg:
                out[r, c] = color
                r += dr
                c += dc
                continue

            break

    return out


DIR_TO_MISSING = {
    (-1, -1): (0, 0),
    (-1, 1): (0, 1),
    (1, -1): (1, 0),
    (1, 1): (1, 1),
}

PATTERNS = (
    ("solid",),
    ("frame", "solid"),
    ("frame", "frame", "solid"),
    ("frame",),
)

GEN_RETRIES = 300


def _ray_steps_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    steps = 0
    while 0 <= r < h and 0 <= c < w:
        steps += 1
        r += dr
        c += dc
    return steps


def _square_bbox(entry: tuple[int, int], direction: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    r, c = entry
    dr, dc = direction
    if (dr, dc) == (1, 1):
        return r, c, r + size - 1, c + size - 1
    if (dr, dc) == (1, -1):
        return r, c - size + 1, r + size - 1, c
    if (dr, dc) == (-1, 1):
        return r - size + 1, c, r, c + size - 1
    assert (dr, dc) == (-1, -1)
    return r - size + 1, c - size + 1, r, c


def _in_bounds_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _paint_emitter(grid: np.ndarray, start: tuple[int, int], direction: tuple[int, int], color: int) -> set[tuple[int, int]]:
    r, c = start
    mr, mc = DIR_TO_MISSING[direction]
    r0 = r - mr
    c0 = c - mc
    cells = {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}
    cells.remove(start)
    for pr, pc in cells:
        grid[pr, pc] = color
    return cells


def _paint_target(grid: np.ndarray, bbox: tuple[int, int, int, int], kind: str, color: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if kind == "solid":
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((r, c))
        return cells

    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            grid[r, c] = color
            cells.add((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            grid[r, c] = color
            cells.add((r, c))
    return cells


def _forbidden_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    intervals: list[tuple[int, int]],
    stop_at_last: bool,
) -> set[tuple[int, int]]:
    limit = _ray_steps_to_edge(shape, start, direction)
    last_t = intervals[-1][0] if stop_at_last else limit - 1
    blocked: set[tuple[int, int]] = set()
    dr, dc = direction
    covered = set()
    for t0, t1 in intervals:
        covered.update(range(t0, t1 + 1))
    for t in range(last_t + 1):
        if t in covered:
            continue
        r = start[0] + dr * t
        c = start[1] + dc * t
        blocked.add((r, c))
    return blocked


def _sample_case() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    grid = np.zeros((h, w), dtype=int)
    used: set[tuple[int, int]] = set()

    n_emitters = random.randint(1, 3)
    patterns = [list(random.choice(PATTERNS)) for _ in range(n_emitters)]
    if not any("frame" in p for p in patterns):
        patterns[0] = ["frame", "solid"]
    if not any(p[-1] == "solid" for p in patterns):
        patterns[0] = ["frame", "solid"]

    emitter_colors = random.sample(range(1, 10), n_emitters)

    for emitter_idx in range(n_emitters):
        pattern = patterns[emitter_idx]
        emitter_color = emitter_colors[emitter_idx]
        placed = False
        for _ in range(120):
            direction = random.choice(list(DIR_TO_MISSING))
            sizes = [random.randint(3, 5) if kind == "frame" else random.randint(2, 4) for kind in pattern]
            gaps = [random.randint(1, 3) for _ in pattern]
            intervals: list[tuple[int, int]] = []
            cursor = 0
            for gap, size in zip(gaps, sizes):
                entry_t = cursor + gap
                exit_t = entry_t + size - 1
                intervals.append((entry_t, exit_t))
                cursor = exit_t
            need_steps = cursor + 1

            mr, mc = DIR_TO_MISSING[direction]
            candidates = []
            for r in range(h):
                for c in range(w):
                    r0 = r - mr
                    c0 = c - mc
                    if not (0 <= r0 and r0 + 1 < h and 0 <= c0 and c0 + 1 < w):
                        continue
                    if _ray_steps_to_edge((h, w), (r, c), direction) < need_steps:
                        continue
                    candidates.append((r, c))
            random.shuffle(candidates)

            for start in candidates:
                emitter_cells = _paint_emitter(np.zeros((h, w), dtype=int), start, direction, 1)
                if emitter_cells & used or start in used:
                    continue

                bboxes = []
                target_cells: set[tuple[int, int]] = set()
                ok = True
                for (t0, _t1), size, kind in zip(intervals, sizes, pattern):
                    entry = (start[0] + direction[0] * t0, start[1] + direction[1] * t0)
                    bbox = _square_bbox(entry, direction, size)
                    if not _in_bounds_bbox((h, w), bbox):
                        ok = False
                        break
                    r0, c0, r1, c1 = bbox
                    bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                    if bbox_cells & used or bbox_cells & emitter_cells or start in bbox_cells:
                        ok = False
                        break
                    bboxes.append((bbox, kind))
                    target_cells |= bbox_cells
                if not ok:
                    continue

                path_cells = _forbidden_path(
                    (h, w),
                    start,
                    direction,
                    intervals,
                    stop_at_last=pattern[-1] == "solid",
                )
                if path_cells & used or path_cells & target_cells or path_cells & emitter_cells:
                    continue

                _paint_emitter(grid, start, direction, emitter_color)
                used |= emitter_cells
                target_palette = [c for c in range(1, 10) if c != emitter_color]
                random.shuffle(target_palette)
                for i, (bbox, kind) in enumerate(bboxes):
                    target_color = target_palette[i % len(target_palette)]
                    used |= _paint_target(grid, bbox, kind, target_color)
                used |= path_cells
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    n_decoys = random.randint(0, 2)
    for _ in range(n_decoys):
        placed = False
        for _ in range(80):
            kind = random.choice(["solid", "frame"])
            size = random.randint(3, 5) if kind == "frame" else random.randint(2, 4)
            r0 = random.randint(0, h - size)
            c0 = random.randint(0, w - size)
            bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
            bbox_cells = {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}
            if bbox_cells & used:
                continue
            color = random.randint(1, 9)
            _paint_target(grid, bbox, kind, color)
            used |= bbox_cells
            placed = True
            break
        if not placed:
            continue

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
