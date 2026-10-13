from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SOURCE_PRIORITY = [RED, GREEN, YELLOW, BLUE, MAGENTA, AZURE]
PRIORITY_INDEX = {color: idx for idx, color in enumerate(SOURCE_PRIORITY)}


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    return len(_components(mask, connectivity=8)) == 1


def _fill_color(color_a: int, color_b: int) -> int:
    if PRIORITY_INDEX[color_a] >= PRIORITY_INDEX[color_b]:
        return color_a
    return color_b


def _find_canvases(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    canvases: list[dict] = []
    border_lookup: dict[tuple[int, int], int] = {}
    for coords in _components(grid == ORANGE, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height < 5:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border != (int(grid[r, c]) == ORANGE):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        idx = len(canvases)
        canvases.append(
            {
                "bbox": (r0, c0, r1, c1),
                "size": height - 2,
            }
        )
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) or c in (c0, c1):
                    border_lookup[(r, c)] = idx
    return canvases, border_lookup


def _find_sources(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    mask = (grid != BLACK) & (grid != GREY) & (grid != ORANGE)
    sources: list[dict] = []
    cell_lookup: dict[tuple[int, int], int] = {}
    for coords in _components(mask, connectivity=8):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        if color not in PRIORITY_INDEX:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height != width or height < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not set(int(v) for v in np.unique(sub)).issubset({BLACK, color}):
            continue
        shape = sub == color
        if not _touches_all_sides(shape):
            continue
        idx = len(sources)
        sources.append(
            {
                "bbox": (r0, c0, r1, c1),
                "size": height,
                "color": color,
                "shape": shape.copy(),
            }
        )
        for r, c in coords:
            cell_lookup[(int(r), int(c))] = idx
    return sources, cell_lookup


def _extract_assemblies(grid: np.ndarray) -> list[dict]:
    canvases, canvas_lookup = _find_canvases(grid)
    sources, source_lookup = _find_sources(grid)

    graph: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for idx in range(len(canvases)):
        graph[("c", idx)] = set()
    for idx in range(len(sources)):
        graph[("s", idx)] = set()

    grey_components = _components(grid == GREY, connectivity=4)
    for gidx, coords in enumerate(grey_components):
        node = ("g", gidx)
        graph[node] = set()
        near_sources: set[int] = set()
        near_canvases: set[int] = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                sid = source_lookup.get((nr, nc))
                if sid is not None:
                    near_sources.add(sid)
                cid = canvas_lookup.get((nr, nc))
                if cid is not None:
                    near_canvases.add(cid)
        for sid in near_sources:
            graph[node].add(("s", sid))
            graph[("s", sid)].add(node)
        for cid in near_canvases:
            graph[node].add(("c", cid))
            graph[("c", cid)].add(node)

    assemblies: list[dict] = []
    visited: set[tuple[str, int]] = set()
    for start, neighbors in graph.items():
        if start in visited or not neighbors:
            continue
        stack = [start]
        visited.add(start)
        part: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            part.add(node)
            for nxt in graph[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        canvas_ids = sorted(idx for kind, idx in part if kind == "c")
        source_ids = sorted(idx for kind, idx in part if kind == "s")
        if len(canvas_ids) != 1 or len(source_ids) != 2:
            continue
        canvas = canvases[canvas_ids[0]]
        source_a = sources[source_ids[0]]
        source_b = sources[source_ids[1]]
        size = canvas["size"]
        if source_a["size"] != size or source_b["size"] != size:
            continue
        assemblies.append({"canvas": canvas, "sources": [source_a, source_b]})
    return assemblies


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for assembly in _extract_assemblies(g):
        canvas = assembly["canvas"]
        source_a, source_b = assembly["sources"]
        mask = source_a["shape"] & source_b["shape"]
        color = _fill_color(int(source_a["color"]), int(source_b["color"]))
        r0, c0, r1, c1 = canvas["bbox"]
        inner = np.full((canvas["size"], canvas["size"]), BLACK, dtype=int)
        inner[mask] = color
        out[r0 + 1 : r1, c0 + 1 : c1] = inner
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _transform(grid)


def _side_cell(n: int, side: str, idx: int) -> tuple[int, int]:
    if side == "top":
        return (0, idx)
    if side == "bottom":
        return (n - 1, idx)
    if side == "left":
        return (idx, 0)
    if side == "right":
        return (idx, n - 1)
    raise ValueError(f"unknown side {side}")


def _random_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = start
    path = [(r, c)]
    while (r, c) != end:
        axes: list[str] = []
        if r != end[0]:
            axes.append("r")
        if c != end[1]:
            axes.append("c")
        axis = rng.choice(axes)
        if axis == "r":
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        path.append((r, c))
    return path


def _connect_target(mask: np.ndarray, target: tuple[int, int], rng: random.Random) -> None:
    occupied = np.argwhere(mask)
    anchor = tuple(occupied[rng.randrange(len(occupied))])
    for r, c in _random_path(target, anchor, rng):
        mask[r, c] = True


def _grow_pattern(mask: np.ndarray, steps: int, rng: random.Random) -> None:
    n = mask.shape[0]
    for _ in range(steps):
        occupied = np.argwhere(mask)
        r, c = occupied[rng.randrange(len(occupied))]
        dirs = DIR4[:]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < n and 0 <= nc < n:
                mask[nr, nc] = True
                break


def _try_remove_cells(mask: np.ndarray, required: tuple[int, int], rng: random.Random) -> np.ndarray:
    out = mask.copy()
    candidates = [tuple(p) for p in np.argwhere(out)]
    rng.shuffle(candidates)
    removals = rng.randint(0, max(1, len(candidates) // 5))
    for r, c in candidates:
        if removals == 0:
            break
        if (r, c) == required:
            continue
        test = out.copy()
        test[r, c] = False
        if test[required] and _touches_all_sides(test) and _is_connected(test):
            out = test
            removals -= 1
    return out


def _make_pattern(
    n: int,
    required_side: str,
    required_idx: int,
    rng: random.Random,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    required = _side_cell(n, required_side, required_idx)
    for _ in range(120):
        if seed_mask is None:
            mask = np.zeros((n, n), dtype=bool)
            center_lo = 1 if n > 2 else 0
            center_hi = n - 2 if n > 2 else n - 1
            seed = (
                rng.randint(center_lo, center_hi),
                rng.randint(center_lo, center_hi),
            )
            mask[seed] = True
        else:
            mask = seed_mask.copy()

        targets = [
            (0, rng.randrange(n)),
            (n - 1, rng.randrange(n)),
            (rng.randrange(n), 0),
            (rng.randrange(n), n - 1),
            required,
        ]
        rng.shuffle(targets)
        seen: set[tuple[int, int]] = set()
        for target in targets:
            if target in seen:
                continue
            seen.add(target)
            if not mask[target]:
                _connect_target(mask, target, rng)
        _grow_pattern(mask, rng.randint(n, n * n // 2 + 1), rng)
        mask = _try_remove_cells(mask, required, rng)
        if mask[required] and _touches_all_sides(mask) and _is_connected(mask):
            return mask
    raise RuntimeError("failed to create source pattern")


def _draw_frame(grid: np.ndarray, top: int, left: int, size: int) -> None:
    bottom = top + size + 1
    right = left + size + 1
    grid[top, left : right + 1] = ORANGE
    grid[bottom, left : right + 1] = ORANGE
    grid[top : bottom + 1, left] = ORANGE
    grid[top : bottom + 1, right] = ORANGE


def _draw_segment(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: int) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = color
    else:
        raise ValueError("segments must be axis-aligned")


def _build_horizontal_assembly(rng: random.Random) -> tuple[np.ndarray, dict]:
    n = rng.randint(3, 5)
    gap_left = rng.randint(1, 4)
    gap_right = rng.randint(1, 4)
    route_above = rng.random() < 0.5
    port_a = rng.randrange(n)
    port_b = rng.randrange(n)

    if rng.random() < 0.5:
        base = _make_pattern(n, rng.choice(["top", "bottom", "left", "right"]), rng.randrange(n), rng)
        pattern_a = _make_pattern(n, "right", port_a, rng, seed_mask=base)
        pattern_b = _make_pattern(n, "left", port_b, rng, seed_mask=base)
    else:
        pattern_a = _make_pattern(n, "right", port_a, rng)
        pattern_b = _make_pattern(n, "left", port_b, rng)

    colors = rng.sample(SOURCE_PRIORITY, 2)
    color_a, color_b = int(colors[0]), int(colors[1])

    height = n + 4
    width = n + gap_left + (n + 2) + gap_right + n
    grid = np.zeros((height, width), dtype=int)

    source_top = 2
    left_source_left = 0
    canvas_left = n + gap_left
    right_source_left = canvas_left + (n + 2) + gap_right

    grid[source_top : source_top + n, left_source_left : left_source_left + n][pattern_a] = color_a
    grid[source_top : source_top + n, right_source_left : right_source_left + n][pattern_b] = color_b
    _draw_frame(grid, 1, canvas_left, n)

    row_a = source_top + port_a
    row_b = source_top + port_b
    left_port_col = canvas_left - 1
    right_port_col = canvas_left + n + 2
    if route_above:
        detour_row = 0
    else:
        detour_row = height - 1

    _draw_segment(grid, (row_a, n), (row_a, left_port_col), GREY)
    _draw_segment(grid, (row_a, left_port_col), (detour_row, left_port_col), GREY)
    _draw_segment(grid, (detour_row, left_port_col), (detour_row, right_port_col), GREY)
    _draw_segment(grid, (detour_row, right_port_col), (row_b, right_port_col), GREY)
    _draw_segment(grid, (row_b, right_port_col), (row_b, right_source_left - 1), GREY)

    meta = {
        "nonempty": bool((pattern_a & pattern_b).any()),
        "nontrivial": bool((pattern_a & pattern_b).any())
        and not np.array_equal(pattern_a & pattern_b, pattern_a)
        and not np.array_equal(pattern_a & pattern_b, pattern_b),
    }
    return grid, meta


def _build_assembly(rng: random.Random) -> tuple[np.ndarray, dict]:
    for _ in range(60):
        local, meta = _build_horizontal_assembly(rng)
        rotated = np.rot90(local, k=rng.randrange(4))
        if len(_extract_assemblies(rotated)) == 1:
            return rotated, meta
    raise RuntimeError("failed to build valid assembly")


def _compose_scene(parts: list[np.ndarray], rng: random.Random) -> np.ndarray:
    if len(parts) == 1:
        part = parts[0]
        top = rng.randint(1, 4)
        left = rng.randint(1, 4)
        bottom = rng.randint(1, 4)
        right = rng.randint(1, 4)
        grid = np.zeros((top + part.shape[0] + bottom, left + part.shape[1] + right), dtype=int)
        grid[top : top + part.shape[0], left : left + part.shape[1]] = part
        return grid

    horizontal = rng.random() < 0.5
    if horizontal:
        gaps = [rng.randint(2, 5) for _ in range(len(parts) + 1)]
        h = max(part.shape[0] for part in parts) + rng.randint(2, 6)
        w = sum(part.shape[1] for part in parts) + sum(gaps)
        grid = np.zeros((h, w), dtype=int)
        x = gaps[0]
        for idx, part in enumerate(parts):
            y = rng.randint(1, h - part.shape[0] - 1)
            grid[y : y + part.shape[0], x : x + part.shape[1]] = part
            x += part.shape[1] + gaps[idx + 1]
        return grid

    gaps = [rng.randint(2, 5) for _ in range(len(parts) + 1)]
    h = sum(part.shape[0] for part in parts) + sum(gaps)
    w = max(part.shape[1] for part in parts) + rng.randint(2, 6)
    grid = np.zeros((h, w), dtype=int)
    y = gaps[0]
    for idx, part in enumerate(parts):
        x = rng.randint(1, w - part.shape[1] - 1)
        grid[y : y + part.shape[0], x : x + part.shape[1]] = part
        y += part.shape[0] + gaps[idx + 1]
    return grid


def _validate_scene(grid: np.ndarray, expected_count: int) -> bool:
    assemblies = _extract_assemblies(grid)
    if len(assemblies) != expected_count:
        return False
    out = _transform(grid)
    if np.array_equal(out, grid):
        return False
    canvas_mask = np.zeros_like(grid, dtype=bool)
    for assembly in assemblies:
        r0, c0, r1, c1 = assembly["canvas"]["bbox"]
        canvas_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if np.any((out != grid) & ~canvas_mask):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        count = 1 if rng.random() < 0.45 else 2
        parts: list[np.ndarray] = []
        metas: list[dict] = []
        ok = True
        for _ in range(count):
            try:
                part, meta = _build_assembly(rng)
            except RuntimeError:
                ok = False
                break
            parts.append(part)
            metas.append(meta)
        if not ok or not any(meta["nonempty"] for meta in metas):
            continue
        scene = _compose_scene(parts, rng)
        if _validate_scene(scene, count):
            return scene
    raise RuntimeError("failed to generate a valid ARC instance")
