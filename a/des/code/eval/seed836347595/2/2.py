from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    pad: int,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _extract_cycle(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, tuple[int, int, int, int], list[int], np.ndarray]] = []

    for comp in _components(g == GREY):
        if len(comp) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == GREY):
            continue
        if np.any(sub[~border] == GREY):
            continue

        cycle: list[int] | None = None
        sh, sw = sub.shape
        if sh == 3 and sw in (7, 9, 11):
            line = sub[1, 1:-1]
            colors = line[::2]
            gaps = line[1::2]
            if (
                3 <= len(colors) <= 5
                and np.all(gaps == BLACK)
                and np.all(colors != BLACK)
                and np.all(colors != GREY)
                and len(set(map(int, colors))) == len(colors)
            ):
                cycle = [int(v) for v in colors]
        elif sw == 3 and sh in (7, 9, 11):
            line = sub[1:-1, 1]
            colors = line[::2]
            gaps = line[1::2]
            if (
                3 <= len(colors) <= 5
                and np.all(gaps == BLACK)
                and np.all(colors != BLACK)
                and np.all(colors != GREY)
                and len(set(map(int, colors))) == len(colors)
            ):
                cycle = [int(v) for v in colors]

        if cycle is None:
            continue
        legend_mask = np.zeros((h, w), dtype=bool)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append((len(cycle), (r0, c0, r1, c1), cycle, legend_mask))

    assert candidates, "expected one grey-framed legend on the edge"
    candidates.sort(key=lambda item: (item[0], -item[1][0], -item[1][1]), reverse=True)
    _, _bbox0, cycle, legend_mask = candidates[0]
    return cycle, legend_mask


def _find_sources(grid: np.ndarray, legend_mask: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    mask = (g != BLACK) & (g != GREY) & (~legend_mask)
    sources: list[dict[str, object]] = []
    for comp in _components(mask):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != 9 or (r1 - r0 + 1) != 3 or (c1 - c0 + 1) != 3:
            continue
        color = next(iter(colors))
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        sources.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + 1, c0 + 1),
            }
        )
    sources.sort(key=lambda src: tuple(src["center"]))  # type: ignore[arg-type]
    return sources


def _ray_relays(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dr, dc = direction
    r = center[0] + 2 * dr
    c = center[1] + 2 * dc
    relays: list[tuple[int, int]] = []

    while 0 <= r < h and 0 <= c < w:
        if legend_mask[r, c]:
            break
        value = int(g[r, c])
        if value == BLACK:
            r += dr
            c += dc
            continue
        if value == GREY:
            relays.append((r, c))
            r += dr
            c += dc
            continue
        break
    return relays


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cycle, legend_mask = _extract_cycle(g)
    cycle_index = {color: i for i, color in enumerate(cycle)}
    sources = _find_sources(g, legend_mask)

    chains_by_source: list[list[dict[str, object]]] = []
    for src in sources:
        color = int(src["color"])
        if color not in cycle_index:
            chains_by_source.append([])
            continue
        center = src["center"]
        src_chains: list[dict[str, object]] = []
        for direction in DIRS.values():
            relays = _ray_relays(g, legend_mask, center, direction)  # type: ignore[arg-type]
            if relays:
                src_chains.append({"seq": relays, "base_idx": cycle_index[color], "active": True})
        chains_by_source.append(src_chains)

    out = g.copy()
    hop = 1
    while True:
        any_pending = False
        for src_chains in chains_by_source:
            for chain in src_chains:
                if not bool(chain["active"]):
                    continue
                seq = chain["seq"]
                if len(seq) < hop:  # type: ignore[arg-type]
                    continue
                any_pending = True
                cell = seq[hop - 1]  # type: ignore[index]
                if int(out[cell]) != GREY:
                    chain["active"] = False
                    continue
                base_idx = int(chain["base_idx"])
                out[cell] = cycle[(base_idx + hop) % len(cycle)]
        if not any_pending:
            break
        hop += 1

    return out


def _draw_legend(grid: np.ndarray, cycle: list[int], side: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    n = len(cycle)
    if side in ("top", "bottom"):
        box_h = 3
        box_w = 2 * n + 1
        r0 = 0 if side == "top" else h - box_h
        c0 = random.randint(0, w - box_w)
        grid[r0 : r0 + box_h, c0 : c0 + box_w] = BLACK
        grid[r0, c0 : c0 + box_w] = GREY
        grid[r0 + box_h - 1, c0 : c0 + box_w] = GREY
        grid[r0 : r0 + box_h, c0] = GREY
        grid[r0 : r0 + box_h, c0 + box_w - 1] = GREY
        for i, color in enumerate(cycle):
            grid[r0 + 1, c0 + 1 + 2 * i] = int(color)
        return r0, c0, r0 + box_h - 1, c0 + box_w - 1

    box_h = 2 * n + 1
    box_w = 3
    r0 = random.randint(0, h - box_h)
    c0 = 0 if side == "left" else w - box_w
    grid[r0 : r0 + box_h, c0 : c0 + box_w] = BLACK
    grid[r0, c0 : c0 + box_w] = GREY
    grid[r0 + box_h - 1, c0 : c0 + box_w] = GREY
    grid[r0 : r0 + box_h, c0] = GREY
    grid[r0 : r0 + box_h, c0 + box_w - 1] = GREY
    for i, color in enumerate(cycle):
        grid[r0 + 1 + 2 * i, c0 + 1] = int(color)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _candidate_centers(
    shape: tuple[int, int],
    legend_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    legend_pad = _expand_bbox(legend_bbox, 1, shape)
    centers: list[tuple[int, int]] = []
    for r in range(3, h - 2, 4):
        for c in range(3, w - 2, 4):
            source_bbox = (r - 1, c - 1, r + 1, c + 1)
            if _rect_overlap(_expand_bbox(source_bbox, 1, shape), legend_pad):
                continue
            centers.append((r, c))
    return centers


def _pick_conflict_pair(
    candidates: list[tuple[int, int]],
    orientation: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if orientation == "horizontal":
        rows = sorted({r for r, _ in candidates})
        for r in rows:
            cols = sorted(c for rr, c in candidates if rr == r)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if cols[j] - cols[i] >= 8:
                        options.append(((r, cols[i]), (r, cols[j])))
    else:
        cols = sorted({c for _, c in candidates})
        for c in cols:
            rows = sorted(r for r, cc in candidates if cc == c)
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if rows[j] - rows[i] >= 8:
                        options.append(((rows[i], c), (rows[j], c)))
    if not options:
        return None
    return random.choice(options)


def _pick_source_centers(
    candidates: list[tuple[int, int]],
    source_count: int,
    conflict_pair: tuple[tuple[int, int], tuple[int, int]] | None,
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        chosen: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        if conflict_pair is not None:
            a, b = conflict_pair
            chosen.extend([a, b])
            used_rows.update([a[0], b[0]])
            used_cols.update([a[1], b[1]])

        pool = candidates[:]
        random.shuffle(pool)
        for center in pool:
            if center in chosen:
                continue
            r, c = center
            if r in used_rows or c in used_cols:
                continue
            chosen.append(center)
            used_rows.add(r)
            used_cols.add(c)
            if len(chosen) == source_count:
                return sorted(chosen)
        if len(chosen) == source_count:
            return sorted(chosen)
    return None


def _available_track(
    grid: np.ndarray,
    center: tuple[int, int],
    direction: str,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    dr, dc = DIRS[direction]
    r = center[0] + 2 * dr
    c = center[1] + 2 * dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BLACK:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _choose_chain_cells(
    track: list[tuple[int, int]],
    *,
    min_relays: int,
) -> list[tuple[int, int]] | None:
    if not track:
        return None
    max_relays = min(4, len(track))
    if min_relays > max_relays:
        return None

    for _ in range(60):
        count = random.randint(min_relays, max_relays)
        first_hi = min(2, len(track) - 1)
        positions = [random.randint(0, first_hi)]
        ok = True
        for idx in range(1, count):
            lo = positions[-1] + 2
            need_tail = 2 * (count - idx - 1)
            hi = min(positions[-1] + 4, len(track) - 1 - need_tail)
            if lo > hi:
                ok = False
                break
            positions.append(random.randint(lo, hi))
        if ok:
            return [track[pos] for pos in positions]
    if min_relays == 1:
        return [track[random.randint(0, len(track) - 1)]]
    return None


def _validate_generated(grid: np.ndarray) -> None:
    cycle, legend_mask = _extract_cycle(grid)
    assert 3 <= len(cycle) <= 5
    relay_mask = (grid == GREY) & (~legend_mask)
    for comp in _components(relay_mask):
        assert len(comp) == 1

    sources = _find_sources(grid, legend_mask)
    assert 2 <= len(sources) <= 5
    assert all(int(src["color"]) in set(cycle) for src in sources)

    out = main(grid)
    changed_relays = int(np.count_nonzero(relay_mask & (out != GREY)))
    assert changed_relays >= 1


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(19, 27)
    w = random.randint(19, 27)
    cycle = random.sample(PALETTE, random.randint(3, 5))
    legend_side = random.choice(["top", "bottom", "left", "right"])

    grid = np.full((h, w), BLACK, dtype=int)
    legend_bbox = _draw_legend(grid, cycle, legend_side)
    candidates = _candidate_centers(grid.shape, legend_bbox)
    if len(candidates) < 2:
        return None

    want_conflict = random.random() < 0.5
    conflict_pair = None
    if want_conflict:
        orientations = ["horizontal", "vertical"]
        random.shuffle(orientations)
        for orientation in orientations:
            conflict_pair = _pick_conflict_pair(candidates, orientation)
            if conflict_pair is not None:
                break

    max_sources = min(5, len(candidates))
    source_count = random.randint(2, max_sources)
    if conflict_pair is not None:
        source_count = max(source_count, 2)

    centers = _pick_source_centers(candidates, source_count, conflict_pair)
    if centers is None:
        return None

    centers = sorted(centers)
    colors = [random.choice(cycle) for _ in centers]
    source_index = {center: idx for idx, center in enumerate(centers)}
    used_dirs: list[set[str]] = [set() for _ in centers]

    for center, color in zip(centers, colors, strict=False):
        r, c = center
        grid[r - 1 : r + 2, c - 1 : c + 2] = int(color)

    if conflict_pair is not None:
        a, b = conflict_pair
        ia = source_index[a]
        ib = source_index[b]
        if a[0] == b[0]:
            left, right = (a, b) if a[1] < b[1] else (b, a)
            left_i = source_index[left]
            right_i = source_index[right]
            relay_c = random.randint(left[1] + 2, right[1] - 2)
            grid[left[0], relay_c] = GREY
            used_dirs[left_i].add("right")
            used_dirs[right_i].add("left")
        else:
            top, bottom = (a, b) if a[0] < b[0] else (b, a)
            top_i = source_index[top]
            bottom_i = source_index[bottom]
            relay_r = random.randint(top[0] + 2, bottom[0] - 2)
            grid[relay_r, top[1]] = GREY
            used_dirs[top_i].add("down")
            used_dirs[bottom_i].add("up")

    multi_options: list[tuple[int, str]] = []
    for idx, center in enumerate(centers):
        for direction in DIRS:
            if direction in used_dirs[idx]:
                continue
            if len(_available_track(grid, center, direction)) >= 3:
                multi_options.append((idx, direction))
    if not multi_options:
        return None

    forced_idx, forced_dir = random.choice(multi_options)
    forced_track = _available_track(grid, centers[forced_idx], forced_dir)
    forced_cells = _choose_chain_cells(forced_track, min_relays=2)
    if forced_cells is None:
        return None
    for r, c in forced_cells:
        grid[r, c] = GREY
    used_dirs[forced_idx].add(forced_dir)

    for idx, center in enumerate(centers):
        available_dirs = [d for d in DIRS if d not in used_dirs[idx] and _available_track(grid, center, d)]
        cap = min(3, len(used_dirs[idx]) + len(available_dirs))
        if cap == 0:
            return None
        want_total = random.randint(max(1, len(used_dirs[idx])), cap)
        need_extra = want_total - len(used_dirs[idx])
        random.shuffle(available_dirs)
        for direction in available_dirs:
            if need_extra == 0:
                break
            track = _available_track(grid, center, direction)
            cells = _choose_chain_cells(track, min_relays=1)
            if cells is None:
                continue
            for r, c in cells:
                grid[r, c] = GREY
            used_dirs[idx].add(direction)
            need_extra -= 1
        if len(used_dirs[idx]) == 0:
            return None

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
