import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
LABEL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # E, S, W, N in clockwise order
TURN_INV = {"L": "R", "R": "L"}
LEGEND_CLASSES = [
    (("L", "L"), ("R", "R")),
    (("L", "R"),),
    (("R", "L"),),
]


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _trace_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(coords)
    nbrs = {
        cell: [n for n in _neighbors4(cell[0], cell[1], 10**9, 10**9) if n in cells]
        for cell in cells
    }
    if len(cells) == 1:
        return [next(iter(cells))]
    endpoints = [cell for cell, ns in nbrs.items() if len(ns) == 1]
    assert len(endpoints) == 2, "grey component must be a simple path with two endpoints"
    start = endpoints[0]
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in nbrs[cur] if n != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path component must not branch"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    assert len(path) == len(cells), "path trace must visit every path pixel exactly once"
    return path


def _turns_from_path(path: list[tuple[int, int]]) -> tuple[str, ...]:
    if len(path) < 3:
        return ()
    dir_ids = []
    dir_map = {d: i for i, d in enumerate(DIRS)}
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        step = (r1 - r0, c1 - c0)
        assert step in dir_map, "path must be 4-connected"
        dir_ids.append(dir_map[step])
    turns: list[str] = []
    for prev, cur in zip(dir_ids, dir_ids[1:]):
        if prev == cur:
            continue
        diff = (cur - prev) % 4
        assert diff in (1, 3), "path must not reverse direction"
        turns.append("R" if diff == 1 else "L")
    return tuple(turns)


def _path_signatures(path: list[tuple[int, int]]) -> set[tuple[str, ...]]:
    return {
        _turns_from_path(path),
        _turns_from_path(list(reversed(path))),
    }


def _starts_with(full: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(full) >= len(prefix) and full[: len(prefix)] == prefix


def _separator_row(grid: np.ndarray, bg: int) -> int:
    h, _w = grid.shape
    candidates = [
        r
        for r in range(1, h - 1)
        if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert candidates, "grid must contain a full-width separator row"
    return int(candidates[-1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each long grey path using the legend color whose turn signature matches the path prefix.

    Input structure:
    - The background is mostly black, with a full-width grey horizontal separator near the bottom.
    - Below the separator sits a legend made of several short grey 4-connected paths, each touching one adjacent colored label pixel.
    - Above the separator sit several longer grey 4-connected paths; each one is isolated and has exactly two endpoints.

    Rule:
    - For every legend path, traverse it from either endpoint and treat its left/right turn sequence as a signature class.
    - For every long path in the upper region, extract its turn sequence from both endpoint orientations.
    - Find the unique legend signature that matches a prefix of one of those long-path sequences, then recolor the entire long path with that legend's label color.
    - Keep the separator line and the whole legend region unchanged.

    Key insight:
    - Geometry and location are distractions; the only matching key is the sequence of left/right turns made by the path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep = _separator_row(g, bg)
    grey = int(g[sep, 0])

    bottom = g[sep + 1 :]
    bh, bw = bottom.shape
    label_cells = {
        (r, c): int(bottom[r, c])
        for r in range(bh)
        for c in range(bw)
        if bottom[r, c] not in (bg, grey)
    }

    legend: dict[tuple[str, ...], int] = {}
    for comp in _components(bottom == grey):
        adjacent_labels = {
            label_cells[(nr, nc)]
            for r, c in comp
            for nr, nc in _neighbors4(r, c, bh, bw)
            if (nr, nc) in label_cells
        }
        assert len(adjacent_labels) == 1, "each legend path must touch exactly one colored label"
        color = next(iter(adjacent_labels))
        path = _trace_path(comp)
        for sig in _path_signatures(path):
            prev = legend.get(sig)
            assert prev in (None, color), "legend signatures must be unambiguous"
            legend[sig] = color

    assert legend, "legend region must define at least one signature"

    out = g.copy()
    top = g[:sep]
    for comp in _components(top == grey):
        path = _trace_path(comp)
        orientations = [
            _turns_from_path(path),
            _turns_from_path(list(reversed(path))),
        ]
        matches = {
            color
            for sig, color in legend.items()
            if any(_starts_with(turns, sig) for turns in orientations)
        }
        assert len(matches) == 1, "each long path must match exactly one legend color"
        new_color = next(iter(matches))
        for r, c in comp:
            out[r, c] = new_color

    return out


def _dual_signature(sig: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(TURN_INV[t] for t in reversed(sig))


def _shift_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    return [(r - min_r, c - min_c) for r, c in path]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return (
        max(r for r, _ in coords) + 1,
        max(c for _, c in coords) + 1,
    )


def _build_path_from_turns(
    turns: tuple[str, ...], lengths: tuple[int, ...], start_dir: int
) -> list[tuple[int, int]] | None:
    r = c = 0
    dir_idx = int(start_dir)
    path = [(r, c)]
    used = {(r, c)}
    for seg_idx, seg_len in enumerate(lengths):
        dr, dc = DIRS[dir_idx]
        for _ in range(seg_len):
            r += dr
            c += dc
            if (r, c) in used:
                return None
            path.append((r, c))
            used.add((r, c))
        if seg_idx < len(turns):
            dir_idx = (dir_idx + (1 if turns[seg_idx] == "R" else -1)) % 4
    return path


def _label_candidates(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(path)
    cands: list[tuple[int, int]] = []
    if len(path) >= 2:
        for endpoint, inner in ((path[0], path[1]), (path[-1], path[-2])):
            dr = endpoint[0] - inner[0]
            dc = endpoint[1] - inner[1]
            cand = (endpoint[0] + dr, endpoint[1] + dc)
            if cand not in cells:
                nbr_count = sum(
                    (cand[0] + rr, cand[1] + cc) in cells for rr, cc in DIRS
                )
                if nbr_count == 1:
                    cands.append(cand)
    for r, c in path:
        for dr, dc in DIRS:
            cand = (r + dr, c + dc)
            if cand in cells or cand in cands:
                continue
            nbr_count = sum(
                (cand[0] + rr, cand[1] + cc) in cells for rr, cc in DIRS
            )
            if nbr_count == 1:
                cands.append(cand)
    return cands


def _normalize_object(
    path: list[tuple[int, int]], label: tuple[int, int] | None = None
) -> tuple[list[tuple[int, int]], tuple[int, int] | None, list[tuple[int, int]]]:
    all_cells = list(path)
    if label is not None:
        all_cells.append(label)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    path2 = [(r - min_r, c - min_c) for r, c in path]
    label2 = None if label is None else (label[0] - min_r, label[1] - min_c)
    full = list(path2) + ([] if label2 is None else [label2])
    return path2, label2, full


def _matched_colors(
    path: list[tuple[int, int]], legend_map: dict[tuple[str, ...], int]
) -> set[int]:
    sig1 = _turns_from_path(path)
    sig2 = _turns_from_path(list(reversed(path)))
    return {
        color
        for sig, color in legend_map.items()
        if _starts_with(sig1, sig) or _starts_with(sig2, sig)
    }


def _make_legend_entry(class_sigs: tuple[tuple[str, ...], ...], color: int):
    short_lengths = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
    ]
    for _ in range(200):
        turns = random.choice(class_sigs)
        path = _build_path_from_turns(turns, random.choice(short_lengths), random.randrange(4))
        if path is None:
            continue
        cands = _label_candidates(path)
        if not cands:
            continue
        label = random.choice(cands)
        path2, label2, full = _normalize_object(path, label)
        h, w = _bbox(full)
        if h > 5 or w > 7:
            continue
        return {
            "color": int(color),
            "class_sigs": class_sigs,
            "path": path2,
            "label": label2,
            "full": full,
        }
    raise RuntimeError("failed to generate legend entry")


def _make_long_path(
    class_sigs: tuple[tuple[str, ...], ...],
    target_color: int,
    legend_map: dict[tuple[str, ...], int],
    extra_turns: int,
):
    assert extra_turns >= 0
    for _ in range(400):
        turns = list(random.choice(class_sigs))
        turns.extend(random.choice(("L", "R")) for _ in range(extra_turns))
        seg_count = len(turns) + 1
        lengths = []
        for idx in range(seg_count):
            lo, hi = (2, 4) if idx in (0, seg_count - 1) else (1, 3)
            lengths.append(random.randint(lo, hi))
        if sum(lengths) < 6:
            continue
        path = _build_path_from_turns(tuple(turns), tuple(lengths), random.randrange(4))
        if path is None:
            continue
        path = _shift_path(path)
        h, w = _bbox(path)
        if h > 9 or w > 9 or len(path) < 7:
            continue
        if _matched_colors(path, legend_map) != {target_color}:
            continue
        return {
            "color": int(target_color),
            "path": path,
            "full": list(path),
        }
    raise RuntimeError("failed to generate long path")


def _can_place(
    occupied: np.ndarray, coords: list[tuple[int, int]], top: int, left: int
) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr, cc = top + r, left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0, r1 = max(0, rr - 1), min(h, rr + 2)
        c0, c1 = max(0, cc - 1), min(w, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_object(
    occupied: np.ndarray,
    coords: list[tuple[int, int]],
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> tuple[int, int] | None:
    obj_h, obj_w = _bbox(coords)
    max_r = row_hi - obj_h
    max_c = col_hi - obj_w
    if max_r < row_lo or max_c < col_lo:
        return None
    candidates = [
        (r, c)
        for r in range(row_lo, max_r + 1)
        for c in range(col_lo, max_c + 1)
    ]
    random.shuffle(candidates)
    for top, left in candidates:
        if _can_place(occupied, coords, top, left):
            for r, c in coords:
                occupied[top + r, left + c] = True
            return top, left
    return None


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _gen_attempt() -> np.ndarray | None:
    class_count = random.choice((2, 3))
    chosen_classes = random.sample(LEGEND_CLASSES, class_count)
    colors = random.sample(LABEL_COLORS, class_count)
    entries = [
        _make_legend_entry(class_sigs, color)
        for class_sigs, color in zip(chosen_classes, colors)
    ]

    legend_map: dict[tuple[str, ...], int] = {}
    for entry in entries:
        for sig in entry["class_sigs"]:
            legend_map[sig] = entry["color"]
            legend_map[_dual_signature(sig)] = entry["color"]

    targets = list(range(class_count))
    for _ in range(random.randint(0, 2)):
        targets.append(random.randrange(class_count))
    random.shuffle(targets)

    extra_plan = [random.randint(0, 2) for _ in targets]
    if extra_plan:
        extra_plan[random.randrange(len(extra_plan))] = random.randint(1, 2)
    if len(extra_plan) >= 2:
        extra_plan[random.randrange(len(extra_plan))] = 0

    long_paths = [
        _make_long_path(
            entries[target]["class_sigs"],
            entries[target]["color"],
            legend_map,
            extra_turns,
        )
        for target, extra_turns in zip(targets, extra_plan)
    ]

    width = random.randint(20, 29)
    bottom_h = random.randint(6, 7)
    top_h = random.randint(10, 20)
    total_h = top_h + 1 + bottom_h
    if total_h > 30:
        top_h -= total_h - 30
        total_h = 30
    if top_h < 10:
        return None

    grid = np.full((total_h, width), BG, dtype=int)
    sep = top_h
    grid[sep, :] = GREY

    top_occ = np.zeros((top_h, width), dtype=bool)
    placed_main = []
    for obj in long_paths:
        pos = _place_object(top_occ, obj["full"], 1, max(2, top_h - 1), 1, max(2, width - 1))
        if pos is None:
            return None
        dr, dc = pos
        abs_path = _translate(obj["path"], dr, dc)
        placed_main.append((abs_path, obj["color"]))
        for r, c in abs_path:
            grid[r, c] = GREY

    bottom_occ = np.zeros((bottom_h, width), dtype=bool)
    for entry in entries:
        pos = _place_object(
            bottom_occ,
            entry["full"],
            1,
            max(2, bottom_h - 1),
            1,
            max(2, width - 1),
        )
        if pos is None:
            return None
        dr, dc = pos
        for r, c in _translate(entry["path"], dr + sep + 1, dc):
            grid[r, c] = GREY
        label_r, label_c = entry["label"]
        grid[sep + 1 + dr + label_r, dc + label_c] = entry["color"]

    out = main(grid)
    if not np.array_equal(out[sep:], grid[sep:]):
        return None
    for abs_path, color in placed_main:
        if any(out[r, c] != color for r, c in abs_path):
            return None
    if np.any(out[:sep] == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except AssertionError:
            grid = None
        except RuntimeError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for path-signature recoloring task")
