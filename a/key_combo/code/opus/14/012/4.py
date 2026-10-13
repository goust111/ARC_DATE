from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
SEP = 4

GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _translate_non_bg(panel: np.ndarray, dr: int, dc: int, bg: int) -> np.ndarray:
    out = np.full_like(panel, bg)
    h, w = panel.shape
    for r, c in np.argwhere(panel != bg):
        nr = int(r) + dr
        nc = int(c) + dc
        assert 0 <= nr < h and 0 <= nc < w, "translated sprite must stay in bounds"
        out[nr, nc] = int(panel[r, c])
    return out


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])

    candidates: list[list[np.ndarray]] = []

    solid_rows_by_color: dict[int, list[int]] = {}
    for r in range(g.shape[0]):
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg:
            solid_rows_by_color.setdefault(int(g[r, 0]), []).append(r)
    solid_cols_by_color: dict[int, list[int]] = {}
    for c in range(g.shape[1]):
        if np.all(g[:, c] == int(g[0, c])) and int(g[0, c]) != bg:
            solid_cols_by_color.setdefault(int(g[0, c]), []).append(c)

    for sep_color, rows in solid_rows_by_color.items():
        if len(rows) < 3:
            continue
        panel_h = rows[0]
        expected = [panel_h + i * (panel_h + 1) for i in range(3)]
        if rows[:3] != expected or g.shape[0] != 4 * panel_h + 3:
            continue
        panels = []
        start = 0
        for sep in expected:
            panels.append(g[start:sep, :].copy())
            start = sep + 1
        panels.append(g[start:, :].copy())
        if len({p.shape for p in panels}) == 1 and all(not np.any(panel == sep_color) for panel in panels):
            candidates.append(panels)

    for sep_color, cols in solid_cols_by_color.items():
        if len(cols) < 3:
            continue
        panel_w = cols[0]
        expected = [panel_w + i * (panel_w + 1) for i in range(3)]
        if cols[:3] != expected or g.shape[1] != 4 * panel_w + 3:
            continue
        panels = []
        start = 0
        for sep in expected:
            panels.append(g[:, start:sep].copy())
            start = sep + 1
        panels.append(g[:, start:].copy())
        if len({p.shape for p in panels}) == 1 and all(not np.any(panel == sep_color) for panel in panels):
            candidates.append(panels)

    assert len(candidates) == 1, "expected a unique horizontal or vertical panel strip"
    return candidates[0], bg


def _extract_frame(panel: np.ndarray, bg: int) -> dict:
    mask = panel != bg
    assert mask.any(), "panel must contain a sprite"
    assert _connected(mask), "sprite must be 4-connected"
    r0, c0, r1, c1 = _bbox(mask)
    crop = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
    local_mask = crop != bg
    red_coords = {tuple(map(int, rc)) for rc in np.argwhere(crop == RED)}
    blue_coords = {tuple(map(int, rc)) for rc in np.argwhere(crop == BLUE)}
    assert red_coords, "each frame must contain red cells"
    assert red_coords | blue_coords == {
        tuple(map(int, rc)) for rc in np.argwhere(local_mask)
    }
    return {
        "anchor": (r0, c0),
        "crop": crop,
        "mask": local_mask,
        "red": red_coords,
        "blue": blue_coords,
    }


def _path_neighbors(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        here = []
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords:
                here.append((nr, nc))
        neighbors[(r, c)] = here
    return neighbors


def _walk_path(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in neighbors[cur] if n != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path walk became ambiguous"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _infer_next_red(mask: np.ndarray, red_sets: list[set[tuple[int, int]]]) -> tuple[int, int]:
    assert len(red_sets) == 4, "expected four observed frames"
    assert int(mask.sum()) >= 6, "sprite must have at least one blue cell after frame 5"

    sizes = [len(rs) for rs in red_sets]
    assert sizes == [1, 2, 3, 4], "red track must grow by one cell per frame"

    sequence = [next(iter(red_sets[0]))]
    for i in range(1, 4):
        prev = red_sets[i - 1]
        cur = red_sets[i]
        assert prev < cur, "red set must strictly grow"
        added = list(cur - prev)
        assert len(added) == 1, "exactly one red cell must be added each frame"
        ar, ac = added[0]
        assert any((ar, ac) == nb for cell in prev for nb in _neighbors4(*cell)), (
            "new red cell must touch the previous red region"
        )
        sequence.append(added[0])

    neighbors = _path_neighbors(mask)
    degrees = sorted(len(v) for v in neighbors.values())
    assert degrees[:2] == [1, 1] and all(d in {1, 2} for d in degrees), (
        "sprite must be a simple unbranched path"
    )
    endpoints = [coord for coord, neigh in neighbors.items() if len(neigh) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    matches = []
    for endpoint in endpoints:
        order = _walk_path(neighbors, endpoint)
        if len(order) != int(mask.sum()):
            continue
        if order[:4] != sequence:
            continue
        if all(red_sets[i] == set(order[: i + 1]) for i in range(4)):
            matches.append(order)

    assert len(matches) == 1, "red growth order must be uniquely determined"
    order = matches[0]
    return order[4]


def _predict_fifth_panel_from_frames(panels: list[np.ndarray], bg: int) -> np.ndarray:
    frames = [_extract_frame(panel, bg) for panel in panels]
    masks = [frame["mask"] for frame in frames]
    assert all(np.array_equal(mask, masks[0]) for mask in masks[1:]), (
        "sprite shape must stay fixed in local coordinates"
    )

    anchors = [frame["anchor"] for frame in frames]
    deltas = [
        (anchors[i + 1][0] - anchors[i][0], anchors[i + 1][1] - anchors[i][1])
        for i in range(3)
    ]
    assert len(set(deltas)) == 1, "sprite translation must be constant"
    dr, dc = deltas[0]
    assert (dr, dc) != (0, 0), "translation must be non-zero"

    next_local = _infer_next_red(masks[0], [frame["red"] for frame in frames])

    out = _translate_non_bg(panels[-1], dr, dc, bg)
    anchor5 = (anchors[-1][0] + dr, anchors[-1][1] + dc)
    rr = anchor5[0] + next_local[0]
    cc = anchor5[1] + next_local[1]
    assert out[rr, cc] == BLUE, "the newly colored cell must belong to the translated sprite"
    out[rr, cc] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the fifth panel by combining a global sprite translation with a local red-growth track.

    Input structure:
    - The grid is an animation strip of four equal panels laid out horizontally or vertically.
    - Adjacent panels are separated by solid yellow rows or columns, and panel backgrounds are black.
    - Each panel contains one connected blue/red sprite whose occupied shape stays the same while its panel position shifts.

    Rule:
    - Split the strip into four panels using the separator rows or columns.
    - In each panel, read the sprite position from the top-left corner of its bounding box and verify a constant per-frame translation.
    - In sprite-local coordinates, track the red cells: they must form a growing prefix of a single unbranched 4-connected path, adding exactly one adjacent cell per frame.
    - Output one panel: translate the fourth-frame sprite by the same offset, then extend the red prefix by one more cell along that same local path; keep all other sprite cells blue and the background black.

    Key insight:
    - The position pattern lives in global panel coordinates, while the color-growth pattern lives inside the sprite's local coordinates; the answer requires inferring both tracks separately and merging them in the new frame.
    """
    g = np.asarray(grid, dtype=int)
    panels, bg = _split_panels(g)
    assert len(panels) == 4, "expected exactly four observed panels"
    return _predict_fifth_panel_from_frames(panels, bg)


def _normalize_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in path)
    c0 = min(c for _, c in path)
    return [(r - r0, c - c0) for r, c in path]


def _has_turn(path: list[tuple[int, int]]) -> bool:
    dirs = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dirs.append((r2 - r1, c2 - c1))
    return any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _generate_induced_path(
    canvas_h: int,
    canvas_w: int,
    length: int,
    require_turn: bool,
) -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        start = (random.randrange(canvas_h), random.randrange(canvas_w))
        path = [start]
        used = {start}

        def dfs() -> bool:
            if len(path) == length:
                return True
            cur = path[-1]
            options = _neighbors4(*cur)
            random.shuffle(options)
            for nr, nc in options:
                if not (0 <= nr < canvas_h and 0 <= nc < canvas_w):
                    continue
                nxt = (nr, nc)
                if nxt in used:
                    continue
                bad = False
                for ar, ac in _neighbors4(nr, nc):
                    if (ar, ac) in used and (ar, ac) != cur:
                        bad = True
                        break
                if bad:
                    continue
                used.add(nxt)
                path.append(nxt)
                if dfs():
                    return True
                path.pop()
                used.remove(nxt)
            return False

        if not dfs():
            continue

        norm = _normalize_path(path)
        mask = np.zeros(
            (max(r for r, _ in norm) + 1, max(c for _, c in norm) + 1),
            dtype=bool,
        )
        for r, c in norm:
            mask[r, c] = True
        if not _connected(mask):
            continue
        if require_turn and not _has_turn(norm):
            continue
        return norm
    return None


def _build_panel(
    panel_h: int,
    panel_w: int,
    anchor: tuple[int, int],
    path: list[tuple[int, int]],
    red_count: int,
) -> np.ndarray:
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    ar, ac = anchor
    for idx, (r, c) in enumerate(path):
        panel[ar + r, ac + c] = RED if idx < red_count else BLUE
    return panel


def _stack_input(panels: list[np.ndarray], horizontal_strip: bool) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    if horizontal_strip:
        out = np.full((panel_h, 4 * panel_w + 3), BG, dtype=int)
        for i, panel in enumerate(panels):
            c0 = i * (panel_w + 1)
            out[:, c0 : c0 + panel_w] = panel
            if i < 3:
                out[:, c0 + panel_w] = SEP
        return out

    out = np.full((4 * panel_h + 3, panel_w), BG, dtype=int)
    for i, panel in enumerate(panels):
        r0 = i * (panel_h + 1)
        out[r0 : r0 + panel_h, :] = panel
        if i < 3:
            out[r0 + panel_h, :] = SEP
    return out


def _full_sequence_semantics(frames: list[np.ndarray]) -> None:
    assert len(frames) == 5
    parsed = [_extract_frame(frame, BG) for frame in frames]
    mask = parsed[0]["mask"]
    assert int(mask.sum()) >= 6
    assert all(np.array_equal(item["mask"], mask) for item in parsed[1:])
    anchors = [item["anchor"] for item in parsed]
    deltas = [
        (anchors[i + 1][0] - anchors[i][0], anchors[i + 1][1] - anchors[i][1])
        for i in range(4)
    ]
    assert len(set(deltas)) == 1 and deltas[0] != (0, 0)
    red_sets = [item["red"] for item in parsed]
    assert [len(rs) for rs in red_sets] == [1, 2, 3, 4, 5]
    next_local = _infer_next_red(mask, red_sets[:4])
    frame5_added = list(red_sets[4] - red_sets[3])
    assert len(frame5_added) == 1 and frame5_added[0] == next_local
    assert len(parsed[4]["blue"]) >= 1, "frame 5 must still leave blue cells"


def _choose_anchor(
    panel_h: int,
    panel_w: int,
    local_h: int,
    local_w: int,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    r_low = max(0, -4 * dr)
    r_high = min(panel_h - local_h, panel_h - local_h - 4 * dr)
    c_low = max(0, -4 * dc)
    c_high = min(panel_w - local_w, panel_w - local_w - 4 * dc)
    if r_low > r_high or c_low > c_high:
        return None
    return random.randint(r_low, r_high), random.randint(c_low, c_high)


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    move_vertical = random.random() < 0.5
    horizontal_strip = random.random() < 0.5
    require_turn = random.random() < 0.7

    if move_vertical:
        panel_h = 6
        panel_w = random.choice([5, 6])
        dr, dc = random.choice([-1, 1]), 0
        canvas_h = 2
        canvas_w = random.randint(4, panel_w)
        max_len = min(7, canvas_h * canvas_w)
        if max_len < 6:
            return None
        length = random.randint(6, max_len)
        path = _generate_induced_path(canvas_h, canvas_w, length, require_turn)
        if path is None:
            return None
        local_h = max(r for r, _ in path) + 1
        local_w = max(c for _, c in path) + 1
        if local_h != 2 or local_w < 3:
            return None
    else:
        panel_h = random.choice([5, 6])
        panel_w = 6
        dr, dc = 0, random.choice([-1, 1])
        canvas_h = random.randint(4, panel_h)
        canvas_w = 2
        max_len = min(7, canvas_h * canvas_w)
        if max_len < 6:
            return None
        length = random.randint(6, max_len)
        path = _generate_induced_path(canvas_h, canvas_w, length, require_turn)
        if path is None:
            return None
        local_h = max(r for r, _ in path) + 1
        local_w = max(c for _, c in path) + 1
        if local_w != 2 or local_h < 3:
            return None

    if random.random() < 0.5:
        path = list(reversed(path))

    anchor1 = _choose_anchor(panel_h, panel_w, local_h, local_w, dr, dc)
    if anchor1 is None:
        return None

    frames = []
    for step in range(5):
        anchor = (anchor1[0] + step * dr, anchor1[1] + step * dc)
        frames.append(_build_panel(panel_h, panel_w, anchor, path, step + 1))

    _full_sequence_semantics(frames)

    inp = _stack_input(frames[:4], horizontal_strip=horizontal_strip)
    out = frames[4]
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 4")
