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

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
VALVE_COLORS = (RED, BLUE, GREEN)
CONTAINER_COLORS = (6, 7, 8, 9, 1, 2, 3)
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _axis_allowed(color: int, dr: int, dc: int) -> bool:
    if color == RED:
        return dc != 0
    if color == BLUE:
        return dr != 0
    if color == GREEN:
        return True
    return False


def _detect_containers(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    container_of = np.full((h, w), -1, dtype=int)
    containers: list[dict[str, object]] = []
    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]

    for color in map(int, np.unique(g)):
        if color in (BLACK, YELLOW, GREY):
            continue
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            area = int(comp.shape[0])
            if hh < 3 or ww < 3:
                continue

            marker = None
            if area == hh * ww:
                ok = True
            elif area == hh * ww - 1:
                missing = []
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if int(g[r, c]) != color:
                            missing.append((r, c))
                ok = len(missing) == 1
                if ok:
                    marker = missing[0]
                    ok = (
                        int(g[marker]) == YELLOW
                        and marker in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
                    )
            else:
                ok = False

            if not ok:
                continue

            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if marker is not None and (r, c) == marker:
                        if int(g[r, c]) != YELLOW:
                            ok = False
                        continue
                    if int(g[r, c]) != color:
                        ok = False
            if not ok:
                continue

            cid = len(containers)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    assert container_of[r, c] == -1, "containers must not overlap"
                    container_of[r, c] = cid

            containers.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "marker": marker,
                }
            )

    source_ids = [i for i, obj in enumerate(containers) if obj["marker"] is not None]
    assert len(source_ids) == 1, "expected exactly one source container"
    assert len(yellow_cells) == 1, "expected exactly one yellow marker"
    assert containers[source_ids[0]]["marker"] == yellow_cells[0]
    return containers, source_ids[0], container_of


def _pipe_masks(grid: np.ndarray, container_of: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    raw_pipe = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if container_of[r, c] != -1:
                continue
            if int(g[r, c]) in (GREY, RED, BLUE, GREEN):
                raw_pipe[r, c] = True

    valves: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if container_of[r, c] != -1 or int(g[r, c]) not in VALVE_COLORS:
                continue
            left = c - 1 >= 0 and raw_pipe[r, c - 1]
            right = c + 1 < w and raw_pipe[r, c + 1]
            up = r - 1 >= 0 and raw_pipe[r - 1, c]
            down = r + 1 < h and raw_pipe[r + 1, c]
            if left and right and up and down:
                valves.add((r, c))

    pipe_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if container_of[r, c] != -1:
                continue
            if int(g[r, c]) == GREY or (r, c) in valves:
                pipe_mask[r, c] = True
    return pipe_mask, valves


def _container_contacts(containers: list[dict[str, object]], pipe_mask: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    h, w = pipe_mask.shape
    contacts: list[list[tuple[int, int, int, int]]] = []
    for obj in containers:
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[assignment]
        seen: set[tuple[int, int, int, int]] = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if _inside((h, w), nr, nc) and pipe_mask[nr, nc]:
                        seen.add((nr, nc, dr, dc))
        contacts.append(sorted(seen))
    return contacts


def _reachable_containers(
    grid: np.ndarray,
    containers: list[dict[str, object]],
    source_id: int,
    container_of: np.ndarray,
    pipe_mask: np.ndarray,
    valves: set[tuple[int, int]],
    contacts: list[list[tuple[int, int, int, int]]],
) -> set[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    reached = {source_id}
    seen_states: set[tuple[int, int, int, int]] = set()
    q: deque[tuple[int, int, int, int]] = deque()

    def seed(container_id: int) -> None:
        for pr, pc, dr, dc in contacts[container_id]:
            if (pr, pc) in valves and not _axis_allowed(int(g[pr, pc]), dr, dc):
                continue
            q.append((pr, pc, dr, dc))

    seed(source_id)

    while q:
        r, c, dr, dc = q.popleft()
        state = (r, c, dr, dc)
        if state in seen_states or not pipe_mask[r, c]:
            continue
        if (r, c) in valves and not _axis_allowed(int(g[r, c]), dr, dc):
            continue
        seen_states.add(state)

        next_dirs = [(dr, dc)] if (r, c) in valves else DIRS4
        for ndr, ndc in next_dirs:
            nr, nc = r + ndr, c + ndc
            if not _inside((h, w), nr, nc):
                continue
            other = int(container_of[nr, nc])
            if other != -1:
                if other not in reached:
                    reached.add(other)
                    seed(other)
                continue
            if not pipe_mask[nr, nc]:
                continue
            if (nr, nc) in valves and not _axis_allowed(int(g[nr, nc]), ndr, ndc):
                continue
            q.append((nr, nc, ndr, ndc))

    return reached


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source container color through a grey pipe network, with valve colors
    filtering which axis may pass through each full cross intersection.
    """
    g = np.asarray(grid, dtype=int)
    containers, source_id, container_of = _detect_containers(g)
    pipe_mask, valves = _pipe_masks(g, container_of)
    contacts = _container_contacts(containers, pipe_mask)
    reached = _reachable_containers(g, containers, source_id, container_of, pipe_mask, valves, contacts)

    source_color = int(containers[source_id]["color"])
    out = g.copy()
    for cid in reached:
        r0, c0, r1, c1 = containers[cid]["bbox"]  # type: ignore[assignment]
        out[r0 : r1 + 1, c0 : c1 + 1] = source_color
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _anchored_interval(anchor: int, length: int, min_start: int, max_end: int) -> tuple[int, int] | None:
    lo = max(min_start, anchor - (length - 2))
    hi = min(max_end - length + 1, anchor - 1)
    if lo > hi:
        return None
    start = random.randint(lo, hi)
    return start, start + length - 1


def _draw_container(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    marker: tuple[int, int] | None = None,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    if marker is not None:
        grid[marker] = YELLOW


def _add_horizontal(pipe_axes: dict[tuple[int, int], set[str]], row: int, c0: int, c1: int) -> None:
    for c in range(min(c0, c1), max(c0, c1) + 1):
        pipe_axes.setdefault((row, c), set()).add("h")


def _add_vertical(pipe_axes: dict[tuple[int, int], set[str]], col: int, r0: int, r1: int) -> None:
    for r in range(min(r0, r1), max(r0, r1) + 1):
        pipe_axes.setdefault((r, col), set()).add("v")


def _pick_container_box(
    h: int,
    w: int,
    boxes: list[tuple[int, int, int, int]],
    row_mode: tuple[str, int],
    col_mode: tuple[str, int],
    box_h: int,
    box_w: int,
) -> tuple[int, int, int, int] | None:
    if row_mode[0] == "around":
        row_iv = _anchored_interval(row_mode[1], box_h, 1, h - 2)
    elif row_mode[0] == "top":
        max_r0 = row_mode[1] - box_h
        if max_r0 < 1:
            return None
        row_iv = (random.randint(1, max_r0), 0)
        row_iv = (row_iv[0], row_iv[0] + box_h - 1)
    else:
        min_r0 = row_mode[1] + 1
        max_r0 = h - box_h - 1
        if min_r0 > max_r0:
            return None
        row_iv = (random.randint(min_r0, max_r0), 0)
        row_iv = (row_iv[0], row_iv[0] + box_h - 1)

    if col_mode[0] == "around":
        col_iv = _anchored_interval(col_mode[1], box_w, 1, w - 2)
    elif col_mode[0] == "left":
        if col_mode[1] + box_w - 1 > w - 2:
            return None
        col_iv = (col_mode[1], col_mode[1] + box_w - 1)
    else:
        c1 = col_mode[1]
        c0 = c1 - box_w + 1
        if c0 < 1:
            return None
        col_iv = (c0, c1)

    box = (row_iv[0], col_iv[0], row_iv[1], col_iv[1])
    if any(_boxes_overlap(box, other, pad=1) for other in boxes):
        return None
    return box


def _render_pipes(
    grid: np.ndarray,
    pipe_axes: dict[tuple[int, int], set[str]],
    valve_colors: dict[tuple[int, int], int],
) -> None:
    h, w = grid.shape
    for (r, c), axes in pipe_axes.items():
        if (r, c) in valve_colors:
            assert "h" in axes and "v" in axes
            assert _inside((h, w), r - 1, c) and _inside((h, w), r + 1, c)
            assert _inside((h, w), r, c - 1) and _inside((h, w), r, c + 1)
            grid[r, c] = valve_colors[(r, c)]
        else:
            grid[r, c] = GREY


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(28, 36)
    trunk_row = random.randint(7, h - 8)
    valve_count = random.choice([2, 3])

    valve_cols = []
    for _ in range(40):
        pick = sorted(random.sample(list(range(8, w - 8)), valve_count))
        if min(b - a for a, b in zip(pick, pick[1:])) >= 5:
            valve_cols = pick
            break
    if not valve_cols:
        return None
    for _ in range(40):
        colors = random.sample(list(CONTAINER_COLORS), 3 + valve_count)
        source_color = colors[0]
        far_color = colors[1]
        local_color = colors[2]
        branch_colors = colors[3:]

        source_h, source_w = random.randint(3, 5), random.randint(3, 5)
        far_h, far_w = random.randint(3, 5), random.randint(3, 5)
        source_box = _pick_container_box(
            h,
            w,
            [],
            ("around", trunk_row),
            ("left", 1),
            source_h,
            source_w,
        )
        far_box = _pick_container_box(
            h,
            w,
            [source_box] if source_box is not None else [],
            ("around", trunk_row),
            ("right", w - 2),
            far_h,
            far_w,
        )
        if source_box is None or far_box is None:
            continue

        sr0, sc0, sr1, sc1 = source_box
        fr0, fc0, fr1, fc1 = far_box
        if sc1 + 6 >= valve_cols[0] or valve_cols[-1] + 6 >= fc0:
            continue

        boxes = [source_box, far_box]
        container_specs: list[tuple[tuple[int, int, int, int], int, tuple[int, int] | None]] = []
        source_corners = [(sr0, sc0), (sr0, sc1), (sr1, sc0), (sr1, sc1)]
        container_specs.append((source_box, source_color, random.choice(source_corners)))
        container_specs.append((far_box, far_color, None))

        local_col_lo = sc1 + 3
        local_col_hi = valve_cols[0] - 3
        if local_col_lo > local_col_hi:
            continue
        local_attach = random.randint(local_col_lo, local_col_hi)
        local_side = random.choice(["top", "bottom"])
        local_box = _pick_container_box(
            h,
            w,
            boxes,
            ("top", trunk_row - 1) if local_side == "top" else ("bottom", trunk_row + 1),
            ("around", local_attach),
            random.randint(3, 4),
            random.randint(3, 4),
        )
        if local_box is None:
            continue
        boxes.append(local_box)
        container_specs.append((local_box, local_color, None))

        branch_specs: list[tuple[tuple[int, int, int, int], int, int, str]] = []
        ok = True
        for col, color in zip(valve_cols, branch_colors, strict=True):
            side = random.choice(["top", "bottom"])
            branch_box = _pick_container_box(
                h,
                w,
                boxes,
                ("top", trunk_row - 1) if side == "top" else ("bottom", trunk_row + 1),
                ("around", col),
                random.randint(3, 4),
                random.randint(3, 4),
            )
            if branch_box is None:
                ok = False
                break
            boxes.append(branch_box)
            branch_specs.append((branch_box, color, col, side))
            container_specs.append((branch_box, color, None))
        if ok:
            break
    else:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for box, color, marker in container_specs:
        _draw_container(grid, box, color, marker)

    pipe_axes: dict[tuple[int, int], set[str]] = {}
    _add_horizontal(pipe_axes, trunk_row, sc1 + 1, fc0 - 1)

    if local_side == "top":
        lr0, lc0, lr1, lc1 = local_box
        assert lc0 < local_attach < lc1
        _add_vertical(pipe_axes, local_attach, lr1 + 1, trunk_row)
    else:
        lr0, lc0, lr1, lc1 = local_box
        assert lc0 < local_attach < lc1
        _add_vertical(pipe_axes, local_attach, trunk_row, lr0 - 1)

    for branch_box, _color, col, side in branch_specs:
        br0, bc0, br1, bc1 = branch_box
        assert bc0 < col < bc1
        if side == "top":
            _add_vertical(pipe_axes, col, br1 + 1, trunk_row)
        else:
            _add_vertical(pipe_axes, col, trunk_row, br0 - 1)

    valve_palette = random.sample(list(VALVE_COLORS), valve_count)
    valve_colors = {(trunk_row, col): color for col, color in zip(valve_cols, valve_palette, strict=True)}

    for (r, c) in pipe_axes:
        if grid[r, c] != BLACK:
            return None
    _render_pipes(grid, pipe_axes, valve_colors)

    try:
        containers, source_id, _ = _detect_containers(grid)
        out = main(grid)
    except Exception:
        return None

    source_color = int(containers[source_id]["color"])
    changed_targets = 0
    unchanged_targets = 0
    for cid, obj in enumerate(containers):
        if cid == source_id:
            continue
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[assignment]
        patch_in = grid[r0 : r1 + 1, c0 : c1 + 1]
        patch_out = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch_out == source_color):
            changed_targets += 1
            if np.array_equal(patch_in, patch_out):
                return None
        else:
            unchanged_targets += 1

    if changed_targets < 1 or unchanged_targets < 1:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
