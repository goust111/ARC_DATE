from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
LEGEND_COMMANDS = (GREEN, RED, BLUE)
MARKER_POOL = (4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components4(mask)) == 1


def _is_marker_interior(mask: np.ndarray, marker: tuple[int, int]) -> bool:
    r, c = marker
    if not mask[r, c]:
        return False
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]):
            return False
    return True


def _trim_mask(mask: np.ndarray, marker: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    trimmed = mask[r0 : r1 + 1, c0 : c1 + 1]
    mr, mc = marker
    return trimmed, (int(mr - r0), int(mc - c0))


def _mirror_mask_horiz(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-left 3x2 legend to decide whether each grey object is kept,
    deleted, or mirrored horizontally, then erase all object markers to grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 2

    legend_left = g[:3, 0]
    legend_right = g[:3, 1]
    assert np.array_equal(legend_right, np.array(LEGEND_COMMANDS, dtype=int))
    assert len(set(map(int, legend_left))) == 3
    marker_to_command = {int(marker): int(command) for marker, command in zip(legend_left, legend_right)}

    work_mask = g != BACKGROUND
    work_mask[:3, :2] = False
    components = _components4(work_mask)
    assert components, "expected at least one foreground object"

    out = g.copy()
    for coords in components:
        colors = {int(g[r, c]) for r, c in coords}
        marker_cells = [(int(r), int(c)) for r, c in coords if int(g[r, c]) != GREY]
        assert len(marker_cells) == 1, "each object must contain exactly one non-grey marker pixel"
        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])
        assert marker_color in marker_to_command

        assert colors <= {GREY, marker_color}, "object may contain only grey plus its marker color"
        command = marker_to_command[marker_color]

        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        c0 = int(cols.min())
        c1 = int(cols.max())

        if command == RED:
            for r, c in coords:
                out[int(r), int(c)] = BACKGROUND
        elif command == GREEN:
            for r, c in coords:
                out[int(r), int(c)] = GREY
        else:
            assert command == BLUE
            for r, c in coords:
                out[int(r), int(c)] = BACKGROUND
            for r, c in coords:
                rr = int(r)
                cc = c0 + c1 - int(c)
                out[rr, cc] = GREY

    return out


GRID_HEIGHT_RANGE = (15, 22)
GRID_WIDTH_RANGE = (15, 22)
OBJECT_COUNT_RANGE = (3, 6)
OBJECT_HEIGHT_RANGE = (3, 7)
OBJECT_WIDTH_RANGE = (3, 7)
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200


def _branch_multiset(count: int) -> list[str]:
    branches = ["keep", "delete", "mirror"]
    extras = count - 3
    start = random.randrange(3)
    for i in range(extras):
        branches.append(("keep", "delete", "mirror")[(start + i) % 3])
    random.shuffle(branches)
    return branches


def _make_object_mask(require_asymmetry: bool) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(OBJECT_ATTEMPTS):
        height = random.randint(*OBJECT_HEIGHT_RANGE)
        width = random.randint(*OBJECT_WIDTH_RANGE)
        mask = np.ones((height, width), dtype=bool)

        marker = (
            random.randint(1, height - 2),
            random.randint(1, width - 2),
        )
        protected = {
            marker,
            (marker[0] - 1, marker[1]),
            (marker[0] + 1, marker[1]),
            (marker[0], marker[1] - 1),
            (marker[0], marker[1] + 1),
        }
        boundary = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r in {0, height - 1} or c in {0, width - 1}) and (r, c) not in protected
        ]
        random.shuffle(boundary)
        max_remove = min(len(boundary), max(0, int(round(len(boundary) * 0.65))))
        target_remove = random.randint(0, max_remove)
        removed = 0
        for r, c in boundary:
            if removed >= target_remove:
                break
            trial = mask.copy()
            trial[r, c] = False
            if _is_connected(trial) and _is_marker_interior(trial, marker):
                mask = trial
                removed += 1

        mask, marker = _trim_mask(mask, marker)
        if not _is_marker_interior(mask, marker):
            continue
        if require_asymmetry and np.array_equal(mask, _mirror_mask_horiz(mask)):
            continue
        return mask, marker
    return None


def _can_place(blocked: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc1 = min(w, c0 + width + 1)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + height + 1)
    cc1 = min(w, c0 + width + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _render_object(
    canvas: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
) -> None:
    rows, cols = np.where(mask)
    canvas[top + rows, left + cols] = int(color)


def _render_expected_object(
    canvas: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    branch: str,
) -> None:
    if branch == "delete":
        return
    if branch == "keep":
        _render_object(canvas, top, left, mask, GREY)
        return
    mirrored = _mirror_mask_horiz(mask)
    _render_object(canvas, top, left, mirrored, GREY)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    input_grid = np.full((height, width), BACKGROUND, dtype=int)
    expected = np.full((height, width), BACKGROUND, dtype=int)

    keep_marker, delete_marker, mirror_marker = random.sample(MARKER_POOL, 3)
    legend_markers = (keep_marker, delete_marker, mirror_marker)
    input_grid[:3, 0] = legend_markers
    expected[:3, 0] = legend_markers
    input_grid[:3, 1] = LEGEND_COMMANDS
    expected[:3, 1] = LEGEND_COMMANDS

    blocked = np.zeros((height, width), dtype=bool)
    blocked[:4, :3] = True

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    branches = _branch_multiset(object_count)
    specs = []
    for branch in branches:
        built = _make_object_mask(require_asymmetry=branch == "mirror")
        if built is None:
            return None
        mask, marker = built
        marker_color = {
            "keep": keep_marker,
            "delete": delete_marker,
            "mirror": mirror_marker,
        }[branch]
        specs.append(
            {
                "branch": branch,
                "mask": mask,
                "marker": marker,
                "marker_color": marker_color,
                "area": int(mask.sum()),
            }
        )

    specs.sort(key=lambda spec: spec["area"], reverse=True)
    for spec in specs:
        mask = spec["mask"]
        oh, ow = mask.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, height - oh)
            left = random.randint(0, width - ow)
            if not _can_place(blocked, top, left, oh, ow):
                continue
            spec["top"] = top
            spec["left"] = left
            _mark_blocked(blocked, top, left, oh, ow)
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        mask = spec["mask"]
        marker_r, marker_c = spec["marker"]
        _render_object(input_grid, top, left, mask, GREY)
        input_grid[top + marker_r, left + marker_c] = int(spec["marker_color"])
        _render_expected_object(expected, top, left, mask, str(spec["branch"]))

    actual = main(input_grid)
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
