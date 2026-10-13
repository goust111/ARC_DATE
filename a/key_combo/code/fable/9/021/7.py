from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_SIDE = 30
GEN_ATTEMPTS = 500

Axis = tuple[str, int]  # ("v", col) or ("h", row)


SPRITES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _on_axis(axis: Axis, r: int, c: int) -> bool:
    return (axis[0] == "v" and c == axis[1]) or (axis[0] == "h" and r == axis[1])


def _mirror(axis: Axis, r: int, c: int) -> tuple[int, int]:
    if axis[0] == "v":
        return int(r), int(2 * axis[1] - c)
    return int(2 * axis[1] - r), int(c)


def _candidate_axes(grid: np.ndarray) -> list[Axis]:
    axes: list[Axis] = []
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == GRAY):
            axes.append(("h", int(r)))
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == GRAY):
            axes.append(("v", int(c)))
    return axes


def _axis_consistency_details(grid: np.ndarray, axis: Axis) -> tuple[bool, bool, bool]:
    """Return (passes, has_visible_mismatch, has_out_of_bounds_contradiction)."""
    h, w = grid.shape
    mismatch = False
    out_of_bounds = False

    for r in range(h):
        for c in range(w):
            if _on_axis(axis, r, c) or int(grid[r, c]) == GRAY:
                continue
            mr, mc = _mirror(axis, r, c)
            if not (0 <= mr < h and 0 <= mc < w):
                out_of_bounds = True
                return False, mismatch, out_of_bounds
            other = int(grid[mr, mc])
            if other != GRAY and other != int(grid[r, c]):
                mismatch = True
                return False, mismatch, out_of_bounds
    return True, mismatch, out_of_bounds


def _choose_true_axis(grid: np.ndarray) -> Axis:
    candidates = _candidate_axes(grid)
    assert candidates, "expected at least one full gray candidate axis bar"
    passing = [axis for axis in candidates if _axis_consistency_details(grid, axis)[0]]
    assert len(passing) == 1, "expected a unique true symmetry axis"
    return passing[0]


def _reconstruct_from_true_axis(grid: np.ndarray, axis: Axis) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    pending = {
        (int(r), int(c))
        for r, c in np.argwhere(out == GRAY)
        if not _on_axis(axis, int(r), int(c))
    }

    progressed = True
    while pending and progressed:
        progressed = False
        for r, c in list(pending):
            mr, mc = _mirror(axis, r, c)
            assert 0 <= mr < out.shape[0] and 0 <= mc < out.shape[1], "occluded cell must have a mirrored partner"
            if _on_axis(axis, mr, mc):
                continue
            value = int(out[mr, mc])
            if value != GRAY:
                out[r, c] = value
                pending.remove((r, c))
                progressed = True

    assert not pending, "all fake-axis cells must be recoverable from the true symmetry"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the unique true gray axis bar, remove false gray bars, and restore the hidden pattern by reflection.

    Input structure:
    - A colored pattern on a black background is overlaid by several one-cell-wide gray full rows or full columns.
    - Gray appears only in those candidate axis bars and may cover both foreground and background cells.
    - Exactly one gray bar is the true strict reflection axis once every gray-covered cell is treated as unknown.

    Rule:
    - Enumerate every full gray row or column as a candidate axis and test reflection symmetry while treating all gray cells as unknown.
    - Keep the unique candidate that is consistent with the visible mirrored evidence and discard every other gray bar.
    - Refill each discarded gray cell with the color of its mirror across the true axis, leaving the true axis bar itself gray.

    Key insight:
    - The same gray bars serve two roles in sequence: first they are unknown occluders used to identify the only valid axis, then the false ones become holes that must be repaired by reflecting the surviving pattern across that axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    true_axis = _choose_true_axis(g)
    return _reconstruct_from_true_axis(g, true_axis)


def _random_sprite() -> np.ndarray:
    sprite = SPRITES[random.randrange(len(SPRITES))].copy()
    sprite = np.rot90(sprite, k=random.randrange(4))
    if random.random() < 0.5:
        sprite = np.flipud(sprite)
    if random.random() < 0.5:
        sprite = np.fliplr(sprite)
    return sprite


def _cells_with_margin(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _place_object(
    base: np.ndarray,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    axis: Axis,
    color: int,
    sprite: np.ndarray,
    r0: int,
    c0: int,
) -> bool:
    h, w = sprite.shape
    cells: set[tuple[int, int]] = set()
    for rr in range(h):
        for cc in range(w):
            if not bool(sprite[rr, cc]):
                continue
            r = r0 + rr
            c = c0 + cc
            mr, mc = _mirror(axis, r, c)
            if _on_axis(axis, r, c) or _on_axis(axis, mr, mc):
                return False
            if not (0 <= mr < base.shape[0] and 0 <= mc < base.shape[1]):
                return False
            cells.add((r, c))
            cells.add((mr, mc))

    if not cells:
        return False
    if any(cell in forbidden for cell in cells):
        return False

    for r, c in cells:
        base[r, c] = int(color)
    occupied.update(cells)
    forbidden.update(_cells_with_margin(cells, base.shape))
    return True


def _build_symmetric_base(shape: tuple[int, int], axis: Axis) -> np.ndarray | None:
    h, w = shape
    base = np.full((h, w), BG, dtype=int)
    colors = random.sample(PALETTE, k=random.randint(2, 4))
    target_objects = random.randint(4, 7)
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()

    placed = 0
    attempts = 0
    while placed < target_objects and attempts < 250:
        attempts += 1
        sprite = _random_sprite()
        sh, sw = sprite.shape
        color = random.choice(colors)

        if axis[0] == "v":
            max_c0 = axis[1] - sw - 1
            max_r0 = h - sh
            if max_c0 < 0 or max_r0 < 0:
                return None
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, max_c0)
        else:
            max_r0 = axis[1] - sh - 1
            max_c0 = w - sw
            if max_r0 < 0 or max_c0 < 0:
                return None
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, max_c0)

        if _place_object(base, occupied, forbidden, axis, color, sprite, r0, c0):
            placed += 1

    if placed < 4:
        return None
    if np.count_nonzero(base != BG) < 16:
        return None
    return base


def _axis_line_mask(shape: tuple[int, int], axis: Axis) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if axis[0] == "v":
        mask[:, axis[1]] = True
    else:
        mask[axis[1], :] = True
    return mask


def _choose_fake_parallel_axes(base: np.ndarray, axis: Axis) -> list[Axis] | None:
    if axis[0] == "v":
        usable = [c for c in range(base.shape[1]) if c != axis[1] and np.any(base[:, c] != BG)]
        target = random.randint(1, min(3, max(1, len(usable))))
        random.shuffle(usable)
        chosen: list[Axis] = []
        blocked = {axis[1]}
        for c in usable:
            mc = _mirror(axis, 0, c)[1]
            if c in blocked:
                continue
            chosen.append(("v", int(c)))
            blocked.add(int(c))
            blocked.add(int(mc))
            if len(chosen) == target:
                break
    else:
        usable = [r for r in range(base.shape[0]) if r != axis[1] and np.any(base[r, :] != BG)]
        target = random.randint(1, min(3, max(1, len(usable))))
        random.shuffle(usable)
        chosen = []
        blocked = {axis[1]}
        for r in usable:
            mr = _mirror(axis, r, 0)[0]
            if r in blocked:
                continue
            chosen.append(("h", int(r)))
            blocked.add(int(r))
            blocked.add(int(mr))
            if len(chosen) == target:
                break

    return chosen if chosen else None


def _render_input(base: np.ndarray, true_axis: Axis, fake_axes: list[Axis]) -> np.ndarray:
    inp = np.array(base, copy=True)
    for axis in [true_axis] + fake_axes:
        mask = _axis_line_mask(inp.shape, axis)
        inp[mask] = GRAY
    return inp


def _render_output(base: np.ndarray, true_axis: Axis) -> np.ndarray:
    out = np.array(base, copy=True)
    out[_axis_line_mask(out.shape, true_axis)] = GRAY
    return out


def _restored_colored_cells(inp: np.ndarray, out: np.ndarray) -> int:
    mask = (inp == GRAY) & (out != GRAY) & (out != BG)
    return int(np.count_nonzero(mask))


def _fake_axes_have_visible_mismatch(inp: np.ndarray, fake_axes: list[Axis]) -> bool:
    for axis in fake_axes:
        passes, mismatch, _oob = _axis_consistency_details(inp, axis)
        if passes or not mismatch:
            return False
    return True


def _semantic_ok(inp: np.ndarray, out: np.ndarray, true_axis: Axis, fake_axes: list[Axis]) -> bool:
    if max(inp.shape) > MAX_SIDE or max(out.shape) > MAX_SIDE:
        return False
    if len(_candidate_axes(inp)) != len(fake_axes) + 1:
        return False
    if _choose_true_axis(inp) != true_axis:
        return False
    if not np.array_equal(main(inp), out):
        return False
    if _restored_colored_cells(inp, out) < 3:
        return False
    if np.count_nonzero((inp != BG) & (inp != GRAY)) < 10:
        return False
    return _fake_axes_have_visible_mismatch(inp, fake_axes)


def _gen_attempt(orientation: str) -> np.ndarray | None:
    if orientation == "v":
        h = random.randint(11, 23)
        w = random.randrange(13, 30, 2)
        true_axis: Axis = ("v", w // 2)
    else:
        h = random.randrange(13, 30, 2)
        w = random.randint(11, 23)
        true_axis = ("h", h // 2)

    base = _build_symmetric_base((h, w), true_axis)
    if base is None:
        return None

    fake_axes = _choose_fake_parallel_axes(base, true_axis)
    if not fake_axes:
        return None

    inp = _render_input(base, true_axis, fake_axes)
    out = _render_output(base, true_axis)

    return inp if _semantic_ok(inp, out, true_axis, fake_axes) else None


def generate_input() -> np.ndarray:
    preferred = random.choice(["v", "h"])
    order = [preferred, "h" if preferred == "v" else "v"]
    per_branch = max(1, GEN_ATTEMPTS // 2)

    for orientation in order:
        for _ in range(per_branch):
            grid = _gen_attempt(orientation)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid true-axis-election input")
