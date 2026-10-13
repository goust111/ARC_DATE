from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

GRID_SIZES = (12, 14, 16)
SPRITE_CELLS = (3, 4, 5, 6)
TARGET_COUNTS = (3, 4, 5)
MAX_GENERATION_ATTEMPTS = 500
_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _expand4(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in tuple(cells):
        for nr, nc in _neighbors4(r, c):
            expanded.add((nr, nc))
    return expanded


def _sort_cells(cells) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in cells)


def _delta_signature(deltas: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(dr), int(dc)) for dr, dc in deltas))


def _mirror_deltas(
    deltas: list[tuple[int, int]],
    *,
    mirror_lr: bool,
    mirror_ud: bool,
) -> list[tuple[int, int]]:
    out = []
    for dr, dc in deltas:
        nr = -dr if mirror_ud else dr
        nc = -dc if mirror_lr else dc
        out.append((int(nr), int(nc)))
    return _sort_cells(out)


def _extract_source(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    red_mask = grid == RED
    red_comps = _component_coords(red_mask)
    assert len(red_comps) == 1, "expected exactly one red sprite component"
    sprite = red_comps[0]
    assert 3 <= len(sprite) <= 6, "sprite size must be 3..6"

    blue_adjacent: set[tuple[int, int]] = set()
    for r, c in sprite:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLUE:
                blue_adjacent.add((nr, nc))
    assert len(blue_adjacent) == 1, "expected exactly one blue anchor adjacent to the sprite"

    anchor = next(iter(blue_adjacent))
    deltas = _sort_cells((int(r) - anchor[0], int(c) - anchor[1]) for r, c in sprite)
    return anchor, deltas


def _output_from_parts(
    shape: tuple[int, int],
    targets: list[tuple[int, int]],
    transformed_deltas: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for tr, tc in sorted(targets):
        for dr, dc in transformed_deltas:
            rr = tr + dr
            cc = tc + dc
            assert _in_bounds(shape, rr, cc), "stamp must stay in bounds"
            assert int(out[rr, cc]) == BLACK, "stamps must not overlap or cover a marker"
            out[rr, cc] = RED
        assert int(out[tr, tc]) == BLACK, "target marker must not lie inside a red footprint"
        out[tr, tc] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the anchored red sprite according to the yellow control pixel's quadrant, then stamp it at every blue target marker.

    Input structure:
    - The background is black.
    - One red 4-connected sprite has exactly one blue anchor pixel that is 4-neighbor adjacent to it.
    - One yellow control pixel is isolated from the other foreground.
    - The remaining blue pixels are standalone target markers.

    Rule:
    - Identify the red sprite and measure every red cell relative to its adjacent blue anchor.
    - If the yellow control is in the left half, reflect those anchor-relative offsets left-right; if it is in the upper half, reflect them top-bottom; both reflections may apply.
    - Clear the grid to black and stamp the transformed red sprite at every blue target marker using that marker as the anchor, keeping each target pixel blue.

    Key insight:
    - The yellow pixel does not select a destination; its quadrant independently encodes the horizontal and vertical mirror flags for the sprite before any stamping happens.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    control_cells = np.argwhere(g == YELLOW)
    assert len(control_cells) == 1, "expected exactly one yellow control pixel"
    control_r, control_c = map(int, control_cells[0])

    anchor, sprite_deltas = _extract_source(g)
    all_blue = _sort_cells(np.argwhere(g == BLUE))
    targets = [cell for cell in all_blue if cell != anchor]
    assert 3 <= len(targets) <= 5, "expected 3..5 blue targets"

    mirror_lr = control_c < (g.shape[1] / 2.0)
    mirror_ud = control_r < (g.shape[0] / 2.0)
    transformed = _mirror_deltas(sprite_deltas, mirror_lr=mirror_lr, mirror_ud=mirror_ud)
    return _output_from_parts(g.shape, targets, transformed)


def _random_connected_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.choice(SPRITE_CELLS)
        if target > h * w:
            continue

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if min(mask.shape) < 2:
            continue
        if np.count_nonzero(mask) != target:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        variants = {
            mask.tobytes(),
            np.fliplr(mask).tobytes(),
            np.flipud(mask).tobytes(),
            np.flipud(np.fliplr(mask)).tobytes(),
        }
        if len(variants) != 4:
            continue
        if len(_component_coords(mask)) != 1:
            continue
        return mask

    raise RuntimeError("failed to generate a non-symmetric sprite")


def _anchor_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates = set()
    for c in range(w):
        if mask[0, c]:
            candidates.add((-1, c))
        if mask[h - 1, c]:
            candidates.add((h, c))
    for r in range(h):
        if mask[r, 0]:
            candidates.add((r, -1))
        if mask[r, w - 1]:
            candidates.add((r, w))
    return _sort_cells(candidates)


def _mask_to_anchor_deltas(mask: np.ndarray, anchor: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = anchor
    coords = np.argwhere(mask)
    return _sort_cells((int(r) - ar, int(c) - ac) for r, c in coords)


def _choose_sprite() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    for _ in range(300):
        mask = _random_connected_mask()
        anchor = random.choice(_anchor_candidates(mask))
        deltas = _mask_to_anchor_deltas(mask, anchor)
        variants = {
            _delta_signature(_mirror_deltas(deltas, mirror_lr=lr, mirror_ud=ud))
            for lr in (False, True)
            for ud in (False, True)
        }
        if len(variants) != 4:
            continue
        return deltas, list(variants)
    raise RuntimeError("failed to generate a sprite-anchor pair")


def _pick_targets(
    shape: tuple[int, int],
    transformed_deltas: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    footprint_offsets = {(0, 0), *transformed_deltas}
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    h, w = shape
    for r in range(h):
        for c in range(w):
            footprint = {(r + dr, c + dc) for dr, dc in footprint_offsets}
            if all(_in_bounds(shape, rr, cc) for rr, cc in footprint):
                candidates.append(((r, c), footprint))

    for _ in range(200):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        occupied_exact: set[tuple[int, int]] = set()
        occupied_margin: set[tuple[int, int]] = set()
        for target, footprint in candidates:
            if footprint & occupied_margin:
                continue
            chosen.append(target)
            occupied_exact |= footprint
            occupied_margin = _expand4(occupied_exact)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _place_source(
    shape: tuple[int, int],
    source_deltas: list[tuple[int, int]],
    targets: list[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    target_forbidden = _expand4(set(targets))
    h, w = shape
    anchors = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(anchors)

    for ar, ac in anchors:
        red_cells = {(ar + dr, ac + dc) for dr, dc in source_deltas}
        if not all(_in_bounds(shape, rr, cc) for rr, cc in red_cells):
            continue
        footprint = set(red_cells)
        footprint.add((ar, ac))
        if footprint & target_forbidden:
            continue
        if set(targets) & red_cells:
            continue
        return (ar, ac), red_cells
    return None


def _choose_control(
    shape: tuple[int, int],
    nonzero_cells: set[tuple[int, int]],
    blue_cells: set[tuple[int, int]],
    *,
    mirror_lr: bool,
    mirror_ud: bool,
) -> tuple[int, int] | None:
    h, w = shape
    row_cut = h / 2.0
    col_cut = w / 2.0
    candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in nonzero_cells:
                continue
            if mirror_ud != (r < row_cut):
                continue
            if mirror_lr != (c < col_cut):
                continue
            if any((nr, nc) in nonzero_cells for nr, nc in _neighbors4(r, c)):
                continue
            if any(r == br or c == bc for br, bc in blue_cells):
                continue
            candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _variant_outputs(
    shape: tuple[int, int],
    targets: list[tuple[int, int]],
    source_deltas: list[tuple[int, int]],
) -> set[bytes]:
    outs = set()
    for mirror_lr in (False, True):
        for mirror_ud in (False, True):
            transformed = _mirror_deltas(source_deltas, mirror_lr=mirror_lr, mirror_ud=mirror_ud)
            out = _output_from_parts(shape, targets, transformed)
            outs.add(out.tobytes() + str(out.shape).encode())
    return outs


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, BLUE, RED, YELLOW}):
        return False

    try:
        control = np.argwhere(grid == YELLOW)
        if len(control) != 1:
            return False
        control_r, control_c = map(int, control[0])

        anchor, source_deltas = _extract_source(grid)
        blue_cells = _sort_cells(np.argwhere(grid == BLUE))
        if len(blue_cells) != 4 and len(blue_cells) != 5 and len(blue_cells) != 6:
            return False
        targets = [cell for cell in blue_cells if cell != anchor]
        if len(targets) not in TARGET_COUNTS:
            return False

        red_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == RED)}
        if any(any((nr, nc) in red_cells for nr, nc in _neighbors4(tr, tc)) for tr, tc in targets):
            return False
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for i, a in enumerate(targets) for b in targets[i + 1 :]):
            return False
        other_blue = set(targets)
        if any(
            (nr, nc) in red_cells or (nr, nc) in other_blue or (nr, nc) == anchor
            for nr, nc in _neighbors4(control_r, control_c)
        ):
            return False
        if any(control_r == br or control_c == bc for br, bc in blue_cells):
            return False

        variant_sigs = {
            _delta_signature(_mirror_deltas(source_deltas, mirror_lr=lr, mirror_ud=ud))
            for lr in (False, True)
            for ud in (False, True)
        }
        if len(variant_sigs) != 4:
            return False
        if len(_variant_outputs(grid.shape, targets, source_deltas)) != 4:
            return False

        out = main(grid)
        if out.shape != grid.shape or max(out.shape) > 30:
            return False
        if set(map(int, np.unique(out))) - {BLACK, BLUE, RED}:
            return False
        if np.count_nonzero(out == BLUE) != len(targets):
            return False

        mirror_lr = control_c < (grid.shape[1] / 2.0)
        mirror_ud = control_r < (grid.shape[0] / 2.0)
        expected = _output_from_parts(grid.shape, targets, _mirror_deltas(source_deltas, mirror_lr=mirror_lr, mirror_ud=mirror_ud))
        return np.array_equal(out, expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    shape = (random.choice(GRID_SIZES), random.choice(GRID_SIZES))
    source_deltas, _ = _choose_sprite()
    mirror_lr = bool(random.getrandbits(1))
    mirror_ud = bool(random.getrandbits(1))
    transformed_deltas = _mirror_deltas(source_deltas, mirror_lr=mirror_lr, mirror_ud=mirror_ud)

    target_count = random.choice(TARGET_COUNTS)
    targets = _pick_targets(shape, transformed_deltas, target_count)
    if targets is None:
        return None

    source_placement = _place_source(shape, source_deltas, targets)
    if source_placement is None:
        return None
    anchor, red_cells = source_placement

    blue_cells = set(targets)
    blue_cells.add(anchor)
    nonzero_cells = set(red_cells) | blue_cells
    control = _choose_control(shape, nonzero_cells, blue_cells, mirror_lr=mirror_lr, mirror_ud=mirror_ud)
    if control is None:
        return None

    grid = np.zeros(shape, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for r, c in blue_cells:
        grid[r, c] = BLUE
    grid[control] = YELLOW

    if not _validate_instance(grid):
        return None

    inp_sig = grid.tobytes() + str(grid.shape).encode()
    out = main(grid)
    out_sig = out.tobytes() + str(out.shape).encode()
    if inp_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
        return None

    _SEEN_INPUT_SIGS.add(inp_sig)
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-controlled mirror stamp instance")
