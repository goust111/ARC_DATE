from __future__ import annotations

from collections import Counter
import random

import numpy as np


PANEL_SIZE = 9
BLOCK_SIZE = 3
LOGICAL_SIZE = 3
INPUT_HEIGHT = 12
INPUT_WIDTH = 19
SEPARATOR_COL = 9
SEPARATOR_COLOR = 5
SOURCE_SIZE = 3

PANEL_COLORS = (1, 2, 3, 4)
STAMP_COLORS = (6, 8)
BACKGROUND = 0

MAX_GENERATION_ATTEMPTS = 400

BLOCK_ORBITS = {
    "corner": ((0, 0), (0, 2), (2, 0), (2, 2)),
    "vertical": ((0, 1), (2, 1)),
    "horizontal": ((1, 0), (1, 2)),
    "center": ((1, 1),),
}

CORNER_MOTIFS = (
    ((0, 0),),
    ((0, 1),),
    ((0, 2),),
    ((1, 0),),
    ((1, 1),),
    ((2, 0),),
    ((0, 0), (1, 1)),
    ((0, 1), (1, 0)),
    ((0, 2), (1, 1)),
    ((1, 0), (2, 0)),
)

VERTICAL_MOTIFS = (
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 2)),
    ((1, 1), (2, 0), (2, 2)),
)

HORIZONTAL_MOTIFS = (
    ((1, 0),),
    ((1, 1),),
    ((1, 2),),
    ((0, 0), (2, 0)),
    ((0, 1), (2, 1)),
    ((0, 2), (2, 2)),
    ((1, 0), (0, 1), (2, 1)),
    ((1, 1), (0, 2), (2, 2)),
)

CENTER_MOTIFS = (
    ((1, 1),),
    ((0, 1), (2, 1)),
    ((1, 0), (1, 2)),
    ((0, 0), (0, 2), (2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _best_axis(panel: np.ndarray, vertical: bool) -> tuple[int, tuple[int, int], int]:
    h, w = map(int, panel.shape)
    limit = w if vertical else h
    best_axis2 = 0
    best_key: tuple[int, int] | None = None
    tie_count = 0

    for axis2 in range(2 * (limit - 1) + 1):
        mismatches = 0
        pairs = 0
        if vertical:
            for r in range(h):
                for c in range(w):
                    if panel[r, c] == BACKGROUND:
                        continue
                    c2 = axis2 - c
                    if 0 <= c2 < w and panel[r, c2] != BACKGROUND:
                        pairs += 1
                        if int(panel[r, c]) != int(panel[r, c2]):
                            mismatches += 1
        else:
            for r in range(h):
                for c in range(w):
                    if panel[r, c] == BACKGROUND:
                        continue
                    r2 = axis2 - r
                    if 0 <= r2 < h and panel[r2, c] != BACKGROUND:
                        pairs += 1
                        if int(panel[r, c]) != int(panel[r2, c]):
                            mismatches += 1
        key = (mismatches, -pairs)
        if best_key is None or key < best_key:
            best_key = key
            best_axis2 = axis2
            tie_count = 1
        elif key == best_key:
            tie_count += 1

    assert best_key is not None
    return best_axis2, best_key, tie_count


def _repair_panel(panel: np.ndarray) -> np.ndarray:
    out = np.array(panel, dtype=int, copy=True)
    axis_c2 = _best_axis(out, vertical=True)[0]
    axis_r2 = _best_axis(out, vertical=False)[0]
    h, w = map(int, out.shape)

    for _ in range(h * w):
        changed = False
        for r, c in np.argwhere(out == BACKGROUND):
            r = int(r)
            c = int(c)
            r2 = axis_r2 - r
            c2 = axis_c2 - c
            candidates: list[int] = []
            if 0 <= c2 < w and out[r, c2] != BACKGROUND:
                candidates.append(int(out[r, c2]))
            if 0 <= r2 < h and out[r2, c] != BACKGROUND:
                candidates.append(int(out[r2, c]))
            if 0 <= r2 < h and 0 <= c2 < w and out[r2, c2] != BACKGROUND:
                candidates.append(int(out[r2, c2]))
            if candidates:
                out[r, c] = Counter(candidates).most_common(1)[0][0]
                changed = True
        if not changed:
            break
    return out


def _consensus_panel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_fixed = _repair_panel(left)
    right_fixed = _repair_panel(right)
    return np.where((left_fixed == right_fixed) & (left_fixed != BACKGROUND), left_fixed, BACKGROUND)


def _block_mask(panel: np.ndarray) -> np.ndarray:
    mask = np.zeros((LOGICAL_SIZE, LOGICAL_SIZE), dtype=bool)
    for br in range(LOGICAL_SIZE):
        for bc in range(LOGICAL_SIZE):
            block = panel[
                br * BLOCK_SIZE : (br + 1) * BLOCK_SIZE,
                bc * BLOCK_SIZE : (bc + 1) * BLOCK_SIZE,
            ]
            mask[br, bc] = bool(np.any(block != BACKGROUND))
    return mask


def _stamp_output(mask: np.ndarray, stamp: np.ndarray) -> np.ndarray:
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for br in range(LOGICAL_SIZE):
        for bc in range(LOGICAL_SIZE):
            if mask[br, bc]:
                out[
                    br * BLOCK_SIZE : (br + 1) * BLOCK_SIZE,
                    bc * BLOCK_SIZE : (bc + 1) * BLOCK_SIZE,
                ] = stamp
    return out


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (INPUT_HEIGHT, INPUT_WIDTH), "expected a 12x19 grid"
    assert np.all(g[:PANEL_SIZE, SEPARATOR_COL] == SEPARATOR_COLOR), "expected grey separator in panel rows"
    left = g[:PANEL_SIZE, :PANEL_SIZE]
    right = g[:PANEL_SIZE, SEPARATOR_COL + 1 :]
    stamp = g[PANEL_SIZE:, :SOURCE_SIZE]
    assert stamp.shape == (SOURCE_SIZE, SOURCE_SIZE)
    return left, right, stamp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the two symmetric panels, keep their shared foreground, and stamp the bottom 3x3 pattern into the active consensus blocks.

    Input structure:
    - The top 9 rows contain two 9x9 panels separated by a grey divider column.
    - Each panel is a biaxially symmetric pattern in colors 1-4 with several erased black cells.
    - The bottom-left 3x3 area contains a magenta/cyan source pattern; the remaining bottom cells are background.

    Rule:
    - Infer each panel's horizontal and vertical symmetry axes from the visible non-black cells.
    - Fill black holes from their mirrored counterparts to reconstruct both 9x9 panels.
    - Keep the non-black pixels that agree in both repaired panels, split that 9x9 consensus into 3x3 blocks,
      and mark every block that contains at least one foreground pixel as active.
    - Create a black 9x9 output and copy the bottom 3x3 source pattern into every active block position.

    Key insight:
    - The output blocks are selected by the repaired-and-compared panel structure, not by any frequency cue inside the source stamp.
    """
    left, right, stamp = _parse_input(grid)
    consensus = _consensus_panel(left, right)
    return _stamp_output(_block_mask(consensus), stamp)


def _transform_local(block_r: int, block_c: int, local_r: int, local_c: int) -> tuple[int, int]:
    if block_r == 2:
        local_r = 2 - local_r
    if block_c == 2:
        local_c = 2 - local_c
    return local_r, local_c


def _place_orbit(
    panel: np.ndarray,
    orbit_type: str,
    motif: tuple[tuple[int, int], ...],
    color: int,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    block_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for block_r, block_c in BLOCK_ORBITS[orbit_type]:
        coords: list[tuple[int, int]] = []
        for local_r, local_c in motif:
            tr, tc = _transform_local(block_r, block_c, local_r, local_c)
            gr = block_r * BLOCK_SIZE + tr
            gc = block_c * BLOCK_SIZE + tc
            panel[gr, gc] = color
            coords.append((gr, gc))
        block_pixels[(block_r, block_c)] = coords
    return block_pixels


def _foreground_orbit(coord: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = coord
    return {
        (r, c),
        (PANEL_SIZE - 1 - r, c),
        (r, PANEL_SIZE - 1 - c),
        (PANEL_SIZE - 1 - r, PANEL_SIZE - 1 - c),
    }


def _make_complete_panel() -> tuple[np.ndarray, dict[str, dict[tuple[int, int], list[tuple[int, int]]]]]:
    while True:
        active_types = sorted(
            random.sample(
                list(BLOCK_ORBITS),
                k=random.randint(2, len(BLOCK_ORBITS)),
            )
        )
        active_blocks = sum(len(BLOCK_ORBITS[name]) for name in active_types)
        if 3 <= active_blocks <= 8 and any(name != "center" for name in active_types):
            break

    sparse_type = random.choice([name for name in active_types if name != "center"])
    colors = random.sample(PANEL_COLORS, k=len(active_types))
    color_map = dict(zip(active_types, colors))

    panel = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    placement_meta: dict[str, dict[tuple[int, int], list[tuple[int, int]]]] = {}
    for orbit_type in active_types:
        if orbit_type == "corner":
            motif_bank = [motif for motif in CORNER_MOTIFS if len(motif) == 1] if orbit_type == sparse_type else list(CORNER_MOTIFS)
        elif orbit_type == "vertical":
            motif_bank = [motif for motif in VERTICAL_MOTIFS if len(motif) == 1] if orbit_type == sparse_type else list(VERTICAL_MOTIFS)
        elif orbit_type == "horizontal":
            motif_bank = [motif for motif in HORIZONTAL_MOTIFS if len(motif) == 1] if orbit_type == sparse_type else list(HORIZONTAL_MOTIFS)
        else:
            motif_bank = list(CENTER_MOTIFS)
        motif = random.choice(motif_bank)
        placement_meta[orbit_type] = _place_orbit(panel, orbit_type, motif, color_map[orbit_type])

    if np.count_nonzero(panel) < 8:
        return _make_complete_panel()
    return panel, placement_meta


def _make_stamp() -> np.ndarray:
    while True:
        values = np.array([random.choice(STAMP_COLORS) for _ in range(SOURCE_SIZE * SOURCE_SIZE)], dtype=int)
        if len(set(map(int, values))) < 2:
            continue
        stamp = values.reshape(SOURCE_SIZE, SOURCE_SIZE)
        if not (
            np.array_equal(stamp, np.flipud(stamp))
            and np.array_equal(stamp, np.fliplr(stamp))
        ):
            return stamp


def _block_mask_from_raw_intersection(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    shared = np.where((left == right) & (left != BACKGROUND), left, BACKGROUND)
    return _block_mask(shared)


def _hole_sets(
    complete: np.ndarray,
    placement_meta: dict[str, dict[tuple[int, int], list[tuple[int, int]]]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    sparse_choices = [
        (orbit_type, pixels)
        for orbit_type, pixels in placement_meta.items()
        if orbit_type != "center" and all(len(coords) == 1 for coords in pixels.values()) and len(pixels) >= 2
    ]
    if not sparse_choices:
        return None

    orbit_type, sparse_pixels = random.choice(sparse_choices)
    forced_blocks = random.sample(list(sparse_pixels), 2)
    left_forced = sparse_pixels[forced_blocks[0]][0]
    right_forced = sparse_pixels[forced_blocks[1]][0]

    foreground = [tuple(map(int, rc)) for rc in np.argwhere(complete != BACKGROUND)]
    candidates = [coord for coord in foreground if len(_foreground_orbit(coord)) > 1]
    if left_forced not in candidates or right_forced not in candidates:
        return None

    left_count = random.randint(2, 4)
    right_count = random.randint(2, 4)
    remaining = [coord for coord in candidates if coord not in {left_forced, right_forced}]

    for _ in range(200):
        random.shuffle(remaining)
        if len(remaining) < (left_count - 1) + (right_count - 1):
            return None
        extra_left = set(remaining[: left_count - 1])
        tail = [coord for coord in remaining if coord not in extra_left]
        extra_right = set(tail[: right_count - 1])
        left_holes = {left_forced} | extra_left
        right_holes = {right_forced} | extra_right
        if left_holes & right_holes:
            continue
        left_panel = complete.copy()
        right_panel = complete.copy()
        for r, c in left_holes:
            left_panel[r, c] = BACKGROUND
        for r, c in right_holes:
            right_panel[r, c] = BACKGROUND
        if not np.array_equal(_repair_panel(left_panel), complete):
            continue
        if not np.array_equal(_repair_panel(right_panel), complete):
            continue
        if _best_axis(left_panel, vertical=True)[2] != 1 or _best_axis(left_panel, vertical=False)[2] != 1:
            continue
        if _best_axis(right_panel, vertical=True)[2] != 1 or _best_axis(right_panel, vertical=False)[2] != 1:
            continue
        expected_mask = _block_mask(complete)
        if np.array_equal(_block_mask(left_panel), expected_mask):
            continue
        if np.array_equal(_block_mask(right_panel), expected_mask):
            continue
        if np.array_equal(_block_mask_from_raw_intersection(left_panel, right_panel), expected_mask):
            continue
        return left_holes, right_holes
    return None


def _assemble_input(left: np.ndarray, right: np.ndarray, stamp: np.ndarray) -> np.ndarray:
    grid = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=int)
    grid[:PANEL_SIZE, :PANEL_SIZE] = left
    grid[:PANEL_SIZE, SEPARATOR_COL] = SEPARATOR_COLOR
    grid[:PANEL_SIZE, SEPARATOR_COL + 1 :] = right
    grid[PANEL_SIZE:, :SOURCE_SIZE] = stamp
    return grid


def _validate_case(
    grid: np.ndarray,
    complete: np.ndarray,
    stamp: np.ndarray,
    left_holes: set[tuple[int, int]],
    right_holes: set[tuple[int, int]],
) -> bool:
    left, right, parsed_stamp = _parse_input(grid)
    if not np.array_equal(parsed_stamp, stamp):
        return False
    if len(left_holes) not in (2, 3, 4) or len(right_holes) not in (2, 3, 4):
        return False
    if left_holes & right_holes:
        return False
    if not np.array_equal(_repair_panel(left), complete):
        return False
    if not np.array_equal(_repair_panel(right), complete):
        return False

    expected_mask = _block_mask(complete)
    if expected_mask.sum() < 3 or expected_mask.sum() > 8:
        return False
    if np.array_equal(_block_mask(left), expected_mask):
        return False
    if np.array_equal(_block_mask(right), expected_mask):
        return False
    if np.array_equal(_block_mask_from_raw_intersection(left, right), expected_mask):
        return False

    expected = _stamp_output(expected_mask, stamp)
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        complete, placement_meta = _make_complete_panel()
        stamp = _make_stamp()
        holes = _hole_sets(complete, placement_meta)
        if holes is None:
            continue
        left_holes, right_holes = holes

        left = complete.copy()
        right = complete.copy()
        for r, c in left_holes:
            left[r, c] = BACKGROUND
        for r, c in right_holes:
            right[r, c] = BACKGROUND

        grid = _assemble_input(left, right, stamp)
        if _validate_case(grid, complete, stamp, left_holes, right_holes):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
