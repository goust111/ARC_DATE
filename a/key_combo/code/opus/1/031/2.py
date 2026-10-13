from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLOCK = 4
BRIDGE = 8
FRAME_COL = 0
SOLID_COL = 5
GAP_COL = 4
INPUT_COLORS = [1, 2, 3, 7]
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_patch(color: int) -> np.ndarray:
    patch = np.full((BLOCK, BLOCK), color, dtype=int)
    patch[1:3, 1:3] = BG
    return patch


def _solid_patch(color: int) -> np.ndarray:
    return np.full((BLOCK, BLOCK), color, dtype=int)


def _classify_patch(patch: np.ndarray) -> tuple[str, int]:
    assert patch.shape == (BLOCK, BLOCK), "expected 4x4 object patch"
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    border_colors = {int(v) for v in border.tolist()}
    assert len(border_colors) == 1, "object border must be monochrome"
    color = next(iter(border_colors))
    assert color != BG, "object border color cannot be background"

    if np.all(patch[1:3, 1:3] == BG):
        assert np.array_equal(patch, _frame_patch(color)), "frame must be a hollow 4x4 square"
        return "frame", color

    assert np.array_equal(patch, _solid_patch(color)), "solid must be a filled 4x4 square"
    return "solid", color


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects: list[dict[str, object]] = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 == BLOCK and c1 - c0 + 1 == BLOCK, "every object must have a 4x4 bbox"
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        kind, color = _classify_patch(patch)
        objects.append({"kind": kind, "color": color, "row": r0, "col": c0, "patch": patch})

    assert objects, "expected at least one object"
    return objects


def _render_output(frames: list[dict[str, object]], solids: list[dict[str, object]]) -> np.ndarray:
    out_height = BLOCK * max(len(frames), len(solids))
    out = np.full((out_height, 9), BG, dtype=int)

    frame_rows: dict[int, int] = {}
    solid_rows: dict[int, int] = {}

    for i, obj in enumerate(frames):
        r0 = BLOCK * i
        out[r0 : r0 + BLOCK, FRAME_COL : FRAME_COL + BLOCK] = obj["patch"]
        frame_rows[int(obj["color"])] = r0

    for i, obj in enumerate(solids):
        r0 = BLOCK * i
        out[r0 : r0 + BLOCK, SOLID_COL : SOLID_COL + BLOCK] = obj["patch"]
        solid_rows[int(obj["color"])] = r0

    assert set(frame_rows) == set(solid_rows), "every color must appear once as a frame and once as a solid"
    for color in frame_rows:
        top = max(frame_rows[color], solid_rows[color])
        bottom = min(frame_rows[color] + BLOCK - 1, solid_rows[color] + BLOCK - 1)
        if bottom - top + 1 >= 3:
            out[top + 1 : bottom, GAP_COL] = BRIDGE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack scattered 4x4 frames and solids into two columns, then bridge same-color pairs across the middle gap.

    Input structure:
    - The canvas has black background and 6 to 8 separated 4x4 monochrome square objects.
    - Each object is either a hollow frame with a black 2x2 interior or a fully filled solid square.
    - Every used color appears exactly twice: once as a frame and once as a solid.

    Rule:
    - Classify each 4x4 object from its interior: black 2x2 means frame, otherwise solid.
    - Sort frames by input top row and stack them on the left at cols 0-3; sort solids the same way and stack them on the right at cols 5-8.
    - Use a 9-column output with black padding and a single empty middle column.
    - For each color, bridge its stacked frame and stacked solid by painting teal in col 4 over the shared row span after shrinking that overlap by one row on top and bottom.

    Key insight:
    - Same-color bridge placement is determined only after the type-based reordering and stacking, so the original rows control the final bridges indirectly through rank within each type.
    """
    objects = _extract_objects(grid)
    frames = [obj for obj in objects if obj["kind"] == "frame"]
    solids = [obj for obj in objects if obj["kind"] == "solid"]

    assert frames and solids, "expected both frames and solids"
    assert len(frames) == len(solids), "frame and solid counts must match"

    frame_colors = [int(obj["color"]) for obj in frames]
    solid_colors = [int(obj["color"]) for obj in solids]
    assert len(set(frame_colors)) == len(frames), "frame colors must be unique"
    assert len(set(solid_colors)) == len(solids), "solid colors must be unique"
    assert set(frame_colors) == set(solid_colors), "colors must pair one frame with one solid"

    frames.sort(key=lambda obj: (int(obj["row"]), int(obj["col"])))
    solids.sort(key=lambda obj: (int(obj["row"]), int(obj["col"])))
    return _render_output(frames, solids)


def _perm_with_fixed_points(items: list[int], fixed_points: int) -> list[int]:
    perms = [
        list(perm)
        for perm in itertools.permutations(items)
        if sum(int(a == b) for a, b in zip(items, perm)) == fixed_points
    ]
    assert perms, "expected at least one permutation with the requested fixed-point count"
    return random.choice(perms)


def _weighted_bridge_count(n_pairs: int) -> int:
    if n_pairs == 3:
        return random.choices([0, 1], weights=[5, 4], k=1)[0]
    return random.choices([0, 1, 2], weights=[4, 3, 3], k=1)[0]


def _slot_choices(length: int, min_count: int) -> list[int]:
    candidates = []
    for offset in range(3):
        slots = list(range(offset, length - BLOCK + 1, BLOCK + 1))
        if len(slots) >= min_count:
            candidates.append(slots)
    assert candidates, "no valid slot configuration"
    return random.choice(candidates)


def _row_by_color(order: list[int], rows: list[int]) -> dict[int, int]:
    return {color: row for color, row in zip(order, rows)}


def _pick_columns(
    frame_rows: list[int], solid_rows: list[int], col_slots: list[int]
) -> tuple[dict[int, int], dict[int, int]] | None:
    frame_cols: dict[int, int] = {}
    solid_cols: dict[int, int] = {}

    for _ in range(100):
        frame_cols.clear()
        solid_cols.clear()
        row_usage: dict[int, set[int]] = {}
        ok = True

        for idx, row in enumerate(frame_rows):
            choices = [col for col in random.sample(col_slots, len(col_slots)) if col not in row_usage.get(row, set())]
            if not choices:
                ok = False
                break
            col = choices[0]
            frame_cols[idx] = col
            row_usage.setdefault(row, set()).add(col)

        if not ok:
            continue

        for idx, row in enumerate(solid_rows):
            choices = [col for col in random.sample(col_slots, len(col_slots)) if col not in row_usage.get(row, set())]
            if not choices:
                ok = False
                break
            col = choices[0]
            solid_cols[idx] = col
            row_usage.setdefault(row, set()).add(col)

        if not ok:
            continue

        frame_values = list(frame_cols.values())
        solid_values = list(solid_cols.values())
        if min(frame_values) < max(solid_values) and min(solid_values) < max(frame_values):
            return frame_cols, solid_cols

    return None


def _render_expected_from_orders(frame_order: list[int], solid_order: list[int]) -> np.ndarray:
    out = np.full((BLOCK * len(frame_order), 9), BG, dtype=int)
    for i, color in enumerate(frame_order):
        out[i * BLOCK : (i + 1) * BLOCK, FRAME_COL : FRAME_COL + BLOCK] = _frame_patch(color)
    for i, color in enumerate(solid_order):
        out[i * BLOCK : (i + 1) * BLOCK, SOLID_COL : SOLID_COL + BLOCK] = _solid_patch(color)
    for i, color in enumerate(frame_order):
        if solid_order[i] == color:
            out[i * BLOCK + 1 : i * BLOCK + 3, GAP_COL] = BRIDGE
    return out


def _candidate_semantics(
    frame_order: list[int],
    solid_order: list[int],
    frame_rows: list[int],
    solid_rows: list[int],
    frame_cols: dict[int, int],
    solid_cols: dict[int, int],
    target_bridges: int,
) -> bool:
    bridge_colors = {color for i, color in enumerate(frame_order) if solid_order[i] == color}
    if len(bridge_colors) != target_bridges or target_bridges >= len(frame_order):
        return False

    frame_row_by_color = _row_by_color(frame_order, frame_rows)
    solid_row_by_color = _row_by_color(solid_order, solid_rows)

    if frame_rows == solid_rows:
        return False

    if bridge_colors:
        if not any(frame_row_by_color[color] != solid_row_by_color[color] for color in bridge_colors):
            return False

    non_bridge_colors = [color for color in frame_order if color not in bridge_colors]
    if non_bridge_colors:
        if not any(abs(frame_row_by_color[color] - solid_row_by_color[color]) <= BLOCK + 1 for color in non_bridge_colors):
            return False

    frame_values = list(frame_cols.values())
    solid_values = list(solid_cols.values())
    if not (min(frame_values) < max(solid_values) and min(solid_values) < max(frame_values)):
        return False

    return True


def _build_grid(
    height: int,
    width: int,
    frame_order: list[int],
    solid_order: list[int],
    frame_rows: list[int],
    solid_rows: list[int],
    frame_cols: dict[int, int],
    solid_cols: dict[int, int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for i, color in enumerate(frame_order):
        r0 = frame_rows[i]
        c0 = frame_cols[i]
        grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = _frame_patch(color)
    for i, color in enumerate(solid_order):
        r0 = solid_rows[i]
        c0 = solid_cols[i]
        grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = _solid_patch(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([3, 4])
    colors = random.sample(INPUT_COLORS, n_pairs)
    frame_order = random.sample(colors, len(colors))
    target_bridges = _weighted_bridge_count(n_pairs)
    solid_order = _perm_with_fixed_points(frame_order, target_bridges)

    min_height = 19 if n_pairs == 3 else 24
    height = random.randint(min_height, 29)
    width = random.randint(19, 29)
    row_slots = _slot_choices(height, n_pairs + 1)
    col_slots = _slot_choices(width, 4)

    for _ in range(200):
        frame_rows = sorted(random.sample(row_slots, n_pairs))
        solid_rows = sorted(random.sample(row_slots, n_pairs))
        picked_cols = _pick_columns(frame_rows, solid_rows, col_slots)
        if picked_cols is None:
            continue
        frame_cols, solid_cols = picked_cols
        if not _candidate_semantics(
            frame_order,
            solid_order,
            frame_rows,
            solid_rows,
            frame_cols,
            solid_cols,
            target_bridges,
        ):
            continue

        grid = _build_grid(
            height,
            width,
            frame_order,
            solid_order,
            frame_rows,
            solid_rows,
            frame_cols,
            solid_cols,
        )
        expected = _render_expected_from_orders(frame_order, solid_order)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        objects = _extract_objects(grid)
        if len(objects) != 2 * n_pairs:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
