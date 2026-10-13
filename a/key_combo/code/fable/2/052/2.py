from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste  # noqa: E402


BG = 0
ROW_COUNT = 4
COL_CHOICES = (3, 4)
SPRITE_SIZE = 3
ROW_GAP_RANGE = (5, 7)
COL_GAP_RANGE = (5, 7)
MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400
CLUSTER_TOLERANCE = 2
MOVE_DELTAS = (-2, -1, 1, 2)
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)


SHAPE_LIBRARY = [
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
]


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _slot_patch(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _unique_mode(values: list) -> object:
    counts = Counter(values)
    top = counts.most_common(2)
    assert top, "expected at least one vote"
    if len(top) > 1:
        assert top[0][1] > top[1][1], "expected a unique majority vote"
    return top[0][0]


def _cluster_axis(values: list[int], tolerance: int = CLUSTER_TOLERANCE) -> list[int]:
    assert values, "expected at least one coordinate"
    ordered = sorted(int(v) for v in values)
    groups: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= int(tolerance):
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(_unique_mode(group)) for group in groups]


def _nearest_index(axis: list[int], value: int) -> int:
    distances = [abs(int(value) - int(anchor)) for anchor in axis]
    best = min(distances)
    assert distances.count(best) == 1, "expected a unique nearest lattice line"
    return distances.index(best)


def _component_records(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    assert comps, "expected foreground sprites"
    records: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        mask = crop_bbox(grid, comp["bbox"]) != bg
        assert mask.any(), "component mask must be non-empty"
        records.append(
            {
                "r0": r0,
                "c0": c0,
                "color": int(comp["color"]),
                "mask": mask,
                "shape": mask.shape,
                "signature": _shape_signature(mask),
            }
        )
    return bg, records


def _recover_structure(grid: np.ndarray) -> dict:
    bg, records = _component_records(grid)
    rows = _cluster_axis([rec["r0"] for rec in records])
    cols = _cluster_axis([rec["c0"] for rec in records])
    assert len(rows) >= 2 and len(cols) >= 2, "expected a row-column lattice"

    for rec in records:
        rec["row"] = _nearest_index(rows, rec["r0"])
        rec["col"] = _nearest_index(cols, rec["c0"])

    row_masks: list[np.ndarray] = []
    row_signatures: list[tuple[tuple[int, ...], ...]] = []
    for row_idx in range(len(rows)):
        row_records = [rec for rec in records if rec["row"] == row_idx]
        assert row_records, "each recovered row must have evidence"
        majority_signature = _unique_mode([rec["signature"] for rec in row_records])
        exemplar = next(rec["mask"] for rec in row_records if rec["signature"] == majority_signature)
        row_signatures.append(majority_signature)
        row_masks.append(exemplar.copy())

    col_colors: list[int] = []
    for col_idx in range(len(cols)):
        col_records = [rec for rec in records if rec["col"] == col_idx]
        assert col_records, "each recovered column must have evidence"
        col_colors.append(int(_unique_mode([rec["color"] for rec in col_records])))

    exact_slots: dict[tuple[int, int], dict] = {}
    offslot_records: list[dict] = []
    row_lookup = {int(r): idx for idx, r in enumerate(rows)}
    col_lookup = {int(c): idx for idx, c in enumerate(cols)}
    for rec in records:
        key = (int(rec["r0"]), int(rec["c0"]))
        if key[0] in row_lookup and key[1] in col_lookup:
            slot = (row_lookup[key[0]], col_lookup[key[1]])
            assert slot not in exact_slots, "expected at most one sprite per lattice slot"
            exact_slots[slot] = rec
        else:
            offslot_records.append(rec)

    assert len(offslot_records) == 1, "expected exactly one displaced sprite"
    empty_slots = [
        (r_idx, c_idx)
        for r_idx in range(len(rows))
        for c_idx in range(len(cols))
        if (r_idx, c_idx) not in exact_slots
    ]
    assert empty_slots, "expected at least one empty slot"

    displaced = offslot_records[0]
    distances = [
        abs(int(displaced["r0"]) - int(rows[r_idx])) + abs(int(displaced["c0"]) - int(cols[c_idx]))
        for r_idx, c_idx in empty_slots
    ]
    best = min(distances)
    assert distances.count(best) == 1, "expected the displaced sprite to have one nearest empty slot"
    move_target = empty_slots[distances.index(best)]

    return {
        "bg": bg,
        "records": records,
        "rows": rows,
        "cols": cols,
        "row_masks": row_masks,
        "row_signatures": row_signatures,
        "col_colors": col_colors,
        "exact_slots": exact_slots,
        "empty_slots": empty_slots,
        "displaced": displaced,
        "move_target": move_target,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the hidden row-shape/column-color sprite table, correcting one displacement, one row-shape fault, one column-color fault, and the remaining empty slots.

    Input structure:
    - A black background contains several small monochrome sprites near the slots of a hidden row/column lattice.
    - Sprites in the same lattice row share one shape mask, and sprites in the same lattice column share one color.
    - Some lattice slots are empty; additionally, one sprite is slightly displaced, one on-slot sprite has the wrong shape, and one on-slot sprite has the wrong color.

    Rule:
    - Recover the lattice row starts and column starts from the near-aligned sprite positions.
    - Use the majority shape within each recovered row and the majority color within each recovered column as the true row/column attributes.
    - Move the displaced sprite back to its unique nearest empty slot, repaint the row-shape fault with that row's shape, repaint the column-color fault with that column's color, and stamp every remaining empty slot with the row-shape/column-color combination.

    Key insight:
    - Position, shape, and color errors are three views of one incomplete Cartesian-product table, so the table structure must be reconstructed first and then used to repair every slot consistently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    info = _recover_structure(g)
    out = np.full_like(g, int(info["bg"]))
    for row_idx, r0 in enumerate(info["rows"]):
        mask = info["row_masks"][row_idx]
        for col_idx, c0 in enumerate(info["cols"]):
            patch = _slot_patch(mask, info["col_colors"][col_idx], info["bg"])
            paste(out, patch, int(r0), int(c0), transparent=info["bg"])
    return out


def _sample_axis(count: int, gap_range: tuple[int, int]) -> tuple[int, list[int]]:
    positions = [random.randint(*MARGIN_RANGE)]
    for _ in range(count - 1):
        positions.append(positions[-1] + random.randint(*gap_range))
    size = positions[-1] + SPRITE_SIZE + random.randint(*MARGIN_RANGE)
    assert size <= 30
    return size, positions


def _render_solution(height: int, width: int, rows: list[int], cols: list[int], row_masks: list[np.ndarray], col_colors: list[int]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for row_idx, r0 in enumerate(rows):
        for col_idx, c0 in enumerate(cols):
            patch = _slot_patch(row_masks[row_idx], col_colors[col_idx], BG)
            paste(out, patch, r0, c0, transparent=BG)
    return out


def _row_shape_setup() -> tuple[list[np.ndarray], tuple[int, int]]:
    duplicate_rows = tuple(sorted(random.sample(range(ROW_COUNT), 2)))
    base_shapes = random.sample(SHAPE_LIBRARY, 3)
    row_masks: list[np.ndarray] = [None] * ROW_COUNT  # type: ignore[list-item]
    row_masks[duplicate_rows[0]] = base_shapes[0]
    row_masks[duplicate_rows[1]] = base_shapes[0]
    other_rows = [idx for idx in range(ROW_COUNT) if idx not in duplicate_rows]
    row_masks[other_rows[0]] = base_shapes[1]
    row_masks[other_rows[1]] = base_shapes[2]
    return [mask.copy() for mask in row_masks], duplicate_rows


def _candidate_wrong_shapes(correct_mask: np.ndarray) -> list[np.ndarray]:
    correct_sig = _shape_signature(correct_mask)
    return [mask.copy() for mask in SHAPE_LIBRARY if _shape_signature(mask) != correct_sig]


def _wrong_color_choices(col_colors: list[int], correct_color: int) -> list[int]:
    others = [color for color in col_colors if int(color) != int(correct_color)]
    extras = [color for color in COLOR_POOL if color not in col_colors and int(color) != int(correct_color)]
    options = others[:]
    if extras:
        options.extend(extras)
    return options


def _candidate_move_positions(width: int, row0: int, col0: int, mask: np.ndarray, occupied: np.ndarray) -> list[tuple[int, int]]:
    patch = _slot_patch(mask, 1, BG)
    candidates = []
    for dc in MOVE_DELTAS:
        new_c0 = int(col0) + int(dc)
        if not (0 <= new_c0 and new_c0 + patch.shape[1] <= width):
            continue
        overlap = occupied[row0 : row0 + patch.shape[0], new_c0 : new_c0 + patch.shape[1]]
        if np.any(overlap[patch != BG]):
            continue
        candidates.append((row0, new_c0))
    return candidates


def _validate_semantics(inp: np.ndarray, solved: np.ndarray, meta: dict) -> bool:
    try:
        info = _recover_structure(inp)
    except Exception:
        return False

    if not np.array_equal(main(inp), solved):
        return False
    if info["rows"] != meta["rows"] or info["cols"] != meta["cols"]:
        return False
    if info["move_target"] != meta["move_slot"]:
        return False
    if set(info["empty_slots"]) != {meta["move_slot"], meta["extra_empty_slot"]}:
        return False
    if info["row_signatures"] != [_shape_signature(mask) for mask in meta["row_masks"]]:
        return False
    if info["col_colors"] != meta["col_colors"]:
        return False

    displaced_pair = (
        info["row_signatures"][meta["move_slot"][0]],
        info["col_colors"][meta["move_slot"][1]],
    )
    pair_empty_candidates = [
        slot
        for slot in info["empty_slots"]
        if (info["row_signatures"][slot[0]], info["col_colors"][slot[1]]) == displaced_pair
    ]
    if set(pair_empty_candidates) != {meta["move_slot"], meta["extra_empty_slot"]}:
        return False

    compliant_per_row = [0] * len(info["rows"])
    compliant_per_col = [0] * len(info["cols"])
    for (row_idx, col_idx), rec in info["exact_slots"].items():
        if rec["signature"] == info["row_signatures"][row_idx] and rec["color"] == info["col_colors"][col_idx]:
            compliant_per_row[row_idx] += 1
            compliant_per_col[col_idx] += 1
    if min(compliant_per_row) < 2 or min(compliant_per_col) < 2:
        return False

    shape_slot = meta["shape_slot"]
    shape_rec = info["exact_slots"].get(shape_slot)
    if shape_rec is None:
        return False
    if shape_rec["color"] != info["col_colors"][shape_slot[1]]:
        return False
    if shape_rec["signature"] == info["row_signatures"][shape_slot[0]]:
        return False

    color_slot = meta["color_slot"]
    color_rec = info["exact_slots"].get(color_slot)
    if color_rec is None:
        return False
    if color_rec["signature"] != info["row_signatures"][color_slot[0]]:
        return False
    if color_rec["color"] == info["col_colors"][color_slot[1]]:
        return False

    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    col_count = random.choice(COL_CHOICES)
    height, rows = _sample_axis(ROW_COUNT, ROW_GAP_RANGE)
    width, cols = _sample_axis(col_count, COL_GAP_RANGE)
    row_masks, duplicate_rows = _row_shape_setup()
    col_colors = random.sample(COLOR_POOL, col_count)
    solved = _render_solution(height, width, rows, cols, row_masks, col_colors)

    ambiguous_col = random.randrange(col_count)
    move_row, extra_empty_row = random.sample(list(duplicate_rows), 2)
    other_rows = [idx for idx in range(ROW_COUNT) if idx not in duplicate_rows]
    shape_row, color_row = random.sample(other_rows, 2)

    non_ambiguous_cols = [idx for idx in range(col_count) if idx != ambiguous_col]
    shape_col = random.choice(non_ambiguous_cols)
    if len(non_ambiguous_cols) == 1 or random.random() < 0.5:
        color_col = shape_col
    else:
        color_col = random.choice(non_ambiguous_cols)

    move_slot = (move_row, ambiguous_col)
    extra_empty_slot = (extra_empty_row, ambiguous_col)
    shape_slot = (shape_row, shape_col)
    color_slot = (color_row, color_col)
    assert len({move_slot, extra_empty_slot, shape_slot, color_slot}) == 4

    inp = solved.copy()
    for row_idx, col_idx in (move_slot, extra_empty_slot):
        inp[rows[row_idx] : rows[row_idx] + SPRITE_SIZE, cols[col_idx] : cols[col_idx] + SPRITE_SIZE] = BG

    shape_mask = row_masks[shape_row]
    wrong_shape = random.choice(_candidate_wrong_shapes(shape_mask))
    inp[rows[shape_row] : rows[shape_row] + SPRITE_SIZE, cols[shape_col] : cols[shape_col] + SPRITE_SIZE] = _slot_patch(
        wrong_shape,
        col_colors[shape_col],
        BG,
    )

    correct_color = col_colors[color_col]
    wrong_color = random.choice(_wrong_color_choices(col_colors, correct_color))
    inp[rows[color_row] : rows[color_row] + SPRITE_SIZE, cols[color_col] : cols[color_col] + SPRITE_SIZE] = _slot_patch(
        row_masks[color_row],
        wrong_color,
        BG,
    )

    occupied = inp != BG
    move_mask = row_masks[move_row]
    move_r0 = rows[move_row]
    move_c0 = cols[ambiguous_col]
    move_candidates = _candidate_move_positions(width, move_r0, move_c0, move_mask, occupied)
    if not move_candidates:
        return None
    moved_r0, moved_c0 = random.choice(move_candidates)
    move_patch = _slot_patch(move_mask, col_colors[ambiguous_col], BG)
    paste(inp, move_patch, moved_r0, moved_c0, transparent=BG)

    meta = {
        "rows": rows,
        "cols": cols,
        "row_masks": row_masks,
        "col_colors": col_colors,
        "move_slot": move_slot,
        "extra_empty_slot": extra_empty_slot,
        "shape_slot": shape_slot,
        "color_slot": color_slot,
    }
    if not _validate_semantics(inp, solved, meta):
        return None
    return inp, solved


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input for puzzle 2")
