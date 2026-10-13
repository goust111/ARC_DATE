from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
EMPTY = -1
GENERATION_RETRIES = 500
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]


def _parse_frame_component(
    g: np.ndarray,
    color: int,
    comp: dict,
) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = g[r0 : r1 + 1, c0 : c1 + 1] == int(color)

    row_lines = [r for r in range(mask.shape[0]) if bool(mask[r, :].all())]
    col_lines = [c for c in range(mask.shape[1]) if bool(mask[:, c].all())]
    if len(row_lines) < 3 or len(col_lines) < 3:
        return None
    if row_lines[0] != 0 or row_lines[-1] != mask.shape[0] - 1:
        return None
    if col_lines[0] != 0 or col_lines[-1] != mask.shape[1] - 1:
        return None
    if any(a + 1 >= b for a, b in zip(row_lines, row_lines[1:])):
        return None
    if any(a + 1 >= b for a, b in zip(col_lines, col_lines[1:])):
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[row_lines, :] = True
    expected[:, col_lines] = True
    if not np.array_equal(mask, expected):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "rows": [r0 + r for r in row_lines],
        "cols": [c0 + c for c in col_lines],
    }


def _find_frames(g: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        comps = find_components(g, bg=bg, colors=[color], connectivity=4, monochromatic=True)
        if len(comps) < 2:
            continue
        frames = []
        for comp in comps:
            parsed = _parse_frame_component(g, color, comp)
            if parsed is None:
                frames = []
                break
            frames.append(parsed)
        if frames:
            candidates.append((int(color), sorted(frames, key=lambda item: item["bbox"])))

    assert len(candidates) == 1, "expected one structurally identifiable lattice-frame color"
    return candidates[0]


def _slot_matrix(g: np.ndarray, bg: int, frame_color: int, frame: dict) -> np.ndarray:
    rows = frame["rows"]
    cols = frame["cols"]
    slots = np.full((len(rows) - 1, len(cols) - 1), EMPTY, dtype=int)

    for sr in range(slots.shape[0]):
        for sc in range(slots.shape[1]):
            region = g[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]]
            vals = region[(region != int(bg)) & (region != int(frame_color))]
            if vals.size == 0:
                continue
            unique = np.unique(vals)
            assert unique.size == 1, "a filled slot must contain exactly one non-frame color"
            color = int(unique[0])
            assert np.all(region == color), "filled slots must be solid colored rectangles"
            slots[sr, sc] = color
    return slots


def _find_anchor_color(slot_mats: list[np.ndarray]) -> int:
    positions_by_color: dict[int, list[tuple[int, int, int]]] = {}
    for frame_idx, mat in enumerate(slot_mats):
        for sr, sc in np.argwhere(mat != EMPTY):
            color = int(mat[int(sr), int(sc)])
            positions_by_color.setdefault(color, []).append((frame_idx, int(sr), int(sc)))

    n_frames = len(slot_mats)
    candidates = []
    for color, positions in positions_by_color.items():
        if len(positions) != n_frames:
            continue
        seen_frames = {idx for idx, _sr, _sc in positions}
        if len(seen_frames) != n_frames:
            continue
        ok = True
        for frame_idx, sr, sc in positions:
            h, w = slot_mats[frame_idx].shape
            if not (sr in (0, h - 1) and sc in (0, w - 1)):
                ok = False
                break
        if ok:
            candidates.append(int(color))

    assert len(candidates) == 1, "expected one anchor color present once per frame in a corner slot"
    return candidates[0]


def _canonical_rotation_for_anchor(mat: np.ndarray, anchor_color: int) -> int:
    locs = np.argwhere(mat == int(anchor_color))
    assert locs.shape[0] == 1, "each frame must contain exactly one anchor slot"
    sr, sc = map(int, locs[0])
    h, w = mat.shape
    if (sr, sc) == (0, 0):
        return 0
    if (sr, sc) == (0, w - 1):
        return 1
    if (sr, sc) == (h - 1, w - 1):
        return 2
    if (sr, sc) == (h - 1, 0):
        return 3
    raise AssertionError("anchor slot must be in a physical corner")


def _infer_canonical_colors(slot_mats: list[np.ndarray], rotations: list[int]) -> dict[tuple[int, int], int]:
    canonical_shape: tuple[int, int] | None = None
    colors: dict[tuple[int, int], int] = {}

    for mat, rot in zip(slot_mats, rotations):
        canon = np.rot90(mat, int(rot))
        if canonical_shape is None:
            canonical_shape = tuple(map(int, canon.shape))
        assert tuple(canon.shape) == canonical_shape, "all frames must share one canonical slot shape"

        for sr, sc in np.argwhere(canon != EMPTY):
            key = (int(sr), int(sc))
            color = int(canon[key])
            assert key not in colors or colors[key] == color, "canonical slot colors must not conflict"
            colors[key] = color

    assert canonical_shape is not None
    expected = {(r, c) for r in range(canonical_shape[0]) for c in range(canonical_shape[1])}
    assert set(colors) == expected, "the visible slots must determine every canonical slot color"
    assert canonical_shape[0] != canonical_shape[1], "row and column counts must make rotation meaningful"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each lattice frame back to its anchor-normalized orientation, infer the shared slot colors, and fill all frames.

    Input structure:
    - A background grid contains several separated one-pixel-thick rectangular lattice frames of one frame color.
    - Every frame divides its rectangle into the same non-square logical row-by-column slot array, but each frame may be rotated independently by 0, 90, 180, or 270 degrees.
    - Some slot interiors are solid filled; exactly one filled slot per frame uses the same anchor color and that anchor lies in the same logical corner.

    Rule:
    - Parse the lattice lines to recover each frame's physical slot matrix.
    - Find the unique color that appears once per frame in a physical corner, then rotate each slot matrix so that anchor slot is upper-left.
    - Merge the visible filled slots in that canonical coordinate system to obtain one color for every logical slot.
    - Rotate the complete canonical color pattern back into each frame's own physical orientation and fill every slot interior, preserving frame lines and outside background.

    Key insight:
    - Apparent contradictions between frames are caused by independent rotations; the repeated corner anchor fixes the shared coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    frame_color, frames = _find_frames(g, bg)
    slot_mats = [_slot_matrix(g, bg, frame_color, frame) for frame in frames]
    anchor_color = _find_anchor_color(slot_mats)
    rotations = [_canonical_rotation_for_anchor(mat, anchor_color) for mat in slot_mats]
    canonical_colors = _infer_canonical_colors(slot_mats, rotations)

    canonical_shape = tuple(np.rot90(slot_mats[0], rotations[0]).shape)
    pattern = np.empty(canonical_shape, dtype=int)
    for key, color in canonical_colors.items():
        pattern[key] = int(color)

    out = g.copy()
    for frame, rot in zip(frames, rotations):
        rows = frame["rows"]
        cols = frame["cols"]
        physical_pattern = np.rot90(pattern, -int(rot))
        assert physical_pattern.shape == (len(rows) - 1, len(cols) - 1)
        for sr in range(physical_pattern.shape[0]):
            for sc in range(physical_pattern.shape[1]):
                out[rows[sr] + 1 : rows[sr + 1], cols[sc] + 1 : cols[sc + 1]] = int(
                    physical_pattern[sr, sc]
                )
    return out


def _line_positions(lengths: tuple[int, ...]) -> list[int]:
    lines = [0]
    pos = 0
    for length in lengths:
        pos += int(length) + 1
        lines.append(pos)
    return lines


def _make_patch(
    slot_heights: tuple[int, ...],
    slot_widths: tuple[int, ...],
    frame_color: int,
    slot_colors: dict[tuple[int, int], int],
    visible_slots: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    row_lines = _line_positions(slot_heights)
    col_lines = _line_positions(slot_widths)
    patch = np.full((row_lines[-1] + 1, col_lines[-1] + 1), BG, dtype=int)
    patch[row_lines, :] = int(frame_color)
    patch[:, col_lines] = int(frame_color)

    for sr in range(len(slot_heights)):
        for sc in range(len(slot_widths)):
            key = (sr, sc)
            if visible_slots is not None and key not in visible_slots:
                continue
            patch[row_lines[sr] + 1 : row_lines[sr + 1], col_lines[sc] + 1 : col_lines[sc + 1]] = int(
                slot_colors[key]
            )
    return patch


def _slot_counts_for_raw_conflicts(
    logical_shape: tuple[int, int],
    rotations: list[int],
    hints: list[set[tuple[int, int]]],
    slot_colors: dict[tuple[int, int], int],
) -> bool:
    raw: dict[tuple[int, int, int, int], set[int]] = {}
    probe = np.arange(logical_shape[0] * logical_shape[1], dtype=int).reshape(logical_shape)
    reverse: dict[int, tuple[int, int]] = {
        int(probe[r, c]): (r, c) for r in range(logical_shape[0]) for c in range(logical_shape[1])
    }
    for rot, visible in zip(rotations, hints):
        physical = np.rot90(probe, -int(rot))
        for pr in range(physical.shape[0]):
            for pc in range(physical.shape[1]):
                logical = reverse[int(physical[pr, pc])]
                if logical not in visible:
                    continue
                key = (physical.shape[0], physical.shape[1], pr, pc)
                raw.setdefault(key, set()).add(int(slot_colors[logical]))
    return any(len(colors) > 1 for colors in raw.values())


def _choose_logical_shape() -> tuple[int, int]:
    return random.choice([(2, 3), (3, 2), (2, 4), (4, 2)])


def _choose_slot_sizes(n: int) -> tuple[int, ...]:
    vals = [random.randint(1, 3) for _ in range(n)]
    if n > 1 and len(set(vals)) == 1:
        vals[random.randrange(n)] = 1 + (vals[0] % 3)
    return tuple(vals)


def _required_hints_by_rotation(rows: int, cols: int) -> dict[int, set[tuple[int, int]]]:
    return {
        0: {(0, 1)},
        1: {(rows - 1, 0)},
        2: {(rows - 1, cols - 2)},
        3: {(0, cols - 1)},
    }


def _build_hints(
    logical_shape: tuple[int, int],
    rotations: list[int],
) -> list[set[tuple[int, int]]]:
    rows, cols = logical_shape
    all_slots = {(r, c) for r in range(rows) for c in range(cols)}
    non_anchor = sorted(all_slots - {(0, 0)})
    hints = [{(0, 0)} for _ in rotations]

    required = _required_hints_by_rotation(rows, cols)
    for frame_idx, rot in enumerate(rotations):
        hints[frame_idx].update(required.get(int(rot), set()))

    for slot in non_anchor:
        if any(slot in frame_hints for frame_hints in hints):
            continue
        provider = random.randrange(len(hints))
        hints[provider].add(slot)

    for frame_hints in hints:
        if len(frame_hints) == 1:
            frame_hints.add(random.choice(non_anchor))

    for slot in non_anchor:
        if random.random() < 0.35:
            choices = [idx for idx, frame_hints in enumerate(hints) if slot not in frame_hints]
            if choices and sum(slot in frame_hints for frame_hints in hints) < len(hints) - 1:
                hints[random.choice(choices)].add(slot)

    return hints


def _place_patches(
    input_patches: list[np.ndarray],
    output_patches: list[np.ndarray],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    occ = np.zeros(shape, dtype=int)

    order = sorted(range(len(input_patches)), key=lambda i: input_patches[i].size, reverse=True)
    positions: dict[int, tuple[int, int]] = {}
    for idx in order:
        patch = np.ones(input_patches[idx].shape, dtype=int)
        try:
            r0, c0 = random_free_location(occ, patch, transparent=BG, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            return None
        paste(occ, patch, r0, c0)
        positions[idx] = (r0, c0)

    for idx in range(len(input_patches)):
        r0, c0 = positions[idx]
        paste(inp, input_patches[idx], r0, c0)
        paste(out, output_patches[idx], r0, c0)
    return inp, out


def _valid_generation(
    inp: np.ndarray,
    expected: np.ndarray,
    logical_shape: tuple[int, int],
    rotations: list[int],
    hints: list[set[tuple[int, int]]],
    slot_colors: dict[tuple[int, int], int],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if len(set(rotations)) < 4:
        return False
    if not _slot_counts_for_raw_conflicts(logical_shape, rotations, hints, slot_colors):
        return False

    anchor_color = int(slot_colors[(0, 0)])
    non_anchor_colors = {int(v) for k, v in slot_colors.items() if k != (0, 0)}
    if any(np.count_nonzero(inp == color) == 0 for color in non_anchor_colors):
        return False
    if np.count_nonzero(inp == anchor_color) == 0:
        return False

    try:
        frame_color, frames = _find_frames(inp, BG)
        mats = [_slot_matrix(inp, BG, frame_color, frame) for frame in frames]
        inferred_anchor = _find_anchor_color(mats)
        inferred_rots = [_canonical_rotation_for_anchor(mat, inferred_anchor) for mat in mats]
        inferred_colors = _infer_canonical_colors(mats, inferred_rots)
        solved = main(inp)
    except Exception:
        return False

    if inferred_anchor != anchor_color:
        return False
    if set(inferred_rots) != {0, 1, 2, 3}:
        return False
    if inferred_colors != slot_colors:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    logical_shape = _choose_logical_shape()
    rows, cols = logical_shape
    slot_count = rows * cols

    chosen = random.sample(PALETTE, slot_count + 1)
    frame_color = int(chosen[0])
    slot_values = list(map(int, chosen[1:]))
    random.shuffle(slot_values)
    slot_colors = {
        (r, c): slot_values[r * cols + c]
        for r in range(rows)
        for c in range(cols)
    }

    frame_count = random.choice([4, 5])
    rotations = [0, 1, 2, 3]
    while len(rotations) < frame_count:
        rotations.append(random.randrange(4))
    random.shuffle(rotations)

    hints = _build_hints(logical_shape, rotations)
    if any(len(frame_hints) <= 1 for frame_hints in hints):
        return None

    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []
    specs_seen = set()
    for frame_idx, rot in enumerate(rotations):
        heights = _choose_slot_sizes(rows)
        widths = _choose_slot_sizes(cols)
        spec = (heights, widths)
        if spec in specs_seen:
            return None
        specs_seen.add(spec)

        full = _make_patch(heights, widths, frame_color, slot_colors)
        partial = _make_patch(heights, widths, frame_color, slot_colors, hints[frame_idx])
        output_patches.append(np.rot90(full, -int(rot)).copy())
        input_patches.append(np.rot90(partial, -int(rot)).copy())

    shape = (random.randint(24, 30), random.randint(24, 30))
    placed = _place_patches(input_patches, output_patches, shape)
    if placed is None:
        return None
    inp, expected = placed

    if int(get_bg(inp)) != BG:
        return None
    if not _valid_generation(inp, expected, logical_shape, rotations, hints, slot_colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotated-lattice input")
