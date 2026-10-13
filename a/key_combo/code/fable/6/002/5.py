from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, draw_line, find_components, get_bg, paste


CORNER_ORDER = ("TL", "TR", "BR", "BL")
ROT_K = {"TL": 0, "TR": 3, "BR": 2, "BL": 1}
UNROT_K = {kind: (-k) % 4 for kind, k in ROT_K.items()}
ANCHOR_OFFSETS = {"TL": (0, 0), "TR": (0, 2), "BR": (2, 2), "BL": (2, 0)}

FRAME_COLOR = 0
TOP_GAP = 1
SEQ_LEN_RANGE = (4, 6)
FRAME_SIDE_RANGE = (9, 13)
MAX_ATTEMPTS = 400


def _border_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    coord_set = set(coords)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in coord_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coord_set)


def _shape_library() -> tuple[np.ndarray, ...]:
    classes: dict[tuple[int, ...], np.ndarray] = {}
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if not bool(mask[0, 0]):
            continue
        area = int(mask.sum())
        if area < 5 or area > 6:
            continue
        if not _is_connected(mask):
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        rotations = [np.rot90(mask, k) for k in range(4)]
        keys = [tuple(int(v) for v in rot.ravel()) for rot in rotations]
        if len(set(keys)) != 4:
            continue
        canon = min(keys)
        classes.setdefault(canon, mask.copy())
    return tuple(classes[key] for key in sorted(classes))


SHAPE_LIBRARY = _shape_library()


def _corner_patch_origin(frame_bbox: tuple[int, int, int, int], kind: str) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    if kind == "TL":
        return r0 + 1, c0 + 1
    if kind == "TR":
        return r0 + 1, c1 - 3
    if kind == "BR":
        return r1 - 3, c1 - 3
    if kind == "BL":
        return r1 - 3, c0 + 1
    raise ValueError(f"unknown corner kind: {kind}")


def _corner_patch(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], kind: str) -> np.ndarray:
    r, c = _corner_patch_origin(frame_bbox, kind)
    return grid[r : r + 3, c : c + 3].copy()


def _orient_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    return np.rot90(mask, k=ROT_K[kind])


def _unorient_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    return np.rot90(mask, k=UNROT_K[kind])


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _draw_shape(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int, bg: int) -> None:
    paste(grid, _mask_to_patch(mask, color, bg), top, left, transparent=bg)


def _draw_frame(grid: np.ndarray, top: int, left: int, side: int, color: int) -> None:
    bottom = top + side - 1
    right = left + side - 1
    draw_line(grid, top, left, end_r=top, end_c=right, color=color)
    draw_line(grid, top, left, end_r=bottom, end_c=left, color=color)
    draw_line(grid, bottom, left, end_r=bottom, end_c=right, color=color)
    draw_line(grid, top, right, end_r=bottom, end_c=right, color=color)


def _frame_info(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    candidates = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < FRAME_SIDE_RANGE[0]:
            continue
        if int(comp["area"]) != 4 * side_h - 4:
            continue
        patch = crop_bbox(grid, comp["bbox"])
        border = _border_mask(side_h)
        if not np.all(patch[border] == color):
            continue
        candidates.append((int(comp["area"]), (r0, c0, r1, c1), color))
    assert candidates, "expected one large hollow frame"
    _area, bbox, color = max(candidates, key=lambda item: item[0])
    return bbox, color


def _parse_sequence(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> list[np.ndarray]:
    r0 = int(frame_bbox[0])
    above = grid[:r0, :]
    seq: list[tuple[int, np.ndarray]] = []
    for comp in find_components(
        above,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(above, bbox)
        if patch.shape != (3, 3):
            continue
        seq.append((bbox[1], patch != bg))
    seq.sort(key=lambda item: item[0])
    shapes = [mask.astype(bool) for _c0, mask in seq]
    assert len(shapes) >= 4, "expected a top sequence of 3x3 guide blocks"
    return shapes


def _corner_colors_and_source(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[dict[str, int], str, np.ndarray]:
    colors: dict[str, int] = {}
    source_kind: str | None = None
    source_mask: np.ndarray | None = None
    for kind in CORNER_ORDER:
        patch = _corner_patch(grid, frame_bbox, kind)
        vals = sorted(set(map(int, patch.flatten())) - {int(bg)})
        assert len(vals) == 1, f"corner {kind} must use exactly one non-background color"
        color = vals[0]
        colors[kind] = color
        ar, ac = ANCHOR_OFFSETS[kind]
        assert int(patch[ar, ac]) == color, f"corner {kind} marker/anchor is misplaced"
        count = int((patch != bg).sum())
        if count == 1:
            continue
        assert source_kind is None, "expected exactly one fully drawn source corner"
        source_kind = kind
        source_mask = _unorient_mask(patch != bg, kind)
    assert source_kind is not None and source_mask is not None
    return colors, source_kind, source_mask


def _sequence_index(shapes: list[np.ndarray], source_mask: np.ndarray) -> int:
    hits = [i for i, mask in enumerate(shapes) if np.array_equal(mask, source_mask)]
    assert len(hits) == 1, "source corner must match exactly one top-sequence shape"
    return hits[0]


def _paint_corners(
    base: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    sequence: list[np.ndarray],
    source_kind: str,
    corner_colors: dict[str, int],
    source_idx: int,
    bg: int,
    *,
    rotate: bool = True,
    idxs_by_offset: list[int] | None = None,
) -> np.ndarray:
    out = base.copy()
    source_pos = CORNER_ORDER.index(source_kind)
    if idxs_by_offset is None:
        idxs_by_offset = [(source_idx + step) % len(sequence) for step in range(4)]
    for offset, seq_idx in enumerate(idxs_by_offset):
        kind = CORNER_ORDER[(source_pos + offset) % 4]
        mask = sequence[int(seq_idx)]
        paint_mask = _orient_mask(mask, kind) if rotate else mask
        pr, pc = _corner_patch_origin(frame_bbox, kind)
        _draw_shape(out, pr, pc, paint_mask, corner_colors[kind], bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance the source-corner shape through the top 3×3 sequence and repaint the other frame corners clockwise.

    Input structure:
    - The upper part of the grid contains several separated 3×3 monochrome guide blocks that define a left-to-right shape cycle in a fixed canonical orientation.
    - Lower in the grid there is one large hollow square frame.
    - Inside the frame, each corner contains a 3×3 corner patch: exactly one patch is a fully drawn source shape, and the other three patches contain a single colored marker pixel.

    Rule:
    - Identify which guide-block shape matches the fully drawn source corner after undoing that corner's rotation.
    - Walk clockwise around the frame from the source corner. For a corner that is k steps ahead clockwise, draw the guide shape k positions ahead in the top sequence, wrapping cyclically.
    - Rotate each drawn shape to the destination corner's orientation and color it with that corner's own marker/source color. The frame and the top guide sequence stay unchanged.

    Key insight:
    - The output at each missing corner is determined only by combining three independent cues: the source corner fixes the starting sequence index, the corner position fixes the rotation, and the local marker fixes the color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_bbox, frame_color = _frame_info(g, bg)
    assert frame_color != bg
    sequence = _parse_sequence(g, frame_bbox, bg)
    corner_colors, source_kind, source_mask = _corner_colors_and_source(g, frame_bbox, bg)
    source_idx = _sequence_index(sequence, source_mask)
    return _paint_corners(g, frame_bbox, sequence, source_kind, corner_colors, source_idx, bg)


def _sample_palette() -> tuple[int, int, dict[str, int]]:
    bg = random.choice(list(range(1, 10)))
    remaining = [c for c in range(1, 10) if c != bg]
    guide = random.choice(remaining)
    remaining = [c for c in remaining if c != guide]
    corner_colors = dict(zip(CORNER_ORDER, random.sample(remaining, 4)))
    return bg, guide, corner_colors


def _sample_sequence() -> list[np.ndarray]:
    n = random.randint(*SEQ_LEN_RANGE)
    return [mask.copy() for mask in random.sample(list(SHAPE_LIBRARY), n)]


def _sample_layout(seq_len: int, frame_side: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    seq_w = 3 * seq_len + TOP_GAP * (seq_len - 1)
    seq_r = random.randint(0, 1)
    frame_r = seq_r + 3 + random.randint(2, 4)
    bottom_margin = random.randint(1, 3)
    H = frame_r + frame_side + bottom_margin
    min_w = max(seq_w, frame_side) + 1
    max_w = min(30, max(seq_w, frame_side) + 7)
    W = random.randint(min_w, max_w)
    seq_c = random.randint(0, W - seq_w)
    frame_c = random.randint(0, W - frame_side)
    return (H, W), (seq_r, seq_c), (frame_r, frame_c)


def _build_base(
    shape: tuple[int, int],
    sequence: list[np.ndarray],
    guide_color: int,
    bg: int,
    seq_origin: tuple[int, int],
    frame_origin: tuple[int, int],
    frame_side: int,
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    seq_r, seq_c = seq_origin
    for i, mask in enumerate(sequence):
        left = seq_c + i * (3 + TOP_GAP)
        _draw_shape(grid, seq_r, left, mask, guide_color, bg)
    _draw_frame(grid, frame_origin[0], frame_origin[1], frame_side, FRAME_COLOR)
    return grid


def _make_input_from_base(
    base: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    sequence: list[np.ndarray],
    source_kind: str,
    source_idx: int,
    corner_colors: dict[str, int],
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected = _paint_corners(base, frame_bbox, sequence, source_kind, corner_colors, source_idx, bg)
    inp = base.copy()
    source_pos = CORNER_ORDER.index(source_kind)
    for offset in range(4):
        kind = CORNER_ORDER[(source_pos + offset) % 4]
        pr, pc = _corner_patch_origin(frame_bbox, kind)
        if offset == 0:
            mask = _orient_mask(sequence[source_idx], kind)
            _draw_shape(inp, pr, pc, mask, corner_colors[kind], bg)
            continue
        ar, ac = ANCHOR_OFFSETS[kind]
        inp[pr : pr + 3, pc : pc + 3] = int(bg)
        inp[pr + ar, pc + ac] = int(corner_colors[kind])
    return inp, expected


def _assert_sequence_semantics(sequence: list[np.ndarray]) -> None:
    assert len(sequence) == len({tuple(int(v) for v in mask.ravel()) for mask in sequence})
    for i, mask in enumerate(sequence):
        rotations = [tuple(int(v) for v in np.rot90(mask, k).ravel()) for k in range(4)]
        assert len(set(rotations)) == 4, f"shape {i} has rotational symmetry"
    rotation_classes = {
        min(tuple(int(v) for v in np.rot90(mask, k).ravel()) for k in range(4))
        for mask in sequence
    }
    assert len(rotation_classes) == len(sequence), "sequence contains rotation-equivalent shapes"


def _wrong_cumulative_idxs(source_idx: int, n: int) -> list[int]:
    idxs = [source_idx]
    cur = source_idx
    for step in range(1, 4):
        cur = (cur + step) % n
        idxs.append(cur)
    return idxs


def _validate_shortcuts(
    base: np.ndarray,
    expected: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    sequence: list[np.ndarray],
    source_kind: str,
    source_idx: int,
    corner_colors: dict[str, int],
    bg: int,
) -> None:
    n = len(sequence)
    wrong_same = _paint_corners(
        base,
        frame_bbox,
        sequence,
        source_kind,
        corner_colors,
        source_idx,
        bg,
        idxs_by_offset=[source_idx] * 4,
    )
    wrong_cumulative = _paint_corners(
        base,
        frame_bbox,
        sequence,
        source_kind,
        corner_colors,
        source_idx,
        bg,
        idxs_by_offset=_wrong_cumulative_idxs(source_idx, n),
    )
    wrong_counter = _paint_corners(
        base,
        frame_bbox,
        sequence,
        source_kind,
        corner_colors,
        source_idx,
        bg,
        idxs_by_offset=[(source_idx - step) % n for step in range(4)],
    )
    wrong_unrotated = _paint_corners(
        base,
        frame_bbox,
        sequence,
        source_kind,
        corner_colors,
        source_idx,
        bg,
        rotate=False,
    )
    assert not np.array_equal(wrong_same, expected), "copying the source shape everywhere must fail"
    assert not np.array_equal(wrong_cumulative, expected), "using cumulative 1+2+3 advances must fail"
    assert not np.array_equal(wrong_counter, expected), "counter-clockwise sequence reading must fail"
    assert not np.array_equal(wrong_unrotated, expected), "ignoring corner rotation must fail"


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    sequence: list[np.ndarray],
    frame_bbox: tuple[int, int, int, int],
    source_kind: str,
    source_idx: int,
    corner_colors: dict[str, int],
    bg: int,
    base: np.ndarray,
) -> None:
    _assert_sequence_semantics(sequence)
    assert len(set(corner_colors.values())) == 4
    assert not np.array_equal(inp, expected)
    solved = main(inp)
    assert np.array_equal(solved, expected)
    parsed_sequence = _parse_sequence(inp, frame_bbox, bg)
    assert len(parsed_sequence) == len(sequence)
    for a, b in zip(parsed_sequence, sequence):
        assert np.array_equal(a, b)
    parsed_colors, parsed_source, parsed_mask = _corner_colors_and_source(inp, frame_bbox, bg)
    assert parsed_source == source_kind
    assert parsed_colors == corner_colors
    assert _sequence_index(sequence, parsed_mask) == source_idx
    _validate_shortcuts(base, expected, frame_bbox, sequence, source_kind, source_idx, corner_colors, bg)


def _gen_attempt() -> np.ndarray | None:
    bg, guide_color, corner_colors = _sample_palette()
    sequence = _sample_sequence()
    frame_side = random.randint(*FRAME_SIDE_RANGE)
    grid_shape, seq_origin, frame_origin = _sample_layout(len(sequence), frame_side)
    base = _build_base(grid_shape, sequence, guide_color, bg, seq_origin, frame_origin, frame_side)
    frame_bbox = (
        frame_origin[0],
        frame_origin[1],
        frame_origin[0] + frame_side - 1,
        frame_origin[1] + frame_side - 1,
    )
    source_kind = random.choice(CORNER_ORDER)
    source_idx = random.randrange(len(sequence))
    inp, expected = _make_input_from_base(
        base,
        frame_bbox,
        sequence,
        source_kind,
        source_idx,
        corner_colors,
        bg,
    )
    try:
        _validate_instance(
            inp,
            expected,
            sequence,
            frame_bbox,
            source_kind,
            source_idx,
            corner_colors,
            bg,
            base,
        )
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
