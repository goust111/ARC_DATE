import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg


def _clockwise_border_positions(bbox):
    r0, c0, r1, c1 = map(int, bbox)
    positions = []
    for c in range(c0, c1 + 1):
        positions.append((r0, c))
    for r in range(r0 + 1, r1):
        positions.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        positions.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        positions.append((r, c0))
    return positions


def _num_layers(bbox):
    r0, c0, r1, c1 = map(int, bbox)
    return max(0, (min(r1 - r0 + 1, c1 - c0 + 1) - 1) // 2)


def _minimal_period(seq):
    seq = [int(x) for x in seq]
    for p in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % p] for i in range(len(seq))):
            return seq[:p]
    return seq


def _extract_frames(grid):
    bg = int(get_bg(grid))
    assert bg == Color.BLACK
    frames = []
    for coords in components_mask(grid != bg, connectivity=4):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3
        border = _clockwise_border_positions(bbox)
        border_set = set(border)
        coord_set = {tuple(map(int, xy)) for xy in coords}
        assert coord_set.issubset(border_set)
        assert coord_set == {pos for pos in border if int(grid[pos]) != bg}

        missing = [pos for pos in border if int(grid[pos]) == bg]
        assert len(missing) in (0, 1)
        if missing:
            gap = missing[0]
            assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        else:
            gap = None

        frames.append({"bbox": bbox, "border": border, "gap": gap})
    return frames


def _read_motif(grid, border, gap):
    gap_index = border.index(gap)
    observed = [
        int(grid[border[(gap_index + step) % len(border)]])
        for step in range(1, len(border))
    ]
    assert observed
    return _minimal_period(observed)


def _apply_fill(out, bbox, motif):
    r0, c0, r1, c1 = map(int, bbox)
    period = len(motif)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            layer = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = int(motif[(layer - 1) % period])


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Read each gapped rectangular color band clockwise from its gap and project that sequence onto the interior rings.

    Input structure:
    - The grid has black background and several one-pixel-thick hollow rectangular frames.
    - Each frame occupies only the border of its bounding box; its interior is black in the input.
    - Most frames have exactly one missing non-corner border cell, while exactly one frame is fully closed.

    Rule:
    - For every frame with a gap, treat the border colors as a clockwise cyclic band and start reading immediately after the gap.
    - Reduce that readout to its repeated base period, then color the interior concentric layers using successive period entries, cycling as needed.
    - Keep every border pixel unchanged, keep each gap black, and leave the closed no-gap frame interior black because it has no read head.

    Key insight:
    - The missing border cell is not repaired; it marks the start position for unwrapping a periodic 1D border pattern and mapping it inward ring by ring.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    out = grid.copy()
    for frame in _extract_frames(grid):
        if frame["gap"] is None:
            continue
        motif = _read_motif(grid, frame["border"], frame["gap"])
        _apply_fill(out, frame["bbox"], motif)
    return out


def _render_input(shape, frames):
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        border = _clockwise_border_positions(frame["bbox"])
        gap = frame["gap"]
        if gap is None:
            for idx, pos in enumerate(border):
                grid[pos] = int(frame["motif"][(frame["offset"] + idx) % len(frame["motif"])])
            continue

        gap_index = border.index(gap)
        for step in range(1, len(border)):
            pos = border[(gap_index + step) % len(border)]
            grid[pos] = int(frame["motif"][(step - 1) % len(frame["motif"])])
    return grid


def _render_output(shape, frames):
    out = _render_input(shape, frames)
    for frame in frames:
        if frame["gap"] is None:
            continue
        _apply_fill(out, frame["bbox"], frame["motif"])
    return out


def _expanded_bbox(bbox, pad, shape):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = map(int, shape)
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _boxes_overlap(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _valid_motif_lengths(height, width, *, require_cycle):
    perimeter = 2 * (height + width) - 4
    layers = _num_layers((0, 0, height - 1, width - 1))
    choices = []
    for length in range(3, min(6, perimeter - 1) + 1):
        if perimeter % length != 0 or perimeter // length < 3:
            continue
        if require_cycle and layers <= length:
            continue
        choices.append(length)
    return choices


def _random_motif(length):
    motif = random.sample(Color.NOT_BLACK, length)
    rev = motif[::-1]
    assert all(motif != motif[k:] + motif[:k] for k in range(1, length))
    assert all(rev != motif[k:] + motif[:k] for k in range(length))
    return motif


def _random_gap(bbox):
    r0, c0, r1, c1 = map(int, bbox)
    options = []
    options.extend((r0, c) for c in range(c0 + 1, c1))
    options.extend((r, c1) for r in range(r0 + 1, r1))
    options.extend((r1, c) for c in range(c0 + 1, c1))
    options.extend((r, c0) for r in range(r0 + 1, r1))
    return random.choice(options)


def _sample_frame_spec(shape, placed_boxes, *, closed, require_cycle):
    h_grid, w_grid = map(int, shape)
    for _ in range(200):
        height = random.randint(4, min(12, h_grid - 2))
        width = random.randint(4, min(12, w_grid - 2))
        motif_lengths = _valid_motif_lengths(height, width, require_cycle=require_cycle)
        if not motif_lengths:
            continue

        r0 = random.randint(1, h_grid - height - 1)
        c0 = random.randint(1, w_grid - width - 1)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        padded = _expanded_bbox(bbox, 1, shape)
        if any(_boxes_overlap(padded, prev) for prev in placed_boxes):
            continue

        motif = _random_motif(random.choice(motif_lengths))
        gap = None if closed else _random_gap(bbox)
        offset = random.randrange(len(motif))
        return {"bbox": bbox, "gap": gap, "motif": motif, "offset": offset}
    return None


def _generate_case(return_meta=False):
    for _ in range(400):
        shape = (random.randint(17, 30), random.randint(17, 30))
        frame_count = random.randint(2, 5)
        closed_index = random.randrange(frame_count)
        cycle_index_choices = [i for i in range(frame_count) if i != closed_index]
        cycle_index = random.choice(cycle_index_choices) if cycle_index_choices and random.random() < 0.75 else None

        frames = []
        placed_boxes = []
        ok = True
        for idx in range(frame_count):
            frame = _sample_frame_spec(
                shape,
                placed_boxes,
                closed=(idx == closed_index),
                require_cycle=(idx == cycle_index),
            )
            if frame is None:
                ok = False
                break
            frames.append(frame)
            placed_boxes.append(_expanded_bbox(frame["bbox"], 1, shape))
        if not ok:
            continue

        grid = _render_input(shape, frames)
        if len(_extract_frames(grid)) != frame_count:
            continue

        if return_meta:
            return grid, frames
        return grid
    raise RuntimeError("failed to generate a valid case")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)
