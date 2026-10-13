from __future__ import annotations

from itertools import product
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
FRAME_COLOR = GREY
MARKER_COLOR = RED
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _count_components(mask: np.ndarray) -> int:
    return len(_components(mask))


def _is_square_frame(component: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(component)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 4:
        return False
    local = np.zeros((side_h, side_w), dtype=bool)
    local[component[:, 0] - r0, component[:, 1] - c0] = True
    if not local[0, :].all() or not local[-1, :].all():
        return False
    if not local[:, 0].all() or not local[:, -1].all():
        return False
    if local[1:-1, 1:-1].any():
        return False
    return True


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(grid == FRAME_COLOR):
        if not _is_square_frame(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner": grid[r0 + 1 : r1, c0 + 1 : c1].copy(),
                "center2": c0 + c1,
            }
        )
    frames.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    assert frames, "expected at least one grey square frame"
    inner_shapes = {tuple(np.asarray(frame["inner"]).shape) for frame in frames}
    assert len(inner_shapes) == 1, "all frames must have the same interior size"
    return frames


def _canonicalize(arr: np.ndarray) -> np.ndarray:
    rotations = [np.rot90(arr, k) for k in range(4)]
    keys = [tuple((rot != BLACK).astype(np.uint8).ravel().tolist()) for rot in rotations]
    best_idx = min(range(4), key=lambda idx: keys[idx])
    return rotations[best_idx]


def _merge_fragments(aligned: list[np.ndarray]) -> np.ndarray | None:
    merged = np.zeros_like(aligned[0])
    for frag in aligned:
        mask = frag != BLACK
        conflict = mask & (merged != BLACK) & (merged != frag)
        if conflict.any():
            return None
        merged[mask] = frag[mask]
    return merged


def _recover_prototype(fragments: list[np.ndarray]) -> np.ndarray:
    best_proto: np.ndarray | None = None
    best_score: tuple[int, int, int, tuple[int, ...]] | None = None
    max_fragment_pixels = max(int(np.count_nonzero(frag)) for frag in fragments)
    for rotations in product(range(4), repeat=len(fragments)):
        aligned = [np.rot90(frag, k) for frag, k in zip(fragments, rotations)]
        merged = _merge_fragments(aligned)
        if merged is None or not (merged != BLACK).any():
            continue
        proto = _canonicalize(merged)
        pixel_count = int(np.count_nonzero(proto))
        if pixel_count <= max_fragment_pixels:
            continue
        mask = proto != BLACK
        bbox_area = 0
        coords = np.argwhere(mask)
        if coords.size:
            r0, c0, r1, c1 = _bbox(coords)
            bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        score = (
            pixel_count,
            _count_components(mask),
            bbox_area,
            tuple(mask.astype(np.uint8).ravel().tolist()),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_proto = proto
    assert best_proto is not None, "failed to align fragments into a common prototype"
    return best_proto


def _decode_marker_rotations(grid: np.ndarray, frames: list[dict[str, object]]) -> list[int]:
    h, _ = grid.shape
    red_coords = [(int(r), int(c)) for r, c in np.argwhere(grid == MARKER_COLOR)]
    assert red_coords, "expected bottom marker pixels"
    assert all(r in (h - 2, h - 1) for r, _ in red_coords), "markers must live in the bottom two rows"
    assert len(red_coords) == len(frames), "expected one marker per frame"
    mapping = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    used: set[int] = set()
    rotations: list[int] = []
    for frame in frames:
        center2 = int(frame["center2"])
        match: tuple[int, int] | None = None
        for idx, (r, c) in enumerate(red_coords):
            if idx in used:
                continue
            row_offset = 0 if r == h - 2 else 1
            if center2 == 2 * c + 1:
                col_offset = 0
            elif center2 == 2 * c - 1:
                col_offset = 1
            else:
                continue
            if match is not None:
                raise AssertionError("marker alignment is ambiguous")
            match = (idx, mapping[(row_offset, col_offset)])
        assert match is not None, "failed to match a marker to a frame"
        used.add(match[0])
        rotations.append(match[1])
    assert len(used) == len(red_coords), "unused marker pixels remain"
    return rotations


def _build_output(proto: np.ndarray, rotations: list[int]) -> np.ndarray:
    n = proto.shape[0]
    out = np.zeros((n, len(rotations) * n + len(rotations) - 1), dtype=int)
    col = 0
    for idx, rot in enumerate(rotations):
        out[:, col : col + n] = np.rot90(proto, rot)
        col += n
        if idx + 1 < len(rotations):
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct one full prototype from framed fragments and emit its marker-requested rotations.

    Input structure:
    - Black is the background. Several same-sized hollow Grey square frames are scattered on the grid.
    - Each frame interior contains an incomplete colored fragment of one shared prototype shape, shown in some rotation and with missing pixels left Black.
    - The bottom two rows contain one Red marker pixel per frame; each marker belongs to an implicit 2x2 block whose pixel position encodes 0°, 90°, 180°, or 270°.
    - Every frame's horizontal center aligns with the horizontal center of exactly one bottom marker block.

    Rule:
    - Find all Grey square frames and extract their interior fragments.
    - Rotate the fragments into a common orientation, overlay them, and take the union to recover the complete prototype.
    - Decode each bottom 2x2 marker from its single Red pixel and use that rotation for the corresponding frame.
    - Output the recovered prototype rotated per marker, ordered by frame position from left to right, with a single Black separator column between copies and no Grey borders.

    Key insight:
    - The visible pieces are complementary rotated subsets of one shared shape, so the task is to first align and union the fragments, then read the external Red marker code to choose the final orientations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frames = _extract_frames(g)
    fragments = [np.asarray(frame["inner"], dtype=int) for frame in frames]
    proto = _recover_prototype(fragments)
    marker_rotations = _decode_marker_rotations(g, frames)
    return _build_output(proto, marker_rotations)


def _random_connected_mask(size: int) -> np.ndarray | None:
    if size == 4:
        min_pixels, max_pixels = 6, 9
    else:
        min_pixels, max_pixels = 9, 15
    target = random.randint(min_pixels, max_pixels)
    cells = {(random.randrange(size), random.randrange(size))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1) < size - 1 or (c1 - c0 + 1) < size - 1:
        return None
    rots = [tuple(np.rot90(mask, k).astype(np.uint8).ravel().tolist()) for k in range(4)]
    if len(set(rots)) != 4:
        return None
    return mask


def _sample_prototype(size: int, color: int) -> np.ndarray | None:
    for _ in range(200):
        mask = _random_connected_mask(size)
        if mask is None:
            continue
        proto = np.zeros((size, size), dtype=int)
        proto[mask] = color
        return _canonicalize(proto)
    return None


def _sample_fragment_subsets(proto: np.ndarray, count: int) -> list[set[int]] | None:
    coords = [tuple(coord) for coord in np.argwhere(proto != BLACK)]
    pixel_ids = list(range(len(coords)))
    low = max(2, int(np.ceil(len(coords) * 0.45)))
    high = min(len(coords) - 1, max(low, int(np.floor(len(coords) * 0.7))))
    for _ in range(400):
        subsets: list[set[int]] = []
        seen: set[frozenset[int]] = set()
        covered: set[int] = set()
        for _j in range(count):
            size = random.randint(low, high)
            keep = set(random.sample(pixel_ids, size))
            subsets.append(keep)
            seen.add(frozenset(keep))
            covered |= keep
        if covered != set(pixel_ids):
            continue
        if len(seen) != count:
            continue
        if any(all(pid in subset for subset in subsets) for pid in pixel_ids):
            continue
        return subsets
    return None


def _subset_to_array(proto: np.ndarray, subset: set[int]) -> np.ndarray:
    arr = np.zeros_like(proto)
    coords = [tuple(coord) for coord in np.argwhere(proto != BLACK)]
    for pid in subset:
        r, c = coords[pid]
        arr[r, c] = proto[r, c]
    return arr


def _distribute_extra(total: int, slots: int) -> list[int]:
    values = [0] * slots
    for _ in range(total):
        values[random.randrange(slots)] += 1
    return values


def _sample_layout(frame_count: int, outer_size: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    base_width = frame_count * outer_size + (frame_count - 1)
    if base_width > 30:
        return None
    extra_width = random.randint(0, 30 - base_width)
    extras = _distribute_extra(extra_width, frame_count + 1)
    left_margin = extras[0]
    gaps = [1 + extras[i + 1] for i in range(frame_count - 1)]
    right_margin = extras[-1]
    cols = [left_margin]
    for gap in gaps:
        cols.append(cols[-1] + outer_size + gap)
    width = cols[-1] + outer_size + right_margin

    row_count = 1
    if frame_count >= 3 and random.random() < 0.5 and 2 * outer_size + 4 <= 30:
        row_count = 2

    top_margin = random.randint(0, 2)
    if row_count == 1:
        row_starts = [top_margin]
        row_ids = [0] * frame_count
    else:
        row_gap = random.randint(1, 3)
        row_starts = [top_margin, top_margin + outer_size + row_gap]
        row_ids = [random.randrange(2) for _ in range(frame_count)]
        if len(set(row_ids)) == 1:
            row_ids[random.randrange(frame_count)] = 1 - row_ids[0]

    marker_gap = random.randint(1, 3)
    frame_bottom = max(row_starts[row_id] + outer_size for row_id in row_ids)
    height = frame_bottom + marker_gap + 2
    if height > 30:
        return None

    placements = [(row_starts[row_id], col) for row_id, col in zip(row_ids, cols)]
    return height, width, placements


def _marker_pixel(col0: int, outer_size: int, rotation: int, marker_rows: tuple[int, int]) -> tuple[int, int]:
    center2 = col0 + (col0 + outer_size - 1)
    left_col = (center2 - 1) // 2
    top_row, bottom_row = marker_rows
    if rotation == 0:
        return top_row, left_col
    if rotation == 1:
        return top_row, left_col + 1
    if rotation == 2:
        return bottom_row, left_col
    return bottom_row, left_col + 1


def _expected_from_spec(proto: np.ndarray, target_rotations: list[int]) -> np.ndarray:
    return _build_output(proto, target_rotations)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((3, 4))
    inner_size = 4 if frame_count == 4 else random.choice((4, 6))
    outer_size = inner_size + 2
    color = random.choice(PATTERN_COLORS)
    proto = _sample_prototype(inner_size, color)
    if proto is None:
        return None

    subsets = _sample_fragment_subsets(proto, frame_count)
    if subsets is None:
        return None
    input_rotations = random.sample(range(4), frame_count)
    target_rotations = [random.randrange(4) for _ in range(frame_count)]
    if len(set(target_rotations)) == 1:
        target_rotations[random.randrange(frame_count)] = (target_rotations[0] + 1) % 4

    layout = _sample_layout(frame_count, outer_size)
    if layout is None:
        return None
    height, width, placements = layout
    grid = np.zeros((height, width), dtype=int)
    marker_rows = (height - 2, height - 1)

    for idx, ((r0, c0), subset, in_rot, out_rot) in enumerate(
        zip(placements, subsets, input_rotations, target_rotations)
    ):
        r1 = r0 + outer_size
        c1 = c0 + outer_size
        grid[r0:r1, c0] = FRAME_COLOR
        grid[r0:r1, c1 - 1] = FRAME_COLOR
        grid[r0, c0:c1] = FRAME_COLOR
        grid[r1 - 1, c0:c1] = FRAME_COLOR

        fragment = _subset_to_array(proto, subset)
        observed = np.rot90(fragment, in_rot)
        grid[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = observed

        mr, mc = _marker_pixel(c0, outer_size, out_rot, marker_rows)
        if grid[mr, mc] != BLACK:
            return None
        grid[mr, mc] = MARKER_COLOR

    expected = _expected_from_spec(proto, target_rotations)
    try:
        solved = main(grid)
    except Exception:
        return None
    if solved.shape != expected.shape or not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid framed-fragment puzzle")
