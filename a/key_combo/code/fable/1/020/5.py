from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, crop_bbox


GRAY = Color.GREY
PATTERN_COLOR_POOL = [Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_ATTEMPTS = 400


def _rot_cw(block: np.ndarray, turns: int = 1) -> np.ndarray:
    return np.rot90(block, -(turns % 4))


def _forward_coord(coord: tuple[int, int], turns: int, size: int) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(turns % 4):
        r, c = c, size - 1 - r
    return int(r), int(c)


def _unrotate_coord(coord: tuple[int, int], turns: int, size: int) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(turns % 4):
        r, c = size - 1 - c, r
    return int(r), int(c)


def _rect_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _find_gray_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == GRAY
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a gray occluder"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    assert np.all(crop_bbox(mask.astype(int), box) == 1), "gray cells must form one solid rectangle"
    return box


def _infer_geometry(grid: np.ndarray) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    assert w % h == 0, "strip width must be an integer number of square blocks"
    k = w // h
    assert k >= 4, "need at least four blocks for a strict majority under one occlusion and one noise"
    return h, int(k)


def _split_blocks(grid: np.ndarray, block_size: int) -> list[np.ndarray]:
    _, w = grid.shape
    return [grid[:, c0 : c0 + block_size].copy() for c0 in range(0, w, block_size)]


def _canonicalize_blocks(blocks: list[np.ndarray], mode: str) -> list[np.ndarray]:
    canonical = []
    for idx, block in enumerate(blocks):
        if mode == "rotation":
            canonical.append(np.rot90(block, idx % 4))
        elif mode == "mirror":
            canonical.append(np.fliplr(block) if idx % 2 else block.copy())
        else:
            raise ValueError(mode)
    return canonical


def _canonical_coord(local_coord: tuple[int, int], block_idx: int, size: int, mode: str) -> tuple[int, int]:
    r, c = map(int, local_coord)
    if mode == "rotation":
        return _unrotate_coord((r, c), block_idx, size)
    if mode == "mirror":
        return (int(r), int(size - 1 - c)) if block_idx % 2 else (int(r), int(c))
    raise ValueError(mode)


def _vote_grid(canonical_blocks: list[np.ndarray], gray: int) -> tuple[np.ndarray, np.ndarray]:
    h = canonical_blocks[0].shape[0]
    consensus = np.full((h, h), -1, dtype=int)
    margin = np.full((h, h), -1, dtype=int)
    for r in range(h):
        for c in range(h):
            votes = [int(block[r, c]) for block in canonical_blocks if int(block[r, c]) != int(gray)]
            assert votes, "every correspondence group must have at least one visible member"
            counts = Counter(votes).most_common()
            assert len(counts) == 1 or counts[0][1] > counts[1][1], "majority must be unique"
            consensus[r, c] = counts[0][0]
            margin[r, c] = counts[0][1] - (counts[1][1] if len(counts) > 1 else 0)
    return consensus, margin


def _restore_rectangle(grid: np.ndarray, *, mode: str = "rotation") -> np.ndarray:
    h, _k = _infer_geometry(grid)
    rect = _find_gray_rect(grid)
    blocks = _split_blocks(np.asarray(grid, dtype=int), h)
    canonical_blocks = _canonicalize_blocks(blocks, mode=mode)
    consensus, _margin = _vote_grid(canonical_blocks, GRAY)

    r0, c0, r1, c1 = rect
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for rr, r in enumerate(range(r0, r1 + 1)):
        for cc, c in enumerate(range(c0, c1 + 1)):
            block_idx = c // h
            local = (r, c % h)
            base_coord = _canonical_coord(local, block_idx, h, mode=mode)
            out[rr, cc] = consensus[base_coord]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden gray rectangle by unrotating a seamless relay strip and majority-voting each correspondence group.

    Input structure:
    - The full canvas is a horizontal strip of adjacent h x h square blocks with no separator lines.
    - The leftmost block is the source motif, and each later block is the previous one rotated 90 degrees clockwise.
    - One solid gray rectangle occludes part of the strip, and a few visible pixels are noisy edits.

    Rule:
    - Infer the hidden block partition, then rotate every block back into the leftmost block's coordinate frame.
    - Group together cells that land on the same canonical coordinate across the strip.
    - For every occluded gray cell, replace it with the majority color among the visible members of its group.
    - Output only the restored contents of the gray rectangle, with the same rectangle size.

    Key insight:
    - The reliable relation is cumulative 90-degree rotation, not alternating mirror symmetry; with one possible noisy vote per group, the correct fill comes from whole-group majority rather than copying any single block.
    """
    return _restore_rectangle(np.asarray(grid, dtype=int), mode="rotation")


def _make_base_block(size: int, palette: list[int], rng: np.random.Generator) -> np.ndarray | None:
    color_probs = np.array([0.42] + [0.58 / (len(palette) - 1)] * (len(palette) - 1), dtype=float)
    block = np.zeros((size, size), dtype=int)
    seen: set[tuple[int, int]] = set()
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (c, r),
                (size - 1 - r, size - 1 - c),
                (size - 1 - c, size - 1 - r),
            }
            seen |= orbit
            color = int(rng.choice(palette, p=color_probs))
            for rr, cc in orbit:
                block[rr, cc] = color

    candidates = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if r < c
        and (r, c) != (size - 1 - r, size - 1 - c)
        and (r, c) != (size - 1 - c, size - 1 - r)
    ]
    rng.shuffle(candidates)
    breaker_count = int(rng.integers(2, min(5, len(candidates) + 1)))
    chosen = candidates[:breaker_count]
    for r, c in chosen:
        new_color = int(rng.choice([col for col in palette if col != int(block[r, c])]))
        block[r, c] = new_color
        rr, cc = size - 1 - r, size - 1 - c
        block[rr, cc] = new_color

    if len(colors_present(block)) < 3:
        return None
    if np.array_equal(block, _rot_cw(block, 1)):
        return None
    diag_mismatch = int(np.count_nonzero(block != block.T))
    half_turn_mismatch = int(np.count_nonzero(block != _rot_cw(block, 2)))
    if not (4 <= diag_mismatch <= 2 * size + 4):
        return None
    if half_turn_mismatch != 0:
        return None
    return block


def _build_clean_strip(base: np.ndarray, block_count: int) -> np.ndarray:
    return np.concatenate([_rot_cw(base, idx) for idx in range(block_count)], axis=1)


def _occluded_groups(
    box: tuple[int, int, int, int],
    block_size: int,
) -> list[dict[str, object]]:
    groups = []
    for r, c in _rect_cells(box):
        block_idx = c // block_size
        local = (r, c % block_size)
        base_coord = _unrotate_coord(local, block_idx, block_size)
        groups.append(
            {
                "grid": (int(r), int(c)),
                "block_idx": int(block_idx),
                "local": local,
                "base_coord": base_coord,
            }
        )
    return groups


def _rect_has_unique_groups(box: tuple[int, int, int, int], block_size: int) -> bool:
    base_coords = [entry["base_coord"] for entry in _occluded_groups(box, block_size)]
    return len(base_coords) == len(set(base_coords))


def _crosses_single_boundary(box: tuple[int, int, int, int], block_size: int) -> bool:
    _r0, c0, _r1, c1 = box
    left_block = c0 // block_size
    right_block = c1 // block_size
    return right_block == left_block + 1


def _candidate_rectangles(
    clean: np.ndarray,
    block_size: int,
    block_count: int,
    *,
    prefer_cross_boundary: bool,
) -> list[tuple[int, int, int, int]]:
    h, w = clean.shape
    min_area = block_count
    boxes = []
    for rect_h in range(2, min(h, 4) + 1):
        max_rect_w = min(block_size + 2, w, max(3, 2 * block_size - 1))
        for rect_w in range(2, max_rect_w + 1):
            if rect_h * rect_w < min_area:
                continue
            for r0 in range(0, h - rect_h + 1):
                for c0 in range(0, w - rect_w + 1):
                    box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                    crosses = _crosses_single_boundary(box, block_size)
                    if prefer_cross_boundary and not crosses:
                        continue
                    if not prefer_cross_boundary and crosses:
                        continue
                    if not _rect_has_unique_groups(box, block_size):
                        continue
                    patch = crop_bbox(clean, box)
                    if len(colors_present(patch)) < 2:
                        continue
                    boxes.append(box)
    return boxes


def _assign_noise(
    entries: list[dict[str, object]],
    source_blocks: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, dict[str, object]]] | None:
    order = list(source_blocks)
    rng.shuffle(order)

    def search(idx: int, used_entries: set[int], acc: list[tuple[int, dict[str, object]]]) -> list[tuple[int, dict[str, object]]] | None:
        if idx == len(order):
            return acc
        source_block = order[idx]
        candidates = [entry_idx for entry_idx, entry in enumerate(entries) if entry_idx not in used_entries and int(entry["block_idx"]) != source_block]
        rng.shuffle(candidates)
        for entry_idx in candidates:
            used_entries.add(entry_idx)
            acc.append((source_block, entries[entry_idx]))
            found = search(idx + 1, used_entries, acc)
            if found is not None:
                return found
            acc.pop()
            used_entries.remove(entry_idx)
        return None

    return search(0, set(), [])


def _global_coord_for_base(base_coord: tuple[int, int], block_idx: int, block_size: int) -> tuple[int, int]:
    local = _forward_coord(base_coord, block_idx, block_size)
    return int(local[0]), int(block_idx * block_size + local[1])


def _apply_occlusion_and_noise(
    clean: np.ndarray,
    box: tuple[int, int, int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    block_size, block_count = _infer_geometry(clean)
    entries = _occluded_groups(box, block_size)
    covered_blocks = {int(entry["block_idx"]) for entry in entries}
    source_blocks = [idx for idx in range(block_count) if idx not in covered_blocks]
    assignments = _assign_noise(entries, source_blocks, rng)
    if assignments is None:
        return None

    inp = clean.copy()
    for source_block, entry in assignments:
        base_coord = entry["base_coord"]
        rr, cc = _global_coord_for_base(base_coord, source_block, block_size)
        true_color = int(clean[rr, cc])
        noise_choices = [col for col in palette if col != true_color]
        inp[rr, cc] = int(rng.choice(noise_choices))

    r0, c0, r1, c1 = box
    inp[r0 : r1 + 1, c0 : c1 + 1] = GRAY
    return inp


def _strict_majority_holds(grid: np.ndarray) -> bool:
    h, _k = _infer_geometry(grid)
    rect = _find_gray_rect(grid)
    blocks = _split_blocks(grid, h)
    canonical_blocks = _canonicalize_blocks(blocks, mode="rotation")
    _consensus, margin = _vote_grid(canonical_blocks, GRAY)
    for entry in _occluded_groups(rect, h):
        base_r, base_c = entry["base_coord"]
        if int(margin[base_r, base_c]) <= 0:
            return False
    return True


def _single_block_copy_fails(grid: np.ndarray) -> bool:
    h, k = _infer_geometry(grid)
    rect = _find_gray_rect(grid)
    blocks = _split_blocks(grid, h)
    canonical_blocks = _canonicalize_blocks(blocks, mode="rotation")
    truth = _restore_rectangle(grid, mode="rotation")

    for source_block in range(k):
        candidate = np.empty_like(truth)
        usable = True
        for rr, r in enumerate(range(rect[0], rect[2] + 1)):
            for cc, c in enumerate(range(rect[1], rect[3] + 1)):
                local = (r, c % h)
                base_coord = _canonical_coord(local, c // h, h, mode="rotation")
                value = int(canonical_blocks[source_block][base_coord])
                if value == GRAY:
                    usable = False
                    break
                candidate[rr, cc] = value
            if not usable:
                break
        if usable and np.array_equal(candidate, truth):
            return False
    return True


def _mirror_trap_fails(grid: np.ndarray) -> bool:
    try:
        false_out = _restore_rectangle(grid, mode="mirror")
    except AssertionError:
        return True
    true_out = _restore_rectangle(grid, mode="rotation")
    return int(np.count_nonzero(false_out != true_out)) >= 2


def _choose_rect(clean: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    h, k = _infer_geometry(clean)
    prefer_cross_first = bool(rng.integers(2))
    for prefer_cross in (prefer_cross_first, not prefer_cross_first, True, False):
        boxes = _candidate_rectangles(clean, h, k, prefer_cross_boundary=prefer_cross)
        rng.shuffle(boxes)
        for box in boxes:
            probe = clean.copy()
            r0, c0, r1, c1 = box
            probe[r0 : r1 + 1, c0 : c1 + 1] = GRAY
            if not _mirror_trap_fails(probe):
                continue
            return box
    return None


def _make_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size_options = [5, 6, 7]
    block_size = int(rng.choice(size_options))
    possible_block_counts = [k for k in range(4, 7) if block_size * k <= 30]
    block_count = int(rng.choice(possible_block_counts))
    palette_size = int(rng.integers(4, 6))
    palette = [int(col) for col in rng.choice(PATTERN_COLOR_POOL, size=palette_size, replace=False)]
    if GRAY in palette:
        return None

    base = _make_base_block(block_size, palette, rng)
    if base is None:
        return None
    clean = _build_clean_strip(base, block_count)
    rect = _choose_rect(clean, rng)
    if rect is None:
        return None

    inp = _apply_occlusion_and_noise(clean, rect, palette, rng)
    if inp is None:
        return None

    expected = crop_bbox(clean, rect)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _strict_majority_holds(inp):
        return None
    if not _single_block_copy_fails(inp):
        return None
    if not _mirror_trap_fails(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rolling-rotation strip input")
