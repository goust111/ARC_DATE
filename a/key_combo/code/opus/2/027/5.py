from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, find_components, tile_fill


BLACK = Color.BLACK
RED = Color.RED
YELLOW = Color.YELLOW
ORANGE = Color.ORANGE
PATTERN_COLORS = (RED, YELLOW)
IGNORED = {BLACK, ORANGE}


def _mask_component(g: np.ndarray) -> dict:
    comps = find_components(
        g,
        bg=-1,
        connectivity=4,
        monochromatic=True,
        colors=[BLACK],
        return_format="dict",
    )
    assert len(comps) == 1, "expected exactly one black occlusion component"
    comp = comps[0]
    r0, c0, r1, c1 = comp["bbox"]
    assert int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1), "black component must be a solid rectangle"
    return comp


def _axis_period_candidates(g: np.ndarray, axis: int) -> list[tuple[int, int]]:
    h, w = g.shape
    limit = h if axis == 0 else w
    candidates: list[tuple[int, int]] = []
    for shift in range(1, limit):
        comparable = 0
        conflict = False
        if axis == 0:
            for r in range(h - shift):
                row_a = g[r]
                row_b = g[r + shift]
                for c in range(w):
                    a = int(row_a[c])
                    b = int(row_b[c])
                    if a in IGNORED or b in IGNORED:
                        continue
                    comparable += 1
                    if a != b:
                        conflict = True
                        break
                if conflict:
                    break
        else:
            for c in range(w - shift):
                col_a = g[:, c]
                col_b = g[:, c + shift]
                for r in range(h):
                    a = int(col_a[r])
                    b = int(col_b[r])
                    if a in IGNORED or b in IGNORED:
                        continue
                    comparable += 1
                    if a != b:
                        conflict = True
                        break
                if conflict:
                    break
        if not conflict and comparable > 0:
            candidates.append((shift, comparable))
    return candidates


def _detect_periods(g: np.ndarray) -> tuple[int, int]:
    row_candidates = _axis_period_candidates(g, axis=0)
    col_candidates = _axis_period_candidates(g, axis=1)
    assert row_candidates and col_candidates, "failed to detect translational periods"
    pr = max(row_candidates, key=lambda item: (item[1], -item[0]))[0]
    pc = max(col_candidates, key=lambda item: (item[1], -item[0]))[0]
    return int(pr), int(pc)


def _phase_samples(g: np.ndarray, r: int, c: int, pr: int, pc: int) -> list[int]:
    h, w = g.shape
    samples: list[int] = []
    for rr in range(r % pr, h, pr):
        for cc in range(c % pc, w, pc):
            color = int(g[rr, cc])
            if color in IGNORED:
                continue
            samples.append(color)
    return samples


def _reconstruct_mask(g: np.ndarray, bbox_rc: tuple[int, int, int, int], pr: int, pc: int) -> np.ndarray:
    out = g.copy()
    r0, c0, r1, c1 = bbox_rc
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            samples = _phase_samples(g, r, c, pr, pc)
            assert samples, "each occluded phase must have at least one visible clean sample"
            color_set = set(samples)
            assert len(color_set) == 1, "phase orbit must resolve to a unique color"
            out[r, c] = samples[0]
    return out


def _row_major(segment: np.ndarray) -> int:
    counts = Counter(int(v) for v in segment if int(v) in PATTERN_COLORS)
    assert counts, "reconstructed segment must contain pattern colors"
    top = max(counts.values())
    winners = [color for color, n in counts.items() if n == top]
    assert len(winners) == 1, "row major color must be unique"
    return int(winners[0])


def _invert_marked_rows(
    reconstructed: np.ndarray,
    bbox_rc: tuple[int, int, int, int],
    marked_rows: set[int],
) -> np.ndarray:
    out = reconstructed.copy()
    r0, c0, r1, c1 = bbox_rc
    for r in range(r0, r1 + 1):
        if r not in marked_rows:
            continue
        segment = reconstructed[r, c0 : c1 + 1]
        major = _row_major(segment)
        out[r, c0 : c1 + 1] = np.where(segment == major, BLACK, major)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the occluded periodic patch, then invert the reconstructed content only on distractor-marked rows.

    Input structure:
    - A black-free red/yellow translational pattern fills the canvas except for one solid black rectangular occlusion.
    - A few orange pixels outside the rectangle replace true pattern cells and mark special row indices.
    - The remaining non-black, non-orange cells still determine one horizontal and one vertical translation period.

    Rule:
    - Find the unique black rectangle and the set of rows containing orange distractor pixels.
    - Ignore black and orange while detecting the minimal row and column translation periods, then reconstruct the rectangle from same-phase visible samples.
    - Inside the rectangle only, for each marked row, swap the reconstructed row-major color against the local background: major-color cells become black and every other cell in that row segment becomes the major color.

    Key insight:
    - Orange pixels serve two independent roles: they must be ignored to recover the periodic texture, but their row coordinates later select exactly which reconstructed rows receive the binary inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mask = _mask_component(g)
    bbox_rc = tuple(map(int, mask["bbox"]))
    marked_rows = {int(r) for r in np.flatnonzero(np.any(g == ORANGE, axis=1))}

    pr, pc = _detect_periods(g)
    reconstructed = _reconstruct_mask(g, bbox_rc, pr, pc)
    return _invert_marked_rows(reconstructed, bbox_rc, marked_rows)


def _tile_has_smaller_period(tile: np.ndarray, axis: int) -> bool:
    size = tile.shape[axis]
    for shift in range(1, size):
        if axis == 0:
            if np.all(tile[:-shift, :] == tile[shift:, :]):
                return True
        else:
            if np.all(tile[:, :-shift] == tile[:, shift:]):
                return True
    return False


def _sample_tile(pr: int, pc: int) -> np.ndarray | None:
    for _ in range(200):
        tile = np.random.choice(PATTERN_COLORS, size=(pr, pc)).astype(int)
        if len(set(map(int, tile.ravel()))) != 2:
            continue
        if _tile_has_smaller_period(tile, axis=0) or _tile_has_smaller_period(tile, axis=1):
            continue
        if not any(len(set(map(int, row))) == 2 for row in tile):
            continue
        if not any(len(set(map(int, col))) == 2 for col in tile.T):
            continue
        return tile
    return None


def _known_output(
    inp: np.ndarray,
    clean: np.ndarray,
    bbox_rc: tuple[int, int, int, int],
    marked_rows: set[int],
) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = bbox_rc
    out[r0 : r1 + 1, c0 : c1 + 1] = clean[r0 : r1 + 1, c0 : c1 + 1]
    return _invert_marked_rows(out, bbox_rc, marked_rows)


def _unique_best_candidate(candidates: list[tuple[int, int]], chosen: int) -> bool:
    chosen_score = None
    for shift, score in candidates:
        if shift == chosen:
            chosen_score = score
            break
    if chosen_score is None:
        return False
    return sum(1 for shift, score in candidates if score == chosen_score and shift != chosen) == 0


def _place_orange_pixels(
    base: np.ndarray,
    mask_bbox: tuple[int, int, int, int],
    marked_rows: set[int],
) -> np.ndarray | None:
    g = base.copy()
    _r0, c0, _r1, c1 = mask_bbox
    outside_cols = [c for c in range(g.shape[1]) if c < c0 or c > c1]
    if not outside_cols:
        return None

    for r in marked_rows:
        n = random.choice([1, 2])
        cols = random.sample(outside_cols, k=min(n, len(outside_cols)))
        for c in cols:
            g[r, c] = ORANGE
    return g


def _rows_with_unique_major(clean: np.ndarray, bbox_rc: tuple[int, int, int, int], rows: set[int]) -> bool:
    _r0, c0, _r1, c1 = bbox_rc
    for r in rows:
        segment = clean[r, c0 : c1 + 1]
        counts = Counter(int(v) for v in segment)
        if set(counts) != set(PATTERN_COLORS):
            return False
        top = max(counts.values())
        if sum(1 for n in counts.values() if n == top) != 1:
            return False
    return True


def _all_mask_phases_visible(inp: np.ndarray, bbox_rc: tuple[int, int, int, int], pr: int, pc: int) -> bool:
    r0, c0, r1, c1 = bbox_rc
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            samples = _phase_samples(inp, r, c, pr, pc)
            if not samples or len(set(samples)) != 1:
                return False
    return True


def _semantic_valid(
    inp: np.ndarray,
    clean: np.ndarray,
    bbox_rc: tuple[int, int, int, int],
    marked_rows: set[int],
    expected: np.ndarray,
    pr: int,
    pc: int,
) -> bool:
    mask_box = bbox(inp == BLACK)
    if mask_box != bbox_rc:
        return False

    detected_pr, detected_pc = _detect_periods(inp)
    if (detected_pr, detected_pc) != (pr, pc):
        return False

    if not _unique_best_candidate(_axis_period_candidates(inp, axis=0), pr):
        return False
    if not _unique_best_candidate(_axis_period_candidates(inp, axis=1), pc):
        return False
    if not _all_mask_phases_visible(inp, bbox_rc, pr, pc):
        return False

    mask_rows = set(range(bbox_rc[0], bbox_rc[2] + 1))
    overlap_rows = marked_rows & mask_rows
    if not overlap_rows or overlap_rows == mask_rows:
        return False
    if not _rows_with_unique_major(clean, bbox_rc, overlap_rows):
        return False

    reconstructed_only = inp.copy()
    r0, c0, r1, c1 = bbox_rc
    reconstructed_only[r0 : r1 + 1, c0 : c1 + 1] = clean[r0 : r1 + 1, c0 : c1 + 1]
    if np.array_equal(reconstructed_only, expected):
        return False
    if not np.any(expected[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(11, 15)
    pr = random.choice([3, 4])
    pc = random.choice([3, 4])
    tile = _sample_tile(pr, pc)
    if tile is None:
        return None

    r_off = random.randint(0, pr - 1)
    c_off = random.randint(0, pc - 1)
    clean = tile_fill((h, w), tile, offset=(r_off, c_off)).astype(int)

    mh = random.randint(4, min(5, h - 2))
    mw = random.randint(3, min(4, w - 2))
    if h - mh - 1 < 1 or w - mw - 1 < 1:
        return None
    r0 = random.randint(1, h - mh - 1)
    c0 = random.randint(1, w - mw - 1)
    bbox_rc = (r0, c0, r0 + mh - 1, c0 + mw - 1)

    total_marked = random.choice([2, 3])
    overlap_max = min(total_marked, mh - 1)
    if overlap_max < 1:
        return None
    overlap_count = random.randint(max(1, total_marked - 1), overlap_max)
    overlap_rows = set(random.sample(range(r0, r0 + mh), k=overlap_count))
    outside_rows = [r for r in range(h) if r < r0 or r > r0 + mh - 1]
    extra = total_marked - overlap_count
    if len(outside_rows) < extra:
        return None
    marked_rows = set(overlap_rows)
    if extra:
        marked_rows.update(random.sample(outside_rows, k=extra))

    inp = clean.copy()
    inp[r0 : r0 + mh, c0 : c0 + mw] = BLACK
    inp = _place_orange_pixels(inp, bbox_rc, marked_rows)
    if inp is None:
        return None

    if not _rows_with_unique_major(clean, bbox_rc, overlap_rows):
        return None

    expected = _known_output(inp, clean, bbox_rc, marked_rows)
    if not _semantic_valid(inp, clean, bbox_rc, marked_rows, expected, pr, pc):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
