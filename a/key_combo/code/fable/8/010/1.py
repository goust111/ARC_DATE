import hashlib
from typing import Iterable

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
FRAME_COLORS = (1, 2, 5, 6, 7, 8, 9)
GEN_RETRIES = 500
_SEEN_OUTPUT_SIGS: set[str] = set()


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _side_inner_indices(h: int, w: int) -> list[list[int]]:
    rel_coords = _perimeter_coords((0, 0, h - 1, w - 1))
    groups = [[] for _ in range(4)]
    for idx, (r, c) in enumerate(rel_coords):
        if r == 0 and 0 < c < w - 1:
            groups[0].append(idx)
        elif c == w - 1 and 0 < r < h - 1:
            groups[1].append(idx)
        elif r == h - 1 and 0 < c < w - 1:
            groups[2].append(idx)
        elif c == 0 and 0 < r < h - 1:
            groups[3].append(idx)
    return groups


def _rotate(seq: Iterable[int], start: int) -> list[int]:
    items = list(seq)
    k = int(start) % len(items)
    return items[k:] + items[:k]


def _xor_row(seq1: list[int], seq2: list[int]) -> np.ndarray:
    assert len(seq1) == len(seq2)
    return np.array([[GREEN if a ^ b else BLACK for a, b in zip(seq1, seq2)]], dtype=int)


def _candidate_rectangle(
    grid: np.ndarray,
    anchor: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> dict | None:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    ar, ac = anchor
    if not (r0 <= ar <= r1 and c0 <= ac <= c1):
        return None
    if ar not in (r0, r1) and ac not in (c0, c1):
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    perim_mask = _perimeter_mask(h, w)
    nonblack_mask = patch != BLACK
    if np.any(nonblack_mask & ~perim_mask):
        return None

    yellow_coords = np.argwhere(patch == YELLOW)
    if yellow_coords.shape[0] != 1:
        return None
    yr, yc = map(int, yellow_coords[0])
    if (r0 + yr, c0 + yc) != anchor:
        return None

    frame_colors = set(map(int, patch[nonblack_mask])) - {YELLOW}
    if len(frame_colors) != 1:
        return None
    frame_color = next(iter(frame_colors))

    for group in _side_inner_indices(h, w):
        if not group:
            return None
        covered = False
        for idx in group:
            rr, cc = _perimeter_coords(bbox)[idx]
            if int(grid[rr, cc]) != BLACK:
                covered = True
                break
        if not covered:
            return None

    perim = _perimeter_coords(bbox)
    tl_seq = [1 if int(grid[r, c]) != BLACK else 0 for r, c in perim]
    anchor_idx = perim.index(anchor)
    if tl_seq[anchor_idx] != 1:
        return None

    return {
        "bbox": bbox,
        "shape": (h, w),
        "anchor": anchor,
        "anchor_idx": anchor_idx,
        "frame_color": frame_color,
        "perim_len": len(perim),
        "tl_seq": tl_seq,
        "anchor_seq": _rotate(tl_seq, anchor_idx),
        "nonblack_coords": {tuple(map(int, coord)) for coord in np.argwhere(nonblack_mask) + np.array([r0, c0])},
    }


def _touching(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    return not (r1a < r0b - 1 or r1b < r0a - 1 or c1a < c0b - 1 or c1b < c0a - 1)


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    anchors = [tuple(map(int, coord)) for coord in np.argwhere(grid == YELLOW)]
    assert len(anchors) == 2, "expected exactly two yellow anchors"

    all_nonblack = {tuple(map(int, coord)) for coord in np.argwhere(grid != BLACK)}
    candidate_lists: list[list[dict]] = []
    H, W = grid.shape
    for anchor in anchors:
        ar, ac = anchor
        candidates: list[dict] = []
        for r0 in range(ar + 1):
            for r1 in range(ar, H):
                if r1 - r0 + 1 < 3:
                    continue
                for c0 in range(ac + 1):
                    for c1 in range(ac, W):
                        if c1 - c0 + 1 < 3:
                            continue
                        cand = _candidate_rectangle(grid, anchor, (r0, c0, r1, c1))
                        if cand is not None:
                            candidates.append(cand)
        assert candidates, "no valid rectangle found for anchor"
        candidate_lists.append(candidates)

    valid_pairs: list[tuple[dict, dict]] = []
    for rect1 in candidate_lists[0]:
        for rect2 in candidate_lists[1]:
            if rect1["bbox"] == rect2["bbox"]:
                continue
            if rect1["perim_len"] != rect2["perim_len"]:
                continue
            if rect1["nonblack_coords"] & rect2["nonblack_coords"]:
                continue
            if rect1["nonblack_coords"] | rect2["nonblack_coords"] != all_nonblack:
                continue
            if _touching(rect1["bbox"], rect2["bbox"]):
                continue
            valid_pairs.append((rect1, rect2))

    assert len(valid_pairs) == 1, "rectangle interpretation must be unique"
    rects = list(valid_pairs[0])
    rects.sort(key=lambda item: item["bbox"])
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each hollow rectangle border into a cyclic solid/void sequence from its yellow anchor, then xor the two sequences into one row.

    Input structure:
    - The black background contains exactly two non-touching hollow axis-aligned rectangles.
    - Each rectangle has one boundary color plus black gaps on its perimeter, and exactly one yellow anchor cell on that perimeter.
    - The two rectangles have equal perimeter length even though their bounding-box shapes differ.

    Rule:
    - For each rectangle, traverse its full bounding-box perimeter clockwise.
    - Read a binary ring where any non-black boundary cell is solid and black boundary cells are void, then rotate that ring so the yellow anchor becomes position 1.
    - Xor the two anchor-aligned rings position by position and output a 1xL row with green for mismatches and black otherwise.

    Key insight:
    - The xor happens in each rectangle's own cyclic border coordinate system: the yellow anchor chooses the origin, so absolute image position or a default top-left start is not reliable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _find_rectangles(g)
    return _xor_row(rects[0]["anchor_seq"], rects[1]["anchor_seq"])


def _random_sequence(h: int, w: int, rng: np.random.Generator) -> list[int]:
    length = 2 * (h + w) - 4
    groups = _side_inner_indices(h, w)
    for _ in range(80):
        seq = (rng.random(length) < rng.uniform(0.35, 0.68)).astype(int)
        for group in groups:
            if not np.any(seq[group]):
                seq[rng.choice(group)] = 1
        if np.all(seq == 1) or np.all(seq == 0):
            continue
        if seq.sum() < max(5, len(groups)) or seq.sum() > length - 2:
            continue
        transitions = int(np.sum(seq != np.roll(seq, 1)))
        if transitions < 4:
            continue
        return seq.tolist()
    raise RuntimeError("failed to build a usable perimeter pattern")


def _draw_rectangle(
    grid: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    tl_seq: list[int],
    anchor_idx: int,
    color: int,
) -> None:
    rel_perim = _perimeter_coords((0, 0, h - 1, w - 1))
    for idx, (rr, cc) in enumerate(rel_perim):
        r = top + rr
        c = left + cc
        if tl_seq[idx]:
            grid[r, c] = color
    ar, ac = rel_perim[anchor_idx]
    grid[top + ar, left + ac] = YELLOW


def _choose_shapes(rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for total in range(8, 15):
        unique_pairs = []
        for h in range(3, total - 2):
            w = total - h
            if h > w:
                continue
            unique_pairs.append((h, w))
        for i in range(len(unique_pairs)):
            for j in range(i + 1, len(unique_pairs)):
                options.append((unique_pairs[i], unique_pairs[j]))
    pair1, pair2 = options[int(rng.integers(len(options)))]
    if bool(rng.integers(2)):
        pair1 = (pair1[1], pair1[0])
    if bool(rng.integers(2)):
        pair2 = (pair2[1], pair2[0])
    return pair1, pair2


def _place_bboxes(
    h1: int,
    w1: int,
    h2: int,
    w2: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    horizontal = bool(rng.integers(2))
    gap_min = 2
    margin = 1
    if horizontal:
        min_h = max(h1, h2) + 2 * margin
        min_w = w1 + w2 + gap_min + 2 * margin
        H = min_h + int(rng.integers(0, min(30 - min_h, 6) + 1))
        W = min_w + int(rng.integers(0, min(30 - min_w, 6) + 1))
        extra = W - min_w
        left_extra = int(rng.integers(extra + 1))
        mid_extra = int(rng.integers(extra - left_extra + 1))
        gap = gap_min + mid_extra
        left1 = margin + left_extra
        left2 = left1 + w1 + gap
        top1 = int(rng.integers(margin, H - h1 - margin + 1))
        top2 = int(rng.integers(margin, H - h2 - margin + 1))
    else:
        min_h = h1 + h2 + gap_min + 2 * margin
        min_w = max(w1, w2) + 2 * margin
        H = min_h + int(rng.integers(0, min(30 - min_h, 6) + 1))
        W = min_w + int(rng.integers(0, min(30 - min_w, 6) + 1))
        extra = H - min_h
        top_extra = int(rng.integers(extra + 1))
        mid_extra = int(rng.integers(extra - top_extra + 1))
        gap = gap_min + mid_extra
        top1 = margin + top_extra
        top2 = top1 + h1 + gap
        left1 = int(rng.integers(margin, W - w1 - margin + 1))
        left2 = int(rng.integers(margin, W - w2 - margin + 1))
    return (H, W), (top1, left1), (top2, left2)


def _validate_generated(grid: np.ndarray) -> np.ndarray:
    rects = _find_rectangles(grid)
    out = main(grid)
    assert out.shape == (1, rects[0]["perim_len"])
    assert rects[0]["perim_len"] == rects[1]["perim_len"]
    assert rects[0]["shape"] != rects[1]["shape"]
    assert np.any(out == GREEN) and np.any(out == BLACK)
    tl_out = _xor_row(rects[0]["tl_seq"], rects[1]["tl_seq"])
    assert not np.array_equal(out, tl_out), "anchor alignment must matter"
    for rect in rects:
        seq = np.array(rect["tl_seq"], dtype=int)
        assert np.any(seq == 0) and np.any(seq == 1)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    (h1, w1), (h2, w2) = _choose_shapes(rng)
    seq1 = _random_sequence(h1, w1, rng)
    seq2 = _random_sequence(h2, w2, rng)

    solid1 = np.flatnonzero(seq1)
    solid2 = np.flatnonzero(seq2)
    anchor1 = int(solid1[rng.integers(len(solid1))])
    anchor2 = int(solid2[rng.integers(len(solid2))])
    anchor_out = _xor_row(_rotate(seq1, anchor1), _rotate(seq2, anchor2))
    tl_out = _xor_row(seq1, seq2)
    if np.array_equal(anchor_out, tl_out):
        return None
    if not (np.any(anchor_out == GREEN) and np.any(anchor_out == BLACK)):
        return None

    colors = rng.choice(FRAME_COLORS, size=2, replace=False)
    (H, W), (top1, left1), (top2, left2) = _place_bboxes(h1, w1, h2, w2, rng)
    bbox1 = (top1, left1, top1 + h1 - 1, left1 + w1 - 1)
    bbox2 = (top2, left2, top2 + h2 - 1, left2 + w2 - 1)
    if _touching(bbox1, bbox2):
        return None

    grid = np.zeros((H, W), dtype=int)
    _draw_rectangle(grid, top1, left1, h1, w1, seq1, anchor1, int(colors[0]))
    _draw_rectangle(grid, top2, left2, h2, w2, seq2, anchor2, int(colors[1]))

    try:
        out = _validate_generated(grid)
    except AssertionError:
        return None

    out_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
