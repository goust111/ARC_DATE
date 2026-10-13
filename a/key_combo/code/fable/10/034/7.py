from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
NON_BG_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            components.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                    "colors": {int(g[rr, cc]) for rr, cc in coords},
                }
            )
    return components


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _find_unique_repeat_pair(exemplar: np.ndarray) -> int:
    matches = [i for i in range(exemplar.shape[1] - 1) if np.array_equal(exemplar[:, i], exemplar[:, i + 1])]
    assert len(matches) == 1, "expected exactly one adjacent repeated-column pair"
    return int(matches[0])


def _stretch_exemplar(exemplar: np.ndarray, target_len: int) -> np.ndarray:
    repeat_i = _find_unique_repeat_pair(exemplar)
    extra = int(target_len) - exemplar.shape[1]
    assert extra > 0, "anchor distance must exceed exemplar length"
    repeated = np.repeat(exemplar[:, repeat_i : repeat_i + 1], extra, axis=1)
    return np.concatenate((exemplar[:, : repeat_i + 1], repeated, exemplar[:, repeat_i + 1 :]), axis=1)


def _detect_puzzle(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[tuple[int, int]], bool, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert len(components) == 3, "expected one exemplar and two anchor pixels"

    exemplar_candidates = []
    anchor_candidates = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if comp["area"] == 1:
            anchor_candidates.append((r0, c0, next(iter(comp["colors"]))))
        elif bbox_area == comp["area"] and len(comp["colors"]) > 1:
            exemplar_candidates.append(comp)

    assert len(exemplar_candidates) == 1, "expected exactly one solid multicolor exemplar rectangle"
    assert len(anchor_candidates) == 2, "expected exactly two anchor pixels"

    anchor_colors = {color for _r, _c, color in anchor_candidates}
    assert len(anchor_colors) == 1, "anchors must share one color"
    anchors = [(r, c) for r, c, _color in anchor_candidates]
    if anchors[0][0] == anchors[1][0]:
        horizontal = True
        anchors = sorted(anchors, key=lambda rc: rc[1])
    else:
        assert anchors[0][1] == anchors[1][1], "anchors must align horizontally or vertically"
        horizontal = False
        anchors = sorted(anchors, key=lambda rc: rc[0])

    exemplar_comp = exemplar_candidates[0]
    exemplar_bbox = exemplar_comp["bbox"]
    exemplar = _crop(g, exemplar_bbox)
    assert exemplar.shape[0] == 1, "generator keeps the exemplar as a 1xW rectangle for unique endpoint placement"
    assert exemplar.shape[1] >= 5, "exemplar must have fixed bands on both sides of the elastic run"
    assert _find_unique_repeat_pair(exemplar) not in (0, exemplar.shape[1] - 2), "elastic run must be interior"

    anchor_distance = (anchors[1][1] - anchors[0][1] if horizontal else anchors[1][0] - anchors[0][0]) + 1
    assert anchor_distance > exemplar.shape[1], "anchors must demand a real stretch"
    return exemplar, exemplar_bbox, anchors, horizontal, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the multicolor exemplar along the two-anchor axis so its endpoints land exactly on the anchors.

    Input structure:
    - The grid has a uniform background, one small multicolor exemplar rectangle, and two isolated same-color anchor pixels.
    - The exemplar is a solid rectangle whose column sequence contains exactly one adjacent repeated-column pair.
    - The anchors are horizontally or vertically aligned and are farther apart than the exemplar's long side.

    Rule:
    - Read the anchor axis. When the anchors are vertical, use the exemplar in a 90-degree clockwise placed orientation.
    - Find the exemplar's unique repeated column and extend only that elastic run until the exemplar length matches the inclusive anchor distance.
    - Clear the exemplar's original location and place the stretched result so its first and last cells cover the ordered anchor pair.

    Key insight:
    - The target length is not drawn explicitly; it is implied by the anchor spacing.
    - This is not geometric scaling: only the single self-repeating band absorbs the extra length while the fixed side bands keep their order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    exemplar, exemplar_bbox, anchors, horizontal, bg = _detect_puzzle(g)
    target_len = (anchors[1][1] - anchors[0][1] if horizontal else anchors[1][0] - anchors[0][0]) + 1
    stretched = _stretch_exemplar(exemplar, target_len)
    placed = stretched if horizontal else np.rot90(stretched, k=3)

    out = g.copy()
    r0, c0, r1, c1 = exemplar_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = bg

    if horizontal:
        row = anchors[0][0]
        col = anchors[0][1]
        assert placed.shape == (1, target_len)
        out[row : row + 1, col : col + target_len] = placed
    else:
        row = anchors[0][0]
        col = anchors[0][1]
        assert placed.shape == (target_len, 1)
        out[row : row + target_len, col : col + 1] = placed
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_sep = ar1 + gap < br0 or br1 + gap < ar0
    col_sep = ac1 + gap < bc0 or bc1 + gap < ac0
    return row_sep or col_sep


def _sample_sequence(base_len: int, palette: list[int]) -> tuple[list[int], int]:
    for _ in range(300):
        repeat_i = random.randint(1, base_len - 3)
        seq: list[int] = []
        for i in range(base_len):
            if i == repeat_i + 1:
                seq.append(seq[-1])
                continue
            choices = list(palette)
            if seq:
                choices = [c for c in choices if c != seq[-1]]
            seq.append(int(random.choice(choices)))

        repeat_positions = [i for i in range(base_len - 1) if seq[i] == seq[i + 1]]
        if repeat_positions != [repeat_i]:
            continue
        if len(set(seq)) < 3:
            continue
        if seq == list(reversed(seq)):
            continue
        return seq, repeat_i
    raise RuntimeError("failed to sample a non-degenerate exemplar sequence")


def _choose_anchor_color(palette: list[int]) -> int:
    if random.random() < 0.5:
        return int(random.choice(palette))
    other = [c for c in NON_BG_COLORS if c not in palette]
    return int(random.choice(other if other else palette))


def _validate_generated_input(grid: np.ndarray) -> None:
    exemplar, _bbox, anchors, horizontal, _bg = _detect_puzzle(grid)
    seq = exemplar[0].tolist()
    assert seq != list(reversed(seq)), "reversed placement would become ambiguous"
    repeat_i = _find_unique_repeat_pair(exemplar)
    assert 0 < repeat_i < exemplar.shape[1] - 2

    out = main(grid)
    if horizontal:
        row = anchors[0][0]
        seg = out[row, anchors[0][1] : anchors[1][1] + 1]
        assert seg.shape[0] > exemplar.shape[1]
    else:
        col = anchors[0][1]
        seg = out[anchors[0][0] : anchors[1][0] + 1, col]
        assert seg.shape[0] > exemplar.shape[1]

    non_bg_out = np.count_nonzero(out != BACKGROUND)
    assert non_bg_out == len(seg), "output should contain only the stretched exemplar"


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.random() < 0.5
    base_len = random.randint(5, 8)
    extra = random.randint(2, 10)
    target_len = base_len + extra

    min_h = max(10, target_len + 3 if not horizontal else 10)
    min_w = max(12, target_len + 3 if horizontal else base_len + 5)
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    palette = random.sample(NON_BG_COLORS, k=random.randint(3, 5))
    seq, _repeat_i = _sample_sequence(base_len, palette)
    exemplar = np.array([seq], dtype=int)
    anchor_color = _choose_anchor_color(palette)

    if horizontal:
        anchor_row = random.randint(1, h - 2)
        anchor_col = random.randint(1, w - target_len - 1)
        anchor_a = (anchor_row, anchor_col)
        anchor_b = (anchor_row, anchor_col + target_len - 1)
        anchor_span = (anchor_row, anchor_col, anchor_row, anchor_col + target_len - 1)
    else:
        anchor_col = random.randint(1, w - 2)
        anchor_row = random.randint(1, h - target_len - 1)
        anchor_a = (anchor_row, anchor_col)
        anchor_b = (anchor_row + target_len - 1, anchor_col)
        anchor_span = (anchor_row, anchor_col, anchor_row + target_len - 1, anchor_col)

    exemplar_bbox = None
    for _ in range(300):
        ex_row = random.randint(1, h - 2)
        ex_col = random.randint(1, w - base_len - 1)
        bbox = (ex_row, ex_col, ex_row, ex_col + base_len - 1)
        if _boxes_separated(bbox, anchor_span, gap=1):
            exemplar_bbox = bbox
            break
    if exemplar_bbox is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    ex_r0, ex_c0, _ex_r1, ex_c1 = exemplar_bbox
    grid[ex_r0, ex_c0 : ex_c1 + 1] = exemplar[0]
    grid[anchor_a] = anchor_color
    grid[anchor_b] = anchor_color

    try:
        _validate_generated_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-anchor elastic-stretch input")
