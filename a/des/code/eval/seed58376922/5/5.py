from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
POSITION_LABELS = {"TL": 2, "TR": 1, "BL": 3, "BR": 4}
LABEL_TO_POSITION = {value: key for key, value in POSITION_LABELS.items()}
POSITIONS = ("TL", "TR", "BL", "BR")
BODY_POOL = (5, 6, 7, 8, 9)
PATTERN_POOL = (1, 2, 3, 4)
TARGET_POOL = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _mode(values: list[int]) -> int:
    counts = Counter(values)
    best = max(counts.values())
    return min(v for v, cnt in counts.items() if cnt == best)


def _find_legend(grid: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    candidates = [
        (0, 0),
        (0, w - 2),
        (h - 4, 0),
        (h - 4, w - 2),
    ]
    expected = set(LABEL_TO_POSITION)
    for r0, c0 in candidates:
        if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 2 > w:
            continue
        patch = grid[r0 : r0 + 4, c0 : c0 + 2]
        if patch.shape != (4, 2):
            continue
        if not bool(np.all(patch != BG)):
            continue
        if set(map(int, patch[:, 0])) != expected:
            continue
        mapping: dict[str, int] = {}
        for label, target in patch:
            mapping[LABEL_TO_POSITION[int(label)]] = int(target)
        if len(mapping) != 4:
            continue
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[r0 : r0 + 4, c0 : c0 + 2] = True
        return mapping, legend_mask
    raise AssertionError("legend not found")


def _classify_piece(crop: np.ndarray) -> dict[str, object]:
    mask = crop != BG
    h, w = crop.shape
    row_left = [int(np.flatnonzero(mask[r])[0]) for r in range(h)]
    row_right = [int(np.flatnonzero(mask[r])[-1]) for r in range(h)]
    col_top = [int(np.flatnonzero(mask[:, c])[0]) for c in range(w)]
    col_bottom = [int(np.flatnonzero(mask[:, c])[-1]) for c in range(w)]

    left_mode = _mode(row_left)
    right_mode = _mode(row_right)
    top_mode = _mode(col_top)
    bottom_mode = _mode(col_bottom)

    left_tab = left_mode == 1
    right_tab = right_mode == w - 2
    top_tab = top_mode == 1
    bottom_tab = bottom_mode == h - 2

    core_r0 = int(top_tab)
    core_c0 = int(left_tab)
    core_r1 = h - 1 - int(bottom_tab)
    core_c1 = w - 1 - int(right_tab)

    sides: dict[str, str] = {}
    sides["left"] = "tab" if left_tab else ("flat" if bool(mask[core_r0 : core_r1 + 1, core_c0].all()) else "notch")
    sides["right"] = "tab" if right_tab else ("flat" if bool(mask[core_r0 : core_r1 + 1, core_c1].all()) else "notch")
    sides["top"] = "tab" if top_tab else ("flat" if bool(mask[core_r0, core_c0 : core_c1 + 1].all()) else "notch")
    sides["bottom"] = "tab" if bottom_tab else ("flat" if bool(mask[core_r1, core_c0 : core_c1 + 1].all()) else "notch")

    if sides["top"] == "flat" and sides["left"] == "flat":
        pos = "TL"
    elif sides["top"] == "flat" and sides["right"] == "flat":
        pos = "TR"
    elif sides["bottom"] == "flat" and sides["left"] == "flat":
        pos = "BL"
    elif sides["bottom"] == "flat" and sides["right"] == "flat":
        pos = "BR"
    else:
        raise AssertionError("piece position ambiguous")

    colors = [int(v) for v in crop[mask]]
    body = Counter(colors).most_common(1)[0][0]
    recolored = crop.copy()
    piece_mask = crop != BG
    recolored[piece_mask & (crop != int(body))] = -1

    return {
        "pos": pos,
        "crop": recolored,
        "body": int(body),
        "core_h": int(core_r1 - core_r0 + 1),
        "core_w": int(core_c1 - core_c0 + 1),
        "top_pad": int(top_tab),
        "left_pad": int(left_tab),
        "mask": piece_mask,
    }


def _paste_piece(out: np.ndarray, piece: dict[str, object], top: int, left: int, pattern_color: int) -> None:
    crop = np.asarray(piece["crop"], dtype=int).copy()
    crop[crop == -1] = int(pattern_color)
    mask = np.asarray(piece["mask"], dtype=bool)
    h, w = crop.shape
    view = out[top : top + h, left : left + w]
    conflict = mask & (view != BG) & (view != crop)
    if bool(conflict.any()):
        raise AssertionError("piece overlap conflict")
    view[mask] = crop[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble four jigsaw-edged rectangles into a single rectangle, then recolor
    each piece's internal pattern according to its assembled quadrant.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _find_legend(g)

    piece_comps = _components((g != BG) & ~legend_mask)
    assert len(piece_comps) == 4

    pieces: dict[str, dict[str, object]] = {}
    for comp in piece_comps:
        r0, c0, r1, c1 = _bbox_from_points(comp)
        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        info = _classify_piece(crop)
        pos = str(info["pos"])
        info["bbox"] = (r0, c0, r1, c1)
        pieces[pos] = info

    assert set(pieces) == set(POSITIONS)

    top_h = int(pieces["TL"]["core_h"])
    bottom_h = int(pieces["BL"]["core_h"])
    left_w = int(pieces["TL"]["core_w"])
    right_w = int(pieces["TR"]["core_w"])
    assert top_h == int(pieces["TR"]["core_h"])
    assert bottom_h == int(pieces["BR"]["core_h"])
    assert left_w == int(pieces["BL"]["core_w"])
    assert right_w == int(pieces["BR"]["core_w"])

    out = np.zeros((top_h + bottom_h, left_w + right_w), dtype=int)
    core_starts = {
        "TL": (0, 0),
        "TR": (0, left_w),
        "BL": (top_h, 0),
        "BR": (top_h, left_w),
    }
    for pos in POSITIONS:
        piece = pieces[pos]
        core_r, core_c = core_starts[pos]
        top = int(core_r - int(piece["top_pad"]))
        left = int(core_c - int(piece["left_pad"]))
        _paste_piece(out, piece, top, left, mapping[pos])
    return out


def _sample_segment(length: int) -> list[tuple[int, int]]:
    max_len = max(1, min(2, length - 2))
    seg_len = random.randint(1, max_len)
    start = random.randint(1, length - seg_len - 1)
    return [(start, start + seg_len)]


def _pattern_mask(height: int, width: int) -> np.ndarray:
    inner_h = height - 2
    inner_w = width - 2
    rr, cc = np.indices((inner_h, inner_w))
    patterns: list[np.ndarray] = []

    patterns.append(rr == inner_h // 2)
    patterns.append(cc == inner_w // 2)
    patterns.append(rr == cc)
    patterns.append(rr + cc == inner_h - 1 if inner_h == inner_w else rr == (cc * inner_h) // max(1, inner_w - 1))

    plus = (rr == inner_h // 2) | (cc == inner_w // 2)
    patterns.append(plus)

    dots = np.zeros((inner_h, inner_w), dtype=bool)
    dots[0, 0] = True
    dots[-1, -1] = True
    if inner_w > 2:
        dots[0, -1] = True
    patterns.append(dots)

    valid = [p for p in patterns if 0 < int(p.sum()) < inner_h * inner_w]
    inner = random.choice(valid).copy()
    if random.random() < 0.5:
        inner = np.fliplr(inner)
    if random.random() < 0.5:
        inner = np.flipud(inner)

    mask = np.zeros((height, width), dtype=bool)
    mask[1:-1, 1:-1] = inner
    return mask


def _build_piece(
    height: int,
    width: int,
    body_color: int,
    pattern_color: int,
    sides: dict[str, tuple[str, list[tuple[int, int]]]],
) -> np.ndarray:
    top_pad = 1 if sides["top"][0] == "tab" else 0
    bottom_pad = 1 if sides["bottom"][0] == "tab" else 0
    left_pad = 1 if sides["left"][0] == "tab" else 0
    right_pad = 1 if sides["right"][0] == "tab" else 0

    core = np.full((height, width), body_color, dtype=int)
    if sides["top"][0] == "notch":
        for start, end in sides["top"][1]:
            core[0, start:end] = BG
    if sides["bottom"][0] == "notch":
        for start, end in sides["bottom"][1]:
            core[-1, start:end] = BG
    if sides["left"][0] == "notch":
        for start, end in sides["left"][1]:
            core[start:end, 0] = BG
    if sides["right"][0] == "notch":
        for start, end in sides["right"][1]:
            core[start:end, -1] = BG

    piece = np.zeros((height + top_pad + bottom_pad, width + left_pad + right_pad), dtype=int)
    r0 = top_pad
    c0 = left_pad
    piece[r0 : r0 + height, c0 : c0 + width] = core

    if sides["top"][0] == "tab":
        for start, end in sides["top"][1]:
            piece[0, c0 + start : c0 + end] = body_color
    if sides["bottom"][0] == "tab":
        for start, end in sides["bottom"][1]:
            piece[-1, c0 + start : c0 + end] = body_color
    if sides["left"][0] == "tab":
        for start, end in sides["left"][1]:
            piece[r0 + start : r0 + end, 0] = body_color
    if sides["right"][0] == "tab":
        for start, end in sides["right"][1]:
            piece[r0 + start : r0 + end, -1] = body_color

    piece_core = piece[r0 : r0 + height, c0 : c0 + width]
    piece_core[_pattern_mask(height, width)] = pattern_color
    return piece


def _legend_block(targets: dict[str, int]) -> np.ndarray:
    rows = [(POSITION_LABELS[pos], targets[pos]) for pos in POSITIONS]
    random.shuffle(rows)
    return np.array(rows, dtype=int)


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = occupied.shape
    occupied[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)] = True


def _place_random_rect(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    candidates = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if bool(occupied[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)].any()):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _assemble_expected(
    pieces: dict[str, np.ndarray],
    core_sizes: dict[str, tuple[int, int]],
    targets: dict[str, int],
) -> np.ndarray:
    top_h = core_sizes["TL"][0]
    bottom_h = core_sizes["BL"][0]
    left_w = core_sizes["TL"][1]
    right_w = core_sizes["TR"][1]
    out = np.zeros((top_h + bottom_h, left_w + right_w), dtype=int)

    starts = {"TL": (0, 0), "TR": (0, left_w), "BL": (top_h, 0), "BR": (top_h, left_w)}
    for pos, piece in pieces.items():
        parsed = _classify_piece(piece)
        top = starts[pos][0] - int(parsed["top_pad"])
        left = starts[pos][1] - int(parsed["left_pad"])
        _paste_piece(out, parsed, top, left, targets[pos])
    return out


def _gen_attempt() -> np.ndarray | None:
    left_w = random.randint(5, 7)
    right_w = random.randint(5, 7)
    top_h = random.randint(5, 7)
    bottom_h = random.randint(5, 7)

    body_colors = random.sample(BODY_POOL, 4)
    pattern_colors = [random.choice([c for c in PATTERN_POOL if c != body_colors[i]]) for i in range(4)]
    targets = {
        pos: random.choice([c for c in TARGET_POOL if c != body_colors[i]])
        for i, pos in enumerate(POSITIONS)
    }

    sides = {
        pos: {
            "top": ("flat", []),
            "bottom": ("flat", []),
            "left": ("flat", []),
            "right": ("flat", []),
        }
        for pos in POSITIONS
    }

    seg = _sample_segment(top_h)
    if random.random() < 0.5:
        sides["TL"]["right"] = ("tab", seg)
        sides["TR"]["left"] = ("notch", seg)
    else:
        sides["TL"]["right"] = ("notch", seg)
        sides["TR"]["left"] = ("tab", seg)

    seg = _sample_segment(bottom_h)
    if random.random() < 0.5:
        sides["BL"]["right"] = ("tab", seg)
        sides["BR"]["left"] = ("notch", seg)
    else:
        sides["BL"]["right"] = ("notch", seg)
        sides["BR"]["left"] = ("tab", seg)

    seg = _sample_segment(left_w)
    if random.random() < 0.5:
        sides["TL"]["bottom"] = ("tab", seg)
        sides["BL"]["top"] = ("notch", seg)
    else:
        sides["TL"]["bottom"] = ("notch", seg)
        sides["BL"]["top"] = ("tab", seg)

    seg = _sample_segment(right_w)
    if random.random() < 0.5:
        sides["TR"]["bottom"] = ("tab", seg)
        sides["BR"]["top"] = ("notch", seg)
    else:
        sides["TR"]["bottom"] = ("notch", seg)
        sides["BR"]["top"] = ("tab", seg)

    core_sizes = {
        "TL": (top_h, left_w),
        "TR": (top_h, right_w),
        "BL": (bottom_h, left_w),
        "BR": (bottom_h, right_w),
    }
    pieces = {
        pos: _build_piece(core_sizes[pos][0], core_sizes[pos][1], body_colors[i], pattern_colors[i], sides[pos])
        for i, pos in enumerate(POSITIONS)
    }

    grid_h = random.randint(22, 30)
    grid_w = random.randint(22, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    legend = _legend_block(targets)
    corner = random.choice(POSITIONS)
    if corner == "TL":
        lr0, lc0 = 0, 0
    elif corner == "TR":
        lr0, lc0 = 0, grid_w - 2
    elif corner == "BL":
        lr0, lc0 = grid_h - 4, 0
    else:
        lr0, lc0 = grid_h - 4, grid_w - 2
    grid[lr0 : lr0 + 4, lc0 : lc0 + 2] = legend
    _mark_rect(occupied, lr0, lc0, 4, 2)

    for pos in sorted(POSITIONS, key=lambda key: pieces[key].shape[0] * pieces[key].shape[1], reverse=True):
        piece = pieces[pos]
        place = _place_random_rect(occupied, piece.shape[0], piece.shape[1])
        if place is None:
            return None
        r0, c0 = place
        view = grid[r0 : r0 + piece.shape[0], c0 : c0 + piece.shape[1]]
        mask = piece != BG
        view[mask] = piece[mask]
        _mark_rect(occupied, r0, c0, piece.shape[0], piece.shape[1])

    expected = _assemble_expected(pieces, core_sizes, targets)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
