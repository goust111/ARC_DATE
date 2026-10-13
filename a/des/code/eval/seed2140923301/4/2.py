from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_target_frame(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    for color in sorted(set(map(int, grid.ravel())) - {BG}):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            if not (
                (r0 == 0 and c0 == 0)
                or (r0 == 0 and c1 == w - 1)
                or (r1 == h - 1 and c0 == 0)
                or (r1 == h - 1 and c1 == w - 1)
            ):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(patch[border] == color):
                continue
            if not np.all(patch[~border] == BG):
                continue
            candidates.append((hh * ww, color, comp))

    assert len(candidates) == 1, "expected a unique corner target frame"
    _area, _color, comp = candidates[0]
    r0, c0, r1, c1 = _bbox(comp)
    mask = np.zeros_like(grid, dtype=bool)
    mask[tuple(comp.T)] = True
    return (r1 - r0 + 1, c1 - c0 + 1), mask


def _extract_pieces(grid: np.ndarray, frame_mask: np.ndarray) -> list[dict]:
    work = np.asarray(grid, dtype=int).copy()
    work[frame_mask] = BG
    pieces: list[dict] = []
    for comp in _components(work != BG):
        r0, c0, r1, c1 = _bbox(comp)
        colors = [int(work[r, c]) for r, c in comp]
        counts: dict[int, int] = {}
        for color in colors:
            counts[color] = counts.get(color, 0) + 1
        body_color = max(counts.items(), key=lambda item: item[1])[0]

        cells: list[tuple[int, int, int]] = []
        seam_sets: dict[int, set[tuple[int, int]]] = {}
        for r, c in comp:
            rr = int(r - r0)
            cc = int(c - c0)
            color = int(work[r, c])
            cells.append((rr, cc, color))
            if color != body_color:
                seam_sets.setdefault(color, set()).add((rr, cc))

        pieces.append(
            {
                "cells": cells,
                "body": int(body_color),
                "seams": seam_sets,
            }
        )

    assert len(pieces) >= 2, "expected multiple puzzle pieces"
    return pieces


def _seam_delta(
    seam_a: set[tuple[int, int]], seam_b: set[tuple[int, int]]
) -> tuple[int, int]:
    pts_a = sorted(seam_a)
    pts_b = sorted(seam_b)
    assert len(pts_a) == len(pts_b) and pts_a, "paired seams must be non-empty and equal-sized"
    ar0, ac0 = pts_a[0]
    br0, bc0 = pts_b[0]
    norm_a = {(r - ar0, c - ac0) for r, c in pts_a}
    norm_b = {(r - br0, c - bc0) for r, c in pts_b}
    assert norm_a == norm_b, "paired seams must share the same toothed outline"
    return ar0 - br0, ac0 - bc0


def _piece_positions(pieces: list[dict]) -> list[tuple[int, int]]:
    seam_to_parts: dict[int, list[tuple[int, set[tuple[int, int]]]]] = {}
    for idx, piece in enumerate(pieces):
        for color, pts in piece["seams"].items():
            seam_to_parts.setdefault(int(color), []).append((idx, pts))

    adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in pieces]
    for color, parts in seam_to_parts.items():
        assert len(parts) == 2, f"seam color {color} must appear on exactly two pieces"
        (a, seam_a), (b, seam_b) = parts
        dr, dc = _seam_delta(seam_a, seam_b)
        adj[a].append((b, (dr, dc)))
        adj[b].append((a, (-dr, -dc)))

    pos: list[tuple[int, int] | None] = [None] * len(pieces)
    pos[0] = (0, 0)
    q: deque[int] = deque([0])
    while q:
        cur = q.popleft()
        cr, cc = pos[cur]
        assert cr is not None and cc is not None
        for nxt, (dr, dc) in adj[cur]:
            want = (int(cr + dr), int(cc + dc))
            if pos[nxt] is None:
                pos[nxt] = want
                q.append(nxt)
            else:
                assert pos[nxt] == want, "inconsistent seam constraints"

    assert all(p is not None for p in pos), "all pieces must connect through seam pairings"
    return [(int(r), int(c)) for r, c in pos]


def _assemble(pieces: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    min_r = min(pr + r for (pr, _pc), piece in zip(positions, pieces, strict=True) for r, _c, _col in piece["cells"])
    min_c = min(pc + c for (_pr, pc), piece in zip(positions, pieces, strict=True) for _r, c, _col in piece["cells"])
    max_r = max(pr + r for (pr, _pc), piece in zip(positions, pieces, strict=True) for r, _c, _col in piece["cells"])
    max_c = max(pc + c for (_pr, pc), piece in zip(positions, pieces, strict=True) for _r, c, _col in piece["cells"])
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)

    for (pr, pc), piece in zip(positions, pieces, strict=True):
        for r, c, color in piece["cells"]:
            rr = int(pr + r - min_r)
            cc = int(pc + c - min_c)
            if out[rr, cc] == BG:
                out[rr, cc] = int(color)
            else:
                assert out[rr, cc] == int(color), "only same-colored seam pixels may overlap"

    assert np.all(out != BG), "assembled rectangle must be gap-free"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered jigsaw strips into their original rectangle using seam-color pairings.

    Input structure:
    - Background is black.
    - Each irregular fragment has one dominant body color plus one or two seam colors on toothed edges.
    - A small grey rectangular frame in a corner marks the target output size.

    Rule:
    - Match seam colors in pairs across different pieces.
    - For each paired seam color, align the identical toothed seam pixel sets.
    - Translate all pieces without rotation/flip, keep overlapping seam pixels in that seam color,
      and crop to the target rectangle.
    """
    g = np.asarray(grid, dtype=int)
    target_shape, frame_mask = _find_target_frame(g)
    pieces = _extract_pieces(g, frame_mask)
    positions = _piece_positions(pieces)
    out = _assemble(pieces, positions)
    assert out.shape == target_shape, "assembled size must match the target frame"
    return out


_GREY = 5
_GEN_RETRIES = 400
_PIECE_COUNT_OPTIONS = (2, 3, 4)
_BODY_COLOR_MIN = 3
_OUTER_SPAN_MIN = 3
_INNER_SPAN_MIN = 4


def _random_walk_sequence(length: int, starts: list[int], *, low: int, high: int, gap: int) -> list[list[int]]:
    out: list[list[int]] = []
    cur = list(starts)
    for _ in range(length):
        if out:
            next_vals = None
            for _retry in range(50):
                cand = [v + random.choice((-1, 0, 1)) for v in cur]
                if cand[0] < low or cand[-1] > high:
                    continue
                if any(cand[i] - cand[i - 1] < gap for i in range(1, len(cand))):
                    continue
                next_vals = cand
                break
            if next_vals is not None:
                cur = next_vals
        out.append(cur.copy())
    return out


def _make_vertical_case(piece_count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    piece_widths = []
    for idx in range(piece_count):
        if idx == 0 or idx == piece_count - 1:
            piece_widths.append(random.randint(3, 6))
        else:
            piece_widths.append(random.randint(4, 7))
    out_w = int(sum(piece_widths) - (piece_count - 1))
    out_h = random.randint(6, 12)

    base_seams = []
    x = piece_widths[0] - 1
    for idx in range(piece_count - 1):
        if idx:
            x += piece_widths[idx] - 1
        base_seams.append(int(x))
    seams_by_row = _random_walk_sequence(out_h, base_seams, low=2, high=out_w - 3, gap=3)

    colors = [c for c in range(1, 10) if c != _GREY]
    random.shuffle(colors)
    body_colors = colors[:piece_count]
    seam_colors = colors[piece_count : piece_count + piece_count - 1]

    out = np.zeros((out_h, out_w), dtype=int)
    pieces = [np.zeros((out_h, out_w), dtype=int) for _ in range(piece_count)]
    for r in range(out_h):
        seam_cols = seams_by_row[r]
        for idx in range(piece_count):
            left = 0 if idx == 0 else seam_cols[idx - 1]
            right = out_w - 1 if idx == piece_count - 1 else seam_cols[idx]
            pieces[idx][r, left : right + 1] = int(body_colors[idx])
            if idx > 0:
                pieces[idx][r, left] = int(seam_colors[idx - 1])
            if idx < piece_count - 1:
                pieces[idx][r, right] = int(seam_colors[idx])
            out[r, left : right + 1] = int(body_colors[idx])
        for seam_idx, col in enumerate(seam_cols):
            out[r, col] = int(seam_colors[seam_idx])

    return out, pieces


def _make_horizontal_case(piece_count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    piece_heights = []
    for idx in range(piece_count):
        if idx == 0 or idx == piece_count - 1:
            piece_heights.append(random.randint(3, 6))
        else:
            piece_heights.append(random.randint(4, 7))
    out_h = int(sum(piece_heights) - (piece_count - 1))
    out_w = random.randint(6, 12)

    base_seams = []
    y = piece_heights[0] - 1
    for idx in range(piece_count - 1):
        if idx:
            y += piece_heights[idx] - 1
        base_seams.append(int(y))
    seams_by_col = _random_walk_sequence(out_w, base_seams, low=2, high=out_h - 3, gap=3)

    colors = [c for c in range(1, 10) if c != _GREY]
    random.shuffle(colors)
    body_colors = colors[:piece_count]
    seam_colors = colors[piece_count : piece_count + piece_count - 1]

    out = np.zeros((out_h, out_w), dtype=int)
    pieces = [np.zeros((out_h, out_w), dtype=int) for _ in range(piece_count)]
    for c in range(out_w):
        seam_rows = seams_by_col[c]
        for idx in range(piece_count):
            top = 0 if idx == 0 else seam_rows[idx - 1]
            bottom = out_h - 1 if idx == piece_count - 1 else seam_rows[idx]
            pieces[idx][top : bottom + 1, c] = int(body_colors[idx])
            if idx > 0:
                pieces[idx][top, c] = int(seam_colors[idx - 1])
            if idx < piece_count - 1:
                pieces[idx][bottom, c] = int(seam_colors[idx])
            out[top : bottom + 1, c] = int(body_colors[idx])
        for seam_idx, row in enumerate(seam_rows):
            out[row, c] = int(seam_colors[seam_idx])

    return out, pieces


def _tight_crop(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _candidate_positions(
    canvas_shape: tuple[int, int], patch_shape: tuple[int, int], blocked: np.ndarray
) -> list[tuple[int, int]]:
    h, w = canvas_shape
    ph, pw = patch_shape
    positions: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            if blocked[r : r + ph, c : c + pw].any():
                continue
            positions.append((r, c))
    return positions


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, patch_shape: tuple[int, int]) -> None:
    ph, pw = patch_shape
    r1 = min(blocked.shape[0], r0 + ph + 1)
    c1 = min(blocked.shape[1], c0 + pw + 1)
    blocked[max(0, r0 - 1) : r1, max(0, c0 - 1) : c1] = True


def _scatter_patches(
    pieces: list[np.ndarray], target_shape: tuple[int, int]
) -> np.ndarray | None:
    target_h, target_w = target_shape
    canvas_h = random.randint(max(target_h + 8, 18), max(target_h + 14, 24))
    canvas_w = random.randint(max(target_w + 8, 18), max(target_w + 14, 24))
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)

    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        fr, fc = 0, 0
    elif corner == "tr":
        fr, fc = 0, canvas_w - target_w
    elif corner == "bl":
        fr, fc = canvas_h - target_h, 0
    else:
        fr, fc = canvas_h - target_h, canvas_w - target_w

    grid[fr, fc : fc + target_w] = _GREY
    grid[fr + target_h - 1, fc : fc + target_w] = _GREY
    grid[fr : fr + target_h, fc] = _GREY
    grid[fr : fr + target_h, fc + target_w - 1] = _GREY
    _mark_blocked(blocked, fr, fc, (target_h, target_w))

    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        patch = pieces[idx]
        positions = _candidate_positions(grid.shape, patch.shape, blocked)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        mask = patch != BG
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
        _mark_blocked(blocked, r0, c0, patch.shape)

    return grid


def _piece_is_valid(patch: np.ndarray) -> bool:
    colors = sorted(set(map(int, patch.ravel())) - {BG})
    if len(colors) < 2:
        return False
    counts = {color: int(np.sum(patch == color)) for color in colors}
    body = max(counts, key=counts.get)
    if counts[body] < _BODY_COLOR_MIN:
        return False
    if sum(color != body for color in colors) > 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.choice(_PIECE_COUNT_OPTIONS)
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        expected, full_pieces = _make_vertical_case(piece_count)
    else:
        expected, full_pieces = _make_horizontal_case(piece_count)

    pieces = [_tight_crop(piece) for piece in full_pieces]
    if not all(_piece_is_valid(piece) for piece in pieces):
        return None

    grid = _scatter_patches(pieces, expected.shape)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
