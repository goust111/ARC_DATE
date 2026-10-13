from __future__ import annotations

from collections import Counter
import random

import numpy as np


_BG = 0
_BOARD = 5
_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
_SIDES = ("top", "bottom", "left", "right")
_OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
_MAX_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_patch(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (int(r0), int(c0), int(r1), int(c1))


def _segment_1d(arr: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(arr)
    if idx.size == 0:
        return None
    start = int(idx[0])
    end = int(idx[-1])
    if end - start + 1 != int(idx.size):
        return None
    return start, end


def _dominant_color(values: np.ndarray) -> int:
    return int(Counter(map(int, values.reshape(-1))).most_common(1)[0][0])


def _find_board_color(grid: np.ndarray, bg: int) -> int:
    if int(_BOARD) != int(bg) and np.any(grid == int(_BOARD)):
        return int(_BOARD)

    best_color = None
    best_area = -1
    for color in sorted(set(map(int, grid.reshape(-1))) - {int(bg)}):
        for comp in _components(grid == int(color)):
            if len(comp) > best_area:
                best_area = len(comp)
                best_color = int(color)
    assert best_color is not None, "expected a board color"
    return int(best_color)


def _find_board_component(grid: np.ndarray, board_color: int) -> list[tuple[int, int]]:
    comps = _components(grid == int(board_color))
    assert comps, "expected board component"
    return max(comps, key=len)


def _parse_notches(
    grid: np.ndarray, board_color: int, board_bbox: tuple[int, int, int, int], bg: int
) -> list[dict]:
    br0, bc0, br1, bc1 = board_bbox
    sub = grid[br0 : br1 + 1, bc0 : bc1 + 1]
    comps = _components(sub != int(board_color))
    notches: list[dict] = []

    for comp in comps:
        lr0, lc0, lr1, lc1 = _bbox(comp)
        touches = []
        if lr0 == 0:
            touches.append("top")
        if lr1 == sub.shape[0] - 1:
            touches.append("bottom")
        if lc0 == 0:
            touches.append("left")
        if lc1 == sub.shape[1] - 1:
            touches.append("right")
        assert len(touches) == 1, "each notch should touch exactly one board side"
        side = touches[0]

        marker = [(r, c, int(sub[r, c])) for r, c in comp if int(sub[r, c]) not in {int(bg), int(board_color)}]
        assert len(marker) == 1, "each notch must contain exactly one marker pixel"
        mr, mc, marker_color = marker[0]

        notches.append(
            {
                "side": side,
                "marker_color": int(marker_color),
                "marker": (int(br0 + mr), int(bc0 + mc)),
                "bbox": (int(br0 + lr0), int(bc0 + lc0), int(br0 + lr1), int(bc0 + lc1)),
                "shape": (int(lr1 - lr0 + 1), int(lc1 - lc0 + 1)),
            }
        )

    assert notches, "expected at least one notch"
    return notches


def _parse_pieces(
    grid: np.ndarray, board_color: int, bg: int
) -> dict[int, dict]:
    pieces: dict[int, dict] = {}
    for comp in _components((grid != int(bg)) & (grid != int(board_color))):
        colors = Counter(int(grid[r, c]) for r, c in comp)
        singletons = [color for color, count in colors.items() if count == 1]
        if len(colors) != 2 or len(singletons) != 1:
            continue
        marker_color = int(singletons[0])
        body_color = next(int(color) for color, count in colors.items() if count > 1)
        patch, bbox = _extract_patch(grid, comp, bg)
        pieces[marker_color] = {
            "marker_color": marker_color,
            "body_color": int(body_color),
            "patch": patch,
            "bbox": bbox,
        }
    assert pieces, "expected scattered pieces"
    return pieces


def _decompose_piece(rot_patch: np.ndarray, bg: int, marker_color: int) -> dict | None:
    """Infer which side carries the rectangular protrusion in this rotation."""
    mask = rot_patch != int(bg)
    h, w = mask.shape
    marker_pos = np.argwhere(rot_patch == int(marker_color))
    assert marker_pos.shape == (1, 2), "piece marker should be unique"
    mr, mc = map(int, marker_pos[0])
    candidates: list[dict] = []

    d = 0
    while d < h and not bool(mask[d, :].all()):
        d += 1
    if 0 < d < h and bool(np.all(mask[d:, :])):
        seg = _segment_1d(mask[0, :])
        if seg is not None and mr == 0:
            valid = True
            for r in range(d):
                if _segment_1d(mask[r, :]) != seg:
                    valid = False
                    break
            if valid and seg[1] - seg[0] + 1 < w and seg[0] <= mc <= seg[1]:
                candidates.append({"side": "top", "protr_bbox": (0, seg[0], d - 1, seg[1])})

    d = h - 1
    while d >= 0 and not bool(mask[d, :].all()):
        d -= 1
    if 0 <= d < h - 1 and bool(np.all(mask[: d + 1, :])):
        seg = _segment_1d(mask[h - 1, :])
        if seg is not None and mr == h - 1:
            valid = True
            for r in range(d + 1, h):
                if _segment_1d(mask[r, :]) != seg:
                    valid = False
                    break
            if valid and seg[1] - seg[0] + 1 < w and seg[0] <= mc <= seg[1]:
                candidates.append({"side": "bottom", "protr_bbox": (d + 1, seg[0], h - 1, seg[1])})

    d = 0
    while d < w and not bool(mask[:, d].all()):
        d += 1
    if 0 < d < w and bool(np.all(mask[:, d:])):
        seg = _segment_1d(mask[:, 0])
        if seg is not None and mc == 0:
            valid = True
            for c in range(d):
                if _segment_1d(mask[:, c]) != seg:
                    valid = False
                    break
            if valid and seg[1] - seg[0] + 1 < h and seg[0] <= mr <= seg[1]:
                candidates.append({"side": "left", "protr_bbox": (seg[0], 0, seg[1], d - 1)})

    d = w - 1
    while d >= 0 and not bool(mask[:, d].all()):
        d -= 1
    if 0 <= d < w - 1 and bool(np.all(mask[:, : d + 1])):
        seg = _segment_1d(mask[:, w - 1])
        if seg is not None and mc == w - 1:
            valid = True
            for c in range(d + 1, w):
                if _segment_1d(mask[:, c]) != seg:
                    valid = False
                    break
            if valid and seg[1] - seg[0] + 1 < h and seg[0] <= mr <= seg[1]:
                candidates.append({"side": "right", "protr_bbox": (seg[0], d + 1, seg[1], w - 1)})

    if len(candidates) != 1:
        return None
    return candidates[0]


def _place_piece(
    out: np.ndarray,
    rot_patch: np.ndarray,
    body_color: int,
    target_bbox: tuple[int, int, int, int],
    protr_bbox: tuple[int, int, int, int],
    bg: int,
) -> None:
    pr0, pc0, _pr1, _pc1 = protr_bbox
    tr0, tc0, _tr1, _tc1 = target_bbox
    dr = int(tr0 - pr0)
    dc = int(tc0 - pc0)
    h, w = rot_patch.shape

    for r in range(h):
        for c in range(w):
            if int(rot_patch[r, c]) == int(bg):
                continue
            rr = int(dr + r)
            cc = int(dc + c)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "piece placement must stay in bounds"
            assert int(out[rr, cc]) == int(bg), "placed pieces should not overlap existing structure"
            out[rr, cc] = int(body_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered marked protrusion pieces into the matching notches of the central board.

    Input structure:
    - The background is black.
    - A large central board is a single grey filled rectangle with one or more rectangular edge notches cut out of it.
    - Each notch contains exactly one colored marker on its deepest edge, and each scattered piece is a two-color component: one solid body color plus one single-pixel marker color at the tip of its protrusion.

    Rule:
    - Detect the central grey board and read each notch by its marker color and rectangular notch shape.
    - Match every scattered piece to the notch with the same marker color.
    - Rotate the piece by 0/90/180/270 degrees so that its rectangular protrusion has the notch's dimensions and points into the notch from outside the board, then translate it so the protrusion fills the notch exactly.
    - Clear the original scattered pieces and cover both the notch marker and the piece marker with the piece body color.

    Key insight:
    - Marker colors solve the pairing, but the final placement still requires geometric reasoning: the correct rotation is the one whose protrusion rectangle is complementary to the notch on the corresponding board side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    board_color = _find_board_color(g, bg)
    board_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(board_color))]
    assert board_coords, "expected grey board pixels"
    board_bbox = _bbox(board_coords)

    notches = _parse_notches(g, board_color, board_bbox, bg)
    pieces = _parse_pieces(g, board_color, bg)
    assert len(notches) == len(pieces), "piece/notch count mismatch"

    out = np.full_like(g, int(bg))
    out[g == int(board_color)] = int(board_color)

    for notch in notches:
        marker_color = int(notch["marker_color"])
        assert marker_color in pieces, "missing matching piece"
        piece = pieces[marker_color]
        wanted_side = _OPPOSITE[str(notch["side"])]
        target_h = int(notch["bbox"][2] - notch["bbox"][0] + 1)
        target_w = int(notch["bbox"][3] - notch["bbox"][1] + 1)

        chosen_patch = None
        chosen_protr = None
        for k in range(4):
            rot_patch = np.rot90(piece["patch"], int(k))
            info = _decompose_piece(rot_patch, bg, marker_color)
            if info is None or info["side"] != wanted_side:
                continue
            pr0, pc0, pr1, pc1 = info["protr_bbox"]
            if (int(pr1 - pr0 + 1), int(pc1 - pc0 + 1)) != (target_h, target_w):
                continue
            chosen_patch = rot_patch
            chosen_protr = info["protr_bbox"]
            break

        assert chosen_patch is not None and chosen_protr is not None, "no valid rotation for piece"
        _place_piece(out, chosen_patch, int(piece["body_color"]), notch["bbox"], chosen_protr, bg)

    return out


def _build_piece_patch(
    notch_side: str,
    span: int,
    depth: int,
    piece_color: int,
    marker_color: int,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    protr_side = _OPPOSITE[str(notch_side)]

    if protr_side in {"top", "bottom"}:
        body_h = random.randint(1, 4)
        body_w = int(span + random.randint(1, 3))
        offset = random.randint(0, body_w - span)
        h = int(body_h + depth)
        patch = np.full((h, body_w), int(bg), dtype=int)
        if protr_side == "top":
            patch[depth:, :] = int(piece_color)
            patch[:depth, offset : offset + span] = int(piece_color)
            marker_r = 0
            protr_bbox = (0, offset, depth - 1, offset + span - 1)
        else:
            patch[:body_h, :] = int(piece_color)
            patch[body_h:, offset : offset + span] = int(piece_color)
            marker_r = h - 1
            protr_bbox = (body_h, offset, h - 1, offset + span - 1)
        marker_c = random.randint(offset, offset + span - 1)
        outward = body_h

    else:
        body_w = random.randint(1, 4)
        body_h = int(span + random.randint(1, 3))
        offset = random.randint(0, body_h - span)
        w = int(body_w + depth)
        patch = np.full((body_h, w), int(bg), dtype=int)
        if protr_side == "left":
            patch[:, depth:] = int(piece_color)
            patch[offset : offset + span, :depth] = int(piece_color)
            marker_c = 0
            protr_bbox = (offset, 0, offset + span - 1, depth - 1)
        else:
            patch[:, :body_w] = int(piece_color)
            patch[offset : offset + span, body_w:] = int(piece_color)
            marker_c = w - 1
            protr_bbox = (offset, body_w, offset + span - 1, w - 1)
        marker_r = random.randint(offset, offset + span - 1)
        outward = body_w

    patch[int(marker_r), int(marker_c)] = int(marker_color)
    return patch, protr_bbox, int(outward)


def _notch_marker_position(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return int(r1), random.randint(c0, c1)
    if side == "bottom":
        return int(r0), random.randint(c0, c1)
    if side == "left":
        return random.randint(r0, r1), int(c1)
    return random.randint(r0, r1), int(c0)


def _stamp_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    transparent: int | None = None,
) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            val = int(patch[r, c])
            if transparent is not None and val == int(transparent):
                continue
            grid[top + r, left + c] = val


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = patch.shape
    hh, ww = occupied.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    for r, c in np.argwhere(patch != int(_BG)):
        rr = int(top + r)
        cc = int(left + c)
        r0 = max(0, rr - pad)
        r1 = min(hh, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(ww, cc + pad + 1)
        if bool(np.any(occupied[r0:r1, c0:c1])):
            return False
    return True


def _mark_occupied(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r, c in np.argwhere(patch != int(_BG)):
        occupied[top + int(r), left + int(c)] = True


def _place_solved_piece(
    out: np.ndarray,
    patch: np.ndarray,
    protr_bbox: tuple[int, int, int, int],
    target_bbox: tuple[int, int, int, int],
    piece_color: int,
) -> None:
    pr0, pc0, _pr1, _pc1 = protr_bbox
    tr0, tc0, _tr1, _tc1 = target_bbox
    top = int(tr0 - pr0)
    left = int(tc0 - pc0)
    body_patch = np.where(patch == int(_BG), int(_BG), int(piece_color))
    _stamp_patch(out, body_patch, top, left, transparent=_BG)


def _generate_instance() -> tuple[np.ndarray, np.ndarray] | None:
    n_pieces = random.randint(2, 4)
    sides = random.sample(list(_SIDES), n_pieces)

    board_h = random.randint(9, 15)
    board_w = random.randint(9, 15)
    if min(board_h, board_w) < 7:
        return None

    notch_specs: list[dict] = []
    used_shapes: set[tuple[int, int]] = set()

    for side in sides:
        if side in {"top", "bottom"}:
            max_span = board_w - 2
            max_depth = min(4, board_h - 3)
            if max_span < 2 or max_depth < 1:
                return None
            shape_choices = [(d, s) for d in range(1, max_depth + 1) for s in range(2, max_span + 1)]
            shape_choices = [choice for choice in shape_choices if choice not in used_shapes]
            if not shape_choices:
                return None
            depth, span = random.choice(shape_choices)
            used_shapes.add((depth, span))
            start = random.randint(1, board_w - span - 1)
            if side == "top":
                bbox = (0, start, depth - 1, start + span - 1)
            else:
                bbox = (board_h - depth, start, board_h - 1, start + span - 1)
        else:
            max_span = board_h - 2
            max_depth = min(4, board_w - 3)
            if max_span < 2 or max_depth < 1:
                return None
            shape_choices = [(s, d) for s in range(2, max_span + 1) for d in range(1, max_depth + 1)]
            shape_choices = [choice for choice in shape_choices if choice not in used_shapes]
            if not shape_choices:
                return None
            span, depth = random.choice(shape_choices)
            used_shapes.add((span, depth))
            start = random.randint(1, board_h - span - 1)
            if side == "left":
                bbox = (start, 0, start + span - 1, depth - 1)
            else:
                bbox = (start, board_w - depth, start + span - 1, board_w - 1)

        notch_specs.append(
            {
                "side": side,
                "local_bbox": tuple(map(int, bbox)),
                "span": int(span),
                "depth": int(depth),
            }
        )

    notch_mask = np.zeros((board_h, board_w), dtype=bool)
    for notch in notch_specs:
        r0, c0, r1, c1 = notch["local_bbox"]
        notch_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    if len(_components(notch_mask)) != len(notch_specs):
        return None
    board_mask = ~notch_mask
    if len(_components(board_mask)) != 1:
        return None

    colors = random.sample(_PALETTE, 2 * n_pieces)
    piece_colors = colors[:n_pieces]
    marker_colors = colors[n_pieces:]

    piece_specs: list[dict] = []
    top_need = 0
    bottom_need = 0
    left_need = 0
    right_need = 0

    for notch, piece_color, marker_color in zip(notch_specs, piece_colors, marker_colors):
        patch, protr_bbox, outward = _build_piece_patch(
            notch["side"], notch["span"], notch["depth"], piece_color, marker_color, _BG
        )
        piece_specs.append(
            {
                "notch": notch,
                "piece_color": int(piece_color),
                "marker_color": int(marker_color),
                "solved_patch": patch,
                "protr_bbox": protr_bbox,
            }
        )
        if notch["side"] == "top":
            top_need = max(top_need, outward)
        elif notch["side"] == "bottom":
            bottom_need = max(bottom_need, outward)
        elif notch["side"] == "left":
            left_need = max(left_need, outward)
        else:
            right_need = max(right_need, outward)

    margin_top = top_need + random.randint(3, 6)
    margin_bottom = bottom_need + random.randint(3, 6)
    margin_left = left_need + random.randint(3, 6)
    margin_right = right_need + random.randint(3, 6)

    h = margin_top + board_h + margin_bottom
    w = margin_left + board_w + margin_right
    if h > 30 or w > 30:
        return None

    br0 = margin_top
    bc0 = margin_left
    br1 = br0 + board_h - 1
    bc1 = bc0 + board_w - 1

    inp = np.full((h, w), int(_BG), dtype=int)
    out = np.full((h, w), int(_BG), dtype=int)

    inp[br0 : br1 + 1, bc0 : bc1 + 1] = int(_BOARD)
    out[br0 : br1 + 1, bc0 : bc1 + 1] = int(_BOARD)

    for spec in piece_specs:
        lr0, lc0, lr1, lc1 = spec["notch"]["local_bbox"]
        gr0 = br0 + lr0
        gc0 = bc0 + lc0
        gr1 = br0 + lr1
        gc1 = bc0 + lc1
        inp[gr0 : gr1 + 1, gc0 : gc1 + 1] = int(_BG)
        mr, mc = _notch_marker_position((gr0, gc0, gr1, gc1), spec["notch"]["side"])
        inp[mr, mc] = int(spec["marker_color"])
        spec["target_bbox"] = (int(gr0), int(gc0), int(gr1), int(gc1))
        _place_solved_piece(out, spec["solved_patch"], spec["protr_bbox"], spec["target_bbox"], spec["piece_color"])

    occupied = np.zeros((h, w), dtype=bool)
    occupied[max(0, br0 - 1) : min(h, br1 + 2), max(0, bc0 - 1) : min(w, bc1 + 2)] = True

    order = list(range(len(piece_specs)))
    random.shuffle(order)
    order.sort(key=lambda idx: -int(np.count_nonzero(piece_specs[idx]["solved_patch"] != int(_BG))))

    for idx in order:
        spec = piece_specs[idx]
        rot_patch = np.rot90(spec["solved_patch"], random.randint(0, 3))
        positions = [(r, c) for r in range(h - rot_patch.shape[0] + 1) for c in range(w - rot_patch.shape[1] + 1)]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            if not _can_place_patch(occupied, rot_patch, top, left, pad=1):
                continue
            _stamp_patch(inp, rot_patch, top, left, transparent=_BG)
            _mark_occupied(occupied, rot_patch, top, left)
            placed = True
            break
        if not placed:
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        built = _generate_instance()
        if built is None:
            continue
        inp, expected = built
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return inp
    raise AssertionError("failed to generate a valid puzzle instance")
