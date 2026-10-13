from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
ARROW = 9
PALETTE = [1, 2, 3, 4, 6, 7, 8]

SIDE_FROM_DIR = {
    "up": "bottom",
    "down": "top",
    "left": "right",
    "right": "left",
}

ARROW_MASKS = {
    "up": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _monochrome_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components(grid == int(color)):
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "area": int(coords.shape[0]),
                "mask": _tight_mask(coords),
            }
        )
    return comps


def _arrow_direction(mask: np.ndarray) -> str | None:
    if mask.shape != (3, 3) or int(mask.sum()) != 5:
        return None
    for direction, pattern in ARROW_MASKS.items():
        if np.array_equal(mask, pattern):
            return direction
    return None


def _find_arrow_components(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[tuple[int, list[dict]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        comps = _monochrome_components(grid, color)
        if not comps:
            continue
        infos = []
        ok = True
        for comp in comps:
            direction = _arrow_direction(comp["mask"])
            if direction is None:
                ok = False
                break
            infos.append({**comp, "direction": direction})
        if ok and infos:
            candidates.append((int(color), infos))
    assert candidates, "expected a dedicated arrow color"
    color, infos = max(candidates, key=lambda item: len(item[1]))
    return int(color), infos


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _pair_arrows_to_pieces(arrows: list[dict], pieces: list[dict]) -> list[tuple[dict, dict]]:
    unused = set(range(len(pieces)))
    pairs: list[tuple[dict, dict]] = []
    for arrow in arrows:
        ar0, ac0, ar1, ac1 = arrow["bbox"]
        best = None
        for idx in list(unused):
            piece = pieces[idx]
            pr0, pc0, pr1, pc1 = piece["bbox"]
            direction = arrow["direction"]
            score = None
            if direction == "down" and pr0 > ar1 and _intervals_overlap(pc0, pc1, ac0, ac1):
                score = pr0 - ar1
            elif direction == "up" and pr1 < ar0 and _intervals_overlap(pc0, pc1, ac0, ac1):
                score = ar0 - pr1
            elif direction == "right" and pc0 > ac1 and _intervals_overlap(pr0, pr1, ar0, ar1):
                score = pc0 - ac1
            elif direction == "left" and pc1 < ac0 and _intervals_overlap(pr0, pr1, ar0, ar1):
                score = ac0 - pc1
            if score is None:
                continue
            key = (int(score), idx)
            if best is None or key < best:
                best = key
        assert best is not None, "every arrow must point to one piece"
        idx = int(best[1])
        unused.remove(idx)
        pairs.append((arrow, pieces[idx]))
    assert not unused, "expected one arrow per piece"
    return pairs


def _extract_slots(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    frame_coords = np.argwhere(grid == int(frame_color))
    assert frame_coords.size > 0
    fr0, fc0, fr1, fc1 = _bbox_from_coords(frame_coords)
    crop = np.asarray(grid[fr0 : fr1 + 1, fc0 : fc1 + 1], dtype=int)

    slots: list[dict] = []
    for coords in _components(crop == int(bg)):
        touch = set()
        if np.any(coords[:, 0] == 0):
            touch.add("top")
        if np.any(coords[:, 0] == crop.shape[0] - 1):
            touch.add("bottom")
        if np.any(coords[:, 1] == 0):
            touch.add("left")
        if np.any(coords[:, 1] == crop.shape[1] - 1):
            touch.add("right")
        if len(touch) != 1:
            continue

        side = next(iter(touch))
        mask = _tight_mask(coords)
        rr0, cc0, rr1, cc1 = _bbox_from_coords(coords)
        global_coords = coords + np.array([[fr0, fc0]])

        if side == "top":
            opening = sorted(int(c) for c in coords[coords[:, 0] == 0, 1])
            s0, s1 = opening[0], opening[-1]
            p1 = (fr0 - 1, fc0 + s0 - 1)
            p2 = (fr0 - 1, fc0 + s1 + 1)
        elif side == "bottom":
            opening = sorted(int(c) for c in coords[coords[:, 0] == crop.shape[0] - 1, 1])
            s0, s1 = opening[0], opening[-1]
            p1 = (fr1 + 1, fc0 + s0 - 1)
            p2 = (fr1 + 1, fc0 + s1 + 1)
        elif side == "left":
            opening = sorted(int(r) for r in coords[coords[:, 1] == 0, 0])
            s0, s1 = opening[0], opening[-1]
            p1 = (fr0 + s0 - 1, fc0 - 1)
            p2 = (fr0 + s1 + 1, fc0 - 1)
        else:
            opening = sorted(int(r) for r in coords[coords[:, 1] == crop.shape[1] - 1, 0])
            s0, s1 = opening[0], opening[-1]
            p1 = (fr0 + s0 - 1, fc1 + 1)
            p2 = (fr0 + s1 + 1, fc1 + 1)

        filter_color = int(grid[p1])
        assert filter_color != int(bg) and filter_color != int(frame_color)
        assert int(grid[p2]) == filter_color
        slots.append(
            {
                "side": side,
                "coords": global_coords,
                "bbox": (fr0 + rr0, fc0 + cc0, fr0 + rr1, fc0 + cc1),
                "mask": mask,
                "filter_color": filter_color,
            }
        )
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each outside fragment into its matching side notch, then recolor it by the notch's filter markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    arrow_color, arrows = _find_arrow_components(g, bg)
    remaining = [int(v) for v in np.unique(g) if int(v) not in {int(bg), int(arrow_color)}]
    assert remaining, "expected a frame color"
    frame_color = max(remaining, key=lambda color: int(np.sum(g == int(color))))

    slots = _extract_slots(g, bg, frame_color)
    pieces: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {int(bg), int(frame_color), int(arrow_color)}):
        for comp in _monochrome_components(g, color):
            if int(comp["area"]) > 1:
                pieces.append(comp)

    assert len(arrows) == len(pieces) == len(slots), "expected one arrow and one slot per piece"
    arrow_piece_pairs = _pair_arrows_to_pieces(arrows, pieces)

    used_slots: set[int] = set()
    assignments: list[tuple[dict, dict]] = []
    for arrow, piece in arrow_piece_pairs:
        need_side = SIDE_FROM_DIR[arrow["direction"]]
        candidates = []
        for idx, slot in enumerate(slots):
            if idx in used_slots or slot["side"] != need_side:
                continue
            if np.array_equal(slot["mask"], piece["mask"]):
                candidates.append((idx, slot))
        assert len(candidates) == 1, "piece-to-slot matching must be unique"
        idx, slot = candidates[0]
        used_slots.add(int(idx))
        assignments.append((piece, slot))

    out = g.copy()
    out[g == int(arrow_color)] = int(bg)
    for piece in pieces:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for _piece, slot in assignments:
        coords = slot["coords"]
        out[coords[:, 0], coords[:, 1]] = int(slot["filter_color"])
    return out


def _is_arrow_like(mask: np.ndarray) -> bool:
    return _arrow_direction(mask) is not None


def _generate_base_shape(depth: int, width: int) -> np.ndarray:
    cells = {(0, c) for c in range(width)}
    max_area = depth * width
    target = random.randint(max(2, width), max_area)
    while len(cells) < target:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < depth and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        frontier = sorted(set(frontier))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((depth, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    tight = _tight_mask(np.argwhere(mask))
    if _is_arrow_like(tight):
        return _generate_base_shape(depth, width)
    return tight


def _orient_shape(base: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return base.copy()
    if side == "bottom":
        return np.flipud(base)
    if side == "left":
        return base.T
    return np.fliplr(base.T)


def _sample_intervals(side_len: int, count: int) -> list[tuple[int, int]] | None:
    choices = []
    for width in (1, 2, 3):
        for start in range(3, side_len - width - 2):
            end = start + width - 1
            if end <= side_len - 4:
                choices.append((start, width))
    if count == 1:
        return [random.choice(choices)] if choices else None

    pairs = []
    for i, first in enumerate(choices):
        s0, w0 = first
        e0 = s0 + w0 - 1
        for s1, w1 in choices[i + 1 :]:
            e1 = s1 + w1 - 1
            if e0 + 4 < s1 or e1 + 4 < s0:
                pairs.append([first, (s1, w1)])
    return random.choice(pairs) if pairs else None


def _slot_cells(side: str, fr0: int, fc0: int, fr1: int, fc1: int, start: int, mask: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    cells = []
    if side == "top":
        rr_base, cc_base = fr0, fc0 + start
    elif side == "bottom":
        rr_base, cc_base = fr1 - sh + 1, fc0 + start
    elif side == "left":
        rr_base, cc_base = fr0 + start, fc0
    else:
        rr_base, cc_base = fr0 + start, fc1 - sw + 1
    for rr, cc in np.argwhere(mask):
        cells.append((int(rr_base + rr), int(cc_base + cc)))
    return cells


def _marker_positions(side: str, fr0: int, fc0: int, fr1: int, fc1: int, start: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (fr0 - 1, fc0 + start - 1), (fr0 - 1, fc0 + start + width)
    if side == "bottom":
        return (fr1 + 1, fc0 + start - 1), (fr1 + 1, fc0 + start + width)
    if side == "left":
        return (fr0 + start - 1, fc0 - 1), (fr0 + start + width, fc0 - 1)
    return (fr0 + start - 1, fc1 + 1), (fr0 + start + width, fc1 + 1)


def _piece_anchor(side: str, fr0: int, fc0: int, fr1: int, fc1: int, start: int, mask: np.ndarray, gap: int = 2) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        return fr0 - gap - sh, fc0 + start
    if side == "bottom":
        return fr1 + gap + 1, fc0 + start
    if side == "left":
        return fr0 + start, fc0 - gap - sw
    return fr0 + start, fc1 + gap + 1


def _arrow_anchor(side: str, piece_r: int, piece_c: int, piece_mask: np.ndarray) -> tuple[int, int]:
    sh, sw = piece_mask.shape
    if side == "top":
        return piece_r - 3, piece_c + sw // 2 - 1
    if side == "bottom":
        return piece_r + sh, piece_c + sw // 2 - 1
    if side == "left":
        return piece_r + sh // 2 - 1, piece_c - 3
    return piece_r + sh // 2 - 1, piece_c + sw


def _draw_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    target = grid[r0 : r0 + h, c0 : c0 + w]
    if np.any(target[patch] != BG):
        return False
    target[patch] = int(color)
    return True


def _slot_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(int(v) for v in mask.shape), mask.tobytes())


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    thickness = random.choice([3, 4])
    outer_h = random.randint(12, 14)
    outer_w = random.randint(12, 14)
    H = random.randint(outer_h + 16, min(30, outer_h + 18))
    W = random.randint(outer_w + 16, min(30, outer_w + 18))
    fr0 = random.randint(8, H - outer_h - 8)
    fc0 = random.randint(8, W - outer_w - 8)
    fr1 = fr0 + outer_h - 1
    fc1 = fc0 + outer_w - 1

    sides = ["top", "bottom", "left", "right"]
    side_counts = {side: 1 for side in sides}
    if random.random() < 0.6:
        side_counts[random.choice(sides)] = 2

    side_lengths = {
        "top": outer_w,
        "bottom": outer_w,
        "left": outer_h,
        "right": outer_h,
    }

    interval_map: dict[str, list[tuple[int, int]]] = {}
    for side in sides:
        intervals = _sample_intervals(side_lengths[side], side_counts[side])
        if intervals is None:
            return None
        interval_map[side] = intervals

    slot_specs = []
    for side in sides:
        seen_keys: set[tuple[tuple[int, int], bytes]] = set()
        for start, width in sorted(interval_map[side]):
            for _ in range(40):
                base = _generate_base_shape(thickness - 1, width)
                oriented = _orient_shape(base, side)
                key = _slot_key(oriented)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                slot_specs.append({"side": side, "start": start, "width": width, "mask": oriented})
                break
            else:
                return None

    n_slots = len(slot_specs)
    piece_colors = random.sample(PALETTE, n_slots)
    filter_colors = piece_colors[1:] + piece_colors[:1]
    random.shuffle(slot_specs)
    for idx, spec in enumerate(slot_specs):
        spec["piece_color"] = piece_colors[idx]
        spec["filter_color"] = filter_colors[idx]

    base = np.full((H, W), BG, dtype=int)
    base[fr0 : fr1 + 1, fc0 : fc1 + 1] = FRAME
    base[fr0 + thickness : fr1 - thickness + 1, fc0 + thickness : fc1 - thickness + 1] = BG

    for spec in slot_specs:
        coords = _slot_cells(spec["side"], fr0, fc0, fr1, fc1, spec["start"], spec["mask"])
        for r, c in coords:
            base[r, c] = BG
        m1, m2 = _marker_positions(spec["side"], fr0, fc0, fr1, fc1, spec["start"], spec["width"])
        if min(m1[0], m1[1], m2[0], m2[1]) < 0 or m1[0] >= H or m2[0] >= H or m1[1] >= W or m2[1] >= W:
            return None
        if base[m1] != BG or base[m2] != BG:
            return None
        base[m1] = spec["filter_color"]
        base[m2] = spec["filter_color"]
        spec["coords"] = coords

    out = base.copy()
    for spec in slot_specs:
        for r, c in spec["coords"]:
            out[r, c] = spec["filter_color"]

    inp = base.copy()
    for spec in slot_specs:
        piece_r, piece_c = _piece_anchor(spec["side"], fr0, fc0, fr1, fc1, spec["start"], spec["mask"])
        if not _draw_patch(inp, spec["mask"], piece_r, piece_c, spec["piece_color"]):
            return None
        arrow_dir = {
            "top": "down",
            "bottom": "up",
            "left": "right",
            "right": "left",
        }[spec["side"]]
        arrow_r, arrow_c = _arrow_anchor(spec["side"], piece_r, piece_c, spec["mask"])
        if not _draw_patch(inp, ARROW_MASKS[arrow_dir], arrow_r, arrow_c, ARROW):
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        sample = _generate_case()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
