from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_BOARD = 5
_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_SIDES = ("top", "right", "bottom", "left")
_MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(vals))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_from_coords(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool).copy()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, exclude: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in exclude or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(int(r), int(c))]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            coords_arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coords_arr)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                local[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "size": int(len(coords)),
                }
            )
    return comps


def _opposite(side: str) -> str:
    return {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[side]


def _side_candidates(mask: np.ndarray) -> set[str]:
    h, w = mask.shape
    row_counts = np.sum(mask, axis=1)
    col_counts = np.sum(mask, axis=0)
    max_line = int(max(int(row_counts.max()), int(col_counts.max())))
    sides: set[str] = set()
    for r, count in enumerate(row_counts):
        if int(count) != max_line:
            continue
        top_gap = int(r)
        bottom_gap = int(h - 1 - r)
        if top_gap < bottom_gap:
            sides.add("top")
        elif bottom_gap < top_gap:
            sides.add("bottom")
        else:
            sides.update(("top", "bottom"))
    for c, count in enumerate(col_counts):
        if int(count) != max_line:
            continue
        left_gap = int(c)
        right_gap = int(w - 1 - c)
        if left_gap < right_gap:
            sides.add("left")
        elif right_gap < left_gap:
            sides.add("right")
        else:
            sides.update(("left", "right"))
    return sides


def _edge_line_score(mask: np.ndarray, side: str) -> int:
    if side in ("top", "bottom"):
        counts = np.sum(mask, axis=1)
        best = -1
        for r, count in enumerate(counts):
            gap = r if side == "top" else mask.shape[0] - 1 - r
            best = max(best, int(count) * 100 - int(gap))
        return int(best)
    counts = np.sum(mask, axis=0)
    best = -1
    for c, count in enumerate(counts):
        gap = c if side == "left" else mask.shape[1] - 1 - c
        best = max(best, int(count) * 100 - int(gap))
    return int(best)


def _choose_rotation(mask: np.ndarray, target_side: str, target_shape: tuple[int, int]) -> np.ndarray:
    options: list[tuple[int, int, int, np.ndarray]] = []
    for k in range(4):
        rot = _crop_mask(np.rot90(mask, int(k)))
        if tuple(map(int, rot.shape)) != tuple(map(int, target_shape)):
            continue
        hits = 1 if target_side in _side_candidates(rot) else 0
        score = _edge_line_score(rot, target_side)
        options.append((hits, score, int(k), rot))
    if not options:
        raise AssertionError("piece must have at least one rotation matching the cavity shape")
    options.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return options[0][3]


def _label_color_for_cavity(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cavity_bbox: tuple[int, int, int, int],
    side: str,
    bg: int,
    board_color: int,
) -> int:
    br0, bc0, br1, bc1 = board_bbox
    r0, c0, r1, c1 = cavity_bbox
    cells: list[int] = []
    if side == "top":
        rr = br0 - 1
        for cc in range(bc0 + c0, bc0 + c1 + 1):
            if 0 <= rr < grid.shape[0]:
                val = int(grid[rr, cc])
                if val not in (bg, board_color):
                    cells.append(val)
    elif side == "bottom":
        rr = br1 + 1
        for cc in range(bc0 + c0, bc0 + c1 + 1):
            if 0 <= rr < grid.shape[0]:
                val = int(grid[rr, cc])
                if val not in (bg, board_color):
                    cells.append(val)
    elif side == "left":
        cc = bc0 - 1
        for rr in range(br0 + r0, br0 + r1 + 1):
            if 0 <= cc < grid.shape[1]:
                val = int(grid[rr, cc])
                if val not in (bg, board_color):
                    cells.append(val)
    else:
        cc = bc1 + 1
        for rr in range(br0 + r0, br0 + r1 + 1):
            if 0 <= cc < grid.shape[1]:
                val = int(grid[rr, cc])
                if val not in (bg, board_color):
                    cells.append(val)
    colors = sorted(set(cells))
    if len(colors) != 1:
        raise AssertionError("each cavity must expose exactly one nearby label color")
    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered color piece and insert it into the matching edge-open cavity of the central gray puzzle board.

    Input structure:
    - Background is black. A large gray monochrome component forms a central rectangular puzzle board whose bbox is the final output canvas.
    - Several black rectangular cavities are carved from the board; each cavity touches exactly one outer side of the board bbox, which determines its opening direction.
    - A single colored label pixel sits just outside each cavity opening, and a scattered monochrome piece of the same color appears elsewhere on the background.

    Rule:
    - Find the central board, crop to its bbox, and locate each black cavity touching one side of that bbox.
    - Pair a cavity with the scattered piece whose color matches the cavity's outside label pixel.
    - Rotate the piece so it matches the cavity bbox and so the densest row or column of the piece faces the cavity's closed bottom side (opposite the opening).
    - Paint the rotated piece into the cavity on the cropped board, leaving any uncovered cavity cells black.

    Key insight:
    - The label color identifies which piece belongs to a cavity, while the cavity opening direction fixes the required rotation by telling you which side of the piece must be the "heavy" end.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _monochrome_components(g, exclude={bg})
    if not comps:
        return g.copy()

    board = max(comps, key=lambda comp: comp["size"])
    board_color = int(board["color"])
    br0, bc0, br1, bc1 = board["bbox"]
    board_crop = np.asarray(g[br0 : br1 + 1, bc0 : bc1 + 1], dtype=int).copy()

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        if int(comp["color"]) == board_color:
            continue
        by_color.setdefault(int(comp["color"]), []).append(comp)
    for color in by_color:
        by_color[color].sort(key=lambda comp: (-comp["size"], comp["bbox"]))

    cavities = []
    for coords in _mask_components(board_crop == bg):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        touching = []
        if r0 == 0:
            touching.append("top")
        if r1 == board_crop.shape[0] - 1:
            touching.append("bottom")
        if c0 == 0:
            touching.append("left")
        if c1 == board_crop.shape[1] - 1:
            touching.append("right")
        if len(touching) != 1:
            continue
        cavities.append({"bbox": (r0, c0, r1, c1), "side": touching[0]})
    cavities.sort(key=lambda cav: cav["bbox"])

    out = board_crop.copy()
    for cavity in cavities:
        r0, c0, r1, c1 = cavity["bbox"]
        open_side = str(cavity["side"])
        label_color = _label_color_for_cavity(
            g,
            (br0, bc0, br1, bc1),
            (r0, c0, r1, c1),
            open_side,
            bg,
            board_color,
        )
        if label_color not in by_color:
            raise AssertionError("every label color must have a matching piece")
        piece = by_color[label_color][0]
        target_shape = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        target_side = _opposite(open_side)
        placed_mask = _choose_rotation(piece["mask"], target_side, target_shape)
        cavity_view = out[r0 : r1 + 1, c0 : c1 + 1]
        cavity_view[placed_mask] = int(label_color)

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rs = slice(max(0, dr), min(h, h + dr))
            rd = slice(max(0, -dr), min(h, h - dr))
            cs = slice(max(0, dc), min(w, w + dc))
            cd = slice(max(0, -dc), min(w, w - dc))
            out[rd, cd] |= mask[rs, cs]
    return out


def _base_bottom_heavy_mask(h: int, w: int) -> np.ndarray:
    if not (2 <= h < w):
        raise AssertionError("base piece must be wider than tall")
    mask = np.zeros((h, w), dtype=bool)
    spine = int(_RNG.integers(0, w))
    mask[:, spine] = True
    mask[h - 1, :] = True
    extra_cols = [c for c in range(w) if c != spine]
    _RNG.shuffle(extra_cols)
    take = int(_RNG.integers(0, min(3, len(extra_cols)) + 1))
    for col in extra_cols[:take]:
        start = int(_RNG.integers(1, h))
        mask[start:, col] = True
    return mask


def _make_piece_mask(base_shape: tuple[int, int], symmetric: bool) -> np.ndarray:
    h, w = map(int, base_shape)
    base = _base_bottom_heavy_mask(h, w)
    if not symmetric:
        return base
    sym = np.asarray(base | np.rot90(base, 2), dtype=bool)
    return _crop_mask(sym)


def _cavity_bbox(side: str, start: int, length: int, depth: int, bh: int, bw: int) -> tuple[int, int, int, int]:
    if side == "top":
        return 0, int(start), int(depth - 1), int(start + length - 1)
    if side == "bottom":
        return int(bh - depth), int(start), int(bh - 1), int(start + length - 1)
    if side == "left":
        return int(start), 0, int(start + length - 1), int(depth - 1)
    return int(start), int(bw - depth), int(start + length - 1), int(bw - 1)


def _sample_cavities(bh: int, bw: int, count: int) -> list[dict] | None:
    sides = list(_SIDES)
    _RNG.shuffle(sides)
    sides = sides[:count]
    occupied = np.zeros((bh, bw), dtype=bool)
    cavities: list[dict] = []
    for side in sides:
        placed = None
        for _ in range(120):
            if side in ("top", "bottom"):
                max_depth = min(4, bh - 3)
                if max_depth < 2:
                    return None
                depth = int(_RNG.integers(2, max_depth + 1))
                min_len = depth + 1
                max_len = min(depth + 3, bw - 2)
                if max_len < min_len:
                    continue
                length = int(_RNG.integers(min_len, max_len + 1))
                start = int(_RNG.integers(1, bw - length))
            else:
                max_depth = min(4, bw - 3)
                if max_depth < 2:
                    return None
                depth = int(_RNG.integers(2, max_depth + 1))
                min_len = depth + 1
                max_len = min(depth + 3, bh - 2)
                if max_len < min_len:
                    continue
                length = int(_RNG.integers(min_len, max_len + 1))
                start = int(_RNG.integers(1, bh - length))
            r0, c0, r1, c1 = _cavity_bbox(side, start, length, depth, bh, bw)
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placed = {
                "side": side,
                "bbox": (r0, c0, r1, c1),
                "shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
            }
            break
        if placed is None:
            return None
        cavities.append(placed)
    return cavities


def _final_rotation_for_target(target_side: str) -> int:
    return {"bottom": 0, "right": 1, "top": 2, "left": 3}[target_side]


def _paint_piece(board: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = bbox
    view = board[r0 : r1 + 1, c0 : c1 + 1]
    view[mask] = int(color)


def _find_slots(blocked: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = map(int, shape)
    slots: list[tuple[int, int]] = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            if not blocked[r : r + ph, c : c + pw].any():
                slots.append((int(r), int(c)))
    return slots


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = int(_RNG.integers(22, 29))
    grid_w = int(_RNG.integers(22, 29))
    max_bh = min(17, grid_h - 8)
    max_bw = min(17, grid_w - 8)
    if max_bh < 10 or max_bw < 10:
        return None
    bh = int(_RNG.integers(10, max_bh + 1))
    bw = int(_RNG.integers(10, max_bw + 1))
    board_r0 = (grid_h - bh) // 2
    board_c0 = (grid_w - bw) // 2

    cavity_count = int(_RNG.integers(2, 5))
    cavities = _sample_cavities(bh, bw, cavity_count)
    if cavities is None:
        return None

    colors = list(_PALETTE)
    _RNG.shuffle(colors)
    colors = colors[:cavity_count]

    inp = np.full((grid_h, grid_w), _BG, dtype=int)
    board_in = np.full((bh, bw), _BOARD, dtype=int)
    out = board_in.copy()

    pieces: list[dict] = []
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[board_r0 : board_r0 + bh, board_c0 : board_c0 + bw] = True

    asym_indices: list[int] = []
    for idx, cavity in enumerate(cavities):
        r0, c0, r1, c1 = cavity["bbox"]
        board_in[r0 : r1 + 1, c0 : c1 + 1] = _BG
        out[r0 : r1 + 1, c0 : c1 + 1] = _BG
        color = int(colors[idx])
        cavity["color"] = color

        if cavity["side"] == "top":
            label_pos = (board_r0 - 1, board_c0 + (c0 + c1) // 2)
        elif cavity["side"] == "bottom":
            label_pos = (board_r0 + bh, board_c0 + (c0 + c1) // 2)
        elif cavity["side"] == "left":
            label_pos = (board_r0 + (r0 + r1) // 2, board_c0 - 1)
        else:
            label_pos = (board_r0 + (r0 + r1) // 2, board_c0 + bw)
        inp[label_pos] = color
        occupied[label_pos] = True

        cavity_h, cavity_w = cavity["shape"]
        target_side = _opposite(str(cavity["side"]))
        if target_side in ("top", "bottom"):
            base_shape = (int(cavity_h), int(cavity_w))
        else:
            base_shape = (int(cavity_w), int(cavity_h))
        if not (base_shape[1] > base_shape[0] >= 2):
            return None

        symmetric = bool(_RNG.random() < 0.3)
        base_mask = _make_piece_mask(base_shape, symmetric)
        final_mask = _crop_mask(np.rot90(base_mask, _final_rotation_for_target(target_side)))
        if tuple(map(int, final_mask.shape)) != tuple(map(int, cavity["shape"])):
            return None
        _paint_piece(out, cavity["bbox"], final_mask, color)

        input_rot = int(_RNG.integers(0, 4))
        input_mask = _crop_mask(np.rot90(base_mask, input_rot))
        pieces.append(
            {
                "color": color,
                "base_mask": base_mask,
                "mask": input_mask,
                "input_rot": input_rot,
                "final_rot": _final_rotation_for_target(target_side),
                "symmetric": symmetric,
            }
        )
        if not symmetric:
            asym_indices.append(len(pieces) - 1)

    if asym_indices and all(
        pieces[i]["input_rot"] % 4 == pieces[i]["final_rot"] % 4 for i in asym_indices
    ):
        pick = int(_RNG.choice(asym_indices))
        new_rot = (int(pieces[pick]["final_rot"]) + int(_RNG.integers(1, 4))) % 4
        pieces[pick]["input_rot"] = new_rot
        pieces[pick]["mask"] = _crop_mask(np.rot90(pieces[pick]["base_mask"], new_rot))

    inp[board_r0 : board_r0 + bh, board_c0 : board_c0 + bw] = board_in

    dilated = _dilate(occupied)
    order = sorted(
        range(len(pieces)),
        key=lambda i: int(pieces[i]["mask"].shape[0] * pieces[i]["mask"].shape[1]),
        reverse=True,
    )
    for idx in order:
        piece = pieces[idx]
        ph, pw = piece["mask"].shape
        slots = _find_slots(dilated, (ph, pw))
        if not slots:
            return None
        rr, cc = slots[int(_RNG.integers(0, len(slots)))]
        patch = np.full((ph, pw), _BG, dtype=int)
        patch[piece["mask"]] = int(piece["color"])
        inp[rr : rr + ph, cc : cc + pw][piece["mask"]] = int(piece["color"])
        occupied[rr : rr + ph, cc : cc + pw] = True
        dilated = _dilate(occupied)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, out = built
        try:
            pred = main(inp)
        except Exception:
            continue
        if pred.shape != out.shape:
            continue
        if not np.array_equal(pred, out):
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
