from __future__ import annotations

from collections import Counter, deque
import numpy as np


BG_COLOR = 8
ARROW_COLOR = 4
FRAME_COLORS = [0, 1, 2, 3, 5, 6, 7, 9]

GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
N_FRAMES_RANGE = (2, 5)
FRAME_H_RANGE = (7, 12)
FRAME_W_RANGE = (7, 12)
GAP_LENGTH_CHOICES = (2, 3, 4)
PIECE_DEPTH_RANGE = (2, 5)

FRAME_PLACE_TRIES = 200
PIECE_PLACE_TRIES = 200
GEN_RETRIES = 500

_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components(grid == color):
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        comps.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "mask": mask,
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _ring_mask(h: int, w: int) -> np.ndarray:
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return ring


def _frame_info(mask: np.ndarray) -> dict | None:
    h, w = mask.shape
    if h < 5 or w < 5:
        return None
    ring = _ring_mask(h, w)
    if bool((mask & ~ring).any()):
        return None
    missing = np.argwhere(ring & ~mask)
    if missing.shape[0] < 2 or missing.shape[0] > 4:
        return None
    if any((r in (0, h - 1) and c in (0, w - 1)) for r, c in missing):
        return None

    sides: list[str] = []
    if np.all(missing[:, 0] == 0):
        cols = sorted(int(c) for c in missing[:, 1])
        if cols == list(range(cols[0], cols[-1] + 1)):
            sides.append("top")
    if np.all(missing[:, 0] == h - 1):
        cols = sorted(int(c) for c in missing[:, 1])
        if cols == list(range(cols[0], cols[-1] + 1)):
            sides.append("bottom")
    if np.all(missing[:, 1] == 0):
        rows = sorted(int(r) for r in missing[:, 0])
        if rows == list(range(rows[0], rows[-1] + 1)):
            sides.append("left")
    if np.all(missing[:, 1] == w - 1):
        rows = sorted(int(r) for r in missing[:, 0])
        if rows == list(range(rows[0], rows[-1] + 1)):
            sides.append("right")
    if len(sides) != 1:
        return None

    return {
        "ring": ring,
        "missing": {tuple(map(int, x)) for x in missing},
        "gap_side": sides[0],
        "shape": (h, w),
    }


def _classify_arrow(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if not bool(mask.all()):
        return None
    if h == 1 and w == 2:
        return "h"
    if h == 2 and w == 1:
        return "v"
    return None


def _match_arrow(frame_bbox: tuple[int, int, int, int], arrows: list[dict], used: set[int]) -> str:
    r0, c0, r1, c1 = map(int, frame_bbox)
    matches: list[tuple[int, str]] = []
    for idx, arrow in enumerate(arrows):
        if idx in used:
            continue
        ar0, ac0, ar1, ac1 = map(int, arrow["bbox"])
        if arrow["kind"] == "h":
            if ac0 == c0 - 2 and ac1 == c0 - 1 and ar0 == ar1 and r0 <= ar0 <= r1:
                matches.append((idx, "h"))
        elif arrow["kind"] == "v":
            if ar0 == r0 - 2 and ar1 == r0 - 1 and ac0 == ac1 and c0 <= ac0 <= c1:
                matches.append((idx, "v"))
    assert len(matches) == 1
    used.add(int(matches[0][0]))
    return matches[0][1]


def _find_piece_placement(frame_info: dict, piece_mask: np.ndarray) -> tuple[int, int] | None:
    h, w = map(int, frame_info["shape"])
    ph, pw = piece_mask.shape
    ring = frame_info["ring"]
    missing = frame_info["missing"]
    piece_coords = np.argwhere(piece_mask)
    candidates: list[tuple[int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            on_ring = set()
            for pr, pc in piece_coords:
                rr = int(r0 + pr)
                cc = int(c0 + pc)
                if ring[rr, cc]:
                    on_ring.add((rr, cc))
            if on_ring == missing:
                candidates.append((int(r0), int(c0)))

    assert len(candidates) <= 1
    return candidates[0] if candidates else None


def _paint_patch(
    color: int,
    bg: int,
    frame_info: dict,
    piece_mask: np.ndarray,
    placement: tuple[int, int],
) -> np.ndarray:
    h, w = map(int, frame_info["shape"])
    out = np.full((h, w), bg, dtype=int)
    out[frame_info["ring"]] = color
    r0, c0 = map(int, placement)
    ph, pw = piece_mask.shape
    window = out[r0 : r0 + ph, c0 : c0 + pw]
    window[piece_mask] = color
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    win = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    win[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each colored fragment into its same-color frame gap, then mirror the assembled frame
    according to the adjacent yellow marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    out = np.full_like(g, bg)

    arrows = []
    for comp in _color_components(g, ARROW_COLOR):
        kind = _classify_arrow(comp["mask"])
        if kind is not None:
            arrows.append({"bbox": comp["bbox"], "kind": kind})

    used_arrows: set[int] = set()
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, ARROW_COLOR)]

    for color in colors:
        comps = _color_components(g, color)
        frame_candidates: list[tuple[dict, dict]] = []
        piece_candidates: list[dict] = []
        for comp in comps:
            info = _frame_info(comp["mask"])
            if info is not None:
                frame_candidates.append((comp, info))
            else:
                piece_candidates.append(comp)
        if not frame_candidates:
            continue

        assert len(frame_candidates) == 1
        frame_comp, frame_info = frame_candidates[0]
        flip_kind = _match_arrow(frame_comp["bbox"], arrows, used_arrows)

        found_piece = None
        found_place = None
        for piece in piece_candidates:
            placement = _find_piece_placement(frame_info, piece["mask"])
            if placement is None:
                continue
            assert found_piece is None
            found_piece = piece
            found_place = placement
        assert found_piece is not None and found_place is not None

        patch = _paint_patch(color, bg, frame_info, found_piece["mask"], found_place)
        patch = np.fliplr(patch) if flip_kind == "h" else np.flipud(patch)
        r0, c0, _r1, _c1 = map(int, frame_comp["bbox"])
        _paste(out, patch, r0, c0, bg)

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _grow_piece_mask(shape: tuple[int, int], required: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, shape)
    total = h * w
    max_extra = total - len(required)
    assert max_extra >= 1
    target = len(required) + int(_RNG.integers(1, max_extra + 1))

    cells = set(required)
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        choice = frontier[int(_RNG.integers(0, len(frontier)))]
        cells.add(choice)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _build_piece(side: str, gap_len: int, frame_h: int, frame_w: int) -> np.ndarray | None:
    if side in ("top", "bottom"):
        max_depth = min(PIECE_DEPTH_RANGE[1] - 1, frame_h - 2)
        if max_depth < PIECE_DEPTH_RANGE[0]:
            return None
        depth = int(_RNG.integers(PIECE_DEPTH_RANGE[0], max_depth + 1))
        shape = (depth, gap_len)
        req_row = 0 if side == "top" else depth - 1
        required = [(req_row, c) for c in range(gap_len)]
    else:
        max_depth = min(PIECE_DEPTH_RANGE[1] - 1, frame_w - 2)
        if max_depth < PIECE_DEPTH_RANGE[0]:
            return None
        depth = int(_RNG.integers(PIECE_DEPTH_RANGE[0], max_depth + 1))
        shape = (gap_len, depth)
        req_col = 0 if side == "left" else depth - 1
        required = [(r, req_col) for r in range(gap_len)]

    for _ in range(50):
        mask = _grow_piece_mask(shape, required)
        if int(mask.sum()) <= len(required):
            continue
        return mask
    return None


def _frame_patch(
    color: int,
    bg: int,
    h: int,
    w: int,
    gap_side: str,
    gap_start: int,
    gap_len: int,
) -> tuple[np.ndarray, dict]:
    patch = np.full((h, w), bg, dtype=int)
    ring = _ring_mask(h, w)
    patch[ring] = color

    if gap_side == "top":
        patch[0, gap_start : gap_start + gap_len] = bg
        missing = {(0, c) for c in range(gap_start, gap_start + gap_len)}
    elif gap_side == "bottom":
        patch[h - 1, gap_start : gap_start + gap_len] = bg
        missing = {(h - 1, c) for c in range(gap_start, gap_start + gap_len)}
    elif gap_side == "left":
        patch[gap_start : gap_start + gap_len, 0] = bg
        missing = {(r, 0) for r in range(gap_start, gap_start + gap_len)}
    else:
        patch[gap_start : gap_start + gap_len, w - 1] = bg
        missing = {(r, w - 1) for r in range(gap_start, gap_start + gap_len)}

    info = {
        "ring": ring,
        "missing": missing,
        "gap_side": gap_side,
        "shape": (h, w),
    }
    return patch, info


def _compose_input_output(
    color: int,
    flip_kind: str,
) -> dict | None:
    for _ in range(80):
        h = int(_RNG.integers(*FRAME_H_RANGE))
        w = int(_RNG.integers(*FRAME_W_RANGE))
        gap_side = str(_RNG.choice(["top", "bottom", "left", "right"]))
        side_span = w if gap_side in ("top", "bottom") else h
        valid_lengths = [x for x in GAP_LENGTH_CHOICES if x <= side_span - 2]
        if not valid_lengths:
            continue
        gap_len = int(_RNG.choice(valid_lengths))
        gap_start = int(_RNG.integers(1, side_span - gap_len))

        piece_mask = _build_piece(gap_side, gap_len, h, w)
        if piece_mask is None:
            continue
        if _frame_info(piece_mask) is not None:
            continue

        frame_patch, frame_info = _frame_patch(color, BG_COLOR, h, w, gap_side, gap_start, gap_len)
        placement = _find_piece_placement(frame_info, piece_mask)
        if placement is None:
            continue

        filled = _paint_patch(color, BG_COLOR, frame_info, piece_mask, placement)
        mirrored = np.fliplr(filled) if flip_kind == "h" else np.flipud(filled)
        if np.array_equal(filled, mirrored):
            continue

        if flip_kind == "h":
            arrow_pos = ("left", int(_RNG.integers(1, h - 1)))
        else:
            arrow_pos = ("top", int(_RNG.integers(1, w - 1)))

        return {
            "color": color,
            "flip_kind": flip_kind,
            "frame_patch": frame_patch,
            "piece_mask": piece_mask,
            "output_patch": mirrored,
            "arrow_pos": arrow_pos,
        }
    return None


def _apply_blocked(blocked: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return blocked | _dilate8(mask)


def _place_frame(spec: dict, blocked: np.ndarray) -> tuple[int, int] | None:
    h, w = spec["frame_patch"].shape
    H, W = blocked.shape
    for _ in range(FRAME_PLACE_TRIES):
        r0 = int(_RNG.integers(2, H - h - 1))
        c0 = int(_RNG.integers(2, W - w - 1))
        cover = np.zeros_like(blocked)
        cover[r0 : r0 + h, c0 : c0 + w] = True
        if spec["flip_kind"] == "h":
            arr_r = int(r0 + spec["arrow_pos"][1])
            cover[arr_r, c0 - 2 : c0] = True
        else:
            arr_c = int(c0 + spec["arrow_pos"][1])
            cover[r0 - 2 : r0, arr_c] = True
        if not bool((blocked & cover).any()):
            spec["frame_pos"] = (r0, c0)
            spec["cover_mask"] = cover
            return r0, c0
    return None


def _place_piece(spec: dict, blocked: np.ndarray) -> tuple[int, int] | None:
    piece = spec["piece_mask"]
    ph, pw = piece.shape
    H, W = blocked.shape
    for _ in range(PIECE_PLACE_TRIES):
        r0 = int(_RNG.integers(1, H - ph))
        c0 = int(_RNG.integers(1, W - pw))
        cover = np.zeros_like(blocked)
        cover[r0 : r0 + ph, c0 : c0 + pw] = True
        if not bool((blocked & cover).any()):
            spec["piece_pos"] = (r0, c0)
            spec["piece_cover"] = cover
            return r0, c0
    return None


def _draw_spec(inp: np.ndarray, out: np.ndarray, spec: dict) -> None:
    color = int(spec["color"])
    frame_patch = spec["frame_patch"]
    output_patch = spec["output_patch"]
    piece_mask = spec["piece_mask"]
    fr0, fc0 = map(int, spec["frame_pos"])
    pr0, pc0 = map(int, spec["piece_pos"])

    _paste(inp, frame_patch, fr0, fc0, BG_COLOR)
    _paste(out, output_patch, fr0, fc0, BG_COLOR)

    piece_patch = np.full(piece_mask.shape, BG_COLOR, dtype=int)
    piece_patch[piece_mask] = color
    _paste(inp, piece_patch, pr0, pc0, BG_COLOR)

    if spec["flip_kind"] == "h":
        arr_r = int(fr0 + spec["arrow_pos"][1])
        inp[arr_r, fc0 - 2 : fc0] = ARROW_COLOR
    else:
        arr_c = int(fc0 + spec["arrow_pos"][1])
        inp[fr0 - 2 : fr0, arr_c] = ARROW_COLOR


def _balanced_flips(n: int) -> list[str]:
    flips = ["h"] * (n // 2) + ["v"] * (n // 2)
    if n % 2 == 1:
        flips.append("h" if int(_RNG.integers(0, 2)) == 0 else "v")
    _RNG.shuffle(flips)
    return flips


def _gen_attempt() -> np.ndarray | None:
    H = int(_RNG.integers(*GRID_H_RANGE))
    W = int(_RNG.integers(*GRID_W_RANGE))
    n_frames = int(_RNG.integers(*N_FRAMES_RANGE))
    colors = [int(x) for x in _RNG.choice(np.asarray(FRAME_COLORS), size=n_frames, replace=False)]
    flips = _balanced_flips(n_frames)

    specs = []
    for color, flip_kind in zip(colors, flips):
        spec = _compose_input_output(color, flip_kind)
        if spec is None:
            return None
        specs.append(spec)

    blocked = np.zeros((H, W), dtype=bool)
    for spec in sorted(specs, key=lambda d: -(d["frame_patch"].shape[0] * d["frame_patch"].shape[1])):
        if _place_frame(spec, blocked) is None:
            return None
        blocked = _apply_blocked(blocked, spec["cover_mask"])

    for spec in sorted(specs, key=lambda d: -(d["piece_mask"].shape[0] * d["piece_mask"].shape[1])):
        if _place_piece(spec, blocked) is None:
            return None
        blocked = _apply_blocked(blocked, spec["piece_cover"])

    inp = np.full((H, W), BG_COLOR, dtype=int)
    out = np.full((H, W), BG_COLOR, dtype=int)
    for spec in specs:
        _draw_spec(inp, out, spec)

    if Counter(int(x) for x in inp.ravel())[BG_COLOR] <= (H * W) // 2:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
