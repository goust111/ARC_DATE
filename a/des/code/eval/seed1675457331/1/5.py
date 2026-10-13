from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BG = 0
FRAME = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDE_ORDER = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
MAX_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    patch = np.full(mask.shape, bg, dtype=int)
    color_vals = grid[coords[:, 0], coords[:, 1]]
    patch[mask] = color_vals
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "patch": patch,
        "area": int(mask.sum()),
        "color": int(color_vals[0]),
    }


def _find_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _pick_frame(components: list[dict]) -> dict:
    def score(comp: dict) -> tuple[int, int, int]:
        r0, c0, r1, c1 = comp["bbox"]
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        hole_score = int(bbox_area - comp["area"])
        return (hole_score, bbox_area, comp["area"])

    return max(components, key=score)


def _pick_frame_color(grid: np.ndarray, bg: int) -> int:
    best_score = None
    best_color = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        r0, c0, r1, c1 = _bbox(coords)
        area = int(len(coords))
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        score = (bbox_area - area, area, bbox_area)
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None, "expected a non-background frame color"
    return int(best_color)


def _find_internal_run(missing: np.ndarray) -> tuple[int, int]:
    n = int(missing.size)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not bool(missing[i]):
            i += 1
            continue
        j = i
        while j < n and bool(missing[j]):
            j += 1
        if i > 0 and j < n:
            runs.append((i, j - 1))
        i = j
    assert len(runs) == 1, f"expected one internal gap run, found {runs}"
    return runs[0]


def _extract_gaps(frame: dict) -> list[dict]:
    mask = frame["mask"]
    fr0, fc0, fr1, fc1 = frame["bbox"]
    h, w = mask.shape
    assert h >= 6 and w >= 6, "frame bbox too small"

    gaps: list[dict] = []

    s, e = _find_internal_run(~mask[0, :])
    top_teeth = tuple(int(i) for i in np.flatnonzero(~mask[1, s : e + 1]))
    gaps.append(
        {
            "side": "top",
            "length": int(e - s + 1),
            "teeth": top_teeth,
            "row": fr0,
            "col": fc0 + s,
        }
    )

    s, e = _find_internal_run(~mask[-1, :])
    bottom_teeth = tuple(int(i) for i in np.flatnonzero(~mask[-2, s : e + 1]))
    gaps.append(
        {
            "side": "bottom",
            "length": int(e - s + 1),
            "teeth": bottom_teeth,
            "row": fr1,
            "col": fc0 + s,
        }
    )

    s, e = _find_internal_run(~mask[:, 0])
    left_teeth = tuple(int(i) for i in np.flatnonzero(~mask[s : e + 1, 1]))
    gaps.append(
        {
            "side": "left",
            "length": int(e - s + 1),
            "teeth": left_teeth,
            "row": fr0 + s,
            "col": fc0,
        }
    )

    s, e = _find_internal_run(~mask[:, -1])
    right_teeth = tuple(int(i) for i in np.flatnonzero(~mask[s : e + 1, -2]))
    gaps.append(
        {
            "side": "right",
            "length": int(e - s + 1),
            "teeth": right_teeth,
            "row": fr0 + s,
            "col": fc1,
        }
    )

    for gap in gaps:
        assert 2 <= int(gap["length"]) <= 4
        assert 0 < len(gap["teeth"]) < int(gap["length"])
    return gaps


def _piece_signatures(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    out: list[dict] = []
    if h >= 3 and bool(mask[1:, :].all()):
        teeth = tuple(int(i) for i in np.flatnonzero(mask[0, :]))
        if 0 < len(teeth) < w:
            out.append({"tooth_side": "top", "length": int(w), "teeth": teeth})
    if h >= 3 and bool(mask[:-1, :].all()):
        teeth = tuple(int(i) for i in np.flatnonzero(mask[-1, :]))
        if 0 < len(teeth) < w:
            out.append({"tooth_side": "bottom", "length": int(w), "teeth": teeth})
    if w >= 3 and bool(mask[:, 1:].all()):
        teeth = tuple(int(i) for i in np.flatnonzero(mask[:, 0]))
        if 0 < len(teeth) < h:
            out.append({"tooth_side": "left", "length": int(h), "teeth": teeth})
    if w >= 3 and bool(mask[:, :-1].all()):
        teeth = tuple(int(i) for i in np.flatnonzero(mask[:, -1]))
        if 0 < len(teeth) < h:
            out.append({"tooth_side": "right", "length": int(h), "teeth": teeth})
    return out


def _piece_options(piece: dict) -> list[dict]:
    options: list[dict] = []
    seen = set()
    for k in range(4):
        mask = np.rot90(piece["mask"], k)
        patch = np.rot90(piece["patch"], k)
        for sig in _piece_signatures(mask):
            key = (mask.shape, mask.tobytes(), sig["tooth_side"])
            if key in seen:
                continue
            seen.add(key)
            options.append(
                {
                    "mask": mask,
                    "patch": patch,
                    "tooth_side": sig["tooth_side"],
                    "length": sig["length"],
                    "teeth": sig["teeth"],
                }
            )
    assert options, "piece has no valid toothed orientation"
    return options


def _placement_for(gap: dict, option: dict) -> tuple[int, int]:
    side = str(gap["side"])
    h, w = option["mask"].shape
    if side == "top":
        assert option["tooth_side"] == "bottom"
        return int(gap["row"] - (h - 2)), int(gap["col"])
    if side == "bottom":
        assert option["tooth_side"] == "top"
        return int(gap["row"] - 1), int(gap["col"])
    if side == "left":
        assert option["tooth_side"] == "right"
        return int(gap["row"]), int(gap["col"] - (w - 2))
    assert side == "right"
    assert option["tooth_side"] == "left"
    return int(gap["row"]), int(gap["col"] - 1)


def _paste(out: np.ndarray, patch: np.ndarray, mask: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = mask.shape
    target = out[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == patch.shape, "placement out of bounds"
    assert not bool(np.any((target != bg) & mask)), "piece overlaps occupied cells"
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble toothed monochrome pieces onto the central frame by shape matching.

    The frame is a thick hollow rectangle with one socket per side. Each loose
    piece is a solid rectangle whose toothed edge is a sparse extra row/column.
    Match sockets to pieces by the socket span and tooth positions, rotate the
    pieces so their teeth face inward, and translate them into place.
    """
    g = np.asarray(grid, dtype=int)
    bg = _find_bg(g)
    frame_color = _pick_frame_color(g, bg)
    frame = _extract_component(g, np.argwhere(g == frame_color), bg)
    pieces = [_extract_component(g, coords, bg) for coords in _components((g != bg) & (g != frame_color))]
    assert len(pieces) == 4

    gaps = _extract_gaps(frame)
    piece_options = [_piece_options(piece) for piece in pieces]

    gap_matches: dict[int, list[tuple[int, dict]]] = {}
    for gi, gap in enumerate(gaps):
        want_side = OPPOSITE[str(gap["side"])]
        matches: list[tuple[int, dict]] = []
        for pi, options in enumerate(piece_options):
            for option in options:
                if (
                    option["tooth_side"] == want_side
                    and int(option["length"]) == int(gap["length"])
                    and tuple(option["teeth"]) == tuple(gap["teeth"])
                ):
                    matches.append((pi, option))
        assert matches, f"no piece matches gap {gap}"
        gap_matches[gi] = matches

    order = sorted(range(len(gaps)), key=lambda gi: len(gap_matches[gi]))
    chosen: dict[int, tuple[int, dict]] = {}
    used_pieces: set[int] = set()

    def search(idx: int) -> bool:
        if idx == len(order):
            return True
        gi = order[idx]
        for pi, option in gap_matches[gi]:
            if pi in used_pieces:
                continue
            chosen[gi] = (pi, option)
            used_pieces.add(pi)
            if search(idx + 1):
                return True
            used_pieces.remove(pi)
            del chosen[gi]
        return False

    assert search(0), "failed to assign pieces to gaps"

    out = g.copy()
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg

    for gi, gap in enumerate(gaps):
        _pi, option = chosen[gi]
        r0, c0 = _placement_for(gap, option)
        _paste(out, option["patch"], option["mask"], r0, c0, bg)

    return out


def _reverse_teeth(length: int, teeth: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(int(length - 1 - t) for t in teeth))


def _canonical_signature(length: int, teeth: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    rev = _reverse_teeth(length, teeth)
    return (int(length), min(tuple(teeth), rev))


def _signature_pool() -> list[dict]:
    pool: list[dict] = []
    for length in range(2, 5):
        groups: dict[tuple[int, tuple[int, ...]], list[tuple[int, ...]]] = {}
        for bits in range(1, (1 << length) - 1):
            teeth = tuple(i for i in range(length) if (bits >> i) & 1)
            groups.setdefault(_canonical_signature(length, teeth), []).append(teeth)
        for (llen, canon), variants in groups.items():
            uniq = []
            seen = set()
            for variant in variants:
                if variant in seen:
                    continue
                seen.add(variant)
                uniq.append(variant)
            pool.append({"length": int(llen), "variants": uniq, "canon": canon})
    return pool


def _frame_mask(height: int, width: int, gaps: dict[str, dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0:2, :] = True
    mask[-2:, :] = True
    mask[:, 0:2] = True
    mask[:, -2:] = True

    top = gaps["top"]
    s = int(top["start"])
    e = s + int(top["length"])
    mask[0, s:e] = False
    for t in top["teeth"]:
        mask[1, s + int(t)] = False

    bottom = gaps["bottom"]
    s = int(bottom["start"])
    e = s + int(bottom["length"])
    mask[-1, s:e] = False
    for t in bottom["teeth"]:
        mask[-2, s + int(t)] = False

    left = gaps["left"]
    s = int(left["start"])
    e = s + int(left["length"])
    mask[s:e, 0] = False
    for t in left["teeth"]:
        mask[s + int(t), 1] = False

    right = gaps["right"]
    s = int(right["start"])
    e = s + int(right["length"])
    mask[s:e, -1] = False
    for t in right["teeth"]:
        mask[s + int(t), -2] = False

    return mask


def _build_piece(tooth_side: str, length: int, teeth: tuple[int, ...], depth: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    if tooth_side in ("top", "bottom"):
        mask = np.ones((depth + 1, length), dtype=bool)
        if tooth_side == "top":
            mask[0, :] = False
            mask[0, list(teeth)] = True
        else:
            mask[-1, :] = False
            mask[-1, list(teeth)] = True
    else:
        mask = np.ones((length, depth + 1), dtype=bool)
        if tooth_side == "left":
            mask[:, 0] = False
            mask[list(teeth), 0] = True
        else:
            mask[:, -1] = False
            mask[list(teeth), -1] = True
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch, mask


def _dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[:-1, :] |= out[1:, :]
        grown[1:, :] |= out[:-1, :]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        grown[:-1, 1:] |= out[1:, :-1]
        grown[1:, :-1] |= out[:-1, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        out = grown
    return out


def _can_place(forbidden: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    H, W = forbidden.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    return not bool(np.any(forbidden[r0 : r0 + h, c0 : c0 + w] & mask))


def _mark_forbidden(forbidden: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    placed = np.zeros_like(forbidden)
    h, w = mask.shape
    placed[r0 : r0 + h, c0 : c0 + w] |= mask
    forbidden |= _dilate_mask(placed, radius=1)


def _place_scattered_pieces(
    rng: random.Random,
    canvas: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    pieces: list[dict],
) -> np.ndarray | None:
    out = canvas.copy()
    H, W = out.shape
    r0, c0, r1, c1 = frame_bbox
    forbidden = np.zeros((H, W), dtype=bool)
    forbidden[max(0, r0 - 1) : min(H, r1 + 2), max(0, c0 - 1) : min(W, c1 + 2)] = True

    rotated_items = []
    for piece in pieces:
        k = rng.randrange(4)
        patch = np.rot90(piece["patch"], k)
        mask = np.rot90(piece["mask"], k)
        rotated_items.append({"patch": patch, "mask": mask})

    order = sorted(range(len(rotated_items)), key=lambda i: int(rotated_items[i]["mask"].sum()), reverse=True)
    for idx in order:
        item = rotated_items[idx]
        h, w = item["mask"].shape
        positions = [(rr, cc) for rr in range(H - h + 1) for cc in range(W - w + 1)]
        rng.shuffle(positions)
        placed = False
        for rr, cc in positions:
            if not _can_place(forbidden, item["mask"], rr, cc):
                continue
            _paste(out, item["patch"], item["mask"], rr, cc, BG)
            _mark_forbidden(forbidden, item["mask"], rr, cc)
            placed = True
            break
        if not placed:
            return None
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(20, 30)
    W = rng.randint(20, 30)
    frame_h = rng.randint(10, min(15, H - 7))
    frame_w = rng.randint(10, min(15, W - 7))
    r0 = (H - frame_h) // 2
    c0 = (W - frame_w) // 2
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    sig_defs = rng.sample(_signature_pool(), 4)
    side_sigs = {}
    for side, sig_def in zip(SIDE_ORDER, sig_defs, strict=False):
        side_sigs[side] = {
            "length": int(sig_def["length"]),
            "teeth": tuple(rng.choice(sig_def["variants"])),
        }

    gaps = {
        "top": {
            "start": rng.randint(2, frame_w - 2 - side_sigs["top"]["length"]),
            **side_sigs["top"],
        },
        "bottom": {
            "start": rng.randint(2, frame_w - 2 - side_sigs["bottom"]["length"]),
            **side_sigs["bottom"],
        },
        "left": {
            "start": rng.randint(2, frame_h - 2 - side_sigs["left"]["length"]),
            **side_sigs["left"],
        },
        "right": {
            "start": rng.randint(2, frame_h - 2 - side_sigs["right"]["length"]),
            **side_sigs["right"],
        },
    }

    frame_mask = _frame_mask(frame_h, frame_w, gaps)
    base = np.full((H, W), BG, dtype=int)
    base[r0 : r1 + 1, c0 : c1 + 1][frame_mask] = FRAME

    colors = rng.sample(PIECE_COLORS, 4)
    output = base.copy()
    loose_pieces: list[dict] = []
    for side, color in zip(SIDE_ORDER, colors, strict=False):
        sig = side_sigs[side]
        depth = rng.randint(2, 4)
        patch, mask = _build_piece(OPPOSITE[side], sig["length"], sig["teeth"], depth, color)
        gap = {
            "side": side,
            "length": sig["length"],
            "teeth": sig["teeth"],
            "row": r0 if side == "top" else r1 if side == "bottom" else r0 + gaps[side]["start"],
            "col": c0 + gaps[side]["start"] if side in ("top", "bottom") else c0 if side == "left" else c1,
        }
        rr, cc = _placement_for(gap, {"mask": mask, "tooth_side": OPPOSITE[side]})
        _paste(output, patch, mask, rr, cc, BG)
        loose_pieces.append({"patch": patch, "mask": mask})

    scattered = _place_scattered_pieces(rng, base, (r0, c0, r1, c1), loose_pieces)
    if scattered is None:
        return None
    if not np.array_equal(main(scattered), output):
        return None
    return scattered


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
