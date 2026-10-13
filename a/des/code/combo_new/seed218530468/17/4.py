from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CORNERS = ("tl", "tr", "bl", "br")

GEN_ATTEMPTS = 500
PAIR_COUNT_OPTIONS = [1, 2, 3, 4]
PAIR_COUNT_WEIGHTS = [1, 3, 3, 2]
INNER_H_RANGE = (3, 7)
INNER_W_RANGE = (3, 7)
PLACEMENT_PAD = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components_from_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in coords:
                mask[r - r0, c - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                    "key": frozenset(coords),
                }
            )
    return comps


def _corner_rectangle(mask: np.ndarray) -> str | None:
    if not mask.any() or mask.all():
        return None
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h, w = mask.shape

    checks = [
        ("tl", r0 == 0 and c0 == 0, slice(0, r1 + 1), slice(0, c1 + 1)),
        ("tr", r0 == 0 and c1 == w - 1, slice(0, r1 + 1), slice(c0, w)),
        ("bl", r1 == h - 1 and c0 == 0, slice(r0, h), slice(0, c1 + 1)),
        ("br", r1 == h - 1 and c1 == w - 1, slice(r0, h), slice(c0, w)),
    ]
    for name, ok, rs, cs in checks:
        if not ok:
            continue
        rect = np.zeros_like(mask, dtype=bool)
        rect[rs, cs] = True
        if np.array_equal(mask, rect):
            return name
    return None


def _frame_from_component(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h, w = comp["mask"].shape
    color = int(comp["color"])
    if h < 5 or w < 5:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((patch != bg) & (patch != color)):
        return None
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return None

    inner = patch[1:-1, 1:-1]
    occupied = inner == color
    free = inner == bg
    if not occupied.any() or not free.any():
        return None
    if _corner_rectangle(occupied) is None:
        return None
    if len(_components_from_mask(free)) != 1:
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "free_mask": free,
        "key": comp["key"],
    }


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered pieces to puzzle frames, insert them into the matching cavity, and tint the inserted edge cells with the frame color.

    Input structure:
    - The background is black.
    - Several colored frame objects are one-pixel-thick rectangular rings whose boundary color also fills one interior corner rectangle.
    - The remaining colored objects are scattered single-color 4-connected pieces, each matching one frame's interior black cavity up to translation only.

    Rule:
    - Detect each frame and extract the black cavity inside its rectangle after excluding the pre-filled corner block.
    - Find the scattered piece whose tight shape exactly matches that cavity without rotation or reflection.
    - Erase the piece from its original location and place it into the frame cavity.
    - Repaint every placed piece cell that lies orthogonally against the frame border to the frame's own color, leaving the rest of the piece in its original color.

    Key insight:
    - The matching cue is the cavity shape carved out by the corner-filled rectangular frame, and after translation the final step is an inner border recoloring rather than a plain copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    comps = _monochrome_components(g, bg)
    frames = []
    frame_keys = set()
    for comp in comps:
        frame = _frame_from_component(g, bg, comp)
        if frame is not None:
            frames.append(frame)
            frame_keys.add(comp["key"])

    assert frames, "expected at least one puzzle frame"
    fragments = [comp for comp in comps if comp["key"] not in frame_keys]
    assert len(fragments) == len(frames), "frame/piece counts must match"

    frame_sigs = [_mask_signature(frame["free_mask"]) for frame in frames]
    frag_by_sig: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for frag in fragments:
        frag_by_sig.setdefault(_mask_signature(frag["mask"]), []).append(frag)

    assert len(set(frame_sigs)) == len(frames), "frame cavities must be uniquely matchable"
    for sig in frame_sigs:
        assert len(frag_by_sig.get(sig, [])) == 1, "each cavity must have exactly one matching piece"

    out = g.copy()
    for frag in fragments:
        for r, c in frag["coords"]:
            out[r, c] = bg

    for frame in frames:
        sig = _mask_signature(frame["free_mask"])
        piece = frag_by_sig[sig][0]
        piece_color = int(piece["color"])
        fr0, fc0, _fr1, _fc1 = frame["bbox"]
        cavity = frame["free_mask"]
        ih, iw = cavity.shape
        for rr, cc in np.argwhere(cavity):
            rr_i = int(rr)
            cc_i = int(cc)
            if rr_i in (0, ih - 1) or cc_i in (0, iw - 1):
                out[fr0 + 1 + rr_i, fc0 + 1 + cc_i] = int(frame["color"])
            else:
                out[fr0 + 1 + rr_i, fc0 + 1 + cc_i] = piece_color

    return out


def _build_pair(
    inner_h: int,
    inner_w: int,
    corner: str,
    block_h: int,
    block_w: int,
    frame_color: int,
    piece_color: int,
) -> dict:
    frame = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    frame[0, :] = frame_color
    frame[-1, :] = frame_color
    frame[:, 0] = frame_color
    frame[:, -1] = frame_color

    free_mask = np.ones((inner_h, inner_w), dtype=bool)
    if corner == "tl":
        free_mask[:block_h, :block_w] = False
        frame[1 : 1 + block_h, 1 : 1 + block_w] = frame_color
    elif corner == "tr":
        free_mask[:block_h, inner_w - block_w :] = False
        frame[1 : 1 + block_h, 1 + inner_w - block_w : 1 + inner_w] = frame_color
    elif corner == "bl":
        free_mask[inner_h - block_h :, :block_w] = False
        frame[1 + inner_h - block_h : 1 + inner_h, 1 : 1 + block_w] = frame_color
    else:
        free_mask[inner_h - block_h :, inner_w - block_w :] = False
        frame[1 + inner_h - block_h : 1 + inner_h, 1 + inner_w - block_w : 1 + inner_w] = frame_color

    piece = np.zeros((inner_h, inner_w), dtype=int)
    piece[free_mask] = piece_color
    return {
        "frame_patch": frame,
        "piece_patch": piece,
        "free_mask": free_mask,
        "frame_color": frame_color,
        "piece_color": piece_color,
    }


def _area_free(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not np.any(occupied[r0:r1, c0:c1])


def _reserve(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    pad: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    if h + 2 > grid.shape[0] or w + 2 > grid.shape[1]:
        return None
    positions = [(r, c) for r in range(1, grid.shape[0] - h) for c in range(1, grid.shape[1] - w)]
    rng.shuffle(positions)
    for top, left in positions:
        if not _area_free(occupied, top, left, h, w, pad):
            continue
        grid[top : top + h, left : left + w] = patch
        _reserve(occupied, top, left, h, w, pad)
        return top, left
    return None


def _expected_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        pr, pc = pair["piece_pos"]
        piece_patch = pair["piece_patch"]
        for rr, cc in np.argwhere(piece_patch != BG):
            out[pr + int(rr), pc + int(cc)] = BG

    for pair in pairs:
        fr, fc = pair["frame_pos"]
        free_mask = pair["free_mask"]
        piece_color = int(pair["piece_color"])
        ih, iw = free_mask.shape
        for rr, cc in np.argwhere(free_mask):
            rr_i = int(rr)
            cc_i = int(cc)
            if rr_i in (0, ih - 1) or cc_i in (0, iw - 1):
                out[fr + 1 + rr_i, fc + 1 + cc_i] = int(pair["frame_color"])
            else:
                out[fr + 1 + rr_i, fc + 1 + cc_i] = piece_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choices(PAIR_COUNT_OPTIONS, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    colors = rng.sample(COLOR_POOL, 2 * pair_count)
    frame_colors = colors[:pair_count]
    piece_colors = colors[pair_count:]

    pairs: list[dict] = []
    sigs = set()
    for i in range(pair_count):
        frame_color = int(frame_colors[i])
        piece_color = int(piece_colors[i])
        built = None
        for _ in range(80):
            inner_h = rng.randint(*INNER_H_RANGE)
            inner_w = rng.randint(*INNER_W_RANGE)
            block_h = rng.randint(1, inner_h - 1)
            block_w = rng.randint(1, inner_w - 1)
            free_area = inner_h * inner_w - block_h * block_w
            if free_area < 4:
                continue
            corner = rng.choice(CORNERS)
            pair = _build_pair(inner_h, inner_w, corner, block_h, block_w, frame_color, piece_color)
            if not np.any(pair["free_mask"][1:-1, 1:-1]):
                continue
            sig = _mask_signature(pair["free_mask"])
            if sig in sigs:
                continue
            sigs.add(sig)
            built = pair
            break
        if built is None:
            return None
        pairs.append(built)

    bbox_area = sum(int(pair["frame_patch"].size + pair["piece_patch"].size) for pair in pairs)
    max_h = max(max(pair["frame_patch"].shape[0], pair["piece_patch"].shape[0]) for pair in pairs)
    max_w = max(max(pair["frame_patch"].shape[1], pair["piece_patch"].shape[1]) for pair in pairs)
    side_hint = max(int(np.ceil(np.sqrt(2.2 * bbox_area))), max_h + 6, max_w + 6, 12 + pair_count)
    if side_hint > 30:
        return None

    min_h = max(max_h + 3, side_hint - rng.randint(0, 3))
    min_w = max(max_w + 3, side_hint - rng.randint(0, 3))
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    objects = []
    for idx, pair in enumerate(pairs):
        objects.append((int(pair["frame_patch"].size), "frame", idx, pair["frame_patch"]))
        objects.append((int(pair["piece_patch"].size), "piece", idx, pair["piece_patch"]))
    objects.sort(reverse=True)

    for _area, kind, idx, patch in objects:
        pos = _place_patch(rng, grid, occupied, patch, PLACEMENT_PAD)
        if pos is None:
            return None
        pairs[idx][f"{kind}_pos"] = pos

    if _most_common_color(grid) != BG:
        return None

    expected = _expected_output(grid, pairs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
