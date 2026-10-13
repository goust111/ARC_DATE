from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
MAROON = 9
RED = 2
BLUE = 1
GREEN = 3
YELLOW = 4
MARKERS = (RED, BLUE, GREEN, YELLOW)
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _run_len(grid: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> int:
    n = 0
    h, w = grid.shape
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == int(color):
        n += 1
        r += dr
        c += dc
    return n


def _frame_info(grid: np.ndarray, bg: int) -> dict:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != int(bg))
    assert counts, "expected a non-background frame"
    frame_color = counts.most_common(1)[0][0]
    coords = np.argwhere(grid == int(frame_color))
    r0, c0, r1, c1 = _bbox(coords)
    thickness = min(
        _run_len(grid, r0, c0, 1, 1, frame_color),
        _run_len(grid, r0, c1, 1, -1, frame_color),
        _run_len(grid, r1, c0, -1, 1, frame_color),
        _run_len(grid, r1, c1, -1, -1, frame_color),
    )
    assert thickness >= 2, "generator uses a ring thicker than 1 so flips stay visible"
    return {
        "color": int(frame_color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "thickness": int(thickness),
    }


def _ring_mask(shape: tuple[int, int], frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    t = int(frame["thickness"])
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mask[r0 + t : r1 - t + 1, c0 + t : c1 - t + 1] = False
    return mask


def _extract_slots(grid: np.ndarray, bg: int, frame: dict) -> list[dict]:
    r0, c0, r1, c1 = frame["bbox"]
    t = int(frame["thickness"])
    ring = _ring_mask(grid.shape, frame)
    slot_mask = ring & (grid == int(bg))
    slots: list[dict] = []
    for coords in _components(slot_mask):
        sr0, sc0, sr1, sc1 = _bbox(coords)
        touched = []
        if sr0 == r0:
            touched.append("top")
        if sr1 == r1:
            touched.append("bottom")
        if sc0 == c0:
            touched.append("left")
        if sc1 == c1:
            touched.append("right")
        assert len(touched) == 1, touched
        side = touched[0]
        if side in ("top", "bottom"):
            assert sr1 - sr0 + 1 == t
        else:
            assert sc1 - sc0 + 1 == t
        slots.append(
            {
                "side": side,
                "bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                "shape": (int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)),
            }
        )
    assert slots
    return slots


def _piece_side(piece_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    pr0, pc0, pr1, pc1 = piece_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if pr1 < fr0:
        return "top"
    if pr0 > fr1:
        return "bottom"
    if pc1 < fc0:
        return "left"
    if pc0 > fc1:
        return "right"
    raise AssertionError("piece must be outside the frame")


def _extract_pieces(grid: np.ndarray, bg: int, frame: dict) -> list[dict]:
    piece_mask = (grid != int(bg)) & (grid != int(frame["color"]))
    pieces: list[dict] = []
    for coords in _components(piece_mask):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        occ = patch != int(bg)
        marker_cells = [(int(r), int(c), int(patch[r, c])) for r, c in np.argwhere((patch != int(bg)) & (patch != int(MAROON)))]
        assert len(marker_cells) == 1
        mr, mc, marker = marker_cells[0]
        assert marker in MARKERS
        pieces.append(
            {
                "side": _piece_side((r0, c0, r1, c1), frame["bbox"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
                "mask": occ,
                "marker": int(marker),
                "marker_pos": (int(mr), int(mc)),
            }
        )
    assert pieces
    return pieces


def _match_slots(pieces: list[dict], slots: list[dict]) -> list[tuple[dict, dict]]:
    by_side: dict[str, list[dict]] = {side: [] for side in SIDES}
    for slot in slots:
        by_side[slot["side"]].append(slot)

    used: set[tuple[str, int]] = set()
    matches: list[tuple[dict, dict]] = []
    for piece in pieces:
        pr0, pc0, pr1, pc1 = piece["bbox"]
        candidates = []
        for idx, slot in enumerate(by_side[piece["side"]]):
            key = (piece["side"], idx)
            if key in used or slot["shape"] != piece["shape"]:
                continue
            sr0, sc0, sr1, sc1 = slot["bbox"]
            if piece["side"] in ("top", "bottom"):
                if sc0 == pc0 and sc1 == pc1:
                    candidates.append((idx, slot))
            else:
                if sr0 == pr0 and sr1 == pr1:
                    candidates.append((idx, slot))
        if len(candidates) != 1:
            fallback = [
                (idx, slot)
                for idx, slot in enumerate(by_side[piece["side"]])
                if (piece["side"], idx) not in used and slot["shape"] == piece["shape"]
            ]
            assert len(fallback) == 1
            candidates = fallback
        idx, slot = candidates[0]
        used.add((piece["side"], idx))
        matches.append((piece, slot))
    return matches


def _transform_mask(mask: np.ndarray, side: str, marker: int) -> tuple[np.ndarray, int]:
    out_mask = np.array(mask, copy=True)
    if marker in (GREEN, YELLOW):
        if side in ("top", "bottom"):
            out_mask = np.fliplr(out_mask)
        else:
            out_mask = np.flipud(out_mask)
    fill = MAROON if marker in (RED, GREEN) else marker
    return out_mask, int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each outside fragment into the matching side slot of the central frame.

    The implementation recovers the grey ring, finds rectangular slot gaps in that ring,
    matches each outside multicolor maroon fragment to a same-side slot with the same bbox,
    then applies the marker-coded local transform before insertion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame = _frame_info(g, bg)
    slots = _extract_slots(g, bg, frame)
    pieces = _extract_pieces(g, bg, frame)
    matches = _match_slots(pieces, slots)

    out = g.copy()
    for piece, _slot in matches:
        pr0, pc0, pr1, pc1 = piece["bbox"]
        patch = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
        patch[piece["mask"]] = int(bg)

    for piece, slot in matches:
        sr0, sc0, sr1, sc1 = slot["bbox"]
        tmask, fill = _transform_mask(piece["mask"], piece["side"], piece["marker"])
        out[sr0 : sr1 + 1, sc0 : sc1 + 1][tmask] = int(fill)

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    out = [1] * parts
    for _ in range(total - parts):
        out[random.randrange(parts)] += 1
    return out


def _random_composition(total: int, bins: int) -> list[int]:
    if bins == 1:
        return [total]
    cuts = sorted(random.sample(range(total + bins - 1), bins - 1))
    values = []
    prev = -1
    for cut in cuts + [total + bins - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _place_intervals(total_len: int, lengths: list[int], gap: int = 1) -> list[int] | None:
    need = sum(lengths) + gap * max(0, len(lengths) - 1)
    if need > total_len:
        return None
    slack = total_len - need
    extras = _random_composition(slack, len(lengths) + 1)
    starts: list[int] = []
    cur = extras[0]
    for i, length in enumerate(lengths):
        starts.append(cur)
        cur += length
        if i + 1 < len(lengths):
            cur += gap + extras[i + 1]
    return starts


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    target = int(mask.sum())
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                stack.append((nr, nc))
    return len(seen) == target


def _make_piece_mask(h: int, w: int, side: str, marker: int) -> np.ndarray:
    cells = [(r, c) for r in range(h) for c in range(w)]
    need_asym = marker in (GREEN, YELLOW)
    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        hole_count = random.choice([0, 1, 1, 2]) if not need_asym else random.choice([1, 2])
        for r, c in random.sample(cells, hole_count):
            mask[r, c] = False
        if mask.sum() < max(3, h + w - 1):
            continue
        if not _full_bbox(mask):
            continue
        if not _connected(mask):
            continue
        if need_asym:
            if side in ("top", "bottom") and np.array_equal(mask, np.fliplr(mask)):
                continue
            if side in ("left", "right") and np.array_equal(mask, np.flipud(mask)):
                continue
        return mask
    raise RuntimeError("failed to build a valid piece mask")


def _make_piece_patch(mask: np.ndarray, marker: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = MAROON
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    rr, cc = random.choice(cells)
    patch[rr, cc] = int(marker)
    return patch


def _build_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        pr0, pc0 = spec["piece_pos"]
        ph, pw = spec["patch"].shape
        piece_view = out[pr0 : pr0 + ph, pc0 : pc0 + pw]
        piece_view[spec["mask"]] = BG

    for spec in specs:
        sr0, sc0, sr1, sc1 = spec["slot_bbox"]
        tmask, fill = _transform_mask(spec["mask"], spec["side"], spec["marker"])
        out[sr0 : sr1 + 1, sc0 : sc1 + 1][tmask] = int(fill)
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    thickness = random.choice([2, 3])
    n_pieces = random.randint(4, 6)
    top_n, bottom_n, left_n, right_n = _random_partition(n_pieces, 4)
    counts = {"top": top_n, "bottom": bottom_n, "left": left_n, "right": right_n}

    top_lengths = [random.randint(3, 6) for _ in range(top_n)]
    bottom_lengths = [random.randint(3, 6) for _ in range(bottom_n)]
    left_lengths = [random.randint(3, 6) for _ in range(left_n)]
    right_lengths = [random.randint(3, 6) for _ in range(right_n)]

    need_w = max(
        sum(top_lengths) + max(0, top_n - 1),
        sum(bottom_lengths) + max(0, bottom_n - 1),
    )
    need_h = max(
        sum(left_lengths) + max(0, left_n - 1),
        sum(right_lengths) + max(0, right_n - 1),
    )
    frame_w = random.randint(max(need_w + 2 * thickness, 12), max(need_w + 2 * thickness + 6, 12))
    frame_h = random.randint(max(need_h + 2 * thickness, 12), max(need_h + 2 * thickness + 6, 12))

    avail_w = frame_w - 2 * thickness
    avail_h = frame_h - 2 * thickness
    top_starts = _place_intervals(avail_w, top_lengths)
    bottom_starts = _place_intervals(avail_w, bottom_lengths)
    left_starts = _place_intervals(avail_h, left_lengths)
    right_starts = _place_intervals(avail_h, right_lengths)
    assert top_starts is not None and bottom_starts is not None
    assert left_starts is not None and right_starts is not None

    markers = [RED, BLUE, GREEN, YELLOW]
    while len(markers) < n_pieces:
        markers.append(random.choice(MARKERS))
    random.shuffle(markers)

    extras = {
        "top": [random.randint(1, 4) for _ in range(top_n)],
        "bottom": [random.randint(1, 4) for _ in range(bottom_n)],
        "left": [random.randint(1, 4) for _ in range(left_n)],
        "right": [random.randint(1, 4) for _ in range(right_n)],
    }

    top_ext = max((thickness + gap for gap in extras["top"]), default=0)
    bottom_ext = max((thickness + gap for gap in extras["bottom"]), default=0)
    left_ext = max((thickness + gap for gap in extras["left"]), default=0)
    right_ext = max((thickness + gap for gap in extras["right"]), default=0)

    pad_top = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_right = random.randint(1, 3)

    frame_r0 = pad_top + top_ext
    frame_c0 = pad_left + left_ext
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1
    grid_h = frame_h + frame_r0 + bottom_ext + pad_bottom
    grid_w = frame_w + frame_c0 + right_ext + pad_right

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = GREY
    grid[
        frame_r0 + thickness : frame_r1 - thickness + 1,
        frame_c0 + thickness : frame_c1 - thickness + 1,
    ] = BG

    queue = list(markers)
    specs: list[dict] = []

    for side, starts, lengths in (
        ("top", top_starts, top_lengths),
        ("bottom", bottom_starts, bottom_lengths),
        ("left", left_starts, left_lengths),
        ("right", right_starts, right_lengths),
    ):
        for idx, (start, length) in enumerate(zip(starts, lengths)):
            marker = int(queue.pop())
            if side == "top":
                sr0 = frame_r0
                sc0 = frame_c0 + thickness + start
                slot_bbox = (sr0, sc0, sr0 + thickness - 1, sc0 + length - 1)
                piece_pos = (frame_r0 - extras[side][idx] - thickness, sc0)
                patch_shape = (thickness, length)
            elif side == "bottom":
                sr0 = frame_r1 - thickness + 1
                sc0 = frame_c0 + thickness + start
                slot_bbox = (sr0, sc0, sr0 + thickness - 1, sc0 + length - 1)
                piece_pos = (frame_r1 + extras[side][idx] + 1, sc0)
                patch_shape = (thickness, length)
            elif side == "left":
                sr0 = frame_r0 + thickness + start
                sc0 = frame_c0
                slot_bbox = (sr0, sc0, sr0 + length - 1, sc0 + thickness - 1)
                piece_pos = (sr0, frame_c0 - extras[side][idx] - thickness)
                patch_shape = (length, thickness)
            else:
                sr0 = frame_r0 + thickness + start
                sc0 = frame_c1 - thickness + 1
                slot_bbox = (sr0, sc0, sr0 + length - 1, sc0 + thickness - 1)
                piece_pos = (sr0, frame_c1 + extras[side][idx] + 1)
                patch_shape = (length, thickness)

            mask = _make_piece_mask(patch_shape[0], patch_shape[1], side, marker)
            patch = _make_piece_patch(mask, marker)
            pr0, pc0 = piece_pos
            ph, pw = patch.shape
            assert 0 <= pr0 and 0 <= pc0
            assert pr0 + ph <= grid_h and pc0 + pw <= grid_w
            view = grid[pr0 : pr0 + ph, pc0 : pc0 + pw]
            assert not np.any((view != BG) & mask)
            view[mask] = patch[mask]
            sr0, sc0, sr1, sc1 = slot_bbox
            grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
            specs.append(
                {
                    "side": side,
                    "slot_bbox": slot_bbox,
                    "piece_pos": piece_pos,
                    "mask": mask,
                    "patch": patch,
                    "marker": marker,
                }
            )

    expected = _build_expected(grid, specs)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, expected = _build_case()
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
