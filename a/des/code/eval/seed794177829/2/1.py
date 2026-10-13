from __future__ import annotations

import numpy as np


OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BG = 0
FRAME_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (26, 34)
GRID_W_RANGE = (26, 34)
FRAME_COUNT_RANGE = (2, 5)
FRAME_H_RANGE = (7, 10)
FRAME_W_RANGE = (7, 10)
SELECTED_DIST_RANGE = (1, 4)
GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _most_common(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _coords_to_local_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mark_perimeter(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _analyze_frame(
    g: np.ndarray, bg: int, frame_color: int, r0: int, c0: int, r1: int, c1: int
) -> dict | None:
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[1:-1, 1:-1] == bg):
        return None

    corners = [int(sub[0, 0]), int(sub[0, -1]), int(sub[-1, 0]), int(sub[-1, -1])]
    if any(v != frame_color for v in corners):
        return None

    sides = {
        "top": sub[0, 1:-1],
        "bottom": sub[-1, 1:-1],
        "left": sub[1:-1, 0],
        "right": sub[1:-1, -1],
    }
    gap_side = None
    gap_idx = None
    marker_side = None
    marker_idx = None
    marker_color = None

    for side, arr in sides.items():
        bg_idx = np.flatnonzero(arr == bg)
        if bg_idx.size:
            if gap_side is not None or int(bg_idx.size) > 3:
                return None
            if int(bg_idx[-1] - bg_idx[0] + 1) != int(bg_idx.size):
                return None
            gap_side = side
            gap_idx = (int(bg_idx[0]), int(bg_idx[-1]))

        other_idx = np.flatnonzero((arr != bg) & (arr != frame_color))
        if other_idx.size:
            if marker_side is not None or int(other_idx.size) != 1:
                return None
            marker_side = side
            marker_idx = int(other_idx[0])
            marker_color = int(arr[marker_idx])

        allowed = (arr == frame_color) | (arr == bg) | ((arr != bg) & (arr != frame_color))
        if not np.all(allowed):
            return None

    if gap_side is None or marker_side is None or marker_color is None:
        return None
    if OPPOSITE[gap_side] != marker_side:
        return None

    if gap_side in ("top", "bottom"):
        band0 = int(c0 + 1 + gap_idx[0])
        band1 = int(c0 + 1 + gap_idx[1])
    else:
        band0 = int(r0 + 1 + gap_idx[0])
        band1 = int(r0 + 1 + gap_idx[1])

    if marker_side == "top":
        marker_pos = (int(r0), int(c0 + 1 + marker_idx))
    elif marker_side == "bottom":
        marker_pos = (int(r1), int(c0 + 1 + marker_idx))
    elif marker_side == "left":
        marker_pos = (int(r0 + 1 + marker_idx), int(c0))
    else:
        marker_pos = (int(r0 + 1 + marker_idx), int(c1))

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": gap_side,
        "gap_band": (int(band0), int(band1)),
        "marker_color": int(marker_color),
        "marker_pos": marker_pos,
    }


def _find_frames(g: np.ndarray) -> tuple[int, list[dict]]:
    bg = _most_common(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected non-background structure"
    frame_color = _most_common(non_bg)

    h, w = g.shape
    frames: list[dict] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            if int(g[r0, c0]) != frame_color:
                continue
            for r1 in range(r0 + 4, h):
                if int(g[r1, c0]) != frame_color:
                    continue
                for c1 in range(c0 + 4, w):
                    if int(g[r0, c1]) != frame_color or int(g[r1, c1]) != frame_color:
                        continue
                    info = _analyze_frame(g, bg, frame_color, r0, c0, r1, c1)
                    if info is not None:
                        frames.append(info)

    assert frames, "expected at least one frame"
    frames.sort(key=lambda item: item["bbox"])
    return int(bg), frames


def _extract_fragments(g: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        _mark_perimeter(frame_mask, frame["bbox"])

    frag_mask = (g != bg) & (~frame_mask)
    comps = _components(frag_mask)
    fragments: list[dict] = []
    for coords in comps:
        bbox = _bbox(coords)
        mask = _coords_to_local_mask(coords, bbox)
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "fragments must be monochrome"
        fragments.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "color": int(colors[0]),
            }
        )
    fragments.sort(key=lambda item: item["bbox"])
    return fragments


def _aligned_distance(frame: dict, fragment: dict) -> int | None:
    fr0, fc0, fr1, fc1 = frame["bbox"]
    r0, c0, r1, c1 = fragment["bbox"]
    band0, band1 = frame["gap_band"]
    side = frame["gap_side"]

    if side == "top":
        if r1 >= fr0 or not _interval_overlap(c0, c1, band0, band1):
            return None
        return int(fr0 - r1 - 1)
    if side == "bottom":
        if r0 <= fr1 or not _interval_overlap(c0, c1, band0, band1):
            return None
        return int(r0 - fr1 - 1)
    if side == "left":
        if c1 >= fc0 or not _interval_overlap(r0, r1, band0, band1):
            return None
        return int(fc0 - c1 - 1)
    if c0 <= fc1 or not _interval_overlap(r0, r1, band0, band1):
        return None
    return int(c0 - fc1 - 1)


def _target_shift(frame: dict, fragment: dict) -> tuple[int, int]:
    fr0, fc0, fr1, fc1 = frame["bbox"]
    r0, c0, r1, c1 = fragment["bbox"]
    side = frame["gap_side"]
    if side == "top":
        return int(fr1 - 1 - r1), 0
    if side == "bottom":
        return int(fr0 + 1 - r0), 0
    if side == "left":
        return 0, int(fc1 - 1 - c1)
    return 0, int(fc0 + 1 - c0)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, frames = _find_frames(g)
    fragments = _extract_fragments(g, bg, frames)

    out = g.copy()
    used: set[int] = set()
    moves: list[tuple[dict, dict]] = []
    for frame in frames:
        candidates = []
        for idx, fragment in enumerate(fragments):
            if idx in used:
                continue
            dist = _aligned_distance(frame, fragment)
            if dist is not None:
                candidates.append((int(dist), idx))
        assert candidates, "every frame must see at least one aligned fragment"
        candidates.sort(key=lambda item: (item[0], fragments[item[1]]["bbox"]))
        best_dist = int(candidates[0][0])
        best = [idx for dist, idx in candidates if int(dist) == best_dist]
        assert len(best) == 1, "nearest aligned fragment must be unique"
        frag_idx = int(best[0])
        used.add(frag_idx)
        moves.append((frame, fragments[frag_idx]))

    for _frame, fragment in moves:
        out[fragment["coords"][:, 0], fragment["coords"][:, 1]] = bg

    for frame, fragment in moves:
        dr, dc = _target_shift(frame, fragment)
        shifted = fragment["coords"] + np.array([dr, dc], dtype=int)
        fr0, fc0, fr1, fc1 = frame["bbox"]
        assert int(shifted[:, 0].min()) >= int(fr0 + 1)
        assert int(shifted[:, 0].max()) <= int(fr1 - 1)
        assert int(shifted[:, 1].min()) >= int(fc0 + 1)
        assert int(shifted[:, 1].max()) <= int(fc1 - 1)
        out[shifted[:, 0], shifted[:, 1]] = int(frame["marker_color"])

    return out


def _random_connected_mask(
    rng: np.random.Generator, h: int, w: int, area: int, *, full_bbox: bool
) -> np.ndarray | None:
    for _ in range(120):
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if full_bbox:
            if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
        return mask
    return None


def _sample_selected_fragment(
    rng: np.random.Generator, direction: str, interior_h: int, interior_w: int
) -> tuple[np.ndarray, int]:
    for _ in range(120):
        if direction in ("top", "bottom"):
            frag_h = int(rng.integers(2, min(4, interior_h + 1)))
            frag_w = int(rng.integers(1, min(4, interior_w + 1)))
            gap_len = int(rng.integers(frag_w, min(3, interior_w) + 1))
        else:
            frag_h = int(rng.integers(1, min(4, interior_h + 1)))
            frag_w = int(rng.integers(2, min(4, interior_w + 1)))
            gap_len = int(rng.integers(frag_h, min(3, interior_h) + 1))
        area_lo = max(3, frag_h + frag_w - 1)
        area_hi = min(6, frag_h * frag_w)
        if area_lo > area_hi:
            continue
        area = int(rng.integers(area_lo, area_hi + 1))
        mask = _random_connected_mask(rng, frag_h, frag_w, area, full_bbox=True)
        if mask is not None:
            return mask, int(gap_len)
    raise RuntimeError("failed to sample selected fragment")


def _sample_free_fragment(rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        h = int(rng.integers(1, 4))
        w = int(rng.integers(1, 4))
        area_lo = max(3, h + w - 1)
        area_hi = min(6, h * w)
        if area_lo > area_hi:
            continue
        area = int(rng.integers(area_lo, area_hi + 1))
        mask = _random_connected_mask(rng, h, w, area, full_bbox=True)
        if mask is not None:
            return mask
    raise RuntimeError("failed to sample fragment")


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = int(color)


def _build_frame_patch(
    h: int,
    w: int,
    gap_side: str,
    gap_start: int,
    gap_len: int,
    marker_color: int,
    marker_offset: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    if gap_side == "top":
        patch[0, gap_start : gap_start + gap_len] = BG
        patch[-1, marker_offset] = marker_color
    elif gap_side == "bottom":
        patch[-1, gap_start : gap_start + gap_len] = BG
        patch[0, marker_offset] = marker_color
    elif gap_side == "left":
        patch[gap_start : gap_start + gap_len, 0] = BG
        patch[marker_offset, -1] = marker_color
    else:
        patch[gap_start : gap_start + gap_len, -1] = BG
        patch[marker_offset, 0] = marker_color

    return patch, (0, 0, h - 1, w - 1)


def _expanded_bbox_has_occupancy(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(occ.shape[0], int(r0 + h + pad))
    cc1 = min(occ.shape[1], int(c0 + w + pad))
    return bool(occ[rr0:rr1, cc0:cc1].any())


def _fragment_bbox(mask: np.ndarray, r0: int, c0: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + mask.shape[0] - 1), int(c0 + mask.shape[1] - 1)


def _latent_frame_expected(frame: dict, fragment: dict, out: np.ndarray) -> None:
    out[fragment["coords"][:, 0], fragment["coords"][:, 1]] = BG
    dr, dc = _target_shift(frame, fragment)
    shifted = fragment["coords"] + np.array([dr, dc], dtype=int)
    out[shifted[:, 0], shifted[:, 1]] = int(frame["marker_color"])


def _attempt_layout(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()

    split_r = int(rng.integers(h // 2 - 1, h // 2 + 2))
    split_c = int(rng.integers(w // 2 - 1, w // 2 + 2))
    slots = [
        (1, split_r - 2, 1, split_c - 2),
        (1, split_r - 2, split_c + 1, w - 2),
        (split_r + 1, h - 2, 1, split_c - 2),
        (split_r + 1, h - 2, split_c + 1, w - 2),
    ]
    slot_ids = list(range(4))
    rng.shuffle(slot_ids)

    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))
    chosen_slots = slot_ids[:n_frames]
    directions = list(np.array(["top", "bottom", "left", "right"])[rng.permutation(4)[:n_frames]])

    marker_colors = list(np.array(PALETTE)[rng.permutation(len(PALETTE))[:n_frames]])
    remaining = [c for c in PALETTE if c not in marker_colors]
    rng.shuffle(remaining)
    selected_colors = remaining[:n_frames]
    if len(selected_colors) < n_frames:
        return None

    occ = np.zeros((h, w), dtype=bool)
    placed_frames: list[dict] = []
    selected_frags: list[dict] = []

    for slot_idx, direction, marker_color, selected_color in zip(
        chosen_slots, directions, marker_colors, selected_colors
    ):
        sr0, sr1, sc0, sc1 = slots[int(slot_idx)]
        if sr1 - sr0 + 1 < 10 or sc1 - sc0 + 1 < 10:
            return None

        frame_h = int(rng.integers(*FRAME_H_RANGE))
        frame_w = int(rng.integers(*FRAME_W_RANGE))
        if frame_h > sr1 - sr0 + 1 or frame_w > sc1 - sc0 + 1:
            return None

        interior_h = int(frame_h - 2)
        interior_w = int(frame_w - 2)
        selected_mask, gap_len = _sample_selected_fragment(rng, direction, interior_h, interior_w)
        frag_h, frag_w = map(int, selected_mask.shape)
        dist = int(rng.integers(*SELECTED_DIST_RANGE))

        r_lo = sr0
        r_hi = sr1 - frame_h + 1
        c_lo = sc0
        c_hi = sc1 - frame_w + 1
        if direction == "top":
            r_lo = max(r_lo, sr0 + frag_h + dist)
        elif direction == "bottom":
            r_hi = min(r_hi, sr1 - frame_h - frag_h - dist + 1)
        elif direction == "left":
            c_lo = max(c_lo, sc0 + frag_w + dist)
        else:
            c_hi = min(c_hi, sc1 - frame_w - frag_w - dist + 1)
        if r_lo > r_hi or c_lo > c_hi:
            return None

        frame_r0 = int(rng.integers(r_lo, r_hi + 1))
        frame_c0 = int(rng.integers(c_lo, c_hi + 1))
        frame_r1 = int(frame_r0 + frame_h - 1)
        frame_c1 = int(frame_c0 + frame_w - 1)

        if direction in ("top", "bottom"):
            gap_start = int(rng.integers(1, frame_w - gap_len))
            marker_choices = [c for c in range(1, frame_w - 1)]
            marker_offset = int(marker_choices[int(rng.integers(0, len(marker_choices)))])
            frag_c0 = int(frame_c0 + gap_start + rng.integers(0, gap_len - frag_w + 1))
            if direction == "top":
                frag_r0 = int(frame_r0 - dist - frag_h)
            else:
                frag_r0 = int(frame_r1 + dist + 1)
        else:
            gap_start = int(rng.integers(1, frame_h - gap_len))
            marker_choices = [r for r in range(1, frame_h - 1)]
            marker_offset = int(marker_choices[int(rng.integers(0, len(marker_choices)))])
            frag_r0 = int(frame_r0 + gap_start + rng.integers(0, gap_len - frag_h + 1))
            if direction == "left":
                frag_c0 = int(frame_c0 - dist - frag_w)
            else:
                frag_c0 = int(frame_c1 + dist + 1)

        patch, _ = _build_frame_patch(
            frame_h, frame_w, direction, gap_start, gap_len, int(marker_color), int(marker_offset)
        )

        if _expanded_bbox_has_occupancy(occ, frame_r0, frame_c0, frame_h, frame_w, 0):
            return None
        if _expanded_bbox_has_occupancy(occ, frag_r0, frag_c0, frag_h, frag_w, 0):
            return None

        _stamp_mask(grid, patch != BG, frame_r0, frame_c0, FRAME_COLOR)
        marker_coords = np.argwhere((patch != BG) & (patch != FRAME_COLOR))
        if marker_coords.shape[0] != 1:
            return None
        mr, mc = map(int, marker_coords[0])
        grid[frame_r0 + mr, frame_c0 + mc] = int(marker_color)
        expected[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = grid[
            frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1
        ]

        _stamp_mask(grid, selected_mask, frag_r0, frag_c0, int(selected_color))
        expected[frag_r0 : frag_r0 + frag_h, frag_c0 : frag_c0 + frag_w] = grid[
            frag_r0 : frag_r0 + frag_h, frag_c0 : frag_c0 + frag_w
        ]

        occ[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = True
        occ[frag_r0 : frag_r0 + frag_h, frag_c0 : frag_c0 + frag_w] |= selected_mask

        frame_info = {
            "bbox": (frame_r0, frame_c0, frame_r1, frame_c1),
            "gap_side": direction,
            "gap_band": (
                int(frame_c0 + gap_start),
                int(frame_c0 + gap_start + gap_len - 1),
            )
            if direction in ("top", "bottom")
            else (
                int(frame_r0 + gap_start),
                int(frame_r0 + gap_start + gap_len - 1),
            ),
            "marker_color": int(marker_color),
        }
        frag_bbox = _fragment_bbox(selected_mask, frag_r0, frag_c0)
        frag_coords = np.argwhere(selected_mask) + np.array([frag_r0, frag_c0], dtype=int)
        fragment = {
            "bbox": frag_bbox,
            "coords": frag_coords,
            "mask": selected_mask.copy(),
            "color": int(selected_color),
        }
        placed_frames.append(frame_info)
        selected_frags.append(fragment)

    def aligns_any(mask: np.ndarray, r0: int, c0: int) -> bool:
        frag = {"bbox": _fragment_bbox(mask, r0, c0)}
        matched = 0
        for frame in placed_frames:
            if _aligned_distance(frame, frag) is not None:
                matched += 1
        return matched > 0

    for frame, fragment in zip(placed_frames, selected_frags):
        seen = 0
        for other in placed_frames:
            if _aligned_distance(other, fragment) is not None:
                seen += 1
        if seen != 1:
            return None
        _latent_frame_expected(frame, fragment, expected)

    marker_decoys = []
    for frame in placed_frames:
        marker_decoys.append((frame["marker_color"], _sample_free_fragment(rng)))

    extra_count = int(rng.integers(0, 4))
    extra_colors = [c for c in PALETTE if c not in marker_colors]
    rng.shuffle(extra_colors)
    extras = [(extra_colors[i % len(extra_colors)], _sample_free_fragment(rng)) for i in range(extra_count)]

    for color, mask in marker_decoys + extras:
        mh, mw = mask.shape
        placed = False
        for _ in range(240):
            r0 = int(rng.integers(1, h - mh - 1))
            c0 = int(rng.integers(1, w - mw - 1))
            if _expanded_bbox_has_occupancy(occ, r0, c0, mh, mw, 1):
                continue
            if aligns_any(mask, r0, c0):
                continue
            _stamp_mask(grid, mask, r0, c0, int(color))
            expected[r0 : r0 + mh, c0 : c0 + mw] = grid[r0 : r0 + mh, c0 : c0 + mw]
            occ[r0 : r0 + mh, c0 : c0 + mw] |= mask
            placed = True
            break
        if not placed:
            return None

    try:
        produced = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _attempt_layout(_RNG)
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid sample")
