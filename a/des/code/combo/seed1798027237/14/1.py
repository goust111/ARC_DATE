from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
GAP_LENGTHS = (2, 3, 4)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_key(coords: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), norm


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _bbox_intersects(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _classify_frame(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]]
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    frame_mask = sub == int(color)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(frame_mask & ~border):
        return None
    missing = border & ~frame_mask
    if not np.any(missing):
        return None
    if np.any(sub[missing] != BG):
        return None
    if np.any(sub[border & ~missing] != int(color)):
        return None

    side_hits = {
        "top": [(0, int(c)) for c in np.where(missing[0, :])[0]],
        "bottom": [(h - 1, int(c)) for c in np.where(missing[-1, :])[0]],
        "left": [(int(r), 0) for r in np.where(missing[:, 0])[0]],
        "right": [(int(r), w - 1) for r in np.where(missing[:, -1])[0]],
    }
    nonempty = [side for side, cells in side_hits.items() if cells]
    if len(nonempty) != 1:
        return None
    side = nonempty[0]
    if side in ("top", "bottom"):
        idxs = sorted(c for _, c in side_hits[side])
        dim = w
    else:
        idxs = sorted(r for r, _ in side_hits[side])
        dim = h
    if idxs[0] == 0 or idxs[-1] == dim - 1:
        return None
    if idxs[-1] - idxs[0] + 1 != len(idxs):
        return None

    gap_cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in side_hits[side]]
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "gap_cells": gap_cells,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for coords in _components(grid == int(color)):
            frame = _classify_frame(grid, int(color), coords)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames


def _fragment_info(grid: np.ndarray, frame: dict) -> tuple[list[tuple[int, int]], int]:
    r0, c0, r1, c1 = frame["bbox"]
    coords: list[tuple[int, int]] = []
    colors: set[int] = set()
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            val = int(grid[r, c])
            if val in (BG, int(frame["color"])):
                continue
            coords.append((int(r), int(c)))
            colors.add(val)
    assert coords, "each frame must contain interior fragments"
    assert len(colors) == 1, "each frame must use exactly one fragment color"
    return coords, next(iter(colors))


def _fragment_direction(
    coords: list[tuple[int, int]], frame_bbox: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = frame_bbox
    mean_r = sum(r for r, _ in coords) / len(coords)
    mean_c = sum(c for _, c in coords) / len(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = mean_r - center_r
    dc = mean_c - center_c
    if abs(dr) >= abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _outside_components_by_color(grid: np.ndarray, frames: list[dict]) -> dict[int, list[list[tuple[int, int]]]]:
    outside = np.ones(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    by_color: dict[int, list[list[tuple[int, int]]]] = defaultdict(list)
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for coords in _components((grid == int(color)) & outside):
            by_color[int(color)].append(coords)
    return by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame gap with the same-color external sprite, then recolor the patch
    to the frame color and erase the guiding fragments inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    outside_by_color = _outside_components_by_color(g, frames)

    for frame in frames:
        frag_coords, frag_color = _fragment_info(g, frame)
        indicator = _fragment_direction(frag_coords, frame["bbox"])
        gap_side = frame["side"]
        if indicator != gap_side:
            indicator = gap_side

        gap_key = _shape_key(frame["gap_cells"])
        candidates = outside_by_color.get(int(frag_color), [])
        match_index = None
        for idx, coords in enumerate(candidates):
            if _shape_key(coords) == gap_key:
                match_index = idx
                break
        assert match_index is not None, "no matching external sprite found"
        sprite_coords = candidates.pop(match_index)

        if indicator == gap_side:
            for r, c in frame["gap_cells"]:
                out[r, c] = int(frame["color"])
        for r, c in frag_coords:
            out[r, c] = BG
        for r, c in sprite_coords:
            out[r, c] = BG

    return out


def _make_frame_patch(
    frame_color: int,
    frag_color: int,
    fh: int,
    fw: int,
    side: str,
    gap_len: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    patch = np.zeros((fh, fw), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)

    if side in ("top", "bottom"):
        start = random.randint(1, fw - gap_len - 1)
        rr = 0 if side == "top" else fh - 1
        gap_rel = [(int(rr), int(c)) for c in range(start, start + gap_len)]
    else:
        start = random.randint(1, fh - gap_len - 1)
        cc = 0 if side == "left" else fw - 1
        gap_rel = [(int(r), int(cc)) for r in range(start, start + gap_len)]
    for r, c in gap_rel:
        patch[r, c] = BG

    inner_h = fh - 2
    inner_w = fw - 2
    band = max(2, min(inner_h if side in ("top", "bottom") else inner_w, 3))
    if side == "top":
        candidates = [(r, c) for r in range(1, 1 + band) for c in range(1, fw - 1)]
    elif side == "bottom":
        candidates = [(r, c) for r in range(fh - 1 - band, fh - 1) for c in range(1, fw - 1)]
    elif side == "left":
        candidates = [(r, c) for r in range(1, fh - 1) for c in range(1, 1 + band)]
    else:
        candidates = [(r, c) for r in range(1, fh - 1) for c in range(fw - 1 - band, fw - 1)]

    random.shuffle(candidates)
    frag_count = random.randint(3, min(6, len(candidates)))
    frag_rel = candidates[:frag_count]
    for r, c in frag_rel:
        patch[r, c] = int(frag_color)

    return patch, gap_rel, frag_rel


def _sprite_patch(side: str, gap_len: int, color: int) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.full((1, gap_len), int(color), dtype=int)
    return np.full((gap_len, 1), int(color), dtype=int)


def _place_bbox(
    canvas: np.ndarray,
    patch: np.ndarray,
    occupied: list[tuple[int, int, int, int]],
    *,
    pad: int,
    border: int,
    tries: int = 200,
) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = canvas.shape
    if H < h + 2 * border or W < w + 2 * border:
        return None
    for _ in range(tries):
        r0 = random.randint(border, H - h - border)
        c0 = random.randint(border, W - w - border)
        bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        if any(_bbox_intersects(bbox, prev, pad=pad) for prev in occupied):
            continue
        _paste(canvas, patch, r0, c0)
        occupied.append(bbox)
        return int(r0), int(c0)
    return None


def _expected_from_specs(
    grid: np.ndarray,
    specs: list[dict],
    sprite_places: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for spec, (sr, sc) in zip(specs, sprite_places):
        fh, fw = spec["frame_patch"].shape
        r0, c0 = spec["frame_pos"]
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if out[rr, cc] == int(spec["frag_color"]):
                    out[rr, cc] = BG
        for gr, gc in spec["gap_rel"]:
            out[r0 + gr, c0 + gc] = int(spec["frame_color"])

        sprite = spec["sprite_patch"]
        sh, sw = sprite.shape
        for rr in range(sh):
            for cc in range(sw):
                if sprite[rr, cc] != BG:
                    out[sr + rr, sc + cc] = BG
    return out


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.randint(2, 4)
    colors = [c for c in range(1, 10)]
    random.shuffle(colors)
    frame_colors = colors[:num_frames]
    frag_colors = colors[num_frames : 2 * num_frames]

    H = random.randint(22, 30)
    W = random.randint(22, 30)
    grid = np.zeros((H, W), dtype=int)

    sides = [random.choice(SIDES) for _ in range(num_frames)]
    gap_lengths = [random.choice(GAP_LENGTHS) for _ in range(num_frames)]
    if num_frames >= 2 and random.random() < 0.5:
        i, j = random.sample(range(num_frames), 2)
        shared = random.choice(GAP_LENGTHS)
        gap_lengths[i] = shared
        gap_lengths[j] = shared

    frame_specs: list[dict] = []
    for i in range(num_frames):
        fh = random.randint(6, 10)
        fw = random.randint(6, 10)
        side = sides[i]
        gap_len = gap_lengths[i]
        frame_patch, gap_rel, frag_rel = _make_frame_patch(
            int(frame_colors[i]),
            int(frag_colors[i]),
            fh,
            fw,
            side,
            gap_len,
        )
        frame_specs.append(
            {
                "frame_color": int(frame_colors[i]),
                "frag_color": int(frag_colors[i]),
                "frame_patch": frame_patch,
                "gap_rel": gap_rel,
                "frag_rel": frag_rel,
                "sprite_patch": _sprite_patch(side, gap_len, int(frag_colors[i])),
            }
        )

    occupied: list[tuple[int, int, int, int]] = []
    for spec in sorted(frame_specs, key=lambda item: -item["frame_patch"].size):
        pos = _place_bbox(grid, spec["frame_patch"], occupied, pad=1, border=1, tries=250)
        if pos is None:
            return None
        spec["frame_pos"] = pos

    sprite_places: list[tuple[int, int]] = []
    for spec in frame_specs:
        pos = _place_bbox(grid, spec["sprite_patch"], occupied, pad=1, border=1, tries=250)
        if pos is None:
            return None
        sprite_places.append(pos)

    expected = _expected_from_specs(grid, frame_specs, sprite_places)
    try:
        produced = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
