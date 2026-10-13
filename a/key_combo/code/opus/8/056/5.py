from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
TARGET_SIZE = 3
TAIL_DIRS = ("up", "down", "left", "right")
PALETTE = tuple(range(1, 10))

_GEN_MAX_ATTEMPTS = 600
_GEN_CALL_COUNT = 0
_RECENT_OUTPUT_SIGS: list[bytes] = []
_RECENT_OUTPUT_SET: set[bytes] = set()
_RECENT_OUTPUT_LIMIT = 128


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_color_components(grid: np.ndarray, color: int, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    if int(connectivity) == 8:
        neighbor_steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    else:
        neighbor_steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in neighbor_steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0 + 1, c1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_frame_component(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict | None:
    comp_set = set(coords)
    cr0, cc0, cr1, cc1 = _bbox(coords)
    comp_h = cr1 - cr0 + 1
    comp_w = cc1 - cc0 + 1
    matches: list[dict] = []

    def try_candidate(direction: str, tail_len: int) -> None:
        if direction == "up":
            fr0, fc0, fr1, fc1 = cr0 + tail_len, cc0, cr1, cc1
        elif direction == "down":
            fr0, fc0, fr1, fc1 = cr0, cc0, cr1 - tail_len, cc1
        elif direction == "left":
            fr0, fc0, fr1, fc1 = cr0, cc0 + tail_len, cr1, cc1
        else:
            fr0, fc0, fr1, fc1 = cr0, cc0, cr1, cc1 - tail_len

        fh = fr1 - fr0 + 1
        fw = fc1 - fc0 + 1
        if fh < 5 or fw < 5:
            return

        corners = [(fr0, fc0), (fr0, fc1), (fr1, fc0), (fr1, fc1)]
        if any(int(grid[r, c]) != BG for r, c in corners):
            return

        border = _border_cells(fr0, fc0, fr1, fc1)
        if any(int(grid[r, c]) != int(color) for r, c in border):
            return
        if len(comp_set) != len(border) + tail_len:
            return
        tail_cells = comp_set - border
        if len(tail_cells) != tail_len:
            return

        if direction in ("up", "down"):
            cols = {c for _, c in tail_cells}
            if len(cols) != 1:
                return
            attach = next(iter(cols))
            if not (fc0 < attach < fc1):
                return
            rows = sorted(r for r, _ in tail_cells)
            expected_rows = list(range(cr0, fr0)) if direction == "up" else list(range(fr1 + 1, cr1 + 1))
            if rows != expected_rows:
                return
        else:
            rows = {r for r, _ in tail_cells}
            if len(rows) != 1:
                return
            attach = next(iter(rows))
            if not (fr0 < attach < fr1):
                return
            cols = sorted(c for _, c in tail_cells)
            expected_cols = list(range(cc0, fc0)) if direction == "left" else list(range(fc1 + 1, cc1 + 1))
            if cols != expected_cols:
                return

        matches.append(
            {
                "color": int(color),
                "frame_bbox": (fr0, fc0, fr1, fc1),
                "tail_dir": direction,
                "tail_len": int(tail_len),
                "tail_attach": int(attach),
                "component": comp_set,
            }
        )

    if comp_h >= 6:
        for tail_len in range(1, min(5, comp_h - 5) + 1):
            try_candidate("up", tail_len)
            try_candidate("down", tail_len)
    if comp_w >= 6:
        for tail_len in range(1, min(5, comp_w - 5) + 1):
            try_candidate("left", tail_len)
            try_candidate("right", tail_len)

    if len(matches) != 1:
        return None
    return matches[0]


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _get_bg(g) == BG, "expected black background"

    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for comp in _find_color_components(g, color, connectivity=8):
            frame = _detect_frame_component(g, color, comp)
            if frame is not None:
                frames.append(frame)

    assert 3 <= len(frames) <= 4, "expected 3-4 rounded frames"
    tail_lengths = [int(frame["tail_len"]) for frame in frames]
    assert len(set(tail_lengths)) == len(tail_lengths), "tail lengths must be distinct"

    used_boxes: list[tuple[int, int, int, int]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["frame_bbox"]
        for or0, oc0, or1, oc1 in used_boxes:
            separated = r1 < or0 or or1 < r0 or c1 < oc0 or oc1 < c0
            assert separated, "frame boxes must not overlap"
        used_boxes.append((r0, c0, r1, c1))
    return frames


def _crop_motif(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    coords = np.argwhere(interior != BG)
    assert coords.size > 0, "each frame must contain a motif"
    ir0, ic0 = coords.min(axis=0)
    ir1, ic1 = coords.max(axis=0)
    crop = interior[int(ir0) : int(ir1) + 1, int(ic0) : int(ic1) + 1]
    assert crop.shape[0] <= TARGET_SIZE and crop.shape[1] <= TARGET_SIZE
    return crop


def _center_in_3x3(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    out = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=int)
    r0 = (TARGET_SIZE - h) // 2
    c0 = (TARGET_SIZE - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = crop
    return out


def _top_left_in_3x3(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape
    out = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=int)
    out[:h, :w] = crop
    return out


def _overlay_layers(layers: list[np.ndarray], order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=int)
    owner = np.full((TARGET_SIZE, TARGET_SIZE), -1, dtype=int)
    for idx in order:
        layer = layers[idx]
        mask = layer != BG
        out[mask] = layer[mask]
        owner[mask] = int(idx)
    return out, owner


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize the motifs inside rounded frames and composite them by reverse tail-length priority.

    Input structure:
    - The background is black.
    - The grid contains 3-4 separated rounded rectangular frames: each frame has black corners, one border color on every other perimeter cell, and one straight one-pixel tail extending from exactly one side.
    - Inside each frame is a small non-black motif on black background; its cropped bounding box fits within 3x3.

    Rule:
    - Detect each rounded frame and its tail, then crop the interior motif to the minimal bounding box of non-black cells.
    - Center each cropped motif inside a 3x3 canvas.
    - Sort frames by tail length from long to short, treating the longest tail as the lowest layer and the shortest tail as the top layer.
    - Transparently overlay the normalized 3x3 motifs in that order: non-black cells overwrite lower layers, black cells stay transparent.

    Key insight:
    - The tail does not move the motif at all; it only encodes layer priority, while the rounded frame is just a carrier whose interior must be cropped and normalized before the motifs can share one coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    layers: list[tuple[int, np.ndarray]] = []
    for frame in frames:
        crop = _crop_motif(g, frame["frame_bbox"])
        layers.append((int(frame["tail_len"]), _center_in_3x3(crop)))

    out = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=int)
    for _tail_len, layer in sorted(layers, key=lambda item: item[0], reverse=True):
        mask = layer != BG
        out[mask] = layer[mask]
    return out


def _make_crop(rng: np.random.Generator, h: int, w: int, colors: list[int]) -> np.ndarray:
    crop = np.zeros((h, w), dtype=int)
    if h == 1 and w == 1:
        coords = [(0, 0)]
    elif h == 1:
        coords = [(0, 0), (0, w - 1)]
    elif w == 1:
        coords = [(0, 0), (h - 1, 0)]
    else:
        if int(rng.integers(2)) == 0:
            coords = [(0, 0), (h - 1, w - 1)]
        else:
            coords = [(0, w - 1), (h - 1, 0)]

    remaining = [(r, c) for r in range(h) for c in range(w) if (r, c) not in coords]
    min_count = len(coords)
    max_count = h * w
    total = int(rng.integers(min_count, max_count + 1))
    if total > min_count:
        pick = rng.choice(len(remaining), size=total - min_count, replace=False)
        coords.extend(remaining[int(i)] for i in np.atleast_1d(pick))

    primary = int(colors[0])
    for r, c in coords:
        crop[r, c] = primary
    if len(colors) >= 2:
        accent = int(colors[1])
        accent_count = min(len(coords), 1 + int(rng.integers(2)))
        pick = rng.choice(len(coords), size=accent_count, replace=False)
        for idx in np.atleast_1d(pick):
            r, c = coords[int(idx)]
            crop[r, c] = accent
    return crop


def _full_bbox_for_object(spec: dict) -> tuple[int, int]:
    if spec["tail_dir"] in ("up", "down"):
        return int(spec["frame_h"] + spec["tail_len"]), int(spec["frame_w"])
    return int(spec["frame_h"]), int(spec["frame_w"] + spec["tail_len"])


def _too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    return not separated


def _render_object(grid: np.ndarray, spec: dict, top: int, left: int) -> dict:
    if spec["tail_dir"] == "up":
        fr0, fc0 = top + spec["tail_len"], left
    elif spec["tail_dir"] == "down":
        fr0, fc0 = top, left
    elif spec["tail_dir"] == "left":
        fr0, fc0 = top, left + spec["tail_len"]
    else:
        fr0, fc0 = top, left

    fr1 = fr0 + spec["frame_h"] - 1
    fc1 = fc0 + spec["frame_w"] - 1
    color = int(spec["frame_color"])

    for c in range(fc0 + 1, fc1):
        grid[fr0, c] = color
        grid[fr1, c] = color
    for r in range(fr0 + 1, fr1):
        grid[r, fc0] = color
        grid[r, fc1] = color

    attach = int(spec["tail_attach"])
    if spec["tail_dir"] == "up":
        for r in range(fr0 - spec["tail_len"], fr0):
            grid[r, fc0 + attach] = color
    elif spec["tail_dir"] == "down":
        for r in range(fr1 + 1, fr1 + spec["tail_len"] + 1):
            grid[r, fc0 + attach] = color
    elif spec["tail_dir"] == "left":
        for c in range(fc0 - spec["tail_len"], fc0):
            grid[fr0 + attach, c] = color
    else:
        for c in range(fc1 + 1, fc1 + spec["tail_len"] + 1):
            grid[fr0 + attach, c] = color

    crop = spec["crop"]
    mh, mw = crop.shape
    ir = fr0 + 1 + spec["motif_top"]
    ic = fc0 + 1 + spec["motif_left"]
    grid[ir : ir + mh, ic : ic + mw] = crop

    return {
        "frame_color": int(spec["frame_color"]),
        "tail_len": int(spec["tail_len"]),
        "tail_dir": spec["tail_dir"],
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "full_bbox": (top, left, top + spec["full_h"] - 1, left + spec["full_w"] - 1),
    }


def _sample_specs(rng: np.random.Generator, n_objects: int, force_multicolor: bool) -> tuple[list[dict], list[np.ndarray], list[int]]:
    frame_colors = [int(x) for x in rng.choice(np.array(PALETTE, dtype=int), size=n_objects, replace=False)]
    motif_palette = [c for c in PALETTE if c not in frame_colors]
    rng.shuffle(motif_palette)

    multi_idx = int(rng.integers(n_objects)) if force_multicolor and len(motif_palette) > n_objects else -1
    tail_lengths = [int(x) for x in rng.choice(np.arange(1, 6), size=n_objects, replace=False)]
    specs: list[dict] = []
    normalized_layers: list[np.ndarray] = []
    size_options = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

    for idx in range(n_objects):
        crop_h, crop_w = size_options[int(rng.integers(len(size_options)))]
        if idx == 0 and crop_h == 3 and crop_w == 3:
            crop_h, crop_w = size_options[int(rng.integers(len(size_options) - 1))]
        colors = [motif_palette[idx]]
        if idx == multi_idx:
            colors.append(motif_palette[n_objects])
        crop = _make_crop(rng, crop_h, crop_w, colors)
        normalized = _center_in_3x3(crop)

        frame_h = int(rng.integers(5, 10))
        frame_w = int(rng.integers(5, 10))
        tail_dir = TAIL_DIRS[int(rng.integers(len(TAIL_DIRS)))]
        tail_attach = int(rng.integers(1, frame_w - 1)) if tail_dir in ("up", "down") else int(rng.integers(1, frame_h - 1))
        interior_h = frame_h - 2
        interior_w = frame_w - 2
        motif_top = int(rng.integers(0, interior_h - crop_h + 1))
        motif_left = int(rng.integers(0, interior_w - crop_w + 1))
        full_h, full_w = _full_bbox_for_object(
            {
                "frame_h": frame_h,
                "frame_w": frame_w,
                "tail_dir": tail_dir,
                "tail_len": tail_lengths[idx],
            }
        )
        specs.append(
            {
                "frame_color": int(frame_colors[idx]),
                "tail_len": int(tail_lengths[idx]),
                "tail_dir": tail_dir,
                "tail_attach": tail_attach,
                "frame_h": frame_h,
                "frame_w": frame_w,
                "full_h": full_h,
                "full_w": full_w,
                "crop": crop,
                "motif_top": motif_top,
                "motif_left": motif_left,
            }
        )
        normalized_layers.append(normalized)

    return specs, normalized_layers, tail_lengths


def _layers_pass_semantics(crops: list[np.ndarray], layers: list[np.ndarray], order: list[int]) -> bool:
    final, owner = _overlay_layers(layers, order)
    if not np.any(final != BG):
        return False
    visible_owners = {int(x) for x in owner.ravel() if int(x) >= 0}
    if len(visible_owners) < 2:
        return False

    overlap_conflict = False
    for r in range(TARGET_SIZE):
        for c in range(TARGET_SIZE):
            vals = [int(layer[r, c]) for layer in layers if int(layer[r, c]) != BG]
            if len(set(vals)) >= 2:
                overlap_conflict = True
                break
        if overlap_conflict:
            break
    if not overlap_conflict:
        return False

    reverse_out, _ = _overlay_layers(layers, list(reversed(order)))
    if np.array_equal(final, reverse_out):
        return False

    top_idx = int(order[-1])
    if np.array_equal(final, layers[top_idx]):
        return False

    alt_layers = [_top_left_in_3x3(crop) for crop in crops]
    alt_out, _ = _overlay_layers(alt_layers, order)
    if np.array_equal(final, alt_out):
        return False

    small_visible = False
    for idx, crop in enumerate(crops):
        if crop.shape != (3, 3) and np.any(owner == idx):
            small_visible = True
            break
    if not small_visible:
        return False
    return True


def _place_specs(rng: np.random.Generator, specs: list[dict], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(specs)), key=lambda i: specs[i]["full_h"] * specs[i]["full_w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        fh, fw = spec["full_h"], spec["full_w"]
        candidates = [(r, c) for r in range(h - fh + 1) for c in range(w - fw + 1)]
        rng.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            box = (top, left, top + fh - 1, left + fw - 1)
            if any(_too_close(box, other, gap=2) for other in boxes):
                continue
            chosen = (top, left)
            boxes.append(box)
            break
        if chosen is None:
            return None
        placed.append((idx, chosen[0], chosen[1]))

    placed.sort()
    return [(top, left) for _idx, top, left in placed]


def _remember_output(out: np.ndarray) -> bool:
    sig = out.tobytes()
    if sig in _RECENT_OUTPUT_SET:
        return False
    _RECENT_OUTPUT_SIGS.append(sig)
    _RECENT_OUTPUT_SET.add(sig)
    if len(_RECENT_OUTPUT_SIGS) > _RECENT_OUTPUT_LIMIT:
        old = _RECENT_OUTPUT_SIGS.pop(0)
        _RECENT_OUTPUT_SET.remove(old)
    return True


def _semantic_validation(grid: np.ndarray, specs: list[dict], placements: list[tuple[int, int]], layers: list[np.ndarray], order: list[int]) -> bool:
    try:
        frames = _extract_frames(grid)
    except Exception:
        return False
    if len(frames) != len(specs):
        return False

    detected_by_color = {int(frame["color"]): frame for frame in frames}
    expected_bboxes: list[tuple[int, int, int, int]] = []
    area_order: list[int] = []
    position_order: list[int] = []
    for idx, spec in enumerate(specs):
        top, left = placements[idx]
        if spec["tail_dir"] == "up":
            fr0, fc0 = top + spec["tail_len"], left
        elif spec["tail_dir"] == "down":
            fr0, fc0 = top, left
        elif spec["tail_dir"] == "left":
            fr0, fc0 = top, left + spec["tail_len"]
        else:
            fr0, fc0 = top, left
        bbox = (fr0, fc0, fr0 + spec["frame_h"] - 1, fc0 + spec["frame_w"] - 1)
        expected_bboxes.append(bbox)

        frame = detected_by_color.get(int(spec["frame_color"]))
        if frame is None:
            return False
        if tuple(frame["frame_bbox"]) != bbox:
            return False
        if int(frame["tail_len"]) != int(spec["tail_len"]) or frame["tail_dir"] != spec["tail_dir"]:
            return False

        crop = _crop_motif(grid, bbox)
        if not np.array_equal(crop, spec["crop"]):
            return False

        area_order.append(idx)
        position_order.append(idx)

    correct, _ = _overlay_layers(layers, order)
    if not np.array_equal(main(grid), correct):
        return False

    area_order.sort(key=lambda idx: specs[idx]["frame_h"] * specs[idx]["frame_w"], reverse=True)
    if area_order != order:
        alt, _ = _overlay_layers(layers, area_order)
        if np.array_equal(correct, alt):
            return False

    position_order.sort(key=lambda idx: expected_bboxes[idx][:2])
    if position_order != order:
        alt, _ = _overlay_layers(layers, position_order)
        if np.array_equal(correct, alt):
            return False

    if not _remember_output(correct):
        return False
    return True


def _gen_attempt(rng: np.random.Generator, call_idx: int) -> np.ndarray | None:
    n_objects = 3 + (call_idx % 2)
    force_multicolor = call_idx % 3 == 0
    specs, layers, tail_lengths = _sample_specs(rng, n_objects, force_multicolor)
    order = sorted(range(n_objects), key=lambda idx: tail_lengths[idx], reverse=True)
    crops = [spec["crop"] for spec in specs]
    if not _layers_pass_semantics(crops, layers, order):
        return None

    grid_h = int(rng.integers(23, 30))
    grid_w = int(rng.integers(23, 30))
    placements = _place_specs(rng, specs, (grid_h, grid_w))
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec, (top, left) in zip(specs, placements):
        _render_object(grid, spec, top, left)

    if not _semantic_validation(grid, specs, placements, layers, order):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng, _GEN_CALL_COUNT)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate valid rounded-frame overlay input")
