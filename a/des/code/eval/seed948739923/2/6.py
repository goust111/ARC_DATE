from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
MAROON = 9
FRAME_THICKNESS = 2
GEN_ATTEMPTS = 400
GRID_H_RANGE = (28, 37)
GRID_W_RANGE = (28, 37)
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]


_NOTCH_BASES = {
    "elbow_left": np.array([[1, 1], [1, 0]], dtype=bool),
    "elbow_right": np.array([[1, 1], [0, 1]], dtype=bool),
    "tee": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "stair": np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
}
_SIDE_ROT = {"top": 0, "left": 1, "bottom": 2, "right": 3}
_PATTERN_BASES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    norm = _normalize_mask(mask)
    return (int(norm.shape[0]), int(norm.shape[1]), norm.astype(np.uint8).tobytes())


def _ring_mask(h: int, w: int, t: int) -> np.ndarray:
    ring = np.zeros((h, w), dtype=bool)
    ring[:t, :] = True
    ring[-t:, :] = True
    ring[:, :t] = True
    ring[:, -t:] = True
    return ring


def _extract_component_objects(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    comps: list[dict[str, Any]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, MAROON)):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _gap_side(missing: np.ndarray, t: int) -> str | None:
    coords = np.argwhere(missing)
    if coords.size == 0:
        return None
    h, w = missing.shape
    sides: list[str] = []
    if np.all(coords[:, 0] < t) and np.any(coords[:, 0] == 0):
        sides.append("top")
    if np.all(coords[:, 0] >= h - t) and np.any(coords[:, 0] == h - 1):
        sides.append("bottom")
    if np.all(coords[:, 1] < t) and np.any(coords[:, 1] == 0):
        sides.append("left")
    if np.all(coords[:, 1] >= w - t) and np.any(coords[:, 1] == w - 1):
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _classify_frame(grid: np.ndarray, bg: int, comp: dict[str, Any]) -> dict[str, Any] | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    mask = comp["mask"]
    h, w = mask.shape
    for t in (FRAME_THICKNESS, 1, 3):
        if h < 2 * t + 3 or w < 2 * t + 3:
            continue
        ring = _ring_mask(h, w, t)
        if np.any(mask & ~ring):
            continue
        missing = ring & ~mask
        missing_area = int(missing.sum())
        if missing_area < 3 or missing_area > 5:
            continue
        comps = _components(missing)
        if len(comps) != 1:
            continue
        side = _gap_side(missing, t)
        if side is None:
            continue
        inner = grid[r0 + t : r1 - t + 1, c0 + t : c1 - t + 1]
        if inner.size == 0:
            continue
        if not np.any(inner != bg):
            continue
        return {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "mask": mask,
            "missing": missing,
            "missing_key": _shape_key(missing),
            "side": side,
            "thickness": t,
        }
    return None


def _assign_markers(frames: list[dict[str, Any]], markers: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cand_lists: list[list[int]] = []
    cand_costs: list[dict[int, tuple[int, int]]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        costs: dict[int, tuple[int, int]] = {}
        for idx, (r, c) in enumerate(markers):
            if frame["side"] == "top" and r < r0 and c0 <= c <= c1:
                costs[idx] = (r0 - r, abs(c - center_c))
            elif frame["side"] == "bottom" and r > r1 and c0 <= c <= c1:
                costs[idx] = (r - r1, abs(c - center_c))
            elif frame["side"] == "left" and c < c0 and r0 <= r <= r1:
                costs[idx] = (c0 - c, abs(r - center_r))
            elif frame["side"] == "right" and c > c1 and r0 <= r <= r1:
                costs[idx] = (c - c1, abs(r - center_r))
        ordered = [idx for idx, _ in sorted(costs.items(), key=lambda item: item[1])]
        assert ordered, "every frame needs a marker on its gap side"
        cand_lists.append(ordered)
        cand_costs.append(costs)

    order = sorted(range(len(frames)), key=lambda i: len(cand_lists[i]))
    assign: list[int | None] = [None] * len(frames)
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        fi = order[pos]
        for marker_idx in cand_lists[fi]:
            if marker_idx in used:
                continue
            used.add(marker_idx)
            assign[fi] = marker_idx
            if backtrack(pos + 1):
                return True
            assign[fi] = None
            used.remove(marker_idx)
        return False

    assert backtrack(0), "markers must admit a one-to-one assignment"
    return [markers[int(idx)] for idx in assign]


def _extract_pattern(inner: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    mask = inner != bg
    box = _bbox_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    patch = inner[r0 : r1 + 1, c0 : c1 + 1].copy()
    return patch, patch != bg


def _paste_non_bg(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != bg
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock each notched frame with the matching key, then release the hidden inner pattern to the maroon anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _extract_component_objects(g, bg)

    frames: list[dict[str, Any]] = []
    frame_boxes: list[tuple[int, int, int, int]] = []
    for comp in comps:
        info = _classify_frame(g, bg, comp)
        if info is not None:
            frames.append(info)
            frame_boxes.append(info["bbox"])
    assert frames, "expected at least one notched frame"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)]
    assert len(markers) >= len(frames), "expected one maroon marker per frame"
    assigned_markers = _assign_markers(frames, markers)
    for frame, marker in zip(frames, assigned_markers):
        frame["marker"] = marker

    key_cells: dict[tuple[int, tuple[int, int, bytes]], list[tuple[int, int]]] = {}
    for comp in comps:
        bbox = comp["bbox"]
        if any(bbox == frame["bbox"] for frame in frames):
            continue
        if any(
            fr0 <= bbox[0] <= bbox[2] <= fr1 and fc0 <= bbox[1] <= bbox[3] <= fc1
            for fr0, fc0, fr1, fc1 in frame_boxes
        ):
            continue
        k = (int(comp["color"]), _shape_key(comp["mask"]))
        cells = [(int(r), int(c)) for r, c in comp["coords"]]
        assert k not in key_cells, "color+shape key pair must be unique"
        key_cells[k] = cells

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        t = int(frame["thickness"])
        marker_r, marker_c = frame["marker"]
        key = (color, frame["missing_key"])
        assert key in key_cells, "missing key component for one frame"

        for kr, kc in key_cells[key]:
            out[kr, kc] = bg

        full_ring = _ring_mask(r1 - r0 + 1, c1 - c0 + 1, t)
        local = out[r0 : r1 + 1, c0 : c1 + 1]
        local[full_ring] = color

        inner_in = g[r0 + t : r1 - t + 1, c0 + t : c1 - t + 1]
        pattern_patch, _ = _extract_pattern(inner_in, bg)
        out[r0 + t : r1 - t + 1, c0 + t : c1 - t + 1][inner_in != bg] = bg
        _paste_non_bg(out, pattern_patch, int(marker_r), int(marker_c), bg)

    return out


def _all_oriented_notches() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, bytes]] = set()
    for name, mask in _NOTCH_BASES.items():
        for side, rot in _SIDE_ROT.items():
            oriented = _normalize_mask(np.rot90(mask, rot))
            key = _shape_key(oriented)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "side": side, "mask": oriented, "key": key})
    return out


def _pattern_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for base in _PATTERN_BASES:
        for rot in range(4):
            cur = np.rot90(base, rot)
            for flipped in (cur, np.fliplr(cur)):
                norm = _normalize_mask(flipped)
                if not norm[0, 0]:
                    continue
                key = _shape_key(norm)
                if key in seen:
                    continue
                seen.add(key)
                variants.append(norm)
    return variants


_ORIENTED_NOTCHES = _all_oriented_notches()
_PATTERN_VARIANTS = _pattern_variants()


def _choose_pattern(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    choices = [mask for mask in _PATTERN_VARIANTS if mask.shape[0] <= max_h and mask.shape[1] <= max_w]
    if not choices:
        return None
    return rng.choice(choices).copy()


def _build_lock_patch(
    rng: random.Random,
    border_color: int,
    pattern_color: int,
    notch: dict[str, Any],
) -> dict[str, Any] | None:
    t = FRAME_THICKNESS
    side = str(notch["side"])
    notch_mask = np.array(notch["mask"], dtype=bool)
    nh, nw = notch_mask.shape

    for _ in range(60):
        outer_h = rng.randint(9, 12)
        outer_w = rng.randint(9, 12)
        safe_h = outer_h - 2 * t - 2
        safe_w = outer_w - 2 * t - 2
        if safe_h < 2 or safe_w < 2:
            continue
        if side in ("top", "bottom") and outer_w < 2 * t + nw + 1:
            continue
        if side in ("left", "right") and outer_h < 2 * t + nh + 1:
            continue

        pattern_mask = _choose_pattern(rng, safe_h, safe_w)
        if pattern_mask is None:
            continue
        ph, pw = pattern_mask.shape

        miss = np.zeros((outer_h, outer_w), dtype=bool)
        if side == "top":
            start = rng.randint(t, outer_w - t - nw)
            miss[:nh, start : start + nw] = notch_mask
            notch_axis = start
        elif side == "bottom":
            start = rng.randint(t, outer_w - t - nw)
            miss[outer_h - nh :, start : start + nw] = notch_mask
            notch_axis = start
        elif side == "left":
            start = rng.randint(t, outer_h - t - nh)
            miss[start : start + nh, :nw] = notch_mask
            notch_axis = start
        else:
            start = rng.randint(t, outer_h - t - nh)
            miss[start : start + nh, outer_w - nw :] = notch_mask
            notch_axis = start

        pattern_r = t + 1 + rng.randint(0, safe_h - ph)
        pattern_c = t + 1 + rng.randint(0, safe_w - pw)
        gap = rng.randint(1, 2)

        if side == "top":
            patch_h, patch_w = ph + gap + outer_h, outer_w
            lock_r, lock_c = ph + gap, 0
            dest_r = 0
            dest_c = min(max(notch_axis + rng.randint(-1, 1), 0), outer_w - pw)
        elif side == "bottom":
            patch_h, patch_w = outer_h + gap + ph, outer_w
            lock_r, lock_c = 0, 0
            dest_r = outer_h + gap
            dest_c = min(max(notch_axis + rng.randint(-1, 1), 0), outer_w - pw)
        elif side == "left":
            patch_h, patch_w = outer_h, pw + gap + outer_w
            lock_r, lock_c = 0, pw + gap
            dest_r = min(max(notch_axis + rng.randint(-1, 1), 0), outer_h - ph)
            dest_c = 0
        else:
            patch_h, patch_w = outer_h, outer_w + gap + pw
            lock_r, lock_c = 0, 0
            dest_r = min(max(notch_axis + rng.randint(-1, 1), 0), outer_h - ph)
            dest_c = outer_w + gap

        input_patch = np.full((patch_h, patch_w), BG, dtype=int)
        output_patch = np.full((patch_h, patch_w), BG, dtype=int)
        ring = _ring_mask(outer_h, outer_w, t)
        input_patch[lock_r : lock_r + outer_h, lock_c : lock_c + outer_w][ring & ~miss] = border_color
        output_patch[lock_r : lock_r + outer_h, lock_c : lock_c + outer_w][ring] = border_color
        input_patch[lock_r + pattern_r : lock_r + pattern_r + ph, lock_c + pattern_c : lock_c + pattern_c + pw][
            pattern_mask
        ] = pattern_color
        output_patch[dest_r : dest_r + ph, dest_c : dest_c + pw][pattern_mask] = pattern_color
        input_patch[dest_r, dest_c] = MAROON

        key_patch = np.full(notch_mask.shape, BG, dtype=int)
        key_patch[notch_mask] = border_color
        return {
            "input_patch": input_patch,
            "output_patch": output_patch,
            "key_patch": key_patch,
            "lock_shape": input_patch.shape,
        }
    return None


def _rect_fits(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occ.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not np.any(occ[r0:r1, c0:c1])


def _mark_rect(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    hh, ww = occ.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    occ[r0:r1, c0:c1] = True


def _place_patch(rng: random.Random, occ: np.ndarray, h: int, w: int, pad: int) -> tuple[int, int] | None:
    hh, ww = occ.shape
    if h > hh or w > ww:
        return None
    for _ in range(250):
        top = rng.randint(0, hh - h)
        left = rng.randint(0, ww - w)
        if _rect_fits(occ, top, left, h, w, pad):
            _mark_rect(occ, top, left, h, w, pad)
            return top, left
    return None


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    border_a, border_b = rng.sample(PALETTE, 2)
    pattern_colors = rng.sample([c for c in PALETTE if c not in {border_a, border_b}], 3)
    notch_x, notch_y = rng.sample(_ORIENTED_NOTCHES, 2)

    lock_specs = [
        (border_a, pattern_colors[0], notch_x),
        (border_a, pattern_colors[1], notch_y),
        (border_b, pattern_colors[2], notch_y),
    ]
    rng.shuffle(lock_specs)

    built: list[dict[str, Any]] = []
    for border_color, pattern_color, notch in lock_specs:
        spec = _build_lock_patch(rng, border_color, pattern_color, notch)
        if spec is None:
            return None
        built.append(spec)

    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(len(built)), key=lambda i: built[i]["input_patch"].size, reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(built)
    for idx in order:
        patch = built[idx]["input_patch"]
        pos = _place_patch(rng, occ, patch.shape[0], patch.shape[1], pad=1)
        if pos is None:
            return None
        positions[idx] = pos
        _draw_patch(inp, built[idx]["input_patch"], pos[0], pos[1])
        _draw_patch(out, built[idx]["output_patch"], pos[0], pos[1])

    for idx, spec in enumerate(built):
        key_patch = spec["key_patch"]
        pos = _place_patch(rng, occ, key_patch.shape[0], key_patch.shape[1], pad=1)
        if pos is None:
            return None
        _draw_patch(inp, key_patch, pos[0], pos[1])

    if np.count_nonzero(inp == MAROON) != 3:
        return None
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    """
    Generate diverse lock-and-key ARC inputs with unique color+shape pairing.
    """
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key input")
