from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 7)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BAR_COUNT_CHOICES = (5,)
DISTRACTOR_COUNT_CHOICES = (2, 3)
FRAME_SIDE_RANGE = (4, 8)
CANVAS_SIDE_RANGE = (15, 18)
MAX_GENERATION_ATTEMPTS = 400
MAX_SHAPE_ATTEMPTS = 120
MAX_LAYOUT_ATTEMPTS = 120
BALANCED_FRAME_CHOICES = tuple(
    (w, h)
    for h in range(FRAME_SIDE_RANGE[0], FRAME_SIDE_RANGE[1] + 1)
    for w in range(FRAME_SIDE_RANGE[0], h + 1)
    if abs(w - h) <= 1
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []
    nbrs = DIR4 if connectivity == 4 else (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _crop_component_mask(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = comp["coords"]
    crop[coords[:, 0] - r0, coords[:, 1] - c0] = comp["color"]
    return crop != bg


def _is_axis_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)))


def _bar_descriptor(mask: np.ndarray) -> tuple[str, int] | None:
    h, w = mask.shape
    if h == 1 and bool(mask.all()):
        return "h", int(w)
    if w == 1 and bool(mask.all()):
        return "v", int(h)
    return None


def _balanced_frame_dims(border_pixels: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for h in range(3, border_pixels + 1):
        for w in range(3, border_pixels + 1):
            if 2 * (h + w) - 4 == border_pixels:
                candidates.append((w, h))
    assert candidates, "bar pixels must match at least one single-cell-thick rectangular frame"
    return min(candidates, key=lambda wh: (abs(wh[0] - wh[1]), wh[0], wh[1]))


def _border_coords(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]


def _placements_for_bar(
    orient: str,
    length: int,
    h: int,
    w: int,
    index_by_coord: dict[tuple[int, int], int],
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    placements: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    if orient == "h":
        for row in (0, h - 1):
            for c0 in range(w - length + 1):
                cells = tuple((row, c) for c in range(c0, c0 + length))
                mask = 0
                for cell in cells:
                    mask |= 1 << index_by_coord[cell]
                placements.append((mask, cells))
    else:
        for col in (0, w - 1):
            for r0 in range(h - length + 1):
                cells = tuple((r, col) for r in range(r0, r0 + length))
                mask = 0
                for cell in cells:
                    mask |= 1 << index_by_coord[cell]
                placements.append((mask, cells))
    return placements


def _assemble_unique_frame(
    bars: list[dict],
    h: int,
    w: int,
    *,
    bg: int,
) -> np.ndarray:
    border = _border_coords(h, w)
    index_by_coord = {coord: i for i, coord in enumerate(border)}
    full_mask = (1 << len(border)) - 1
    placements_by_bar: list[list[tuple[int, tuple[tuple[int, int], ...]]]] = []

    for bar in bars:
        placements = _placements_for_bar(bar["orient"], bar["length"], h, w, index_by_coord)
        assert placements, "each bar must fit the target frame somewhere"
        placements_by_bar.append(placements)

    order = sorted(
        range(len(bars)),
        key=lambda i: (
            len(placements_by_bar[i]),
            -bars[i]["length"],
            bars[i].get("bbox", (0, 0, 0, 0))[0],
            bars[i].get("bbox", (0, 0, 0, 0))[1],
        ),
    )
    unique_outputs: dict[tuple[bytes, tuple[int, int]], np.ndarray] = {}
    total_length = sum(bar["length"] for bar in bars)
    assert total_length == len(border)

    def search(
        step: int,
        used_mask: int,
        used_cells: int,
        chosen: list[tuple[int, tuple[tuple[int, int], ...]]],
    ) -> None:
        if len(unique_outputs) > 1:
            return
        if step == len(order):
            if used_mask != full_mask:
                return
            out = np.full((h, w), bg, dtype=int)
            for idx, cells in chosen:
                for r, c in cells:
                    out[r, c] = int(bars[idx]["color"])
            key = (out.tobytes(), tuple(map(int, out.shape)))
            unique_outputs[key] = out
            return

        remaining = sum(bars[order[i]]["length"] for i in range(step, len(order)))
        if len(border) - used_cells != remaining:
            return

        idx = order[step]
        for placement_mask, cells in placements_by_bar[idx]:
            if used_mask & placement_mask:
                continue
            search(
                step + 1,
                used_mask | placement_mask,
                used_cells + bars[idx]["length"],
                chosen + [(idx, cells)],
            )

    search(0, 0, 0, [])
    assert unique_outputs, "bars must tile the inferred frame"
    assert len(unique_outputs) == 1, "frame tiling must be uniquely determined"
    return next(iter(unique_outputs.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the symmetric bars and translate them into the uniquely tiled most-balanced rectangular frame.

    Input structure:
    - A black background contains several 4-connected monochromatic foreground objects in colors {1,2,3,7}.
    - Some objects are straight one-pixel-wide horizontal or vertical bars, while the distractors are wider irregular shapes.
    - Each object's role is determined from the foreground mask inside its tight bounding box, not from its absolute position.

    Rule:
    - Split the foreground into monochromatic 4-connected objects and crop each one to its tight bbox.
    - Keep only objects whose cropped mask is mirror-symmetric across the vertical axis or the horizontal axis; every kept object must therefore be a straight bar.
    - Let P be the total number of kept pixels. Because the output is a one-cell-thick rectangular border, choose the integer frame size (W,H) with W>=3, H>=3, 2*(W+H)-4 = P, minimal |W-H|, and W<=H on ties.
    - Translate the kept bars without rotation or reflection so they cover the border of that W x H frame exactly once; fill the interior with black.

    Key insight:
    - Symmetry does not select the final visible object. It selects which input pieces are valid building materials, and only their pixel count leads to the correct balanced frame size and border tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg=bg, connectivity=4)
    bars: list[dict] = []
    for comp in components:
        mask = _crop_component_mask(g, comp, bg)
        if not _is_axis_symmetric(mask):
            continue
        desc = _bar_descriptor(mask)
        assert desc is not None, "symmetric survivors must be straight bars"
        orient, length = desc
        bars.append(
            {
                "color": int(comp["color"]),
                "orient": orient,
                "length": length,
                "bbox": comp["bbox"],
            }
        )

    assert bars, "expected at least one symmetric bar"
    bars.sort(key=lambda bar: (bar["bbox"][0], bar["bbox"][1], bar["orient"], -bar["length"]))
    border_pixels = sum(bar["length"] for bar in bars)
    w, h = _balanced_frame_dims(border_pixels)
    return _assemble_unique_frame(bars, h, w, bg=bg)


def _first_uncovered_bit(mask: int, total_bits: int) -> int:
    for bit in range(total_bits):
        if not (mask >> bit) & 1:
            return bit
    return -1


def _random_tiling(h: int, w: int, target_count: int) -> list[dict] | None:
    border = _border_coords(h, w)
    index_by_coord = {coord: i for i, coord in enumerate(border)}
    total_bits = len(border)
    full_mask = (1 << total_bits) - 1
    segments: list[dict] = []

    for row, side in ((0, "top"), (h - 1, "bottom")):
        for length in range(2, min(7, w) + 1):
            for c0 in range(w - length + 1):
                cells = tuple((row, c) for c in range(c0, c0 + length))
                mask = 0
                for cell in cells:
                    mask |= 1 << index_by_coord[cell]
                segments.append(
                    {"orient": "h", "length": length, "side": side, "cells": cells, "mask": mask}
                )

    for col, side in ((0, "left"), (w - 1, "right")):
        for length in range(2, min(7, h) + 1):
            for r0 in range(h - length + 1):
                cells = tuple((r, col) for r in range(r0, r0 + length))
                mask = 0
                for cell in cells:
                    mask |= 1 << index_by_coord[cell]
                segments.append(
                    {"orient": "v", "length": length, "side": side, "cells": cells, "mask": mask}
                )

    by_bit: dict[int, list[int]] = {i: [] for i in range(total_bits)}
    for i, seg in enumerate(segments):
        for r, c in seg["cells"]:
            by_bit[index_by_coord[(r, c)]].append(i)

    best: list[dict] | None = None

    def is_good(solution: list[dict]) -> bool:
        if len({seg["orient"] for seg in solution}) < 2:
            return False
        if len({seg["length"] for seg in solution}) < 2:
            return False
        side_counts = {side: 0 for side in ("top", "bottom", "left", "right")}
        for seg in solution:
            side_counts[seg["side"]] += 1
        if not any(count >= 2 for count in side_counts.values()) and target_count == 4:
            return False
        return True

    def search(used_mask: int, used_cells: int, chosen: list[dict]) -> bool:
        nonlocal best
        if used_mask == full_mask:
            if len(chosen) == target_count and is_good(chosen):
                best = chosen[:]
                return True
            return False
        if len(chosen) >= target_count:
            return False
        remaining_slots = target_count - len(chosen)
        uncovered = total_bits - used_cells
        if uncovered < 2 * remaining_slots or uncovered > 7 * remaining_slots:
            return False

        bit = _first_uncovered_bit(used_mask, total_bits)
        candidate_ids = by_bit[bit][:]
        random.shuffle(candidate_ids)
        for seg_id in candidate_ids:
            seg = segments[seg_id]
            if used_mask & seg["mask"]:
                continue
            if search(used_mask | seg["mask"], used_cells + seg["length"], chosen + [seg]):
                return True
        return False

    search(0, 0, [])
    return best


def _bar_patch(color: int, orient: str, length: int) -> np.ndarray:
    shape = (1, length) if orient == "h" else (length, 1)
    return np.full(shape, int(color), dtype=int)


def _shape_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    pts = {tuple(map(int, rc)) for rc in coords}
    q = deque([tuple(map(int, coords[0]))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            nxt = (nr, nc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(pts)


def _random_asymmetric_patch(color: int) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(6, h * w - 1))
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < area:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
        if int(mask.sum()) != area:
            continue
        if not _shape_connected(mask):
            continue
        if h == 1 or w == 1:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if _is_axis_symmetric(mask):
            continue
        patch = np.full((h, w), BG, dtype=int)
        patch[mask] = int(color)
        return patch
    return None


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    r1 = max(0, r0 - padding)
    c1 = max(0, c0 - padding)
    r2 = min(H, r0 + h + padding)
    c2 = min(W, c0 + w + padding)
    return not bool(np.any(canvas[r1:r2, c1:c2] != BG))


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _render_solution(h: int, w: int, segments: list[dict], colors: list[int]) -> tuple[np.ndarray, list[dict]]:
    out = np.full((h, w), BG, dtype=int)
    bars: list[dict] = []
    for seg, color in zip(segments, colors):
        for r, c in seg["cells"]:
            out[r, c] = int(color)
        bars.append({"color": int(color), "orient": seg["orient"], "length": seg["length"]})
    return out, bars


def _split_two(total: int) -> tuple[int, int]:
    cut = random.randint(2, total - 2)
    return cut, total - cut


def _h_segment(row: int, c0: int, length: int, side: str) -> dict:
    return {
        "orient": "h",
        "length": int(length),
        "side": side,
        "cells": tuple((int(row), c) for c in range(int(c0), int(c0 + length))),
    }


def _v_segment(col: int, r0: int, length: int, side: str) -> dict:
    return {
        "orient": "v",
        "length": int(length),
        "side": side,
        "cells": tuple((r, int(col)) for r in range(int(r0), int(r0 + length))),
    }


def _structured_segments(h: int, w: int) -> tuple[list[dict], list[int]] | None:
    families = ["horizontal_corners", "vertical_corners"]
    random.shuffle(families)
    color_h, color_v = random.sample(PALETTE, 2)

    for family in families:
        if family == "horizontal_corners":
            if w < 4 or h < 4:
                continue
            a, b = _split_two(w)
            segments = [
                _h_segment(0, 0, a, "top"),
                _h_segment(0, a, b, "top"),
                _h_segment(h - 1, 0, w, "bottom"),
                _v_segment(0, 1, h - 2, "left"),
                _v_segment(w - 1, 1, h - 2, "right"),
            ]
            colors = [color_h, color_h, color_h, color_v, color_v]
            return segments, colors

        if h < 4 or w < 4:
            continue
        a, b = _split_two(h)
        segments = [
            _v_segment(0, 0, a, "left"),
            _v_segment(0, a, b, "left"),
            _v_segment(w - 1, 0, h, "right"),
            _h_segment(0, 1, w - 2, "top"),
            _h_segment(h - 1, 1, w - 2, "bottom"),
        ]
        colors = [color_v, color_v, color_v, color_h, color_h]
        return segments, colors

    return None


def _balanced_frame_dims_if_possible(total_pixels: int) -> tuple[int, int] | None:
    candidates = []
    for h in range(3, total_pixels + 1):
        for w in range(3, total_pixels + 1):
            if 2 * (h + w) - 4 == total_pixels:
                candidates.append((w, h))
    if not candidates:
        return None
    return min(candidates, key=lambda wh: (abs(wh[0] - wh[1]), wh[0], wh[1]))


def _validate_instance(inp: np.ndarray, expected: np.ndarray, w: int, h: int) -> None:
    comps = _find_components(inp, bg=BG, connectivity=4)
    assert 6 <= len(comps) <= 8
    symmetric = 0
    asymmetric = 0
    survivor_pixels = 0
    for comp in comps:
        mask = _crop_component_mask(inp, comp, BG)
        if _is_axis_symmetric(mask):
            desc = _bar_descriptor(mask)
            assert desc is not None
            symmetric += 1
            survivor_pixels += int(comp["area"])
        else:
            asymmetric += 1
            assert mask.shape[0] > 1 and mask.shape[1] > 1
    assert symmetric in BAR_COUNT_CHOICES
    assert asymmetric in DISTRACTOR_COUNT_CHOICES
    assert expected.shape == (h, w)
    assert survivor_pixels == int(np.count_nonzero(expected))
    wrong_dims = _balanced_frame_dims_if_possible(int(np.count_nonzero(inp)))
    assert wrong_dims is None or wrong_dims != (w, h)
    solved = main(inp)
    assert np.array_equal(solved, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        bar_count = BAR_COUNT_CHOICES[0]
        w, h = random.choice(BALANCED_FRAME_CHOICES)
        border_pixels = 2 * (h + w) - 4
        if not (2 * bar_count <= border_pixels <= 7 * bar_count):
            continue

        structured = _structured_segments(h, w)
        if structured is None:
            continue
        segments, colors = structured
        expected, bars = _render_solution(h, w, segments, colors)
        try:
            solved = _assemble_unique_frame(bars, h, w, bg=BG)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue

        distractor_count = random.choice(DISTRACTOR_COUNT_CHOICES)
        distractors: list[np.ndarray] = []
        ok = True
        for _j in range(distractor_count):
            patch = _random_asymmetric_patch(random.choice(PALETTE))
            if patch is None:
                ok = False
                break
            distractors.append(patch)
        if not ok:
            continue

        objects = [_bar_patch(bar["color"], bar["orient"], bar["length"]) for bar in bars] + distractors
        random.shuffle(objects)

        for _layout in range(MAX_LAYOUT_ATTEMPTS):
            H = random.randint(*CANVAS_SIDE_RANGE)
            W = random.randint(*CANVAS_SIDE_RANGE)
            canvas = np.zeros((H, W), dtype=int)
            placed = True
            for patch in sorted(objects, key=lambda arr: (arr.shape[0] * arr.shape[1], arr.shape[0] + arr.shape[1]), reverse=True):
                positions = [
                    (r0, c0)
                    for r0 in range(H - patch.shape[0] + 1)
                    for c0 in range(W - patch.shape[1] + 1)
                    if _can_place(canvas, patch, r0, c0, padding=1)
                ]
                if not positions:
                    placed = False
                    break
                r0, c0 = random.choice(positions)
                _paste(canvas, patch, r0, c0)
            if not placed:
                continue
            _validate_instance(canvas, expected, w, h)
            return canvas

    raise RuntimeError("failed to generate a valid ARC instance")
