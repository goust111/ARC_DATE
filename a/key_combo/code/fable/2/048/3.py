from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste


BG = 0
FRAME_COLOR = 5
COMB_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CANVAS_SIZE = 30
PLACEMENT_MARGIN = 1
PLACEMENT_BORDER = 1
MAX_GEN_ATTEMPTS = 400

SIGNATURE_LIBRARY = {
    3: [
        (0, 2, 4),
        (0, 1, 4),
        (0, 2, 5),
        (0, 1, 5),
        (0, 3, 5),
        (0, 2, 6),
        (0, 3, 6),
        (0, 4, 6),
    ],
    4: [
        (0, 1, 3, 5),
        (0, 2, 4, 6),
        (0, 1, 4, 6),
        (0, 2, 5, 7),
        (0, 1, 3, 6),
        (0, 2, 4, 7),
        (0, 1, 4, 7),
        (0, 3, 5, 7),
    ],
}


def _relative_signature(cols: list[int] | tuple[int, ...] | np.ndarray) -> tuple[int, ...]:
    seq = [int(c) for c in cols]
    return tuple(c - seq[0] for c in seq)


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = comp["coords"] - np.array([[r0, c0]])
    mask[local[:, 0], local[:, 1]] = True
    return mask


def _parse_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 5:
        return None

    mask = _component_mask(comp)
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]) or not np.all(mask[-1, :]):
        return None
    if not mask[0, 0] or not mask[0, -1]:
        return None
    if not np.all(~mask[1:-1, 1:-1]):
        return None

    top_inside = mask[0, 1:-1]
    if np.all(top_inside) or not np.any(top_inside):
        return None

    gap_local_cols = [idx + 1 for idx, filled in enumerate(top_inside) if not filled]
    if len(gap_local_cols) < 2:
        return None

    return {
        "kind": "frame",
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": (r0, c0, r1, c1),
        "shape": (h, w),
        "gap_local_cols": gap_local_cols,
        "gap_abs_cols": [c0 + col for col in gap_local_cols],
        "signature": _relative_signature(gap_local_cols),
    }


def _parse_comb(comp: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != 2 or w < 4:
        return None

    mask = _component_mask(comp)
    if np.all(mask[0]) and 2 <= int(mask[1].sum()) < w:
        teeth_row = 1
        flip_needed = False
    elif np.all(mask[1]) and 2 <= int(mask[0].sum()) < w:
        teeth_row = 0
        flip_needed = True
    else:
        return None

    teeth_local_cols = [int(c) for c in np.flatnonzero(mask[teeth_row])]
    patch = np.full((2, w), BG, dtype=int)
    patch[mask] = int(comp["color"])
    patch_down = np.flipud(patch) if flip_needed else patch.copy()

    return {
        "kind": "comb",
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": (r0, c0, r1, c1),
        "shape": (2, w),
        "flip_needed": flip_needed,
        "teeth_local_cols": teeth_local_cols,
        "signature": _relative_signature(teeth_local_cols),
        "patch_down": patch_down,
    }


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    fg = grid[grid != bg]
    assert fg.size > 0, "expected foreground"
    colors, counts = np.unique(fg, return_counts=True)
    frame_color = int(colors[np.argmax(counts)])

    h, w = grid.shape
    frames: list[dict] = []
    for r0 in range(h - 3):
        for c0 in range(w - 4):
            if int(grid[r0, c0]) != frame_color:
                continue
            for r1 in range(r0 + 3, h):
                if int(grid[r1, c0]) != frame_color:
                    continue
                if not np.all(grid[r0 : r1 + 1, c0] == frame_color):
                    continue
                for c1 in range(c0 + 4, w):
                    if int(grid[r0, c1]) != frame_color or int(grid[r1, c1]) != frame_color:
                        continue
                    if not np.all(grid[r0 : r1 + 1, c1] == frame_color):
                        continue
                    if not np.all(grid[r1, c0 : c1 + 1] == frame_color):
                        continue

                    top_inside = grid[r0, c0 + 1 : c1]
                    if top_inside.size < 3:
                        continue
                    if np.any((top_inside != frame_color) & (top_inside != bg)):
                        continue
                    gap_local_cols = [idx + 1 for idx, value in enumerate(top_inside) if int(value) == bg]
                    if len(gap_local_cols) < 2 or len(gap_local_cols) == top_inside.size:
                        continue
                    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                        continue

                    frames.append(
                        {
                            "kind": "frame",
                            "color": frame_color,
                            "bbox": (r0, c0, r1, c1),
                            "shape": (r1 - r0 + 1, c1 - c0 + 1),
                            "gap_local_cols": gap_local_cols,
                            "gap_abs_cols": [c0 + col for col in gap_local_cols],
                            "signature": _relative_signature(gap_local_cols),
                        }
                    )

    seen_bboxes: set[tuple[int, int, int, int]] = set()
    unique_frames: list[dict] = []
    for frame in frames:
        if frame["bbox"] in seen_bboxes:
            continue
        seen_bboxes.add(frame["bbox"])
        unique_frames.append(frame)

    return frame_color, unique_frames


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = get_bg(grid)
    frame_color, frames = _find_frames(grid, bg)
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[color for color in np.unique(grid) if int(color) not in {bg, frame_color}],
        return_format="dict",
    )

    combs: list[dict] = []
    for idx, frame in enumerate(frames):
        frame["id"] = idx
    for idx, comp in enumerate(components, start=len(frames)):
        comb = _parse_comb(comp)
        if comb is not None:
            comb["id"] = idx
            combs.append(comb)
            continue

        raise AssertionError("unexpected non-frame foreground component")

    return bg, frames, combs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each colored comb into the grey frame whose top-edge gap spacing it matches, then fill the newly closed interior.

    Input structure:
    - The grid has a black background, several grey rectangular frames, and several separate monochrome combs.
    - Each frame is a one-cell-thick rectangle whose top edge is missing several single cells.
    - Each comb is a two-row horizontal bar with several one-cell teeth on exactly one side; some combs point upward and need vertical flipping.

    Rule:
    - Read each frame by the relative spacing pattern of its missing top-edge cells, and read each comb by the relative spacing of its teeth columns.
    - Match every comb to the unique frame with the same spacing pattern; if a comb's teeth point upward, flip it vertically first.
    - Move the comb so its teeth fill the frame's missing top cells and its bar sits one row above the frame, then erase the comb's original position and fill that frame interior with the comb color.
    - Leave any unmatched frame unchanged.

    Key insight:
    - The decisive clue is not frame width or tooth count, but the normalized left-to-right spacing pattern of the gaps and teeth; only after those gaps are plugged does the frame become a closed region eligible for filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, frames, combs = _analyze_scene(g)
    assert frames and combs, "expected both frames and combs"

    frame_by_signature: dict[tuple[int, ...], dict] = {}
    for frame in frames:
        sig = frame["signature"]
        assert sig not in frame_by_signature, "frame gap signatures must be unique"
        frame_by_signature[sig] = frame

    seen_comb_signatures: set[tuple[int, ...]] = set()
    out = g.copy()
    filled_frames: set[int] = set()

    for comb in combs:
        sig = comb["signature"]
        assert sig not in seen_comb_signatures, "comb signatures must be unique"
        seen_comb_signatures.add(sig)

        frame = frame_by_signature.get(sig)
        if frame is None:
            continue

        assert frame["id"] not in filled_frames, "multiple combs cannot target the same frame"
        filled_frames.add(frame["id"])

        gap_abs_cols = frame["gap_abs_cols"]
        teeth_local_cols = comb["teeth_local_cols"]
        target_r0 = int(frame["bbox"][0]) - 1
        target_c0 = gap_abs_cols[0] - teeth_local_cols[0]

        aligned_cols = [target_c0 + col for col in teeth_local_cols]
        assert aligned_cols == gap_abs_cols, "teeth must land exactly in the frame gaps"

        out[comb["coords"][:, 0], comb["coords"][:, 1]] = bg
        paste(out, comb["patch_down"], target_r0, target_c0, transparent=bg)

        r0, c0, r1, c1 = map(int, frame["bbox"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(comb["color"])

    return out


def _make_frame_patch(height: int, inner_width: int, start: int, signature: tuple[int, ...]) -> tuple[np.ndarray, list[int]]:
    width = inner_width + 2
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    gap_local_cols = [1 + start + offset for offset in signature]
    patch[0, gap_local_cols] = BG
    return patch, gap_local_cols


def _make_comb_patch(
    bar_width: int,
    start: int,
    signature: tuple[int, ...],
    color: int,
    orientation: str,
) -> tuple[np.ndarray, list[int]]:
    patch = np.full((2, bar_width), BG, dtype=int)
    tooth_cols = [start + offset for offset in signature]
    if orientation == "down":
        patch[0, :] = color
        patch[1, tooth_cols] = color
    else:
        patch[1, :] = color
        patch[0, tooth_cols] = color
    return patch, tooth_cols


def _find_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    limit_r = occupied.shape[0] - h - PLACEMENT_BORDER
    limit_c = occupied.shape[1] - w - PLACEMENT_BORDER
    for r0 in range(PLACEMENT_BORDER, limit_r + 1):
        for c0 in range(PLACEMENT_BORDER, limit_c + 1):
            rr0 = max(0, r0 - PLACEMENT_MARGIN)
            cc0 = max(0, c0 - PLACEMENT_MARGIN)
            rr1 = min(occupied.shape[0], r0 + h + PLACEMENT_MARGIN)
            cc1 = min(occupied.shape[1], c0 + w + PLACEMENT_MARGIN)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _place_reservations(
    items: list[dict],
    occupied: np.ndarray | None = None,
) -> tuple[dict[int, tuple[int, int]], np.ndarray] | tuple[None, None]:
    if occupied is None:
        occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    else:
        occupied = occupied.copy()
    placements: dict[int, tuple[int, int]] = {}

    ordered = sorted(
        items,
        key=lambda item: item["reserve_h"] * item["reserve_w"],
        reverse=True,
    )
    for item in ordered:
        candidates = _find_positions(occupied, item["reserve_h"], item["reserve_w"])
        if not candidates:
            return None, None

        r0, c0 = candidates[0]
        placements[item["id"]] = (r0, c0)
        rr0 = max(0, r0 - PLACEMENT_MARGIN)
        cc0 = max(0, c0 - PLACEMENT_MARGIN)
        rr1 = min(CANVAS_SIZE, r0 + item["reserve_h"] + PLACEMENT_MARGIN)
        cc1 = min(CANVAS_SIZE, c0 + item["reserve_w"] + PLACEMENT_MARGIN)
        occupied[rr0:rr1, cc0:cc1] = True

    return placements, occupied


def _interior_reachable_from_outside(frame_patch: np.ndarray) -> bool:
    padded = np.pad(frame_patch == BG, 1, constant_values=True)
    visited = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    visited[0, 0] = True

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1]:
                if padded[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))

    interior = visited[2:-2, 2:-2]
    return bool(interior.any())


def _build_expected(scene: dict) -> np.ndarray:
    out = scene["input"].copy()
    for comb in scene["combs"]:
        coords = comb["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for frame in scene["frames"]:
        if frame["matched_color"] is None:
            continue
        paste(
            out,
            frame["matched_patch_down"],
            frame["output_bar_r0"],
            frame["output_bar_c0"],
            transparent=BG,
        )
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = frame["matched_color"]

    return out


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    union_mask = (inp != BG) | (out != BG)
    union_bbox = bbox(union_mask)
    assert union_bbox is not None
    r0, c0, r1, c1 = map(int, union_bbox)

    top_room = r0
    left_room = c0
    bottom_room = inp.shape[0] - 1 - r1
    right_room = inp.shape[1] - 1 - c1

    pad_top = random.randint(1, min(3, top_room)) if top_room >= 1 else 0
    pad_left = random.randint(1, min(3, left_room)) if left_room >= 1 else 0
    pad_bottom = random.randint(1, min(3, bottom_room)) if bottom_room >= 1 else 0
    pad_right = random.randint(1, min(3, right_room)) if right_room >= 1 else 0

    crop = (
        r0 - pad_top,
        c0 - pad_left,
        r1 + pad_bottom,
        c1 + pad_right,
    )
    return crop_bbox(inp, crop), crop_bbox(out, crop)


def _validate_scene(scene: dict) -> bool:
    frame_signatures = [frame["signature"] for frame in scene["frames"]]
    comb_signatures = [comb["signature"] for comb in scene["combs"]]
    if len(frame_signatures) != len(set(frame_signatures)):
        return False
    if len(comb_signatures) != len(set(comb_signatures)):
        return False
    if set(comb_signatures) != {frame["signature"] for frame in scene["frames"] if frame["matched_color"] is not None}:
        return False
    if not any(comb["flip_needed"] for comb in scene["combs"]):
        return False
    if len(scene["frames"]) < 2 or len(scene["combs"]) < 2:
        return False
    if len({len(sig) for sig in frame_signatures}) != 1:
        return False
    if len(set(frame_signatures)) == 1:
        return False
    if not any(frame["frame_start"] != frame["comb_start"] for frame in scene["frames"] if frame["comb_start"] is not None):
        return False

    for frame in scene["frames"]:
        if not _interior_reachable_from_outside(frame["patch"]):
            return False

        closed = frame["patch"].copy()
        closed[0, frame["gap_local_cols"]] = 9
        if _interior_reachable_from_outside(closed):
            return False

        if frame["matched_color"] is None:
            continue

        aligned = [frame["output_bar_c0"] + col for col in frame["comb_teeth_cols"]]
        if aligned != frame["gap_abs_cols"]:
            return False

    expected = _build_expected(scene)
    cropped_inp, cropped_expected = _crop_pair(scene["input"], expected)
    if get_bg(cropped_inp) != BG:
        return False
    if cropped_inp.ndim != 2 or cropped_expected.ndim != 2:
        return False
    if max(cropped_inp.shape) > 30 or max(cropped_expected.shape) > 30:
        return False
    if np.array_equal(cropped_inp, cropped_expected):
        return False
    try:
        actual = main(cropped_inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, cropped_expected):
        return False

    scene["cropped_input"] = cropped_inp
    return True


def _generate_scene() -> dict | None:
    tooth_count = random.choice((3, 4))
    matched_count = random.randint(2, 3)
    extra_frame = 1 if random.random() < 0.65 else 0
    total_frames = matched_count + extra_frame

    signatures = random.sample(SIGNATURE_LIBRARY[tooth_count], total_frames)
    random.shuffle(signatures)
    matched_signatures = signatures[:matched_count]
    unmatched_signatures = signatures[matched_count:]

    max_span = max(max(signature) for signature in signatures)
    frame_inner_width = random.randint(max_span + 2, min(10, max_span + 4))
    bar_width = random.randint(max_span + 2, min(10, max_span + 4))

    colors = random.sample(COMB_COLORS, matched_count)
    forced_up_idx = random.randrange(matched_count)

    frame_by_signature: dict[tuple[int, ...], dict] = {}
    frames: list[dict] = []
    for idx, signature in enumerate(signatures):
        start_limit = frame_inner_width - (max(signature) + 1)
        frame_start = random.randint(0, start_limit)
        height = random.randint(4, 7)
        patch, gap_local_cols = _make_frame_patch(height, frame_inner_width, frame_start, signature)

        frame = {
            "id": idx,
            "signature": signature,
            "patch": patch,
            "gap_local_cols": gap_local_cols,
            "frame_start": frame_start,
            "comb_start": None,
            "matched_color": None,
            "matched_patch_down": None,
            "comb_teeth_cols": None,
            "reserve_h": patch.shape[0],
            "reserve_w": patch.shape[1],
            "frame_offset_r": 0,
            "frame_offset_c": 0,
            "output_bar_r0": None,
            "output_bar_c0": None,
        }
        frame_by_signature[signature] = frame
        frames.append(frame)

    combs: list[dict] = []
    for idx, (signature, color) in enumerate(zip(matched_signatures, colors)):
        start_limit = bar_width - (max(signature) + 1)
        comb_start = random.randint(0, start_limit)
        orientation = "up" if idx == forced_up_idx else random.choice(("up", "down"))
        patch, tooth_cols = _make_comb_patch(bar_width, comb_start, signature, color, orientation)
        combs.append(
            {
                "id": idx,
                "signature": signature,
                "patch": patch,
                "patch_down": np.flipud(patch) if orientation == "up" else patch.copy(),
                "teeth_cols": tooth_cols,
                "color": color,
                "flip_needed": orientation == "up",
                "reserve_h": patch.shape[0],
                "reserve_w": patch.shape[1],
            }
        )
        frame_by_signature[signature]["comb_start"] = comb_start
        frame_by_signature[signature]["matched_color"] = color
        frame_by_signature[signature]["matched_patch_down"] = np.flipud(patch) if orientation == "up" else patch.copy()
        frame_by_signature[signature]["comb_teeth_cols"] = tooth_cols

    if not any(frame["frame_start"] != frame["comb_start"] for frame in frames if frame["comb_start"] is not None):
        adjustable = [frame for frame in frames if frame["comb_start"] is not None]
        random.shuffle(adjustable)
        changed = False
        for frame in adjustable:
            limit = bar_width - (max(frame["signature"]) + 1)
            if limit > 0:
                frame["comb_start"] = (frame["comb_start"] + 1) % (limit + 1)
                frame["comb_teeth_cols"] = [frame["comb_start"] + offset for offset in frame["signature"]]
                for comb in combs:
                    if comb["signature"] == frame["signature"]:
                        orientation = "up" if comb["flip_needed"] else "down"
                        comb["patch"], comb["teeth_cols"] = _make_comb_patch(
                            bar_width,
                            frame["comb_start"],
                            frame["signature"],
                            comb["color"],
                            orientation,
                        )
                        comb["patch_down"] = np.flipud(comb["patch"]) if comb["flip_needed"] else comb["patch"].copy()
                        frame["matched_patch_down"] = comb["patch_down"]
                        frame["comb_teeth_cols"] = comb["teeth_cols"]
                        break
                changed = True
                break
        if not changed:
            return None

    for frame in frames:
        if frame["comb_start"] is None:
            continue
        bar_left = 1 + frame["frame_start"] - frame["comb_start"]
        left_expand = max(0, -bar_left)
        right_expand = max(0, bar_left + bar_width - frame["patch"].shape[1])
        frame["reserve_h"] = frame["patch"].shape[0] + 1
        frame["reserve_w"] = frame["patch"].shape[1] + left_expand + right_expand
        frame["frame_offset_r"] = 1
        frame["frame_offset_c"] = left_expand

    frame_placements, occupied = _place_reservations(frames)
    if frame_placements is None:
        return None

    comb_placements, _occupied = _place_reservations(combs, occupied=occupied)
    if comb_placements is None:
        return None

    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    for frame in frames:
        reserve_r0, reserve_c0 = frame_placements[frame["id"]]
        frame_r0 = reserve_r0 + frame["frame_offset_r"]
        frame_c0 = reserve_c0 + frame["frame_offset_c"]
        paste(grid, frame["patch"], frame_r0, frame_c0)

        r0, c0 = frame_r0, frame_c0
        r1 = frame_r0 + frame["patch"].shape[0] - 1
        c1 = frame_c0 + frame["patch"].shape[1] - 1
        frame["bbox"] = (r0, c0, r1, c1)
        frame["gap_abs_cols"] = [c0 + col for col in frame["gap_local_cols"]]

        if frame["matched_color"] is not None:
            frame["output_bar_r0"] = r0 - 1
            frame["output_bar_c0"] = frame["gap_abs_cols"][0] - frame["comb_teeth_cols"][0]

    for comb in combs:
        comb_r0, comb_c0 = comb_placements[comb["id"]]
        paste(grid, comb["patch"], comb_r0, comb_c0, transparent=BG)
        rows, cols = np.where(comb["patch"] != BG)
        comb["coords"] = np.column_stack((rows + comb_r0, cols + comb_c0))
        comb["bbox"] = (
            comb_r0,
            comb_c0,
            comb_r0 + comb["patch"].shape[0] - 1,
            comb_c0 + comb["patch"].shape[1] - 1,
        )

    scene = {
        "input": grid,
        "frames": frames,
        "combs": combs,
        "matched_signatures": matched_signatures,
        "unmatched_signatures": unmatched_signatures,
        "frame_inner_width": frame_inner_width,
        "bar_width": bar_width,
    }
    return scene if _validate_scene(scene) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return scene["cropped_input"]
    raise RuntimeError("failed to generate a valid comb-frame ARC input")
