from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste


BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 400
SHAPE_RETRIES = 120
EXTRA_ANCHOR_PROB = 0.55


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"].astype(int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_kind(mask: np.ndarray) -> str:
    vertical_axis = bool(np.array_equal(mask, mask[:, ::-1]))
    horizontal_axis = bool(np.array_equal(mask, mask[::-1, :]))
    if vertical_axis and horizontal_axis:
        return "anchor"
    if vertical_axis:
        return "vertical"
    if horizontal_axis:
        return "horizontal"
    raise AssertionError("every object must have vertical-axis, horizontal-axis, or biaxial symmetry")


def _movement(bbox: tuple[int, int, int, int], kind: str, shape: tuple[int, int], *, swap_axes: bool = False) -> tuple[tuple[int, int], int, str]:
    H, W = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    moving_kind = kind
    if swap_axes:
        moving_kind = "horizontal" if kind == "vertical" else "vertical"

    if moving_kind == "vertical":
        top = r0
        bottom = H - 1 - r1
        assert top != bottom, "vertical movers may not be equidistant from top and bottom edges"
        if top < bottom:
            return (-1, 0), int(top), "top"
        return (1, 0), int(bottom), "bottom"

    assert moving_kind == "horizontal"
    left = c0
    right = W - 1 - c1
    assert left != right, "horizontal movers may not be equidistant from left and right edges"
    if left < right:
        return (0, -1), int(left), "left"
    return (0, 1), int(right), "right"


def _sort_key(obj: dict) -> tuple[int, int, int, int, int]:
    dr, dc = obj["step"]
    r0, c0, r1, c1 = obj["bbox"]
    if (dr, dc) == (-1, 0):
        axis_key = r0
        cross_key = c0
        direction_rank = 0
    elif (dr, dc) == (1, 0):
        axis_key = -r1
        cross_key = c0
        direction_rank = 1
    elif (dr, dc) == (0, -1):
        axis_key = c0
        cross_key = r0
        direction_rank = 2
    else:
        axis_key = -c1
        cross_key = r0
        direction_rank = 3
    return (int(obj["target_dist"]), direction_rank, int(axis_key), int(cross_key), int(obj["color"]))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_objects(grid: np.ndarray, *, swap_axes: bool = False) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one foreground object"

    anchors: list[dict] = []
    movers: list[dict] = []
    for idx, comp in enumerate(comps):
        mask = _component_mask(comp)
        kind = _symmetry_kind(mask)
        record = {
            "id": idx,
            "color": int(comp["color"]),
            "coords": comp["coords"].astype(int).copy(),
            "bbox": tuple(map(int, comp["bbox"])),
            "area": int(comp["area"]),
            "kind": kind,
        }
        if kind == "anchor":
            anchors.append(record)
            continue
        step, target_dist, target_edge = _movement(record["bbox"], kind, g.shape, swap_axes=swap_axes)
        record["step"] = step
        record["target_dist"] = int(target_dist)
        record["target_edge"] = target_edge
        movers.append(record)

    assert anchors, "expected at least one biaxial anchor"
    assert movers, "expected at least one sliding object"
    return bg, anchors, movers


def _transform(
    grid: np.ndarray,
    *,
    ignore_anchors: bool = False,
    reverse_order: bool = False,
    swap_axes: bool = False,
    return_trace: bool = False,
):
    g = np.asarray(grid, dtype=int)
    bg, anchors, movers = _parse_objects(g, swap_axes=swap_axes)
    out = np.full_like(g, bg)
    anchor_occ = np.zeros(g.shape, dtype=bool)
    settled_occ = np.zeros(g.shape, dtype=bool)

    for anchor in anchors:
        coords = anchor["coords"]
        out[coords[:, 0], coords[:, 1]] = anchor["color"]
        anchor_occ[coords[:, 0], coords[:, 1]] = True

    ordered = sorted(movers, key=_sort_key)
    if reverse_order:
        ordered = list(reversed(ordered))

    trace: list[dict] = []
    H, W = map(int, g.shape)
    for obj in ordered:
        coords = obj["coords"].copy()
        dr, dc = obj["step"]
        step = np.array([dr, dc], dtype=int)

        while True:
            nxt = coords + step
            in_bounds = (
                np.all(0 <= nxt[:, 0])
                and np.all(nxt[:, 0] < H)
                and np.all(0 <= nxt[:, 1])
                and np.all(nxt[:, 1] < W)
            )
            if not in_bounds:
                stop_reason = "edge"
                break
            if not ignore_anchors and np.any(anchor_occ[nxt[:, 0], nxt[:, 1]]):
                stop_reason = "anchor"
                break
            if np.any(settled_occ[nxt[:, 0], nxt[:, 1]]):
                stop_reason = "settled"
                break
            coords = nxt

        out[coords[:, 0], coords[:, 1]] = obj["color"]
        settled_occ[coords[:, 0], coords[:, 1]] = True
        if return_trace:
            trace.append(
                {
                    "id": obj["id"],
                    "kind": obj["kind"],
                    "step": obj["step"],
                    "target_dist": obj["target_dist"],
                    "target_edge": obj["target_edge"],
                    "stop_reason": stop_reason,
                    "final_bbox": _bbox_from_coords(coords),
                }
            )

    if return_trace:
        return out, anchors, ordered, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each single-axis-symmetric object along its own symmetry axis toward the nearer edge, while biaxial objects stay put as blockers.

    Input structure:
    - A black background contains several separated monochromatic 4-connected foreground objects.
    - Every object is exactly one of three symmetry classes: vertical-axis only, horizontal-axis only, or biaxial.
    - No sliding object is equally distant from the two opposite edges on its own motion axis.

    Rule:
    - Biaxial objects remain fixed in place and act as obstacles.
    - Vertical-axis-only objects slide vertically toward the nearer of the top and bottom edges.
    - Horizontal-axis-only objects slide horizontally toward the nearer of the left and right edges.
    - Objects nearer to their destination edge settle first; each object moves rigidly until one more step would leave the grid, hit a fixed biaxial anchor, or hit an already settled object.

    Key insight:
    - Reflection symmetry is the motion instruction: one mirror axis chooses the sliding axis, while two mirror axes mean the object becomes a stationary blocker instead of a mover.
    """
    return _transform(np.asarray(grid, dtype=int))


def _mask_from_profile(profile: list[int], width: int) -> np.ndarray:
    h = len(profile)
    w = int(width)
    mask = np.zeros((h, w), dtype=bool)
    if w % 2 == 1:
        center = w // 2
        for r, ext in enumerate(profile):
            mask[r, center - ext : center + ext + 1] = True
    else:
        left_center = w // 2 - 1
        right_center = w // 2
        for r, ext in enumerate(profile):
            mask[r, left_center - ext : right_center + ext + 1] = True
    return mask


def _make_vertical_mask(height: int, width: int) -> np.ndarray:
    max_ext = width // 2 if width % 2 else width // 2 - 1
    for _ in range(SHAPE_RETRIES):
        profile = [random.randint(0, max_ext) for _ in range(height)]
        if len(set(profile)) < 2:
            continue
        if profile == profile[::-1]:
            continue
        mask = _mask_from_profile(profile, width)
        if _symmetry_kind(mask) == "vertical":
            return mask
    raise RuntimeError("failed to generate a vertical-axis-only shape")


def _make_horizontal_mask(height: int, width: int) -> np.ndarray:
    mask = _make_vertical_mask(width, height).T
    assert _symmetry_kind(mask) == "horizontal"
    return mask


def _make_anchor_mask(height: int, width: int) -> np.ndarray:
    max_ext = width // 2 if width % 2 else width // 2 - 1
    half = (height + 1) // 2
    for _ in range(SHAPE_RETRIES):
        front = [random.randint(0, max_ext) for _ in range(half)]
        if max_ext > 0 and max(front) == 0:
            continue
        if height % 2 == 1:
            profile = front + front[-2::-1]
        else:
            profile = front + front[::-1]
        mask = _mask_from_profile(profile, width)
        if _symmetry_kind(mask) == "anchor":
            return mask
    raise RuntimeError("failed to generate a biaxial anchor shape")


def _fits(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, *, padding: int = 1) -> bool:
    H, W = occupied.shape
    coords = np.argwhere(mask)
    for rr, cc in coords:
        r = int(r0 + rr)
        c = int(c0 + cc)
        if not (0 <= r < H and 0 <= c < W):
            return False
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                    return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = np.where(mask, int(color), BG).astype(int)
    paste(grid, patch, int(r0), int(c0), transparent=BG)
    coords = np.argwhere(mask)
    occupied[coords[:, 0] + int(r0), coords[:, 1] + int(c0)] = True


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _spans_overlap(a: dict, b: dict) -> bool:
    if a["step"] != b["step"]:
        return False
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    if a["step"][0] != 0:
        return _overlap_1d(ac0, ac1, bc0, bc1)
    return _overlap_1d(ar0, ar1, br0, br1)


def _transformed(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    return np.rot90(out, k=k).copy()


def _try_alt_transform(grid: np.ndarray, **kwargs) -> np.ndarray | None:
    try:
        return _transform(grid, **kwargs)
    except AssertionError:
        return None


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    try:
        out, anchors, movers, trace = _transform(g, return_trace=True)
    except AssertionError:
        return False

    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if len(anchors) < 1 or len(movers) < 6:
        return False
    if np.array_equal(g, out):
        return False

    directions = {tuple(item["step"]) for item in movers}
    if directions != {(-1, 0), (1, 0), (0, -1), (0, 1)}:
        return False

    kinds = {item["kind"] for item in movers}
    if kinds != {"vertical", "horizontal"}:
        return False

    if not any(item["stop_reason"] == "anchor" for item in trace):
        return False
    if not any(item["stop_reason"] == "settled" for item in trace):
        return False

    for i, mover_a in enumerate(movers):
        for mover_b in movers[i + 1 :]:
            if _spans_overlap(mover_a, mover_b) and mover_a["target_dist"] == mover_b["target_dist"]:
                return False

    ignore_anchors = _try_alt_transform(g, ignore_anchors=True)
    if ignore_anchors is not None and np.array_equal(ignore_anchors, out):
        return False

    reverse_order = _try_alt_transform(g, reverse_order=True)
    if reverse_order is not None and np.array_equal(reverse_order, out):
        return False

    swap_axes = _try_alt_transform(g, swap_axes=True)
    if swap_axes is not None and np.array_equal(swap_axes, out):
        return False

    return True


def _build_canonical() -> np.ndarray:
    H = random.randint(26, 29)
    W = random.randint(26, 29)
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    extra_anchor = random.random() < EXTRA_ANCHOR_PROB
    colors = random.sample(PALETTE, k=7 + int(extra_anchor))
    color_idx = 0

    anchor_h = random.randint(4, 6)
    anchor_w = random.randint(4, 6)
    anchor_mask = _make_anchor_mask(anchor_h, anchor_w)
    up_w = random.randint(3, 5)
    up_near_h = random.randint(3, 5)
    up_far_h = random.randint(3, 5)
    up_near_mask = _make_vertical_mask(up_near_h, up_w)
    up_far_mask = _make_vertical_mask(up_far_h, up_w)
    up_c0_max = W // 3
    if up_c0_max < 2:
        raise AssertionError("no room for the upward lane")
    up_c0 = random.randint(2, up_c0_max)
    up_near_r0 = random.randint(1, 3)
    up_far_r0_min = up_near_r0 + up_near_h + random.randint(2, 4)
    up_far_r0_max = (H - up_far_h - 1) // 2
    if up_far_r0_min > up_far_r0_max:
        raise AssertionError("no room for the stacked upward pair")

    left_h = random.randint(3, 5)
    left_near_w = random.randint(3, 5)
    left_far_w = random.randint(3, 5)
    left_near_mask = _make_horizontal_mask(left_h, left_near_w)
    left_far_mask = _make_horizontal_mask(left_h, left_far_w)
    up_far_r0 = random.randint(up_far_r0_min, up_far_r0_max)
    left_r0_min = up_far_r0 + up_far_h + 2
    left_near_c0 = random.randint(1, 3)
    left_far_c0_min = left_near_c0 + left_near_w + random.randint(2, 4)
    left_far_c0_max = (W - left_far_w - 1) // 2
    if left_far_c0_min > left_far_c0_max:
        raise AssertionError("no room for the stacked leftward pair")

    down_w = random.randint(3, min(5, anchor_w))
    down_h = random.randint(3, 5)
    down_mask = _make_vertical_mask(down_h, down_w)

    right_h = random.randint(3, min(5, anchor_h))
    right_w = random.randint(3, 5)
    right_mask = _make_horizontal_mask(right_h, right_w)
    min_down_r0 = (H - down_h) // 2 + 1
    min_right_c0 = (W - right_w) // 2 + 1
    anchor_r0_lo = max((2 * H) // 3 - 1, min_down_r0 + down_h + 2)
    anchor_r0_hi = H - anchor_h - 1
    anchor_c0_lo = max((2 * W) // 3 - 1, min_right_c0 + right_w + 2)
    anchor_c0_hi = W - anchor_w - 1
    if anchor_r0_lo > anchor_r0_hi or anchor_c0_lo > anchor_c0_hi:
        raise AssertionError("anchor cannot satisfy both blocking constraints")
    anchor_r0 = random.randint(anchor_r0_lo, anchor_r0_hi)
    anchor_c0 = random.randint(anchor_c0_lo, anchor_c0_hi)
    left_r0_max = min(H - left_h - 2, anchor_r0 - left_h - 2)
    if left_r0_min > left_r0_max:
        raise AssertionError("no room for the leftward lane")
    left_r0 = random.randint(left_r0_min, left_r0_max)

    _stamp(grid, occupied, anchor_mask, anchor_r0, anchor_c0, colors[color_idx])
    color_idx += 1
    _stamp(grid, occupied, up_near_mask, up_near_r0, up_c0, colors[color_idx])
    color_idx += 1
    _stamp(grid, occupied, up_far_mask, up_far_r0, up_c0, colors[color_idx])
    color_idx += 1
    left_far_c0 = random.randint(left_far_c0_min, left_far_c0_max)
    _stamp(grid, occupied, left_near_mask, left_r0, left_near_c0, colors[color_idx])
    color_idx += 1
    _stamp(grid, occupied, left_far_mask, left_r0, left_far_c0, colors[color_idx])
    color_idx += 1

    down_c0 = random.randint(anchor_c0, anchor_c0 + anchor_w - down_w)
    valid_down_gaps = [
        gap
        for gap in range(2, 5)
        if anchor_r0 - gap - down_h >= min_down_r0 and anchor_r0 - gap - down_h >= 1
    ]
    if not valid_down_gaps:
        raise AssertionError("no valid down-moving anchor blocker")
    down_gap = random.choice(valid_down_gaps)
    down_r0 = anchor_r0 - down_gap - down_h
    if not _fits(occupied, down_mask, down_r0, down_c0):
        raise AssertionError("down mover touches another object")
    _stamp(grid, occupied, down_mask, down_r0, down_c0, colors[color_idx])
    color_idx += 1

    right_r0 = random.randint(anchor_r0, anchor_r0 + anchor_h - right_h)
    valid_right_gaps = [
        gap
        for gap in range(2, 5)
        if anchor_c0 - gap - right_w >= min_right_c0 and anchor_c0 - gap - right_w >= 1
    ]
    if not valid_right_gaps:
        raise AssertionError("no valid right-moving anchor blocker")
    right_gap = random.choice(valid_right_gaps)
    right_c0 = anchor_c0 - right_gap - right_w
    if not _fits(occupied, right_mask, right_r0, right_c0):
        raise AssertionError("right mover touches another object")
    _stamp(grid, occupied, right_mask, right_r0, right_c0, colors[color_idx])
    color_idx += 1

    if extra_anchor:
        extra_h = random.randint(3, 5)
        extra_w = random.randint(3, 5)
        extra_mask = _make_anchor_mask(extra_h, extra_w)
        row_hi = max(1, H // 3)
        col_lo = max(W // 2, anchor_c0)
        col_hi = W - extra_w - 1
        if col_lo <= col_hi:
            for _ in range(40):
                r0 = random.randint(1, row_hi)
                c0 = random.randint(col_lo, col_hi)
                if _fits(occupied, extra_mask, r0, c0):
                    _stamp(grid, occupied, extra_mask, r0, c0, colors[color_idx])
                    color_idx += 1
                    break

    return _transformed(grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _build_canonical()
        except (AssertionError, RuntimeError, ValueError):
            continue
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid symmetry-directed sliding instance")
