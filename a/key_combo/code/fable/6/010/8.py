from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in comps:
        color = int(comp["color"])
        b = tuple(map(int, comp["bbox"]))
        h = bbox_h(b)
        w = bbox_w(b)
        if h < 4 or w < 4:
            continue
        patch = crop_bbox(grid, b)
        border = _border_mask(h, w)
        if int(comp["area"]) != int(border.sum()):
            continue
        if not np.array_equal(patch == color, border):
            continue
        r0, c0, r1, c1 = b
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        quota = int(np.count_nonzero(interior != bg))
        frames.append(
            {
                "bbox": b,
                "color": color,
                "quota": quota,
            }
        )
    frames.sort(key=lambda fr: (fr["bbox"][0], fr["bbox"][1], fr["color"]))
    return frames


def _outside_all_bboxes(pos: tuple[int, int], frames: list[dict[str, Any]]) -> bool:
    r, c = pos
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return False
    return True


def _detect_markers(grid: np.ndarray, bg: int, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        if _outside_all_bboxes((rr, cc), frames):
            markers.append({"pos": (rr, cc), "color": int(grid[rr, cc])})
    markers.sort(key=lambda mk: (mk["pos"][0], mk["pos"][1], mk["color"]))
    return markers


def _candidate_for_pair(
    grid: np.ndarray,
    bg: int,
    marker: dict[str, Any],
    frame: dict[str, Any],
) -> dict[str, Any] | None:
    r, c = marker["pos"]
    r0, c0, r1, c1 = frame["bbox"]
    inside: tuple[int, int]
    dock: tuple[int, int]
    side: str
    distance: int

    if r0 < r < r1 and c < c0 and np.all(grid[r, c + 1 : c0] == bg):
        inside = (r, c0 + 1)
        dock = (r, c0 - 1)
        side = "L"
        distance = int(c0 + 1 - c)
    elif r0 < r < r1 and c > c1 and np.all(grid[r, c1 + 1 : c] == bg):
        inside = (r, c1 - 1)
        dock = (r, c1 + 1)
        side = "R"
        distance = int(c - (c1 - 1))
    elif c0 < c < c1 and r < r0 and np.all(grid[r + 1 : r0, c] == bg):
        inside = (r0 + 1, c)
        dock = (r0 - 1, c)
        side = "U"
        distance = int(r0 + 1 - r)
    elif c0 < c < c1 and r > r1 and np.all(grid[r1 + 1 : r, c] == bg):
        inside = (r1 - 1, c)
        dock = (r1 + 1, c)
        side = "D"
        distance = int(r - (r1 - 1))
    else:
        return None

    H, W = grid.shape
    ir, ic = inside
    dr, dc = dock
    if not (0 <= ir < H and 0 <= ic < W and 0 <= dr < H and 0 <= dc < W):
        return None
    if int(grid[ir, ic]) != int(bg):
        return None
    if (dr, dc) != (r, c) and int(grid[dr, dc]) != int(bg):
        return None
    return {
        "distance": distance,
        "inside": inside,
        "dock": dock,
        "side": side,
    }


def _all_candidates(
    grid: np.ndarray,
    bg: int,
    frames: list[dict[str, Any]],
    markers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mi, marker in enumerate(markers):
        for fi, frame in enumerate(frames):
            cand = _candidate_for_pair(grid, bg, marker, frame)
            if cand is None:
                continue
            cand["marker_idx"] = mi
            cand["frame_idx"] = fi
            out.append(cand)
    out.sort(key=lambda cand: (cand["distance"], cand["marker_idx"], cand["frame_idx"]))
    return out


def _simulate_assignment(
    grid: np.ndarray,
    frames: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    out = np.array(grid, dtype=int, copy=True)
    for marker in markers:
        r, c = marker["pos"]
        out[r, c] = get_bg(grid)

    per_marker: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(markers))}
    for cand in candidates:
        per_marker[int(cand["marker_idx"])].append(cand)

    remaining = [int(frame["quota"]) for frame in frames]
    placements: dict[int, dict[str, Any]] = {}

    for cand in candidates:
        mi = int(cand["marker_idx"])
        fi = int(cand["frame_idx"])
        if mi in placements:
            continue
        if remaining[fi] <= 0:
            continue
        remaining[fi] -= 1
        placements[mi] = {
            "mode": "inside",
            "pos": tuple(cand["inside"]),
            "color": int(frames[fi]["color"]),
            "frame_idx": fi,
            "distance": int(cand["distance"]),
            "nearest_distance": int(per_marker[mi][0]["distance"]),
        }

    for mi, marker in enumerate(markers):
        if mi in placements:
            continue
        nearest = per_marker[mi][0]
        placements[mi] = {
            "mode": "dock",
            "pos": tuple(nearest["dock"]),
            "color": int(marker["color"]),
            "frame_idx": int(nearest["frame_idx"]),
            "distance": int(nearest["distance"]),
            "nearest_distance": int(nearest["distance"]),
        }

    for info in placements.values():
        r, c = info["pos"]
        out[r, c] = int(info["color"])
    return out, placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Greedily absorb outside markers into multiple quota-limited hollow rectangles by global distance order.

    Input structure:
    - A background grid contains at least two one-cell-thick hollow rectangular frames, each in its own border color.
    - Each frame contains some interior dot pixels; the number of such dots is that frame's quota.
    - Additional singleton markers lie outside the frames, each aligned with at least one frame along a clear row or column.

    Rule:
    - For every reachable (marker, frame) pair, measure the straight slide distance to the cell just inside the entry side.
    - Sort all reachable pairs globally by increasing distance. In that order, a marker enters its frame if that frame still has quota, landing on the just-inside cell and recoloring to the frame color; otherwise that pair is skipped and the marker may still enter a farther frame later.
    - After all pairs are processed, every still-unassigned marker slides to the nearest reachable frame's touching outside cell and keeps its original color.

    Key insight:
    - Quotas are local to frames, but ownership is decided by one global distance ordering over all pairs, so a marker can miss its nearest frame and later be absorbed by a farther one.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frames = _detect_frames(g, bg)
    markers = _detect_markers(g, bg, frames)
    candidates = _all_candidates(g, bg, frames, markers)
    out, _placements = _simulate_assignment(g, frames, markers, candidates)
    return out


def _frame_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _render_scene(scene: dict[str, Any]) -> np.ndarray:
    grid = np.full((scene["H"], scene["W"]), int(scene["bg"]), dtype=int)
    for frame in scene["frames"]:
        for r, c in _frame_cells(frame["bbox"]):
            grid[r, c] = int(frame["color"])
        for r, c in frame["dots"]:
            grid[r, c] = int(scene["dot_color"])
    for marker in scene["markers"]:
        r, c = marker["pos"]
        grid[r, c] = int(scene["marker_color"])
    return grid


def _marker_roles_to_scene(
    H: int,
    W: int,
    bg: int,
    frame_colors: list[int],
    dot_color: int,
    marker_color: int,
    bboxes: list[tuple[int, int, int, int]],
    marker_positions: dict[str, tuple[int, int]],
    dot_positions: list[list[tuple[int, int]]],
    quotas: list[int],
) -> dict[str, Any]:
    frames = []
    for idx, bbox in enumerate(bboxes):
        frames.append(
            {
                "name": chr(ord("A") + idx),
                "bbox": bbox,
                "color": int(frame_colors[idx]),
                "quota": int(quotas[idx]),
                "dots": list(dot_positions[idx]),
            }
        )
    markers = [
        {"name": name, "pos": pos}
        for name, pos in marker_positions.items()
    ]
    markers.sort(key=lambda mk: mk["name"])
    return {
        "H": H,
        "W": W,
        "bg": int(bg),
        "dot_color": int(dot_color),
        "marker_color": int(marker_color),
        "frames": frames,
        "markers": markers,
    }


def _naive_nearest_only_output(
    grid: np.ndarray,
    frames: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    bg = int(get_bg(grid))
    for marker in markers:
        r, c = marker["pos"]
        out[r, c] = bg

    remaining = [int(frame["quota"]) for frame in frames]
    per_marker: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(markers))}
    for cand in candidates:
        per_marker[int(cand["marker_idx"])].append(cand)

    nearest_pairs = [cands[0] for cands in per_marker.values()]
    nearest_pairs.sort(key=lambda cand: cand["distance"])

    for cand in nearest_pairs:
        mi = int(cand["marker_idx"])
        fi = int(cand["frame_idx"])
        marker = markers[mi]
        if remaining[fi] > 0:
            remaining[fi] -= 1
            r, c = cand["inside"]
            out[r, c] = int(frames[fi]["color"])
        else:
            r, c = cand["dock"]
            out[r, c] = int(marker["color"])
    return out


def _choose_dot_positions(
    rng: np.random.Generator,
    bboxes: list[tuple[int, int, int, int]],
    blocked_inside: dict[int, set[tuple[int, int]]],
    quotas: list[int],
) -> list[list[tuple[int, int]]]:
    dot_positions: list[list[tuple[int, int]]] = []
    for fi, bbox in enumerate(bboxes):
        r0, c0, r1, c1 = bbox
        candidates = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if (r, c) not in blocked_inside[fi]
        ]
        rng.shuffle(candidates)
        need = int(quotas[fi])
        if len(candidates) < need:
            raise ValueError("not enough free interior cells for dots")
        dot_positions.append(candidates[:need])
    return dot_positions


def _search_canonical_scene(rng: np.random.Generator) -> dict[str, Any] | None:
    bg = 0
    palette = list(range(1, 10))
    rng.shuffle(palette)
    frame_colors = palette[:3]
    dot_color = palette[3]
    marker_color = palette[4]
    quotas = [1, 1, 1]

    h = int(rng.integers(6, 9))
    widths = [4, 4, 4]
    left_margin = 2
    right_margin = int(rng.integers(1, 3))
    gap_ab = int(rng.integers(4, 6))
    gap_bc = int(rng.integers(8, 10))

    W = left_margin + sum(widths) + gap_ab + gap_bc + right_margin
    if W > 30:
        return None

    top_margin = int(rng.integers(1, 6))
    bottom_margin = int(rng.integers(1, 6))
    H = top_margin + h + bottom_margin
    if H > 30:
        return None

    r0 = top_margin
    a = (r0, left_margin, r0 + h - 1, left_margin + widths[0] - 1)
    b = (r0, a[3] + 1 + gap_ab, r0 + h - 1, a[3] + gap_ab + widths[1])
    cbox = (r0, b[3] + 1 + gap_bc, r0 + h - 1, b[3] + gap_bc + widths[2])
    bboxes = [a, b, cbox]

    rows = list(range(r0 + 1, r0 + h - 1))
    if len(rows) < 4:
        return None
    rng.shuffle(rows)
    role_rows = {
        "enter_a": rows[0],
        "dock_a": rows[1],
        "enter_b": rows[2],
        "spill_to_c": rows[3],
    }

    gap_bc_start = b[3] + 1
    marker_positions = {
        "enter_a": (role_rows["enter_a"], 1),
        "dock_a": (role_rows["dock_a"], 0),
        "enter_b": (role_rows["enter_b"], gap_bc_start + 2),
        "spill_to_c": (role_rows["spill_to_c"], gap_bc_start + 3),
    }

    temp_scene = _marker_roles_to_scene(
        H=H,
        W=W,
        bg=bg,
        frame_colors=frame_colors,
        dot_color=dot_color,
        marker_color=marker_color,
        bboxes=bboxes,
        marker_positions=marker_positions,
        dot_positions=[[], [], []],
        quotas=quotas,
    )
    temp_grid = _render_scene(temp_scene)
    frames = _detect_frames(temp_grid, bg)
    markers = _detect_markers(temp_grid, bg, frames)
    candidates = _all_candidates(temp_grid, bg, frames, markers)
    if len(markers) != 4 or len(candidates) != 6:
        return None
    dists = [int(cand["distance"]) for cand in candidates]
    if len(set(dists)) != len(dists):
        return None

    blocked_inside: dict[int, set[tuple[int, int]]] = {0: set(), 1: set(), 2: set()}
    for cand in candidates:
        blocked_inside[int(cand["frame_idx"])].add(tuple(cand["inside"]))
    dot_positions = _choose_dot_positions(rng, bboxes, blocked_inside, quotas)

    return _marker_roles_to_scene(
        H=H,
        W=W,
        bg=bg,
        frame_colors=frame_colors,
        dot_color=dot_color,
        marker_color=marker_color,
        bboxes=bboxes,
        marker_positions=marker_positions,
        dot_positions=dot_positions,
        quotas=quotas,
    )


def _validate_scene(scene: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    grid = _render_scene(scene)
    bg = int(scene["bg"])
    frames = _detect_frames(grid, bg)
    markers = _detect_markers(grid, bg, frames)
    assert len(frames) == 3
    assert len(markers) == 4
    assert sum(int(frame["quota"]) for frame in frames) == 3

    candidates = _all_candidates(grid, bg, frames, markers)
    assert max(grid.shape) <= 30
    assert len(candidates) == 6
    dists = [int(cand["distance"]) for cand in candidates]
    assert len(set(dists)) == len(dists)

    per_marker: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(markers))}
    for cand in candidates:
        per_marker[int(cand["marker_idx"])].append(cand)
    assert any(len(v) > 1 for v in per_marker.values())

    out, placements = _simulate_assignment(grid, frames, markers, candidates)
    assert out.ndim == 2 and max(out.shape) <= 30

    reassigned = [
        info
        for info in placements.values()
        if info["mode"] == "inside" and info["distance"] > info["nearest_distance"]
    ]
    parked = [info for info in placements.values() if info["mode"] == "dock"]
    assert reassigned, "must contain at least one farther-frame absorption"
    assert parked, "must contain at least one unassigned parked marker"

    naive = _naive_nearest_only_output(grid, frames, markers, candidates)
    assert not np.array_equal(naive, out), "quota-skipping retry step must matter"

    marker_pos_to_name = {tuple(marker["pos"]): marker["name"] for marker in scene["markers"]}
    frame_bbox_to_name = {tuple(frame["bbox"]): frame["name"] for frame in scene["frames"]}
    named = {
        marker_pos_to_name[tuple(markers[idx]["pos"])]: {
            **info,
            "frame": frame_bbox_to_name[tuple(frames[int(info["frame_idx"])]["bbox"])],
        }
        for idx, info in placements.items()
    }
    assert named["enter_a"]["mode"] == "inside" and named["enter_a"]["frame"] == "A"
    assert named["dock_a"]["mode"] == "dock" and named["dock_a"]["frame"] == "A"
    assert named["enter_b"]["mode"] == "inside" and named["enter_b"]["frame"] == "B"
    assert named["spill_to_c"]["mode"] == "inside" and named["spill_to_c"]["frame"] == "C"
    assert named["spill_to_c"]["distance"] > named["spill_to_c"]["nearest_distance"]

    solved = main(grid)
    assert np.array_equal(solved, out)
    return grid, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(600):
        scene = _search_canonical_scene(rng)
        if scene is None:
            continue
        grid, out = _validate_scene(scene)
        k = int(rng.integers(0, 4))
        if k:
            grid = np.rot90(grid, k)
            out = np.rot90(out, k)
        assert np.array_equal(main(grid), out)
        return grid
    raise RuntimeError("failed to generate a valid scene")
