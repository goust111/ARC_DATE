# Combo variant of a25697e4 (index 18)
# combo_spec: D1/2+D3/3+D5/2
# Name: 空心洞数匹配T形凹槽
#
# Design analysis
# a) Concrete behavior and trace:
#    Original a25697e4 pairs a tool to a frame by a color-region area equal to one
#    staircase cavity area, then docks a solid plug into that cavity. This combo
#    changes all three relevant stages together.
#    Small fused trace:
#    - Frame A touches the left border and contains a right-opening thick T socket.
#      The socket's outer cavity area is 21 cells, and its branch count is 3.
#    - Frame B touches the top border and contains a down-opening thick cross
#      socket. Its outer cavity area is also 21 cells, but its branch count is 4.
#    - Tool A is a 2-color component whose plug is the same thick T silhouette
#      with 3 enclosed 1-cell background holes; Tool B is the analogous cross with
#      4 enclosed holes. Both tools have the same handle size, and neither tool has
#      a color-region size equal to 21.
#    - Fused logic must:
#      1. detect border-touching frame blocks and recover each cavity silhouette,
#      2. classify each cavity as T or cross, hence infer branch count 3 or 4,
#      3. count internal background-hole components inside each movable tool,
#      4. pair tool-to-frame by `tool_holes == cavity_branches`,
#      5. rotate/flip the tool so one color region plus its preserved holes matches
#         the cavity exactly, leaving those holes as background in the output.
#    The visible difference from the original is strong: cavity area no longer
#    identifies the pair, and the docked plug remains hollow instead of solid.
#
# b) Complexity calibration:
#    The generator uses only two cavity families, thick T and thick cross, and it
#    makes both families share the same outer area (21). That keeps the puzzle
#    visually clean while explicitly disabling the original area-matching shortcut.
#    With 2 pairs, one sample always contains one T and one cross so the count cue
#    is supported by repetition rather than a single one-off case.
#
# c) Hidden assumptions broken from the original main():
#    - `cavity_size == one tool color count` is false here by construction; plug
#      counts are 18 or 17 while cavity area is always 21.
#    - The plug is no longer solid, so "fill the cavity exactly" must treat some
#      cavity cells as intentionally background hole cells.
#    - The frame cavity is no longer the original staircase; matching by raw area
#      or by the original notch geometry would be wrong.
#    - Pairing cannot rely on one global cavity-size permutation because both T and
#      cross cavities share the same area; the topological key is branch/hole count.
#    The rewritten solver therefore extracts cavities and tools independently,
#    counts tool holes, classifies cavity silhouette, and only then searches D4
#    placements for an exact hollow-plug match.
#
# d) Placement legality:
#    Generator builds the docked output first, then removes each tool and places a
#    transformed copy on background-only cells away from every frame. Frames touch
#    exactly one canvas border, scenes never overlap, tools stay strictly inside the
#    canvas, and internal holes are enclosed by foreground on all four sides.
#
# e) Visual observability:
#    A human can see that tools are hollow and that some frames show three-way
#    sockets while others show four-way sockets. Because T and cross cavities have
#    identical area but different branch counts, the examples visually force the
#    solver to notice topology rather than pixel count.
#
# f) Visual cleanliness:
#    Each frame is a single monochrome border block with one carved socket. Each
#    tool is one detached 2-color component with a straight handle. No region mixes
#    several cues at once, and internal holes are single pixels separated inside a
#    thick plug, so the board remains readable.
#
# g) Color inference:
#    There is still a reflection-triggered color swap when the chosen docking
#    transform is a flip and plug/handle sizes differ. This is structurally hinted
#    by the same two-region tool family across samples; no color-id ordering or
#    invisible numeric property is used.
#
# h) Quantitative anti-degeneration target:
#    Valid samples must satisfy all of:
#    - every tool has 3 or 4 internal holes,
#    - every cavity has area 21,
#    - no tool color-region count equals its frame cavity area,
#    - at least one generated tool is placed in input with a reflected variant.
#    This ensures the combo does not collapse back to the original area-matching
#    task and that the hollow + reflection behaviors are exercised.
#
# i) New solver assumptions and generator guarantees:
#    - Solver assumes every border-touching foreground component is exactly one
#      frame block and each non-border foreground component is exactly one tool.
#      Generator enforces 1-2 frames and 1-2 tools with generous spacing.
#    - Solver assumes each frame bbox contains exactly one bg cavity component and
#      that this cavity is a T or cross template under D4. Generator uses only
#      those templates.
#    - Solver assumes each tool has exactly two non-bg colors and that all internal
#      holes belong to the plug footprint, not the handle. Generator constructs the
#      tools directly from that geometry.

from __future__ import annotations

from collections import Counter
import itertools
import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import STRUCT4, get_bg, paste


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["left", "right", "top", "bottom"]
OPEN_DIR_BY_SIDE = {"left": "right", "right": "left", "top": "down", "bottom": "up"}
BRANCH_COUNT = {"T": 3, "PLUS": 4}

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
# Keep both socket families visible in every sample so the hole-count cue is
# observable and the task does not collapse into a single-pair coincidence.
NUM_PAIRS_CHOICES = [2]
HANDLE_LEN = 4
FRAME_GAP_MIN = 3
SCENE_GAP_MIN = 2
TOOL_BORDER = 1
TOOL_FRAME_GAP = 4
TOOL_TOOL_GAP = 3
GEN_RETRIES = 400


def _normalize_coords(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not cells:
        return set()
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _coords_key(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_normalize_coords(cells)))


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(a[0], a[2], b[0], b[2]) + _gap1(a[1], a[3], b[1], b[3])


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def _rot90_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _normalize_coords({(int(c), int(-r)) for r, c in cells})


def _fliplr_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _normalize_coords({(int(r), int(-c)) for r, c in cells})


def _transform_cells(cells: set[tuple[int, int]], k: int, flip: bool) -> set[tuple[int, int]]:
    out = set(cells)
    for _ in range(int(k) % 4):
        out = _rot90_cells(out)
    if flip:
        out = _fliplr_cells(out)
    return _normalize_coords(out)


def _transform_aligned_sets(
    *sets: set[tuple[int, int]],
    k: int,
    flip: bool,
) -> tuple[set[tuple[int, int]], ...]:
    transformed: list[set[tuple[int, int]]] = []
    for cells in sets:
        out = set(cells)
        for _ in range(int(k) % 4):
            out = {(int(c), int(-r)) for r, c in out}
        if flip:
            out = {(int(r), int(-c)) for r, c in out}
        transformed.append(out)
    union = set().union(*transformed)
    r0 = min(r for r, _ in union)
    c0 = min(c for _, c in union)
    return tuple({_shift for _shift in {(int(r - r0), int(c - c0)) for r, c in cells}} for cells in transformed)


def _render_patch(
    *,
    bg: int,
    frame_cells: set[tuple[int, int]] | None = None,
    plug_cells: set[tuple[int, int]] | None = None,
    handle_cells: set[tuple[int, int]] | None = None,
    frame_color: int | None = None,
    plug_color: int | None = None,
    handle_color: int | None = None,
) -> tuple[np.ndarray, dict[str, set[tuple[int, int]]]]:
    named = {
        "frame": set(frame_cells or set()),
        "plug": set(plug_cells or set()),
        "handle": set(handle_cells or set()),
    }
    union = set().union(*named.values())
    if not union:
        return np.full((1, 1), int(bg), dtype=int), named
    r0, c0, r1, c1 = _bbox_from_cells(union)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    shifted = {name: _shift_cells(cells, -r0, -c0) for name, cells in named.items()}
    patch = np.full((h, w), int(bg), dtype=int)
    if frame_color is not None:
        for r, c in shifted["frame"]:
            patch[r, c] = int(frame_color)
    if plug_color is not None:
        for r, c in shifted["plug"]:
            patch[r, c] = int(plug_color)
    if handle_color is not None:
        for r, c in shifted["handle"]:
            patch[r, c] = int(handle_color)
    return patch, shifted


def _internal_hole_mask(mask: np.ndarray) -> np.ndarray:
    holes = np.zeros_like(mask, dtype=bool)
    bg_mask = ~mask
    if not bg_mask.any():
        return holes
    lab, n = label(bg_mask, structure=STRUCT4)
    h, w = mask.shape
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        if not len(pts):
            continue
        touches = any(int(r) in (0, h - 1) or int(c) in (0, w - 1) for r, c in pts)
        if not touches:
            holes[lab == cid] = True
    return holes


def _count_holes(mask: np.ndarray) -> int:
    holes = _internal_hole_mask(mask)
    if not holes.any():
        return 0
    _lab, n = label(holes, structure=STRUCT4)
    return int(n)


def _shape_variants(cells: set[tuple[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    out: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        for flip in (False, True):
            out.add(_coords_key(_transform_cells(cells, k, flip)))
    return out


def _base_t_cells() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    outer = {(r, c) for r in range(3) for c in range(5)}
    outer |= {(r, c) for r in range(2, 5) for c in range(1, 4)}
    holes = {(1, 1), (1, 3), (3, 2)}
    return _normalize_coords(outer), holes


def _base_plus_cells() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    outer = {(r, c) for r in range(5) for c in range(1, 4)}
    outer |= {(r, c) for r in range(1, 4) for c in range(5)}
    holes = {(1, 1), (1, 3), (3, 1), (3, 3)}
    return _normalize_coords(outer), holes


T_OUTER_BASE, T_HOLES_BASE = _base_t_cells()
PLUS_OUTER_BASE, PLUS_HOLES_BASE = _base_plus_cells()
SHAPE_LIBRARY = {
    "T": {"outer": T_OUTER_BASE, "holes": T_HOLES_BASE, "variants": _shape_variants(T_OUTER_BASE)},
    "PLUS": {"outer": PLUS_OUTER_BASE, "holes": PLUS_HOLES_BASE, "variants": _shape_variants(PLUS_OUTER_BASE)},
}


def _rotation_for_open_dir(open_dir: str) -> int:
    # Base T uses "down" as its outward branch.
    return {"down": 0, "right": 1, "up": 2, "left": 3}[open_dir]


def _oriented_shape(kind: str, open_dir: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    k = _rotation_for_open_dir(open_dir)
    outer, holes = _transform_aligned_sets(
        SHAPE_LIBRARY[kind]["outer"],
        SHAPE_LIBRARY[kind]["holes"],
        k=k,
        flip=False,
    )
    return outer, holes


def _classify_shape(cells: set[tuple[int, int]]) -> str:
    key = _coords_key(cells)
    for kind, info in SHAPE_LIBRARY.items():
        if key in info["variants"]:
            return kind
    raise ValueError("unknown cavity silhouette")


def _frame_side_from_bbox(bbox: tuple[int, int, int, int], H: int, W: int) -> str:
    sides = []
    if bbox[1] == 0:
        sides.append("left")
    if bbox[3] == W - 1:
        sides.append("right")
    if bbox[0] == 0:
        sides.append("top")
    if bbox[2] == H - 1:
        sides.append("bottom")
    if len(sides) != 1:
        raise ValueError("frame must touch exactly one canvas border")
    return sides[0]


def _choose_attach_and_handle(plug_cells: set[tuple[int, int]], open_dir: str) -> set[tuple[int, int]]:
    base_offsets = [(0, 1), (0, 2), (0, 3), (-1, 3)]
    rot_k = {"right": 0, "down": 1, "left": 2, "up": 3}[open_dir]
    offsets = list(base_offsets)
    for _ in range(rot_k):
        offsets = [(int(dc), int(-dr)) for dr, dc in offsets]
    if open_dir == "right":
        edge = sorted((r, c) for r, c in plug_cells if c == max(cc for _rr, cc in plug_cells))
        anchor_r, anchor_c = edge[len(edge) // 2]
    elif open_dir == "left":
        edge = sorted((r, c) for r, c in plug_cells if c == min(cc for _rr, cc in plug_cells))
        anchor_r, anchor_c = edge[len(edge) // 2]
    elif open_dir == "down":
        edge = sorted((r, c) for r, c in plug_cells if r == max(rr for rr, _cc in plug_cells))
        anchor_r, anchor_c = edge[len(edge) // 2]
    else:
        edge = sorted((r, c) for r, c in plug_cells if r == min(rr for rr, _cc in plug_cells))
        anchor_r, anchor_c = edge[len(edge) // 2]
    return {(int(anchor_r + dr), int(anchor_c + dc)) for dr, dc in offsets}


def _build_pair_geometry(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    kind: str,
    side: str,
) -> dict:
    open_dir = OPEN_DIR_BY_SIDE[side]
    outer, holes = _oriented_shape(kind, open_dir)
    if open_dir == "right":
        offset = (1, 2)
        frame_rect = {(r, c) for r in range(7) for c in range(7)}
    elif open_dir == "left":
        offset = (1, 0)
        frame_rect = {(r, c) for r in range(7) for c in range(7)}
    elif open_dir == "down":
        offset = (2, 1)
        frame_rect = {(r, c) for r in range(7) for c in range(7)}
    else:
        offset = (0, 1)
        frame_rect = {(r, c) for r in range(7) for c in range(7)}
    cavity = _shift_cells(outer, *offset)
    hole_cells = _shift_cells(holes, *offset)
    frame_cells = frame_rect - cavity
    plug_cells = cavity - hole_cells
    handle_cells = _choose_attach_and_handle(plug_cells, open_dir)
    scene_patch, scene_sets = _render_patch(
        bg=bg,
        frame_cells=frame_cells,
        plug_cells=plug_cells,
        handle_cells=handle_cells,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
    )
    tool_patch, tool_sets = _render_patch(
        bg=bg,
        plug_cells=plug_cells,
        handle_cells=handle_cells,
        plug_color=plug_color,
        handle_color=handle_color,
    )
    frame_patch = np.full_like(scene_patch, int(bg))
    for r, c in scene_sets["frame"]:
        frame_patch[r, c] = int(frame_color)
    frame_bbox = _bbox_from_cells(scene_sets["frame"])
    return {
        "scene_patch": scene_patch,
        "frame_patch": frame_patch,
        "tool_patch": tool_patch,
        "frame_bbox": frame_bbox,
        "kind": kind,
        "side": side,
        "open_dir": open_dir,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "plug_area": int(len(scene_sets["plug"])),
        "handle_area": int(len(scene_sets["handle"])),
        "branch_count": int(BRANCH_COUNT[kind]),
        "hole_count": int(BRANCH_COUNT[kind]),
        "cavity_area": int(len(cavity)),
    }


def _crop_non_bg(patch: np.ndarray, bg: int) -> np.ndarray:
    mask = patch != int(bg)
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scene_origins(
    side: str,
    scene_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    h, w = map(int, scene_shape)
    origins: list[tuple[int, int]] = []
    if side == "left":
        c0 = -int(frame_bbox[1])
        if 0 <= c0 <= W - w:
            origins = [(r0, c0) for r0 in range(H - h + 1)]
    elif side == "right":
        c0 = int(W - 1 - frame_bbox[3])
        if 0 <= c0 <= W - w:
            origins = [(r0, c0) for r0 in range(H - h + 1)]
    elif side == "top":
        r0 = -int(frame_bbox[0])
        if 0 <= r0 <= H - h:
            origins = [(r0, c0) for c0 in range(W - w + 1)]
    else:
        r0 = int(H - 1 - frame_bbox[2])
        if 0 <= r0 <= H - h:
            origins = [(r0, c0) for c0 in range(W - w + 1)]
    random.shuffle(origins)
    return origins


def _variant_choices(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    out: list[tuple[np.ndarray, bool]] = []
    seen: set[bytes] = set()
    for k in range(4):
        p = np.rot90(patch, k)
        for flip in (False, True):
            q = np.fliplr(p) if flip else p.copy()
            key = q.tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((q, bool(flip)))
    random.shuffle(out)
    return out


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _place_scene(
    *,
    side: str,
    scene_patch: np.ndarray,
    frame_patch: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    scene_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
    bg: int,
) -> tuple[int, int] | None:
    h, w = scene_patch.shape
    for r0, c0 in _scene_origins(side, (h, w), frame_bbox, H, W):
        scene_bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        abs_frame_bbox = (
            int(r0 + frame_bbox[0]),
            int(c0 + frame_bbox[1]),
            int(r0 + frame_bbox[2]),
            int(c0 + frame_bbox[3]),
        )
        if side in ("left", "right") and abs_frame_bbox[0] == 0:
            continue
        if side in ("left", "right") and abs_frame_bbox[2] == H - 1:
            continue
        if side in ("top", "bottom") and abs_frame_bbox[1] == 0:
            continue
        if side in ("top", "bottom") and abs_frame_bbox[3] == W - 1:
            continue
        if any(_bbox_gap(scene_bbox, prev) < SCENE_GAP_MIN for prev in scene_bboxes):
            continue
        if any(_bbox_gap(abs_frame_bbox, prev) < FRAME_GAP_MIN for prev in frame_bboxes):
            continue
        sub = output[r0 : r0 + h, c0 : c0 + w]
        if np.any((sub != int(bg)) & (scene_patch != int(bg))):
            continue
        paste(output, scene_patch, r0, c0, transparent=bg)
        paste(input_grid, frame_patch, r0, c0, transparent=bg)
        scene_bboxes.append(scene_bbox)
        frame_bboxes.append(abs_frame_bbox)
        return int(r0), int(c0)
    return None


def _component_info(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    frames: list[dict] = []
    tools: list[dict] = []
    H, W = g.shape
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
        touches = any(int(r) in (0, H - 1) or int(c) in (0, W - 1) for r, c in pts)
        mask = lab[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == cid
        patch = np.full(mask.shape, int(bg), dtype=int)
        patch[mask] = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1][mask]
        if touches:
            side = _frame_side_from_bbox(bbox, H, W)
            sub_bg = patch == int(bg)
            cav_lab, cav_n = label(sub_bg, structure=STRUCT4)
            if cav_n == 0:
                raise ValueError("frame without cavity")
            best_id = max(range(1, cav_n + 1), key=lambda x: int(np.sum(cav_lab == x)))
            cavity_local = cav_lab == best_id
            cavity_cells_local = {(int(r), int(c)) for r, c in np.argwhere(cavity_local)}
            kind = _classify_shape(cavity_cells_local)
            cavity_abs = {(bbox[0] + r, bbox[1] + c) for r, c in cavity_cells_local}
            frames.append(
                {
                    "bbox": bbox,
                    "side": side,
                    "open_dir": OPEN_DIR_BY_SIDE[side],
                    "kind": kind,
                    "branch_count": int(BRANCH_COUNT[kind]),
                    "mask_abs": {(int(r), int(c)) for r, c in pts},
                    "cavity_abs": cavity_abs,
                    "cavity_bbox": (
                        int(bbox[0] + np.argwhere(cavity_local)[:, 0].min()),
                        int(bbox[1] + np.argwhere(cavity_local)[:, 1].min()),
                        int(bbox[0] + np.argwhere(cavity_local)[:, 0].max()),
                        int(bbox[1] + np.argwhere(cavity_local)[:, 1].max()),
                    ),
                    "cavity_norm": _coords_key(cavity_cells_local),
                    "color": int(patch[mask][0]),
                }
            )
        else:
            colors = sorted({int(x) for x in patch[mask]})
            hole_mask = _internal_hole_mask(mask)
            hole_count = _count_holes(mask)
            tools.append(
                {
                    "bbox": bbox,
                    "mask": mask,
                    "patch": patch,
                    "hole_mask": hole_mask,
                    "hole_count": hole_count,
                    "colors": colors,
                    "color_counts": Counter(int(x) for x in patch[mask]),
                }
            )
    frames.sort(key=lambda d: d["bbox"])
    tools.sort(key=lambda d: d["bbox"])
    return frames, tools


def _tool_variants(tool: dict, bg: int) -> list[dict]:
    patch = tool["patch"]
    mask = tool["mask"]
    holes = tool["hole_mask"]
    out: list[dict] = []
    seen: set[tuple[bytes, bytes, bytes]] = set()
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        h = np.rot90(holes, k)
        for flip in (False, True):
            q = np.fliplr(p) if flip else p.copy()
            qm = np.fliplr(m) if flip else m.copy()
            qh = np.fliplr(h) if flip else h.copy()
            key = (q.tobytes(), qm.tobytes(), qh.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append({"patch": q, "mask": qm, "hole_mask": qh, "is_flip": bool(flip)})
    return out


def _choose_frame_assignment(frames: list[dict], tools: list[dict]) -> list[int]:
    if len(frames) != len(tools):
        raise ValueError("tool/frame count mismatch")
    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(frames))):
        ok = True
        dist_sum = 0
        for ti, fi in enumerate(perm):
            if int(tools[ti]["hole_count"]) != int(frames[fi]["branch_count"]):
                ok = False
                break
            dist_sum += _bbox_gap(tools[ti]["bbox"], frames[fi]["bbox"])
        if not ok:
            continue
        score = (-int(dist_sum),)
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    if best_perm is None:
        raise ValueError("no valid tool-frame assignment by hole/branch count")
    return [int(x) for x in best_perm]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock hollow two-color tools into border sockets by matching hole count to socket branch count.

    Input structure:
    - The background is the most frequent color.
    - Border-touching monochrome components are anchor frames; each frame bbox contains one bg socket whose outer silhouette is a thick T or a thick cross.
    - Non-border components are movable 2-color tools; each tool contains 3 or 4 enclosed bg holes inside the plug region.

    Rule:
    - Classify each frame socket as T or cross, giving it branch count 3 or 4.
    - Pair each tool to the frame whose branch count equals the tool's internal-hole count.
    - Rotate/flip the tool so one color region together with its preserved holes matches the socket exactly; place that plug into the socket and leave the holes as background while the other color protrudes outward as the handle.
    - If the winning docking transform is a reflection and the two color regions have unequal sizes, swap the two tool colors before placing it.

    Key insight:
    - The match key is topological, not areal: tool hole count identifies the target socket, and exact docking still requires a hollow T/cross plug whose hole pattern is carried into the frame.
    """
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    H, W = g.shape
    frames, tools = _component_info(g, bg)
    if not frames or not tools:
        raise ValueError("expected at least one frame and one tool")

    out = np.full_like(g, int(bg))
    occupied = np.zeros((H, W), dtype=bool)
    for fr in frames:
        for r, c in fr["mask_abs"]:
            out[r, c] = int(fr["color"])
            occupied[r, c] = True

    assignment = _choose_frame_assignment(frames, tools)
    for tool, frame_idx in zip(tools, assignment):
        frame = frames[frame_idx]
        best = None
        best_score = None
        cavity_abs = set(frame["cavity_abs"])
        cavity_bbox = frame["cavity_bbox"]
        frame_bbox = frame["bbox"]
        for var in _tool_variants(tool, bg):
            patch = var["patch"]
            mask = var["mask"]
            hole_mask = var["hole_mask"]
            colors = sorted({int(x) for x in patch[mask]})
            for plug_color in colors:
                plug_mask = (patch == int(plug_color)) & mask
                if plug_mask.sum() + hole_mask.sum() != len(cavity_abs):
                    continue
                outer = plug_mask | hole_mask
                if _coords_key({(int(r), int(c)) for r, c in np.argwhere(outer)}) != frame["cavity_norm"]:
                    continue
                outer_bbox = _bbox_from_mask(outer)
                r0 = int(cavity_bbox[0] - outer_bbox[0])
                c0 = int(cavity_bbox[1] - outer_bbox[1])
                h, w = patch.shape
                if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
                    continue
                translated_outer = {
                    (int(r0 + r), int(c0 + c))
                    for r, c in np.argwhere(outer)
                }
                if translated_outer != cavity_abs:
                    continue
                translated_mask = {
                    (int(r0 + r), int(c0 + c))
                    for r, c in np.argwhere(mask)
                }
                if any(occupied[r, c] for r, c in translated_mask):
                    continue
                handle_coords = {
                    (int(r0 + r), int(c0 + c))
                    for r, c in np.argwhere(mask & (patch != int(plug_color)))
                }
                outside = 0
                for r, c in handle_coords:
                    if frame["open_dir"] == "right" and c > frame_bbox[3]:
                        outside += 1
                    elif frame["open_dir"] == "left" and c < frame_bbox[1]:
                        outside += 1
                    elif frame["open_dir"] == "down" and r > frame_bbox[2]:
                        outside += 1
                    elif frame["open_dir"] == "up" and r < frame_bbox[0]:
                        outside += 1
                if outside == 0:
                    continue
                score = (int(outside), int(len(handle_coords)), int(var["is_flip"]))
                if best_score is None or score > best_score:
                    best_score = score
                    best = (patch.copy(), mask.copy(), r0, c0, bool(var["is_flip"]))
        if best is None:
            raise ValueError("failed to dock tool")
        patch, mask, r0, c0, is_flip = best
        if len(tool["color_counts"]) == 2 and len(set(tool["color_counts"].values())) == 2 and is_flip:
            a, b = sorted(int(x) for x in tool["color_counts"])
            patch = _swap_colors(patch, a, b)
        paste(out, patch, r0, c0, transparent=bg)
        occupied[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] |= mask
    return out


def _tool_place_candidates(
    patch: np.ndarray,
    bg: int,
    input_grid: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    tool_bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    H, W = input_grid.shape
    h, w = patch.shape
    mask = patch != int(bg)
    occ = input_grid != int(bg)
    candidates: list[tuple[int, int]] = []
    for r0 in range(TOOL_BORDER, H - TOOL_BORDER - h + 1):
        for c0 in range(TOOL_BORDER, W - TOOL_BORDER - w + 1):
            if np.any(occ[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            if any(_bbox_gap(bbox, fb) < TOOL_FRAME_GAP for fb in frame_bboxes):
                continue
            if any(_bbox_gap(bbox, tb) < TOOL_TOOL_GAP for tb in tool_bboxes):
                continue
            candidates.append((int(r0), int(c0)))
    random.shuffle(candidates)
    return candidates


def _undock_tool(
    *,
    info: dict,
    bg: int,
    input_grid: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    tool_bboxes: list[tuple[int, int, int, int]],
    require_flip: bool,
) -> bool:
    variants = _variant_choices(info["tool_patch"], bg)
    if require_flip:
        variants = [x for x in variants if x[1]]
        if not variants:
            return False
    for patch, is_flip in variants:
        q = patch.copy()
        if is_flip and int(info["plug_area"]) != int(info["handle_area"]):
            q = _swap_colors(q, info["plug_color"], info["handle_color"])
        for r0, c0 in _tool_place_candidates(q, bg, input_grid, frame_bboxes, tool_bboxes):
            paste(input_grid, q, r0, c0, transparent=bg)
            tool_bboxes.append((int(r0), int(c0), int(r0 + q.shape[0] - 1), int(c0 + q.shape[1] - 1)))
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randrange(10)
    num_pairs = random.choice(NUM_PAIRS_CHOICES)
    sides = random.sample(SIDES, k=num_pairs)
    if num_pairs == 2:
        shape_kinds = random.sample(["T", "PLUS"], k=2)
    else:
        shape_kinds = [random.choice(["T", "PLUS"])]

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    output = np.full((H, W), int(bg), dtype=int)
    input_grid = np.full((H, W), int(bg), dtype=int)
    scene_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    tools_info: list[dict] = []

    for i in range(num_pairs):
        frame_color = colors[3 * i]
        plug_color = colors[3 * i + 1]
        handle_color = colors[3 * i + 2]
        pair = _build_pair_geometry(
            bg=bg,
            frame_color=frame_color,
            plug_color=plug_color,
            handle_color=handle_color,
            kind=shape_kinds[i],
            side=sides[i],
        )
        placed = _place_scene(
            side=sides[i],
            scene_patch=pair["scene_patch"],
            frame_patch=pair["frame_patch"],
            frame_bbox=pair["frame_bbox"],
            H=H,
            W=W,
            output=output,
            input_grid=input_grid,
            scene_bboxes=scene_bboxes,
            frame_bboxes=frame_bboxes,
            bg=bg,
        )
        if placed is None:
            return None
        pair["tool_patch"] = _crop_non_bg(pair["tool_patch"], bg)
        tools_info.append(pair)

    tool_bboxes: list[tuple[int, int, int, int]] = []
    for i, info in enumerate(tools_info):
        ok = _undock_tool(
            info=info,
            bg=bg,
            input_grid=input_grid,
            frame_bboxes=frame_bboxes,
            tool_bboxes=tool_bboxes,
            require_flip=True,
        )
        if not ok:
            return None

    try:
        solved = main(input_grid)
    except Exception:
        return None
    if np.array_equal(solved, output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
