# Combo variant of a25697e4 (index 27)
# combo_spec: D12/0+D13/0
# Name: 包围结构角色互换
#
# Design analysis
# a) Concrete behavior and trace:
#    Original a25697e4 pairs each movable 2-color tool to a border frame by the
#    area of the color-region that will be inserted, and that region itself is the
#    exposed "plug". This combo changes both the topology and the matching target.
#    Small fused trace:
#    - Frame A touches the left border and contains one empty socket chamber shaped
#      like a 3x4 rectangle plus one extra staircase cell on the open side. The
#      whole empty chamber has area 13, but its 4-neighbor interior core has area
#      2: a 1x2 slit.
#    - Tool A is one 2-color component whose outer color is the chamber boundary
#      ring (11 cells) and whose inner color is exactly that 1x2 core (2 cells).
#      The outer ring fully surrounds the inner bar.
#    - Frame B analogously has chamber area 16 but interior-core area 3, matched by
#      a second tool whose enclosing shell has 13 cells around a 1x3 inner bar.
#    - Fused logic must:
#      1. detect border-touching frame components and recover each full empty socket,
#      2. erode that socket by one 4-neighbor layer to recover the true cavity core,
#      3. detect movable nested tools whose mask equals a socket silhouette while
#         one color sits exactly on the tool's interior core,
#      4. pair tool-to-frame by `tool_core_area == frame_core_area`, not by the
#         full socket area,
#      5. rotate/flip the whole nested tool so the inner color lands in the core
#         cavity and the surrounding outer color occupies the socket rim,
#      6. if the winning docking transform is a reflection and the two color-region
#         sizes differ, swap the two colors.
#    The visual difference from the original is strong: the matched area is no
#    longer the whole visible empty socket, and the inserted color is the enclosed
#    inner core rather than the enclosing shell.
#
# b) Complexity calibration:
#    The generator uses only two handle lengths, 2 and 3, and always emits two
#    pairs so that samples visibly repeat the same fused mechanism. Whole socket
#    areas become 13 and 16 while core areas become 2 and 3, so the original
#    "largest empty chamber area == one tool color count" shortcut is explicitly
#    disabled without introducing many unrelated families.
#
# c) Hidden assumptions broken from the original main():
#    - "largest cavity area equals one tool color count" is false here by
#      construction: chamber areas are 13/16 while tool color counts are 11/13 and
#      2/3.
#    - The inserted color is not the exposed shell; it is the enclosed inner core.
#    - Matching by any color count alone is insufficient unless the solver first
#      identifies which pixels are the topological core of the socket and the tool.
#    - Reflection parity now matters on an asymmetric nested silhouette, so the
#      solver cannot ignore flip-vs-rotation equivalence.
#    The rewritten solver therefore extracts socket chambers, computes their
#    4-neighbor interior cores, extracts the analogous interior of each tool mask,
#    and only then searches D4 variants.
#
# d) Placement legality:
#    Generator builds each docked output first, using a single border-touching frame
#    component plus a disjoint full-socket tool patch. Frames stay in-bounds, each
#    touches exactly one border, pair scenes do not overlap, and detached tools are
#    later placed on background-only interior locations with padding.
#
# e) Visual observability:
#    A human can see two repeated motifs: detached tools are nested shells around
#    short bars, while border frames expose much larger empty chambers whose inner
#    slits are smaller than the chambers themselves. In the outputs those inner bars
#    land in the middle and the shells sit around them, making the role swap
#    directly visible.
#
# f) Visual cleanliness:
#    Every frame is a single monochrome bracket, every tool is one detached 2-color
#    component, and each pair uses only three colors. No legends, hidden numeric
#    ordering, or overlapping decorations are introduced, so the boards stay easy to
#    parse.
#
# g) Color inference:
#    Color swapping is still structurally indicated by reflection, not by color-id
#    magnitude. The asymmetric staircase bump on the socket silhouette makes flips
#    geometrically observable, and unequal region sizes ensure the swap affects a
#    visible large shell versus small core.
#
# h) Quantitative anti-degeneration target:
#    Valid samples must satisfy all of:
#    - every frame socket area is strictly larger than both of that pair's tool
#      color-region areas,
#    - every frame core area equals exactly the enclosed tool-core area,
#    - every tool has an enclosing outer color and an enclosed inner color,
#    - at least one detached tool uses a reflected transform in a substantial share
#      of generated samples.
#    This prevents collapse back to the original whole-cavity area rule.
#
# i) New solver assumptions and generator guarantees:
#    - Solver assumes each border-touching foreground component is one frame and
#      each non-border foreground component is one tool; generator emits exactly two
#      of each with ample spacing.
#    - Solver assumes each frame bbox contains exactly one intended empty socket
#      chamber; generator uses one fixed bracket template per pair.
#    - Solver assumes the interior of the tool mask and the interior of the socket
#      chamber are both non-empty and unique; generator uses only the 2-cell and
#      3-cell core families.

from __future__ import annotations

from collections import Counter
import itertools
import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, get_bg, paste, random_free_location


SIDES = ["left", "right", "top", "bottom"]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [2]
HANDLE_LENGTHS = [2, 3]

FRAME_GAP_MIN = 4
SCENE_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4
GEN_RETRIES = 400


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(a[0], a[2], b[0], b[2]) + _gap1(a[1], a[3], b[1], b[3])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return out


def _orient_left_to_side(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, 3)
    if side == "bottom":
        return np.rot90(arr, 1)
    raise ValueError(side)


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _mask_interior(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        ok = True
        for dr, dc in DIR4:
            r2 = rr + dr
            c2 = cc + dc
            if not (0 <= r2 < h and 0 <= c2 < w and mask[r2, c2]):
                ok = False
                break
        if ok:
            interior[rr, cc] = True
    return interior


def _base_socket_mask(handle_len: int) -> np.ndarray:
    mask = np.zeros((3, handle_len + 3), dtype=bool)
    mask[:, : handle_len + 2] = True
    mask[0, handle_len + 2] = True
    return mask


def _build_left_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    handle_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    socket = _base_socket_mask(handle_len)
    core = _mask_interior(socket)

    tool = np.full(socket.shape, int(bg), dtype=int)
    tool[socket & (~core)] = int(plug_color)
    tool[core] = int(handle_color)

    frame = np.full((5, handle_len + 5), int(bg), dtype=int)
    frame[0, :] = int(frame_color)
    frame[-1, :] = int(frame_color)
    frame[:, :2] = int(frame_color)
    # Keep the socket asymmetric: only the top-right cell of the open side remains
    # empty, while the lower two cells become part of the frame wall.
    frame[2:, -1] = int(frame_color)
    return frame, tool


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        origins = [(r0, 0) for r0 in range(H - ph + 1)]
    elif side == "right":
        origins = [(r0, W - pw) for r0 in range(H - ph + 1)]
    elif side == "top":
        origins = [(0, c0) for c0 in range(W - pw + 1)]
    elif side == "bottom":
        origins = [(H - ph, c0) for c0 in range(W - pw + 1)]
    else:
        raise ValueError(side)
    random.shuffle(origins)
    return origins


def _frame_socket_mask(frame_patch: np.ndarray, bg: int) -> np.ndarray:
    frame_mask = frame_patch != int(bg)
    bg_mask = frame_patch == int(bg)
    lab, n = label(bg_mask, structure=STRUCT4)
    best_mask = None
    best_area = -1
    for cid in range(1, n + 1):
        comp = lab == cid
        if not comp.any():
            continue
        adj = False
        for r, c in np.argwhere(comp):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR4:
                r2 = rr + dr
                c2 = cc + dc
                if 0 <= r2 < frame_patch.shape[0] and 0 <= c2 < frame_patch.shape[1] and frame_mask[r2, c2]:
                    adj = True
                    break
            if adj:
                break
        if not adj:
            continue
        area = int(comp.sum())
        if area > best_area:
            best_area = area
            best_mask = comp
    assert best_mask is not None
    return best_mask


def _analyze_nested_tool(patch: np.ndarray, bg: int) -> dict:
    mask = patch != int(bg)
    core_mask = _mask_interior(mask)
    assert core_mask.any()
    shell_mask = mask & (~core_mask)
    assert shell_mask.any()

    core_colors = sorted({int(x) for x in patch[core_mask].flatten()})
    shell_colors = sorted({int(x) for x in patch[shell_mask].flatten()})
    assert len(core_colors) == 1
    assert len(shell_colors) == 1
    assert core_colors[0] != shell_colors[0]
    assert int(core_mask.sum()) < int(mask.sum())

    _core_lab, core_n = label(core_mask, structure=STRUCT4)
    _shell_lab, shell_n = label(shell_mask, structure=STRUCT4)
    assert core_n == 1
    assert shell_n == 1

    return {
        "mask": mask,
        "core_mask": core_mask,
        "shell_mask": shell_mask,
        "inner_color": int(core_colors[0]),
        "outer_color": int(shell_colors[0]),
        "inner_count": int(core_mask.sum()),
        "outer_count": int(shell_mask.sum()),
    }


def _gen_variants(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    mask = patch != int(bg)
    raw: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        raw.append((p, m, False))
        raw.append((np.fliplr(p), np.fliplr(m), True))
    out: list[tuple[np.ndarray, np.ndarray, bool]] = []
    seen = set()
    for p, m, is_flip in raw:
        pc, mc, _ = _crop_to_mask(p, m)
        key = (mc.tobytes(), pc[mc].astype(np.int16).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((pc, mc, bool(is_flip)))
    return out


def _tool_variant_choices(tool_patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    variants = [(p, is_flip) for p, _m, is_flip in _gen_variants(tool_patch, bg)]
    random.shuffle(variants)
    return variants


def _place_pair(
    *,
    spec: dict,
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    scene_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> dict | None:
    frame_left, tool_left = _build_left_pair(
        bg=bg,
        frame_color=spec["frame_color"],
        plug_color=spec["plug_color"],
        handle_color=spec["handle_color"],
        handle_len=spec["handle_len"],
    )
    tool_full_left = np.full_like(frame_left, int(bg))
    paste(tool_full_left, tool_left, 1, 2, transparent=int(bg))
    scene_left = np.where(tool_full_left != int(bg), tool_full_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)

    ph, pw = scene.shape
    frame_mask = frame_scene != int(bg)
    fr0, fc0, fr1, fc1 = _bbox_from_mask(frame_mask)
    scene_bbox_local = _bbox_from_mask(scene != int(bg))

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        scene_bbox = (
            int(r0 + scene_bbox_local[0]),
            int(c0 + scene_bbox_local[1]),
            int(r0 + scene_bbox_local[2]),
            int(c0 + scene_bbox_local[3]),
        )
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(scene_bbox, prev) < SCENE_GAP_MIN for prev in scene_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_GAP_MIN for prev in frame_bboxes):
            continue
        if np.any((scene != int(bg)) & (output[r0 : r0 + ph, c0 : c0 + pw] != int(bg))):
            continue
        origin = (int(r0), int(c0))
        scene_bboxes.append(scene_bbox)
        frame_bboxes.append(frame_bbox)
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=int(bg))
    paste(input_grid, frame_scene, r0, c0, transparent=int(bg))

    tool_vals = [int(x) for x in tool[tool != int(bg)]]
    counts = Counter(tool_vals)
    plug_count = int(counts[spec["plug_color"]])
    handle_count = int(counts[spec["handle_color"]])
    return {
        "tool": tool,
        "plug_color": int(spec["plug_color"]),
        "handle_color": int(spec["handle_color"]),
        "plug_count": plug_count,
        "handle_count": handle_count,
    }


def _undock_tool(
    info: dict,
    bg: int,
    input_grid: np.ndarray,
    placement_canvas: np.ndarray,
    prefer_flip: bool | None = None,
) -> bool | None:
    placed_reflected = False
    variants = _tool_variant_choices(info["tool"], bg)
    if prefer_flip is not None:
        preferred = [item for item in variants if item[1] == prefer_flip]
        fallback = [item for item in variants if item[1] != prefer_flip]
        variants = preferred + fallback
    for patch, is_flip in variants:
        q = patch.copy()
        if is_flip and info["plug_count"] != info["handle_count"]:
            q = _swap_colors(q, info["plug_color"], info["handle_color"])
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                q,
                transparent=int(bg),
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, q, r0, c0, transparent=int(bg))
        paste(placement_canvas, q, r0, c0, transparent=int(bg))
        placed_reflected = bool(is_flip)
        return placed_reflected
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock nested shell-and-core tools back into border sockets.

    Input structure:
    - The background is the most frequent color.
    - Border-touching monochrome components are anchor frames; each frame bbox
      contains one larger empty socket chamber whose one-cell-thick interior core is
      a much smaller slit.
    - Non-border components are movable 2-color tools whose outer color surrounds a
      smaller enclosed inner color.

    Rule:
    - Pair each tool with the frame whose socket-core area matches the tool's
      enclosed inner-core area, not the whole empty socket area.
    - Rotate/flip the whole nested tool so its full silhouette fills the socket
      chamber, with the inner color landing in the socket core and the outer color
      wrapping around it.
    - If the winning docking transform uses a reflection and the two color-region
      sizes differ, swap the tool's two colors before placing it.

    Key insight: the true match key is the 4-neighbor interior core of each socket
    and tool mask; the visible empty socket is larger than the matched core because
    the enclosing shell occupies the surrounding rim.
    """
    g = grid.copy()
    bg = int(get_bg(g))
    H, W = g.shape

    fg = g != bg
    lab, n = label(fg, structure=STRUCT4)
    frames = []
    tools = []

    for cid in range(1, n + 1):
        comp = lab == cid
        if not comp.any():
            continue
        r0, c0, r1, c1 = _bbox_from_mask(comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = comp[r0 : r1 + 1, c0 : c1 + 1]
        patch[~mask] = bg
        touches_border = r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1
        item = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "patch": patch,
            "mask": mask.copy(),
            "size": int(mask.sum()),
        }
        if touches_border:
            frames.append(item)
        else:
            tools.append(item)

    assert frames and tools
    assert len(frames) == len(tools)

    parsed_frames = []
    for frame in frames:
        bbox = frame["bbox"]
        patch = frame["patch"]
        socket_mask = _frame_socket_mask(patch, bg)
        core_mask = _mask_interior(socket_mask)
        assert core_mask.any()
        assert int(core_mask.sum()) < int(socket_mask.sum())
        _, socket_crop, (sr0, sc0) = _crop_to_mask(socket_mask.astype(np.uint8), socket_mask)
        socket_crop = socket_crop.astype(bool)
        core_crop = core_mask[sr0 : sr0 + socket_crop.shape[0], sc0 : sc0 + socket_crop.shape[1]].copy()
        side_hits = []
        if bbox[1] == 0:
            side_hits.append("left")
        if bbox[3] == W - 1:
            side_hits.append("right")
        if bbox[0] == 0:
            side_hits.append("top")
        if bbox[2] == H - 1:
            side_hits.append("bottom")
        assert len(side_hits) == 1
        parsed_frames.append(
            {
                "bbox": bbox,
                "patch": patch,
                "socket_mask": socket_crop,
                "core_mask": core_crop,
                "core_area": int(core_crop.sum()),
                "socket_origin": (int(bbox[0] + sr0), int(bbox[1] + sc0)),
                "size": frame["size"],
                "side": side_hits[0],
            }
        )

    parsed_tools = []
    for tool in tools:
        patch = tool["patch"]
        nested = _analyze_nested_tool(patch, bg)
        colors = sorted([nested["inner_color"], nested["outer_color"]])
        counts = Counter(int(x) for x in patch[nested["mask"]].flatten())
        variants = _gen_variants(patch, bg)
        parsed_tools.append(
            {
                "bbox": tool["bbox"],
                "patch": patch,
                "mask": nested["mask"],
                "core_area": nested["inner_count"],
                "inner_color": nested["inner_color"],
                "outer_color": nested["outer_color"],
                "inner_count": nested["inner_count"],
                "outer_count": nested["outer_count"],
                "colors": colors,
                "counts": counts,
                "variants": variants,
            }
        )

    def compatible(ti: int, fi: int) -> bool:
        tool = parsed_tools[ti]
        frame = parsed_frames[fi]
        if tool["inner_count"] != frame["core_area"]:
            return False
        if tool["outer_count"] <= tool["inner_count"]:
            return False
        for p, m, _is_flip in tool["variants"]:
            if m.shape == frame["socket_mask"].shape and np.array_equal(m, frame["socket_mask"]):
                return True
        return False

    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(parsed_frames))):
        ok = True
        dist = 0
        for ti, fi in enumerate(perm):
            if not compatible(ti, fi):
                ok = False
                break
            dist += _bbox_gap(parsed_tools[ti]["bbox"], parsed_frames[fi]["bbox"])
        if not ok:
            continue
        score = -int(dist)
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None

    out = np.full_like(g, bg)
    for frame in parsed_frames:
        paste(out, frame["patch"], frame["bbox"][0], frame["bbox"][1], transparent=bg)

    for ti, fi in enumerate(best_perm):
        tool = parsed_tools[ti]
        frame = parsed_frames[fi]
        chosen = None
        for p, m, is_flip in tool["variants"]:
            if m.shape == frame["socket_mask"].shape and np.array_equal(m, frame["socket_mask"]):
                chosen = (p.copy(), bool(is_flip))
                break
        assert chosen is not None
        docked, is_flip = chosen
        a, b = tool["colors"]
        if is_flip and tool["counts"][a] != tool["counts"][b]:
            docked = _swap_colors(docked, a, b)
        paste(out, docked, frame["socket_origin"][0], frame["socket_origin"][1], transparent=bg)

    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    if H < 18 or W < 25:
        return None

    lengths = HANDLE_LENGTHS[:]
    random.shuffle(lengths)
    lengths = lengths[:num_pairs]

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    sides = random.sample(SIDES, k=num_pairs)
    output = np.full((H, W), int(bg), dtype=int)
    input_grid = np.full((H, W), int(bg), dtype=int)
    scene_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    tools_info: list[dict] = []

    for i in range(num_pairs):
        spec = {
            "handle_len": int(lengths[i]),
            "frame_color": int(colors[3 * i]),
            "plug_color": int(colors[3 * i + 1]),
            "handle_color": int(colors[3 * i + 2]),
        }
        info = _place_pair(
            spec=spec,
            side=sides[i],
            bg=bg,
            H=H,
            W=W,
            output=output,
            input_grid=input_grid,
            scene_bboxes=scene_bboxes,
            frame_bboxes=frame_bboxes,
        )
        if info is None:
            return None
        tools_info.append(info)

    placement_canvas = output.copy()
    reflected_flags = []
    desired_flips = [True, False]
    random.shuffle(desired_flips)
    for info, prefer_flip in zip(tools_info, desired_flips):
        placed = _undock_tool(info, bg, input_grid, placement_canvas, prefer_flip=prefer_flip)
        if placed is None:
            return None
        reflected_flags.append(bool(placed))

    if np.array_equal(main(input_grid), output):
        if sorted(reflected_flags) != [False, True]:
            return None
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
