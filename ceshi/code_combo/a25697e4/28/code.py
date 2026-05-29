# Combo variant of a25697e4 (index 28)
# combo_spec: D1/3+D4/3
# Name: 浮空框架连接器匹配
#
# Design analysis:
# - Concrete trace:
#   Start from the original left-opening frame: a top bar, a bottom bar with a staircase notch,
#   and a plug-shaped cavity between them. The original task paired tools to frames by cavity area,
#   assuming frames touched the outer border. In this combo, the whole frame floats inside the grid,
#   so border contact is gone. Each frame now carries one visible connector pixel on a cavity wall,
#   and the tool carries a matching connector pixel on its plug edge. The tool still fills the cavity
#   exactly, but the pair is chosen by connector color plus connector side rather than by area.
# - Complexity calibration:
#   To make connector matching genuinely necessary, the generator always creates two pairs with the
#   same cavity geometry and therefore the same plug area. If we left areas distinct, the old rule
#   would still solve many examples. Shared geometry forces the new visual cue to matter.
# - Hidden assumptions broken from the original solver:
#   1. "Border-touching components are frames" is invalid because every frame is internal here.
#   2. "Area equality identifies the destination" is invalid because both frames expose the same
#      cavity area and same cavity shape.
#   3. "Opening direction comes from border side" is invalid because the frame orientation must now
#      be inferred from the cavity opening inside the frame bbox.
#   4. Tool color logic must tolerate a third connector color while preserving the original
#      flip-triggered plug/handle swap for the two large regions only.
# - Placement legality:
#   The generator builds the docked output first, places both floating scenes with interior margins
#   and bbox gaps, then uses that finished output as an occupancy canvas when scattering source tools.
#   That guarantees in-bounds placement and prevents tools from colliding with frames or with each
#   other's destination regions.
# - Visual observability:
#   Two frames have the same cavity geometry, so shape and area alone do not explain the mapping.
#   The only reliable cue is the tiny connector marker: same color, same side on the plug/frame pair.
#   The source tools are also placed closer to the wrong frame than the correct one, so distance is
#   intentionally misleading rather than explanatory.
# - Visual cleanliness:
#   Each pair uses four distinct non-background colors: frame, plug, handle, connector. The marker is
#   only one pixel per object, so it is visible without overwhelming the frame/tool silhouettes.
# - New solver assumptions and how the generator guarantees them:
#   1. Every frame object is exactly two nearby 4-connected components; each tool is one component.
#   2. Every frame has exactly one connector pixel and one intended cavity component.
#   3. Every tool has exactly one connector pixel whose exposed side is unique.
#   4. The plug occupancy is "connector pixel + one major color region"; the handle is the other.
#   5. For each tool there is exactly one frame with matching connector color; connector side becomes
#      the pose constraint after the solver tries D4 transforms of that tool.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_sprite


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_gap(a, b) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def gap1(x0, x1, y0, y1):
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


def _bbox_of_pts(pts: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return out


def _orient_open_right_to(arr: np.ndarray, open_side: str) -> np.ndarray:
    if open_side == "right":
        return arr.copy()
    if open_side == "left":
        return np.fliplr(arr)
    if open_side == "up":
        return np.rot90(arr, 1)
    if open_side == "down":
        return np.rot90(arr, 3)
    raise ValueError(open_side)


def _swap_two_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _gen_variants(patch: np.ndarray, mask: np.ndarray):
    base = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        base.append((p, m, False))
        base.append((np.fliplr(p), np.fliplr(m), True))
    uniq = []
    seen = set()
    for p, m, is_flip in base:
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m, is_flip))
    return uniq


def _largest_adjacent_cavity(mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    sub_lab, sub_n = label(bg_mask, structure=STRUCT4)
    best_id = None
    best_key = None
    for cid in range(1, int(sub_n) + 1):
        pts = np.argwhere(sub_lab == int(cid))
        area = int(len(pts))
        adj = False
        for rr, cc in pts:
            for dr, dc in DIRS4:
                r2 = int(rr + dr)
                c2 = int(cc + dc)
                if 0 <= r2 < mask.shape[0] and 0 <= c2 < mask.shape[1] and mask[r2, c2]:
                    adj = True
                    break
            if adj:
                break
        if not adj:
            continue
        key = (area, -int(cid))
        if best_key is None or key > best_key:
            best_key = key
            best_id = int(cid)
    assert best_id is not None, "frame must have one intended cavity"
    return sub_lab == int(best_id)


def _open_side_from_cavity(cavity: np.ndarray) -> tuple[int, int]:
    h, w = cavity.shape
    counts = {
        (-1, 0): int(cavity[0, :].sum()),
        (1, 0): int(cavity[h - 1, :].sum()),
        (0, -1): int(cavity[:, 0].sum()),
        (0, 1): int(cavity[:, w - 1].sum()),
    }
    best_dir = max(counts, key=lambda d: counts[d])
    assert counts[best_dir] > 0, "cavity must touch its bbox on the opening side"
    return best_dir


def _tool_connector_direction(mask: np.ndarray, conn_mask: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(conn_mask)
    assert len(pts) == 1, "tool connector must be a single pixel"
    r, c = map(int, pts[0])
    exposed = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if not (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]):
            exposed.append((dr, dc))
    assert len(exposed) == 1, "tool connector side must be visually unique"
    return exposed[0]


def _frame_connector_direction(mask: np.ndarray, cavity: np.ndarray, conn_mask: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(conn_mask)
    assert len(pts) == 1, "frame connector must be a single pixel"
    r, c = map(int, pts[0])
    dirs = []
    for dr, dc in DIRS4:
        rr = int(r - dr)
        cc = int(c - dc)
        if 0 <= rr < cavity.shape[0] and 0 <= cc < cavity.shape[1] and cavity[rr, cc]:
            dirs.append((dr, dc))
    assert len(dirs) == 1, "frame connector must mark one cavity wall"
    return dirs[0]


def _normalize_shape(mask: np.ndarray) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    shp = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shp, (r0, c0)


def _shift_for_exact_match(local_mask: np.ndarray, target_global_mask: np.ndarray) -> tuple[int, int] | None:
    if int(local_mask.sum()) != int(target_global_mask.sum()):
        return None
    shp_a, (ar0, ac0) = _normalize_shape(local_mask)
    shp_b, (br0, bc0) = _normalize_shape(target_global_mask)
    if shp_a != shp_b:
        return None
    return int(br0 - ar0), int(bc0 - ac0)


def _analyze_grid(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    fg = g != bg
    lab, n = label(fg, structure=STRUCT4)

    raw_components = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        raw_components.append({"pts": pts, "bbox": _bbox_of_pts(pts)})

    parent = list(range(len(raw_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(raw_components)):
        for j in range(i + 1, len(raw_components)):
            if _bbox_gap(raw_components[i]["bbox"], raw_components[j]["bbox"]) <= 2:
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for i in range(len(raw_components)):
        grouped.setdefault(find(i), []).append(i)

    frames = []
    tools = []
    for idxs in grouped.values():
        pts = np.concatenate([raw_components[i]["pts"] for i in idxs], axis=0)
        bbox = _bbox_of_pts(pts)
        r0, c0, r1, c1 = bbox
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[int(r - r0), int(c - c0)] = True
        colors = [int(x) for x in sub[mask].flatten()]
        counts = Counter(colors)

        if len(idxs) == 2:
            assert len(counts) == 2, "frame should contain frame color + connector color"
            frame_color, _ = max(counts.items(), key=lambda kv: kv[1])
            connector_colors = [col for col in counts if int(col) != int(frame_color)]
            assert len(connector_colors) == 1
            conn_color = int(connector_colors[0])
            conn_mask = (sub == int(conn_color)) & mask
            cavity = _largest_adjacent_cavity(mask, ~mask)
            frames.append(
                {
                    "bbox": bbox,
                    "mask_local": mask,
                    "mask_global": _embed_mask(mask, bbox, (H, W)),
                    "subgrid": sub,
                    "frame_color": int(frame_color),
                    "connector_color": conn_color,
                    "connector_mask_local": conn_mask,
                    "connector_dir": _frame_connector_direction(mask, cavity, conn_mask),
                    "cavity_local": cavity,
                    "cavity_global": _embed_mask(cavity, bbox, (H, W)),
                    "open_dir": _open_side_from_cavity(cavity),
                    "connector_pos_global": (
                        int(r0 + np.argwhere(conn_mask)[0, 0]),
                        int(c0 + np.argwhere(conn_mask)[0, 1]),
                    ),
                }
            )
        else:
            assert len(idxs) == 1, "generator guarantees tools are single connected components"
            assert len(counts) == 3, "tool should contain plug, handle, connector colors"
            conn_color, conn_count = min(counts.items(), key=lambda kv: (kv[1], kv[0]))
            assert int(conn_count) == 1, "tool connector must be a singleton color"
            conn_mask = (sub == int(conn_color)) & mask
            tools.append(
                {
                    "bbox": bbox,
                    "mask": mask,
                    "patch": np.where(mask, sub, 0),
                    "connector_color": int(conn_color),
                    "connector_dir": _tool_connector_direction(mask, conn_mask),
                }
            )

    frames.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    tools.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return {"bg": bg, "shape": (H, W), "frames": frames, "tools": tools}


def _embed_mask(local_mask: np.ndarray, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    r0, c0, r1, c1 = bbox
    out = np.zeros((H, W), dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = local_mask
    return out


def _find_docking_state(
    *,
    frame: dict,
    patch: np.ndarray,
    mask: np.ndarray,
    conn_color: int,
    major_colors: list[int],
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray, int, int, bool, int] | None:
    for p, m, is_flip in _gen_variants(patch, mask):
        conn_mask = (p == int(conn_color)) & m
        if tuple(_tool_connector_direction(m, conn_mask)) != tuple(frame["connector_dir"]):
            continue
        conn_local = tuple(map(int, np.argwhere(conn_mask)[0]))
        for plug_color in major_colors:
            plug_local = (((p == int(plug_color)) | (p == int(conn_color))) & m)
            shift = _shift_for_exact_match(plug_local, frame["cavity_global"])
            if shift is None:
                continue
            dr, dc = shift
            coords = np.argwhere(m)
            ok = True
            handle_outside = 0
            for rr, cc in coords:
                r = int(rr + dr)
                c = int(cc + dc)
                if not (0 <= r < H and 0 <= c < W):
                    ok = False
                    break
                if frame["mask_global"][r, c]:
                    ok = False
                    break
                if not frame["cavity_global"][r, c]:
                    fr0, fc0, fr1, fc1 = frame["bbox"]
                    odr, odc = frame["open_dir"]
                    if odr == -1 and r < fr0:
                        handle_outside += 1
                    elif odr == 1 and r > fr1:
                        handle_outside += 1
                    elif odc == -1 and c < fc0:
                        handle_outside += 1
                    elif odc == 1 and c > fc1:
                        handle_outside += 1
            if not ok or handle_outside == 0:
                continue
            tool_conn_global = (int(conn_local[0] + dr), int(conn_local[1] + dc))
            dir_r, dir_c = frame["connector_dir"]
            expect_frame_conn = (int(tool_conn_global[0] + dir_r), int(tool_conn_global[1] + dir_c))
            if expect_frame_conn != tuple(frame["connector_pos_global"]):
                continue
            return (p.copy(), m.copy(), dr, dc, bool(is_flip), int(plug_color))
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock floating interior tools into floating frames by connector-guided cavity filling.

    Input structure:
    - `bg` is the most common color.
    - Each frame is a floating two-piece structure of one frame color plus one tiny connector color.
    - Each tool is a floating 4-connected multi-color component: plug color, handle color, and one
      connector pixel whose exposed side marks how it should dock.

    Rule:
    - Group nearby components into objects, identify frame cavities, and infer each frame's opening
      direction from the cavity rather than from the outer border.
    - Match every tool to the unique frame whose connector cue can actually be satisfied. In the
      generated tasks connector color already narrows the candidate set, and the final validation
      also enforces the connector side plus exact cavity alignment.
    - Rotate/flip the tool until its connector side agrees with that frame and `connector + one tool
      color` fills the cavity exactly; the remaining color forms the handle outside the frame. If
      the winning docking transform is a reflection and the two large tool-color counts are unequal,
      swap those two colors after docking.

    Key insight: The cavity shape still constrains the final pose, but pairing no longer comes from
    border anchoring or area equality; the decisive cue is the tiny connector marker and its side.
    """
    g = np.array(grid, dtype=int)
    analysis = _analyze_grid(g)
    bg = int(analysis["bg"])
    H, W = analysis["shape"]
    frames = analysis["frames"]
    tools = analysis["tools"]

    assert frames, "expected at least one frame"
    assert len(frames) == len(tools), "generator creates one floating tool per frame"

    out = np.full((H, W), bg, dtype=int)
    for frame in frames:
        out[frame["mask_global"]] = g[frame["mask_global"]]

    used_frames: set[int] = set()
    for tool in tools:
        patch = tool["patch"]
        mask = tool["mask"]
        conn_color = int(tool["connector_color"])
        present = sorted({int(x) for x in patch[mask].flatten() if int(x) != 0})
        major_colors = [col for col in present if int(col) != conn_color]
        assert len(major_colors) == 2
        major_counts = {col: int(((patch == int(col)) & mask).sum()) for col in major_colors}

        candidate_frames = [
            i for i, frame in enumerate(frames)
            if i not in used_frames and int(frame["connector_color"]) == conn_color
        ]
        assert candidate_frames, "every tool should see at least one connector-color candidate"
        feasible = []
        for gi in candidate_frames:
            best_state = _find_docking_state(
                frame=frames[gi],
                patch=patch,
                mask=mask,
                conn_color=conn_color,
                major_colors=major_colors,
                H=H,
                W=W,
            )
            if best_state is not None:
                feasible.append((gi, best_state))

        assert len(feasible) == 1, "connector-guided docking should identify one frame"
        gi, best_state = feasible[0]
        used_frames.add(int(gi))
        p, m, dr, dc, is_flip, _plug_color = best_state
        if is_flip and len(set(major_counts.values())) == 2:
            a, b = major_colors
            p = _swap_two_colors(p, a, b)
        for rr, cc in np.argwhere(m):
            out[int(rr + dr), int(cc + dc)] = int(p[int(rr), int(cc)])

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (28, 30)
NUM_PAIRS = 2

FRAME_W_RANGE = (5, 8)       # exclusive upper bound
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
NOTCH_CHOICES = [2, 3]

HANDLE_AREA_RANGE = (4, 9)   # exclusive upper bound
HANDLE_DIM_RANGE = (2, 5)    # exclusive upper bound
TOOL_TOTAL_BOUNDS = (10, 16)

OUTER_MARGIN = 1
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
SOURCE_FRAME_GAP_MIN = 4
GEN_ATTEMPT_RETRIES = 400
HANDLE_SPRITE_RETRIES = 40


def _notch_offsets(notch_n: int, fw: int, bot_h: int) -> list[tuple[int, int]]:
    if notch_n == 2:
        return [(0, fw - 2), (0, fw - 1)]
    if notch_n == 3:
        if bot_h >= 2:
            return [(0, fw - 2), (0, fw - 1), (1, fw - 1)]
        return [(0, fw - 3), (0, fw - 2), (0, fw - 1)]
    raise ValueError(notch_n)


def _make_handle_sprite(handle_area: int, bg: int, handle_color: int) -> np.ndarray | None:
    lo, hi = HANDLE_DIM_RANGE
    dims = [
        (h, w)
        for h in range(lo, hi)
        for w in range(lo, hi)
        if handle_area <= h * w <= handle_area * 2
    ]
    for _ in range(HANDLE_SPRITE_RETRIES):
        h, w = random.choice(dims)
        sprite = random_sprite(
            h,
            w,
            density=handle_area / (h * w),
            color_palette=[handle_color],
            connectivity=4,
            background=bg,
        )
        if int(np.sum(sprite != bg)) == handle_area:
            return sprite
    return None


def _build_open_right_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    connector_color: int,
    fw: int,
    top_h: int,
    bot_h: int,
    notch_n: int,
    handle_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = frame_color
    frame[top_h + 1 :, :] = frame_color
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = bg

    conn_c = int(fw - 2)
    frame[top_h - 1, conn_c] = connector_color

    cavity = frame == bg
    tool = np.full((fh, fw), bg, dtype=int)
    tool[cavity] = plug_color
    tool[top_h, conn_c] = connector_color

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None
    hs_h, hs_w = handle_sprite.shape
    attach_rows = {r for r in range(fh) if cavity[r, fw - 1]}
    edge_rows = {r for r in range(hs_h) if handle_sprite[r, 0] != bg}
    valid_offsets = [
        voff
        for voff in range(fh)
        if any((voff + sr) in attach_rows for sr in edge_rows)
    ]
    if not valid_offsets:
        return None
    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + hs_h)
    total_w = fw + hs_w
    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame
    tool_full = np.full((total_h, total_w), bg, dtype=int)
    tool_full[:fh, :fw] = tool
    handle_mask = handle_sprite != bg
    tool_full[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = handle_color
    return frame_full, tool_full


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    return _bbox_of_pts(pts)


def _candidate_origins_internal(ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    origins = [
        (r0, c0)
        for r0 in range(OUTER_MARGIN, H - OUTER_MARGIN - ph + 1)
        for c0 in range(OUTER_MARGIN, W - OUTER_MARGIN - pw + 1)
    ]
    random.shuffle(origins)
    return origins


def _place_docked_pair(
    *,
    spec: dict,
    colors: dict,
    open_side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> dict | None:
    built = _build_open_right_pair(
        bg=bg,
        frame_color=int(colors["frame"]),
        plug_color=int(colors["plug"]),
        handle_color=int(colors["handle"]),
        connector_color=int(colors["connector"]),
        fw=int(spec["fw"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
    )
    if built is None:
        return None
    frame_right, tool_right = built
    scene_right = np.where(tool_right != bg, tool_right, frame_right)

    frame = _orient_open_right_to(frame_right, open_side)
    tool = _orient_open_right_to(tool_right, open_side)
    scene = _orient_open_right_to(scene_right, open_side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _mask_bbox(frame != bg)

    origin = None
    for r0, c0 in _candidate_origins_internal(ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if frame_bbox[0] <= 0 or frame_bbox[1] <= 0 or frame_bbox[2] >= H - 1 or frame_bbox[3] >= W - 1:
            continue
        if any(_bbox_gap(union_bbox, prev) < UNION_BBOX_GAP_MIN for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_BBOX_GAP_MIN for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != bg) & (scene != bg)):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame, r0, c0, transparent=bg)
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))
    return {
        "tool": tool,
        "plug_color": int(colors["plug"]),
        "handle_color": int(colors["handle"]),
        "connector_color": int(colors["connector"]),
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
    }


def _variant_choices(patch: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    seen = set()
    out = []
    mask = patch != 0
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, flip))
    random.shuffle(out)
    return out


def _placement_candidates(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    mask = patch != bg
    out = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            if np.any((canvas[r0 : r0 + h, c0 : c0 + w] != bg) & mask):
                continue
            out.append((int(r0), int(c0)))
    random.shuffle(out)
    return out


def _place_source_tool(
    *,
    info: dict,
    bg: int,
    input_grid: np.ndarray,
    placement_canvas: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    source_bboxes: list[tuple[int, int, int, int]],
    correct_index: int,
) -> bool:
    wrong_index = 1 - int(correct_index)
    for patch, is_flip in _variant_choices(info["tool"]):
        q = patch.copy()
        if is_flip and int(info["plug_area"]) != int(info["handle_area"]):
            q = _swap_two_colors(q, int(info["plug_color"]), int(info["handle_color"]))
        h, w = q.shape
        good = []
        fallback = []
        for r0, c0 in _placement_candidates(placement_canvas, q, bg):
            bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            d_ok = True
            for fb in frame_bboxes:
                if _bbox_gap(bbox, fb) < SOURCE_FRAME_GAP_MIN:
                    d_ok = False
                    break
            if not d_ok:
                continue
            if any(_bbox_gap(bbox, prev) < FRAME_BBOX_GAP_MIN for prev in source_bboxes):
                continue
            d_correct = _bbox_gap(bbox, frame_bboxes[correct_index])
            d_wrong = _bbox_gap(bbox, frame_bboxes[wrong_index])
            if d_wrong + 1 < d_correct:
                good.append((r0, c0))
            elif d_wrong < d_correct:
                fallback.append((r0, c0))
        for bucket in (good, fallback):
            if bucket:
                r0, c0 = random.choice(bucket)
                bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
                paste(input_grid, q, r0, c0, transparent=bg)
                paste(placement_canvas, q, r0, c0, transparent=bg)
                source_bboxes.append(bbox)
                return True
    return False


def _spec_options() -> list[dict]:
    opts = []
    for fw in range(*FRAME_W_RANGE):
        for top_h in range(*TOP_H_RANGE):
            for bot_h in range(*BOT_H_RANGE):
                for notch_n in NOTCH_CHOICES:
                    plug_area = int(fw + notch_n)
                    if 5 <= plug_area <= 9 and top_h >= 1 and bot_h >= 2:
                        opts.append(
                            {
                                "fw": int(fw),
                                "top_h": int(top_h),
                                "bot_h": int(bot_h),
                                "notch_n": int(notch_n),
                                "plug_area": plug_area,
                            }
                        )
    return opts


SPEC_OPTIONS = _spec_options()


def _pick_specs() -> list[dict] | None:
    base = random.choice(SPEC_OPTIONS)
    pa = int(base["plug_area"])
    handle_choices = [
        h
        for h in range(*HANDLE_AREA_RANGE)
        if TOOL_TOTAL_BOUNDS[0] <= pa + h < TOOL_TOTAL_BOUNDS[1] and h != pa
    ]
    if len(handle_choices) < NUM_PAIRS:
        return None
    chosen = random.sample(handle_choices, k=NUM_PAIRS)
    return [{**base, "handle_area": int(chosen[i])} for i in range(NUM_PAIRS)]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))

    specs = _pick_specs()
    if specs is None:
        return None

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 4 * NUM_PAIRS:
        return None

    pair_colors = []
    for i in range(NUM_PAIRS):
        block = colors[4 * i : 4 * i + 4]
        pair_colors.append(
            {
                "frame": int(block[0]),
                "plug": int(block[1]),
                "handle": int(block[2]),
                "connector": int(block[3]),
            }
        )

    open_sides = random.sample(["right", "left", "up", "down"], k=NUM_PAIRS)
    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for i in range(NUM_PAIRS):
        info = _place_docked_pair(
            spec=specs[i],
            colors=pair_colors[i],
            open_side=open_sides[i],
            bg=bg,
            H=H,
            W=W,
            output=output,
            input_grid=input_grid,
            union_bboxes=union_bboxes,
            frame_bboxes=frame_bboxes,
        )
        if info is None:
            return None
        placed_tools.append(info)

    placement_canvas = output.copy()
    source_bboxes: list[tuple[int, int, int, int]] = []
    order = [0, 1]
    random.shuffle(order)
    for idx in order:
        if not _place_source_tool(
            info=placed_tools[idx],
            bg=bg,
            input_grid=input_grid,
            placement_canvas=placement_canvas,
            frame_bboxes=frame_bboxes,
            source_bboxes=source_bboxes,
            correct_index=idx,
        ):
            return None

    solved = main(input_grid)
    if np.array_equal(solved, output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
