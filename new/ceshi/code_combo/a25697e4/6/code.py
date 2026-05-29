# Combo variant of a25697e4 (index 6)
# combo_spec: D3/3+D4/1
# Name: 空心工具嵌入空心框架
#
# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original task: the border object is a two-piece stepped frame, and the movable tool is solid.
#      One tool color fills the frame cavity exactly by area; the other color stays outside.
#    - Fused task here: the border object is now a single hollow rectangular frame with a doorway
#      cut out on the side facing inward. The movable tool still has exactly two colors, but the
#      plug region is no longer solid: it is a rectangle with 1-2 enclosed background holes.
#    - Small example: a 3x3 cavity frame on the left border has a 1-cell doorway on its right wall.
#      The matching tool has a 3x3 plug whose center pixel is missing, plus a 3-cell handle that
#      extends to the right. In the output, the 3x3 plug sits inside the frame cavity, but the
#      center hole remains background, while the handle occupies the doorway and protrudes outside.
#    - Main difference from the original code: area equality alone is insufficient. The solver must
#      align the plug bbox to the frame cavity bbox and preserve enclosed holes.
#
# b) Complexity calibration:
#    - The frame keeps a small discrete family of cavity sizes (3x3 or 3x4), so pairing remains
#      human-inferable from 2-4 examples.
#    - The tool holes are restricted to 1-2 pixels and are always inside the plug, making the
#      topological change visually obvious without exploding the search space.
#    - Handle templates are simple and one-sided, so the task is still about docking rather than
#      free-form packing.
#
# c) Hidden-assumption audit against the original solver:
#    - Original assumption broken: "largest bg component inside frame bbox" equals the fill shape.
#      Here the true fill shape is the inner rectangle, while the doorway belongs to the handle path.
#    - Original assumption broken: plug area equals cavity area. Here plug area is cavity area minus
#      the enclosed hole count, so raw pixel counts no longer identify the match.
#    - Original assumption broken: all bg inside the tool bbox is external padding. Here some bg
#      cells are genuine enclosed holes and must survive docking.
#    - New maintained assumptions: each frame touches exactly one outer border; each frame has a
#      single doorway on the opposite side; each tool has exactly two colors; the handle lies wholly
#      on one side of the plug rectangle; the plug rectangle boundary is solid so holes are enclosed.
#
# d) Placement legality:
#    - Generator builds the output first, then removes the tool and relocates it.
#    - Each frame/scene is placed with bbox-gap constraints, and free tools are placed with border
#      and padding constraints, so there is no overlap and no border-touching tool in the input.
#
# e) Visual observability:
#    - Humans can see three coupled cues at once: a hollow border frame, a missing doorway on one
#      side, and a tool whose plug has visible internal holes.
#    - The output preserves those holes inside the frame, so the fused behavior is visible rather
#      than hidden in bookkeeping.
#
# f) Visual cleanliness:
#    - Only one layer of semantics exists in each local region: frame color, plug color, handle
#      color, and background holes. There are no extra markers or invisible ordering keys.
#
# g) Inferability of change:
#    - Color swapping still follows the original reflection rule, but the rule is anchored to clear
#      structure: the same 2-color tool is docked into a clearly matching hollow frame. Because the
#      plug and handle remain distinct geometric regions, the color behavior is not arbitrary.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location


DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _bbox_gap(a, b) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0, x1, y0, y1):
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _touches_border(bbox: tuple[int, int, int, int], H: int, W: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1


def _border_side(bbox: tuple[int, int, int, int], H: int, W: int) -> str | None:
    r0, c0, r1, c1 = bbox
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == H - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == W - 1:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


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


def _gen_variants(patch: np.ndarray, mask: np.ndarray):
    base: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        base.append((p, m, False))
        base.append((np.fliplr(p), np.fliplr(m), True))
    out: list[tuple[np.ndarray, np.ndarray, bool]] = []
    seen = set()
    for p, m, is_flip in base:
        key = (m.tobytes(), p.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((p, m, is_flip))
    return out


def _rectangle_boundary_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def _parse_frame(mask: np.ndarray, bbox: tuple[int, int, int, int], H: int, W: int):
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None
    touch = _border_side(bbox, H, W)
    if touch is None:
        return None
    open_side = OPPOSITE[touch]
    local = mask[r0 : r1 + 1, c0 : c1 + 1]
    if local.shape != (h, w):
        return None
    if np.any(local[1:-1, 1:-1]):
        return None

    if open_side == "right":
        edge = local[1:-1, -1]
        gap_idx = np.flatnonzero(~edge)
        expected = np.ones((h, w), dtype=bool)
        expected[1:-1, 1:-1] = False
        if gap_idx.size == 0 or gap_idx.size > 2 or gap_idx[-1] - gap_idx[0] + 1 != gap_idx.size:
            return None
        expected[1 + gap_idx, -1] = False
        door_cells = {(int(r0 + 1 + rr), int(c1)) for rr in gap_idx.tolist()}
    elif open_side == "left":
        edge = local[1:-1, 0]
        gap_idx = np.flatnonzero(~edge)
        expected = np.ones((h, w), dtype=bool)
        expected[1:-1, 1:-1] = False
        if gap_idx.size == 0 or gap_idx.size > 2 or gap_idx[-1] - gap_idx[0] + 1 != gap_idx.size:
            return None
        expected[1 + gap_idx, 0] = False
        door_cells = {(int(r0 + 1 + rr), int(c0)) for rr in gap_idx.tolist()}
    elif open_side == "bottom":
        edge = local[-1, 1:-1]
        gap_idx = np.flatnonzero(~edge)
        expected = np.ones((h, w), dtype=bool)
        expected[1:-1, 1:-1] = False
        if gap_idx.size == 0 or gap_idx.size > 2 or gap_idx[-1] - gap_idx[0] + 1 != gap_idx.size:
            return None
        expected[-1, 1 + gap_idx] = False
        door_cells = {(int(r1), int(c0 + 1 + cc)) for cc in gap_idx.tolist()}
    else:
        edge = local[0, 1:-1]
        gap_idx = np.flatnonzero(~edge)
        expected = np.ones((h, w), dtype=bool)
        expected[1:-1, 1:-1] = False
        if gap_idx.size == 0 or gap_idx.size > 2 or gap_idx[-1] - gap_idx[0] + 1 != gap_idx.size:
            return None
        expected[0, 1 + gap_idx] = False
        door_cells = {(int(r0), int(c0 + 1 + cc)) for cc in gap_idx.tolist()}

    if not np.array_equal(local, expected):
        return None

    cavity_bbox = (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1))
    return {
        "bbox": bbox,
        "mask": mask.copy(),
        "touch_side": touch,
        "open_side": open_side,
        "door_cells": door_cells,
        "cavity_bbox": cavity_bbox,
        "cavity_shape": (int(cavity_bbox[2] - cavity_bbox[0] + 1), int(cavity_bbox[3] - cavity_bbox[1] + 1)),
        "size": int(mask.sum()),
    }


def _find_frames(g: np.ndarray, bg: int):
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    frames = []
    fixed_mask = np.zeros((H, W), dtype=bool)
    for cid in range(1, n + 1):
        comp_mask = lab == cid
        bb = _bbox(comp_mask)
        if bb is None or not _touches_border(bb, H, W):
            continue
        vals = {int(x) for x in g[comp_mask]}
        if len(vals) != 1:
            continue
        info = _parse_frame(comp_mask, bb, H, W)
        if info is None:
            continue
        info["color"] = int(next(iter(vals)))
        frames.append(info)
        fixed_mask |= comp_mask
    frames.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
    return frames, fixed_mask


def _find_tools(g: np.ndarray, bg: int, fixed_mask: np.ndarray):
    fg = (g != int(bg)) & (~fixed_mask)
    lab, n = label(fg, structure=STRUCT4)
    tools = []
    H, W = g.shape
    for cid in range(1, n + 1):
        comp_mask = lab == cid
        bb = _bbox(comp_mask)
        assert bb is not None
        if _touches_border(bb, H, W):
            continue
        vals = sorted({int(x) for x in g[comp_mask]})
        if len(vals) != 2:
            continue
        r0, c0, r1, c1 = bb
        local_mask = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        tools.append(
            {
                "bbox": bb,
                "patch": patch,
                "mask": local_mask,
                "colors": vals,
                "counts": Counter(int(x) for x in patch[local_mask]),
                "size": int(local_mask.sum()),
            }
        )
    tools.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
    return tools


def _candidate_roles(patch: np.ndarray, mask: np.ndarray, bg: int):
    roles = []
    colors = sorted({int(x) for x in patch[mask]})
    for col in colors:
        plug = (patch == int(col)) & mask
        handle = mask & (~plug)
        if not plug.any() or not handle.any():
            continue
        pb = _bbox(plug)
        assert pb is not None
        pr0, pc0, pr1, pc1 = pb
        ph = int(pr1 - pr0 + 1)
        pw = int(pc1 - pc0 + 1)
        if ph < 3 or pw < 3:
            continue
        if handle[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
            continue
        rect = patch[pr0 : pr1 + 1, pc0 : pc1 + 1]
        rect_mask = mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
        boundary = _rectangle_boundary_mask(ph, pw)
        if not np.all(rect_mask[boundary] & (rect[boundary] == int(col))):
            continue
        hole_mask = (~rect_mask) & (rect == int(bg))
        hole_count = int(hole_mask.sum())
        if hole_count not in (1, 2):
            continue
        handle_pts = np.argwhere(handle)
        side = None
        if np.all(handle_pts[:, 1] < pc0):
            side = "left"
        elif np.all(handle_pts[:, 1] > pc1):
            side = "right"
        elif np.all(handle_pts[:, 0] < pr0):
            side = "top"
        elif np.all(handle_pts[:, 0] > pr1):
            side = "bottom"
        if side is None:
            continue
        roles.append(
            {
                "plug_color": int(col),
                "plug_bbox": (int(pr0), int(pc0), int(pr1), int(pc1)),
                "plug_mask": plug,
                "handle_mask": handle,
                "hole_count": hole_count,
                "handle_side": side,
            }
        )
    return roles


def _fit_tool_to_frame(tool: dict, frame: dict, fixed_mask: np.ndarray, bg: int):
    cav = frame["cavity_bbox"]
    ch = int(cav[2] - cav[0] + 1)
    cw = int(cav[3] - cav[1] + 1)
    door_cells = frame["door_cells"]
    best = None
    best_score = None

    for patch, mask, is_flip in _gen_variants(tool["patch"], tool["mask"]):
        roles = _candidate_roles(patch, mask, bg)
        if not roles:
            continue
        coords = np.argwhere(mask)
        for role in roles:
            pb = role["plug_bbox"]
            ph = int(pb[2] - pb[0] + 1)
            pw = int(pb[3] - pb[1] + 1)
            if (ph, pw) != (ch, cw):
                continue
            if role["handle_side"] != frame["open_side"]:
                continue
            dr = int(cav[0] - pb[0])
            dc = int(cav[1] - pb[1])

            ok = True
            door_hits = 0
            outside = 0
            adj = 0
            for rr, cc in coords:
                r = int(rr + dr)
                c = int(cc + dc)
                if not (0 <= r < fixed_mask.shape[0] and 0 <= c < fixed_mask.shape[1]):
                    ok = False
                    break
                if fixed_mask[r, c]:
                    ok = False
                    break
                if role["handle_mask"][rr, cc]:
                    if (r, c) in door_cells:
                        door_hits += 1
                    br0, bc0, br1, bc1 = frame["bbox"]
                    if frame["open_side"] == "right" and c > bc1:
                        outside += 1
                    elif frame["open_side"] == "left" and c < bc0:
                        outside += 1
                    elif frame["open_side"] == "bottom" and r > br1:
                        outside += 1
                    elif frame["open_side"] == "top" and r < br0:
                        outside += 1
            if not ok or door_hits == 0 or outside == 0:
                continue

            plug = role["plug_mask"]
            handle = role["handle_mask"]
            for rr, cc in np.argwhere(plug):
                r = int(rr + dr)
                c = int(cc + dc)
                if not (cav[0] <= r <= cav[2] and cav[1] <= c <= cav[3]):
                    ok = False
                    break
                for drr, dcc in DIRS.values():
                    r2 = r + drr
                    c2 = c + dcc
                    if 0 <= r2 < fixed_mask.shape[0] and 0 <= c2 < fixed_mask.shape[1]:
                        if handle[rr - drr, cc - dcc] if (0 <= rr - drr < handle.shape[0] and 0 <= cc - dcc < handle.shape[1]) else False:
                            adj += 1
            if not ok:
                continue

            score = (
                int(role["hole_count"]),
                int(door_hits),
                int(outside),
                int(adj),
                -int(_bbox_gap(tool["bbox"], frame["bbox"])),
                int(not is_flip),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "patch": patch.copy(),
                    "mask": mask.copy(),
                    "dr": dr,
                    "dc": dc,
                    "is_flip": bool(is_flip),
                }
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock hollow tools into border-anchored hollow frames while preserving internal voids.

    Input structure:
    - bg = most common color.
    - Border-touching monochrome components are hollow rectangular frames with a single doorway on
      the side opposite the touched border; their inner rectangles are fully background in the input.
    - Non-border foreground components are 2-color tools. One color forms a rectangular plug with
      1-2 enclosed background holes; the other color is a one-sided handle extending away from the plug.

    Rule:
    - Match each tool to the frame whose inner cavity rectangle matches the tool plug bbox under a
      rotation/flip, not by raw area.
    - Place the plug inside the frame cavity so its enclosed holes remain background.
    - Route the handle through the frame doorway so it protrudes outward.
    - If the docking transform uses a reflection and the two tool colors have unequal counts, swap
      those two colors after docking.

    Key insight: The pairing key is the plug's rectangular support with preserved holes nested inside
    the frame cavity; the handle only determines which side of the frame the tool can enter from.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frames, fixed_mask = _find_frames(g, bg)
    assert frames, "must find at least one hollow frame"
    tools = _find_tools(g, bg, fixed_mask)
    assert tools and len(tools) == len(frames), "tool/frame count mismatch"

    candidates: dict[tuple[int, int], dict] = {}
    for ti, tool in enumerate(tools):
        for fi, frame in enumerate(frames):
            state = _fit_tool_to_frame(tool, frame, fixed_mask, bg)
            if state is not None:
                candidates[(ti, fi)] = state

    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(frames)), len(tools)):
        score = 0
        ok = True
        for ti, fi in enumerate(perm):
            if (ti, fi) not in candidates:
                ok = False
                break
            score -= _bbox_gap(tools[ti]["bbox"], frames[fi]["bbox"])
        if ok and (best_score is None or score > best_score):
            best_score = score
            best_perm = perm
    assert best_perm is not None, "must find a valid tool-to-frame assignment"

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    occupied = fixed_mask.copy()

    for ti, fi in enumerate(best_perm):
        tool = tools[ti]
        state = candidates[(ti, fi)]
        patch = state["patch"].copy()
        mask = state["mask"]
        if state["is_flip"] and len(set(tool["counts"].values())) == 2:
            a, b = tool["colors"]
            patch = _swap_colors(patch, a, b)
        for rr, cc in np.argwhere(mask):
            r = int(rr + state["dr"])
            c = int(cc + state["dc"])
            assert not occupied[r, c]
            out[r, c] = int(patch[rr, cc])
            occupied[r, c] = True

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4
GEN_ATTEMPT_RETRIES = 300

PAIR_TEMPLATES = [
    {
        "name": "A",
        "inner_h": 3,
        "inner_w": 3,
        "holes": [(1, 1)],
        "handle": [(1, 3), (1, 4), (1, 5)],
        "gap_rows": [1],
    },
    {
        "name": "B",
        "inner_h": 3,
        "inner_w": 3,
        "holes": [(1, 1)],
        "handle": [(1, 3), (1, 4), (1, 5), (0, 4)],
        "gap_rows": [1],
    },
    {
        "name": "C",
        "inner_h": 3,
        "inner_w": 4,
        "holes": [(1, 1), (1, 2)],
        "handle": [(0, 4), (1, 4), (0, 5), (1, 5)],
        "gap_rows": [0, 1],
    },
    {
        "name": "D",
        "inner_h": 3,
        "inner_w": 4,
        "holes": [(1, 1)],
        "handle": [(1, 4), (1, 5), (1, 6), (0, 5)],
        "gap_rows": [1],
    },
]


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


def _variant_choices(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    seen = set()
    out = []
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = q != int(bg)
            key = (m.tobytes(), q.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, flip))
    random.shuffle(out)
    return out


def _build_left_pair(spec: dict, bg: int, frame_color: int, plug_color: int, handle_color: int):
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    scene_h = outer_h
    scene_w = max(outer_w, 2 + max(c for _r, c in spec["handle"]))
    frame = np.full((scene_h, scene_w), bg, dtype=int)
    frame[0, :outer_w] = int(frame_color)
    frame[outer_h - 1, :outer_w] = int(frame_color)
    frame[:outer_h, 0] = int(frame_color)
    frame[:outer_h, outer_w - 1] = int(frame_color)
    for rr in spec["gap_rows"]:
        frame[1 + int(rr), outer_w - 1] = int(bg)

    tool = np.full((scene_h, scene_w), bg, dtype=int)
    tool[1 : 1 + inner_h, 1 : 1 + inner_w] = int(plug_color)
    for rr, cc in spec["holes"]:
        tool[1 + int(rr), 1 + int(cc)] = int(bg)
    for rr, cc in spec["handle"]:
        tool[1 + int(rr), 1 + int(cc)] = int(handle_color)

    mask = tool != int(bg)
    lab, n = label(mask, structure=STRUCT4)
    if int(n) != 1:
        return None
    return frame, tool


def _pick_specs(num_pairs: int) -> list[dict]:
    by_area: dict[int, list[dict]] = {}
    for spec in PAIR_TEMPLATES:
        by_area.setdefault(int(spec["inner_h"] * spec["inner_w"]), []).append(spec)
    areas = sorted(by_area)
    if num_pairs == 1:
        return [dict(random.choice(PAIR_TEMPLATES))]
    if len(areas) < num_pairs:
        return []
    chosen = []
    for area in random.sample(areas, k=num_pairs):
        chosen.append(dict(random.choice(by_area[area])))
    random.shuffle(chosen)
    return chosen


def _place_docked_pair(
    i: int,
    spec: dict,
    colors: list[int],
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list,
    frame_bboxes: list,
):
    frame_color = colors[3 * i]
    plug_color = colors[3 * i + 1]
    handle_color = colors[3 * i + 2]
    built = _build_left_pair(spec, bg, frame_color, plug_color, handle_color)
    if built is None:
        return None
    frame_left, tool_left = built
    scene_left = np.where(tool_left != int(bg), tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fb = _bbox(frame_scene != int(bg))
    assert fb is not None
    fr0, fc0, fr1, fc1 = fb

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (scene != int(bg))):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))

    plug_area = int(sum(1 for rr in range(spec["inner_h"]) for cc in range(spec["inner_w"]) if (rr, cc) not in spec["holes"]))
    handle_area = int(len(spec["handle"]))
    return {
        "tool": tool_scene,
        "plug_color": int(plug_color),
        "handle_color": int(handle_color),
        "plug_area": plug_area,
        "handle_area": handle_area,
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    placements = []
    for patch, is_flip in _variant_choices(info["tool"], bg):
        q = patch.copy()
        if is_flip and info["plug_area"] != info["handle_area"]:
            q = _swap_colors(q, info["plug_color"], info["handle_color"])
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                q,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        placements.append((q, bool(is_flip), int(r0), int(c0)))

    if not placements:
        return False

    # Keep the reflection-triggered color-swap branch visible in generated samples.
    if info["plug_area"] != info["handle_area"]:
        flipped = [p for p in placements if p[1]]
        unflipped = [p for p in placements if not p[1]]
        if flipped and unflipped:
            placements = flipped if random.random() < 0.5 else unflipped

    q, _is_flip, r0, c0 = random.choice(placements)
    paste(input_grid, q, r0, c0, transparent=bg)
    paste(placement_canvas, q, r0, c0, transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    specs = _pick_specs(num_pairs)
    if len(specs) != num_pairs:
        return None

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes = []
    frame_bboxes = []
    placed_tools = []

    for i, spec in enumerate(specs):
        info = _place_docked_pair(
            i,
            spec,
            colors,
            sides[i],
            bg,
            H,
            W,
            output,
            input_grid,
            union_bboxes,
            frame_bboxes,
        )
        if info is None:
            return None
        placed_tools.append(info)

    placement_canvas = output.copy()
    for info in placed_tools:
        if not _undock_tool(info, bg, input_grid, placement_canvas):
            return None

    if np.array_equal(main(input_grid), output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
