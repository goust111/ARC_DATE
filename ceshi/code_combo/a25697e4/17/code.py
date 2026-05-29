# Combo variant of a25697e4 (index 17)
# combo_spec: D12/0+D13/0+D17/2
# Name: 包围结构内核填充融合
#
# Design analysis
# a) Concrete behavior trace:
# - Original task: one tool color matched the cavity area, that color filled the notch, the other
#   color stayed outside, and a reflected docking could swap the two colors.
# - This combo flips the area role (D13/0): the INNER color now matches the cavity, not the outer
#   color. It also changes the tool topology (D12/0): the inner color is surrounded by the outer
#   color in the standalone tool. Finally D17/2 ties the inner color to the frame color, so after
#   docking the filled cavity visually merges into the frame.
# - Small example: a 2x3 inner block of frame color sits inside a 5x4 shell of plug color. In the
#   output, that 2x3 inner block fills the notch exactly; the shell is projected outward from the
#   notch opening. This makes the filled part disappear into the frame while the plug remains the
#   only visible protrusion.
# b) Complexity calibration:
# - The generator uses a small library of nested tool templates with two distinct inner areas. That
#   keeps the puzzle legible and prevents accidental many-to-many matches.
# - Frame colors are distinct when there are two pairs, and the inner color equals the frame color,
#   so humans get both a shape cue and a color cue.
# c) Hidden-assumption audit:
# - The original solver assumed "whichever color has cavity-sized area is the plug" and optionally
#   swapped colors after a reflection. That assumption is invalid here because the cavity-sized
#   region must remain the frame color after docking.
# - A stricter rigid-body interpretation creates a geometric conflict: a fully surrounding shell
#   cannot be inserted wholesale into a border notch without some shell pixels landing on the frame.
#   This combo therefore uses a consistent projection rule instead: the inner region must align
#   exactly with the cavity, while shell pixels are only kept where the transformed shell lands on
#   empty cells outside the frame.
# d) Placement legality:
# - Frames are border-touching rectangles with a carved notch; tools are non-border standalone
#   2-color components; random placement uses explicit border and padding constraints.
# - Output composition never writes plug pixels on top of frame pixels.
# e) Visual observability:
# - Across examples, the cavity always disappears into the frame color, not the plug color. That is
#   the main observable difference from the original task.
# - The standalone tools visibly show an inner kernel inside an outer shell, so the nesting rule is
#   readable from a few examples.
# f) Visual cleanliness:
# - Only the shell is left visible outside the frame after docking; no extra symbols or hidden
#   sorting criteria are introduced.
# g) Inference support:
# - The key control variables are repeated and low-cardinality: two handle areas, one-to-one
#   frame/tool color agreement, and a stable "kernel fills, shell protrudes" action.

from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]
FRAME_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 2
GEN_ATTEMPTS = 300


TOOL_LIBRARY = [
    {
        "name": "inner5",
        "patch": np.array(
            [
                [1, 1, 1, 1],
                [1, 2, 2, 1],
                [1, 2, 2, 1],
                [1, 2, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=int,
        ),
        "handle_mask": np.array(
            [
                [1, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=bool,
        ),
    },
    {
        "name": "inner6",
        "patch": np.array(
            [
                [1, 1, 1, 1],
                [1, 2, 2, 1],
                [1, 2, 2, 1],
                [1, 2, 2, 1],
                [1, 1, 1, 1],
            ],
            dtype=int,
        ),
        "handle_mask": np.array(
            [
                [1, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=bool,
        ),
    },
]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


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


def _gen_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    seen = set()
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        for q, qm in ((p, m), (np.fliplr(p), np.fliplr(m))):
            key = (qm.tobytes(), q[qm].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, qm))
    return out


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return tuple()
    r0, c0 = pts.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))


def _touches_patch_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _is_enclosed(inner: np.ndarray, outer: np.ndarray) -> bool:
    h, w = inner.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if outer[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if seen[rr, cc] or outer[rr, cc]:
                continue
            seen[rr, cc] = True
            q.append((rr, cc))
    return not bool((seen & inner).any())


def _tool_roles(patch: np.ndarray, mask: np.ndarray) -> dict:
    colors = sorted({int(x) for x in patch[mask].flatten()})
    assert len(colors) == 2, "tool must have exactly two colors"
    for handle_color in colors:
        plug_color = colors[0] if colors[1] == handle_color else colors[1]
        handle_mask = (patch == int(handle_color)) & mask
        plug_mask = (patch == int(plug_color)) & mask
        if _touches_patch_border(handle_mask):
            continue
        if _is_enclosed(handle_mask, plug_mask):
            return {
                "handle_color": int(handle_color),
                "plug_color": int(plug_color),
                "handle_mask": handle_mask,
                "plug_mask": plug_mask,
                "handle_sig": _shape_signature(handle_mask),
                "handle_area": int(handle_mask.sum()),
            }
    # Controlled fallback for the generator's subfamily: the inner color is the smaller non-border one.
    best = None
    for handle_color in colors:
        handle_mask = (patch == int(handle_color)) & mask
        if _touches_patch_border(handle_mask):
            continue
        score = (int(handle_mask.sum()),)
        if best is None or score < best[0]:
            plug_color = colors[0] if colors[1] == handle_color else colors[1]
            best = (
                score,
                {
                    "handle_color": int(handle_color),
                    "plug_color": int(plug_color),
                    "handle_mask": handle_mask,
                    "plug_mask": ((patch == int(plug_color)) & mask),
                    "handle_sig": _shape_signature(handle_mask),
                    "handle_area": int(handle_mask.sum()),
                },
            )
    assert best is not None, "tool role detection failed"
    return best[1]


def _frame_groups(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    fg = grid != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros_like(fg, dtype=bool)
    groups: list[dict] = []
    H, W = grid.shape
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        if len(pts) == 0:
            continue
        if not any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts):
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = lab == cid
        fixed_mask |= mask
        sub = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        sub_bg = sub == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        cavity = np.zeros_like(mask, dtype=bool)
        best_id = None
        best_area = -1
        for sid in range(1, sub_n + 1):
            loc = np.argwhere(sub_lab == sid)
            area = int(len(loc))
            adj = False
            for rr, cc in loc:
                rr0 = int(r0 + rr)
                cc0 = int(c0 + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2, c2 = rr0 + dr, cc0 + dc
                    if r0 <= r2 <= r1 and c0 <= c2 <= c1 and mask[r2, c2]:
                        adj = True
                        break
                if adj:
                    break
            if adj and area > best_area:
                best_area = area
                best_id = sid
        assert best_id is not None, "frame cavity not found"
        for rr, cc in np.argwhere(sub_lab == best_id):
            cavity[int(r0 + rr), int(c0 + cc)] = True
        edge_counts = {
            (-1, 0): int(sum(1 for rr, cc in np.argwhere(cavity) if rr == r0)),
            (1, 0): int(sum(1 for rr, cc in np.argwhere(cavity) if rr == r1)),
            (0, -1): int(sum(1 for rr, cc in np.argwhere(cavity) if cc == c0)),
            (0, 1): int(sum(1 for rr, cc in np.argwhere(cavity) if cc == c1)),
        }
        open_dir = max(edge_counts, key=edge_counts.get)
        frame_colors = [int(x) for x in grid[mask].flatten() if int(x) != int(bg)]
        groups.append(
            {
                "mask": mask,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cavity_mask": cavity,
                "cavity_sig": _shape_signature(cavity),
                "cavity_area": int(cavity.sum()),
                "open_dir": open_dir,
                "frame_color": int(Counter(frame_colors).most_common(1)[0][0]),
            }
        )
    groups.sort(key=lambda g: (g["bbox"][0], g["bbox"][1]))
    return groups, fixed_mask


def _movable_components(grid: np.ndarray, bg: int, fixed_mask: np.ndarray) -> list[dict]:
    fg = (grid != int(bg)) & (~fixed_mask)
    lab, n = label(fg, structure=STRUCT4)
    comps: list[dict] = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        if len(pts) == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = lab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == cid
        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
        patch[~mask] = 0
        roles = _tool_roles(patch, mask)
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "roles": roles,
            }
        )
    comps.sort(key=lambda c: (c["bbox"][0], c["bbox"][1]))
    return comps


def _align_variant_to_cavity(comp: dict, grp: dict, fixed_mask: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    roles = comp["roles"]
    best = None
    best_score = None
    cavity_pts = np.argwhere(grp["cavity_mask"])
    cav_r0, cav_c0 = cavity_pts.min(axis=0)
    br0, bc0, br1, bc1 = grp["bbox"]
    odr, odc = grp["open_dir"]
    for p, m in _gen_variants(comp["patch"], comp["mask"]):
        handle_mask = (p == int(roles["handle_color"])) & m
        if _shape_signature(handle_mask) != grp["cavity_sig"]:
            continue
        handle_pts = np.argwhere(handle_mask)
        dr = int(cav_r0 - handle_pts[:, 0].min())
        dc = int(cav_c0 - handle_pts[:, 1].min())
        mapped = handle_pts + np.array([dr, dc])
        if set(map(tuple, mapped.tolist())) != set(map(tuple, cavity_pts.tolist())):
            continue
        plug_mask = (p == int(roles["plug_color"])) & m
        visible = 0
        outward = 0
        overlaps = 0
        for rr, cc in np.argwhere(plug_mask):
            r = int(rr + dr)
            c = int(cc + dc)
            if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                overlaps += 1
                continue
            if fixed_mask[r, c]:
                overlaps += 1
                continue
            if out[r, c] != int(bg):
                overlaps += 1
                continue
            visible += 1
            if odr == -1 and r < br0:
                outward += 1
            elif odr == 1 and r > br1:
                outward += 1
            elif odc == -1 and c < bc0:
                outward += 1
            elif odc == 1 and c > bc1:
                outward += 1
        score = (int(outward), int(visible), -int(overlaps))
        if best_score is None or score > best_score:
            best_score = score
            best = (p, m, dr, dc)
    assert best is not None, "no docking alignment found"
    return best


def _pair_components(comps: list[dict], groups: list[dict]) -> list[int]:
    assert len(comps) == len(groups), "generator guarantees equal tool/frame counts"

    def compatible(ci: int, gi: int) -> bool:
        roles = comps[ci]["roles"]
        grp = groups[gi]
        return (
            int(roles["handle_color"]) == int(grp["frame_color"])
            and int(roles["handle_area"]) == int(grp["cavity_area"])
            and any(
                _shape_signature((p == int(roles["handle_color"])) & m) == grp["cavity_sig"]
                for p, m in _gen_variants(comps[ci]["patch"], comps[ci]["mask"])
            )
        )

    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(groups))):
        ok = True
        dist = 0
        for ci, gi in enumerate(perm):
            if not compatible(ci, gi):
                ok = False
                break
            dist += _bbox_gap(comps[ci]["bbox"], groups[gi]["bbox"])
        if not ok:
            continue
        score = (-int(dist),)
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None, "tool/frame pairing failed"
    return [int(x) for x in best_perm]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock nested two-color tools into border notches by merging the inner kernel into the frame.

    Input structure:
    - The background is the most common color.
    - Border-touching components are solid frame slabs with one staircase-like bg notch.
    - Non-border components are movable 2-color tools whose inner color is enclosed by the outer one.

    Rule:
    - The enclosed inner region is the handle/kernel; its pixel count and shape match exactly one
      frame cavity, and its color is the same as that frame.
    - Move a tool to the matching frame so the inner region fills the cavity; after filling, that
      region visually fuses into the frame because the colors agree.
    - The outer shell stays as the visible protrusion outside the notch, and the tool disappears
      from its original location.

    Key insight: pairing is determined by an enclosed inner shape that matches the cavity, while the
    visible color change happens outside the frame because the cavity itself is filled by frame color.
    """
    g = grid.copy()
    bg = get_bg(g)
    groups, fixed_mask = _frame_groups(g, bg)
    comps = _movable_components(g, bg, fixed_mask)
    mapping = _pair_components(comps, groups)

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    occupied = fixed_mask.copy()

    for comp, gi in zip(comps, mapping):
        grp = groups[gi]
        roles = comp["roles"]
        # New implicit assumptions, all guaranteed by generate_input():
        # - the inner color equals the frame color;
        # - one D4 variant of the inner mask exactly matches the cavity;
        # - at least one outer-shell pixel lands on empty space outside the frame.
        p, m, dr, dc = _align_variant_to_cavity(comp, grp, occupied, out, bg)
        out[grp["cavity_mask"]] = int(grp["frame_color"])
        plug_mask = (p == int(roles["plug_color"])) & m
        for rr, cc in np.argwhere(plug_mask):
            r = int(rr + dr)
            c = int(cc + dc)
            if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                continue
            if occupied[r, c] or out[r, c] != int(bg):
                continue
            out[r, c] = int(roles["plug_color"])
            occupied[r, c] = True

    return out


def _frame_and_tool_from_spec(spec: dict, bg: int, frame_color: int, plug_color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    handle = spec["handle_mask"]
    hh, hw = handle.shape
    frame_h = hh + 2
    frame_w = hw + 2
    frame = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    cavity = np.zeros((frame_h, frame_w), dtype=bool)
    r0 = 1
    c0 = frame_w - hw
    cavity[r0 : r0 + hh, c0 : c0 + hw] = handle
    frame[cavity] = int(bg)

    tool = np.full(spec["patch"].shape, int(bg), dtype=int)
    tool[spec["patch"] == 1] = int(plug_color)
    tool[spec["patch"] == 2] = int(frame_color)
    return frame, tool, cavity


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        out = [(r0, 0) for r0 in range(0, H - ph + 1)]
    elif side == "right":
        out = [(r0, W - pw) for r0 in range(0, H - ph + 1)]
    elif side == "top":
        out = [(0, c0) for c0 in range(0, W - pw + 1)]
    elif side == "bottom":
        out = [(H - ph, c0) for c0 in range(0, W - pw + 1)]
    else:
        raise ValueError(side)
    random.shuffle(out)
    return out


def _place_frame(
    frame: np.ndarray,
    cavity: np.ndarray,
    side: str,
    H: int,
    W: int,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    frame_bboxes: list[tuple[int, int, int, int]],
    bg: int,
) -> dict | None:
    fr = _orient_left_to_side(frame, side)
    cav = _orient_left_to_side(cavity.astype(np.uint8), side).astype(bool)
    ph, pw = fr.shape
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        if any(_bbox_gap(bbox, prev) < FRAME_GAP_MIN for prev in frame_bboxes):
            continue
        if np.any((input_grid[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (fr != int(bg))):
            continue
        paste(input_grid, fr, r0, c0, transparent=bg)
        paste(output_grid, fr, r0, c0, transparent=bg)
        global_cavity = np.zeros((H, W), dtype=bool)
        global_cavity[r0 : r0 + ph, c0 : c0 + pw] = cav
        frame_bboxes.append(bbox)
        open_dir = {
            "left": (0, 1),
            "right": (0, -1),
            "top": (1, 0),
            "bottom": (-1, 0),
        }[side]
        return {"bbox": bbox, "cavity_mask": global_cavity, "open_dir": open_dir}
    return None


def _dock_tool_onto_output(output_grid: np.ndarray, frame_info: dict, tool: np.ndarray, frame_color: int, plug_color: int, bg: int) -> None:
    H, W = output_grid.shape
    mask = tool != int(bg)
    comp = {"patch": tool, "mask": mask, "roles": _tool_roles(tool, mask)}
    grp = {
        "bbox": frame_info["bbox"],
        "cavity_mask": frame_info["cavity_mask"],
        "cavity_sig": _shape_signature(frame_info["cavity_mask"]),
        "frame_color": int(frame_color),
        "open_dir": frame_info["open_dir"],
    }
    occupied = output_grid != int(bg)
    p, m, dr, dc = _align_variant_to_cavity(comp, grp, occupied, output_grid, bg)
    output_grid[frame_info["cavity_mask"]] = int(frame_color)
    plug_mask = (p == int(plug_color)) & m
    for rr, cc in np.argwhere(plug_mask):
        r = int(rr + dr)
        c = int(cc + dc)
        if not (0 <= r < H and 0 <= c < W):
            continue
        if output_grid[r, c] != int(bg):
            continue
        output_grid[r, c] = int(plug_color)


def _random_tool_variant(tool: np.ndarray, bg: int) -> np.ndarray:
    mask = tool != int(bg)
    return random.choice(_gen_variants(tool, mask))[0]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)
    if num_pairs > len(TOOL_LIBRARY):
        return None

    specs = random.sample(TOOL_LIBRARY, k=num_pairs)
    frame_colors = random.sample([c for c in range(10) if c != bg], k=num_pairs)
    remaining = [c for c in range(10) if c != bg and c not in frame_colors]
    plug_colors = random.sample(remaining, k=num_pairs)
    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)

    input_grid = np.full((H, W), int(bg), dtype=int)
    output_grid = np.full((H, W), int(bg), dtype=int)
    frame_bboxes: list[tuple[int, int, int, int]] = []
    tools_to_place: list[np.ndarray] = []

    for spec, frame_color, plug_color, side in zip(specs, frame_colors, plug_colors, sides):
        frame, tool, cavity = _frame_and_tool_from_spec(spec, bg, frame_color, plug_color)
        frame_info = _place_frame(frame, cavity, side, H, W, input_grid, output_grid, frame_bboxes, bg)
        if frame_info is None:
            return None
        dock_tool = _orient_left_to_side(tool, side)
        _dock_tool_onto_output(output_grid, frame_info, dock_tool, frame_color, plug_color, bg)
        tools_to_place.append(tool)

    placement_canvas = input_grid.copy()
    for tool in tools_to_place:
        patch = _random_tool_variant(tool, bg)
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                patch,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=4,
            )
        except ValueError:
            return None
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(placement_canvas, patch, r0, c0, transparent=bg)

    if np.array_equal(main(input_grid), output_grid):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
