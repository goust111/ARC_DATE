from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORTHO8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    neigh = ORTHO8 if connectivity == 8 else ORTHO4
    out: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or vis[r0, c0]:
                continue
            stack = [(r0, c0)]
            vis[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((int(r), int(c)))
                for dr, dc in neigh:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        stack.append((rr, cc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    return _bbox_from_coords(np.argwhere(mask))


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    mask: np.ndarray | None = None,
) -> None:
    h, w = patch.shape
    if mask is None:
        if transparent is None:
            mask = np.ones((h, w), dtype=bool)
        else:
            mask = patch != int(transparent)
    dst[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return out


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros_like(patch, dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    patch[mask] = grid[r0 : r1 + 1, c0 : c1 + 1][mask]
    return patch, mask, bbox


def _all_variants(patch: np.ndarray, mask: np.ndarray, bg: int) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    seen = set()
    out: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        for is_flip in (False, True):
            q = np.fliplr(p) if is_flip else p
            qm = np.fliplr(m) if is_flip else m
            key = (qm.shape, qm.tobytes(), q[qm].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q.copy(), qm.copy(), bool(is_flip)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock movable 2-color tools into border-anchored frames.

    One color-region of each tool matches some frame cavity by area. Rotate / reflect
    the tool so that region exactly fills the cavity and the other region extends outside.
    If the docking transform uses a reflection, swap the tool's two colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    fg = g != int(bg)
    fg_components = _components(fg, connectivity=4)

    fixed_mask = np.zeros((h, w), dtype=bool)
    fixed_components: list[dict] = []
    movable_components: list[dict] = []

    for coords in fg_components:
        patch, mask, bbox = _extract_patch(g, coords, bg)
        item = {
            "coords": coords,
            "bbox": bbox,
            "patch": patch,
            "mask": mask,
            "size": int(coords.shape[0]),
        }
        if _touches_border(coords, h, w):
            fixed_components.append(item)
            fixed_mask[coords[:, 0], coords[:, 1]] = True
        else:
            movable_components.append(item)

    assert fixed_components, "expected at least one border-touching frame fragment"
    assert movable_components, "expected at least one movable tool"

    parent = list(range(len(fixed_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(fixed_components)):
        for j in range(i + 1, len(fixed_components)):
            if _bbox_gap(fixed_components[i]["bbox"], fixed_components[j]["bbox"]) <= 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        groups.setdefault(find(i), []).append(i)

    anchors: list[dict] = []
    for idxs in groups.values():
        coords = np.concatenate([fixed_components[i]["coords"] for i in idxs], axis=0)
        bbox = _bbox_from_coords(coords)
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True

        r0, c0, r1, c1 = bbox
        socket = np.zeros((h, w), dtype=bool)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in ORTHO4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if r0 <= rr <= r1 and c0 <= cc <= c1 and mask[rr, cc]:
                        socket[r, c] = True
                        break

        sub_bg = g[r0 : r1 + 1, c0 : c1 + 1] == int(bg)
        cavity = np.zeros((h, w), dtype=bool)
        best_area = -1
        best_comp: np.ndarray | None = None
        for comp in _components(sub_bg, connectivity=4):
            adjacent = False
            for rr, cc in comp:
                gr = int(r0 + rr)
                gc = int(c0 + cc)
                for dr, dc in ORTHO4:
                    nr = int(gr + dr)
                    nc = int(gc + dc)
                    if r0 <= nr <= r1 and c0 <= nc <= c1 and mask[nr, nc]:
                        adjacent = True
                        break
                if adjacent:
                    break
            if adjacent and int(comp.shape[0]) > best_area:
                best_area = int(comp.shape[0])
                best_comp = comp
        assert best_comp is not None, "each anchor group must expose a cavity"
        for rr, cc in best_comp:
            cavity[int(r0 + rr), int(c0 + cc)] = True

        dir_counts = [0, 0, 0, 0]
        for r, c in coords:
            for i, (dr, dc) in enumerate(ORTHO4):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == int(bg):
                    dir_counts[i] += 1
        open_dir = ORTHO4[int(np.argmax(dir_counts))]

        anchors.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": socket,
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "open_dir": open_dir,
                "size": int(coords.shape[0]),
            }
        )

    anchors.sort(key=lambda item: -item["size"])
    movable_components.sort(key=lambda item: -item["size"])

    color_counts = [
        Counter(int(v) for v in item["patch"][item["mask"]].tolist())
        for item in movable_components
    ]

    def compatible(ci: int, gi: int) -> bool:
        need = int(anchors[gi]["cavity_size"])
        return any(int(cnt) == need for cnt in color_counts[ci].values())

    assignment: list[int] = []
    if len(movable_components) == len(anchors) and len(anchors) <= 6:
        best_perm: tuple[int, ...] | None = None
        best_score: tuple[int, int] | None = None
        for perm in itertools.permutations(range(len(anchors))):
            ok = True
            dist_sum = 0
            for ci, gi in enumerate(perm):
                if not compatible(ci, gi):
                    ok = False
                    break
                dist_sum += _bbox_gap(movable_components[ci]["bbox"], anchors[gi]["bbox"])
            if not ok:
                continue
            score = (len(perm), -int(dist_sum))
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None, "must find an unambiguous tool-to-frame matching"
        assignment = [int(x) for x in best_perm]
    else:
        used: set[int] = set()
        for ci, comp in enumerate(movable_components):
            best_g = None
            best_d = None
            for gi, grp in enumerate(anchors):
                if gi in used or not compatible(ci, gi):
                    continue
                dist = _bbox_gap(comp["bbox"], grp["bbox"])
                if best_d is None or dist < best_d:
                    best_d = dist
                    best_g = gi
            if best_g is None:
                for gi, grp in enumerate(anchors):
                    if gi in used:
                        continue
                    dist = _bbox_gap(comp["bbox"], grp["bbox"])
                    if best_d is None or dist < best_d:
                        best_d = dist
                        best_g = gi
            assert best_g is not None
            used.add(int(best_g))
            assignment.append(int(best_g))

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchors]

    for ci, item in enumerate(movable_components):
        gi = assignment[ci]
        grp = anchors[gi]
        placed_grp = placed_groups[gi]
        cavity = grp["cavity_mask"]
        socket = grp["socket_mask"]
        cavity_size = int(grp["cavity_size"])
        odr, odc = grp["open_dir"]
        bbox = grp["bbox"]

        best_state = None
        best_score = None
        comp_counts = color_counts[ci]
        colors_here = sorted(int(k) for k in comp_counts)

        for patch, mask, is_flip in _all_variants(item["patch"], item["mask"], bg):
            coords = np.argwhere(mask)
            ph, pw = mask.shape
            plug_colors = [col for col in colors_here if int(np.sum((patch == int(col)) & mask)) == cavity_size]
            if not plug_colors:
                continue
            for r0 in range(h - ph + 1):
                for c0 in range(w - pw + 1):
                    overlap = False
                    for rr, cc in coords:
                        gr = int(r0 + rr)
                        gc = int(c0 + cc)
                        if placed_global[gr, gc]:
                            overlap = True
                            break
                    if overlap:
                        continue

                    adj = 0
                    cavity_hits = 0
                    socket_hits = 0
                    outside = 0
                    br0, bc0, br1, bc1 = bbox
                    for rr, cc in coords:
                        gr = int(r0 + rr)
                        gc = int(c0 + cc)
                        for dr, dc in ORTHO4:
                            nr = int(gr + dr)
                            nc = int(gc + dc)
                            if 0 <= nr < h and 0 <= nc < w and placed_grp[nr, nc]:
                                adj += 1
                        if cavity[gr, gc]:
                            cavity_hits += 1
                        if socket[gr, gc]:
                            socket_hits += 1
                        if odc == -1 and gc < bc0:
                            outside += 1
                        elif odc == 1 and gc > bc1:
                            outside += 1
                        elif odr == -1 and gr < br0:
                            outside += 1
                        elif odr == 1 and gr > br1:
                            outside += 1
                    if adj == 0:
                        continue

                    best_plug = None
                    for plug_col in plug_colors:
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cavity = 0
                        for rr, cc in coords:
                            gr = int(r0 + rr)
                            gc = int(c0 + cc)
                            in_cavity = bool(cavity[gr, gc])
                            col = int(patch[rr, cc])
                            if col == int(plug_col):
                                if in_cavity:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cavity:
                                nonplug_in_cavity += 1
                        plug_full = int(
                            plug_hits == cavity_size and plug_outside == 0 and nonplug_in_cavity == 0
                        )
                        plug_score = (
                            int(plug_full),
                            int(plug_hits),
                            -int(nonplug_in_cavity),
                            -int(plug_outside),
                        )
                        if best_plug is None or plug_score > best_plug:
                            best_plug = plug_score

                    dist = _bbox_gap(bbox, (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)))
                    score = tuple(best_plug) + (
                        int(cavity_hits),
                        int(adj),
                        int(outside),
                        int(socket_hits),
                        -int(dist),
                        -int(r0),
                        -int(c0),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (patch.copy(), mask.copy(), int(r0), int(c0), bool(is_flip))

        assert best_state is not None, "must place every movable tool"
        patch, mask, r0, c0, is_flip = best_state
        if is_flip and len(colors_here) == 2:
            patch = _swap_colors(patch, colors_here[0], colors_here[1])

        for rr, cc in np.argwhere(mask):
            gr = int(r0 + rr)
            gc = int(c0 + cc)
            out[gr, gc] = int(patch[rr, cc])
            placed_global[gr, gc] = True
            placed_grp[gr, gc] = True

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (22, 31)
NUM_PAIRS_CHOICES = [1, 2]

FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

UNION_BBOX_GAP_MIN = 3
FRAME_BBOX_GAP_MIN = 4
TOOL_BORDER = 1
TOOL_PADDING = 3

GEN_RETRIES = 300
HANDLE_RETRIES = 60
PAIR_SPEC_RETRIES = 300


def _notch_offsets(notch_n: int, frame_w: int, bot_h: int) -> list[tuple[int, int]]:
    if notch_n == 1:
        return [(0, frame_w - 1)]
    if notch_n == 2:
        return [(0, frame_w - 2), (0, frame_w - 1)]
    if notch_n == 3:
        if bot_h >= 2:
            return [(0, frame_w - 2), (0, frame_w - 1), (1, frame_w - 1)]
        return [(0, frame_w - 3), (0, frame_w - 2), (0, frame_w - 1)]
    raise ValueError("unsupported notch size")


def _grow_connected_mask(h: int, w: int, area: int, *, require_left: bool) -> np.ndarray | None:
    for _ in range(HANDLE_RETRIES):
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), 0 if require_left else random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ORTHO4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        frontier.add((rr, cc))
            if not frontier:
                break
            rr, cc = random.choice(tuple(frontier))
            mask[rr, cc] = True
        if int(mask.sum()) == area and (not require_left or bool(mask[:, 0].any())):
            return mask
    return None


def _make_handle_sprite(handle_area: int, bg: int, handle_color: int) -> np.ndarray | None:
    lo, hi = HANDLE_SPRITE_DIM
    dims = [
        (hh, ww)
        for hh in range(lo, hi)
        for ww in range(lo, hi)
        if handle_area <= hh * ww <= 2 * handle_area
    ]
    random.shuffle(dims)
    for hh, ww in dims:
        mask = _grow_connected_mask(hh, ww, handle_area, require_left=True)
        if mask is None:
            continue
        sprite = np.full((hh, ww), int(bg), dtype=int)
        sprite[mask] = int(handle_color)
        return sprite
    return None


def _build_left_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    frame_w: int,
    top_h: int,
    bot_h: int,
    notch_n: int,
    handle_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    frame_h = top_h + 1 + bot_h
    frame = np.full((frame_h, frame_w), int(bg), dtype=int)
    frame[:top_h, :] = int(frame_color)
    frame[top_h + 1 :, :] = int(frame_color)
    for dr, dc in _notch_offsets(notch_n, frame_w, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = {int(r) for r in range(frame_h) if cavity[r, frame_w - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    left_rows = {int(r) for r in range(hs_h) if handle_sprite[r, 0] != int(bg)}
    valid_offsets = [off for off in range(frame_h) if any(off + rr in attach_rows for rr in left_rows)]
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(frame_h, voff + hs_h)
    total_w = frame_w + hs_w

    frame_full = np.full((total_h, total_w), int(bg), dtype=int)
    frame_full[:frame_h, :frame_w] = frame

    tool = np.full((total_h, total_w), int(bg), dtype=int)
    tool[:frame_h, :frame_w][cavity] = int(plug_color)
    handle_mask = handle_sprite != int(bg)
    tool[voff : voff + hs_h, frame_w : frame_w + hs_w][handle_mask] = int(handle_color)

    return frame_full, tool


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


def _param_options() -> list[tuple[int, int, int, int]]:
    out = []
    for frame_w in range(*FRAME_W_RANGE):
        for notch_n in range(*NOTCH_N_RANGE):
            plug_area = frame_w + notch_n
            if not (PLUG_AREA_BOUNDS[0] <= plug_area <= PLUG_AREA_BOUNDS[1]):
                continue
            for top_h in range(*TOP_H_RANGE):
                for bot_h in range(*BOT_H_RANGE):
                    out.append((frame_w, top_h, bot_h, notch_n))
    return out


PARAM_OPTIONS = _param_options()


def _pick_pair_specs(num_pairs: int) -> list[dict]:
    for _ in range(PAIR_SPEC_RETRIES):
        specs: list[dict] = []
        used_plug_areas: set[int] = set()
        for _idx in range(num_pairs):
            choices = [opt for opt in PARAM_OPTIONS if (opt[0] + opt[3]) not in used_plug_areas]
            if not choices:
                specs = []
                break
            frame_w, top_h, bot_h, notch_n = random.choice(choices)
            plug_area = frame_w + notch_n
            specs.append(
                {
                    "frame_w": frame_w,
                    "top_h": top_h,
                    "bot_h": bot_h,
                    "notch_n": notch_n,
                    "plug_area": plug_area,
                }
            )
            used_plug_areas.add(int(plug_area))
        if len(specs) != num_pairs:
            continue

        plug_areas = {int(spec["plug_area"]) for spec in specs}
        ok = True
        for spec in specs:
            plug_area = int(spec["plug_area"])
            handle_choices = [
                area
                for area in range(*HANDLE_AREA_RANGE)
                if TOOL_TOTAL_BOUNDS[0] <= plug_area + area <= TOOL_TOTAL_BOUNDS[1]
                and area != plug_area
                and area not in plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = int(random.choice(handle_choices))
        if ok:
            return specs
    return []


def _candidate_origins(side: str, ph: int, pw: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "left":
        origins = [(r0, 0) for r0 in range(h - ph + 1)]
    elif side == "right":
        origins = [(r0, w - pw) for r0 in range(h - ph + 1)]
    elif side == "top":
        origins = [(0, c0) for c0 in range(w - pw + 1)]
    elif side == "bottom":
        origins = [(h - ph, c0) for c0 in range(w - pw + 1)]
    else:
        raise ValueError(side)
    random.shuffle(origins)
    return origins


def _variant_choices(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    seen = set()
    out: list[tuple[np.ndarray, bool]] = []
    mask = patch != int(bg)
    for k in range(4):
        rot_patch = np.rot90(patch, k)
        rot_mask = np.rot90(mask, k)
        for flip in (False, True):
            q = np.fliplr(rot_patch) if flip else rot_patch
            qm = np.fliplr(rot_mask) if flip else rot_mask
            key = (qm.shape, qm.tobytes(), q[qm].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q.copy(), bool(flip)))
    random.shuffle(out)
    return out


def _fits_with_padding(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int) -> bool:
    h, w = canvas.shape
    fg_coords = np.argwhere(patch != int(bg))
    if fg_coords.size == 0:
        return False
    for rr, cc in fg_coords:
        gr = int(r0 + rr)
        gc = int(c0 + cc)
        if int(canvas[gr, gc]) != int(bg):
            return False
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                if abs(dr) + abs(dc) > padding:
                    continue
                nr = int(gr + dr)
                nc = int(gc + dc)
                if 0 <= nr < h and 0 <= nc < w and int(canvas[nr, nc]) != int(bg):
                    return False
    return True


def _find_free_location(canvas: np.ndarray, patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    h, w = canvas.shape
    ph, pw = patch.shape
    if ph + 2 * TOOL_BORDER > h or pw + 2 * TOOL_BORDER > w:
        return None
    origins = [
        (r0, c0)
        for r0 in range(TOOL_BORDER, h - ph - TOOL_BORDER + 1)
        for c0 in range(TOOL_BORDER, w - pw - TOOL_BORDER + 1)
    ]
    random.shuffle(origins)
    for r0, c0 in origins:
        if _fits_with_padding(canvas, patch, r0, c0, bg, TOOL_PADDING):
            return int(r0), int(c0)
    return None


def _place_docked_pair(
    pair_idx: int,
    spec: dict,
    colors: list[int],
    side: str,
    bg: int,
    h: int,
    w: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> dict | None:
    frame_color = int(colors[3 * pair_idx])
    plug_color = int(colors[3 * pair_idx + 1])
    handle_color = int(colors[3 * pair_idx + 2])

    built = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        frame_w=int(spec["frame_w"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
    )
    if built is None:
        return None
    frame_left, tool_left = built

    scene_left = np.where(tool_left != int(bg), tool_left, frame_left)
    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)

    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _bbox_from_mask(frame_scene != int(bg))

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, h, w):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == h - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == w - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) < UNION_BBOX_GAP_MIN for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_BBOX_GAP_MIN for prev in frame_bboxes):
            continue
        scene_mask = scene != int(bg)
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & scene_mask):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    _paste(output, scene, r0, c0, transparent=bg)
    _paste(input_grid, frame_scene, r0, c0, transparent=bg)
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))

    return {
        "tool": tool,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"], bg):
        if is_flip:
            patch = _swap_colors(patch, int(info["plug_color"]), int(info["handle_color"]))
        loc = _find_free_location(placement_canvas, patch, bg)
        if loc is None:
            continue
        r0, c0 = loc
        _paste(input_grid, patch, r0, c0, transparent=bg)
        _paste(placement_canvas, patch, r0, c0, transparent=bg)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.randrange(10)
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    specs = _pick_pair_specs(num_pairs)
    if len(specs) != num_pairs:
        return None

    colors = [col for col in range(10) if col != bg]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    output = np.full((h, w), int(bg), dtype=int)
    input_grid = np.full((h, w), int(bg), dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    tools: list[dict] = []

    for pair_idx, spec in enumerate(specs):
        info = _place_docked_pair(
            pair_idx,
            spec,
            colors,
            sides[pair_idx],
            bg,
            h,
            w,
            output,
            input_grid,
            union_bboxes,
            frame_bboxes,
        )
        if info is None:
            return None
        tools.append(info)

    placement_canvas = output.copy()
    for info in tools:
        if not _undock_tool(info, bg, input_grid, placement_canvas):
            return None

    if np.array_equal(main(input_grid), output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
