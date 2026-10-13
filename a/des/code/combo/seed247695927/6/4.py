from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_patch(grid: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    submask = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    patch[~submask] = 0
    return patch, submask, (r0, c0)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    if len(coords) == 0:
        return ()
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


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


def _variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    out: list[tuple[np.ndarray, np.ndarray, bool]] = []
    seen = set()
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        for flip in (False, True):
            pp = np.fliplr(p) if flip else p.copy()
            mm = np.fliplr(m) if flip else m.copy()
            key = (mm.tobytes(), tuple(sorted(int(x) for x in pp[mm])))
            if key in seen:
                continue
            seen.add(key)
            out.append((pp, mm, flip))
    return out


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _frame_touch_side(coords: np.ndarray, h: int, w: int) -> str:
    if np.any(coords[:, 1] == 0):
        return "left"
    if np.any(coords[:, 1] == w - 1):
        return "right"
    if np.any(coords[:, 0] == 0):
        return "top"
    return "bottom"


def _extract_cavity(grid: np.ndarray, comp_mask: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(comp_mask)
    r0, c0, r1, c1 = _bbox(pts)
    sub_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)
    best = None
    for comp in _components(sub_bg):
        abs_comp = comp + np.array([[r0, c0]])
        adj = False
        for rr, cc in abs_comp:
            for dr, dc in DIR4:
                nr, nc = int(rr + dr), int(cc + dc)
                if r0 <= nr <= r1 and c0 <= nc <= c1 and comp_mask[nr, nc]:
                    adj = True
                    break
            if adj:
                break
        if not adj:
            continue
        score = int(len(comp))
        if best is None or score > best[0]:
            best = (score, abs_comp)
    assert best is not None, "frame must contain a cavity"
    cav = np.zeros_like(comp_mask)
    for rr, cc in best[1]:
        cav[int(rr), int(cc)] = True
    return cav


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock movable two-color tools into border-anchored frames by filling cavities.

    A tool matches the frame whose cavity area equals one of the tool's color counts.
    The tool may rotate and reflect; if the winning orientation uses reflection,
    swap the two tool colors before placing it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    fg = g != int(bg)
    comps = _components(fg)
    frames = []
    tools = []
    for pts in comps:
        mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            mask[int(r), int(c)] = True
        colors = sorted(int(x) for x in np.unique(g[mask]))
        if _touches_border(pts, h, w):
            cavity = _extract_cavity(g, mask, bg)
            frames.append(
                {
                    "mask": mask,
                    "cavity": cavity,
                    "touch_side": _frame_touch_side(pts, h, w),
                    "cavity_size": int(cavity.sum()),
                }
            )
        elif len(colors) == 2:
            patch, submask, _ = _crop_patch(g, mask)
            counts = {int(c): int(np.sum((patch == int(c)) & submask)) for c in colors}
            tools.append({"patch": patch, "mask": submask, "colors": colors, "counts": counts})

    assert frames and tools

    sizes = [int(fr["cavity_size"]) for fr in frames]
    assert len(set(sizes)) == len(sizes), "cavity sizes must be unique"

    tool_for_size: dict[int, dict] = {}
    for tool in tools:
        matches = [sz for sz in sizes if sz in tool["counts"].values()]
        assert len(matches) == 1, "each tool must match exactly one cavity size"
        tool_for_size[int(matches[0])] = tool
    assert len(tool_for_size) == len(frames)

    out = np.full_like(g, int(bg))
    occupied = np.zeros((h, w), dtype=bool)
    for fr in frames:
        out[fr["mask"]] = g[fr["mask"]]
        occupied |= fr["mask"]

    for fr in frames:
        tool = tool_for_size[int(fr["cavity_size"])]
        cav_coords = np.argwhere(fr["cavity"])
        cav_norm = _normalize(cav_coords)
        placed = None

        for patch0, mask0, used_flip in _variants(tool["patch"], tool["mask"]):
            colors = tool["colors"]
            patch = _swap_colors(patch0, colors[0], colors[1]) if used_flip else patch0
            for plug_color in colors:
                plug_coords = np.argwhere((patch == int(plug_color)) & mask0)
                if len(plug_coords) != int(fr["cavity_size"]):
                    continue
                if _normalize(plug_coords) != cav_norm:
                    continue
                dr = int(cav_coords[:, 0].min() - plug_coords[:, 0].min())
                dc = int(cav_coords[:, 1].min() - plug_coords[:, 1].min())
                abs_all = np.argwhere(mask0) + np.array([[dr, dc]])
                if np.any(abs_all[:, 0] < 0) or np.any(abs_all[:, 0] >= h) or np.any(abs_all[:, 1] < 0) or np.any(abs_all[:, 1] >= w):
                    continue
                ok = True
                for rr, cc in abs_all:
                    rr = int(rr)
                    cc = int(cc)
                    if occupied[rr, cc] and not fr["cavity"][rr, cc]:
                        ok = False
                        break
                if not ok:
                    continue
                other_color = colors[0] if int(plug_color) == int(colors[1]) else colors[1]
                if np.any(((patch == int(other_color)) & mask0) & (fr["cavity"][dr : dr + patch.shape[0], dc : dc + patch.shape[1]] if 0 <= dr and 0 <= dc and dr + patch.shape[0] <= h and dc + patch.shape[1] <= w else False)):
                    continue
                placed = (patch, mask0, dr, dc)
                break
            if placed is not None:
                break

        assert placed is not None, "failed to dock a tool"
        patch, mask0, dr, dc = placed
        for rr, cc in np.argwhere(mask0):
            ar = int(rr + dr)
            ac = int(cc + dc)
            out[ar, ac] = int(patch[rr, cc])
            occupied[ar, ac] = True

    return out


PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _random_connected_mask(rng: random.Random, hh: int, ww: int, area: int, must_touch_side: str) -> np.ndarray:
    cells = [(r, c) for r in range(hh) for c in range(ww)]
    side_cells = []
    if must_touch_side == "right":
        side_cells = [(r, ww - 1) for r in range(hh)]
    elif must_touch_side == "left":
        side_cells = [(r, 0) for r in range(hh)]
    elif must_touch_side == "top":
        side_cells = [(0, c) for c in range(ww)]
    else:
        side_cells = [(hh - 1, c) for c in range(ww)]
    for _ in range(200):
        chosen = {rng.choice(side_cells)}
        frontier = set(chosen)
        while len(chosen) < area:
            cand = []
            for r, c in frontier:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in chosen:
                        cand.append((nr, nc))
            if not cand:
                break
            pick = rng.choice(cand)
            chosen.add(pick)
            frontier.add(pick)
        if len(chosen) == area:
            mask = np.zeros((hh, ww), dtype=bool)
            for r, c in chosen:
                mask[r, c] = True
            return mask
    raise RuntimeError("failed to generate cavity mask")


def _grow_handle(
    rng: random.Random,
    frame_h: int,
    frame_w: int,
    cavity_mask: np.ndarray,
    handle_area: int,
) -> np.ndarray:
    candidates = set()
    for r, c in np.argwhere(cavity_mask):
        if c == frame_w - 1:
            candidates.add((int(r), int(frame_w)))
            candidates.add((int(r), int(frame_w + 1)))
    chosen = {rng.choice(sorted(candidates))}
    while len(chosen) < handle_area:
        cand = []
        for r, c in chosen:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < frame_h and nc >= frame_w and nc <= frame_w + handle_area + 1 and (nr, nc) not in chosen:
                    cand.append((nr, nc))
        if not cand:
            break
        chosen.add(rng.choice(cand))
    assert len(chosen) == handle_area
    max_c = max(c for _, c in chosen)
    mask = np.zeros((frame_h, max_c + 1), dtype=bool)
    for r, c in chosen:
        mask[int(r), int(c)] = True
    return mask


def _build_left_pair(
    rng: random.Random,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    cavity_area: int,
    handle_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    cav_h = rng.randint(2, 4)
    cav_w = rng.randint(max(2, (cavity_area + cav_h - 1) // cav_h), min(4, cavity_area))
    while cav_h * cav_w < cavity_area:
        cav_h = rng.randint(2, 4)
        cav_w = rng.randint(2, 4)
    cavity = _random_connected_mask(rng, cav_h, cav_w, cavity_area, "right")
    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    left = rng.randint(1, 2)
    frame_h = top + cav_h + bottom
    frame_w = left + cav_w
    frame = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame[top : top + cav_h, left : left + cav_w][cavity] = -1
    frame[frame == -1] = -1

    cavity_global = np.zeros((frame_h, frame_w), dtype=bool)
    cavity_global[top : top + cav_h, left : left + cav_w] = cavity
    handle = _grow_handle(rng, frame_h, frame_w, cavity_global, handle_area)
    tool_w = max(frame_w, handle.shape[1])
    tool = np.zeros((frame_h, tool_w), dtype=int)
    tool[:, :frame_w][cavity_global] = int(plug_color)
    tool[: handle.shape[0], : handle.shape[1]][handle] = int(handle_color)
    return frame, tool


def _orient_left(arr: np.ndarray, side: str, fill: int) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, 1)
    if side == "bottom":
        return np.rot90(arr, 3)
    raise ValueError(side)


def _paste_mask(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _can_place(mask_canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, gap: int = 1) -> bool:
    h, w = mask.shape
    hh, ww = mask_canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    pts = np.argwhere(mask)
    for rr, cc in pts:
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        r1 = max(0, ar - gap)
        r2 = min(hh, ar + gap + 1)
        c1 = max(0, ac - gap)
        c2 = min(ww, ac + gap + 1)
        if np.any(mask_canvas[r1:r2, c1:c2]):
            return False
    return True


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr.copy()
    return np.rot90(out, -k)


def _tool_mask(tool: np.ndarray) -> np.ndarray:
    return tool != 0


def _frame_mask(frame: np.ndarray) -> np.ndarray:
    return frame != -1


def _frame_patch(frame: np.ndarray, bg: int) -> np.ndarray:
    out = frame.copy()
    out[out == -1] = int(bg)
    return out


def _expected_output(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        frame_patch = _frame_patch(spec["frame"], bg)
        _paste_mask(out, frame_patch, spec["frame_r"], spec["frame_c"], _frame_mask(spec["frame"]))
        _paste_mask(out, spec["docked_tool"], spec["frame_r"], spec["frame_c"], _tool_mask(spec["docked_tool"]))
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    bg = rng.choice(PALETTE)
    colors = [c for c in PALETTE if c != bg]
    h = rng.randint(16, 26)
    w = rng.randint(16, 26)
    grid = np.full((h, w), int(bg), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    pair_count = rng.randint(1, 3)
    cavity_sizes_all = [3, 4, 5, 6, 7]
    rng.shuffle(cavity_sizes_all)
    cavity_sizes = cavity_sizes_all[:pair_count]
    handle_sizes = [sz for sz in [2, 3, 4, 5, 6] if sz not in cavity_sizes]
    rng.shuffle(handle_sizes)

    specs: list[dict] = []
    for i in range(pair_count):
        frame_color, plug_color, handle_color = rng.sample(colors, 3)
        side = rng.choice(["left", "right", "top", "bottom"])
        cavity_area = cavity_sizes[i]
        handle_area = handle_sizes[i]
        frame_left, tool_left = _build_left_pair(rng, frame_color, plug_color, handle_color, cavity_area, handle_area)
        frame = _orient_left(frame_left, side, bg)
        docked_tool = _orient_left(tool_left, side, 0)
        fm = _frame_mask(frame)
        th, tw = frame.shape

        placed = False
        for _ in range(200):
            if side == "left":
                r0 = rng.randint(1, h - th - 2)
                c0 = 0
            elif side == "right":
                r0 = rng.randint(1, h - th - 2)
                c0 = w - tw
            elif side == "top":
                r0 = 0
                c0 = rng.randint(1, w - tw - 2)
            else:
                r0 = h - th
                c0 = rng.randint(1, w - tw - 2)
            if _can_place(occ, fm, r0, c0, gap=2):
                _paste_mask(grid, _frame_patch(frame, bg), r0, c0, fm)
                occ[r0 : r0 + th, c0 : c0 + tw] |= fm
                specs.append(
                    {
                        "frame": frame,
                        "frame_r": r0,
                        "frame_c": c0,
                        "docked_tool": docked_tool,
                    }
                )
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place frame")

    expected = _expected_output(specs, (h, w), bg)

    flip_target = rng.choice([False, True])
    used_flip = False
    for idx, spec in enumerate(specs):
        docked_tool = spec["docked_tool"]
        colors_here = sorted(int(x) for x in np.unique(docked_tool[docked_tool != 0]))
        choose_flip = flip_target if idx == 0 else rng.choice([False, True])
        k = rng.randint(0, 3)
        inv = docked_tool.copy()
        if choose_flip:
            inv = _swap_colors(inv, colors_here[0], colors_here[1])
            used_flip = True
        tool_in = _inverse_transform(inv, k, choose_flip)
        tm = _tool_mask(tool_in)
        th, tw = tool_in.shape

        placed = False
        positions = [(r, c) for r in range(1, h - th) for c in range(1, w - tw)]
        rng.shuffle(positions)
        for r0, c0 in positions:
            if _can_place(occ, tm, r0, c0, gap=1):
                _paste_mask(grid, tool_in, r0, c0, tm)
                occ[r0 : r0 + th, c0 : c0 + tw] |= tm
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place tool")

    if flip_target and not used_flip:
        raise RuntimeError("reflection branch was requested but not used")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        try:
            grid, expected = _generate_case(rng)
        except Exception:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
