from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import label


BG = 0
FRAME_COLORS = (1, 2, 3)
FRAGMENT = 6
FILL = 9

STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)

GEN_MASTER = 26
GEN_MIN_SIZE = 18
GEN_MAX_ATTEMPTS = 400
GEN_BLOCK_GAP = 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    lab, n = label(mask.astype(bool), structure=struct)
    return lab, int(n)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in FRAME_COLORS:
        lab, n = label_mask(grid == int(color), connectivity=4)
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.array_equal(patch == int(color), _frame_mask(h, w)):
                continue
            frames.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                }
            )
    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)
    assert len(frames) in (4, 6), "expected 4 or 6 hollow frames"
    assert all(len(parts) == 2 for parts in by_color.values()), "each frame color must appear exactly twice"
    return frames


def _pair_edge(grid: np.ndarray, bg: int, a: dict, b: dict) -> dict:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    candidates: list[dict] = []

    if ac1 < bc0 or bc1 < ac0:
        if ac1 < bc0:
            left, right = a, b
        else:
            left, right = b, a
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        r0 = max(lr0 + 1, rr0 + 1)
        r1 = min(lr1 - 1, rr1 - 1)
        c0 = lc1 + 1
        c1 = rc0 - 1
        if r0 <= r1 and c0 <= c1 and bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)).all()):
            candidates.append(
                {
                    "kind": "h",
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "left": left["bbox"],
                    "right": right["bbox"],
                }
            )

    if ar1 < br0 or br1 < ar0:
        if ar1 < br0:
            upper, lower = a, b
        else:
            upper, lower = b, a
        ur0, uc0, ur1, uc1 = upper["bbox"]
        lr0, lc0, lr1, lc1 = lower["bbox"]
        c0 = max(uc0 + 1, lc0 + 1)
        c1 = min(uc1 - 1, lc1 - 1)
        r0 = ur1 + 1
        r1 = lr0 - 1
        if c0 <= c1 and r0 <= r1 and bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)).all()):
            candidates.append(
                {
                    "kind": "v",
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "upper": upper["bbox"],
                    "lower": lower["bbox"],
                }
            )

    assert len(candidates) == 1, "same-color frames must have one visible horizontal or vertical gap"
    return candidates[0]


def _apply_corridor(out: np.ndarray, bg: int, color: int, edge: dict) -> None:
    r0 = int(edge["r0"])
    r1 = int(edge["r1"])
    c0 = int(edge["c0"])
    c1 = int(edge["c1"])
    if edge["kind"] == "h":
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        left = edge["left"]
        right = edge["right"]
        if r0 + 1 <= r1 - 1:
            out[r0 + 1 : r1, left[3]] = bg
            out[r0 + 1 : r1, right[1]] = bg
    else:
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        upper = edge["upper"]
        lower = edge["lower"]
        if c0 + 1 <= c1 - 1:
            out[upper[2], c0 + 1 : c1] = bg
            out[lower[0], c0 + 1 : c1] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored hollow rectangles, open a corridor between each pair, and fill each fragment's complement inside the rooms.

    Input structure:
    - The background is black.
    - There are 4 or 6 one-pixel-thick hollow rectangular frames in blue, red, and/or green; each used frame color appears exactly twice.
    - Each frame interior contains one 4-connected magenta fragment on background, and same-colored frame pairs have equal interior size plus a clear horizontal or vertical visible gap.

    Rule:
    - Pair frames by border color.
    - For each pair, find the unique clear horizontal or vertical gap whose perpendicular overlap is between the two interiors.
    - Draw corridor walls in the pair color along the two boundary rows or columns of that overlap across the gap, and erase the facing wall pixels strictly between those boundaries to open the passage.
    - Inside every frame, replace all remaining background cells with maroon while keeping the magenta fragment unchanged; corridor interiors stay background.

    Key insight:
    - Color determines which two rooms belong together; the corridor geometry comes from the overlap of their interiors, and each room is then completed locally by filling the fragment's missing cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = _detect_frames(g)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)

    out = g.copy()
    for color, pair in by_color.items():
        assert len(pair) == 2
        edge = _pair_edge(g, bg, pair[0], pair[1])
        _apply_corridor(out, bg, int(color), edge)

    for frame in frames:
        r0, c0, r1, c1 = frame["inner_bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == int(bg)] = FILL
    return out


def _component_count(mask: np.ndarray, connectivity: int = 4) -> int:
    return label_mask(mask, connectivity=connectivity)[1]


def _make_fragment_pair(rng: random.Random, small: bool) -> tuple[np.ndarray, np.ndarray]:
    h_choices = (3, 4) if small else (3, 4, 5)
    w_choices = (3, 4) if small else (3, 4, 5)
    for _ in range(300):
        h = rng.choice(h_choices)
        w = rng.choice(w_choices)
        lo = math.ceil(0.4 * h * w)
        hi = math.floor(0.6 * h * w)
        widths = [rng.randint(1, w - 1)]
        for _row in range(1, h):
            prev = widths[-1]
            if rng.random() < 0.75:
                low = max(1, prev - 1)
                high = min(w - 1, prev + 1)
                widths.append(rng.randint(low, high))
            else:
                widths.append(rng.randint(1, w - 1))
        area = int(sum(widths))
        if not (lo <= area <= hi):
            continue
        if len(set(widths)) == 1:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, width in enumerate(widths):
            mask[r, :width] = True
        other = ~mask
        if _component_count(mask, connectivity=4) != 1 or _component_count(other, connectivity=4) != 1:
            continue

        k = rng.randrange(4)
        mask = np.rot90(mask, k)
        other = np.rot90(other, k)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
            other = np.flipud(other)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
            other = np.fliplr(other)
        if rng.random() < 0.5:
            mask, other = other, mask
        return mask.astype(bool), other.astype(bool)
    raise RuntimeError("failed to generate complementary fragments")


def _make_pair_spec(color: int, orientation: str, rng: random.Random, small: bool) -> dict:
    mask_a, mask_b = _make_fragment_pair(rng, small=small)
    h, w = map(int, mask_a.shape)
    fh, fw = h + 2, w + 2
    gap = rng.randint(1, 4)

    if orientation == "h":
        overlap = rng.randint(1, h)
        delta = h - overlap
        if rng.random() < 0.5:
            a_r0, b_r0 = 0, delta
        else:
            a_r0, b_r0 = delta, 0
        a_c0, b_c0 = 0, fw + gap
        block_h = fh + delta
        block_w = fw * 2 + gap
    else:
        overlap = rng.randint(1, w)
        delta = w - overlap
        if rng.random() < 0.5:
            a_c0, b_c0 = 0, delta
        else:
            a_c0, b_c0 = delta, 0
        a_r0, b_r0 = 0, fh + gap
        block_h = fh * 2 + gap
        block_w = fw + delta

    return {
        "color": int(color),
        "orientation": orientation,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "inner_h": int(h),
        "inner_w": int(w),
        "a_local": (int(a_r0), int(a_c0)),
        "b_local": (int(b_r0), int(b_c0)),
        "block_h": int(block_h),
        "block_w": int(block_w),
    }


def _rect_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return bool(ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_specs(specs: list[dict], rng: random.Random) -> dict[int, tuple[int, int]] | None:
    order = sorted(range(len(specs)), key=lambda i: specs[i]["block_h"] * specs[i]["block_w"], reverse=True)
    placed: dict[int, tuple[int, int]] = {}
    rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        spec = specs[idx]
        bh = int(spec["block_h"])
        bw = int(spec["block_w"])
        options: list[tuple[float, int, int]] = []
        for r0 in range(GEN_MASTER - bh + 1):
            for c0 in range(GEN_MASTER - bw + 1):
                rect = (r0, c0, r0 + bh - 1, c0 + bw - 1)
                if all(_rect_far(rect, other, GEN_BLOCK_GAP) for other in rects):
                    cr = abs((r0 + r0 + bh - 1) / 2.0 - (GEN_MASTER - 1) / 2.0)
                    cc = abs((c0 + c0 + bw - 1) / 2.0 - (GEN_MASTER - 1) / 2.0)
                    options.append((cr + cc + rng.random() * 0.25, r0, c0))
        if not options:
            return None
        options.sort(key=lambda item: item[0])
        _, r0, c0 = rng.choice(options[: min(12, len(options))])
        placed[idx] = (int(r0), int(c0))
        rects.append((int(r0), int(c0), int(r0 + bh - 1), int(c0 + bw - 1)))
    return placed


def _frame_bboxes_for_spec(spec: dict, origin: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    base_r, base_c = origin
    h = int(spec["inner_h"])
    w = int(spec["inner_w"])
    ar0 = base_r + int(spec["a_local"][0])
    ac0 = base_c + int(spec["a_local"][1])
    br0 = base_r + int(spec["b_local"][0])
    bc0 = base_c + int(spec["b_local"][1])
    a = (int(ar0), int(ac0), int(ar0 + h + 1), int(ac0 + w + 1))
    b = (int(br0), int(bc0), int(br0 + h + 1), int(bc0 + w + 1))
    return a, b


def _render_input(specs: list[dict], placements: dict[int, tuple[int, int]]) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((GEN_MASTER, GEN_MASTER), dtype=int)
    frames: list[dict] = []
    for idx, spec in enumerate(specs):
        a_bbox, b_bbox = _frame_bboxes_for_spec(spec, placements[idx])
        _draw_frame(grid, a_bbox, int(spec["color"]))
        _draw_frame(grid, b_bbox, int(spec["color"]))

        ar0, ac0, ar1, ac1 = a_bbox
        br0, bc0, br1, bc1 = b_bbox
        a_patch = grid[ar0 + 1 : ar1, ac0 + 1 : ac1]
        b_patch = grid[br0 + 1 : br1, bc0 + 1 : bc1]
        a_patch[spec["mask_a"]] = FRAGMENT
        b_patch[spec["mask_b"]] = FRAGMENT

        frames.append({"color": int(spec["color"]), "bbox": a_bbox, "inner_bbox": (ar0 + 1, ac0 + 1, ar1 - 1, ac1 - 1)})
        frames.append({"color": int(spec["color"]), "bbox": b_bbox, "inner_bbox": (br0 + 1, bc0 + 1, br1 - 1, bc1 - 1)})
    return grid, frames


def _render_expected(inp: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg = get_bg(inp)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        by_color[int(frame["color"])].append(frame)
    for color, pair in by_color.items():
        edge = _pair_edge(inp, bg, pair[0], pair[1])
        _apply_corridor(out, bg, int(color), edge)
    for frame in frames:
        r0, c0, r1, c1 = frame["inner_bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = FILL
    return out


def _crop_with_random_margin(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    r0, c0, r1, c1 = bbox_coords(coords)
    content_h = int(r1 - r0 + 1)
    content_w = int(c1 - c0 + 1)

    target_h = rng.randint(max(GEN_MIN_SIZE, content_h), GEN_MASTER)
    target_w = rng.randint(max(GEN_MIN_SIZE, content_w), GEN_MASTER)

    free_top = int(r0)
    free_bottom = int(GEN_MASTER - 1 - r1)
    extra_h = int(target_h - content_h)
    top_pad_lo = max(0, extra_h - free_bottom)
    top_pad_hi = min(free_top, extra_h)
    top_pad = rng.randint(top_pad_lo, top_pad_hi) if top_pad_lo <= top_pad_hi else 0
    bottom_pad = extra_h - top_pad

    free_left = int(c0)
    free_right = int(GEN_MASTER - 1 - c1)
    extra_w = int(target_w - content_w)
    left_pad_lo = max(0, extra_w - free_right)
    left_pad_hi = min(free_left, extra_w)
    left_pad = rng.randint(left_pad_lo, left_pad_hi) if left_pad_lo <= left_pad_hi else 0
    right_pad = extra_w - left_pad

    rr0 = int(r0 - top_pad)
    rr1 = int(r1 + bottom_pad)
    cc0 = int(c0 - left_pad)
    cc1 = int(c1 + right_pad)
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _valid_fragment_areas(grid: np.ndarray, frames: list[dict]) -> bool:
    for frame in frames:
        r0, c0, r1, c1 = frame["inner_bbox"]
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] == FRAGMENT
        area = int(mask.sum())
        total = int(mask.size)
        if _component_count(mask, connectivity=4) != 1:
            return False
        if not (math.ceil(0.4 * total) <= area <= math.floor(0.6 * total)):
            return False
    return True


def _sample_orientations(n_pairs: int, rng: random.Random) -> list[str]:
    if n_pairs == 2:
        if rng.random() < 0.75:
            orientations = ["h", "v"]
            rng.shuffle(orientations)
            return orientations
        return [rng.choice(["h", "v"]) for _ in range(2)]
    orientations = ["h", "v", rng.choice(["h", "v"])]
    rng.shuffle(orientations)
    return orientations


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.choice([2, 3])
    colors = list(rng.sample(FRAME_COLORS, n_pairs))
    orientations = _sample_orientations(n_pairs, rng)
    small = n_pairs == 3

    specs: list[dict] = []
    for color, orientation in zip(colors, orientations):
        spec = _make_pair_spec(int(color), orientation, rng, small=small)
        if spec["block_h"] > GEN_MASTER or spec["block_w"] > GEN_MASTER:
            return None
        specs.append(spec)

    placements = _place_specs(specs, rng)
    if placements is None:
        return None

    inp, frames = _render_input(specs, placements)
    out = _render_expected(inp, frames)
    inp, out = _crop_with_random_margin(inp, out, rng)

    try:
        detected = _detect_frames(inp)
    except AssertionError:
        return None
    if len(detected) != 2 * n_pairs:
        return None
    if not _valid_fragment_areas(inp, detected):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
