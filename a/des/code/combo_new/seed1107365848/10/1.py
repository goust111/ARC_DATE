from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
NON_BG_COLORS = list(range(1, 10))
OPEN_SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
PAD = 1


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = [
    _mask_from_rows(["11"]),
    _mask_from_rows(["111"]),
    _mask_from_rows(["1111"]),
    _mask_from_rows(["11111"]),
    _mask_from_rows(["011", "110"]),
    _mask_from_rows(["011", "110"]),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _tight_component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = _crop_mask(np.rot90(mask, k))
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(rot)
    return variants


def _anchored_offset(
    inner_h: int, inner_w: int, piece_h: int, piece_w: int, open_side: str
) -> tuple[int, int]:
    if open_side == "top":
        return inner_h - piece_h, (inner_w - piece_w) // 2
    if open_side == "bottom":
        return 0, (inner_w - piece_w) // 2
    if open_side == "left":
        return (inner_h - piece_h) // 2, inner_w - piece_w
    if open_side == "right":
        return (inner_h - piece_h) // 2, 0
    raise ValueError(f"unknown open side: {open_side}")


def _classify_frame(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    rr, cc = np.argwhere(mask).T
    if np.any((rr > 0) & (rr < h - 1) & (cc > 0) & (cc < w - 1)):
        return None

    vals = patch[mask]
    colors, counts = np.unique(vals, return_counts=True)
    if len(colors) != 2:
        return None
    if np.any(colors == bg):
        return None
    count_map = {int(col): int(cnt) for col, cnt in zip(colors, counts)}
    tag_colors = [col for col, cnt in count_map.items() if cnt == 1]
    if len(tag_colors) != 1:
        return None
    tag_color = int(tag_colors[0])
    frame_color = int([col for col in count_map if col != tag_color][0])

    tag_pts = np.argwhere(patch == tag_color)
    if tag_pts.shape[0] != 1:
        return None
    tag_r, tag_c = map(int, tag_pts[0])
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    if (tag_r, tag_c) not in corners:
        return None
    if not all(mask[r, c] for r, c in corners):
        return None

    side_counts = {
        "top": int(mask[0, :].sum()),
        "bottom": int(mask[h - 1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, w - 1].sum()),
    }
    full_counts = {"top": w, "bottom": w, "left": h, "right": h}
    open_sides = [side for side, cnt in side_counts.items() if cnt == 2]
    if len(open_sides) != 1:
        return None
    open_side = open_sides[0]
    for side in OPEN_SIDES:
        want = 2 if side == open_side else full_counts[side]
        if side_counts[side] != want:
            return None

    inner_h, inner_w = h - 2, w - 2
    if inner_h < 1 or inner_w < 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": frame_color,
        "tag_color": tag_color,
        "open_side": open_side,
        "inner_h": inner_h,
        "inner_w": inner_w,
    }


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: list[dict] = []
    fragments: dict[int, dict] = {}

    for comp in _components(g != bg):
        frame = _classify_frame(g, comp, bg)
        if frame is not None:
            frames.append(frame)
            continue

        vals = [int(g[r, c]) for r, c in comp]
        palette = sorted(set(vals))
        assert len(palette) == 1, "non-frame fragments must be monochrome"
        color = palette[0]
        assert color != bg
        area = len(comp)
        assert 2 <= area <= 5, "fragment size must be between 2 and 5"
        fragments[color] = {
            "color": color,
            "coords": comp,
            "mask": _tight_component_mask(comp),
        }

    assert frames, "at least one frame is required"
    assert len(frames) == len(fragments), "frame/fragment counts must match"
    assert {frame["tag_color"] for frame in frames} == set(fragments), "tags must match fragments"
    return frames, fragments, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each corner-tagged loose fragment to its open frame, rotate it to fit, and recolor it with the frame color.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-thick rectangular U-frames appear in different colors; each frame keeps three full sides and has one open side.
    - Each frame has exactly one corner pixel replaced by a distinct tag color.
    - Elsewhere, there is exactly one small monochrome 4-connected fragment in each tag color.

    Rule:
    - Pair every frame with the loose fragment whose color equals the frame's corner tag color.
    - For each pair, test the fragment under 0/90/180/270-degree rotation and keep the unique orientation whose tight bounding box fits inside the frame interior while resting against the wall opposite the opening and staying centered along that wall.
    - Recolor the embedded fragment to the frame's own color, erase the original loose fragment to background, and leave the frame border plus the corner tag unchanged.

    Key insight: The corner tag is only a lookup key; the open side determines where the matched shape must be anchored, and the frame color becomes the fragment's final color.
    """
    g = np.asarray(grid, dtype=int)
    frames, fragments, bg = _extract_scene(g)
    out = g.copy()

    for frag in fragments.values():
        for r, c in frag["coords"]:
            out[r, c] = bg

    for frame in frames:
        frag = fragments[frame["tag_color"]]
        fits: list[tuple[np.ndarray, int, int]] = []
        for rot in _unique_rotations(frag["mask"]):
            piece_h, piece_w = map(int, rot.shape)
            if piece_h > frame["inner_h"] or piece_w > frame["inner_w"]:
                continue
            off_r, off_c = _anchored_offset(
                int(frame["inner_h"]),
                int(frame["inner_w"]),
                piece_h,
                piece_w,
                str(frame["open_side"]),
            )
            fits.append((rot, int(off_r), int(off_c)))

        assert len(fits) == 1, "every fragment must have a unique fitting rotation"
        rot, off_r, off_c = fits[0]
        r0, c0, _r1, _c1 = frame["bbox"]
        top = r0 + 1 + off_r
        left = c0 + 1 + off_c
        sub = out[top : top + rot.shape[0], left : left + rot.shape[1]]
        sub[rot] = int(frame["frame_color"])

    return out


def _make_frame_patch(
    inner_h: int, inner_w: int, open_side: str, frame_color: int, tag_color: int, tag_corner: tuple[int, int]
) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    if open_side == "top":
        patch[0, 1:-1] = BG
    elif open_side == "bottom":
        patch[-1, 1:-1] = BG
    elif open_side == "left":
        patch[1:-1, 0] = BG
    elif open_side == "right":
        patch[1:-1, -1] = BG
    else:
        raise ValueError(open_side)
    patch[tag_corner] = tag_color
    return patch


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _place_box(occupied: np.ndarray, box_h: int, box_w: int, pad: int = PAD) -> tuple[int, int] | None:
    H, W = occupied.shape
    if box_h > H or box_w > W:
        return None
    r_lo, r_hi = 1, H - box_h - 1
    c_lo, c_hi = 1, W - box_w - 1
    if r_lo > r_hi or c_lo > c_hi:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(r_lo, r_hi)
        left = random.randint(c_lo, c_hi)
        rr0 = max(0, top - pad)
        rr1 = min(H, top + box_h + pad)
        cc0 = max(0, left - pad)
        cc1 = min(W, left + box_w + pad)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        occupied[top : top + box_h, left : left + box_w] = True
        return int(top), int(left)
    return None


def _build_pair_specs() -> list[dict] | None:
    n_pairs = random.choice([2, 3, 4])
    palette = random.sample(NON_BG_COLORS, 2 * n_pairs)
    frame_colors = palette[:n_pairs]
    tag_colors = palette[n_pairs:]

    directions = list(random.sample(OPEN_SIDES, n_pairs))
    specs: list[dict] = []
    rotated_flags: list[bool] = []

    for idx in range(n_pairs):
        frame_color = frame_colors[idx]
        tag_color = tag_colors[idx]
        open_side = directions[idx]
        base_mask = np.array(random.choice(SHAPE_LIBRARY), copy=True)
        final_mask = base_mask if open_side in ("top", "bottom") else np.rot90(base_mask)
        final_mask = _crop_mask(final_mask)
        final_h, final_w = map(int, final_mask.shape)
        alt_h, alt_w = map(int, np.rot90(final_mask).shape)
        assert (final_h, final_w) != (alt_h, alt_w)

        if open_side in ("top", "bottom"):
            inner_h = final_h
            inner_w = final_w + random.randint(0, 2)
            assert alt_h > inner_h
        else:
            inner_h = final_h + random.randint(0, 2)
            inner_w = final_w
            assert alt_w > inner_w

        variants = _unique_rotations(final_mask)
        input_mask = np.array(random.choice(variants), copy=True)
        rotated_flags.append(not np.array_equal(input_mask, final_mask))

        tag_corner = random.choice(
            [(0, 0), (0, inner_w + 1), (inner_h + 1, 0), (inner_h + 1, inner_w + 1)]
        )
        frame_patch = _make_frame_patch(inner_h, inner_w, open_side, frame_color, tag_color, tag_corner)
        frag_patch = np.full(input_mask.shape, BG, dtype=int)
        frag_patch[input_mask] = tag_color

        specs.append(
            {
                "frame_color": frame_color,
                "tag_color": tag_color,
                "open_side": open_side,
                "frame_patch": frame_patch,
                "frag_patch": frag_patch,
                "frag_mask": input_mask,
                "final_mask": final_mask,
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )

    if not any(rotated_flags):
        idx = random.randrange(len(specs))
        variants = _unique_rotations(specs[idx]["final_mask"])
        alt_variants = [m for m in variants if not np.array_equal(m, specs[idx]["final_mask"])]
        if alt_variants:
            new_mask = np.array(random.choice(alt_variants), copy=True)
            specs[idx]["frag_mask"] = new_mask
            specs[idx]["frag_patch"] = np.full(new_mask.shape, BG, dtype=int)
            specs[idx]["frag_patch"][new_mask] = specs[idx]["tag_color"]

    return specs


def _render_from_specs(specs: list[dict], H: int, W: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    frame_positions: list[tuple[int, int]] = []
    fragment_positions: list[tuple[int, int]] = []

    frame_order = sorted(range(len(specs)), key=lambda i: specs[i]["frame_patch"].size, reverse=True)
    for idx in frame_order:
        patch = specs[idx]["frame_patch"]
        pos = _place_box(occupied, int(patch.shape[0]), int(patch.shape[1]))
        if pos is None:
            return None
        while len(frame_positions) <= idx:
            frame_positions.append((0, 0))
        frame_positions[idx] = pos

    frag_order = sorted(range(len(specs)), key=lambda i: specs[i]["frag_patch"].size, reverse=True)
    for idx in frag_order:
        patch = specs[idx]["frag_patch"]
        pos = _place_box(occupied, int(patch.shape[0]), int(patch.shape[1]))
        if pos is None:
            return None
        while len(fragment_positions) <= idx:
            fragment_positions.append((0, 0))
        fragment_positions[idx] = pos

    for idx, spec in enumerate(specs):
        fr, fc = frame_positions[idx]
        pr, pc = fragment_positions[idx]
        _paint_patch(inp, spec["frame_patch"], fr, fc)
        _paint_patch(inp, spec["frag_patch"], pr, pc)

        _paint_patch(out, spec["frame_patch"], fr, fc)
        off_r, off_c = _anchored_offset(
            int(spec["inner_h"]),
            int(spec["inner_w"]),
            int(spec["final_mask"].shape[0]),
            int(spec["final_mask"].shape[1]),
            str(spec["open_side"]),
        )
        top = fr + 1 + off_r
        left = fc + 1 + off_c
        sub = out[top : top + spec["final_mask"].shape[0], left : left + spec["final_mask"].shape[1]]
        sub[spec["final_mask"]] = int(spec["frame_color"])

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    specs = _build_pair_specs()
    if not specs:
        return None

    total_area = sum(int(spec["frame_patch"].size + spec["frag_patch"].size) for spec in specs)
    base_side = max(18, int(np.ceil(np.sqrt(total_area * 3.2))))
    H = random.randint(base_side, 30)
    W = random.randint(base_side, 30)

    rendered = _render_from_specs(specs, H, W)
    if rendered is None:
        return None
    inp, expected = rendered

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
