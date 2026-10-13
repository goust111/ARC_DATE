import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask must be non-empty"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            key = (int(cand.shape[0]), int(cand.shape[1]), cand.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cand.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (int(v.shape[0]), int(v.shape[1]), v.astype(np.uint8).tobytes())
        for v in _dihedral_variants(mask)
    )


def _find_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY):
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            if (r1 - r0 + 1) * (c1 - c0 + 1) != int(coords.shape[0]):
                continue
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            fr0, fc0, fr1, fc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
            if (fr1 - fr0 + 1) < 5 or (fc1 - fc0 + 1) < 5:
                continue
            sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
            if not np.all(sub[1:-1, 1:-1] == color):
                continue
            border = _perimeter_mask(*sub.shape)
            border_vals = sub[border]
            if np.any((border_vals != BG) & (border_vals != GREY)):
                continue
            if not np.any(border_vals == BG) or not np.any(border_vals == GREY):
                continue
            gap_mask = border & (sub == BG)
            frames.append(
                {
                    "bbox": (fr0, fc0, fr1, fc1),
                    "interior_color": color,
                    "gap_mask": gap_mask,
                    "gap_crop": _crop_mask(gap_mask),
                    "key": _canonical_key(_crop_mask(gap_mask)),
                }
            )
    frames.sort(key=lambda fr: fr["bbox"])
    return frames


def _extract_patches(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    blocked = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    patches: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY):
            continue
        pts = np.argwhere((grid == color) & ~blocked)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(pts)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
        patches.append(
            {
                "color": color,
                "coords": pts,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "key": _canonical_key(mask),
            }
        )
    patches.sort(key=lambda p: p["bbox"])
    return patches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each broken frame to its shape-equal repair patch, then recolor the whole frame border with the patch's color.

    Input structure:
    - Background is black.
    - Several grey one-cell-thick rectangular frames enclose solid monochrome interiors.
    - Each frame border has one missing pattern where border cells are replaced by black background.
    - Outside the frames are small monochrome repair-patch sprites whose shapes match frame gap patterns up to rotation or reflection.

    Rule:
    - Detect each frame by finding a solid interior rectangle whose surrounding ring consists only of grey border cells plus black missing cells.
    - Extract the missing border pattern for every frame and compare it to each outside patch shape under all dihedral variants.
    - For every unique match, erase the outside patch, restore the missing border cells, and recolor the entire border of that frame to the patch's color.
    - Leave each frame interior unchanged and keep the black background elsewhere.

    Key insight:
    - The outside patch shape identifies which frame it repairs, while the patch color is transferred onto the repaired frame border.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert frames, "expected at least one broken frame"
    patches = _extract_patches(g, frames)
    assert patches, "expected at least one repair patch"

    frames_by_key: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for frame in frames:
        frames_by_key[frame["key"]].append(frame)
    patches_by_key: dict[tuple[int, int, bytes], list[dict]] = defaultdict(list)
    for patch in patches:
        patches_by_key[patch["key"]].append(patch)

    assert set(frames_by_key) == set(patches_by_key), "gap shapes and patch shapes must match exactly"
    for key in frames_by_key:
        assert len(frames_by_key[key]) == 1, "each gap shape must identify a unique frame"
        assert len(patches_by_key[key]) == 1, "each patch shape must identify a unique frame"

    out = g.copy()
    for patch in patches:
        out[patch["coords"][:, 0], patch["coords"][:, 1]] = BG

    for key, frame_list in frames_by_key.items():
        frame = frame_list[0]
        patch = patches_by_key[key][0]
        r0, c0, r1, c1 = frame["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[_perimeter_mask(*sub.shape)] = int(patch["color"])

    return out


SHAPE_LIBRARY = [
    ("line", 1),
    ("line", 2),
    ("line", 3),
    ("line", 4),
    ("line", 5),
    ("corner", 2, 2),
    ("corner", 2, 3),
    ("corner", 2, 4),
    ("corner", 3, 3),
    ("corner", 3, 4),
]


def _shape_kind(spec: tuple) -> str:
    return str(spec[0])


def _spec_fits(spec: tuple, frame_h: int, frame_w: int) -> bool:
    if spec[0] == "line":
        return int(spec[1]) <= max(frame_h, frame_w)
    a, b = int(spec[1]), int(spec[2])
    return (frame_h >= a and frame_w >= b) or (frame_h >= b and frame_w >= a)


def _gap_mask_from_spec(frame_h: int, frame_w: int, spec: tuple) -> np.ndarray:
    mask = np.zeros((frame_h, frame_w), dtype=bool)
    if spec[0] == "line":
        length = int(spec[1])
        options: list[tuple[str, int, int]] = []
        if frame_w >= length:
            for row in (0, frame_h - 1):
                for start in range(frame_w - length + 1):
                    options.append(("h", row, start))
        if frame_h >= length:
            for col in (0, frame_w - 1):
                for start in range(frame_h - length + 1):
                    options.append(("v", col, start))
        mode, fixed, start = random.choice(options)
        if mode == "h":
            mask[fixed, start : start + length] = True
        else:
            mask[start : start + length, fixed] = True
        return mask

    a, b = int(spec[1]), int(spec[2])
    corner_options: list[tuple[str, int, int]] = []
    for vert, horiz in {(a, b), (b, a)}:
        if frame_h >= vert and frame_w >= horiz:
            for corner in ("tl", "tr", "bl", "br"):
                corner_options.append((corner, vert, horiz))
    corner, vert, horiz = random.choice(corner_options)
    if corner == "tl":
        mask[0, :horiz] = True
        mask[:vert, 0] = True
    elif corner == "tr":
        mask[0, frame_w - horiz :] = True
        mask[:vert, frame_w - 1] = True
    elif corner == "bl":
        mask[frame_h - 1, :horiz] = True
        mask[frame_h - vert :, 0] = True
    else:
        mask[frame_h - 1, frame_w - horiz :] = True
        mask[frame_h - vert :, frame_w - 1] = True
    return mask


def _random_free_top_left(occupied: np.ndarray, h: int, w: int, pad: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    choices: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - pad)
            c0 = max(0, c - pad)
            r1 = min(H, r + h + pad)
            c1 = min(W, c + w + pad)
            if not occupied[r0:r1, c0:c1].any():
                choices.append((r, c))
    if not choices:
        return None
    return random.choice(choices)


def _mark_occupied(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    H, W = occupied.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(H, r + h + pad)
    c1 = min(W, c + w + pad)
    occupied[r0:r1, c0:c1] = True


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    coords = np.argwhere(mask)
    grid[top + coords[:, 0], left + coords[:, 1]] = int(color)


def _build_expected(
    inp: np.ndarray, frame_specs: list[dict], patch_specs: list[dict]
) -> np.ndarray:
    out = inp.copy()
    for patch in patch_specs:
        _paint_mask(out, int(patch["top"]), int(patch["left"]), patch["mask"], BG)
    for frame, patch in zip(frame_specs, patch_specs):
        top, left = int(frame["top"]), int(frame["left"])
        h, w = int(frame["h"]), int(frame["w"])
        sub = out[top : top + h, left : left + w]
        sub[_perimeter_mask(h, w)] = int(patch["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choices([2, 3, 4], weights=[4, 4, 2], k=1)[0]
    H = random.randint(18, 30)
    W = random.randint(18, 30)

    colors = random.sample(COLOR_POOL, 2 * n_frames)
    interior_colors = colors[:n_frames]
    patch_colors = colors[n_frames:]

    frame_dims = [(random.randint(5, 8), random.randint(5, 8)) for _ in range(n_frames)]
    specs: list[tuple] = []
    used_specs: set[tuple] = set()
    for idx, (fh, fw) in enumerate(frame_dims):
        eligible_lines = [
            spec for spec in SHAPE_LIBRARY if spec not in used_specs and spec[0] == "line" and _spec_fits(spec, fh, fw)
        ]
        eligible_corners = [
            spec for spec in SHAPE_LIBRARY if spec not in used_specs and spec[0] == "corner" and _spec_fits(spec, fh, fw)
        ]
        if not eligible_lines and not eligible_corners:
            return None
        if idx == 0:
            kind = random.choice([k for k, ok in [("line", bool(eligible_lines)), ("corner", bool(eligible_corners))] if ok])
        else:
            kind = random.choice([k for k, ok in [("line", bool(eligible_lines)), ("corner", bool(eligible_corners))] if ok])
        if kind == "line":
            spec = random.choice(eligible_lines)
        else:
            spec = random.choice(eligible_corners)
        used_specs.add(spec)
        specs.append(spec)

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = sorted(range(n_frames), key=lambda i: frame_dims[i][0] * frame_dims[i][1], reverse=True)
    frame_specs: list[dict | None] = [None] * n_frames
    for i in order:
        fh, fw = frame_dims[i]
        pos = _random_free_top_left(occupied, fh, fw, pad=1)
        if pos is None:
            return None
        top, left = pos
        _mark_occupied(occupied, top, left, fh, fw, pad=1)
        gap_mask = _gap_mask_from_spec(fh, fw, specs[i])
        frame_specs[i] = {
            "top": top,
            "left": left,
            "h": fh,
            "w": fw,
            "gap_mask": gap_mask,
            "interior_color": interior_colors[i],
        }
        frame_view = grid[top : top + fh, left : left + fw]
        frame_view[_perimeter_mask(fh, fw)] = GREY
        grid[top + 1 : top + fh - 1, left + 1 : left + fw - 1] = interior_colors[i]
        _paint_mask(grid, top, left, gap_mask, BG)

    patch_specs: list[dict] = []
    patch_order = list(range(n_frames))
    random.shuffle(patch_order)
    for i in patch_order:
        frame = frame_specs[i]
        assert frame is not None
        gap_crop = _crop_mask(frame["gap_mask"])
        patch_mask = random.choice(_dihedral_variants(gap_crop))
        ph, pw = patch_mask.shape
        pos = _random_free_top_left(occupied, ph, pw, pad=1)
        if pos is None:
            return None
        top, left = pos
        _mark_occupied(occupied, top, left, ph, pw, pad=1)
        _paint_mask(grid, top, left, patch_mask, patch_colors[i])
        patch_specs.append(
            {
                "frame_index": i,
                "top": top,
                "left": left,
                "mask": patch_mask,
                "color": patch_colors[i],
            }
        )

    patch_specs.sort(key=lambda p: int(p["frame_index"]))
    frame_specs_final = [frame for frame in frame_specs if frame is not None]
    expected = _build_expected(grid, frame_specs_final, patch_specs)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
