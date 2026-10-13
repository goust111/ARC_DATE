from __future__ import annotations

import random
from collections import deque

import numpy as np


TRANSFORMS = [(rot, flip) for flip in (False, True) for rot in range(4)]
GEN_RETRIES = 500
MAX_GRID = 30


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


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
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_true(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return (int(arr.shape[0]), int(arr.shape[1]), arr.tobytes())


def _apply_transform(mask: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.fliplr(mask) if flip else mask
    return np.rot90(out, int(rot)).copy()


def _canonical_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_signature(_apply_transform(mask, rot, flip)) for rot, flip in TRANSFORMS)


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _fill_holes(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = occ.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if occ[r, c] or ext[r, c]:
                continue
            ext[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not occ[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    holes = (~occ) & (~ext)
    return occ | holes, holes


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frames: list[dict] = []
    frame_mask = np.zeros(grid.shape, dtype=bool)
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            occ = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            occ[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = np.zeros_like(occ)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(occ, border):
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            inner_colors = set(map(int, np.unique(inner))) - {bg, color}
            if not inner_colors:
                continue
            frame_mask[coords[:, 0], coords[:, 1]] = True
            frames.append({"color": int(color), "bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda fr: fr["bbox"])
    assert frames, "expected at least one frame"
    return frames, frame_mask


def _extract_templates(grid: np.ndarray, bg: int, frames: list[dict], frame_mask: np.ndarray) -> dict[int, dict]:
    templates: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["color"])
        comps = _components((grid == color) & (~frame_mask))
        assert len(comps) == 1, f"expected exactly one template for frame color {color}"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox_coords(coords)
        occ = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        occ[coords[:, 0] - r0, coords[:, 1] - c0] = True
        filled, holes = _fill_holes(occ)
        assert holes.any(), "template must contain at least one hole"
        templates[color] = {
            "bbox": (r0, c0, r1, c1),
            "occ": occ,
            "silhouette": filled,
            "holes": holes,
        }
    return templates


def _frame_contents(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int, frame_color: int
) -> dict:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    vals, cnts = np.unique(inner, return_counts=True)
    colors = [(int(v), int(cnt)) for v, cnt in zip(vals.tolist(), cnts.tolist(), strict=False) if int(v) not in (bg, frame_color)]
    assert len(colors) == 2, "frame interior must contain exactly fill and key colors"
    colors.sort(key=lambda item: (-item[1], item[0]))
    fill_color = int(colors[0][0])
    key_color = int(colors[1][0])

    key_full = inner == key_color
    key_comps = _components(key_full)
    assert len(key_comps) == 1, "key must be a single component"
    key_coords = key_comps[0]
    kr0, kc0, kr1, kc1 = _bbox_coords(key_coords)
    key_mask = np.zeros((kr1 - kr0 + 1, kc1 - kc0 + 1), dtype=bool)
    key_mask[key_coords[:, 0] - kr0, key_coords[:, 1] - kc0] = True

    return {
        "fill_color": fill_color,
        "key_color": key_color,
        "key_full": key_full,
        "key_bbox": (kr0, kc0, kr1, kc1),
        "key_mask": key_mask,
    }


def _matched_holes(template: dict, key_mask: np.ndarray) -> np.ndarray:
    candidates: dict[tuple[int, int, bytes], np.ndarray] = {}
    for rot, flip in TRANSFORMS:
        sil = _apply_transform(template["silhouette"], rot, flip)
        if sil.shape != key_mask.shape:
            continue
        if not np.array_equal(sil, key_mask):
            continue
        holes = _apply_transform(template["holes"], rot, flip)
        candidates[_mask_signature(holes)] = holes
    assert candidates, "key must match its template silhouette"
    assert len(candidates) == 1, "template transform must determine a unique hole pattern"
    return next(iter(candidates.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed key-shape to an external template sprite and carve the template's holes into the frame fill.

    Input structure:
    - The background is the most common color.
    - Several 1-pixel-thick monochrome rectangular frames appear, each in a distinct border color and enclosing a solid fill color.
    - Outside the frames are small two-color template sprites whose shell color matches a frame border color and whose interiors contain background-colored holes.
    - Inside every frame is a smaller key-colored shape whose silhouette matches one template shell up to rotation or reflection.

    Rule:
    - Detect each rectangular frame and isolate its interior fill color and key shape.
    - Find the external template with the same shell color, then match the key shape to the template's filled silhouette under the allowed rotations/reflections.
    - Apply that same geometric transform to the template's hole mask and align it to the key shape's position inside the frame.
    - Restore the key pixels back to the frame's fill color, then erase the transformed hole pixels to background while leaving the rest of the grid unchanged.

    Key insight:
    - The inner colored blob is a geometric key: it selects a same-colored template shell and also fixes the exact orientation needed to transfer the template's hole pattern back into the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames, frame_mask = _find_frames(g, bg)
    templates = _extract_templates(g, bg, frames, frame_mask)

    out = g.copy()
    for frame in frames:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        info = _frame_contents(g, frame["bbox"], bg, frame_color)
        fill_color = int(info["fill_color"])
        key_full = info["key_full"]
        kr0, kc0, kr1, kc1 = info["key_bbox"]
        key_mask = info["key_mask"]

        holes = _matched_holes(templates[frame_color], key_mask)

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[key_full] = fill_color
        key_region = interior[kr0 : kr1 + 1, kc0 : kc1 + 1]
        key_region[holes] = bg

    return out


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    out.append((r, c))
                    break
    return out


def _interior_cells(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if all(mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                out[r, c] = True
    return out


def _random_silhouette() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        mask = np.ones((h, w), dtype=bool)
        target_removals = random.randint(1, min(5, h + w - 4))
        attempts = 0
        while target_removals > 0 and attempts < 60:
            attempts += 1
            boundary = _boundary_cells(mask)
            if not boundary:
                break
            r, c = random.choice(boundary)
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) < 9:
                continue
            if not _is_connected(trial):
                continue
            cropped, _ = _crop_true(trial)
            if int(_interior_cells(cropped).sum()) < 1:
                continue
            mask = cropped
            target_removals -= 1
        mask, _ = _crop_true(mask)
        if int(mask.sum()) < 9:
            continue
        if int(_interior_cells(mask).sum()) < 1:
            continue
        return mask
    return None


def _random_holes(silhouette: np.ndarray) -> np.ndarray | None:
    candidates = np.argwhere(_interior_cells(silhouette))
    if candidates.size == 0:
        return None
    cand_set = {tuple(map(int, xy)) for xy in candidates}
    max_holes = min(3, len(cand_set))
    for _ in range(120):
        hole_count = random.randint(1, max_holes)
        if random.random() < 0.5:
            start = tuple(map(int, random.choice(candidates)))
            chosen = {start}
            while len(chosen) < hole_count:
                frontier = []
                for r, c in chosen:
                    for nr, nc in _neighbors4(r, c):
                        if (nr, nc) in cand_set and (nr, nc) not in chosen:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                chosen.add(random.choice(frontier))
            if len(chosen) < hole_count:
                continue
        else:
            sample = random.sample(list(cand_set), hole_count)
            chosen = {tuple(map(int, xy)) for xy in sample}

        holes = np.zeros_like(silhouette, dtype=bool)
        for r, c in chosen:
            holes[r, c] = True
        shell = silhouette & (~holes)
        if not _is_connected(shell):
            continue
        filled, recovered_holes = _fill_holes(shell)
        if not np.array_equal(filled, silhouette):
            continue
        if not np.array_equal(recovered_holes, holes):
            continue
        return holes
    return None


def _make_shape() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(120):
        silhouette = _random_silhouette()
        if silhouette is None:
            continue
        holes = _random_holes(silhouette)
        if holes is None:
            continue
        return silhouette, holes
    return None


def _template_match_outputs(template_silhouette: np.ndarray, template_holes: np.ndarray, key_mask: np.ndarray) -> dict[tuple[int, int, bytes], np.ndarray]:
    out: dict[tuple[int, int, bytes], np.ndarray] = {}
    for rot, flip in TRANSFORMS:
        sil = _apply_transform(template_silhouette, rot, flip)
        if sil.shape != key_mask.shape:
            continue
        if np.array_equal(sil, key_mask):
            holes = _apply_transform(template_holes, rot, flip)
            out[_mask_signature(holes)] = holes
    return out


def _build_frame_patches(
    bg: int,
    border_color: int,
    fill_color: int,
    key_color: int,
    key_mask: np.ndarray,
    hole_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kh, kw = map(int, key_mask.shape)
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    while (kh * kw) >= ((kh + top + bottom) * (kw + left + right) - kh * kw):
        top += 1
        bottom += 1
    inner_h = kh + top + bottom
    inner_w = kw + left + right

    inp = np.full((inner_h + 2, inner_w + 2), fill_color, dtype=int)
    inp[0, :] = border_color
    inp[-1, :] = border_color
    inp[:, 0] = border_color
    inp[:, -1] = border_color

    key_region = inp[1 + top : 1 + top + kh, 1 + left : 1 + left + kw]
    key_region[key_mask] = key_color

    out = inp.copy()
    out_region = out[1 + top : 1 + top + kh, 1 + left : 1 + left + kw]
    out_region[key_mask] = fill_color
    out_region[hole_mask] = bg
    return inp, out


def _build_template_patch(bg: int, border_color: int, shell_mask: np.ndarray) -> np.ndarray:
    patch = np.full(shell_mask.shape, bg, dtype=int)
    patch[shell_mask] = border_color
    return patch


def _can_place(occupied: np.ndarray, shape: tuple[int, int], r0: int, c0: int, gap: int) -> bool:
    h, w = shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(H, r0 + h + gap)
    cc1 = min(W, c0 + w + gap)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_random(occupied: np.ndarray, shape: tuple[int, int], gap: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        if _can_place(occupied, shape, r, c, gap):
            occupied[r : r + h, c : c + w] = True
            return int(r), int(c)
    return None


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int | None = None) -> None:
    h, w = patch.shape
    if bg is None:
        grid[top : top + h, left : left + w] = patch
        return
    sub = grid[top : top + h, left : left + w]
    mask = patch != bg
    sub[mask] = patch[mask]


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.choice([1, 2, 3])
    bg = random.randrange(10)
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    if len(colors) < 3 * n_frames:
        return None

    used_shapes: set[tuple[int, int, bytes]] = set()
    objects: list[dict] = []

    for idx in range(n_frames):
        border_color = int(colors[3 * idx])
        fill_color = int(colors[3 * idx + 1])
        key_color = int(colors[3 * idx + 2])

        built = None
        for _ in range(120):
            shape = _make_shape()
            if shape is None:
                continue
            base_silhouette, base_holes = shape
            canon = _canonical_signature(base_silhouette)
            if canon in used_shapes:
                continue

            tpl_rot, tpl_flip = random.choice(TRANSFORMS)
            key_rot, key_flip = random.choice(TRANSFORMS)

            template_sil = _apply_transform(base_silhouette, tpl_rot, tpl_flip)
            template_holes = _apply_transform(base_holes, tpl_rot, tpl_flip)
            template_shell = template_sil & (~template_holes)
            key_mask = _apply_transform(base_silhouette, key_rot, key_flip)

            hole_outputs = _template_match_outputs(template_sil, template_holes, key_mask)
            if len(hole_outputs) != 1:
                continue
            matched_holes = next(iter(hole_outputs.values()))

            frame_in, frame_out = _build_frame_patches(
                bg, border_color, fill_color, key_color, key_mask, matched_holes
            )
            template_patch = _build_template_patch(bg, border_color, template_shell)

            built = {
                "frame_in": frame_in,
                "frame_out": frame_out,
                "template": template_patch,
                "shape_sig": canon,
            }
            break

        if built is None:
            return None

        used_shapes.add(built["shape_sig"])
        objects.append(built)

    patches: list[tuple[str, int, np.ndarray]] = []
    for i, obj in enumerate(objects):
        patches.append((f"frame_{i}", i, obj["frame_in"]))
        patches.append((f"template_{i}", i, obj["template"]))

    max_h = max(int(patch.shape[0]) for _, _, patch in patches)
    max_w = max(int(patch.shape[1]) for _, _, patch in patches)
    total_area = sum(int(patch.shape[0] * patch.shape[1]) for _, _, patch in patches)

    min_dim = max(max_h + 2, max_w + 2, 14)
    side_floor = int(np.ceil(np.sqrt(total_area * 2.3)))
    min_dim = max(min_dim, side_floor)
    if min_dim > MAX_GRID:
        return None

    for _ in range(80):
        H = random.randint(min_dim, MAX_GRID)
        W = random.randint(min_dim, MAX_GRID)
        occupied = np.zeros((H, W), dtype=bool)
        placements: dict[str, tuple[int, int]] = {}
        order = sorted(patches, key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)
        if random.random() < 0.5:
            random.shuffle(order)
            order.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)
        ok = True
        for name, idx, patch in order:
            pos = _place_random(occupied, patch.shape, gap=1)
            if pos is None:
                ok = False
                break
            placements[name] = pos
        if not ok:
            continue

        inp = np.full((H, W), bg, dtype=int)
        out = np.full((H, W), bg, dtype=int)

        for i, obj in enumerate(objects):
            fr, fc = placements[f"frame_{i}"]
            tr, tc = placements[f"template_{i}"]
            _paste(inp, obj["frame_in"], fr, fc)
            _paste(out, obj["frame_out"], fr, fc)
            _paste(inp, obj["template"], tr, tc, bg=bg)
            _paste(out, obj["template"], tr, tc, bg=bg)

        if int((inp == bg).sum()) <= int(inp.size // 2):
            continue
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sample = _generate_case()
        if sample is None:
            continue
        inp, expected = sample
        try:
            got = main(inp)
        except Exception:
            continue
        if got.shape != expected.shape:
            continue
        if np.array_equal(got, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
