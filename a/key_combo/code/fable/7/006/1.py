from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _bbox(mask: np.ndarray) -> BBox | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _connected_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_false_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (not mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                hole = np.zeros_like(mask, dtype=bool)
                for rr, cc in comp:
                    hole[rr, cc] = True
                holes.append(hole)
    return holes


def _apply_transform(arr: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(rot))
    if flip:
        out = np.fliplr(out)
    return out.copy()


def _transform_variants(arr: np.ndarray) -> list[tuple[int, bool, np.ndarray]]:
    return [(rot, flip, _apply_transform(arr, rot, flip)) for rot in range(4) for flip in (False, True)]


def _exact_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    data = np.ascontiguousarray(mask.astype(np.uint8))
    return (int(data.shape[0]), int(data.shape[1])), data.tobytes()


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_exact_signature(trans) for _, _, trans in _transform_variants(mask))


def _has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    for rot, flip, trans in _transform_variants(mask):
        if rot == 0 and not flip:
            continue
        if trans.shape == mask.shape and np.array_equal(trans, mask):
            return True
    return False


def _is_connected(mask: np.ndarray) -> bool:
    return len(_connected_components(mask)) == 1


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    mask = patch != int(bg)
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    return bool(np.all(region[mask] == int(bg)))


def _drop_row(canvas: np.ndarray, patch: np.ndarray, start_r: int, c0: int, bg: int) -> int:
    h = int(patch.shape[0])
    for r0 in range(canvas.shape[0] - h, int(start_r) - 1, -1):
        if _can_place(canvas, patch, r0, int(c0), bg):
            return int(r0)
    raise AssertionError("no valid downward landing row")


def _fits_with_padding(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    pts = np.argwhere(patch != int(bg))
    for pr, pc in pts:
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        r1 = max(0, rr - padding)
        r2 = min(canvas.shape[0], rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(canvas.shape[1], cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != int(bg)):
            return False
    return True


def _random_place(canvas: np.ndarray, patch: np.ndarray, bg: int, *, row_min: int, row_max: int, padding: int = 1) -> tuple[int, int] | None:
    h, w = patch.shape
    positions: list[Coord] = []
    last_row = min(int(row_max), canvas.shape[0] - h)
    for r0 in range(max(0, int(row_min)), last_row + 1):
        for c0 in range(0, canvas.shape[1] - w + 1):
            if _fits_with_padding(canvas, patch, r0, c0, bg, padding=padding):
                positions.append((r0, c0))
    if not positions:
        return None
    return random.choice(positions)


def _random_connected_shape() -> np.ndarray:
    for _ in range(400):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_area = min(8, h * w - 1)
        if max_area < 4:
            continue
        target = random.randint(4, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        bbox = _bbox(mask)
        if bbox is None:
            continue
        mask = _crop_bbox(mask, bbox)
        if min(mask.shape) < 2:
            continue
        if not _is_connected(mask):
            continue
        if _enclosed_false_components(mask):
            continue
        if _has_nontrivial_symmetry(mask):
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric connected shape")


def _make_frame_patch(cavity_mask: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, Coord]:
    ch, cw = cavity_mask.shape
    top = random.randint(1, 2)
    left = random.randint(1, 2)
    bottom = random.randint(1, 2)
    right = random.randint(1, 2)
    patch = np.full((ch + top + bottom, cw + left + right), int(color), dtype=int)
    patch[top : top + ch, left : left + cw][cavity_mask] = int(bg)
    return patch, (top, left)


def _make_sprite_patch(shape_mask: np.ndarray, body_color: int, marker_color: int, bg: int) -> np.ndarray:
    patch = np.full(shape_mask.shape, int(bg), dtype=int)
    patch[shape_mask] = int(body_color)
    pts = [tuple(map(int, rc)) for rc in np.argwhere(shape_mask)]
    mr, mc = random.choice(pts)
    patch[mr, mc] = int(marker_color)
    return patch


def _mask_match_count(sprite_patch: np.ndarray, cavity_mask: np.ndarray, bg: int) -> list[tuple[int, bool]]:
    matches: list[tuple[int, bool]] = []
    sprite_mask = sprite_patch != int(bg)
    for rot, flip, trans in _transform_variants(sprite_mask):
        if trans.shape == cavity_mask.shape and np.array_equal(trans, cavity_mask):
            matches.append((rot, flip))
    return matches


def _has_strict_majority_color(grid: np.ndarray, color: int) -> bool:
    counts = Counter(map(int, np.asarray(grid).ravel())).most_common(2)
    if not counts or int(counts[0][0]) != int(color):
        return False
    if len(counts) == 1:
        return True
    return int(counts[0][1]) > int(counts[1][1])


def _input_invariants_hold(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    bg = get_bg(g)
    if not _has_strict_majority_color(g, bg):
        return False

    frame_count = 0
    sprite_count = 0
    for comp in _connected_components(g != int(bg)):
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = patch != int(bg)
        colors = sorted(set(map(int, patch[mask])))
        holes = _enclosed_false_components(mask)
        if len(colors) == 1 and len(holes) == 1:
            frame_count += 1
            continue

        if len(colors) != 2:
            return False
        counts = Counter(map(int, patch[mask]))
        marker_colors = [color for color, count in counts.items() if int(count) == 1]
        if len(marker_colors) != 1:
            return False
        sprite_count += 1

    return frame_count > 0 and sprite_count > 0


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _connected_components(g != int(bg))
    frames: list[dict] = []
    sprites: list[dict] = []
    for comp in comps:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = patch != int(bg)
        colors = sorted(set(map(int, patch[mask])))
        holes = _enclosed_false_components(mask)
        if len(colors) == 1 and len(holes) == 1:
            hole_bbox = _bbox(holes[0])
            assert hole_bbox is not None
            cavity_mask = _crop_bbox(holes[0], hole_bbox)
            frames.append(
                {
                    "bbox": bbox,
                    "color": int(colors[0]),
                    "patch": patch,
                    "cavity_mask": cavity_mask,
                    "cavity_bbox_local": hole_bbox,
                    "cavity_top_left": (int(r0 + hole_bbox[0]), int(c0 + hole_bbox[1])),
                }
            )
            continue

        counts = Counter(map(int, patch[mask]))
        assert len(colors) == 2, "sprites must be exactly two-colored"
        marker_colors = [color for color, count in counts.items() if int(count) == 1]
        assert len(marker_colors) == 1, "sprites must have exactly one marker pixel"
        marker_color = int(marker_colors[0])
        body_color = int(next(color for color in colors if int(color) != marker_color))
        marker_rc = tuple(map(int, np.argwhere(patch == marker_color)[0]))
        sprites.append(
            {
                "bbox": bbox,
                "patch": patch,
                "mask": mask,
                "body_color": body_color,
                "marker_color": marker_color,
                "marker_rc": marker_rc,
            }
        )
    return bg, frames, sprites


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Insert each marked free sprite into the uniquely matching cavity shape; unmatched sprites fall straight down and stack.

    Input structure:
    - The dominant color is background. Several monochrome closed frame objects each enclose one background cavity of a distinct asymmetric shape.
    - Separate free sprites lie in empty space. Each sprite is 4-connected, has one body color, and has exactly one differently colored marker pixel.
    - Sprite body colors may agree with unrelated frame colors, so color alone does not identify the destination cavity.

    Rule:
    - Detect every frame cavity and every free sprite.
    - If a sprite's occupied shape matches exactly one cavity after some rotation or mirror transform, remove the sprite from its original position and place the transformed two-color sprite into that cavity.
    - If a sprite matches no cavity, remove it from its original position and let it fall vertically in its original columns until it reaches the bottom edge or a previously settled object; lower unmatched sprites settle first.

    Key insight:
    - Pairing is controlled only by shape under D4 transforms, while the single marker pixel reveals the required orientation inside the cavity; gravity applies only to the leftover non-matching sprites.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, frames, sprites = _parse_scene(g)
    assert frames and sprites

    matches: list[tuple[dict, dict, int, bool]] = []
    unmatched: list[dict] = []
    used_cavities: set[int] = set()

    for sprite in sprites:
        sprite_matches: list[tuple[int, int, bool]] = []
        for frame_idx, frame in enumerate(frames):
            for rot, flip in _mask_match_count(sprite["patch"], frame["cavity_mask"], bg):
                sprite_matches.append((frame_idx, rot, flip))
        assert len(sprite_matches) <= 1, "sprite must have at most one valid cavity/orientation"
        if not sprite_matches:
            unmatched.append(sprite)
            continue
        frame_idx, rot, flip = sprite_matches[0]
        assert frame_idx not in used_cavities, "a cavity cannot receive multiple sprites"
        used_cavities.add(frame_idx)
        matches.append((sprite, frames[frame_idx], rot, flip))

    out = g.copy()
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        patch = sprite["patch"]
        region = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        region[patch != int(bg)] = int(bg)

    for sprite, frame, rot, flip in matches:
        patch = _apply_transform(sprite["patch"], rot, flip)
        rr, cc = frame["cavity_top_left"]
        _paste(out, patch, rr, cc, transparent=bg)

    unmatched.sort(key=lambda obj: (-int(obj["bbox"][2]), -int(obj["bbox"][0]), int(obj["bbox"][1])))
    for sprite in unmatched:
        r0, c0, _r1, _c1 = sprite["bbox"]
        landing_r = _drop_row(out, sprite["patch"], int(r0), int(c0), bg)
        _paste(out, sprite["patch"], landing_r, int(c0), transparent=bg)

    return out


def _build_expected_from_spec(spec: dict) -> np.ndarray:
    bg = int(spec["bg"])
    out = spec["input"].copy()
    for sprite in spec["sprites"]:
        r0, c0 = map(int, sprite["pos"])
        patch = sprite["patch"]
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch != bg] = bg

    for sprite in spec["sprites"]:
        if sprite["target_frame"] is None:
            continue
        frame = spec["frames"][int(sprite["target_frame"])]
        patch = _apply_transform(sprite["patch"], int(sprite["fit_rot"]), bool(sprite["fit_flip"]))
        rr = int(frame["pos"][0] + frame["cavity_offset"][0])
        cc = int(frame["pos"][1] + frame["cavity_offset"][1])
        _paste(out, patch, rr, cc, transparent=bg)

    leftovers = [sprite for sprite in spec["sprites"] if sprite["target_frame"] is None]
    leftovers.sort(key=lambda obj: (-int(obj["pos"][0] + obj["patch"].shape[0] - 1), -int(obj["pos"][0]), int(obj["pos"][1])))
    for sprite in leftovers:
        start_r, start_c = map(int, sprite["pos"])
        landing_r = _drop_row(out, sprite["patch"], start_r, start_c, bg)
        sprite["expected_fall_to"] = landing_r
        _paste(out, sprite["patch"], landing_r, start_c, transparent=bg)
    return out


def _validate_spec(spec: dict) -> None:
    bg = int(spec["bg"])
    frame_sigs = [_canonical_signature(frame["cavity_mask"]) for frame in spec["frames"]]
    assert len(frame_sigs) == len(set(frame_sigs))

    decoy_seen = False
    for sprite in spec["sprites"]:
        all_hits: list[tuple[int, int, bool]] = []
        for frame_idx, frame in enumerate(spec["frames"]):
            for rot, flip in _mask_match_count(sprite["patch"], frame["cavity_mask"], bg):
                all_hits.append((frame_idx, rot, flip))
        if sprite["target_frame"] is None:
            assert not all_hits
        else:
            assert len(all_hits) == 1
            frame_idx, rot, flip = all_hits[0]
            assert frame_idx == int(sprite["target_frame"])
            assert rot == int(sprite["fit_rot"])
            assert bool(flip) == bool(sprite["fit_flip"])
            if int(sprite["body_color"]) != int(spec["frames"][frame_idx]["color"]):
                decoy_seen = True

    assert decoy_seen
    assert any(sprite["target_frame"] is None for sprite in spec["sprites"])
    assert any(sprite["target_frame"] is not None for sprite in spec["sprites"])
    assert any(int(sprite["fit_rot"]) != 0 or bool(sprite["fit_flip"]) for sprite in spec["sprites"] if sprite["target_frame"] is not None)

    unmatched = [sprite for sprite in spec["sprites"] if sprite["target_frame"] is None]
    overlap_seen = False
    for i in range(len(unmatched)):
        c0a = int(unmatched[i]["pos"][1])
        c1a = c0a + int(unmatched[i]["patch"].shape[1]) - 1
        for j in range(i + 1, len(unmatched)):
            c0b = int(unmatched[j]["pos"][1])
            c1b = c0b + int(unmatched[j]["patch"].shape[1]) - 1
            if not (c1a < c0b or c1b < c0a):
                overlap_seen = True
    assert overlap_seen


def _generate_once() -> dict | None:
    bg = random.randint(0, 9)
    colors = [color for color in range(10) if color != bg]
    random.shuffle(colors)

    height = random.randint(20, 29)
    width = random.randint(18, 30)
    canvas = np.full((height, width), bg, dtype=int)

    n_frames = random.randint(2, 4)
    n_unmatched = random.randint(2, 3)
    n_matched = random.randint(1, n_frames)

    cavity_signatures: set[tuple[tuple[int, int], bytes]] = set()
    frames: list[dict] = []
    frame_colors = random.sample(colors, n_frames)
    for frame_color in frame_colors:
        cavity_mask = _random_connected_shape()
        sig = _canonical_signature(cavity_mask)
        if sig in cavity_signatures:
            return None
        cavity_signatures.add(sig)
        patch, cavity_offset = _make_frame_patch(cavity_mask, frame_color, bg)
        pos = _random_place(canvas, patch, bg, row_min=1, row_max=max(2, height - patch.shape[0] - 8), padding=1)
        if pos is None:
            return None
        _paste(canvas, patch, pos[0], pos[1])
        frames.append(
            {
                "color": frame_color,
                "patch": patch,
                "cavity_mask": cavity_mask,
                "cavity_offset": cavity_offset,
                "pos": pos,
            }
        )

    matched_indices = random.sample(range(n_frames), n_matched)
    other_colors = [color for color in colors if color not in frame_colors]
    if not other_colors:
        other_colors = [color for color in colors if color != bg]

    sprites: list[dict] = []
    decoy_frame_idx = random.choice(matched_indices)
    non_identity_done = False

    for frame_idx in matched_indices:
        cavity_mask = frames[frame_idx]["cavity_mask"]
        transform_options = [(rot, flip) for rot in range(4) for flip in (False, True) if rot != 0 or flip]
        rot, flip = random.choice(transform_options)
        sprite_mask = _apply_transform(cavity_mask, rot, flip)

        if frame_idx == decoy_frame_idx and n_frames >= 2:
            wrong_choices = [frames[i]["color"] for i in range(n_frames) if i != frame_idx]
            body_color = random.choice(wrong_choices)
        else:
            body_color = random.choice([frames[frame_idx]["color"], random.choice(other_colors)])

        marker_choices = [color for color in colors if color != body_color]
        marker_color = random.choice(marker_choices)
        patch = _make_sprite_patch(sprite_mask, body_color, marker_color, bg)
        fit_hits = []
        for rr, ff, trans in _transform_variants(patch != bg):
            if trans.shape == cavity_mask.shape and np.array_equal(trans, cavity_mask):
                fit_hits.append((rr, ff))
        if len(fit_hits) != 1:
            return None
        fit_rot, fit_flip = fit_hits[0]
        non_identity_done = non_identity_done or bool(fit_rot != 0 or fit_flip)
        sprites.append(
            {
                "patch": patch,
                "body_color": body_color,
                "marker_color": marker_color,
                "target_frame": frame_idx,
                "fit_rot": fit_rot,
                "fit_flip": fit_flip,
            }
        )

    unmatched_signatures = set(cavity_signatures)
    while len(sprites) < n_matched + n_unmatched:
        shape = _random_connected_shape()
        sig = _canonical_signature(shape)
        if sig in unmatched_signatures:
            continue
        unmatched_signatures.add(sig)
        body_color = random.choice([random.choice(frame_colors), random.choice(other_colors)])
        marker_color = random.choice([color for color in colors if color != body_color])
        patch = _make_sprite_patch(shape, body_color, marker_color, bg)
        sprites.append(
            {
                "patch": patch,
                "body_color": body_color,
                "marker_color": marker_color,
                "target_frame": None,
                "fit_rot": 0,
                "fit_flip": False,
            }
        )

    if not non_identity_done:
        return None

    random.shuffle(sprites)
    lower_band_top = max(1, height // 2 - 1)
    upper_band_bottom = max(1, height // 2 - 4)
    unmatched_widths = [int(sprite["patch"].shape[1]) for sprite in sprites if sprite["target_frame"] is None]
    shared_unmatched_c = random.randint(0, max(0, width - max(unmatched_widths)))
    forced_overlap_count = 0
    for sprite in sprites:
        patch = sprite["patch"]
        if sprite["target_frame"] is None and forced_overlap_count < 2:
            pos = None
            for _ in range(40):
                start_r = random.randint(1, upper_band_bottom)
                if _fits_with_padding(canvas, patch, start_r, shared_unmatched_c, bg, padding=1):
                    pos = (start_r, shared_unmatched_c)
                    break
            if pos is None:
                return None
            if pos is None:
                return None
            forced_overlap_count += 1
        else:
            pos = _random_place(canvas, patch, bg, row_min=1, row_max=lower_band_top, padding=1)
            if pos is None:
                return None
        sprite["pos"] = pos
        _paste(canvas, patch, pos[0], pos[1], transparent=bg)

    if not _has_strict_majority_color(canvas, bg):
        return None
    if not _input_invariants_hold(canvas):
        return None

    spec = {"bg": bg, "input": canvas.copy(), "frames": frames, "sprites": sprites}
    _validate_spec(spec)
    expected = _build_expected_from_spec(spec)
    actual = main(canvas)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(canvas, expected):
        return None
    fallers = [sprite for sprite in sprites if sprite["target_frame"] is None]
    if not any(int(sprite["expected_fall_to"]) > int(sprite["pos"][0]) for sprite in fallers):
        return None
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None
    return spec


def generate_input() -> np.ndarray:
    for _ in range(500):
        spec = _generate_once()
        if spec is not None:
            return spec["input"].copy()
    raise RuntimeError("failed to generate a valid ARC instance")
