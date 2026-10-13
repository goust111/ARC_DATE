from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load arc_utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
get_bg = _ARC.get_bg
find_components = _ARC.find_components
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
in_bounds = _ARC.in_bounds
neighbors4 = _ARC.neighbors4


BG = Color.BLACK
CANVAS_SIZE = 26
MIN_MARGIN = 2
MAX_ATTEMPTS = 400

SPRITE_TEMPLATES = [
    {"name": "tri_l", "shape": np.array([[1, 0], [1, 1]], dtype=int)},
    {"name": "hook4", "shape": np.array([[1, 0], [1, 0], [1, 1]], dtype=int)},
    {"name": "hook5", "shape": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int)},
    {"name": "zig5", "shape": np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int)},
    {"name": "step6", "shape": np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=int)},
    {"name": "hook6", "shape": np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=int)},
]


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _rotate_point(r: int, c: int, cr: int, cc: int, turns: int) -> tuple[int, int]:
    dr = int(r) - int(cr)
    dc = int(c) - int(cc)
    turns = int(turns) % 4
    if turns == 0:
        nr, nc = dr, dc
    elif turns == 1:
        nr, nc = dc, -dr
    elif turns == 2:
        nr, nc = -dr, -dc
    else:
        nr, nc = -dc, dr
    return int(cr + nr), int(cc + nc)


def _rotate_coords(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int], turns: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cr = int(r0 + bbox_h((r0, c0, r1, c1)) // 2)
    cc = int(c0 + bbox_w((r0, c0, r1, c1)) // 2)
    return [_rotate_point(r, c, cr, cc, turns) for r, c in coords]


def _compute_interior_mask(frame_mask: np.ndarray) -> np.ndarray:
    h, w = map(int, frame_mask.shape)
    free = ~frame_mask.astype(bool)
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return free & (~outside)


def _boundary_of_region(region_mask: np.ndarray) -> np.ndarray:
    h, w = map(int, region_mask.shape)
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not region_mask[r, c]:
                continue
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not region_mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _diag_pixels_for_frame(grid: np.ndarray, frame_mask: np.ndarray, interior_mask: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = map(int, grid.shape)
    coords: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != int(bg):
                continue
            has_diag = False
            has_orth = False
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and frame_mask[nr, nc]:
                    has_diag = True
                    break
            if not has_diag:
                continue
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and frame_mask[nr, nc]:
                    has_orth = True
                    break
            if not has_orth:
                coords.append((int(r), int(c)))
    return coords


def _is_asymmetric(shape: np.ndarray) -> bool:
    coords = _coords_from_mask(shape.astype(bool))
    bbox = (0, 0, int(shape.shape[0]) - 1, int(shape.shape[1]) - 1)
    base = set(coords)
    for turns in (1, 2, 3):
        if set(_rotate_coords(coords, bbox, turns)) == base:
            return False
    return True


def _sprite_templates_for_turns(turns: int) -> list[dict]:
    candidates = []
    for template in SPRITE_TEMPLATES:
        shape = template["shape"]
        if not _is_asymmetric(shape):
            continue
        if turns == 0:
            candidates.append(template)
            continue
        coords = _coords_from_mask(shape.astype(bool))
        bbox = (0, 0, int(shape.shape[0]) - 1, int(shape.shape[1]) - 1)
        if set(_rotate_coords(coords, bbox, turns)) != set(coords):
            candidates.append(template)
    return candidates


def _build_rectangle_patch(rng: np.random.Generator) -> dict:
    h = int(rng.integers(5, 8))
    w_choices = [x for x in range(6, 10) if x != h]
    w = int(rng.choice(w_choices))
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    interior = np.zeros((h, w), dtype=bool)
    interior[1:-1, 1:-1] = True
    return {"kind": "rect", "frame": frame, "interior": interior, "diag_count": 4}


def _make_l_region(h: int, w: int, cut_h: int, cut_w: int, corner: str) -> np.ndarray:
    region = np.ones((h, w), dtype=bool)
    if corner == "TL":
        region[:cut_h, :cut_w] = False
    elif corner == "TR":
        region[:cut_h, w - cut_w :] = False
    elif corner == "BL":
        region[h - cut_h :, :cut_w] = False
    else:
        region[h - cut_h :, w - cut_w :] = False
    return region


def _diag_count_for_patch(frame: np.ndarray, interior: np.ndarray) -> int:
    h, w = map(int, frame.shape)
    grid = np.zeros((h + 4, w + 4), dtype=int)
    frame_global = np.zeros_like(grid, dtype=bool)
    interior_global = np.zeros_like(grid, dtype=bool)
    frame_global[2 : 2 + h, 2 : 2 + w] = frame
    interior_global[2 : 2 + h, 2 : 2 + w] = interior
    return len(_diag_pixels_for_frame(grid, frame_global, interior_global, BG))


def _precompute_l_patches() -> dict[int, list[dict]]:
    candidates: dict[int, list[dict]] = {5: [], 6: [], 7: []}
    for h in range(6, 10):
        for w in range(6, 10):
            for cut_h in range(2, h - 1):
                for cut_w in range(2, w - 1):
                    if h - cut_h < 3 or w - cut_w < 3:
                        continue
                    for corner in ("TL", "TR", "BL", "BR"):
                        region = _make_l_region(h, w, cut_h, cut_w, corner)
                        frame = _boundary_of_region(region)
                        interior = region & (~frame)
                        if interior.sum() < 6:
                            continue
                        if len(find_components(frame.astype(int), bg=0, connectivity=4, monochromatic=True, colors=[1], return_format="dict")) != 1:
                            continue
                        diag_count = _diag_count_for_patch(frame, interior)
                        if diag_count in candidates:
                            candidates[diag_count].append(
                                {"kind": "l", "frame": frame.copy(), "interior": interior.copy(), "diag_count": diag_count}
                            )
    return candidates


_L_PATCHES_BY_DIAG = _precompute_l_patches()


def _build_l_patch(rng: np.random.Generator, target_diag: int) -> dict:
    candidates = _L_PATCHES_BY_DIAG.get(int(target_diag), [])
    if candidates:
        choice = candidates[int(rng.integers(0, len(candidates)))]
        return {
            "kind": "l",
            "frame": choice["frame"].copy(),
            "interior": choice["interior"].copy(),
            "diag_count": int(choice["diag_count"]),
        }
    raise RuntimeError(f"unable to build L frame with diagonal count {target_diag}")


def _bbox_of_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _place_sprite(
    rng: np.random.Generator,
    interior: np.ndarray,
    turns: int,
    require_clip: bool,
) -> dict | None:
    h, w = map(int, interior.shape)
    templates = _sprite_templates_for_turns(turns)
    order = list(rng.permutation(len(templates)))
    for idx in order:
        template = templates[int(idx)]
        shape = template["shape"]
        th, tw = map(int, shape.shape)
        if th > h or tw > w:
            continue
        offsets = [(r, c) for r in range(h - th + 1) for c in range(w - tw + 1)]
        rng.shuffle(offsets)
        base_coords = _coords_from_mask(shape.astype(bool))
        for off_r, off_c in offsets:
            placed = [(int(off_r + r), int(off_c + c)) for r, c in base_coords]
            if not all(interior[r, c] for r, c in placed):
                continue
            bbox = _bbox_of_coords(placed)
            rotated = _rotate_coords(placed, bbox, turns)
            kept = [(r, c) for r, c in rotated if 0 <= r < h and 0 <= c < w and interior[r, c]]
            if not kept:
                continue
            if turns != 0 and set(kept) == set(placed):
                continue
            clipped = len(kept) < len(placed)
            if require_clip and not clipped:
                continue
            if interior.sum() - len(set(placed)) < 2:
                continue
            return {
                "template_name": template["name"],
                "coords_local": placed,
                "bbox_local": bbox,
                "rotated_local": kept,
                "clipped": clipped,
            }
    return None


def _place_patch(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    occupied_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = map(int, patch.shape)
    positions = [
        (r, c)
        for r in range(MIN_MARGIN, grid.shape[0] - h - MIN_MARGIN + 1)
        for c in range(MIN_MARGIN, grid.shape[1] - w - MIN_MARGIN + 1)
    ]
    rng.shuffle(positions)
    for r0, c0 in positions:
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        ok = True
        for er0, ec0, er1, ec1 in occupied_bboxes:
            if not (
                bbox[2] + 3 < er0
                or er1 + 3 < bbox[0]
                or bbox[3] + 3 < ec0
                or ec1 + 3 < bbox[1]
            ):
                ok = False
                break
        if not ok:
            continue
        patch_mask = patch.astype(bool)
        if np.any(grid[r0 : r0 + h, c0 : c0 + w][patch_mask] != BG):
            continue
        return int(r0), int(c0)
    return None


def _apply_metadata_transform(grid: np.ndarray, metadata: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for item in metadata:
        for r, c in item["sprite_coords"]:
            out[r, c] = BG
        rotated = _rotate_coords(item["sprite_coords"], item["sprite_bbox"], item["turns"])
        for r, c in rotated:
            if item["interior_global"][r, c]:
                out[r, c] = int(item["sprite_color"])
        for r, c in item["diag_pixels"]:
            out[r, c] = int(item["sprite_color"])
    return out


def _crop_case(grid: np.ndarray, metadata: list[dict]) -> tuple[np.ndarray, list[dict]]:
    coords = np.argwhere(grid != BG)
    r0 = max(0, int(coords[:, 0].min()) - 2)
    c0 = max(0, int(coords[:, 1].min()) - 2)
    r1 = min(grid.shape[0] - 1, int(coords[:, 0].max()) + 2)
    c1 = min(grid.shape[1] - 1, int(coords[:, 1].max()) + 2)
    cropped = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    new_meta: list[dict] = []
    for item in metadata:
        frame_global = item["frame_global"][r0 : r1 + 1, c0 : c1 + 1].copy()
        interior_global = item["interior_global"][r0 : r1 + 1, c0 : c1 + 1].copy()
        sprite_coords = [(int(r - r0), int(c - c0)) for r, c in item["sprite_coords"]]
        sr0, sc0, sr1, sc1 = item["sprite_bbox"]
        new_meta.append(
            {
                **item,
                "frame_global": frame_global,
                "interior_global": interior_global,
                "sprite_coords": sprite_coords,
                "sprite_bbox": (int(sr0 - r0), int(sc0 - c0), int(sr1 - r0), int(sc1 - c0)),
                "diag_pixels": [(int(r - r0), int(c - c0)) for r, c in item["diag_pixels"]],
            }
        )
    return cropped, new_meta


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        patch_mask = np.zeros((h, w), dtype=bool)
        for r, c in comp["coords"]:
            patch_mask[int(r) - r0, int(c) - c0] = True
        interior_local = _compute_interior_mask(patch_mask)
        if not interior_local.any():
            continue
        region = patch_mask | interior_local
        if not np.array_equal(patch_mask, _boundary_of_region(region)):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        sprite_colors = sorted({int(v) for v in patch[interior_local] if int(v) != bg})
        if len(sprite_colors) != 1:
            continue
        sprite_color = int(sprite_colors[0])
        sprite_mask = (patch == sprite_color) & interior_local
        sprite_comps = find_components(
            sprite_mask.astype(int),
            bg=0,
            connectivity=4,
            monochromatic=True,
            colors=[1],
            return_format="dict",
        )
        if len(sprite_comps) != 1:
            continue
        sprite_coords_local = [tuple(map(int, rc)) for rc in sprite_comps[0]["coords"]]
        if not (3 <= len(sprite_coords_local) <= 6):
            continue
        bbox_local = tuple(map(int, sprite_comps[0]["bbox"]))
        shape = np.zeros((bbox_local[2] - bbox_local[0] + 1, bbox_local[3] - bbox_local[1] + 1), dtype=bool)
        for rr, cc in sprite_coords_local:
            shape[rr - bbox_local[0], cc - bbox_local[1]] = True
        if not _is_asymmetric(shape.astype(int)):
            continue
        frame_global = np.zeros_like(g, dtype=bool)
        frame_global[r0 : r1 + 1, c0 : c1 + 1] = patch_mask
        interior_global = np.zeros_like(g, dtype=bool)
        interior_global[r0 : r1 + 1, c0 : c1 + 1] = interior_local
        diag_pixels = _diag_pixels_for_frame(g, frame_global, interior_global, bg)
        sprite_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in sprite_coords_local]
        sprite_bbox = _bbox_of_coords(sprite_coords)
        frames.append(
            {
                "frame_color": color,
                "frame_bbox": (r0, c0, r1, c1),
                "frame_global": frame_global,
                "interior_global": interior_global,
                "sprite_color": sprite_color,
                "sprite_coords": sprite_coords,
                "sprite_bbox": sprite_bbox,
                "diag_pixels": diag_pixels,
                "diag_count": len(diag_pixels),
                "turns": len(diag_pixels) % 4,
            }
        )
    frames.sort(key=lambda item: (item["frame_bbox"][0], item["frame_bbox"][1]))
    return frames


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rotate each boxed sprite by a quarter-turn count derived from that frame's diagonal-only background contacts, then recolor those same contacts.

    Input structure:
    - The background is black and the grid contains 2 or 3 separated single-color closed frames.
    - Each frame is a 1-pixel-wide non-square rectangle ring or an L-shaped corner ring, and each encloses exactly one small asymmetric monochrome sprite of another color.
    - Around every frame, some black pixels lie diagonally adjacent to the frame without touching it orthogonally.

    Rule:
    - For each frame, collect the black pixels that are diagonally adjacent to at least one frame pixel but orthogonally adjacent to none of that frame's pixels.
    - Let D be the size of that diagonal-only set and rotate the enclosed sprite around its own bounding-box center by D mod 4 quarter turns clockwise, clipping any rotated pixels that leave the frame interior.
    - Recolor every pixel from the same diagonal-only set to the sprite's original color, while leaving the frame and all unrelated background unchanged.

    Key insight:
    - The same diagonal-contact analysis serves two roles at once: its count determines the rotation amount and its locations determine which outside pixels are recolored.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2
    frames = _extract_frames(g)
    assert 2 <= len(frames) <= 3, "expected 2 or 3 valid framed sprites"
    return _apply_metadata_transform(g, frames)


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, list[dict]] | None:
    n_frames = int(rng.integers(2, 4))
    target_ds = [4]
    if n_frames == 2:
        target_ds.append(int(rng.choice([5, 6, 7])))
    else:
        extra = list(rng.choice([5, 6, 7], size=2, replace=False))
        target_ds.extend(int(x) for x in extra)
    rng.shuffle(target_ds)

    patch_specs = []
    for d in target_ds:
        patch_specs.append(_build_rectangle_patch(rng) if d == 4 else _build_l_patch(rng, d))

    palette = list(rng.choice(np.array(Color.NOT_BLACK, dtype=int), size=2 * n_frames, replace=False))
    frame_colors = palette[:n_frames]
    sprite_colors = palette[n_frames:]
    require_clip_index = int(rng.integers(0, n_frames)) if rng.random() < 0.6 else -1

    grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occupied_bboxes: list[tuple[int, int, int, int]] = []
    metadata: list[dict] = []

    for idx, patch_spec in enumerate(patch_specs):
        frame_mask = patch_spec["frame"]
        interior_mask = patch_spec["interior"]
        pos = _place_patch(rng, grid, frame_mask.astype(int), occupied_bboxes)
        if pos is None:
            return None
        r0, c0 = pos
        h, w = map(int, frame_mask.shape)
        frame_color = int(frame_colors[idx])
        sprite_color = int(sprite_colors[idx])
        grid[r0 : r0 + h, c0 : c0 + w][frame_mask] = frame_color

        turns = int(patch_spec["diag_count"] % 4)
        sprite = _place_sprite(rng, interior_mask, turns, require_clip=(idx == require_clip_index))
        if sprite is None:
            return None

        sprite_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in sprite["coords_local"]]
        for rr, cc in sprite_coords:
            grid[rr, cc] = sprite_color

        frame_global = np.zeros_like(grid, dtype=bool)
        frame_global[r0 : r0 + h, c0 : c0 + w] = frame_mask
        interior_global = np.zeros_like(grid, dtype=bool)
        interior_global[r0 : r0 + h, c0 : c0 + w] = interior_mask
        diag_pixels = _diag_pixels_for_frame(grid, frame_global, interior_global, BG)
        if len(diag_pixels) != int(patch_spec["diag_count"]):
            return None

        occupied_bboxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
        metadata.append(
            {
                "kind": patch_spec["kind"],
                "frame_color": frame_color,
                "frame_global": frame_global,
                "interior_global": interior_global,
                "sprite_color": sprite_color,
                "sprite_coords": sprite_coords,
                "sprite_bbox": _bbox_of_coords(sprite_coords),
                "diag_pixels": diag_pixels,
                "diag_count": len(diag_pixels),
                "turns": turns,
                "clipped": bool(sprite["clipped"]),
                "template_name": sprite["template_name"],
            }
        )

    if len({item["turns"] for item in metadata}) < 2:
        return None
    if not any(item["kind"] == "l" for item in metadata):
        return None
    if not any((item["sprite_bbox"][2] - item["sprite_bbox"][0] + 1) != (item["sprite_bbox"][3] - item["sprite_bbox"][1] + 1) for item in metadata):
        return None

    cropped_grid, cropped_meta = _crop_case(grid, metadata)
    if max(cropped_grid.shape) > 30:
        return None
    return cropped_grid, cropped_meta


def _validate_case(grid: np.ndarray, metadata: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.any(grid == BG):
        return False
    if len(metadata) < 2 or len(metadata) > 3:
        return False
    if len({item["turns"] for item in metadata}) < 2:
        return False
    if not any(item["kind"] == "rect" for item in metadata):
        return False
    if not any(item["kind"] == "l" for item in metadata):
        return False
    if not all(1 <= item["diag_count"] <= 7 for item in metadata):
        return False

    expected = _apply_metadata_transform(grid, metadata)
    if np.array_equal(expected, grid):
        return False
    solved = main(grid)
    if not np.array_equal(expected, solved):
        return False

    extracted = _extract_frames(grid)
    if len(extracted) != len(metadata):
        return False
    extracted_counts = sorted(item["diag_count"] for item in extracted)
    meta_counts = sorted(item["diag_count"] for item in metadata)
    if extracted_counts != meta_counts:
        return False
    if not any(item["clipped"] for item in metadata) and not any(item["turns"] == 0 for item in metadata):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        built = _build_case(rng)
        if built is None:
            continue
        grid, metadata = built
        if _validate_case(grid, metadata):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
