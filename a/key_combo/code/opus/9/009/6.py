from __future__ import annotations

from collections import Counter, deque
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _load_arc_utils():
    try:
        from arc_utils import find_components, get_bg, paste, random_free_location, scale_sprite

        return find_components, get_bg, paste, random_free_location, scale_sprite
    except Exception:
        pass

    if ARC_UTILS_PATH.exists():
        try:
            mod = SourceFileLoader("_arc_utils_for_task_6", str(ARC_UTILS_PATH)).load_module()
            return (
                mod.find_components,
                mod.get_bg,
                mod.paste,
                mod.random_free_location,
                mod.scale_sprite,
            )
        except Exception:
            pass

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
        return np.kron(np.asarray(sprite), np.ones((int(factor), int(factor)), dtype=np.asarray(sprite).dtype))

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        assert connectivity == 4
        if not monochromatic or return_format != "dict":
            raise ValueError("fallback find_components only supports monochromatic dict components")
        if colors is None:
            color_list = sorted(set(map(int, g.ravel())) - {int(bg)})
        else:
            color_list = [int(c) for c in colors if int(c) != int(bg)]
        out = []
        h, w = g.shape
        for color in color_list:
            mask = g == color
            seen = np.zeros_like(mask, dtype=bool)
            for r in range(h):
                for c in range(w):
                    if not mask[r, c] or seen[r, c]:
                        continue
                    q = deque([(r, c)])
                    seen[r, c] = True
                    coords = []
                    while q:
                        rr, cc = q.popleft()
                        coords.append((rr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                    if len(coords) < int(min_area):
                        continue
                    arr = np.array(coords, dtype=int)
                    r0, c0 = arr.min(axis=0)
                    r1, c1 = arr.max(axis=0)
                    out.append(
                        {
                            "color": color,
                            "coords": arr,
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "area": int(arr.shape[0]),
                        }
                    )
        return out

    def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None, **_kwargs):
        h, w = patch.shape
        view = dst[r0 : r0 + h, c0 : c0 + w]
        if transparent is None:
            view[:] = patch
        else:
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return dst

    def random_free_location(
        canvas: np.ndarray,
        patch: np.ndarray,
        *,
        transparent: int,
        border: int = 0,
        padding: int = 0,
        padding_connectivity: int = 8,
    ):
        del padding_connectivity
        bg = int(transparent)
        h, w = patch.shape
        occ = canvas != bg
        mask = patch != bg
        candidates = []
        for r0 in range(int(border), canvas.shape[0] - int(border) - h + 1):
            for c0 in range(int(border), canvas.shape[1] - int(border) - w + 1):
                rr0 = max(0, r0 - int(padding))
                cc0 = max(0, c0 - int(padding))
                rr1 = min(canvas.shape[0], r0 + h + int(padding))
                cc1 = min(canvas.shape[1], c0 + w + int(padding))
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                if (canvas[r0 : r0 + h, c0 : c0 + w] != bg)[mask].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            raise ValueError("no free location")
        return random.choice(candidates)

    return find_components, get_bg, paste, random_free_location, scale_sprite


find_components, get_bg, paste, random_free_location, scale_sprite = _load_arc_utils()


BG = 0
FRAME_COLORS = (1, 2, 3, 4)
SPRITE_COLORS = (5, 6, 7, 8, 9)
GRID_SIDE_RANGE = (20, 28)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
GENERATION_RETRIES = 400


MATCH_FAMILIES = (
    {
        "name": "square",
        "sprite_mask": np.ones((2, 2), dtype=bool),
        "choices": ((2, (4, 4)), (3, (6, 6))),
    },
    {
        "name": "wide",
        "sprite_mask": np.ones((2, 3), dtype=bool),
        "choices": ((2, (4, 6)), (3, (6, 9))),
    },
    {
        "name": "tall",
        "sprite_mask": np.ones((3, 2), dtype=bool),
        "choices": ((2, (6, 4)), (3, (9, 6))),
    },
)

UNMATCHED_FRAME_SIZES = (
    (5, 5),
    (5, 7),
    (7, 5),
    (7, 7),
    (5, 8),
    (8, 5),
    (7, 8),
    (8, 7),
    (8, 8),
    (10, 7),
    (7, 10),
)

DISTRACTOR_MASKS = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
)


def _mask_from_component(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(component["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_hollow_rectangle(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 4 or w < 4:
        return False
    return np.array_equal(mask, _frame_mask(h, w))


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _pool_blocks(mask: np.ndarray, factor: int) -> np.ndarray | None:
    h, w = mask.shape
    k = int(factor)
    if h % k != 0 or w % k != 0:
        return None
    pooled = mask.reshape(h // k, k, w // k, k).any(axis=(1, 3))
    return pooled


def _sprite_matches_frame(sprite_mask: np.ndarray, frame_mask_local: np.ndarray, factor: int) -> bool:
    pooled = _pool_blocks(frame_mask_local, factor)
    if pooled is None or pooled.shape != sprite_mask.shape:
        return False
    scaled_sprite = scale_sprite(sprite_mask.astype(int), factor).astype(bool)
    scaled_pooled = scale_sprite(pooled.astype(int), factor).astype(bool)
    return np.array_equal(scaled_sprite, scaled_pooled)


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames = []
    sprites = []
    for component in components:
        mask = _mask_from_component(component)
        item = {
            "color": int(component["color"]),
            "coords": np.asarray(component["coords"], dtype=int),
            "bbox": tuple(map(int, component["bbox"])),
            "mask": mask,
        }
        if _is_hollow_rectangle(mask):
            frames.append(item)
        else:
            sprites.append(item)

    frames.sort(key=lambda item: item["bbox"])
    sprites.sort(key=lambda item: item["bbox"])

    matches: dict[int, tuple[int, int]] = {}
    used_frames: set[int] = set()
    for sprite_idx, sprite in enumerate(sprites):
        hits = []
        for frame_idx, frame in enumerate(frames):
            for factor in (2, 3):
                if _sprite_matches_frame(sprite["mask"], frame["mask"], factor):
                    hits.append((frame_idx, factor))
        frame_ids = {frame_idx for frame_idx, _factor in hits}
        assert len(frame_ids) <= 1, "sprite-to-frame match must be unique"
        if hits:
            frame_idx, factor = hits[0]
            assert frame_idx not in used_frames, "each frame can match at most one sprite"
            matches[sprite_idx] = (frame_idx, factor)
            used_frames.add(frame_idx)
    return frames, sprites, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each small sprite to a same-shape hollow rectangle across scale, then fill the rectangle interior with the sprite color.

    Input structure:
    - The background is black and the foreground contains several one-pixel-thick hollow rectangular frames plus several separated monochrome sprites.
    - Frame colors come from a frame palette and sprite colors come from a different sprite palette, so the object classes are visually distinct.
    - Each relevant sprite is a small 4-connected mask whose 2x or 3x scaled shape agrees with exactly one frame border, while one extra sprite is a distractor.

    Rule:
    - Find every monochrome 4-connected component and classify components whose local mask equals a hollow rectangle border as frames.
    - For every non-frame sprite, test scale factors 2 and 3 and compare the scaled sprite mask against each frame border shape on the frame's own bounding box.
    - If a sprite uniquely matches a frame, fill that frame's interior cells with the sprite color; leave unmatched frames, all borders, and all sprites in place.

    Key insight:
    - Cross-scale shape matching is not the output itself here; it creates a unique frame-to-sprite color mapping that drives a later interior fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames, sprites, matches = _analyze_grid(g)
    assert 3 <= len(frames) <= 4, "expected 3 or 4 hollow frames"
    assert 3 <= len(sprites) <= 4, "expected 3 or 4 sprites"

    out = g.copy()
    for sprite_idx, (frame_idx, _factor) in matches.items():
        r0, c0, r1, c1 = frames[frame_idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(sprites[sprite_idx]["color"])
    return out


def _sample_matched_specs(total_frames: int) -> tuple[list[dict], int]:
    matched_count = total_frames - 1
    family_indices = random.sample(range(len(MATCH_FAMILIES)), matched_count)
    specs = []
    for family_idx in family_indices:
        family = MATCH_FAMILIES[family_idx]
        factor, (h, w) = random.choice(family["choices"])
        specs.append(
            {
                "family": family["name"],
                "sprite_mask": family["sprite_mask"].copy(),
                "factor": int(factor),
                "frame_hw": (int(h), int(w)),
            }
        )
    random.shuffle(specs)
    return specs, matched_count


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> tuple[int, int]:
    r0, c0 = random_free_location(
        grid,
        patch,
        transparent=BG,
        border=PLACEMENT_BORDER,
        padding=PLACEMENT_PADDING,
        padding_connectivity=8,
    )
    paste(grid, patch, r0, c0, transparent=BG)
    return int(r0), int(c0)


def _build_instance() -> np.ndarray | None:
    total_frames = random.choice((3, 4))
    matched_specs, matched_count = _sample_matched_specs(total_frames)
    unmatched_hw = random.choice(UNMATCHED_FRAME_SIZES)
    frame_colors = random.sample(FRAME_COLORS, total_frames)
    sprite_colors = random.sample(SPRITE_COLORS, total_frames)

    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    frame_items = []
    for idx, spec in enumerate(matched_specs):
        patch = _patch_from_mask(_frame_mask(*spec["frame_hw"]), frame_colors[idx])
        frame_items.append(
            {
                "kind": "matched",
                "patch": patch,
                "frame_color": frame_colors[idx],
                "sprite_mask": spec["sprite_mask"],
                "sprite_color": sprite_colors[idx],
                "factor": spec["factor"],
            }
        )
    frame_items.append(
        {
            "kind": "unmatched",
            "patch": _patch_from_mask(_frame_mask(*unmatched_hw), frame_colors[-1]),
            "frame_color": frame_colors[-1],
        }
    )
    frame_items.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    sprite_items = []
    for item in frame_items:
        if item["kind"] == "matched":
            sprite_items.append(
                {
                    "patch": _patch_from_mask(item["sprite_mask"], item["sprite_color"]),
                    "sprite_color": item["sprite_color"],
                    "sprite_mask": item["sprite_mask"],
                    "matched_kind": item["kind"],
                }
            )
    distractor_mask = random.choice(DISTRACTOR_MASKS).copy()
    sprite_items.append(
        {
            "patch": _patch_from_mask(distractor_mask, sprite_colors[-1]),
            "sprite_color": sprite_colors[-1],
            "sprite_mask": distractor_mask,
            "matched_kind": "distractor",
        }
    )
    random.shuffle(sprite_items)

    try:
        for item in frame_items:
            item["top_left"] = _place_patch(grid, item["patch"])
        for item in sprite_items:
            item["top_left"] = _place_patch(grid, item["patch"])
    except ValueError:
        return None

    if matched_count != len(sprite_items) - 1:
        return None
    return grid


def _semantic_checks(inp: np.ndarray) -> None:
    frames, sprites, matches = _analyze_grid(inp)
    assert len(frames) == len(sprites), "generator must create equally many frames and sprites"
    assert len(matches) == len(frames) - 1, "generator must create exactly one distractor sprite"
    matched_frames = {frame_idx for frame_idx, _factor in matches.values()}
    assert len(matched_frames) == len(matches)
    assert len(frames) - len(matched_frames) == 1, "generator must leave exactly one unmatched frame"

    out = main(inp)
    assert out.shape == inp.shape
    assert np.any(out != inp), "output must differ from input"

    for sprite_idx, (frame_idx, _factor) in matches.items():
        r0, c0, r1, c1 = frames[frame_idx]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert np.all(interior == sprites[sprite_idx]["color"])

    for frame_idx, frame in enumerate(frames):
        if frame_idx in matched_frames:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        assert np.all(interior == BG)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_instance()
        if grid is None:
            continue
        try:
            _semantic_checks(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid task after {GENERATION_RETRIES} attempts")
