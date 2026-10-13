from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MIN_GAP = 2
GEN_RETRIES = 400

VERTICAL_TEMPLATES = (
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=int),
)

BOTH_TEMPLATES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]], dtype=int),
)

NONE_TEMPLATES = (
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pixels = []
            while stack:
                cr, cc = stack.pop()
                pixels.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pixels, dtype=int)
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "area": int(coords.shape[0]),
                }
            )
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return components


def _classify(mask: np.ndarray) -> str:
    h, w = mask.shape
    vertical = w % 2 == 1 and np.array_equal(mask, np.fliplr(mask)) and bool(mask[:, w // 2].any())
    horizontal = h % 2 == 1 and np.array_equal(mask, np.flipud(mask)) and bool(mask[h // 2, :].any())
    if vertical and horizontal:
        return "both"
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    return "none"


def _seed_data(comp: dict) -> tuple[list[tuple[int, int]], tuple[tuple[int, int], ...]]:
    mask = comp["mask"]
    r0, c0, _, _ = comp["bbox"]
    h, w = mask.shape
    kind = comp["kind"]

    if kind == "vertical":
        center_c = w // 2
        seeds = [(r0 + r, c0 + center_c) for r in range(h) if mask[r, center_c]]
        return seeds, ((0, -1), (0, 1))
    if kind == "horizontal":
        center_r = h // 2
        seeds = [(r0 + center_r, c0 + c) for c in range(w) if mask[center_r, c]]
        return seeds, ((-1, 0), (1, 0))
    if kind == "both":
        center_r = h // 2
        center_c = w // 2
        return [(r0 + center_r, c0 + center_c)], ((-1, 0), (1, 0), (0, -1), (0, 1))
    return [], ()


def _simulate(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    components = _find_components(g)
    owner = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(components):
        owner[comp["coords"][:, 0], comp["coords"][:, 1]] = idx
        comp["kind"] = _classify(comp["mask"])

    out = g.copy()
    for comp in components:
        if comp["kind"] == "none":
            out[comp["coords"][:, 0], comp["coords"][:, 1]] = BG

    paint_events: Counter[tuple[int, int]] = Counter()
    ray_info = []
    for idx, comp in enumerate(components):
        if comp["kind"] == "none":
            continue
        seeds, directions = _seed_data(comp)
        color = int(comp["color"])
        for seed in seeds:
            for dr, dc in directions:
                r, c = seed
                painted = []
                skipped_self = 0
                stop = "edge"
                while True:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        stop = "edge"
                        break
                    cell_owner = int(owner[nr, nc])
                    if cell_owner == idx:
                        skipped_self += 1
                        r, c = nr, nc
                        continue
                    if cell_owner != -1:
                        stop = "object"
                        break
                    out[nr, nc] = color
                    paint_events[(nr, nc)] += 1
                    painted.append((nr, nc))
                    r, c = nr, nc
                ray_info.append(
                    {
                        "kind": comp["kind"],
                        "color": color,
                        "seed": seed,
                        "direction": (dr, dc),
                        "painted": painted,
                        "skipped_self": skipped_self,
                        "stop": stop,
                    }
                )

    if return_meta:
        return out, {"components": components, "paint_events": paint_events, "ray_info": ray_info}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each sprite's vertical/horizontal symmetry type and use that type to choose its ray direction.

    Input structure:
    - The grid is a black background containing 3-5 separated monochrome 4-connected sprites.
    - Each sprite belongs to exactly one class: vertical-only symmetry, horizontal-only symmetry, both symmetries, or no symmetry.
    - Sprites are small irregular shapes whose bounding boxes are separated by visible gaps.

    Rule:
    - Crop every sprite to its bounding box and classify its symmetry by comparing the cropped mask to its left-right and up-down flips.
    - Vertical-only sprites emit left and right rays from every pixel on their vertical symmetry column; horizontal-only sprites emit up and down rays from every pixel on their horizontal symmetry row.
    - Sprites with both symmetries emit rays from the center pixel in the four cardinal directions, while non-symmetric sprites are erased completely.
    - A ray may pass through its own sprite without repainting those existing body pixels, then paints background cells until the next step would hit another sprite or leave the grid; later sprites in top-left order overwrite earlier ray crossings.

    Key insight:
    - Symmetry is not just a keep-or-drop filter here: the detected symmetry axis encodes the ray-launch direction, and failure to detect symmetry triggers erasure instead of emission.
    """
    return _simulate(np.asarray(grid, dtype=int), return_meta=False)


def _same_kind_transform(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "vertical":
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask.copy()
    if kind == "horizontal":
        mask = np.rot90(mask, 1)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return mask.copy()
    if kind == "both":
        mask = np.rot90(mask, random.randint(0, 3))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return mask.copy()
    k = random.randint(0, 3)
    mask = np.rot90(mask, k)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.copy()


def _sample_patch(kind: str) -> np.ndarray:
    if kind == "vertical":
        base = random.choice(VERTICAL_TEMPLATES)
    elif kind == "horizontal":
        base = random.choice(VERTICAL_TEMPLATES)
    elif kind == "both":
        base = random.choice(BOTH_TEMPLATES)
    else:
        base = random.choice(NONE_TEMPLATES)
    patch = _same_kind_transform(base, kind)
    assert _classify(_crop_mask(patch.astype(bool))) == kind
    return patch.copy()


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_GAP) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _candidate_positions(h: int, w: int, patch: np.ndarray, target: tuple[int, int]) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    tr, tc = target
    desired_r = max(0, min(h - ph, int(round(tr - (ph - 1) / 2))))
    desired_c = max(0, min(w - pw, int(round(tc - (pw - 1) / 2))))
    scored = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            score = abs(r - desired_r) + abs(c - desired_c) + random.random() * 0.25
            scored.append((score, r, c))
    scored.sort(key=lambda item: item[0])
    return [(r, c) for _, r, c in scored]


def _render_grid(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for item in placements:
        patch = item["patch"] * int(item["color"])
        r0, c0 = item["top_left"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != 0] = patch[patch != 0]
    return grid


def _object_specs(h: int, w: int) -> list[dict]:
    specs = [
        {"kind": "horizontal", "target": (max(2, h // 4), w // 2)},
        {"kind": "vertical", "target": (h // 2, max(2, w // 5))},
        {"kind": "both", "target": (h // 2, min(w - 3, (4 * w) // 5))},
        {"kind": "none", "target": (min(h - 3, (3 * h) // 4), w // 2)},
    ]
    if random.random() < 0.5:
        specs.append(
            {
                "kind": random.choice(("vertical", "horizontal", "both", "none")),
                "target": (min(h - 3, (3 * h) // 4), max(2, w // 4)),
            }
        )
    return specs


def _semantic_ok(grid: np.ndarray) -> bool:
    out, meta = _simulate(grid, return_meta=True)
    components = meta["components"]
    kinds = [comp["kind"] for comp in components]
    if len(components) not in (4, 5):
        return False
    if {comp["color"] for comp in components} != set(int(x) for x in np.unique(grid) if int(x) != BG):
        return False
    if {"vertical", "horizontal", "both", "none"} - set(kinds):
        return False
    for comp in components:
        if not 4 <= comp["area"] <= 8:
            return False
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1 :]:
            if not _boxes_separated(comp_a["bbox"], comp_b["bbox"]):
                return False
    if not any(info["painted"] for info in meta["ray_info"]):
        return False
    if not any(info["stop"] == "object" for info in meta["ray_info"]):
        return False
    if not any(count >= 2 for count in meta["paint_events"].values()):
        return False
    erased = [comp for comp in components if comp["kind"] == "none"]
    if not erased:
        return False
    for comp in erased:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        if np.any(out[rows, cols] != BG):
            return False
    return not np.array_equal(grid, out)


def _build_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    specs = _object_specs(h, w)
    colors = random.sample(list(range(1, 10)), len(specs))
    placements = []

    for color, spec in zip(colors, specs):
        patch = _sample_patch(spec["kind"])
        candidates = _candidate_positions(h, w, patch, spec["target"])
        chosen = None
        ph, pw = patch.shape
        for r0, c0 in candidates[: min(len(candidates), 80)]:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if all(_boxes_separated(box, placed["bbox"]) for placed in placements):
                chosen = (r0, c0, box)
                break
        if chosen is None:
            return None
        r0, c0, box = chosen
        placements.append(
            {
                "kind": spec["kind"],
                "color": color,
                "patch": patch,
                "top_left": (r0, c0),
                "bbox": box,
            }
        )

    grid = _render_grid((h, w), placements)
    if not _semantic_ok(grid):
        return None
    expected = _simulate(grid, return_meta=False)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-ray input")
