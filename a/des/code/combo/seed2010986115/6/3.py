from __future__ import annotations

import numpy as np


BG = 0
NON_BG_COLORS = np.arange(1, 10, dtype=int)
MAX_ATTEMPTS = 400


def _components_by_color(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue

            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )

    return comps


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    perimeter = 2 * (h + w) - 4
    if int(comp["area"]) != int(perimeter):
        return False

    for r, c in comp["coords"]:
        if int(r) not in (int(r0), int(r1)) and int(c) not in (int(c0), int(c1)):
            return False
    return True


def _crop_non_bg(patch: np.ndarray, bg: int = BG) -> np.ndarray:
    pts = np.argwhere(np.asarray(patch) != int(bg))
    if pts.size == 0:
        raise ValueError("patch has no foreground")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(patch[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(patch, dtype=int), int(factor), axis=0), int(factor), axis=1)


def _extract_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = [comp for comp in _components_by_color(g, bg=BG) if _is_frame_component(comp)]
    frame_by_bbox = {tuple(map(int, comp["bbox"])): idx for idx, comp in enumerate(frames)}

    child_of: dict[int, int] = {}
    parent_of: dict[int, int] = {}
    for idx, comp in enumerate(frames):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        parent_bbox = (int(r0 - 1), int(c0 - 1), int(r1 + 1), int(c1 + 1))
        parent_idx = frame_by_bbox.get(parent_bbox)
        if parent_idx is not None:
            parent_of[idx] = int(parent_idx)
            child_of[int(parent_idx)] = int(idx)

    groups: list[dict] = []
    for outer_idx, outer in enumerate(frames):
        if outer_idx in parent_of:
            continue

        layers = 1
        inner_idx = int(outer_idx)
        while inner_idx in child_of:
            inner_idx = int(child_of[inner_idx])
            layers += 1

        r0, c0, r1, c1 = map(int, frames[inner_idx]["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        obj = _crop_non_bg(interior, bg=BG)
        groups.append(
            {
                "layers": int(layers),
                "left": int(outer["bbox"][1]),
                "top": int(outer["bbox"][0]),
                "object": obj,
            }
        )

    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each nested target object, scale it by frame depth, then pack by depth.
    """
    groups = _extract_groups(np.asarray(grid, dtype=int))
    if not groups:
        return np.asarray(grid, dtype=int).copy()

    ordered = sorted(groups, key=lambda item: (int(item["layers"]), int(item["left"]), int(item["top"])))
    scaled = [_scale_patch(item["object"], int(item["layers"])) for item in ordered]

    out_h = max(int(patch.shape[0]) for patch in scaled)
    out_w = sum(int(patch.shape[1]) for patch in scaled) + max(0, len(scaled) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for patch in scaled:
        h, w = map(int, patch.shape)
        out[:h, col : col + w] = patch
        col += w + 1

    return out


def _has_frame_like_component(patch: np.ndarray) -> bool:
    for comp in _components_by_color(np.asarray(patch, dtype=int), bg=BG):
        if _is_frame_component(comp):
            return True
    return False


def _random_object(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(1, 4))
        w = int(rng.integers(1, 4))
        n_cells = int(rng.integers(1, h * w + 1))
        mask = np.zeros((h, w), dtype=bool)
        picks = rng.choice(h * w, size=n_cells, replace=False)
        mask.flat[picks] = True

        n_colors = int(rng.integers(1, min(3, n_cells) + 1))
        palette = rng.choice(NON_BG_COLORS, size=n_colors, replace=False)

        patch = np.full((h, w), BG, dtype=int)
        for rr, cc in np.argwhere(mask):
            patch[int(rr), int(cc)] = int(palette[int(rng.integers(0, len(palette)))])

        patch = _crop_non_bg(patch, bg=BG)
        if _has_frame_like_component(patch):
            continue
        return patch

    raise RuntimeError("failed to sample object")


def _wrap_with_frames(obj: np.ndarray, frame_colors: list[int]) -> np.ndarray:
    cur = np.asarray(obj, dtype=int)
    for color in frame_colors:
        h, w = map(int, cur.shape)
        nxt = np.full((h + 2, w + 2), BG, dtype=int)
        nxt[0, :] = int(color)
        nxt[-1, :] = int(color)
        nxt[:, 0] = int(color)
        nxt[:, -1] = int(color)
        nxt[1:-1, 1:-1] = cur
        cur = nxt
    return cur


def _render_expected(groups: list[dict]) -> np.ndarray:
    ordered = sorted(groups, key=lambda item: (int(item["layers"]), int(item["pos"][1]), int(item["pos"][0])))
    scaled = [_scale_patch(item["object"], int(item["layers"])) for item in ordered]
    out_h = max(int(patch.shape[0]) for patch in scaled)
    out_w = sum(int(patch.shape[1]) for patch in scaled) + len(scaled) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for patch in scaled:
        h, w = map(int, patch.shape)
        out[:h, col : col + w] = patch
        col += w + 1
    return out


def _place_patches(rng: np.random.Generator, patches: list[np.ndarray], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: (-int(patches[idx].shape[0] * patches[idx].shape[1]), -int(patches[idx].shape[1])))

    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        candidates: list[tuple[int, int]] = []
        for r in range(1, h - ph):
            for c in range(1, w - pw):
                rr0 = int(max(0, r - 1))
                rr1 = int(min(h, r + ph + 1))
                cc0 = int(max(0, c - 1))
                cc1 = int(min(w, c + pw + 1))
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((int(r), int(c)))
        if not candidates:
            return None

        r0, c0 = candidates[int(rng.integers(0, len(candidates)))]
        positions[idx] = (int(r0), int(c0))
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_groups = int(rng.integers(3, 5))
    layers = [1, 2, 3]
    while len(layers) < n_groups:
        layers.append(int(rng.integers(1, 4)))
    rng.shuffle(layers)

    groups: list[dict] = []
    patches: list[np.ndarray] = []
    for depth in layers:
        obj = _random_object(rng)
        obj_colors = {int(v) for v in obj[obj != BG]}
        frame_pool = np.array([c for c in NON_BG_COLORS if int(c) not in obj_colors], dtype=int)
        if len(frame_pool) < depth:
            return None
        frame_colors = [int(v) for v in rng.choice(frame_pool, size=depth, replace=False)]
        patch = _wrap_with_frames(obj, frame_colors)
        patches.append(patch)
        groups.append({"layers": int(depth), "object": obj, "patch": patch})

    max_patch_h = max(int(patch.shape[0]) for patch in patches)
    max_patch_w = max(int(patch.shape[1]) for patch in patches)
    grid_h_lo = max(16, max_patch_h + 2)
    grid_w_lo = max(22, max_patch_w + 2)
    grid_h = int(rng.integers(grid_h_lo, 31))
    grid_w = int(rng.integers(grid_w_lo, 31))

    positions = _place_patches(rng, patches, (grid_h, grid_w))
    if positions is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected_groups: list[dict] = []
    for group, (r0, c0) in zip(groups, positions):
        patch = group["patch"]
        ph, pw = map(int, patch.shape)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        expected_groups.append(
            {
                "layers": int(group["layers"]),
                "object": np.asarray(group["object"], dtype=int),
                "pos": (int(r0), int(c0)),
            }
        )

    expected = _render_expected(expected_groups)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
