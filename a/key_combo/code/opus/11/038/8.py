from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
OBSTACLE = 5
GROUND = 9
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400

HEIGHT1_SHAPES = (
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
)

HEIGHT2_SHAPES = (
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=int),
)


@dataclass(frozen=True)
class Sprite:
    color: int
    pixels: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]

    @property
    def left(self) -> int:
        return self.bbox[1]

    @property
    def right(self) -> int:
        return self.bbox[3]

    @property
    def top(self) -> int:
        return self.bbox[0]

    @property
    def bottom(self) -> int:
        return self.bbox[2]

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_sprites(grid: np.ndarray) -> list[Sprite]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, OBSTACLE, GROUND)]
    sprites: list[Sprite] = []
    for color in colors:
        comps = _components(grid == color)
        for coords in comps:
            box = _bbox(coords)
            sprites.append(
                Sprite(
                    color=color,
                    pixels=tuple(sorted((int(r), int(c)) for r, c in coords)),
                    bbox=box,
                )
            )
    return sprites


def _target_left(sprite: Sprite, width: int, *, mirror: bool) -> int:
    if not mirror:
        return sprite.left
    return int(width - sprite.width - sprite.left)


def _simulate_sprite(
    sprite: Sprite,
    grid_shape: tuple[int, int],
    obstacles: set[tuple[int, int]],
    *,
    respect_obstacles: bool,
    mirror: bool,
) -> dict:
    h, w = map(int, grid_shape)
    target_left = _target_left(sprite, w, mirror=mirror)
    direction = 0
    if target_left > sprite.left:
        direction = 1
    elif target_left < sprite.left:
        direction = -1

    pixels = [(idx, r, c) for idx, (r, c) in enumerate(sprite.pixels)]
    current_left = int(sprite.left)
    detours = 0
    clipped = False

    while current_left != target_left:
        next_pixels = [(idx, r, c + direction) for idx, r, c in pixels]
        blocked = respect_obstacles and any((r, c) in obstacles for _idx, r, c in next_pixels)
        if blocked:
            detours += 1
            lifted = []
            for idx, r, c in pixels:
                if r - 1 >= 0:
                    lifted.append((idx, r - 1, c))
                else:
                    clipped = True
            pixels = lifted
        pixels = [(idx, r, c + direction) for idx, r, c in pixels]
        current_left += direction

    final_pixels = [(r, c) for _idx, r, c in pixels if 0 <= r < h and 0 <= c < w]
    return {
        "color": sprite.color,
        "original_left": sprite.left,
        "target_left": target_left,
        "final_pixels": final_pixels,
        "detours": detours,
        "clipped": clipped,
        "all_lost": len(final_pixels) == 0,
    }


def _transform(
    grid: np.ndarray,
    *,
    respect_obstacles: bool = True,
    draw_trails: bool = True,
    mirror: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 4, "grid too small"
    assert np.all(g[-1] == GROUND), "bottom row must be the solid ground row"

    sprites = _extract_sprites(g)
    assert sprites, "expected at least one sprite"
    seen_colors = set()
    for sprite in sprites:
        assert sprite.color not in seen_colors, "sprite colors must be unique"
        seen_colors.add(sprite.color)
        assert 3 <= len(sprite.pixels) <= 5, "sprite area must be 3..5"
        assert sprite.height <= 2, "sprite bbox height must be at most 2"
        assert sprite.bottom == h - 2, "sprite must rest immediately above the ground row"
        assert (sprite.left == 0) ^ (sprite.right == w - 1), "sprite must touch exactly one side edge"

    ordering = sorted(sprites, key=lambda s: (-s.bottom, s.left))
    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)}

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite.pixels:
            out[r, c] = BG

    metas: list[dict] = []
    for sprite in ordering:
        meta = _simulate_sprite(
            sprite,
            g.shape,
            obstacles,
            respect_obstacles=respect_obstacles,
            mirror=mirror,
        )
        metas.append(meta)
        if draw_trails:
            a = min(meta["original_left"], meta["target_left"])
            b = max(meta["original_left"], meta["target_left"])
            out[h - 1, a : b + 1] = int(sprite.color)
        for r, c in meta["final_pixels"]:
            out[r, c] = int(sprite.color)
    return out, metas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each edge-attached sprite toward its horizontal mirror target, detour upward around gray blockers, and paint its ground-track projection.

    Input structure:
    - Black background fills the board except for a solid maroon ground row at the bottom.
    - Several small monochrome 4-connected sprites of distinct colors touch the left or right edge and rest directly above the ground.
    - Gray single-pixel obstacles lie in interior columns and may intersect a sprite's next horizontal step.

    Rule:
    - For each sprite, compute the left column of its horizontally mirrored bounding box and move the sprite one column at a time toward that target without flipping its internal layout.
    - If any sprite pixel would enter a gray obstacle on the next horizontal step, shift the whole sprite up by one row first, clip any pixels that rise above the grid, then complete that horizontal step.
    - After all moves, erase each source sprite, draw the sprite at its final reached position, and recolor the ground row between its original and final left columns inclusive with that sprite's color; later sprites overwrite earlier ground marks.

    Key insight:
    - The mirror location fixes only the destination column, while gray blockers change the sprite's final height, so the solver must simulate the horizontal walk step by step before the ground trace can be known.
    """
    out, _ = _transform(np.asarray(grid, dtype=int), respect_obstacles=True, draw_trails=True, mirror=True)
    return out


def _sprite_from_shape(shape: np.ndarray, color: int, side: str, floor_row: int, width: int) -> Sprite:
    h, w = shape.shape
    c0 = 0 if side == "left" else width - w
    r0 = floor_row - h
    coords = np.argwhere(shape.astype(bool))
    pixels = tuple(sorted((int(r0 + r), int(c0 + c)) for r, c in coords))
    return Sprite(color=color, pixels=pixels, bbox=(r0, c0, r0 + h - 1, c0 + w - 1))


def _choose_shape(rng: random.Random, *, height: int) -> np.ndarray:
    bank = HEIGHT1_SHAPES if height == 1 else HEIGHT2_SHAPES
    return rng.choice(bank).copy()


def _anchor_ids(sprite: Sprite, row_kind: str) -> list[int]:
    top = min(r for r, _ in sprite.pixels)
    bottom = max(r for r, _ in sprite.pixels)
    wanted = top if row_kind == "top" else bottom
    return [idx for idx, (r, _c) in enumerate(sprite.pixels) if r == wanted]


def _plan_obstacles_for_sprite(
    sprite: Sprite,
    *,
    desired_detours: int,
    row_kind: str,
    used_cols: set[int],
    occupied_now: set[tuple[int, int]],
    width: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    anchor_choices = _anchor_ids(sprite, row_kind)
    if not anchor_choices:
        return None

    target_left = _target_left(sprite, width, mirror=True)
    path_len = abs(target_left - sprite.left)
    direction = 1 if target_left > sprite.left else -1
    margin_left = 2
    margin_right = width - 3

    rng.shuffle(anchor_choices)
    for anchor_id in anchor_choices:
        anchor_r, anchor_c = sprite.pixels[anchor_id]
        max_usable = int(anchor_r + 1)
        if row_kind == "bottom":
            max_usable = int(anchor_r + 1)
        if desired_detours > max_usable:
            continue

        feasible_steps = []
        for step in range(1, path_len + 1):
            col = int(anchor_c + direction * step)
            if not (margin_left <= col <= margin_right):
                continue
            if col in used_cols:
                continue
            feasible_steps.append(step)
        if len(feasible_steps) < desired_detours:
            continue

        # Spread blockers so the path visibly alternates between free motion and detours.
        if desired_detours == 1:
            steps = [rng.choice(feasible_steps)]
        else:
            cut = max(2, len(feasible_steps) // desired_detours)
            buckets = [feasible_steps[i : i + cut] for i in range(0, len(feasible_steps), cut)]
            picked = []
            for bucket in buckets:
                if not bucket:
                    continue
                picked.append(rng.choice(bucket))
                if len(picked) == desired_detours:
                    break
            if len(picked) < desired_detours:
                extra = [s for s in feasible_steps if s not in picked]
                rng.shuffle(extra)
                picked.extend(extra[: desired_detours - len(picked)])
            steps = sorted(picked[:desired_detours])

        obstacles: list[tuple[int, int]] = []
        ok = True
        for i, step in enumerate(steps):
            rr = int(anchor_r - i)
            cc = int(anchor_c + direction * step)
            if rr < 0:
                ok = False
                break
            if (rr, cc) in occupied_now:
                ok = False
                break
            obstacles.append((rr, cc))
        if ok:
            for _r, c in obstacles:
                used_cols.add(int(c))
            return obstacles
    return None


def _sample_profile(rng: random.Random) -> str:
    return rng.choices(
        ["single", "single_heavy", "top_clip", "both"],
        weights=[4, 2, 2, 2],
        k=1,
    )[0]


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    profile = _sample_profile(rng)

    if profile == "top_clip":
        height = rng.randint(6, 8)
    else:
        height = rng.randint(8, 15)
    width = rng.randint(12, 18)
    floor_row = height - 1

    detour_side = rng.choice(["left", "right"])
    clean_side = "right" if detour_side == "left" else "left"

    if profile == "both":
        left_shape = _choose_shape(rng, height=2)
        right_shape = _choose_shape(rng, height=2)
    else:
        detour_shape = _choose_shape(rng, height=2)
        clean_shape = _choose_shape(rng, height=1)
        if detour_side == "left":
            left_shape, right_shape = detour_shape, clean_shape
        else:
            left_shape, right_shape = clean_shape, detour_shape

    if left_shape.shape[1] + right_shape.shape[1] > width - 3:
        return None

    colors = rng.sample(list(SPRITE_COLORS), 2)
    left_sprite = _sprite_from_shape(left_shape, colors[0], "left", floor_row, width)
    right_sprite = _sprite_from_shape(right_shape, colors[1], "right", floor_row, width)
    sprites = [left_sprite, right_sprite]
    occupied_now = {p for sprite in sprites for p in sprite.pixels}

    used_cols: set[int] = set()
    obstacles: list[tuple[int, int]] = []

    if profile == "single":
        desired = 1
        target = left_sprite if detour_side == "left" else right_sprite
        planned = _plan_obstacles_for_sprite(
            target,
            desired_detours=desired,
            row_kind="top",
            used_cols=used_cols,
            occupied_now=occupied_now,
            width=width,
            rng=rng,
        )
        if planned is None:
            return None
        obstacles.extend(planned)
    elif profile == "single_heavy":
        desired = rng.randint(2, 3)
        target = left_sprite if detour_side == "left" else right_sprite
        planned = _plan_obstacles_for_sprite(
            target,
            desired_detours=desired,
            row_kind="top",
            used_cols=used_cols,
            occupied_now=occupied_now,
            width=width,
            rng=rng,
        )
        if planned is None:
            return None
        obstacles.extend(planned)
    elif profile == "top_clip":
        target = left_sprite if detour_side == "left" else right_sprite
        desired = target.top + 1
        planned = _plan_obstacles_for_sprite(
            target,
            desired_detours=desired,
            row_kind="top",
            used_cols=used_cols,
            occupied_now=occupied_now,
            width=width,
            rng=rng,
        )
        if planned is None:
            return None
        obstacles.extend(planned)
    else:
        left_plan = _plan_obstacles_for_sprite(
            left_sprite,
            desired_detours=rng.randint(1, 2),
            row_kind="top",
            used_cols=used_cols,
            occupied_now=occupied_now,
            width=width,
            rng=rng,
        )
        if left_plan is None:
            return None
        obstacles.extend(left_plan)
        right_plan = _plan_obstacles_for_sprite(
            right_sprite,
            desired_detours=rng.randint(1, 2),
            row_kind="top",
            used_cols=used_cols,
            occupied_now=occupied_now,
            width=width,
            rng=rng,
        )
        if right_plan is None:
            return None
        obstacles.extend(right_plan)

    grid = np.zeros((height, width), dtype=int)
    grid[-1, :] = GROUND
    for sprite in sprites:
        for r, c in sprite.pixels:
            grid[r, c] = int(sprite.color)
    for r, c in obstacles:
        if grid[r, c] != BG:
            return None
        grid[r, c] = OBSTACLE

    return grid


def _validate_input(grid: np.ndarray) -> dict | None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return None
    h, w = grid.shape
    if max(h, w) > 30 or h < 3 or w < 6:
        return None
    if not np.all(grid[-1] == GROUND):
        return None

    sprites = _extract_sprites(grid)
    if len(sprites) != 2:
        return None
    if len({sprite.color for sprite in sprites}) != len(sprites):
        return None
    for sprite in sprites:
        if not (3 <= len(sprite.pixels) <= 5):
            return None
        if sprite.height > 2 or sprite.bottom != h - 2:
            return None
        if not ((sprite.left == 0) ^ (sprite.right == w - 1)):
            return None

    obstacle_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == OBSTACLE)]
    if not obstacle_coords:
        return None
    if any(c in (0, 1, w - 2, w - 1) for _r, c in obstacle_coords):
        return None
    used_cols = set()
    for r, c in obstacle_coords:
        if c in used_cols:
            return None
        used_cols.add(c)
        if r >= h - 1:
            return None

    true_out, metas = _transform(grid, respect_obstacles=True, draw_trails=True, mirror=True)
    mirror_only, no_obstacle_metas = _transform(grid, respect_obstacles=False, draw_trails=True, mirror=True)
    no_trails, _ = _transform(grid, respect_obstacles=True, draw_trails=False, mirror=True)
    no_mirror, _ = _transform(grid, respect_obstacles=True, draw_trails=True, mirror=False)

    if np.array_equal(true_out, no_trails):
        return None
    if np.array_equal(true_out, no_mirror):
        return None

    detours = [int(meta["detours"]) for meta in metas]
    clipped = [bool(meta["clipped"]) for meta in metas]
    if max(detours) <= 0:
        return None
    if np.array_equal(true_out, mirror_only):
        return None
    if any(meta["all_lost"] for meta in metas):
        return None

    # At least one sprite should move cleanly in most cases, but allow both-detour samples too.
    clean_count = sum(d == 0 for d in detours)
    if clean_count == 0 and min(detours) >= 3:
        return None

    # Trails should stay visually readable: keep some ground visible at the far right.
    if int(np.sum(true_out[-1] == GROUND)) == 0:
        return None

    # The moved final sprites must encode the obstacle effect, not just the trail.
    if all(meta["detours"] == alt["detours"] and meta["final_pixels"] == alt["final_pixels"] for meta, alt in zip(metas, no_obstacle_metas)):
        return None

    return {
        "out": true_out,
        "metas": metas,
        "detours": detours,
        "clipped": clipped,
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is None:
            continue
        info = _validate_input(grid)
        if info is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
